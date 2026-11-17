"""Fault injection for the pipeline stages data -> template -> reward -> GRPO grouping -> LoRA -> streaming -> config.

Every test runs the unaffected control and the injected fault through the same check. CPU only, no model weights.
The LoRA tests read the reference run's step-0 and step-90 adapters (runs/qwen, kept in the repository)."""
import copy
from pathlib import Path

import pytest
import torch
import yaml
from peft import load_peft_weights
from transformers import AutoTokenizer

from qwen_grpo import config
from qwen_grpo.diagnostics import invariants
from qwen_grpo.sentiment import build_prompts, load_splits, reward
from qwen_grpo.training import check_grouping

ROOT = Path(__file__).parents[2]


@pytest.fixture(scope="module")
def splits():
    return load_splits()


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")


# --- data ---------------------------------------------------------------------------------------------------------

def test_split_leakage_and_duplicates_are_reported(splits):
    assert invariants.split_problems(splits) == []  # control: the repository's splits
    leaked = copy.deepcopy(splits)
    leaked["train"].append(dict(leaked["test"][7], text="  " + leaked["test"][7]["text"].upper() + " "))
    leaked["validation"].append(dict(leaked["validation"][0]))
    problems = invariants.split_problems(leaked)
    assert any("share 1 texts (split leakage)" in p and "train" in p and "test" in p for p in problems), problems
    assert any(p.startswith("validation: 1 duplicate") for p in problems), problems


def test_invalid_labels_are_reported(splits):
    bad = copy.deepcopy(splits)
    bad["train"][3]["label"] = "positive"  # not canonical: the reward would never pay out for this prompt
    assert any("label outside" in p for p in invariants.split_problems(bad))


# --- template -----------------------------------------------------------------------------------------------------

class ThinkingTokenizer:
    """The real tokenizer with enable_thinking forced on (the injected template fault)."""

    def __init__(self, tok):
        self.tok = tok

    def apply_chat_template(self, *a, **k):
        return self.tok.apply_chat_template(*a, **{**k, "enable_thinking": True})


def test_a_thinking_template_is_refused(tok):
    assert build_prompts(tok, ["fine ."])[0].endswith("<think>\n\n</think>\n\n")  # control
    with pytest.raises(AssertionError):
        build_prompts(ThinkingTokenizer(tok), ["fine ."])


def test_prompts_are_never_truncated(tok, splits):
    longest = max((r["text"] for s in ("train", "validation", "test") for r in splits[s]), key=len)
    prompt = build_prompts(tok, [longest])[0]
    ids = tok(prompt, add_special_tokens=False)["input_ids"]  # what policy.generate_rollouts tokenizes
    assert tok.decode(ids) == prompt  # control: the full prompt survives tokenization
    cut = tok(prompt, add_special_tokens=False, truncation=True, max_length=16)["input_ids"]  # injected truncation
    assert tok.decode(cut) != prompt


# --- reward -------------------------------------------------------------------------------------------------------

def test_a_permissive_reward_is_caught():
    assert invariants.reward_problems(reward) == []  # control: the training reward
    substring = lambda r, label: float(label in r.upper())  # noqa: E731 - the injected, gameable reward
    problems = invariants.reward_problems(substring)
    assert any("POSITIVE NEGATIVE" in p for p in problems) and any("The sentiment is" in p for p in problems)
    lenient_empty = lambda r, label: 1.0 if not r.strip() else reward(r, label)  # noqa: E731
    assert any("reward(''" in p for p in invariants.reward_problems(lenient_empty))
    soft = lambda r, label: 0.5  # noqa: E731 - not a 0/1 reward
    assert all("not 0.0 / 1.0" in p for p in invariants.reward_problems(soft))


# --- GRPO grouping ------------------------------------------------------------------------------------------------

def test_misgrouped_rollouts_are_refused():
    B, G, P, R = 3, 4, 5, 2
    prompt_ids = torch.arange(B * P).reshape(B, P) + 10
    good = {"input_ids": torch.cat([prompt_ids.repeat_interleave(G, 0), torch.zeros(B * G, R, dtype=torch.long)], 1),
            "prompt_len": P}
    check_grouping(good, prompt_ids, G)  # control
    bad = {**good, "input_ids": good["input_ids"][[0, 1, 2, 4, 3, *range(5, B * G)]]}  # rows 3/4 swapped across groups
    with pytest.raises(AssertionError, match="not grouped by prompt"):
        check_grouping(bad, prompt_ids, G)


# --- LoRA ---------------------------------------------------------------------------------------------------------

def adapter(step):
    return load_peft_weights(str(ROOT / "runs" / "qwen" / "checkpoints" / f"step{step}"), device="cpu")


def test_an_inactive_or_unchanged_adapter_is_caught():
    init, trained = adapter(0), adapter(90)
    assert invariants.adapter_problems(init, trained) == []  # control: the reference run's real adapters
    problems = invariants.adapter_problems(init, init)  # injected: training never changed anything
    assert any("inactive" in p for p in problems) and any("unexpectedly unchanged" in p for p in problems)
    zeroed = {k: (torch.zeros_like(v) if "lora_B" in k else v) for k, v in trained.items()}
    assert any("inactive" in p for p in invariants.adapter_problems(init, zeroed))


def test_missing_extra_and_nonfinite_adapter_tensors_are_caught():
    init, trained = adapter(0), adapter(90)
    key = next(k for k in trained if "lora_A" in k)
    missing = {k: v for k, v in trained.items() if k != key}
    assert any("missing" in p for p in invariants.adapter_problems(init, missing))
    extra = {**trained, "base_model.model.lm_head.lora_A.weight": torch.zeros(1)}
    assert any("unexpected tensors" in p for p in invariants.adapter_problems(init, extra))
    nan = {**trained, key: trained[key].clone().index_fill_(0, torch.tensor([0]), float("nan"))}
    assert any("non-finite" in p for p in invariants.adapter_problems(init, nan))


# --- streaming ----------------------------------------------------------------------------------------------------

def records():
    """Committed rollout records of a run: 3 groups of 2 prompts over a 4-sample epoch, then epoch 1."""
    occ = [[(0, 0, 11), (0, 1, 7)], [(0, 2, 3), (0, 3, 5)], [(1, 0, 7), (1, 1, 3)]]
    return [{"group_id": g + 1, "policy_version": g, "occurrences": o} for g, o in enumerate(occ)]


def test_duplicate_or_skipped_stream_records_are_caught():
    assert invariants.rollout_record_problems(records(), epoch_size=4) == []  # control
    dup = records() + [dict(records()[2], group_id=4)]  # a group committed twice (replayed after a crash)
    assert any("committed twice" in p for p in invariants.rollout_record_problems(dup))
    gap = [r for r in records() if r["group_id"] != 2]  # a group lost: epoch 0 positions 2-3 never trained
    assert invariants.rollout_record_problems(gap) == []  # missed without the epoch size: positions 0-1 look complete
    assert any("2 of 4 samples" in p for p in invariants.rollout_record_problems(gap, epoch_size=4))
    hole = [records()[0], dict(records()[1], occurrences=[(0, 3, 5)]), records()[2]]  # position 2 skipped mid-epoch
    assert any("not contiguous" in p for p in invariants.rollout_record_problems(hole))
    back = records()
    back[2] = dict(back[2], policy_version=0)  # resumed with an older policy
    assert any("backwards" in p for p in invariants.rollout_record_problems(back))
    same_id = records()
    same_id[1] = dict(same_id[1], group_id=1)
    assert any("duplicate rollout groups" in p for p in invariants.rollout_record_problems(same_id))


# --- configuration --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("override, reason", [
    ({"activation_checkpointing": False}, "activation_checkpointing false"),
    ({"vocab_chunk_size": 4096}, "vocab_chunk_size"),
    ({"resident_layers": 2}, "resident_layers > 0"),
])
def test_ignored_execution_settings_are_refused(tmp_path, override, reason):
    raw = config.read(ROOT / "configs" / "profiles" / "qwen3_4b.yaml")
    assert config.unimplemented(config.load(ROOT / "configs" / "profiles" / "qwen3_4b.yaml")) == []  # control
    raw["execution"].update(override)
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(raw))
    assert any(reason in r for r in config.unimplemented(config.load(p)))
