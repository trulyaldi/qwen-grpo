"""TRL reference (research/trl_reference): equivalence with the manual pipeline, without loading Qwen3-0.6B weights.

The trainer tests use a tiny fp32 Qwen3 (28 layers so the layer-26/27 targets exist) with the real tokenizer, on CPU.
"""
import json
from pathlib import Path

import pytest
import torch
from peft import get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from baseline import EOS_IDS, MODEL, build_prompts
from grpo import group_advantages
from research.trl_reference.compare_gradients import (compare, manual_gradient, manual_rollouts, make_trainer, trl_advantages,
                                                      trl_gradient)
from research.trl_reference.evaluate import shift_stats
from research.trl_reference.train import CONFIG, G, MANUAL_TRAINING, check_lora, grpo_args, make_dataset, make_reward, stream, train_examples
from sentiment import _key, load_splits
from train_grpo import decode
from train_lora import lora_config
from train_multistep import EXPECTED_MODULES, TARGETS, group_rewards, lora_modules, metrics

ROOT = Path(__file__).parents[1]
POS, NEG, IM_START = [17456, 44350, 151645], [97127, 23215, 151645], 151644
# Two groups of G=6 for a [POSITIVE, NEGATIVE] prompt pair. Covers: canonical, wrong label, pad id as the EOS, EOS only,
# truncated (no EOS, R = 8 columns), a special token inside the label, lowercase (filled in by the fixture), trailing ".".
COMPLETIONS = [
    POS, POS, NEG, [151645], [9707] * 8, [17456, IM_START, 44350, 151645],
    NEG, [97127, 23215, 151643], POS, NEG, None, [97127, 23215, 13, 151645],
]


@pytest.fixture(scope="module", autouse=True)
def float32_default():
    """tests/test_grpo.py sets a float64 default dtype at import; these tests mirror the fp32/bf16 training setup."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL, padding_side="left")


@pytest.fixture(scope="module")
def completions(tok):
    out = list(COMPLETIONS)
    out[10] = tok("negative", add_special_tokens=False)["input_ids"] + [151645]
    return out


@pytest.fixture(scope="module")
def examples():
    return train_examples()


@pytest.fixture(scope="module")
def pair(examples):
    return stream(examples, 1)[0][0]


def tiny_model(tok):
    """fp32 Qwen3 with the real vocabulary and 28 tiny layers, LoRA on the six real target names, B made non-zero."""
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=28, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=8, tie_word_embeddings=True, max_position_embeddings=1024)
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(cfg).float().eval()
    base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
    model = get_peft_model(base, lora_config(TARGETS["last2-attention"])).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".lora_B." in n:
                p.normal_(std=0.5)  # with B = 0 the A-gradient is exactly zero; make it a real test
    return model


@pytest.fixture(scope="module")
def tiny(tok, examples, completions, tmp_path_factory):
    model = tiny_model(tok)
    params, _ = check_lora(model, dtype=torch.float32)
    log = []
    trainer = make_trainer(model, tok, examples, completions, log,
                           grpo_args(tmp_path_factory.mktemp("trl"), use_cpu=True))
    return model, params, trainer, log


def test_reward_matches_manual_parser(tok, examples, pair, completions):
    prompts = build_prompts(tok, [e["text"] for e in pair])
    labels = [e["label"] for e in pair]
    manual = group_rewards(decode(tok, manual_rollouts(tok, prompts, completions, "cpu")), labels, G).flatten().tolist()
    rows = [(p, e) for p, e in zip(prompts, pair) for _ in range(G)]
    trl_texts = tok.batch_decode(completions, skip_special_tokens=True)  # what TRL passes as `completions`
    trl = make_reward(tok, examples)(prompts=[p for p, _ in rows], completions=trl_texts, completion_ids=completions,
                                      ground_truth=[e["label"] for _, e in rows], example_id=[e["id"] for _, e in rows])
    assert trl == manual == [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]  # the parser upper-cases
    assert trl_texts[5] == "POSITIVE"  # TRL's skip_special_tokens text would score 1 here; the id-based reward gives 0


def test_dataset_matches_manual_subset(tok, examples):
    saved = json.loads(MANUAL_TRAINING.read_text())["config"]["train_ids"]
    assert [e["id"] for e in examples] == saved and len(examples) == CONFIG["train_samples"] == 32
    assert sum(e["label"] == "POSITIVE" for e in examples) == 16
    ds = make_dataset(tok, examples)
    assert ds["prompt"] == build_prompts(tok, [e["text"] for e in examples])
    assert ds["ground_truth"] == [e["label"] for e in examples] and ds["example_id"] == saved
    splits = load_splits()
    held_out = {_key(r["text"]) for s in ("validation", "test") for r in splits[s]}
    assert not held_out & {_key(e["text"]) for e in examples}


def test_trl_tokenizes_the_manual_prompt(tok, examples, tiny):
    prompts = build_prompts(tok, [e["text"] for e in examples])
    trl_ids = tiny[2]._tokenize_prompts(prompts)[0]
    assert trl_ids == tok(prompts, add_special_tokens=False)["input_ids"]
    assert all(t.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for t in prompts)  # non-thinking mode


def test_lora_targets_on_real_architecture():
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(MODEL), dtype=torch.bfloat16)
        model = get_peft_model(base, lora_config(TARGETS["last2-attention"]))
    assert [n for n, _ in lora_modules(model)] == EXPECTED_MODULES["last2-attention"]
    params = [p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params) == CONFIG["lora"]["expected_trainable_params"] == 131_072
    assert all(p.dtype == torch.float32 for p in params)  # PEFT upcasts adapters on a bf16 base
    assert all(p.dtype == torch.bfloat16 for p in model.parameters() if not p.requires_grad)


def test_trl_batches_are_the_manual_balanced_stream(examples, tok, tmp_path):
    pairs, order = stream(examples, CONFIG["grpo_config"]["max_steps"])
    trainer = make_trainer(tiny_model(tok), tok, examples, [], [], grpo_args(tmp_path, use_cpu=True))
    trainer.sampler_order = order
    args = trainer.args
    assert (args.per_device_train_batch_size, args.steps_per_generation, args.gradient_accumulation_steps) == (2 * G, 1, 1)
    assert args.generation_batch_size == 2 * G and trainer.num_generations == G == 6
    batches = list(trainer.get_train_dataloader())
    assert len(batches) == len(pairs) == 30
    for b, p in zip(batches, pairs):
        assert [r["example_id"] for r in b] == [e["id"] for e in p for _ in range(G)]
        assert [r["ground_truth"] for r in b] == ["POSITIVE"] * G + ["NEGATIVE"] * G


def test_reward_rejects_misaligned_batches(tok, examples, pair, completions):
    pairs = [pair]
    fn = make_reward(tok, examples, pairs, log=[])
    prompts = build_prompts(tok, [e["text"] for e in pair])
    ok = dict(prompts=[p for p in prompts for _ in range(G)], completion_ids=completions,
              ground_truth=[e["label"] for e in pair for _ in range(G)], example_id=[e["id"] for e in pair for _ in range(G)])
    fn(**ok)
    with pytest.raises(AssertionError):  # labels shifted by one row relative to prompts
        make_reward(tok, examples, pairs, log=[])(**{**ok, "ground_truth": ok["ground_truth"][1:] + ok["ground_truth"][:1]})
    with pytest.raises(AssertionError):  # NEGATIVE group first
        make_reward(tok, examples, pairs, log=[])(**{k: v[G:] + v[:G] for k, v in ok.items()})


def test_fp32_gradient_trl_equals_manual(tok, pair, completions, tiny):
    """Exact equivalence in fp32 once TRL's Bessel-corrected std is used; the raw gap is that rescaling alone."""
    model, params, trainer, log = tiny
    prompts = build_prompts(tok, [e["text"] for e in pair])
    rollouts = manual_rollouts(tok, prompts, completions, "cpu")
    rewards = group_rewards(decode(tok, rollouts), [e["label"] for e in pair], G)
    by_id = {k: i for i, k in enumerate(trainer.train_dataset["example_id"])}
    rows = [trainer.train_dataset[by_id[e["id"]]] for e in pair for _ in range(G)]
    g_trl, loss_trl, out = trl_gradient(trainer, model, params, rows)

    assert torch.equal(torch.tensor(log[-1]["rewards"]).reshape(2, G), rewards)
    assert torch.allclose(out["advantages"], trl_advantages(rewards).flatten(), atol=1e-6)
    assert torch.equal(out["completion_ids"], rollouts["response_ids"]) and torch.equal(out["completion_mask"].bool(), rollouts["completion_mask"])
    g_aligned, loss_aligned = manual_gradient(model, params, rollouts, trl_advantages(rewards))
    c = compare(g_aligned, g_trl)
    assert c["rel_diff"] < 1e-5 and abs(loss_aligned - loss_trl) < 1e-6, c
    assert all(g.abs().max() > 0 for g in g_trl.values())

    g_raw, _ = manual_gradient(model, params, rollouts, group_advantages(rewards))
    raw = compare(g_raw, g_trl)
    # 2 and 4 of 6 correct: equal p(1-p), so one std for both groups and the advantage ratio is a single scalar.
    s_pop = rewards.std(1, unbiased=False)
    assert s_pop[0] == s_pop[1]
    s_pop = s_pop[0].item()
    s_unb = (G / (G - 1)) ** 0.5 * s_pop
    assert raw["cosine"] > 1 - 1e-9 and abs(raw["norm_ratio"] - (s_pop + 1e-4) / (s_unb + 1e-4)) < 1e-5, raw


def test_score_shift_stats_by_hand():
    s = shift_stats([0.0, 1.0, -1.0, 2.0], [0.5, 1.5, -0.25, 2.25], ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"])
    assert s["mean"] == pytest.approx(0.5) and s["median"] == pytest.approx(0.5) and s["frac_positive"] == 1.0
    assert s["mean_POSITIVE_gold"] == pytest.approx(0.625) and s["mean_NEGATIVE_gold"] == pytest.approx(0.375)
    assert s["sd"] == pytest.approx(torch.tensor([0.5, 0.5, 0.75, 0.25]).std().item())


def test_complete_label_scoring_matches_saved_manual_results(tok):
    """The label continuations scored by label_scores, and the saved manual numbers this comparison is anchored to."""
    assert tok("POSITIVE", add_special_tokens=False)["input_ids"] + [EOS_IDS[0]] == POS
    assert tok("NEGATIVE", add_special_tokens=False)["input_ids"] + [EOS_IDS[0]] == NEG
    saved = json.loads(MANUAL_TRAINING.read_text())["final"]
    preds = json.loads((ROOT / "results" / "grpo_lora_last2_b2_balanced_step30_validation.json").read_text())["predictions"]
    base = json.loads((ROOT / "results" / "baseline_validation.json").read_text())["predictions"]
    labels = [p["label"] for p in base]
    assert [p["id"] for p in base] == [p["id"] for p in preds] == saved["label_scores"]["ids"] == list(range(1066))
    assert round(metrics(base)["accuracy"], 4) == 0.7486 and round(metrics(preds)["accuracy"], 4) == 0.7261
    s = shift_stats(saved["label_scores"]["base"], saved["label_scores"]["trained"], labels)
    assert s["frac_positive"] == 1.0 and 0.6 < s["mean"] < 0.7 and s["pearson_base_trained"] > 0.99
