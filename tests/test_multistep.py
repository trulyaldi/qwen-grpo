import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from qwen_grpo import evaluation
from qwen_grpo.evaluation import metrics, paired_bootstrap
from qwen_grpo.sentiment import load_splits
from qwen_grpo.training import balanced_subset, group_rewards, grpo_step

ROOT = Path(__file__).parents[1]


class Tiny(torch.nn.Module):
    """Stand-in for a causal LM: logits = Linear(Embedding(ids)); ignores masks and positions."""

    def __init__(self, V=5, d=4):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.out = torch.nn.Embedding(V, d), torch.nn.Linear(d, V)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(logits=self.out(self.emb(input_ids)))


def rollouts(G=3, P=2, R=3):
    ids = torch.randint(0, 5, (G, P + R), generator=torch.Generator().manual_seed(1))
    mask = torch.ones(G, R, dtype=torch.bool)
    return {"input_ids": ids, "attention_mask": torch.ones(G, P + R, dtype=torch.long), "completion_mask": mask, "prompt_len": P}


def setup():
    model = Tiny()
    params = dict(model.named_parameters())
    return model, params, torch.optim.AdamW(params.values(), lr=1e-2, weight_decay=0.0)


def test_training_subset_balanced_and_fixed():
    rows = load_splits()["train"]
    a, b = balanced_subset(rows, 32, 42), balanced_subset(rows, 32, 42)
    assert a == b and len({e["id"] for e in a}) == 32
    assert sum(e["label"] == "POSITIVE" for e in a) == sum(e["label"] == "NEGATIVE" for e in a) == 16
    assert all(rows[e["id"]]["text"] == e["text"] and rows[e["id"]]["label"] == e["label"] for e in a)


def test_rewards_follow_rollout_grouping():
    responses = ["POSITIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "x"]
    with pytest.raises(AssertionError):
        group_rewards(responses, ["POSITIVE", "NEGATIVE"], 3)
    r = group_rewards(responses + ["POSITIVE"], ["POSITIVE", "NEGATIVE"], 3)  # rows 0-2 -> prompt 0, 3-5 -> prompt 1
    assert r.tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_zero_signal_group_skips_step():
    model, params, opt = setup()
    before = {n: p.detach().clone() for n, p in params.items()}
    assert grpo_step(model, params, opt, rollouts(), torch.ones(1, 3)) is None
    assert all(torch.equal(p, before[n]) for n, p in params.items()) and not opt.state


def test_nonfinite_gradient_blocks_step():
    model, params, opt = setup()
    before = {n: p.detach().clone() for n, p in params.items()}
    model.out.weight.register_hook(lambda g: g * float("nan"))
    with pytest.raises(RuntimeError, match="non-finite gradients"):
        grpo_step(model, params, opt, rollouts(), torch.tensor([[1.0, 0.0, 0.0]]))
    assert all(torch.equal(p, before[n]) for n, p in params.items()) and not opt.state


def test_step_updates_and_logs_plain_scalars():
    model, params, opt = setup()
    before = {n: p.detach().clone() for n, p in params.items()}
    out = grpo_step(model, params, opt, rollouts(), torch.tensor([[1.0, 0.0, 0.0]]))
    assert all(not torch.equal(p, before[n]) for n, p in params.items())
    assert all(p.grad is None for p in params.values())
    flat = [v for v in out.values() if not isinstance(v, dict)] + list(out["grad_norms"].values())
    assert all(type(v) is float for v in flat)


def test_predictions_keep_baseline_schema(monkeypatch):
    monkeypatch.setattr(evaluation, "generate", lambda model, tok, s: [("POSITIVE", 2, False), ("meh", 2, False)][: len(s)])
    ex = [{"id": 7, "text": "good", "label": "POSITIVE"}, {"id": 9, "text": "bad", "label": "NEGATIVE"}]
    preds = evaluation.predict(None, None, ex, [7, 9])[0]
    saved = json.loads((ROOT / "results" / "baseline_validation.json").read_text())["predictions"][0]
    assert all(p.keys() == saved.keys() for p in preds)
    assert [(p["id"], p["prediction"], p["correct"]) for p in preds] == [(7, "POSITIVE", True), (9, None, False)]


def test_metrics_reproduce_baseline_confusion():
    cells = [("POSITIVE", "POSITIVE", 381), ("POSITIVE", "NEGATIVE", 116), ("NEGATIVE", "POSITIVE", 149),
             ("NEGATIVE", "NEGATIVE", 417), (None, "POSITIVE", 3)]
    preds = [{"prediction": p, "label": g, "correct": p == g} for p, g, n in cells for _ in range(n)]
    m = metrics(preds)
    assert m["n"] == 1066 and round(m["accuracy"], 4) == 0.7486 and abs(m["macro_f1"] - (762 / 1030 + 834 / 1099) / 2) < 1e-12
    assert round(m["recall_POSITIVE"], 4) == 0.7148 and round(m["recall_NEGATIVE"], 4) == 0.7824
    assert m["confusion"]["None"] == {"NEGATIVE": 0, "POSITIVE": 3}


def test_paired_bootstrap_identical_models_has_zero_interval():
    c = [True, False] * 50
    assert paired_bootstrap(c, c, n=2000) == (0.0, 0.0, 0.0)


from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo.evaluation import paired_compare
from qwen_grpo.model import EXPECTED_MODULES, TARGET, TARGETS, expected_lora_params, lora_config, lora_modules
from research.train_multistep import output_name


def tiny_qwen3():
    """Random 28-layer Qwen3 with the real module naming; q: 16->16, v: 16->8, o: 16->16."""
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=28,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    return Qwen3ForCausalLM(cfg)


def test_original_target_unchanged():
    assert TARGETS["last-o-proj"] == TARGET == r"model\.layers\.27\.self_attn\.o_proj"
    assert lora_config().target_modules == TARGET


@pytest.mark.parametrize("targets", list(TARGETS))
def test_peft_attaches_exactly_the_requested_projections(targets):
    model = get_peft_model(tiny_qwen3(), lora_config(TARGETS[targets]))
    modules = lora_modules(model)
    assert [n for n, _ in modules] == EXPECTED_MODULES[targets]
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 2 * len(modules) and all(".lora_A." in n or ".lora_B." in n for n in trainable)
    assert all(any(n.startswith(f"base_model.model.{m}.") for m in EXPECTED_MODULES[targets]) for n in trainable)


def test_trainable_parameter_count():
    model = get_peft_model(tiny_qwen3(), lora_config(TARGETS["last2-attention"]))
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_hand = 2 * 8 * ((16 + 16) + (16 + 8) + (16 + 16))  # 2 layers x r x (in + out) for q, v, o
    assert n == expected_lora_params(lora_modules(model)) == by_hand == 1408


def test_output_names_differ():
    assert output_name("last-o-proj") == "grpo_lora" and output_name("last-o-proj", "_lr0") == "grpo_lora_lr0"
    assert output_name("last2-attention") == "grpo_lora_last2" and output_name("last2-attention", "_smoke") == "grpo_lora_last2_smoke"


def records(correct, ids=None):
    ids = ids or list(range(len(correct)))
    return [{"id": i, "text": f"t{i}", "label": LABEL[i % 2], "prediction": LABEL[i % 2] if c else None, "correct": c}
            for i, c in zip(ids, correct)]


LABEL = ("NEGATIVE", "POSITIVE")


def test_paired_compare_rejects_misaligned_examples():
    ref = records([True, False, True])
    with pytest.raises(AssertionError):
        paired_compare(ref, records([True, False, True], ids=[0, 2, 1]))


def test_bootstrap_pairs_examples():
    ref = [True] * 50 + [False] * 50
    swapped = [False] * 50 + [True] * 50  # same accuracy, but every example flips
    assert paired_compare(records(ref), records(ref))["ci95"] == [0.0, 0.0]
    c = paired_compare(records(ref), records(swapped))
    assert c["accuracy_diff"] == 0.0 and c["ci95"][0] < -0.1 and c["ci95"][1] > 0.1
    one_fix = ref[:]
    one_fix[60] = True
    c = paired_compare(records(ref), records(one_fix))
    assert c["accuracy_diff"] == pytest.approx(0.01) and c["ci95"][0] >= 0.0


# ---- Stage 3C: B=2 batches, grouping, label scoring ----
from transformers import AutoTokenizer

from qwen_grpo.evaluation import label_scores, paired_auc_bootstrap, paired_changes, win_matrix
from qwen_grpo.grpo import group_advantages
from qwen_grpo.model import EOS_IDS, MODEL
from qwen_grpo.sentiment import build_prompts
from qwen_grpo.training import batches, check_grouping
from research.train_multistep import REFERENCES

TRAIN = balanced_subset(load_splits()["train"], 32, 42)


def test_check_grouping_accepts_prompt_major_rows_only():
    prompt_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    rows = prompt_ids.repeat_interleave(6, dim=0)  # rows 0-5 prompt 0, rows 6-11 prompt 1
    r = {"input_ids": torch.cat([rows, torch.zeros(12, 4, dtype=torch.long)], 1), "prompt_len": 3}
    check_grouping(r, prompt_ids, 6)
    r["input_ids"][:, :3] = prompt_ids.repeat(6, 1)  # interleaved 0,1,0,1,...
    with pytest.raises(AssertionError):
        check_grouping(r, prompt_ids, 6)


def test_b2_rewards_scored_against_own_prompt_label():
    responses = ["POSITIVE", "NEGATIVE", "POSITIVE", "POSITIVE", "x", "NEGATIVE",    # prompt 0: POSITIVE
                 "NEGATIVE", "POSITIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "x"]  # prompt 1: NEGATIVE
    r = group_rewards(responses, ["POSITIVE", "NEGATIVE"], 6)
    assert r.shape == (2, 6)
    assert r.tolist() == [[1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0]]


def test_advantages_normalized_per_group_not_jointly():
    r = torch.tensor([[1.0, 0, 1, 1, 0, 0], [1.0, 1, 1, 1, 1, 0]])
    a = group_advantages(r)
    assert a.shape == (2, 6)
    for row, rr in zip(a, r):
        assert torch.allclose(row, (rr - rr.mean()) / (rr.std(unbiased=False) + 1e-4))
    joint = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    assert not torch.allclose(a, joint)


def b2_step(rewards):
    model, params, opt = setup()
    out = grpo_step(model, params, opt, rollouts(G=6), rewards)  # 6 rows = B=2 x G=3
    return out, {n: p.detach().clone() for n, p in params.items()}


def test_one_zero_signal_group_still_steps_and_contributes_nothing():
    out_a, pa = b2_step(torch.tensor([[1.0, 0, 0], [1.0, 1, 1]]))
    out_b, pb = b2_step(torch.tensor([[1.0, 0, 0], [0.0, 0, 0]]))  # different constant group, same zero advantages
    assert out_a is not None and all(torch.equal(pa[n], pb[n]) for n in pa)
    assert any(not torch.equal(pa[n], p) for n, p in setup()[1].items())


def test_both_groups_zero_signal_skip_step():
    model, params, opt = setup()
    before = {n: p.detach().clone() for n, p in params.items()}
    assert grpo_step(model, params, opt, rollouts(G=6), torch.tensor([[1.0, 1, 1], [0.0, 0, 0]])) is None
    assert all(torch.equal(p, before[n]) for n, p in params.items()) and not opt.state


def test_frozen_parameters_not_updated():
    model = Tiny()
    model.emb.weight.requires_grad_(False)
    emb = model.emb.weight.detach().clone()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(params.values(), lr=1e-2, weight_decay=0.0)
    assert grpo_step(model, params, opt, rollouts(G=6), torch.tensor([[1.0, 0, 0], [0.0, 1, 1]])) is not None
    assert torch.equal(model.emb.weight, emb) and model.emb.weight.grad is None


def test_balanced_batches_deterministic_one_per_class_without_replacement():
    take = lambda: [next(s) for s in [batches(TRAIN, 2, "balanced")] for _ in range(32)]
    a, b = take(), take()
    assert [[e["id"] for e in x] for x in a] == [[e["id"] for e in x] for x in b]
    assert all([e["label"] for e in x] == ["POSITIVE", "NEGATIVE"] for x in a)
    for k, label in enumerate(("POSITIVE", "NEGATIVE")):
        pool = sorted(e["id"] for e in TRAIN if e["label"] == label)
        first, second = [x[k]["id"] for x in a[:16]], [x[k]["id"] for x in a[16:]]
        assert sorted(first) == sorted(second) == pool and first != second  # full pass, then a reshuffle


def test_unconstrained_batches_cover_subset_each_pass():
    s = batches(TRAIN, 2, "unconstrained")
    first = [next(s) for _ in range(16)]
    ids = [e["id"] for x in first for e in x]
    assert sorted(ids) == sorted(e["id"] for e in TRAIN)
    assert any(x[0]["label"] == x[1]["label"] for x in first)  # labels are not forced apart


def test_cycle_mode_is_original_b1_order():
    s = batches(TRAIN, 1, "cycle")
    assert [next(s)[0]["id"] for _ in range(40)] == [TRAIN[i % 32]["id"] for i in range(40)]


def test_output_files_distinct_and_flips_by_direction():
    names = {output_name("last2-attention", "", 2, m) for m in ("balanced", "unconstrained")}
    names |= {output_name(t) for t in TARGETS}
    assert len(names) == 4
    assert len({p for p, _, _ in REFERENCES.values()}) == len(REFERENCES)
    base = records([True, True, False, False])
    new = [dict(p) for p in base]
    new[1].update(prediction="NEGATIVE", correct=False)  # gold POSITIVE broken
    new[2].update(prediction="NEGATIVE", correct=True)   # gold NEGATIVE, was invalid
    assert paired_changes(base, new)["flips"] == {"POSITIVE->NEGATIVE": 1, "None->NEGATIVE": 1}


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL, padding_side="left")


def test_label_score_sums_full_label_tokenization_and_eos(tok):
    model = Tiny(V=len(tok))  # logits depend only on the current token, so log-probs factorize by hand
    model.device = torch.device("cpu")
    texts = ["a fine film", "dull, overlong and badly acted"]
    ex = [{"text": t, "label": "POSITIVE"} for t in texts]
    got = label_scores(model, tok, ex, batch=2)
    lsm = lambda a, b: torch.log_softmax(model.out(model.emb(torch.tensor(a))), -1)[b].item()
    for t, s in zip(texts, got):
        last = tok(build_prompts(tok, [t]), add_special_tokens=False)["input_ids"][0][-1]
        by_hand = {}
        for label in ("POSITIVE", "NEGATIVE"):
            c = tok(label, add_special_tokens=False)["input_ids"] + [EOS_IDS[0]]
            assert len(c) == 3  # two label tokens + <|im_end|>
            by_hand[label] = sum(lsm(prev, nxt) for prev, nxt in zip([last] + c[:-1], c))
        assert s == pytest.approx(by_hand["POSITIVE"] - by_hand["NEGATIVE"], abs=1e-4)


def test_auc_from_scores_and_paired_bootstrap():
    labels = ["POSITIVE", "NEGATIVE"] * 30
    g = torch.Generator().manual_seed(0)
    s = torch.randn(60, generator=g).tolist()
    W = win_matrix(s, labels)
    pos = [x for x, l in zip(s, labels) if l == "POSITIVE"]
    neg = [x for x, l in zip(s, labels) if l == "NEGATIVE"]
    brute = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    assert W.mean().item() == pytest.approx(brute)
    perfect = [1.0 if l == "POSITIVE" else 0.0 for l in labels]
    assert win_matrix(perfect, labels).mean() == 1.0
    assert paired_auc_bootstrap(W, W, n=1000)["ci95"] == [0.0, 0.0]
    c = paired_auc_bootstrap(win_matrix(perfect, labels), W, n=1000)
    assert c["auc_diff"] == pytest.approx(1 - brute) and c["ci95"][0] > 0


class PositionLogits(torch.nn.Module):
    """logits[:, j] = Z[j] for every row: log pi of each response token is log_softmax(Z[column - 1]), known in closed form."""

    def __init__(self, T, V):
        super().__init__()
        self.Z = torch.nn.Parameter(torch.randn(T, V, generator=torch.Generator().manual_seed(3), dtype=torch.float64))

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(logits=self.Z.expand(input_ids.shape[0], -1, -1))


def test_b2_opposite_labels_hand_rewards_and_independent_reference_gradient():
    POS, NEG, TAIL, EOS, JUNK = range(5)
    toks = {"POSITIVE": [POS, TAIL, EOS], "NEGATIVE": [NEG, TAIL, EOS], "x": [JUNK] * 4}  # "x": truncated, |o| = 4
    responses = ["POSITIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "x", "NEGATIVE",   # prompt 0, gold POSITIVE
                 "NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "POSITIVE"]  # prompt 1, gold NEGATIVE
    rewards = group_rewards(responses, ["POSITIVE", "NEGATIVE"], 6)
    assert rewards.tolist() == [[1, 1, 0, 1, 0, 0], [1, 1, 1, 1, 1, 0]]
    a0 = 0.5 / (0.5 + 1e-4)                            # group 0: mean 1/2, population std 1/2
    s1 = math.sqrt(5) / 6                              # group 1: mean 5/6, population std sqrt(5/36)
    adv = [a0, a0, -a0, a0, -a0, -a0] + [(1 / 6) / (s1 + 1e-4)] * 5 + [-(5 / 6) / (s1 + 1e-4)]
    torch.testing.assert_close(group_advantages(rewards).flatten().double(), torch.tensor(adv, dtype=torch.float64))

    P, R = 2, 4
    ids = torch.full((12, P + R), EOS)
    for i, s in enumerate(responses):
        ids[i, P : P + len(toks[s])] = torch.tensor(toks[s])
    mask = torch.zeros(12, R, dtype=torch.bool)
    for i, s in enumerate(responses):
        mask[i, : len(toks[s])] = True
    r = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "completion_mask": mask, "prompt_len": P}

    model = PositionLogits(P + R, 5)
    Z0 = model.Z.detach().clone()
    opt = torch.optim.SGD([model.Z], lr=1.0)
    assert grpo_step(model, {"Z": model.Z}, opt, r, rewards) is not None

    # reference: dL/dZ[P-1+t] = -(1/N) sum_i (A_i / |o_i|) (e_{y_it} - softmax(Z[P-1+t])), written without grpo_loss
    grad = torch.zeros_like(Z0)
    for i, s in enumerate(responses):
        for t, y in enumerate(toks[s]):
            e = torch.zeros(5, dtype=torch.float64)
            e[y] = 1.0
            grad[P - 1 + t] -= adv[i] / len(toks[s]) * (e - torch.softmax(Z0[P - 1 + t], -1)) / 12
    torch.testing.assert_close(model.Z.detach() - Z0, -grad)  # SGD, lr 1: dZ = -grad
    # decision position: d = (1/12) [sum_i (A_i/|o_i|) e_{y_i0} - C pi], C = sum_i A_i/|o_i|.
    # C = a0 - 2a0/3 - a0/4 = a0/12 from group 0 (the truncated |o| = 4 row) and 5c/3 + w/3 = 0 from group 1.
    d = (model.Z.detach() - Z0)[P - 1].tolist()
    pi0, C = torch.softmax(Z0[P - 1], -1).tolist(), a0 / 12
    assert d[TAIL] == pytest.approx(-C * pi0[TAIL] / 12, abs=1e-12)
    # POS: +a0 from group 0's correct rows, -0.745 from group 1's wrong POSITIVE row (A < 0), -C pi; JUNK: A < 0 and -C pi < 0
    assert d[POS] > 0 and d[JUNK] < 0
    assert d[POS] - d[NEG] == pytest.approx((5 * a0 / 3 - (10 / 6) / (s1 + 1e-4) / 3 - C * (pi0[POS] - pi0[NEG])) / 12, abs=1e-12)
