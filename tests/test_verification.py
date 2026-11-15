"""Verification and reporting (CPU only): paired statistics, policy change, parameter deltas, merge equivalence on a tiny
Qwen3 with LoRA, adapter loading without fallback, reward integrity, control provenance, and reports from saved files."""
import copy
import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch
from peft import get_peft_model
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import config, reporting, verification
from qwen_grpo.evaluation import metrics, paired_f1_bootstrap
from qwen_grpo.model import TARGETS, lora_config, lora_modules

ROOT = Path(__file__).parents[1]
CFG = config.load(ROOT / "configs" / "default.yaml")
STATS = CFG.verification.statistics


@pytest.fixture(scope="module", autouse=True)
def float32_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")


def preds(labels, predicted, start=0):
    return [{"id": start + i, "text": f"t{i}", "label": g, "response": str(p), "prediction": p, "correct": p == g}
            for i, (g, p) in enumerate(zip(labels, predicted))]


def ev(labels, predicted, scores):
    p = preds(labels, predicted)
    from qwen_grpo.evaluation import win_matrix
    return {"predictions": p, "scores": scores, "metrics": metrics(p), "auc": win_matrix(scores, labels).mean().item()}


LABELS = ["POSITIVE", "NEGATIVE"] * 20
BASE = ev(LABELS, ["POSITIVE", "POSITIVE", "NEGATIVE", None] * 10, [float(i % 7) - 3 for i in range(40)])
NEW = ev(LABELS, ["POSITIVE", "NEGATIVE"] * 20, [(1.0 if g == "POSITIVE" else -1.0) + 0.01 * i for i, g in enumerate(LABELS)])


# --- statistics --------------------------------------------------------------------------------------------------

def test_f1_bootstrap_observed_value_reproducible_and_zero_for_identical_models():
    a = paired_f1_bootstrap(BASE["predictions"], NEW["predictions"], n=500, chunk=100)
    assert a["f1_diff"] == pytest.approx(NEW["metrics"]["macro_f1"] - BASE["metrics"]["macro_f1"], abs=1e-12)
    assert a == paired_f1_bootstrap(BASE["predictions"], NEW["predictions"], n=500, chunk=100)
    assert a["ci"][0] <= a["f1_diff"] <= a["ci"][1]
    same = paired_f1_bootstrap(BASE["predictions"], BASE["predictions"], n=500, chunk=100)
    assert same == {"f1_diff": 0.0, "ci": [0.0, 0.0]}
    narrow = paired_f1_bootstrap(BASE["predictions"], NEW["predictions"], n=500, chunk=100, level=0.5)
    assert narrow["ci"][1] - narrow["ci"][0] < a["ci"][1] - a["ci"][0]


def test_compare_is_paired_and_flags_intervals_excluding_zero():
    c = verification.compare(BASE, NEW, STATS)
    assert c["n"] == 40 and c["accuracy"]["diff"] == pytest.approx(1.0 - BASE["metrics"]["accuracy"])
    assert c["accuracy"]["ci_excludes_zero"] and c["accuracy"]["ci"][0] > 0
    same = verification.compare(BASE, BASE, STATS)
    assert not any(same[k]["ci_excludes_zero"] for k in ("accuracy", "macro_f1", "auc"))
    shifted = copy.deepcopy(NEW)
    shifted["predictions"] = shifted["predictions"][1:] + shifted["predictions"][:1]
    with pytest.raises(AssertionError):
        verification.compare(BASE, shifted, STATS)


def test_invalid_predictions_count_as_wrong_and_stay_in_denominators():
    m = BASE["metrics"]
    assert m["n"] == 40 and m["invalid_rate"] == 0.25 and m["confusion"]["None"] == {"NEGATIVE": 10, "POSITIVE": 0}
    assert m["accuracy"] == 10 / 40  # only position 0 of every 4 is right; the invalid position 3 counts as wrong


def test_policy_change_by_hand():
    pc = verification.policy_change(NEW, BASE)
    ds = [n - b for n, b in zip(NEW["scores"], BASE["scores"])]
    assert pc["max_abs_score_change"] == pytest.approx(max(map(abs, ds)))
    assert pc["mean_abs_score_change"] == pytest.approx(sum(map(abs, ds)) / 40)
    assert pc["prediction_disagreement_rate"] == 0.75  # positions 1, 2, 3 of every 4 differ
    assert (pc["improved"], pc["degraded"], pc["correctness_changed"]) == (30, 0, 30)


# --- parameters and merging --------------------------------------------------------------------------------------

def tiny_lora(tok, targets="last2-attention", seed=0):
    torch.manual_seed(seed)
    base = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=28,
                                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True)).eval()
    model = get_peft_model(base, lora_config(TARGETS[targets])).eval()
    for _, m in lora_modules(model):  # a trained-looking adapter: B != 0
        torch.nn.init.normal_(m.lora_B["default"].weight, std=0.5)
    return model


def test_parameter_updates_raw_and_effective_by_hand(tok):
    model = tiny_lora(tok, "last-o-proj")
    trained = verification.adapter_state(model)
    init = {k: (torch.zeros_like(v) if "lora_B" in k else v.clone()) for k, v in trained.items()}
    init[next(k for k in init if "lora_A" in k)] += 1.0  # A also moved by +1 everywhere
    u = verification.parameter_updates(init, trained, lora_modules(model), 2.0)
    kA, kB = (next(k for k in trained if f"lora_{x}" in k) for x in "AB")
    assert u["tensors"][kB]["delta"] == pytest.approx(trained[kB].norm().item()) and u["tensors"][kB]["relative"] is None
    assert u["tensors"][kA]["delta"] == pytest.approx(math.sqrt(trained[kA].numel()))
    dW = 2.0 * trained[kB] @ trained[kA]  # B0 = 0, so B0 A0 = 0
    (name, m), = lora_modules(model)
    assert u["effective"][name]["delta_frobenius"] == pytest.approx(dW.norm().item(), rel=1e-6)
    assert u["effective"][name]["relative_to_W0"] == pytest.approx(dW.norm().item() / m.base_layer.weight.norm().item(), rel=1e-6)
    assert u["tensors_changed"] == u["tensors_total"] == 2


def test_merge_equivalence_fp32_passes_with_power(tok):
    model = tiny_lora(tok)
    subset = [{"text": "A gorgeous, witty film."}, {"text": "Stale and tedious."}]
    merged, e = verification.merge_check(model, tok, subset, 2.0)
    assert not verification.lora_modules(merged)
    assert e["adapter_vs_merged"]["relative"] < verification.FP32_MERGE_RTOL
    assert e["power_ratio"] > verification.POWER_RATIO and verification.merge_status(e) == "PASS"
    assert all(w["max_abs_error"] <= 4 * w["dtype_eps_times_max_weight"] for w in e["weights"].values())


def test_wrong_merge_fails_and_dropped_adapter_is_not_a_pass(tok):
    subset = [{"text": "A gorgeous, witty film."}]
    model = tiny_lora(tok)
    z_adapter = verification.sequence_logits(model, tok, subset)
    with model.disable_adapter():
        z_base = verification.sequence_logits(model, tok, subset)
    merged = model.merge_and_unload()
    with torch.no_grad():
        merged.model.layers[27].self_attn.o_proj.weight.mul_(0.9)  # a merge that got one module wrong
    bad = {"adapter_vs_merged": verification.logit_diff(z_adapter, verification.sequence_logits(merged, tok, subset)),
           "adapter_vs_base": verification.logit_diff(z_adapter, z_base)}
    bad["power_ratio"] = bad["adapter_vs_base"]["max_abs"] / bad["adapter_vs_merged"]["max_abs"]
    assert verification.merge_status(bad) == "FAIL"
    # a merge that silently dropped the adapter equals the base: it fails unless the adapter has no effect at all
    dropped = {"adapter_vs_merged": verification.logit_diff(z_adapter, z_base), "adapter_vs_base": verification.logit_diff(z_adapter, z_base)}
    dropped["power_ratio"] = 1.0
    assert verification.merge_status(dropped) != "PASS"


def test_adapter_loading_takes_effect_and_rejects_other_targets(tok, tmp_path):
    trained = tiny_lora(tok, seed=1)
    trained.save_pretrained(tmp_path / "ckpt")
    fresh = tiny_lora(tok, seed=1)
    for _, m in lora_modules(fresh):
        torch.nn.init.zeros_(m.lora_B["default"].weight)  # behaves as the base model
    subset = [{"text": "Stale."}]
    before = verification.sequence_logits(fresh, tok, subset)
    verification.load_adapter(fresh, tmp_path / "ckpt")
    after = verification.sequence_logits(fresh, tok, subset)
    assert verification.logit_diff(after, verification.sequence_logits(trained, tok, subset))["max_abs"] == 0
    assert verification.logit_diff(after, before)["max_abs"] > 1e-3  # no silent fallback to the base model
    with pytest.raises(SystemExit, match="adapter tensors differ"):
        verification.load_adapter(tiny_lora(tok, "last-o-proj"), tmp_path / "ckpt")


# --- reward integrity --------------------------------------------------------------------------------------------

def test_reward_probes_and_independent_check():
    ri = verification.reward_integrity({"m": {"predictions": preds(["POSITIVE"] * 3, ["POSITIVE"] * 3)}}, None, [])
    accepted = {p["response"] for p in ri["probes"] if p["training_reward"] == 1.0}
    assert accepted == {"POSITIVE", "positive", " POSITIVE ", "POSITIVE\n\n"}  # strip + upper, exact; no substrings
    assert {p["response"] for p in ri["probes"] if p["strict_correct"]} == {"POSITIVE"}
    p = preds(["POSITIVE", "NEGATIVE", "POSITIVE"], ["POSITIVE", "NEGATIVE", "POSITIVE"])
    p[1]["response"], p[2]["response"] = "negative", "POSITIVE because good"
    p[2]["prediction"], p[2]["correct"] = None, False
    r = verification.reward_integrity({"m": {"predictions": p}}, None, [])["models"]["m"]
    assert r["disagreements"] == 1 and r["examples"][0]["response"] == "negative" and r["logged_correct_mismatches"] == 0


def test_training_rollouts_rescored_from_the_log():
    log = {"training": [{"attempt": 1, "prompt_ids": [5, 6], "labels": ["POSITIVE", "NEGATIVE"],
                         "responses": ["POSITIVE", "positive", "NEGATIVE", "POSITIVE"], "rewards": [[1.0, 1.0], [1.0, 0.0]]}],
           "zero_signal": [{"attempt": 2, "prompt_ids": [7], "labels": ["NEGATIVE"], "responses": ["x", "y"], "rewards": [[0.0, 0.0]]}]}
    ex = [{"id": i, "text": f"s{i}"} for i in range(8)]
    tr = verification.reward_integrity({}, log, ex)["training_rollouts"]
    assert tr["n"] == 6 and tr["disagreements"] == 1 and tr["logged_reward_rescore_mismatches"] == 0
    assert tr["examples"][0] == {"attempt": 1, "input": "s5", "response": "positive", "label": "POSITIVE",
                                 "training_reward": 1.0, "strict_correct": False}


# --- provenance --------------------------------------------------------------------------------------------------

def control_log(tmp_path, cfg, **edit):
    y = dataclasses.asdict(cfg)
    y["run"]["name"], y["training"]["learning_rate"], y["training"]["full_val"] = "exp_lr0", 0.0, False
    for k, v in edit.items():
        sec, key = k.split("__")
        y[sec][key] = v
    ckpt = tmp_path / "checkpoints" / "exp_lr0_step30"
    log = {"yaml_config": y, "final_checkpoint": {"path": verification.rel(ckpt)},
           "control": {"params_equal_init": True, "subset_predictions_unchanged": True}, "frozen_changed": []}
    (tmp_path / "results" / "exp_lr0").mkdir(parents=True, exist_ok=True)
    (tmp_path / "results" / "exp_lr0" / "training.json").write_text(json.dumps(log))
    return ckpt


def test_control_provenance(tmp_path):
    ok, why, _ = verification.control_provenance(CFG, control_log(tmp_path, CFG), tmp_path / "results")
    assert ok, why
    for edit, reason in [({"training__learning_rate": 1e-4}, "not 0"), ({"training__group_size": 4}, "differ"),
                         ({"lora__rank": 4}, "differ")]:
        ok, why, _ = verification.control_provenance(CFG, control_log(tmp_path, CFG, **edit), tmp_path / "results")
        assert not ok and reason in why
    ckpt = control_log(tmp_path, CFG)
    log = json.loads((tmp_path / "results" / "exp_lr0" / "training.json").read_text())
    for broken, reason in [({**log, "control": {"params_equal_init": False}}, "control checks"),
                           ({k: v for k, v in log.items() if k != "yaml_config"}, "predates YAML")]:
        (tmp_path / "results" / "exp_lr0" / "training.json").write_text(json.dumps(broken))
        ok, why, _ = verification.control_provenance(CFG, ckpt, tmp_path / "results")
        assert not ok and reason in why


def test_saved_baseline_reused_only_when_compatible():
    from qwen_grpo.sentiment import load_splits
    examples = [{"id": i, **e} for i, e in enumerate(load_splits()["validation"])]
    base, src = verification.saved_baseline(CFG, examples, "validation")
    assert src == "results/baseline/validation.json" and round(base["metrics"]["accuracy"], 4) == 0.7486
    assert round(base["auc"], 4) == 0.8277
    fp32 = dataclasses.replace(CFG, model=dataclasses.replace(CFG.model, dtype="float32"))
    assert verification.saved_baseline(fp32, examples, "validation")[0] is None
    assert verification.saved_baseline(CFG, examples[:-1], "validation")[0] is None


# --- reporting ---------------------------------------------------------------------------------------------------

def synthetic_log():
    rows = [{"step": s, "attempt": s + (s > 2), "mixed_groups": 1 + (s % 2), "mean_reward": 0.5 + 0.01 * s, "grad_norm": 0.3,
             "loss": 1e-8, "init_diff": 0.0} for s in range(1, 6)]
    return {"config": {"B": 2, "lr": 1e-4, "grad_clipping": None}, "training": rows,
            "zero_signal": [{"attempt": 3, "mixed_groups": 0}], "groups_generated": 6, "successful_steps": 5,
            "zero_signal_groups": 1, "responses_generated": 72, "prompt_groups": 12, "frozen_changed": [], "params_equal_init": False,
            "evaluations": [{"step": s, "metrics": {"accuracy": 0.78 + 0.001 * s, "n": 160}} for s in (0, 5)]}


def synthetic_evidence(log_source):
    evals = {"base": BASE, "adapter": NEW, "merged": NEW}
    diff = {"max_abs": 1e-6, "relative": 1e-7, "logit_scale": 10.0, "last_position_argmax_agree": True}
    big = {**diff, "max_abs": 1.0, "relative": 0.1}
    return {
        "checkpoint": "checkpoints/exp_step5", "split": "validation", "n": 40, "label_balance": {"NEGATIVE": 20, "POSITIVE": 20},
        "config": dataclasses.asdict(CFG), "notes": {k: f"{k} note" for k in evals}, "evaluations": evals,
        "training_log": {"source": log_source, "missing_reason": None if log_source else "no training log at results/exp/training.json"},
        "training_diagnostics": verification.training_diagnostics(synthetic_log()) if log_source else None,
        "parameter_updates": {"status": "INCONCLUSIVE", "init_source": "reconstructed", "init_validated_by_control": None,
                              "total_delta": 0.1, "tensors_changed": 12, "tensors_total": 12, "only_lora_tensors_in_checkpoint": True,
                              "effective": {"model.layers.27.self_attn.o_proj": {"delta_frobenius": 0.02, "relative_to_W0": 1e-3}}},
        "policy_change": {k: verification.policy_change(evals[k], BASE) for k in ("adapter", "merged")},
        "merge_equivalence": {"fp32_cpu": {"status": "PASS", "n_examples": 8, "adapter_vs_merged": diff, "adapter_vs_base": big,
                                           "power_ratio": 1e6, "tolerance_relative": 1e-4, "weights": {"m": {"max_abs_error": 1e-8}}},
                              "bf16_gpu": {"status": "PASS", "adapter_vs_merged": diff, "adapter_vs_base": big,
                                           "full_split": verification.policy_change(NEW, NEW)}},
        "negative_control": {"status": "NOT_RUN", "reason": "verification.negative_control is not configured"},
        "confidence_intervals": {"adapter_vs_base": verification.compare(BASE, NEW, STATS)},
        "reward_integrity": verification.reward_integrity(evals, None, []),
        "gpu_memory": {"peak_allocated_gib": 1.5, "peak_reserved_gib": 1.9},
    }


def test_report_without_training_log_marks_missing_plots_and_models(tmp_path):
    reporting.save(synthetic_evidence(None), tmp_path)
    report = reporting.render(tmp_path).read_text()
    assert sorted(p.name for p in (tmp_path / "plots").iterdir()) == ["model_comparison.png"]
    assert "not generated: training plots" in report and "not run" in report  # LR=0 column
    assert "Training log: not available" in report and "NOT_RUN" in report
    assert json.loads((tmp_path / "evaluation" / "adapter.json").read_text())["predictions"][0]["response"] == "POSITIVE"


def test_report_with_training_log_plots_recorded_values(tmp_path):
    log_path = tmp_path / "training.json"
    log_path.write_text(json.dumps(synthetic_log()))
    reporting.save(synthetic_evidence(str(log_path)), tmp_path)
    reporting.render(tmp_path)
    names = {p.name for p in (tmp_path / "plots").iterdir()}
    assert names == {"reward.png", "validation_accuracy.png", "mixed_reward_fraction.png", "gradient_norm.png", "loss.png", "model_comparison.png"}
    d = verification.training_diagnostics(synthetic_log())
    assert d["mixed_reward_group_fraction"] == pytest.approx((2 + 1 + 2 + 1 + 2 + 0) / (2 * 6))  # all attempts in the denominator
    assert d["mean_completion_tokens"] is None and d["step_seconds"] is None  # not collected by that run: None, not invented
