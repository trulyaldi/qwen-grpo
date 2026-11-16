"""Verification of a trained LoRA checkpoint: base vs adapter-attached vs merged vs an LR=0 control, on held-out data.

Four kinds of evidence are kept apart: (1) adapter parameters changed, (2) the policy's outputs changed, (3) held-out
performance changed (paired bootstrap CIs), (4) merging reproduces the adapter-attached model. Each function returns
plain JSON-serializable data; reporting.py renders it. Statuses: PASS, FAIL, INCONCLUSIVE, NOT_RUN.
"""
import dataclasses
import gc
import json
import resource
from collections import Counter
from pathlib import Path

import torch
from peft import load_peft_weights

from qwen_grpo import runs
from qwen_grpo.checkpointing import adapter_state, check_adapter_matches, load_adapter, rel  # noqa: F401 (re-exported)
from qwen_grpo.config import normalized, sha256
from qwen_grpo.evaluation import (evaluate, label_scores, metrics, paired_auc_bootstrap, paired_bootstrap, paired_changes,
                                  paired_f1_bootstrap, settings, win_matrix)
from qwen_grpo.model import EOS_IDS, lora_model, lora_modules
from qwen_grpo.sentiment import LABELS, build_prompts, parse_label, reward

FP32_MERGE_RTOL = 1e-4  # see merge_check(): fp32 rounding of two algebraically equal computations, with margin
POWER_RATIO = 100  # the adapter must move logits >= 100x more than merging does, or the merge check has no power
DRIFT_GUARD = 32  # examples re-scored in-process to confirm a saved baseline still matches the current code
PROBES = ["POSITIVE", "NEGATIVE", "positive", " POSITIVE ", "POSITIVE\n\n", "POSITIVE.", "POSITIVE NEGATIVE",
          "POSITIVE because the movie was good", "The answer is POSITIVE", "**POSITIVE**", "POS", "NEGATIVE POSITIVE"]


def strict_correct(response: str, label: str) -> bool:
    """Independent correctness check: the response is exactly the canonical label (no case or whitespace folding)."""
    return response == label


# --- checkpoints -------------------------------------------------------------------------------------------------









def comparable(cfg_a: dict, cfg_b: dict, ignore):
    """Training configs equal except for the dotted keys in `ignore`. Both are normalized to the current schema first, so
    logs written before a key existed compare by its default. Settings that cannot change the trained adapter never
    matter: the verification and evaluation sections, run.gpu_budget_gib and run.overwrite."""
    ignore = {*ignore, "run.gpu_budget_gib", "run.overwrite"}
    strip = lambda c: {s: {k: v for k, v in sec.items() if f"{s}.{k}" not in ignore}
                       for s, sec in normalized(c).items() if s not in ("verification", "evaluation")}
    return strip(cfg_a) == strip(cfg_b)


def training_log(cfg, checkpoint):
    """The training log of this checkpoint, if it exists, was produced by this config, and lists this checkpoint."""
    path = runs.training_log_for(checkpoint)
    if path is None or not path.exists():
        return None, f"no training log at {rel(path) if path else f'for {checkpoint}'}"
    log = json.loads(path.read_text())
    if "yaml_config" not in log or not comparable(log["yaml_config"], dataclasses.asdict(cfg), set()):
        return None, f"{path.name} was not produced by this config"
    ckpts = {e.get("checkpoint") for e in log["evaluations"]} | {log.get("final_checkpoint", {}).get("path")}
    if rel(checkpoint) not in ckpts:
        return None, f"{path.name} does not list {rel(checkpoint)}"
    return log, rel(path)


def control_provenance(cfg, checkpoint):
    """A reusable LR=0 control: same config except run name, learning rate 0 and full_val, its own log lists the
    checkpoint, and that run's in-training control checks passed. Returns (ok, reason, log)."""
    path = runs.training_log_for(checkpoint)
    if path is None or not path.exists():
        return False, f"no training log for {checkpoint}", None
    log = json.loads(path.read_text())
    y = log.get("yaml_config")
    if y is None:
        return False, f"{path} predates YAML configs; its settings cannot be compared with this config", None
    if y["training"]["learning_rate"] != 0:
        return False, f"{path} has learning_rate {y['training']['learning_rate']}, not 0", None
    if not comparable(y, dataclasses.asdict(cfg), {"run.name", "training.learning_rate", "training.full_val"}):
        return False, f"{path}: settings other than run.name/learning_rate/full_val differ from this config", None
    if log.get("final_checkpoint", {}).get("path") != rel(checkpoint):
        return False, f"{path} does not end at {checkpoint}", None
    if not (log.get("control", {}).get("params_equal_init") and not log.get("frozen_changed")):
        return False, f"{path}: in-training control checks did not pass ({log.get('control')})", None
    return True, rel(path), log


# --- evidence ----------------------------------------------------------------------------------------------------

def evaluate_model(model, tok, examples, ev):
    """Greedy predictions and complete-label scores under config.Evaluation `ev`; the model is used exactly as given."""
    gen, score = settings(ev)
    preds = evaluate(model, tok, examples, **gen)
    scores = label_scores(model, tok, examples, **score)
    return {"predictions": preds, "scores": scores, "metrics": metrics(preds),
            "auc": win_matrix(scores, [e["label"] for e in examples]).mean().item()}


def saved_baseline(cfg, examples, split):
    """The saved base predictions if they were produced under the current evaluation settings for these examples."""
    path = runs.baseline_path(split)
    if not path.exists():
        return None, f"{path} not found"
    b = json.loads(path.read_text())
    checks = {
        "model": b.get("model") == cfg.model.name,
        "dtype": cfg.model.dtype == "bfloat16",  # research/baseline.py evaluates in bf16
        "split": b.get("split") == split,
        "decoding": b.get("decoding") == {"do_sample": False, "max_new_tokens": cfg.evaluation.max_new_tokens},
        "examples": [(p["id"], p["text"], p["label"]) for p in b["predictions"]] == [(e["id"], e["text"], e["label"]) for e in examples],
        "score_ids": b.get("label_scores", {}).get("ids") == [e["id"] for e in examples],
    }
    if not all(checks.values()):
        return None, f"incompatible: {[k for k, ok in checks.items() if not ok]}"
    preds, scores = b["predictions"], b["label_scores"]["scores"]
    return {"predictions": preds, "scores": scores, "metrics": metrics(preds),
            "auc": win_matrix(scores, [e["label"] for e in examples]).mean().item()}, rel(path)


def initial_adapter(init_ckpt, reconstructed, seed, control_ev):
    """(state on CPU, source, validated): the run's saved <run>_step0 adapter if present, else the seeded reconstruction,
    which counts as validated only if an LR=0 control of the same config equals it bitwise."""
    if Path(init_ckpt).exists():
        return {k: t.float().cpu() for k, t in load_peft_weights(str(init_ckpt)).items()}, f"saved {rel(init_ckpt)}", True
    return (reconstructed, f"reconstructed: torch.manual_seed({seed}) + lora_model(), as in train()",
            control_ev["equals_reconstructed_init"] if control_ev else None)


def parameter_updates(init, trained, modules, scale):
    """Raw tensor changes ||theta - theta0|| and effective changes ||(a/r)(BA - B0A0)||_F relative to ||W0||_F.

    init, trained: adapter state dicts (PEFT keys); modules: [(name, module)] with base weights W0."""
    tensors = {}
    for k in trained:
        d, n0 = (trained[k] - init[k]).norm().item(), init[k].norm().item()
        tensors[k] = {"delta": d, "init_norm": n0, "relative": d / n0 if n0 > 0 else None, "changed": d > 0}
    effective = {}
    for name, m in modules:
        key = lambda f: next(k for k in trained if k.endswith(f"{name}.{f}.weight"))
        A, B, A0, B0 = trained[key("lora_A")], trained[key("lora_B")], init[key("lora_A")], init[key("lora_B")]
        dW = scale * (B @ A - B0 @ A0)
        W0 = m.base_layer.weight.detach().float().cpu()
        effective[name] = {"delta_frobenius": dW.norm().item(), "relative_to_W0": dW.norm().item() / W0.norm().item(),
                           "trained_frobenius": (scale * B @ A).norm().item()}
    total = sum(v["delta"] ** 2 for v in tensors.values()) ** 0.5
    return {"tensors": tensors, "effective": effective, "total_delta": total,
            "tensors_changed": sum(v["changed"] for v in tensors.values()), "tensors_total": len(tensors)}


def policy_change(new, ref):
    """Output-level change of `new` relative to `ref` on the same examples (no claim about benefit)."""
    assert [p["id"] for p in new["predictions"]] == [p["id"] for p in ref["predictions"]]
    ds = torch.tensor(new["scores"], dtype=torch.float64) - torch.tensor(ref["scores"], dtype=torch.float64)
    pairs = list(zip(ref["predictions"], new["predictions"]))
    return {"n": len(pairs), "mean_abs_score_change": ds.abs().mean().item(), "max_abs_score_change": ds.abs().max().item(),
            "mean_score_change": ds.mean().item(),
            "prediction_disagreement_rate": sum(a["prediction"] != b["prediction"] for a, b in pairs) / len(pairs),
            "correctness_changed": sum(a["correct"] != b["correct"] for a, b in pairs),
            "improved": sum(not a["correct"] and b["correct"] for a, b in pairs),
            "degraded": sum(a["correct"] and not b["correct"] for a, b in pairs)}


def compare(ref, new, stats):
    """new - ref on identical examples: accuracy, macro-F1 and AUC differences with paired bootstrap CIs."""
    kw = dict(n=stats.bootstrap_samples, seed=stats.seed, chunk=100, level=stats.confidence_level)
    changes = paired_changes(ref["predictions"], new["predictions"])  # asserts (id, text, label) alignment
    d, lo, hi = paired_bootstrap([p["correct"] for p in ref["predictions"]], [p["correct"] for p in new["predictions"]], **kw)
    labels = [p["label"] for p in ref["predictions"]]
    auc = paired_auc_bootstrap(win_matrix(new["scores"], labels), win_matrix(ref["scores"], labels), **kw)
    f1 = paired_f1_bootstrap(ref["predictions"], new["predictions"], **kw)
    out = {"n": len(labels), "level": stats.confidence_level, "resamples": stats.bootstrap_samples, "seed": stats.seed,
           "accuracy": {"ref": ref["metrics"]["accuracy"], "new": new["metrics"]["accuracy"], "diff": d, "ci": [lo, hi]},
           "macro_f1": {"ref": ref["metrics"]["macro_f1"], "new": new["metrics"]["macro_f1"], "diff": f1["f1_diff"], "ci": f1["ci"]},
           "auc": {"ref": ref["auc"], "new": new["auc"], "diff": auc["auc_diff"], "ci": auc["ci"]},
           "paired_changes": changes}
    for k in ("accuracy", "macro_f1", "auc"):
        lo, hi = out[k]["ci"]
        out[k]["ci_excludes_zero"] = lo > 0 or hi < 0
    return out


def sequence_logits(model, tok, examples):
    """Full logits [T, V] (fp32, CPU) of prompt + POSITIVE<|im_end|>, one unpadded example at a time."""
    cont = tok("POSITIVE", add_special_tokens=False)["input_ids"] + [EOS_IDS[0]]
    out = []
    with torch.no_grad():
        for p in build_prompts(tok, [e["text"] for e in examples]):
            ids = torch.tensor([tok(p, add_special_tokens=False)["input_ids"] + cont], device=model.device)
            out.append(model(input_ids=ids, use_cache=False).logits[0].float().cpu())
    return out


def logit_diff(a, b):
    """E_inf = max |a - b|; relative = E_inf / max |a| (the logit scale), over all positions of all examples.
    One pass over zip(a, b), so b may be a generator: its logits are never all held at once."""
    e = scale = 0.0
    agree = True
    for x, y in zip(a, b, strict=True):
        e, scale = max(e, (x - y).abs().max().item()), max(scale, x.abs().max().item())
        agree &= bool(x[-1].argmax() == y[-1].argmax())
    return {"max_abs": e, "relative": e / scale, "logit_scale": scale, "last_position_argmax_agree": agree}


def iter_logits(model, tok, examples):
    for e in examples:
        yield sequence_logits(model, tok, [e])[0]


def weight_merge_error(model, expected_deltas, W0):
    """After merge_and_unload: max |(W_merged - W0) - (a/r)BA| per module, and the rounding scale of W_merged's dtype."""
    out = {}
    named = dict(model.named_modules())
    for name, dW in expected_deltas.items():
        Wm = named[name].weight.detach()
        residual = (Wm.float().cpu() - W0[name].float()) - dW
        out[name] = {"max_abs_error": residual.abs().max().item(),
                     "dtype_eps_times_max_weight": torch.finfo(Wm.dtype).eps * Wm.abs().max().item(),
                     # how much of the intended update is lost or distorted by storing W0 + dW in the base dtype
                     "error_relative_to_delta": (residual.norm() / dW.norm()).item()}
    return out


def merge_inputs(model, modules, scale):
    """Before merging: base weights W0 and expected deltas (a/r)BA per module, on CPU."""
    W0 = {n: m.base_layer.weight.detach().cpu().clone() for n, m in modules}
    dW = {n: scale * (m.lora_B["default"].weight.detach().float().cpu() @ m.lora_A["default"].weight.detach().float().cpu())
          for n, m in modules}
    return W0, dW


def merge_check(model, tok, subset, scale):
    """In place: logits of base (adapter disabled) and adapter-attached, then merge_and_unload and logits of the merged
    model. Returns (merged plain model, evidence). The adapter-vs-base distance gives the check its power: a merge that
    silently dropped the adapter would agree with base, not with the adapter."""
    modules = lora_modules(model)
    W0, dW = merge_inputs(model, modules, scale)
    z_adapter = sequence_logits(model, tok, subset)  # the only full-logit set held in memory
    with model.disable_adapter():
        vs_base = logit_diff(z_adapter, iter_logits(model, tok, subset))
    merged = model.merge_and_unload()
    assert not lora_modules(merged), "LoRA layers remain after merge_and_unload"
    ev = {"dtype": str(next(merged.parameters()).dtype), "n_examples": len(subset),
          "adapter_vs_merged": logit_diff(z_adapter, iter_logits(merged, tok, subset)), "adapter_vs_base": vs_base,
          "weights": weight_merge_error(merged, dW, W0)}
    del z_adapter
    ev["power_ratio"] = ev["adapter_vs_base"]["max_abs"] / max(ev["adapter_vs_merged"]["max_abs"], 1e-30)
    return merged, ev


def fp32_merge_check(cfg, checkpoint, subset, tok):
    """Merge correctness in fp32 on CPU: adapter path W0x + (a/r)B(Ax) and merged path (W0 + (a/r)BA)x are algebraically
    equal, so in fp32 they differ only by rounding (~1e-7 per op, accumulated over 28 layers): relative error far below
    FP32_MERGE_RTOL = 1e-4, while a wrong merge (missing scale, wrong module, dropped adapter) shifts logits by about
    the adapter's own effect."""
    lo = cfg.lora
    model = lora_model(torch.float32, "cpu", lo.spec(), lo.rank, lo.alpha, lo.dropout, cfg.model.name)
    load_adapter(model, checkpoint)
    merged, ev = merge_check(model, tok, subset, lo.alpha / lo.rank)
    del merged, model
    ev["tolerance_relative"], ev["status"] = FP32_MERGE_RTOL, merge_status(ev)
    return ev


def merge_status(ev):
    """FAIL if merged logits leave the fp32 tolerance; INCONCLUSIVE if the adapter barely moves logits (no power)."""
    d = ev["adapter_vs_merged"]
    if d["relative"] > FP32_MERGE_RTOL or not d["last_position_argmax_agree"]:
        return "FAIL"
    return "PASS" if ev["power_ratio"] >= POWER_RATIO else "INCONCLUSIVE"


def reward_integrity(evals, train_log, train_examples):
    """Which strings the training reward accepts, and where it disagrees with the strict canonical-label check, on the
    logged training rollouts and on every evaluated model's saved responses."""
    probes = [{"response": r, "label": "POSITIVE", "parsed": parse_label(r), "training_reward": reward(r, "POSITIVE"),
               "strict_correct": strict_correct(r, "POSITIVE")} for r in PROBES]

    def scan(rows):
        dis = [r for r in rows if (r["training_reward"] == 1.0) != r["strict_correct"]]
        return {"n": len(rows), "disagreements": len(dis), "disagreement_rate": len(dis) / len(rows) if rows else None,
                "examples": dis[:50]}

    out = {"probes": probes, "parser_contract": "strip() + upper() + exact match to POSITIVE/NEGATIVE",
           "strict_contract": "response == canonical label", "models": {}, "training_rollouts": None}
    for name, ev in evals.items():
        rows = [{"id": p["id"], "input": p["text"], "response": p["response"], "label": p["label"],
                 "training_reward": reward(p["response"], p["label"]), "strict_correct": strict_correct(p["response"], p["label"]),
                 "logged_correct": p["correct"]} for p in ev["predictions"]]
        out["models"][name] = {**scan(rows), "logged_correct_mismatches": sum(r["logged_correct"] != (r["training_reward"] == 1.0) for r in rows),
                               "response_forms": dict(Counter(p["response"] for p in ev["predictions"]).most_common(10)),
                               "invalid_rate": sum(p["prediction"] is None for p in ev["predictions"]) / len(ev["predictions"])}
    if train_log is not None:
        by_id = {e["id"]: e for e in train_examples}
        rows, rescored = [], 0
        for g in {g["attempt"]: g for g in train_log["training"] + train_log["zero_signal"]}.values():  # one per rollout batch
            G = len(g["responses"]) // len(g["labels"])
            for i, r in enumerate(g["responses"]):
                label, logged = g["labels"][i // G], g["rewards"][i // G][i % G]
                rescored += reward(r, label) != logged
                rows.append({"attempt": g["attempt"], "input": by_id[g["prompt_ids"][i // G]]["text"], "response": r, "label": label,
                             "training_reward": logged, "strict_correct": strict_correct(r, label)})
        out["training_rollouts"] = {**scan(rows), "logged_reward_rescore_mismatches": rescored,
                                    "response_forms": dict(Counter(r["response"] for r in rows).most_common(10))}
    return out


def training_diagnostics(log):
    """Summary of the saved training log; fields the run did not record are reported as None (not collected)."""
    if log is None:
        return None
    rows, cfg = log["training"], log["config"]
    B = cfg["B"]
    attempts = sorted({g["attempt"]: g for g in rows + log["zero_signal"]}.values(), key=lambda g: g["attempt"])  # K rows per batch
    get = lambda k: [r.get(k) for r in rows] if rows and all(k in r for r in rows) else None
    return {
        "rollout_attempts": log["groups_generated"], "optimizer_updates": log["successful_steps"],
        "zero_signal_batches": log["zero_signal_groups"], "completions_generated": log["responses_generated"],
        "prompt_groups": log["prompt_groups"],
        "mixed_reward_group_fraction": sum(g["mixed_groups"] for g in attempts) / (B * len(attempts)) if attempts else None,
        "frozen_changed": log["frozen_changed"], "params_equal_init": log["params_equal_init"],
        "max_init_diff": max(r["init_diff"] for r in rows if "init_diff" in r) if rows else None,  # first update per batch
        "num_iterations": cfg.get("num_iterations", 1),
        "ratio_range": [min(r["ratio_min"] for r in rows), max(r["ratio_max"] for r in rows)] if get("ratio_min") else None,
        "max_clip_frac": max(r["clip_frac"] for r in rows) if get("clip_frac") else None,
        "grad_norm_range": [min(r["grad_norm"] for r in rows), max(r["grad_norm"] for r in rows)] if rows else None,
        "grad_clipping": cfg.get("grad_clipping"),
        "learning_rate": cfg["lr"], "per_step_learning_rate": get("learning_rate"),
        "mean_completion_tokens": get("mean_completion_tokens"), "step_seconds": get("step_seconds"),
        "peak_reserved_gib": log.get("memory", {}).get("peak_reserved_gib"), "status": log.get("status"),
    }


def label_balance(examples):
    return {label: sum(e["label"] == label for e in examples) for label in LABELS}


# --- orchestration -----------------------------------------------------------------------------------------------

def verify(cfg, checkpoint, control=None):
    """All evidence for one checkpoint, one model instance on the GPU at a time. `control` is an LR=0 checkpoint whose
    provenance the caller has checked (control_provenance), or None. Returns the evidence dict (also what is saved)."""
    from transformers import AutoTokenizer

    from qwen_grpo.config import DTYPES
    from qwen_grpo.sentiment import load_splits

    v, lo = cfg.verification, cfg.lora
    scale = lo.alpha / lo.rank
    checkpoint = Path(checkpoint).resolve()
    check_adapter_matches(cfg, checkpoint)
    splits = load_splits()
    examples = [{"id": i, **e} for i, e in enumerate(splits[v.split])]
    subset = examples[: v.merge_subset]
    log, log_source = training_log(cfg, checkpoint)
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")

    fp32 = fp32_merge_check(cfg, checkpoint, subset, tok)  # first: its ~4 GiB host peak never overlaps the CUDA process state
    gc.collect()
    rss_fp32 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GiB (ru_maxrss is in KiB on Linux)
    torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(cfg.training.seed)  # the same seeded construction as train(): a fresh adapter equals the run's init
    model = lora_model(DTYPES[cfg.model.dtype], "cuda", lo.spec(), lo.rank, lo.alpha, lo.dropout, cfg.model.name)
    reconstructed = adapter_state(model)
    assert all(t.abs().max() == 0 for k, t in reconstructed.items() if "lora_B" in k)
    evals, notes = {}, {}

    base, why = saved_baseline(cfg, examples, v.split)
    if base is not None:  # drift guard: the saved file must match the current prompt/tokenization/model code exactly
        with model.disable_adapter():
            head = evaluate_model(model, tok, examples[:DRIFT_GUARD], cfg.evaluation)
        same = ([p["response"] for p in head["predictions"]] == [p["response"] for p in base["predictions"][:DRIFT_GUARD]]
                and head["scores"] == base["scores"][:DRIFT_GUARD])
        if not same:
            base, why = None, f"saved baseline ({why}) differs from in-process base on the first {DRIFT_GUARD} examples"
        else:
            notes["base"] = f"reused {why}; first {DRIFT_GUARD} examples re-scored in-process: identical"
    if base is None:
        with model.disable_adapter():
            base = evaluate_model(model, tok, examples, cfg.evaluation)
        notes["base"] = f"evaluated in-process with the adapter disabled ({why})"
    evals["base"] = base

    init_ckpt = runs.sibling(checkpoint, 0)
    control_ev = None
    if control is not None:
        ctrl_state = load_adapter(model, control)
        evals["lr_zero"] = evaluate_model(model, tok, examples, cfg.evaluation)
        control_ev = {"checkpoint": rel(control),
                      "equals_reconstructed_init": all(torch.equal(ctrl_state[k], reconstructed[k]) for k in reconstructed),
                      "lora_B_all_zero": all(t.abs().max() == 0 for k, t in ctrl_state.items() if "lora_B" in k),
                      "policy_change_vs_base": policy_change(evals["lr_zero"], base)}
        c = control_ev
        c["status"] = "PASS" if (c["equals_reconstructed_init"] and c["policy_change_vs_base"]["prediction_disagreement_rate"] == 0
                                 and c["policy_change_vs_base"]["max_abs_score_change"] == 0) else "FAIL"
        notes["lr_zero"] = f"LR=0 control {rel(control)} evaluated with the adapter attached"

    init, init_source, init_validated = initial_adapter(init_ckpt, reconstructed, cfg.training.seed, control_ev)
    trained = load_adapter(model, checkpoint)
    updates = parameter_updates(init, trained, lora_modules(model), scale)
    updates.update(init_source=init_source, init_validated_by_control=init_validated,
                   only_lora_tensors_in_checkpoint=all(".lora_A." in k or ".lora_B." in k for k in trained),
                   status="PASS" if updates["total_delta"] > 0 and init_validated is not False else "FAIL")
    if init_validated is None:
        updates["status"] = "INCONCLUSIVE"  # deltas measured against an initialization nothing has confirmed
    evals["adapter"] = evaluate_model(model, tok, examples, cfg.evaluation)
    notes["adapter"] = f"{rel(checkpoint)} attached to {cfg.model.name} ({cfg.model.dtype})"

    merged, bf16 = merge_check(model, tok, subset, scale)  # in place on the GPU: no second model instance
    evals["merged"] = evaluate_model(merged, tok, examples, cfg.evaluation)
    notes["merged"] = f"PEFT merge_and_unload in {bf16['dtype']} (W0 + (a/r)BA rounded to the base dtype), in place"
    bf16["full_split"] = policy_change(evals["merged"], evals["adapter"])
    # behavior preservation at deployment precision; whether a failure is precision or a bug is decided by the fp32 check
    bf16["status"] = "PASS" if bf16["full_split"]["prediction_disagreement_rate"] == 0 else "FAIL"
    peak = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    del merged, model
    torch.cuda.empty_cache()
    comparisons = {f"{new}_vs_{ref}": compare(evals[ref], evals[new], v.statistics)
                   for new, ref in [("adapter", "base"), ("merged", "base"), ("merged", "adapter"), ("lr_zero", "base"), ("adapter", "lr_zero")]
                   if new in evals and ref in evals}
    train_examples = [{"id": i, **e} for i, e in enumerate(splits["train"])]
    return {
        "checkpoint": rel(checkpoint), "split": v.split, "n": len(examples), "label_balance": label_balance(examples),
        "config": dataclasses.asdict(cfg), "config_sha256": sha256(cfg), "notes": notes, "evaluations": evals,
        "training_log": {"source": log_source if log else None, "missing_reason": None if log else log_source},
        "training_diagnostics": training_diagnostics(log),
        "parameter_updates": updates,
        "policy_change": {k: policy_change(evals[k], base) for k in ("adapter", "merged", "lr_zero") if k in evals},
        "merge_equivalence": {"fp32_cpu": fp32, "bf16_gpu": bf16},
        "negative_control": control_ev or {"status": "NOT_RUN", "reason": "verification.negative_control is not configured"},
        "confidence_intervals": comparisons,
        "reward_integrity": reward_integrity(evals, log, train_examples),
        "gpu_memory": {**peak, "host_peak_rss_gib_after_fp32_check": rss_fp32,
                       "host_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20},
    }
