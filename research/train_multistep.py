"""Multi-step GRPO + LoRA on Qwen3-0.6B: fp32 adapters on selected projections of a frozen bf16 model.

Rollouts: T=1 sampling, MAX_NEW_TOKENS. Evaluation: evaluation.predict (greedy, evaluation.MAX_NEW tokens, BATCH chunks).
Checkpoints hold adapter weights only; optimizer and RNG state are not saved, so they cannot resume training exactly.
Outputs: results/<run>/training.json (config, per-step log, subset evaluations, memory), results/<run>/validation.json
(raw full-validation predictions and label scores of the final checkpoint), results/benchmarks/<run>.{json,md} (metrics).
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer

from qwen_grpo.evaluation import (BATCH, BOOTSTRAP, MAX_NEW, confusion_table, evaluate, label_scores, metrics, metrics_table,
                                  paired_auc_bootstrap, paired_bootstrap, paired_changes, paired_compare, summary, win_matrix)
from qwen_grpo.model import (ALPHA, DTYPE, EOS_IDS, EXPECTED_MODULES, MODEL, RANK, TARGETS, digest, expected_lora_params,
                             lora_model, lora_modules)
from qwen_grpo.policy import generate_rollouts
from qwen_grpo.sentiment import build_prompts, load_splits, parse_label
from qwen_grpo.training import balanced_subset, batches, check_grouping, decode, group_rewards, grpo_step
from research.baseline import BENCHMARKS, RESULTS, predictions_path

ROOT = Path(__file__).parents[1]
BASE_FILE = predictions_path("validation")
SUFFIX = {"last-o-proj": "", "last2-attention": "_last2"}
BATCH_MODES = ("cycle", "balanced", "unconstrained")
TRAIN_SAMPLES, VAL_SUBSET, G = 32, 160, 6
MAX_NEW_TOKENS, MAX_STEPS, EVAL_EVERY = 8, 30, 10
LEARNING_RATE, CLIP_EPS, SEED = 1e-4, 0.2, 42
GPU_BUDGET_GIB = 4.0


def output_name(targets, tag="", B=1, mode="cycle"):
    return f"grpo_lora{SUFFIX[targets]}" + (f"_b{B}_{mode}" if B > 1 else "") + tag


def _ref(targets, B=1, mode="cycle"):
    name = output_name(targets, B=B, mode=mode)
    return RESULTS / name / "validation.json", ROOT / "checkpoints" / f"{name}_step{MAX_STEPS}", targets


# Saved final (step-30) predictions (and adapters, for label scoring) of earlier runs: (predictions, checkpoint, targets).
REFERENCES = {
    "base": (BASE_FILE, None, None),
    "last-o-proj B=1": _ref("last-o-proj"),
    "last2 B=1": _ref("last2-attention"),
    "last2 B=2 balanced": _ref("last2-attention", 2, "balanced"),
    "last2 B=2 unconstrained": _ref("last2-attention", 2, "unconstrained"),
}


def l2(norms):
    """Aggregate L2 norm sqrt(sum_k ||theta_k||^2) from per-tensor norms."""
    return math.sqrt(sum(x * x for x in norms))


def write_benchmark(name, result):
    """results/benchmarks/<run>.json (final full-validation metrics vs base) and a Markdown rendering of it."""
    f, cfg = result["final"], result["config"]
    bench = {"run": name, "model": MODEL, "seed": result["seed"], "split": "validation",
             "checkpoint": result["final_checkpoint"]["path"], "step": result["final_checkpoint"]["step"],
             "predictions": str((RESULTS / name / "validation.json").relative_to(ROOT)),
             "training_log": str((RESULTS / name / "training.json").relative_to(ROOT)),
             "baseline_predictions": str(BASE_FILE.relative_to(ROOT)),
             "config": {k: v for k, v in cfg.items() if k not in ("train_ids", "val_subset_ids")},
             **f, "memory": result["memory"]}
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    (BENCHMARKS / f"{name}.json").write_text(json.dumps(bench, indent=1))

    ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
    lora = cfg["lora"]
    md = [f"# {name}: GRPO + LoRA vs baseline", "",
          f"{MODEL}, LoRA r={lora['r']} α={lora['alpha']} on {lora.get('targets', lora.get('target'))}, lr {cfg['lr']}, "
          f"{bench['step']} steps, B={cfg['B']} ({cfg['batch_mode']}), G={cfg['G']}, seed {result['seed']}. "
          f"Full validation (n={f['trained']['n']}), greedy, max_new_tokens = {cfg['eval_max_new_tokens']}. One training seed: CIs cover "
          "evaluation sampling only.", "",
          metrics_table({"base": {**f["base"], "auc": f["auc"]["base"]}, "trained": {**f["trained"], "auc": f["auc"]["trained"]}}),
          "", f"Accuracy trained − base: {f['accuracy_diff']:+.4f}, paired bootstrap 95% CI {ci(f['bootstrap_ci95'])}.",
          f"AUC trained − base: {f['auc_vs']['base']['auc_diff']:+.4f}, class-stratified paired bootstrap 95% CI "
          f"{ci(f['auc_vs']['base']['ci95'])}.", "",
          "## Paired changes vs base", "", "```", json.dumps(f["paired_changes"]), "```", "",
          "## Trained confusion", "", confusion_table(f["trained"])]
    others = [k for k in f["auc_vs"] if k != "base"]
    if others or f["vs_saved"]:
        md += ["", "## Trained vs saved runs", "", "| reference | Δ accuracy | 95% CI | Δ AUC | 95% CI |", "|---|---|---|---|---|"]
        for ref in dict.fromkeys([*f["vs_saved"], *others]):
            a, u = f["vs_saved"].get(ref), f["auc_vs"].get(ref)
            md.append(f"| {ref} | " + (f"{a['accuracy_diff']:+.4f} | {ci(a['ci95'])}" if a else "— | —") + " | "
                      + (f"{u['auc_diff']:+.4f} | {ci(u['ci95'])}" if u else "— | —") + " |")
    m = result["memory"]
    md += ["", f"Peak reserved {m['peak_reserved_gib']:.2f} GiB of {GPU_BUDGET_GIB} GiB budget. "
           f"Frozen weights changed: {len(result['frozen_changed'])}. Raw predictions: `{bench['predictions']}`; "
           f"training log: `{bench['training_log']}`."]
    (BENCHMARKS / f"{name}.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=LEARNING_RATE)
    ap.add_argument("--steps", type=int, default=MAX_STEPS)
    ap.add_argument("--train-samples", type=int, default=TRAIN_SAMPLES)
    ap.add_argument("--val-subset", type=int, default=VAL_SUBSET)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--max-attempts", type=int, default=None, help="rollout-group budget (default 10 * steps)")
    ap.add_argument("--full-val", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora-targets", choices=TARGETS, default="last-o-proj")
    ap.add_argument("--batch-size", type=int, default=1, help="prompts per optimizer step (B)")
    ap.add_argument("--batch-mode", choices=BATCH_MODES, default=None, help="default: cycle for B=1")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. _smoke or _lr0")
    args = ap.parse_args()
    B, mode = args.batch_size, args.batch_mode or ("cycle" if args.batch_size == 1 else None)
    if (mode == "cycle") != (B == 1) or (mode == "balanced" and B != 2):
        ap.error("B=1 uses --batch-mode cycle; balanced needs --batch-size 2; B>1 needs an explicit mode")
    max_attempts = args.max_attempts or 10 * args.steps
    name = output_name(args.lora_targets, args.tag, B, mode)

    splits = load_splits()
    train = balanced_subset(splits["train"], args.train_samples, SEED)
    val_all = [{"id": i, **r} for i, r in enumerate(splits["validation"])]
    val_sub = sorted(balanced_subset(splits["validation"], args.val_subset, SEED), key=lambda e: e["id"])
    base_file = json.loads(BASE_FILE.read_text())["predictions"]
    base_file_m = metrics(base_file)
    assert len(base_file) == 1066 and round(base_file_m["accuracy"], 4) == 0.7486 and sum(p["prediction"] is None for p in base_file) == 3
    assert [(p["id"], p["text"], p["label"]) for p in base_file] == [(e["id"], e["text"], e["label"]) for e in val_all]

    torch.manual_seed(SEED)  # LoRA A init
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    assert tok.pad_token_id == EOS_IDS[1]  # baseline.py pads with tok.pad_token_id, lora_model with EOS_IDS[1]
    model = lora_model(DTYPE, "cuda", TARGETS[args.lora_targets])
    torch.cuda.reset_peak_memory_stats()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    modules = lora_modules(model)
    module_names = [n for n, _ in modules]
    assert module_names == EXPECTED_MODULES[args.lora_targets], module_names
    n_trainable = sum(p.numel() for p in params.values())
    assert n_trainable == expected_lora_params(modules) and len(params) == 2 * len(modules)
    assert all(".lora_A." in n or ".lora_B." in n for n in params)
    assert all(p.dtype == torch.float32 for p in params.values())
    assert all(p.dtype == DTYPE for p in frozen.values())
    names_A, names_B = [n for n in params if ".lora_A." in n], [n for n in params if ".lora_B." in n]
    pairs = [(m.lora_A["default"].weight, m.lora_B["default"].weight) for _, m in modules]
    assert all(params[n].abs().max() == 0 for n in names_B)  # adapter contributes exactly zero: initial model = base
    n_total = n_trainable + sum(p.numel() for p in frozen.values())
    print(f"LoRA modules ({len(modules)}): {module_names}")
    print(f"trainable parameters: {n_trainable:,} of {n_total:,} ({100 * n_trainable / n_total:.4f}%)")
    init = {n: p.detach().clone() for n, p in params.items()}
    frozen_digest = {n: digest(p) for n, p in frozen.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=0.0)

    result = {
        "model": MODEL, "method": "GRPO+LoRA", "seed": SEED,
        "config": {"lr": args.lr, "steps": args.steps, "max_attempts": max_attempts, "train_samples": args.train_samples,
                   "train_ids": [e["id"] for e in train], "val_subset_ids": [e["id"] for e in val_sub],
                   "B": B, "G": G, "batch_mode": mode,
                   "batch_policy": f"{mode} (see batches()); a zero-signal batch is consumed, not retried", "rollout_max_new_tokens": MAX_NEW_TOKENS, "eval_max_new_tokens": MAX_NEW,
                   "eval_batch": BATCH, "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0},
                   "clip_eps": CLIP_EPS, "kl_beta": 0.0, "optimizer": "AdamW(weight_decay=0)", "grad_clipping": None,
                   "lora": {"targets": args.lora_targets, "target_regex": TARGETS[args.lora_targets], "modules": module_names,
                            "trainable_params": n_trainable, "r": RANK, "alpha": ALPHA, "dropout": 0.0},
                   "dtypes": {"base": str(DTYPE), "lora": "torch.float32"}, "gpu_budget_gib": GPU_BUDGET_GIB,
                   "checkpoint_contents": "adapter weights + adapter_config only (no optimizer/RNG state)"},
        "training": [], "zero_signal": [], "evaluations": [],
    }

    base_full = None
    if args.full_val:
        base_full = evaluate(model, tok, val_all)
        mism = [p["id"] for p, q in zip(base_full, base_file) if p["response"] != q["response"]]
        result["base_reproduction"] = {"n": len(base_full), "mismatched_ids": mism, "metrics": metrics(base_full)}
        print(f"base full validation (in-process): {summary(metrics(base_full))}\n  mismatches vs {BASE_FILE.name}: {len(mism)}")

    def eval_subset(step, ckpt=None):
        preds = evaluate(model, tok, val_sub)
        result["evaluations"].append({"step": step, "checkpoint": ckpt, "split": "validation", "metrics": metrics(preds), "predictions": preds})
        print(f"  [val subset, step {step}] {summary(metrics(preds))}")

    eval_subset(0)
    torch.manual_seed(SEED)  # rollout sampling
    step = attempts = 0
    ckpt = None
    stream, mixed_counts, t0 = batches(train, B, mode), Counter(), time.time()
    while step < args.steps and attempts < max_attempts:
        batch = next(stream)
        attempts += 1
        prompts = build_prompts(tok, [e["text"] for e in batch])
        rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=MAX_NEW_TOKENS)
        check_grouping(rollouts, tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].cuda(), G)
        responses = decode(tok, rollouts)
        rewards = group_rewards(responses, [e["label"] for e in batch], G)  # [B, G]
        mixed = int((rewards != rewards[:, :1]).any(dim=1).sum())
        mixed_counts[mixed] += 1
        group = {"attempt": attempts, "prompt_ids": [e["id"] for e in batch], "labels": [e["label"] for e in batch],
                 "mixed_groups": mixed, "responses": responses, "rewards": rewards.tolist(), "mean_reward": rewards.mean().item(),
                 "invalid_responses": sum(parse_label(r) is None for r in responses),
                 "rollout_sha256": hashlib.sha256(rollouts["response_ids"].cpu().numpy().tobytes()).hexdigest()[:16]}
        out = grpo_step(model, params, optimizer, rollouts, rewards)
        del rollouts
        if out is None:
            result["zero_signal"].append(group)
            continue
        step += 1
        with torch.no_grad():  # norms over several modules aggregate as sqrt(sum of squares)
            row = {"step": step, **group, "loss": out["loss"], "init_diff": out["init_diff"], "grad_norm": out["grad_norm"],
                   "grad_norm_A": l2(out["grad_norms"][n] for n in names_A), "grad_norm_B": l2(out["grad_norms"][n] for n in names_B),
                   "update_norm": out["update_norm"],
                   "lora_A_norm": l2(params[n].norm().item() for n in names_A),
                   "lora_B_norm": l2(params[n].norm().item() for n in names_B),
                   "effective_adapter_norm": l2(((ALPHA / RANK) * Bm @ A).norm().item() for A, Bm in pairs),
                   "change_from_init": math.sqrt(sum((p - init[n]).pow(2).sum().item() for n, p in params.items())),
                   "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        result["training"].append(row)
        ids = ",".join(f"{e['id']}{e['label'][0]}" for e in batch)
        print(f"step {step:2d} (attempt {attempts:3d}) {ids:12} mixed {mixed}  R {row['mean_reward']:.2f}  "
              f"loss {row['loss']:+.4f}  |g| {row['grad_norm']:.3e} (A {row['grad_norm_A']:.2e}, B {row['grad_norm_B']:.2e})  "
              f"|dtheta| {row['update_norm']:.2e}  |(a/r)BA| {row['effective_adapter_norm']:.3e}  "
              f"alloc {row['cuda_allocated_gib']:.2f}  peak res {row['peak_reserved_gib']:.2f} GiB")
        if step % args.eval_every == 0 or step == args.steps:
            ckpt = f"checkpoints/{name}_step{step}"
            model.save_pretrained(ROOT / ckpt)
            eval_subset(step, ckpt)

    result["train_seconds"] = time.time() - t0
    result["groups_generated"], result["successful_steps"] = attempts, step  # groups_generated counts batches
    result["prompt_groups"], result["responses_generated"] = attempts * B, attempts * B * G
    result["batches_by_mixed_groups"] = {str(k): mixed_counts[k] for k in range(B + 1)}
    result["zero_signal_groups"] = len(result["zero_signal"])  # batches skipped: no group had mixed rewards
    if step < args.steps:
        print(f"Attempt budget exhausted: {step} of {args.steps} steps after {attempts} groups")

    result["frozen_changed"] = [n for n, p in frozen.items() if digest(p) != frozen_digest[n]]
    result["params_equal_init"] = all(torch.equal(p, init[n]) for n, p in params.items())
    if args.lr == 0:
        subset = [e["predictions"] for e in result["evaluations"]]
        result["control"] = {"params_equal_init": result["params_equal_init"],
                             "subset_predictions_unchanged": all(s == subset[0] for s in subset)}

    if ckpt is not None:  # evaluate the saved artifact, not just the in-memory adapter
        live = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        loaded = load_peft_weights(str(ROOT / ckpt))
        assert live.keys() == loaded.keys() and all(torch.equal(live[k], loaded[k].to(live[k].device)) for k in live), ckpt
        set_peft_model_state_dict(model, loaded)
        assert all(torch.equal(p, live[k]) for k, p in get_peft_model_state_dict(model).items())
        result["final_checkpoint"] = {"path": ckpt, "step": step, "reloaded_bitwise_equal": True}

    if args.full_val and ckpt is not None:
        trained_full = evaluate(model, tok, val_all)
        tm, bm = metrics(trained_full), metrics(base_full)
        diff, lo, hi = paired_bootstrap([p["correct"] for p in base_full], [p["correct"] for p in trained_full])
        out_path = RESULTS / name / "validation.json"
        result["final"] = {"base": bm, "trained": tm, "accuracy_diff": diff, "bootstrap_ci95": [lo, hi],
                           "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "paired_by": "id"},
                           "paired_changes": paired_changes(base_full, trained_full),
                           "vs_saved": {ref: paired_compare(json.loads(p.read_text())["predictions"], trained_full)
                                        for ref, (p, _, _) in REFERENCES.items() if p.exists() and p != out_path}}
        scores = {"trained": label_scores(model, tok, val_all)}
        with model.disable_adapter():
            scores["base"] = label_scores(model, tok, val_all)
        for ref, (_, ref_ckpt, targets) in REFERENCES.items():  # same targets -> load into this model
            if ref_ckpt is not None and targets == args.lora_targets and ref_ckpt.exists() and ref_ckpt != ROOT / ckpt:
                set_peft_model_state_dict(model, load_peft_weights(str(ref_ckpt)))
                scores[ref] = label_scores(model, tok, val_all)
        set_peft_model_state_dict(model, loaded)
        W = {k: win_matrix(v, [e["label"] for e in val_all]) for k, v in scores.items()}
        result["final"]["auc"] = {k: w.mean().item() for k, w in W.items()}
        result["final"]["auc_vs"] = {k: paired_auc_bootstrap(W["trained"], W[k]) for k in W if k != "trained"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": MODEL, "checkpoint": ckpt, "step": step, "split": "validation",
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW},
            "label_scores": {"ids": [e["id"] for e in val_all], **scores}, "predictions": trained_full,
        }, indent=1))

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    mem = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "device_gib": total / 2**30}
    mem["within_budget"] = mem["peak_reserved_gib"] < GPU_BUDGET_GIB
    result["memory"] = mem
    (RESULTS / name).mkdir(parents=True, exist_ok=True)
    (RESULTS / name / "training.json").write_text(json.dumps({k: v for k, v in result.items() if k != "final"}, indent=1))
    if "final" in result:
        write_benchmark(name, result)

    rows = result["training"]
    print(f"\nbatches by number of mixed groups: {result['batches_by_mixed_groups']}, prompt groups {attempts * B}, "
          f"responses {attempts * B * G}, train time {result['train_seconds']:.0f} s")
    print(f"steps {step}/{args.steps}, batches {attempts}, zero-signal batches {result['zero_signal_groups']}, "
          f"frozen changed {len(result['frozen_changed'])}/{len(frozen)}, params equal init: {result['params_equal_init']}")
    if rows:
        print(f"grad norm range [{min(r['grad_norm'] for r in rows):.3e}, {max(r['grad_norm'] for r in rows):.3e}]  "
              f"change from init {rows[-1]['change_from_init']:.3e}  |(a/r)BA| {rows[-1]['effective_adapter_norm']:.3e}  "
              f"allocated after step: {min(r['cuda_allocated_gib'] for r in rows):.3f}-{max(r['cuda_allocated_gib'] for r in rows):.3f} GiB")
    if "control" in result:
        print(f"LR=0 control: {result['control']}")
    if "final" in result:
        f = result["final"]
        print(f"\nfull validation  base:    {summary(f['base'])}\nfull validation  trained: {summary(f['trained'])}")
        print(f"accuracy diff {f['accuracy_diff']:+.4f}  paired bootstrap 95% CI [{f['bootstrap_ci95'][0]:+.4f}, {f['bootstrap_ci95'][1]:+.4f}]")
        print(f"paired changes: {f['paired_changes']}")
        for ref, c in f["vs_saved"].items():
            print(f"\nvs saved {ref} predictions: {summary(c['ref_metrics'])}")
            print(f"  accuracy diff {c['accuracy_diff']:+.4f}  paired bootstrap 95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
            print(f"  paired changes: {c['paired_changes']}")
        for k, a in f["auc"].items():
            vs = f["auc_vs"].get(k)
            print(f"ROC AUC {k:24} {a:.4f}" + (f"   trained - this: {vs['auc_diff']:+.4f}  CI [{vs['ci95'][0]:+.4f}, {vs['ci95'][1]:+.4f}]" if vs else ""))
    print(f"peak allocated {mem['peak_allocated_gib']:.2f} GiB, peak reserved {mem['peak_reserved_gib']:.2f} GiB, "
          f"device {mem['device_gib']:.2f} GiB, budget {GPU_BUDGET_GIB} GiB: {'PASS' if mem['within_budget'] else 'FAIL'}"
          "  (allocator stats exclude the CUDA context)")
    if result["frozen_changed"] or ("control" in result and not all(result["control"].values())):
        raise SystemExit("FAILED: frozen weights changed or LR=0 control changed")


if __name__ == "__main__":
    main()
