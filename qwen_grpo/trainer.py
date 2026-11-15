"""Multi-step GRPO + LoRA: fp32 adapters on selected projections of a frozen Qwen3 base model.

Run through scripts/run.py train --config <yaml> [--resume checkpoints/<run>_step<N>]; every setting comes from the config
(qwen_grpo/config.py). Rollouts: sampled at training.temperature, training.max_new_tokens. Evaluation: greedy, in chunks,
with the YAML evaluation settings (defaults: those of the saved baseline, so runs stay comparable with it).
Checkpoints hold the adapter and the training state (optimizer, progress, log so far, RNG), so a run resumes from any
checkpoint it saved and continues as the uninterrupted run would have (tests/test_trainer.py checks this bitwise on CPU).
Outputs: results/<run>/training.json (config, environment, per-step log, subset evaluations, memory), results/<run>/validation.json
(raw full-validation predictions and label scores of the final checkpoint), results/<run>/plots/ (training diagnostics:
reward, validation-subset accuracy, mixed-reward fraction, gradient norm, loss), results/benchmarks/<run>.{json,md} (metrics).
"""
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer

from qwen_grpo.checkpointing import (load_adapter, load_training_state, lock_path, rel, run_of, save_checkpoint, set_rng_state,
                                     training_lock)
from qwen_grpo.config import DTYPES, Config, as_dict, normalized
from qwen_grpo.evaluation import (BENCHMARKS, RESULTS, confusion_table, evaluate, label_scores, metrics, metrics_table,
                                  paired_auc_bootstrap, paired_bootstrap, paired_changes, paired_compare, predictions_path,
                                  settings, summary, win_matrix)
from qwen_grpo.model import EOS_IDS, digest, expected_lora_params, lora_model, lora_modules
from qwen_grpo.policy import generate_rollouts
from qwen_grpo.reporting import training_plots
from qwen_grpo.sentiment import SPLITS_PATH, build_prompts, load_splits, parse_label
from qwen_grpo.training import balanced_subset, batches, check_grouping, decode, group_rewards, grpo_step

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
BASE_FILE = predictions_path("validation")
SUFFIX = {"last-o-proj": "", "last2-attention": "_last2"}
REFERENCE_STEPS = 30  # final step of the saved historical runs in REFERENCES
SEED = 42  # bootstrap seed of the historical runs (research/trl_reference uses it)
PACKAGES = ("torch", "transformers", "peft", "datasets", "pyyaml")


def output_name(targets, tag="", B=1, mode="cycle"):
    """Naming scheme of the historical runs (before YAML configs), used only to locate REFERENCES."""
    return f"grpo_lora{SUFFIX[targets]}" + (f"_b{B}_{mode}" if B > 1 else "") + tag


def _ref(targets, B=1, mode="cycle"):
    name = output_name(targets, B=B, mode=mode)
    return RESULTS / name / "validation.json", CHECKPOINTS / f"{name}_step{REFERENCE_STEPS}", targets


# Saved final (step-30) predictions (and adapters, for label scoring) of earlier Qwen3-0.6B runs:
# (predictions, checkpoint, targets). A run compares with them only if model and evaluation decoding match.
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


def environment(splits, device):
    """Provenance of a training session: code version, library versions, dataset identity and hardware."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    env = {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
           "dataset": {"name": splits["dataset"], "split_seed": splits["seed"], "file": rel(SPLITS_PATH),
                       "sha256": hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest(),
                       "sizes": {s: len(splits[s]) for s in ("train", "validation", "test")}},
           "device": str(device)}
    if torch.device(device).type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_gib": p.total_memory / 2**30,
                      "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    return env


def write_benchmark(name, result, budget):
    """results/benchmarks/<run>.json (final full-validation metrics vs base) and a Markdown rendering of it."""
    f, cfg = result["final"], result["config"]
    bench = {"run": name, "model": result["model"], "seed": result["seed"], "split": "validation",
             "checkpoint": result["final_checkpoint"]["path"], "step": result["final_checkpoint"]["step"],
             "predictions": rel(RESULTS / name / "validation.json"),
             "training_log": rel(RESULTS / name / "training.json"),
             "baseline_predictions": rel(BASE_FILE),
             "config": {k: v for k, v in cfg.items() if k not in ("train_ids", "val_subset_ids")},
             **f, "memory": result["memory"]}
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    (BENCHMARKS / f"{name}.json").write_text(json.dumps(bench, indent=1))

    ci = lambda c: f"[{c[0]:+.4f}, {c[1]:+.4f}]"
    lora = cfg["lora"]
    md = [f"# {name}: GRPO + LoRA vs baseline", "",
          f"{result['model']}, LoRA r={lora['r']} α={lora['alpha']} on {lora.get('targets', lora.get('target'))}, lr {cfg['lr']}, "
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
    peak = "n/a (CPU run)" if m["peak_reserved_gib"] is None else f"{m['peak_reserved_gib']:.2f} GiB"
    md += ["", f"Peak reserved {peak} of {budget} GiB budget. "
           f"Frozen weights changed: {len(result['frozen_changed'])}. Raw predictions: `{bench['predictions']}`; "
           f"training log: `{bench['training_log']}`."]
    (BENCHMARKS / f"{name}.md").write_text("\n".join(md) + "\n")


def existing_outputs(name):
    return sorted([*RESULTS.glob(name), *(BENCHMARKS.glob(f"{name}.*")), *CHECKPOINTS.glob(f"{name}_step*")])


def check_resume(cfg, checkpoint, state):
    """Resume only the run this checkpoint belongs to, with the same settings, without overwriting later checkpoints."""
    name, step = run_of(checkpoint)
    if name != cfg.run.name or step != state["step"]:
        raise SystemExit(f"{checkpoint} is not a checkpoint of run {cfg.run.name!r}")
    strip = lambda c: {k: ({**v, "overwrite": None} if k == "run" else v) for k, v in c.items() if k != "verification"}
    if strip(normalized(state["config"])) != strip(as_dict(cfg)):
        raise SystemExit(f"{checkpoint} was trained with a different config; resume needs the same YAML settings")
    if later := sorted(rel(p) for p in CHECKPOINTS.glob(f"{name}_step*") if (run_of(p)[1] or 0) > step):
        raise SystemExit(f"resuming from step {step} would overwrite later checkpoints {later}")


def train(cfg: Config, resume=None, device="cuda"):
    """Train under results/.<run>.lock so that `scripts/run.py remove` cannot delete a run while it is training."""
    with training_lock(lock_path(RESULTS, cfg.run.name)):
        return _train(cfg, resume, device)


def _train(cfg: Config, resume=None, device="cuda"):
    t, lo, d, ev = cfg.training, cfg.lora, cfg.data, cfg.evaluation
    B, mode, G, name = t.batch_size, t.batch_mode, t.group_size, cfg.run.name
    gen_kw, score_kw = settings(ev)
    on_cuda = torch.device(device).type == "cuda"
    gib = lambda f: f() / 2**30 if on_cuda else None
    if resume is None:
        if (clash := existing_outputs(name)) and not cfg.run.overwrite:
            raise SystemExit(f"run.name {name!r} already has outputs, refusing to overwrite: {[rel(c) for c in clash]}\n"
                             "  set run.overwrite: true to delete them first, or choose another run.name")
        for c in clash:
            shutil.rmtree(c) if c.is_dir() else c.unlink()
        if clash:
            print(f"run.overwrite: deleted earlier outputs of {name!r}: {[rel(c) for c in clash]}")
    else:
        state = load_training_state(resume)
        check_resume(cfg, resume, state)
    dtype = DTYPES[cfg.model.dtype]
    decoding = {"do_sample": False, "max_new_tokens": ev.max_new_tokens}

    splits = load_splits()
    train_set = balanced_subset(splits["train"], d.train_samples, d.subset_seed)
    val_all = [{"id": i, **r} for i, r in enumerate(splits["validation"])]
    val_sub = sorted(balanced_subset(splits["validation"], d.val_subset, d.subset_seed), key=lambda e: e["id"])

    def saved_predictions(path):
        """Saved predictions of the same model under the same decoding, aligned with validation, else None."""
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if saved.get("model") != cfg.model.name or saved.get("decoding") != decoding:
            return None
        preds = saved["predictions"]
        assert [(p["id"], p["text"], p["label"]) for p in preds] == [(e["id"], e["text"], e["label"]) for e in val_all], path
        return preds

    torch.manual_seed(t.seed)  # LoRA A init
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    assert tok.pad_token_id == EOS_IDS[1]  # baseline.py pads with tok.pad_token_id, lora_model with EOS_IDS[1]
    model = lora_model(dtype, device, lo.spec(), lo.rank, lo.alpha, lo.dropout, cfg.model.name)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    modules = lora_modules(model)
    module_names = [n for n, _ in modules]
    assert sorted(module_names) == sorted(model.peft_config["default"].target_modules), module_names
    n_trainable = sum(p.numel() for p in params.values())
    assert n_trainable == expected_lora_params(modules, lo.rank) and len(params) == 2 * len(modules)
    assert all(m.r["default"] == lo.rank and m.scaling["default"] == lo.alpha / lo.rank for _, m in modules)
    assert all(".lora_A." in n or ".lora_B." in n for n in params)
    assert all(p.dtype == torch.float32 for p in params.values())
    assert all(p.dtype == dtype for p in frozen.values())
    names_A, names_B = [n for n in params if ".lora_A." in n], [n for n in params if ".lora_B." in n]
    pairs = [(m.lora_A["default"].weight, m.lora_B["default"].weight) for _, m in modules]
    assert all(params[n].abs().max() == 0 for n in names_B)  # adapter contributes exactly zero: initial model = base
    n_total = n_trainable + sum(p.numel() for p in frozen.values())
    print(f"LoRA modules ({len(modules)}): {module_names}")
    print(f"trainable parameters: {n_trainable:,} of {n_total:,} ({100 * n_trainable / n_total:.4f}%)")
    init = {n: p.detach().clone() for n, p in params.items()}  # the seeded init, equal to checkpoints/<run>_step0
    frozen_digest = {n: digest(p) for n, p in frozen.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=t.learning_rate, weight_decay=0.0)

    def eval_subset(step, ckpt=None):
        preds = evaluate(model, tok, val_sub, **gen_kw)
        result["evaluations"].append({"step": step, "checkpoint": ckpt, "split": "validation", "metrics": metrics(preds), "predictions": preds})
        print(f"  [val subset, step {step}] {summary(metrics(preds))}")

    if resume is None:
        model.save_pretrained(CHECKPOINTS / f"{name}_step0")  # initial adapter: reference for parameter deltas
        result = {
            "model": cfg.model.name, "method": "GRPO+LoRA", "seed": t.seed, "yaml_config": as_dict(cfg),
            "environment": environment(splits, device),
            "config": {"lr": t.learning_rate, "steps": t.max_steps, "max_attempts": t.max_attempts, "train_samples": d.train_samples,
                       "train_ids": [e["id"] for e in train_set], "val_subset_ids": [e["id"] for e in val_sub],
                       "B": B, "G": G, "batch_mode": mode,
                       "batch_policy": f"{mode} (see batches()); a zero-signal batch is consumed, not retried",
                       "rollout_max_new_tokens": t.max_new_tokens, "eval_max_new_tokens": ev.max_new_tokens,
                       "eval_batch": ev.batch_size, "sampling": {"do_sample": True, "temperature": t.temperature, "top_p": 1.0, "top_k": 0},
                       "clip_eps": t.clip_eps, "kl_beta": 0.0, "optimizer": "AdamW(weight_decay=0)", "grad_clipping": None,
                       "lora": {"targets": as_dict(cfg)["lora"]["targets"], "target_modules": module_names,
                                "trainable_params": n_trainable, "r": lo.rank, "alpha": lo.alpha, "dropout": lo.dropout},
                       "dtypes": {"base": str(dtype), "lora": "torch.float32"}, "gpu_budget_gib": cfg.run.gpu_budget_gib,
                       "checkpoint_contents": "adapter weights + adapter_config + training_state.pt (optimizer, progress, log, RNG)",
                       "initial_checkpoint": rel(CHECKPOINTS / f"{name}_step0")},
            "training": [], "zero_signal": [], "evaluations": [],
        }
        base_full = None
        if t.full_val:
            base_full = evaluate(model, tok, val_all, **gen_kw)
            saved = saved_predictions(BASE_FILE)
            mism = None if saved is None else [p["id"] for p, q in zip(base_full, saved) if p["response"] != q["response"]]
            result["base_reproduction"] = {"n": len(base_full), "mismatched_ids": mism, "metrics": metrics(base_full)}
            print(f"base full validation (in-process): {summary(metrics(base_full))}\n  mismatches vs {BASE_FILE.name}: "
                  + ("not compared (different model or decoding)" if mism is None else str(len(mism))))
        eval_subset(0)
        step = attempts = 0
        ckpt, mixed_counts, elapsed = None, Counter(), 0.0
        torch.manual_seed(t.seed)  # rollout sampling
    else:
        load_adapter(model, resume)
        optimizer.load_state_dict(state["optimizer"])
        result, base_full = state["result"], state["base_full"]
        step, attempts, ckpt, elapsed = state["step"], state["attempts"], state["checkpoint"], state["train_seconds"]
        mixed_counts = Counter(state["mixed_counts"])
        result.setdefault("resumed", []).append({"from": rel(resume), "step": step, "attempts": attempts,
                                                 "environment": environment(splits, device)})
        print(f"resumed from {rel(resume)}: step {step}, attempt {attempts}")

    def save(step):
        """Adapter + resumable state; the subset evaluation runs first so the saved log and RNG state include it."""
        path = CHECKPOINTS / f"{name}_step{step}"
        eval_subset(step, rel(path))
        save_checkpoint(model, optimizer, path, {
            "config": as_dict(cfg), "step": step, "attempts": attempts, "checkpoint": rel(path),
            "mixed_counts": dict(mixed_counts), "train_seconds": time.time() - t0, "result": result, "base_full": base_full})
        return rel(path)

    stream = batches(train_set, B, mode, seed=t.seed)
    for _ in range(attempts):  # replay: every attempt, zero-signal or not, consumed one batch
        next(stream)
    if resume is not None:
        set_rng_state(state["rng"])  # last: nothing between here and the next rollout may draw random numbers
    t0 = time.time() - elapsed
    while step < t.max_steps and attempts < t.max_attempts:
        batch = next(stream)
        attempts += 1
        t_step = time.time()
        prompts = build_prompts(tok, [e["text"] for e in batch])
        rollouts = generate_rollouts(model, tok, prompts, G=G, max_new_tokens=t.max_new_tokens, temperature=t.temperature)
        check_grouping(rollouts, tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(model.device), G)
        responses = decode(tok, rollouts)
        rewards = group_rewards(responses, [e["label"] for e in batch], G)  # [B, G]
        mixed = int((rewards != rewards[:, :1]).any(dim=1).sum())
        mixed_counts[mixed] += 1
        group = {"attempt": attempts, "prompt_ids": [e["id"] for e in batch], "labels": [e["label"] for e in batch],
                 "mixed_groups": mixed, "responses": responses, "rewards": rewards.tolist(), "mean_reward": rewards.mean().item(),
                 "invalid_responses": sum(parse_label(r) is None for r in responses),
                 "rollout_sha256": hashlib.sha256(rollouts["response_ids"].cpu().numpy().tobytes()).hexdigest()[:16],
                 "mean_completion_tokens": rollouts["completion_mask"].sum(1).float().mean().item()}  # through the first EOS
        out = grpo_step(model, params, optimizer, rollouts, rewards, clip_eps=t.clip_eps)
        del rollouts
        if out is None:
            result["zero_signal"].append(group)
            continue
        step += 1
        with torch.no_grad():  # norms over several modules aggregate as sqrt(sum of squares)
            row = {"step": step, **group, "loss": out["loss"], "init_diff": out["init_diff"], "grad_norm": out["grad_norm"],
                   "grad_norm_A": l2(out["grad_norms"][n] for n in names_A), "grad_norm_B": l2(out["grad_norms"][n] for n in names_B),
                   "update_norm": out["update_norm"], "learning_rate": optimizer.param_groups[0]["lr"],
                   "step_seconds": time.time() - t_step,  # rollout generation + rewards + GRPO update
                   "lora_A_norm": l2(params[n].norm().item() for n in names_A),
                   "lora_B_norm": l2(params[n].norm().item() for n in names_B),
                   "effective_adapter_norm": l2(((lo.alpha / lo.rank) * Bm @ A).norm().item() for A, Bm in pairs),
                   "change_from_init": math.sqrt(sum((p - init[n]).pow(2).sum().item() for n, p in params.items())),
                   "cuda_allocated_gib": gib(torch.cuda.memory_allocated),
                   "peak_allocated_gib": gib(torch.cuda.max_memory_allocated),
                   "peak_reserved_gib": gib(torch.cuda.max_memory_reserved)}
        result["training"].append(row)
        ids = ",".join(f"{e['id']}{e['label'][0]}" for e in batch)
        mem = f"alloc {row['cuda_allocated_gib']:.2f}  peak res {row['peak_reserved_gib']:.2f} GiB" if on_cuda else ""
        print(f"step {step:2d} (attempt {attempts:3d}) {ids:12} mixed {mixed}  R {row['mean_reward']:.2f}  "
              f"loss {row['loss']:+.4f}  |g| {row['grad_norm']:.3e} (A {row['grad_norm_A']:.2e}, B {row['grad_norm_B']:.2e})  "
              f"|dtheta| {row['update_norm']:.2e}  |(a/r)BA| {row['effective_adapter_norm']:.3e}  {mem}")
        if step % t.eval_every == 0 or step == t.max_steps:
            ckpt = save(step)
    if step and ckpt != rel(CHECKPOINTS / f"{name}_step{step}"):  # attempt budget ran out between checkpoints
        ckpt = save(step)

    result["train_seconds"] = time.time() - t0
    result["groups_generated"], result["successful_steps"] = attempts, step  # groups_generated counts batches
    result["prompt_groups"], result["responses_generated"] = attempts * B, attempts * B * G
    result["batches_by_mixed_groups"] = {str(k): mixed_counts[k] for k in range(B + 1)}
    result["zero_signal_groups"] = len(result["zero_signal"])  # batches skipped: no group had mixed rewards
    if step < t.max_steps:
        print(f"Attempt budget exhausted: {step} of {t.max_steps} steps after {attempts} groups")

    result["frozen_changed"] = [n for n, p in frozen.items() if digest(p) != frozen_digest[n]]
    result["params_equal_init"] = all(torch.equal(p, init[n]) for n, p in params.items())
    if t.learning_rate == 0:
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

    def save_log(status):
        """results/<run>/training.json; written when training ends and again after the final evaluation."""
        mem = {"peak_allocated_gib": None, "peak_reserved_gib": None, "device_gib": None, "within_budget": None}
        if on_cuda:
            torch.cuda.synchronize()
            mem.update(peak_allocated_gib=gib(torch.cuda.max_memory_allocated), peak_reserved_gib=gib(torch.cuda.max_memory_reserved),
                       device_gib=torch.cuda.mem_get_info()[1] / 2**30)
            mem["within_budget"] = mem["peak_reserved_gib"] < cfg.run.gpu_budget_gib
        result["status"], result["memory"] = status, mem
        (RESULTS / name).mkdir(parents=True, exist_ok=True)
        (RESULTS / name / "training.json").write_text(json.dumps({k: v for k, v in result.items() if k != "final"}, indent=1))
        return mem

    save_log("training complete; final evaluation pending")  # survives a failure in the evaluation below

    if t.full_val and ckpt is not None:
        boot = {"n": ev.bootstrap_samples, "seed": ev.bootstrap_seed}
        trained_full = evaluate(model, tok, val_all, **gen_kw)
        tm, bm = metrics(trained_full), metrics(base_full)
        diff, ci_lo, ci_hi = paired_bootstrap([p["correct"] for p in base_full], [p["correct"] for p in trained_full], **boot)
        out_path = RESULTS / name / "validation.json"
        refs = {ref: preds for ref, (p, _, _) in REFERENCES.items() if p != out_path and (preds := saved_predictions(p)) is not None}
        result["final"] = {"base": bm, "trained": tm, "accuracy_diff": diff, "bootstrap_ci95": [ci_lo, ci_hi],
                           "bootstrap": {"resamples": ev.bootstrap_samples, "seed": ev.bootstrap_seed, "paired_by": "id"},
                           "paired_changes": paired_changes(base_full, trained_full),
                           "vs_saved": {ref: paired_compare(preds, trained_full, **boot) for ref, preds in refs.items()}}
        scores = {"trained": label_scores(model, tok, val_all, **score_kw)}
        with model.disable_adapter():
            scores["base"] = label_scores(model, tok, val_all, **score_kw)
        for ref, (_, ref_ckpt, targets) in REFERENCES.items():  # same model and targets -> load into this model
            if (ref_ckpt is not None and targets == lo.targets and ref_ckpt.exists() and ref_ckpt != ROOT / ckpt
                    and json.loads((ref_ckpt / "adapter_config.json").read_text())["base_model_name_or_path"] == cfg.model.name):
                set_peft_model_state_dict(model, load_peft_weights(str(ref_ckpt)))
                scores[ref] = label_scores(model, tok, val_all, **score_kw)
        set_peft_model_state_dict(model, loaded)
        W = {k: win_matrix(v, [e["label"] for e in val_all]) for k, v in scores.items()}
        result["final"]["auc"] = {k: w.mean().item() for k, w in W.items()}
        result["final"]["auc_vs"] = {k: paired_auc_bootstrap(W["trained"], W[k], **boot) for k in W if k != "trained"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": cfg.model.name, "checkpoint": ckpt, "step": step, "split": "validation", "decoding": decoding,
            "label_scores": {"ids": [e["id"] for e in val_all], **scores}, "predictions": trained_full,
        }, indent=1))

    mem = save_log("complete")  # memory peaks now include the final evaluation
    (RESULTS / name / "plots").mkdir(exist_ok=True)
    training_plots(result, RESULTS / name / "plots")
    if "final" in result:
        write_benchmark(name, result, cfg.run.gpu_budget_gib)

    rows = result["training"]
    print(f"\nbatches by number of mixed groups: {result['batches_by_mixed_groups']}, prompt groups {attempts * B}, "
          f"responses {attempts * B * G}, train time {result['train_seconds']:.0f} s")
    print(f"steps {step}/{t.max_steps}, batches {attempts}, zero-signal batches {result['zero_signal_groups']}, "
          f"frozen changed {len(result['frozen_changed'])}/{len(frozen)}, params equal init: {result['params_equal_init']}")
    if rows:
        print(f"grad norm range [{min(r['grad_norm'] for r in rows):.3e}, {max(r['grad_norm'] for r in rows):.3e}]  "
              f"change from init {rows[-1]['change_from_init']:.3e}  |(a/r)BA| {rows[-1]['effective_adapter_norm']:.3e}")
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
    if on_cuda:
        print(f"peak allocated {mem['peak_allocated_gib']:.2f} GiB, peak reserved {mem['peak_reserved_gib']:.2f} GiB, "
              f"device {mem['device_gib']:.2f} GiB, budget {cfg.run.gpu_budget_gib} GiB: {'PASS' if mem['within_budget'] else 'FAIL'}"
              "  (allocator stats exclude the CUDA context)")
    if result["frozen_changed"] or ("control" in result and not all(result["control"].values())):
        raise SystemExit("FAILED: frozen weights changed or LR=0 control changed")
    return result
