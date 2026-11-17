"""Ticket 13: held-out evaluation of the final run on the test split, through the L2L engine only.

    python scripts/final_eval.py --run qwen3_4b_final --control qwen3_4b_final_lr0 --device cuda

Execution: the run's own execution section (resident, l2l_sync or l2l_async; DFlash is never used for evaluation),
greedy decoding with its evaluation settings (max_new_tokens, batch sizes). Models (the same test prompts):
  base      the untuned model (the trained run's PEFT model with the adapter disabled)
  trained   the adapter-attached model at the run's last checkpoint
  merged    the trained adapter merged tensor by tensor (export.merge) and loaded as a plain HF model
  control   the LR=0 negative-control run's last checkpoint (adapter attached)
Reported per model: accuracy, macro F1, invalid-response rate, prediction distribution (evaluation.metrics; reward =
1 for a correct canonical label, so mean reward = accuracy). Paired bootstrap 95% CIs of the accuracy difference
(evaluation.paired_bootstrap, paired by test example): trained - base, merged - trained, control - base. Policy change:
the fraction of test predictions that changed vs base; ROC AUC from label scores (log P(POSITIVE) - log P(NEGATIVE),
through the same backend) with paired bootstrap CIs; LoRA change: norms of the trained and control adapters minus
their step-0 adapters. Data separation: the run's training prompt ids (from its log) must not occur in the test split
(different split, checked by text). Writes research/l2l/experiments/t13_final_eval_<run>.json (or --out) and refuses to
overwrite an existing result. `scripts/run.py verify --run <run>` calls this for runs whose config selects an L2L
backend (with --step and --out under runs/<run>/verification/).
"""
import argparse
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qwen_grpo import runs  # noqa: E402
from qwen_grpo.checkpointing import load_adapter  # noqa: E402
from qwen_grpo.config import DTYPES, as_dict, load, sha256  # noqa: E402
from qwen_grpo.evaluation import (BOOTSTRAP_SEED, evaluate, label_scores, metrics, paired_auc_bootstrap,  # noqa: E402
                                  paired_bootstrap, settings, win_matrix)
from qwen_grpo.execution.backend import ResidentBackend, load_device, make_evaluation_backend  # noqa: E402
from qwen_grpo.export.merge import load_merged_l2l, stream_merge  # noqa: E402
from qwen_grpo.model import lora_model  # noqa: E402
from qwen_grpo.sentiment import load_splits  # noqa: E402


def adapter_change(ckpt, step0):
    from peft import load_peft_weights
    a, b = load_peft_weights(str(ckpt), device="cpu"), load_peft_weights(str(step0), device="cpu")
    return {"l2_change": sum((a[k].float() - b[k].float()).pow(2).sum().item() for k in a) ** 0.5,
            "max_abs_change": max((a[k].float() - b[k].float()).abs().max().item() for k in a)}


def with_storage(cfg, storage):
    """cfg with execution.weight_storage replaced (L2L only; None keeps the config's)."""
    if storage is None or cfg.execution.backend == "resident":
        return cfg
    return dataclasses.replace(cfg, execution=dataclasses.replace(cfg.execution, weight_storage=storage))


def policy(cfg, ckpt, device):
    """The PEFT model with `ckpt` loaded, and its evaluation backend (cfg.execution)."""
    lo = cfg.lora
    m = lora_model(DTYPES[cfg.model.dtype], load_device(cfg, device), lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
    be = make_evaluation_backend(cfg, m, device)
    load_adapter(m, ckpt)
    return m, be


def merged_policy(cfg, path, device):
    """The merged checkpoint as a plain HF model, executed like the run (resident on the device, or L2L)."""
    x = cfg.execution
    if x.backend == "resident":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(path, dtype=DTYPES[cfg.model.dtype]).to(device).eval()
        return m, ResidentBackend(m)
    return load_merged_l2l(path, device, DTYPES[cfg.model.dtype], x.weight_storage, x.weight_buffers,
                           kv_device=torch.device(device) if x.kv_cache == "gpu" else torch.device("cpu"),
                           prefetch_depth=x.prefetch_depth if x.backend == "l2l_async" else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--control")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, help="first n test examples (default: all)")
    ap.add_argument("--storage", choices=["pinned", "pageable"], help="L2L weight storage (default: the run's)")
    ap.add_argument("--step", type=int, help="checkpoint step of --run (default: the latest)")
    ap.add_argument("--out", type=Path, help="result file (default research/l2l/experiments/t13_final_eval_<run>.json)")
    a = ap.parse_args(argv)
    out = a.out or ROOT / "research" / "l2l" / "experiments" / f"t13_final_eval_{a.run}.json"
    if out.exists():
        raise SystemExit(f"{out} exists: evaluation results are never overwritten")
    from qwen_grpo.trainer import check_device, environment, git_state, models
    cfg = with_storage(load(runs.run_dir(a.run) / "config.yaml", historical=True), a.storage)
    check_device(cfg, a.device)
    gen_kw, score_kw = settings(cfg.evaluation)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    splits = load_splits()
    test = [{"id": i, **r} for i, r in enumerate(splits["test"])][: a.n]
    labels = [e["label"] for e in test]
    log = json.loads((runs.training_dir(a.run) / "training.json").read_text())
    train_texts = {r["text"] for r in splits["train"]}
    rep = {"run": a.run, "control": a.control, "model": cfg.model.name, "n_test": len(test), "split": "test",
           "config_sha256": sha256(cfg), "execution": as_dict(cfg)["execution"],
           "decoding": {"do_sample": False, "max_new_tokens": cfg.evaluation.max_new_tokens,
                        "batch": cfg.evaluation.batch_size, "score_batch": cfg.evaluation.score_batch_size},
           "provenance": {**git_state(), "environment": environment(splits, a.device, models(cfg))},
           "data_separation": {"test_texts_in_train": sum(e["text"] in train_texts for e in test),
                               "train_ids_used": len(log["config"]["train_ids"])},
           "training": {"steps": log.get("successful_steps"), "attempts": log.get("groups_generated"),
                        "zero_signal": log.get("zero_signal_groups"), "frozen_changed": log.get("frozen_changed"),
                        "reward_curve": [r.get("mean_reward") for r in log.get("training", [])]}, "models": {}}
    preds, scores = {}, {}

    def run_model(name, model, be):
        preds[name] = evaluate(model, tok, test, generator=be.greedy, **gen_kw)
        scores[name] = label_scores(model, tok, test, logprobs=be.logprobs, **score_kw)
        rep.setdefault("backends", {})[name] = be.name

    ckpt = runs.checkpoint_path(a.run, a.step) if a.step is not None else runs.latest_checkpoint(a.run)
    rep["checkpoint"] = runs.rel(ckpt)
    model, be = policy(cfg, ckpt, a.device)
    run_model("trained", model, be)
    with model.disable_adapter():
        run_model("base", model, be)
    if torch.cuda.is_available():
        rep["peak_gpu_allocated_bytes_eval"] = torch.cuda.max_memory_allocated()
    be.close()
    del model, be
    with tempfile.TemporaryDirectory(prefix="t13_merged_") as tmp:
        merge_rep = stream_merge(cfg.model.name, ckpt, Path(tmp) / "merged")
        merged, mbe = merged_policy(cfg, Path(tmp) / "merged", a.device)
        run_model("merged", merged, mbe)
        mbe.close()
        del merged, mbe
    rep["merge"] = {"all_within_half_ulp": all(v["within_half_ulp"] for v in merge_rep["merged"].values()),
                    "adapted_tensors": len(merge_rep["merged"])}
    rep["lora_change"] = {"trained": adapter_change(ckpt, runs.checkpoint_path(a.run, 0))}
    if a.control:
        cckpt = runs.latest_checkpoint(a.control)
        cm, cbe = policy(cfg, cckpt, a.device)
        run_model("control", cm, cbe)
        cbe.close()
        rep["lora_change"]["control"] = adapter_change(cckpt, runs.checkpoint_path(a.control, 0))
        clog = json.loads((runs.training_dir(a.control) / "training.json").read_text())
        rep["control_log"] = {"params_equal_init": clog.get("params_equal_init"), "control": clog.get("control")}
    correct = {k: [p["correct"] for p in v] for k, v in preds.items()}
    W = {k: win_matrix(v, labels) for k, v in scores.items()}
    for k, v in preds.items():
        rep["models"][k] = {**metrics(v), "mean_reward": sum(correct[k]) / len(v), "auc": W[k].mean().item(),
                            "changed_vs_base": sum(p["response"] != q["response"] for p, q in zip(v, preds["base"])) / len(v)}
    pairs = [("trained", "base"), ("merged", "trained")] + ([("control", "base")] if a.control else [])
    rep["paired"] = {f"{x}_minus_{y}": dict(zip(("diff", "ci_low", "ci_high"), paired_bootstrap(correct[y], correct[x])))
                     for x, y in pairs}
    rep["paired_auc"] = {f"{x}_minus_{y}": paired_auc_bootstrap(W[x], W[y], seed=BOOTSTRAP_SEED) for x, y in pairs}
    rep["merged_vs_trained_prediction_agreement"] = sum(
        p["response"] == q["response"] for p, q in zip(preds["merged"], preds["trained"])) / len(test)
    rep["predictions"] = {k: [p["response"] for p in v] for k, v in preds.items()}
    rep["label_scores"] = {"ids": [e["id"] for e in test], **scores}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k, m in rep["models"].items():
        print(f"{k:8s} acc {m['accuracy']:.4f}  macro-F1 {m.get('macro_f1', 0):.4f}  AUC {m['auc']:.4f}  "
              f"invalid {m['invalid_rate']:.4f}  pred POS {m['predicted_positive_rate']:.3f}  "
              f"changed vs base {m['changed_vs_base']:.3f}  ({rep['backends'][k]})")
    print(json.dumps(rep["paired"]), f"\nwritten: {out}")


if __name__ == "__main__":
    main()
