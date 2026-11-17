"""Ticket 13: held-out evaluation of the final run on the test split, through the L2L engine only.

    python scripts/final_eval.py --run qwen3_4b_final --control qwen3_4b_final_lr0 --device cuda

Models (the same test prompts, greedy decoding through execution.generator; never model.generate):
  base      the untuned model (the trained run's PEFT model with the adapter disabled)
  trained   the adapter-attached model at the run's last checkpoint
  merged    the trained adapter merged tensor by tensor (export.merge) and loaded as a plain HF model
  control   the LR=0 negative-control run's last checkpoint (adapter attached)
Reported per model: accuracy, macro F1, invalid-response rate, prediction distribution (evaluation.metrics; reward =
1 for a correct canonical label, so mean reward = accuracy). Paired bootstrap 95% CIs of the accuracy difference
(evaluation.paired_bootstrap, paired by test example): trained - base, merged - trained, control - base. Policy change:
the fraction of test predictions that changed vs base; LoRA change: norms of the trained and control adapters minus
their step-0 adapters. Data separation: the run's training prompt ids (from its log) must not occur in the test split
(different split, checked by text). Writes research/l2l/experiments/t13_final_eval_<run>.json.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qwen_grpo import runs  # noqa: E402
from qwen_grpo.checkpointing import load_adapter  # noqa: E402
from qwen_grpo.config import DTYPES, load  # noqa: E402
from qwen_grpo.evaluation import evaluate, metrics, paired_bootstrap  # noqa: E402
from qwen_grpo.execution.backend import L2LBackend  # noqa: E402
from qwen_grpo.export.merge import load_merged_l2l, stream_merge  # noqa: E402
from qwen_grpo.model import lora_model  # noqa: E402
from qwen_grpo.sentiment import load_splits  # noqa: E402


def adapter_change(ckpt, step0):
    from peft import load_peft_weights
    a, b = load_peft_weights(str(ckpt), device="cpu"), load_peft_weights(str(step0), device="cpu")
    return {"l2_change": sum((a[k].float() - b[k].float()).pow(2).sum().item() for k in a) ** 0.5,
            "max_abs_change": max((a[k].float() - b[k].float()).abs().max().item() for k in a)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--control")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, help="first n test examples (default: all)")
    ap.add_argument("--storage", default="pinned")
    a = ap.parse_args(argv)
    cfg = load(runs.run_dir(a.run) / "config.yaml", historical=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    test = [{"id": i, **r} for i, r in enumerate(load_splits()["test"])][: a.n]
    log = json.loads((runs.training_dir(a.run) / "training.json").read_text())
    train_texts = {r["text"] for r in load_splits()["train"]}
    rep = {"run": a.run, "control": a.control, "model": cfg.model.name, "n_test": len(test),
           "data_separation": {"test_texts_in_train": sum(e["text"] in train_texts for e in test),
                               "train_ids_used": len(log["config"]["train_ids"])},
           "training": {"steps": log.get("successful_steps"), "attempts": log.get("groups_generated"),
                        "zero_signal": log.get("zero_signal_groups"), "frozen_changed": log.get("frozen_changed"),
                        "reward_curve": [r.get("mean_reward") for r in log.get("training", [])]}, "models": {}}
    ckpt = runs.latest_checkpoint(a.run)
    lo = cfg.lora
    model = lora_model(DTYPES[cfg.model.dtype], "cpu", lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
    load_adapter(model, ckpt)
    be = L2LBackend(model, a.device, a.storage, 2)
    preds = {}
    preds["trained"] = evaluate(model, tok, test, generator=be.greedy)
    with model.disable_adapter():
        preds["base"] = evaluate(model, tok, test, generator=be.greedy)
    if torch.cuda.is_available():
        rep["peak_gpu_allocated_bytes_eval"] = torch.cuda.max_memory_allocated()
    be.close()
    del model, be
    with tempfile.TemporaryDirectory(prefix="t13_merged_") as tmp:
        merge_rep = stream_merge(cfg.model.name, ckpt, Path(tmp) / "merged")
        merged, mbe = load_merged_l2l(Path(tmp) / "merged", a.device, DTYPES[cfg.model.dtype], a.storage, 2)
        preds["merged"] = evaluate(merged, tok, test, generator=mbe.greedy)
        mbe.close()
        del merged, mbe
    rep["merge"] = {"all_within_half_ulp": all(v["within_half_ulp"] for v in merge_rep["merged"].values()),
                    "adapted_tensors": len(merge_rep["merged"])}
    rep["lora_change"] = {"trained": adapter_change(ckpt, runs.checkpoint_path(a.run, 0))}
    if a.control:
        cckpt = runs.latest_checkpoint(a.control)
        cm = lora_model(DTYPES[cfg.model.dtype], "cpu", lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
        load_adapter(cm, cckpt)
        cbe = L2LBackend(cm, a.device, a.storage, 2)
        preds["control"] = evaluate(cm, tok, test, generator=cbe.greedy)
        cbe.close()
        rep["lora_change"]["control"] = adapter_change(cckpt, runs.checkpoint_path(a.control, 0))
        clog = json.loads((runs.training_dir(a.control) / "training.json").read_text())
        rep["control_log"] = {"params_equal_init": clog.get("params_equal_init"), "control": clog.get("control")}
    correct = {k: [p["correct"] for p in v] for k, v in preds.items()}
    for k, v in preds.items():
        rep["models"][k] = {**metrics(v), "mean_reward": sum(correct[k]) / len(v),
                            "changed_vs_base": sum(p["response"] != q["response"] for p, q in zip(v, preds["base"])) / len(v)}
    pairs = [("trained", "base"), ("merged", "trained")] + ([("control", "base")] if a.control else [])
    rep["paired"] = {f"{x}_minus_{y}": dict(zip(("diff", "ci_low", "ci_high"), paired_bootstrap(correct[y], correct[x])))
                     for x, y in pairs}
    rep["merged_vs_trained_prediction_agreement"] = sum(
        p["response"] == q["response"] for p, q in zip(preds["merged"], preds["trained"])) / len(test)
    rep["predictions"] = {k: [p["response"] for p in v] for k, v in preds.items()}
    out = ROOT / "research" / "l2l" / "experiments" / f"t13_final_eval_{a.run}.json"
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    for k, m in rep["models"].items():
        print(f"{k:8s} acc {m['accuracy']:.4f}  macro-F1 {m.get('macro_f1', 0):.4f}  invalid {m['invalid_rate']:.4f}  "
              f"pred POS {m['predicted_positive_rate']:.3f}  changed vs base {m['changed_vs_base']:.3f}")
    print(json.dumps(rep["paired"]), f"\nwritten: {out}")


if __name__ == "__main__":
    main()
