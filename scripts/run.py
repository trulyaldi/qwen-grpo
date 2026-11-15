"""Config-driven entry point.

python scripts/run.py train    --config configs/default.yaml [--resume checkpoints/<run>_step<N>]
python scripts/run.py validate --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
python scripts/run.py test     --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
python scripts/run.py verify   --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
python scripts/run.py report   --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>

train --resume continues a run from one of its checkpoints (same YAML; see qwen_grpo/checkpointing.py).
validate/test evaluate a saved adapter on the full split with the baseline's prompt, greedy decoding and parser (YAML
evaluation section), compare it with the base model (paired: the saved baseline predictions if they match model and
decoding, else the base evaluated in-process), and write results/eval/<checkpoint>/<split>.json.
verify compares base, adapter-attached, merged and (if configured) an LR=0 control on verification.split and writes
results/verification/<checkpoint>/ (evidence JSON, plots, report.md); report rebuilds plots and report.md from those files
without loading a model. Neither retrains the checkpoint; only verification.negative_control.train runs a (control) training.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script from anywhere: make qwen_grpo/ importable

SPLITS = {"validate": "validation", "test": "test"}


def evaluate_checkpoint(cfg, checkpoint, split):
    from transformers import AutoTokenizer

    from qwen_grpo.checkpointing import check_adapter_matches, load_adapter
    from qwen_grpo.config import DTYPES, as_dict
    from qwen_grpo.evaluation import (RESULTS, evaluate, label_scores, metrics, paired_auc_bootstrap, paired_compare,
                                      predictions_path, settings, summary, win_matrix)
    from qwen_grpo.model import lora_model
    from qwen_grpo.sentiment import load_splits
    from qwen_grpo.trainer import environment

    checkpoint = Path(checkpoint).resolve()
    check_adapter_matches(cfg, checkpoint)
    decoding = {"do_sample": False, "max_new_tokens": cfg.evaluation.max_new_tokens}
    path = predictions_path(split)
    base = json.loads(path.read_text()) if path.exists() else None
    if base is not None and (base["model"], base["decoding"]) != (cfg.model.name, decoding):
        print(f"saved baseline {path.name} ({base['model']}, {base['decoding']}) does not match; evaluating the base in-process")
        base = None

    lo, (gen_kw, score_kw) = cfg.lora, settings(cfg.evaluation)
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    model = lora_model(DTYPES[cfg.model.dtype], "cuda", lo.spec(), lo.rank, lo.alpha, lo.dropout, cfg.model.name)
    load_adapter(model, checkpoint)

    splits = load_splits()
    examples = [{"id": i, **e} for i, e in enumerate(splits[split])]
    preds = evaluate(model, tok, examples, **gen_kw)
    scores = {"trained": label_scores(model, tok, examples, **score_kw)}
    with model.disable_adapter():
        scores["base"] = label_scores(model, tok, examples, **score_kw)
        base_preds = base["predictions"] if base else evaluate(model, tok, examples, **gen_kw)
    labels = [e["label"] for e in examples]
    W = {k: win_matrix(v, labels) for k, v in scores.items()}
    m = metrics(preds)
    boot = {"n": cfg.evaluation.bootstrap_samples, "seed": cfg.evaluation.bootstrap_seed}
    out = {
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        "split": split, "config": as_dict(cfg), "decoding": decoding,
        "metrics": m, "auc": {k: w.mean().item() for k, w in W.items()},
        "vs_baseline": {**paired_compare(base_preds, preds, **boot), "auc": paired_auc_bootstrap(W["trained"], W["base"], **boot)},
        "baseline_source": str(path.relative_to(ROOT)) if base else "in-process (adapter disabled)",
        "base_scores_max_abs_diff_vs_saved": (max(abs(a - b) for a, b in zip(scores["base"], base["label_scores"]["scores"]))
                                              if base else None),
        "environment": environment(splits, "cuda"),
        "label_scores": {"ids": [e["id"] for e in examples], **scores}, "predictions": preds,
    }
    path = RESULTS / "eval" / checkpoint.name / f"{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))

    v = out["vs_baseline"]
    print(f"{split} ({out['checkpoint']}): {summary(m)}")
    print(f"  baseline:  {summary(v['ref_metrics'])}")
    print(f"  accuracy diff {v['accuracy_diff']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(f"  AUC trained {out['auc']['trained']:.4f}  base {out['auc']['base']:.4f}  diff {v['auc']['auc_diff']:+.4f}  "
          f"95% CI [{v['auc']['ci95'][0]:+.4f}, {v['auc']['ci95'][1]:+.4f}]")
    if base:
        print(f"  base label scores vs saved baseline: max |diff| {out['base_scores_max_abs_diff_vs_saved']:.2e}")
    print(f"  wrote {path.relative_to(ROOT)}")
    return out


def control_checkpoint(cfg):
    """The LR=0 control for verify: an existing checkpoint, or a fresh run of train() with learning_rate 0 if (and only if)
    verification.negative_control.train is set. Either way its provenance is checked against this config."""
    import dataclasses

    from qwen_grpo.verification import control_provenance

    nc = cfg.verification.negative_control
    if nc.train:
        from qwen_grpo.trainer import train

        name = f"{cfg.run.name}_lr0"
        train(dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, name=name),
                                  training=dataclasses.replace(cfg.training, learning_rate=0.0, full_val=False)))
        ckpt = ROOT / json.loads((ROOT / "results" / name / "training.json").read_text())["final_checkpoint"]["path"]
        import gc

        import torch

        gc.collect()  # train() returns with its model and optimizer unreferenced; release them before verify loads a model
        torch.cuda.empty_cache()
    elif nc.checkpoint:
        ckpt = ROOT / nc.checkpoint
    else:
        return None
    ok, why, _ = control_provenance(cfg, ckpt)
    if not ok:
        raise SystemExit(f"LR=0 control rejected: {why}")
    return ckpt


def verify(cfg, checkpoint):
    from qwen_grpo import reporting
    from qwen_grpo.verification import verify as collect

    out = ROOT / "results" / "verification" / Path(checkpoint).resolve().name
    evidence = collect(cfg, checkpoint, control_checkpoint(cfg))
    reporting.save(evidence, out)
    path = reporting.render(out)
    print(f"wrote {out.relative_to(ROOT)}/ ; report: {path.relative_to(ROOT)}")
    return evidence


def report(checkpoint):
    from qwen_grpo import reporting

    out = ROOT / "results" / "verification" / Path(checkpoint).resolve().name
    if not (out / "manifest.json").exists():
        raise SystemExit(f"no saved verification for {checkpoint}: run verify first ({out.relative_to(ROOT)} missing)")
    print(f"wrote {reporting.render(out).relative_to(ROOT)}")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", *SPLITS, "verify", "report"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        if name == "train":
            p.add_argument("--resume", type=Path, help="continue the run from this checkpoint of it")
        else:
            p.add_argument("--checkpoint", required=True, type=Path)
    return ap.parse_args(argv)


def main(argv=None):
    from qwen_grpo.config import load

    args = parse(argv)
    cfg = load(args.config)
    if args.command == "train":
        from qwen_grpo.trainer import train

        return train(cfg, resume=args.resume)
    if not (args.checkpoint / "adapter_config.json").exists():
        raise SystemExit(f"not a LoRA checkpoint (no adapter_config.json): {args.checkpoint}")
    if args.command == "verify":
        return verify(cfg, args.checkpoint)
    if args.command == "report":
        return report(args.checkpoint)
    return evaluate_checkpoint(cfg, args.checkpoint, SPLITS[args.command])


if __name__ == "__main__":
    main()
