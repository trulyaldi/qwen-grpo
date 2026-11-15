"""Entry point. Every run trained from now on lives in runs/<run>/ (layout: qwen_grpo/runs.py).

python scripts/run.py train    --config configs/experiments/<exp>.yaml
python scripts/run.py train    --config runs/<run>/config.yaml --resume runs/<run>/checkpoints/step<N>
python scripts/run.py validate --run <run> [--step N]            held-out validation split -> runs/<run>/evaluation/step<N>/
python scripts/run.py test     --run <run> [--step N]            test split (final report only)
python scripts/run.py verify   --run <run> [--step N]            -> runs/<run>/verification/step<N>/
python scripts/run.py report   --run <run>                       rebuild plots, summary.json, report.md (no model, no GPU)
python scripts/run.py remove   --run <run> [--confirm]           dry run unless --confirm

--run takes runs/<run>/config.yaml and the checkpoint at --step (default: the latest). The older form
--config <yaml> --checkpoint <dir> still works for every command except train, for checkpoints under runs/ and for runs
from before runs/ existed (checkpoints/<run>_step<N>/), whose outputs stay in results/eval/ and results/verification/.

validate/test evaluate a saved adapter on the full split with the baseline's prompt, greedy decoding and parser (YAML
evaluation section) and compare it with the base model (paired: the saved baseline predictions if they match model and
decoding, else the base evaluated in-process). verify compares base, adapter-attached, merged and (if configured) an
LR=0 control on verification.split. Neither retrains the checkpoint; only verification.negative_control.train runs a
(control) training, as the separate run <run>_lr0. remove lists what it would delete with sizes and deletes only with
--confirm; it refuses when ownership is uncertain (qwen_grpo/cleanup.py).
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
    from qwen_grpo.config import DTYPES, as_dict, sha256
    from qwen_grpo import runs
    from qwen_grpo.evaluation import (evaluate, label_scores, metrics, paired_auc_bootstrap, paired_compare,
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
        "checkpoint": runs.rel(checkpoint),
        "split": split, "config": as_dict(cfg), "config_sha256": sha256(cfg), "decoding": decoding,
        "metrics": m, "auc": {k: w.mean().item() for k, w in W.items()},
        "vs_baseline": {**paired_compare(base_preds, preds, **boot), "auc": paired_auc_bootstrap(W["trained"], W["base"], **boot)},
        "baseline_source": runs.rel(path) if base else "in-process (adapter disabled)",
        "base_scores_max_abs_diff_vs_saved": (max(abs(a - b) for a, b in zip(scores["base"], base["label_scores"]["scores"]))
                                              if base else None),
        "environment": environment(splits, "cuda"),
        "label_scores": {"ids": [e["id"] for e in examples], **scores}, "predictions": preds,
    }
    path = runs.write_json(runs.evaluation_output(checkpoint, split), out)

    v = out["vs_baseline"]
    print(f"{split} ({out['checkpoint']}): {summary(m)}")
    print(f"  baseline:  {summary(v['ref_metrics'])}")
    print(f"  accuracy diff {v['accuracy_diff']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
    print(f"  AUC trained {out['auc']['trained']:.4f}  base {out['auc']['base']:.4f}  diff {v['auc']['auc_diff']:+.4f}  "
          f"95% CI [{v['auc']['ci95'][0]:+.4f}, {v['auc']['ci95'][1]:+.4f}]")
    if base:
        print(f"  base label scores vs saved baseline: max |diff| {out['base_scores_max_abs_diff_vs_saved']:.2e}")
    print(f"  wrote {runs.rel(path)}")
    updated(checkpoint)
    return out


def updated(checkpoint):
    """Rebuild the run's summary and report after new results for one of its checkpoints (runs/ layout only)."""
    from qwen_grpo import runs

    name, _, layout = runs.parse_checkpoint(checkpoint)
    if layout == "run":
        from qwen_grpo.reporting import update_run
        print(f"updated {runs.rel(update_run(name))}")


def control_checkpoint(cfg):
    """The LR=0 control for verify: an existing checkpoint, or a fresh run of train() with learning_rate 0 (as the run
    <run>_lr0) if (and only if) verification.negative_control.train is set. Its provenance is checked either way."""
    import dataclasses

    from qwen_grpo import runs
    from qwen_grpo.verification import control_provenance

    nc = cfg.verification.negative_control
    if nc.train:
        from qwen_grpo.trainer import train

        name = f"{cfg.run.name}_lr0"
        train(dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, name=name),
                                  training=dataclasses.replace(cfg.training, learning_rate=0.0, full_val=False)))
        ckpt = runs.ROOT / runs.read_json(runs.training_dir(name) / "training.json")["final_checkpoint"]["path"]
        import gc

        import torch

        gc.collect()  # train() returns with its model and optimizer unreferenced; release them before verify loads a model
        torch.cuda.empty_cache()
    elif nc.checkpoint:
        ckpt = runs.ROOT / nc.checkpoint
    else:
        return None
    ok, why, _ = control_provenance(cfg, ckpt)
    if not ok:
        raise SystemExit(f"LR=0 control rejected: {why}")
    return ckpt


def verify(cfg, checkpoint):
    from qwen_grpo import reporting, runs
    from qwen_grpo.verification import verify as collect

    out, data = runs.verification_output(checkpoint)
    evidence = collect(cfg, checkpoint, control_checkpoint(cfg))
    reporting.save(evidence, data)
    path = reporting.render(out, data)
    print(f"wrote {runs.rel(out)}/ ; report: {runs.rel(path)}")
    updated(checkpoint)
    return evidence


def report(checkpoint):
    """Rebuild one verification report (plots and report.md) from its saved evidence."""
    from qwen_grpo import reporting, runs

    out, data = runs.verification_output(checkpoint)
    if not (data / "manifest.json").exists():
        raise SystemExit(f"no saved verification for {checkpoint}: run verify first ({runs.rel(data)} missing)")
    print(f"wrote {runs.rel(reporting.render(out, data))}")
    updated(checkpoint)


def report_run(name):
    from qwen_grpo import runs
    from qwen_grpo.reporting import update_run

    if not runs.run_dir(name).is_dir():
        runs.resolve(name)  # raises, with a hint for a run from before runs/ existed
    path = update_run(name)
    print(f"wrote {runs.rel(runs.run_dir(name) / 'summary.json')} and {runs.rel(path)}")
    return path


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", *SPLITS, "verify", "report", "remove"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=name == "train")
        if name == "train":
            p.add_argument("--resume", type=Path, help="continue the run from this checkpoint of it")
            continue
        p.add_argument("--checkpoint", type=Path)
        p.add_argument("--run", help="a run under runs/ (instead of --config and --checkpoint)")
        if name not in ("report", "remove"):
            p.add_argument("--step", type=int, help="checkpoint step of --run (default: the latest)")
        if name == "remove":
            p.add_argument("--confirm", action="store_true", help="delete the listed paths (default: dry run)")
    args = ap.parse_args(argv)
    if args.command != "train":
        by_run = args.run is not None and args.config is None and args.checkpoint is None
        by_paths = args.run is None and args.config is not None and args.checkpoint is not None
        if not (by_run or by_paths):
            ap.error(f"{args.command}: give --run <name>, or both --config and --checkpoint")
        if getattr(args, "step", None) is not None and not by_run:
            ap.error("--step selects a checkpoint of --run")
    return args


def main(argv=None):
    from qwen_grpo import runs
    from qwen_grpo.config import load

    args = parse(argv)
    if args.command == "train":
        from qwen_grpo.trainer import train

        return train(load(args.config), resume=args.resume)
    if args.run is not None:
        if args.command == "report":
            return report_run(args.run)
        if args.command == "remove":
            from qwen_grpo.cleanup import remove_run

            return remove_run(args.run, confirm=args.confirm)
        config_path, checkpoint = runs.resolve(args.run, args.step)
        cfg = load(config_path)
    else:
        cfg, checkpoint = load(args.config), args.checkpoint
    if args.command == "remove":
        from qwen_grpo.cleanup import remove

        return remove(cfg, runs.ROOT, checkpoint, confirm=args.confirm)
    if not (checkpoint / "adapter_config.json").exists():
        raise SystemExit(f"not a LoRA checkpoint (no adapter_config.json): {checkpoint}")
    if args.command == "verify":
        return verify(cfg, checkpoint)
    if args.command == "report":
        return report(checkpoint)
    return evaluate_checkpoint(cfg, checkpoint, SPLITS[args.command])


if __name__ == "__main__":
    main()
