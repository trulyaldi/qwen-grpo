"""Config-driven entry point.

python scripts/run.py train    --config configs/default.yaml
python scripts/run.py validate --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
python scripts/run.py test     --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>

validate/test evaluate a saved adapter on the full split with the baseline's prompt, greedy decoding and parser, compare it
with the saved baseline predictions of that split (paired), and write results/eval/<checkpoint>/<split>.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run as a script from anywhere: make qwen_grpo/ and research/ importable

SPLITS = {"validate": "validation", "test": "test"}


def check_adapter_matches(cfg, checkpoint):
    """The adapter's saved config must be the one the YAML describes; otherwise loading would silently mismatch."""
    from qwen_grpo.model import TARGETS

    saved = json.loads((checkpoint / "adapter_config.json").read_text())
    expected = {"r": cfg.lora.rank, "lora_alpha": cfg.lora.alpha, "lora_dropout": cfg.lora.dropout,
                "target_modules": TARGETS[cfg.lora.targets], "base_model_name_or_path": cfg.model.name}
    got = {k: saved.get(k) for k in expected}
    if got != expected:
        raise SystemExit(f"{checkpoint} does not match the config:\n  checkpoint {got}\n  config     {expected}")


def evaluate_checkpoint(cfg, checkpoint, split):
    import torch
    from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
    from transformers import AutoTokenizer

    from qwen_grpo.config import DTYPES, as_dict
    from qwen_grpo.evaluation import (MAX_NEW, evaluate, label_scores, metrics, paired_auc_bootstrap, paired_compare, summary,
                                      win_matrix)
    from qwen_grpo.model import TARGETS, lora_model
    from qwen_grpo.sentiment import load_splits
    from research.baseline import RESULTS, predictions_path

    checkpoint = Path(checkpoint).resolve()
    check_adapter_matches(cfg, checkpoint)
    base = json.loads(predictions_path(split).read_text())
    if base["decoding"] != {"do_sample": False, "max_new_tokens": MAX_NEW}:
        raise SystemExit(f"baseline decoding {base['decoding']} differs from evaluation decoding (max_new_tokens={MAX_NEW})")

    lo = cfg.lora
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    model = lora_model(DTYPES[cfg.model.dtype], "cuda", TARGETS[lo.targets], lo.rank, lo.alpha, lo.dropout, cfg.model.name)
    loaded = load_peft_weights(str(checkpoint))
    live = get_peft_model_state_dict(model)
    if live.keys() != loaded.keys():
        raise SystemExit(f"adapter tensors differ: missing {sorted(live.keys() - loaded.keys())}, extra {sorted(loaded.keys() - live.keys())}")
    set_peft_model_state_dict(model, loaded)
    assert all(torch.equal(p, loaded[k].to(p.device)) for k, p in get_peft_model_state_dict(model).items()), "adapter not loaded"

    examples = [{"id": i, **e} for i, e in enumerate(load_splits()[split])]
    preds = evaluate(model, tok, examples)
    scores = {"trained": label_scores(model, tok, examples)}
    with model.disable_adapter():
        scores["base"] = label_scores(model, tok, examples)
    labels = [e["label"] for e in examples]
    W = {k: win_matrix(v, labels) for k, v in scores.items()}
    m = metrics(preds)
    out = {
        "checkpoint": str(checkpoint.relative_to(ROOT)) if checkpoint.is_relative_to(ROOT) else str(checkpoint),
        "split": split, "config": as_dict(cfg), "decoding": base["decoding"],
        "metrics": m, "auc": {k: w.mean().item() for k, w in W.items()},
        "vs_baseline": {**paired_compare(base["predictions"], preds), "auc": paired_auc_bootstrap(W["trained"], W["base"])},
        "base_scores_max_abs_diff_vs_saved": max(abs(a - b) for a, b in zip(scores["base"], base["label_scores"]["scores"])),
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
    print(f"  base label scores vs saved baseline: max |diff| {out['base_scores_max_abs_diff_vs_saved']:.2e}")
    print(f"  wrote {path.relative_to(ROOT)}")
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("train", *SPLITS):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True, type=Path)
        if name in SPLITS:
            p.add_argument("--checkpoint", required=True, type=Path)
    return ap.parse_args(argv)


def main(argv=None):
    from qwen_grpo.config import load

    args = parse(argv)
    cfg = load(args.config)
    if args.command == "train":
        from research.train_multistep import train

        return train(cfg)
    if not (args.checkpoint / "adapter_config.json").exists():
        raise SystemExit(f"not a LoRA checkpoint (no adapter_config.json): {args.checkpoint}")
    return evaluate_checkpoint(cfg, args.checkpoint, SPLITS[args.command])


if __name__ == "__main__":
    main()
