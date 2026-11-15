"""Full-validation evaluation of the TRL adapter, exactly as the manual runs were evaluated, vs base and manual balanced GRPO.

Run from the repo root:  .venv/bin/python -m research.trl_reference.evaluate
Greedy evaluation.predict (16 new tokens, batch 32) and evaluation.label_scores on a fresh bf16 model; adapters
are loaded from disk. Base = the same model with B = 0 adapters (exactly the base model).
"""
import json
from collections import Counter

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer

from qwen_grpo.evaluation import (BOOTSTRAP, evaluate, label_scores, metrics, paired_auc_bootstrap, paired_bootstrap,
                                  paired_changes, summary, win_matrix)
from qwen_grpo.model import DTYPE, MODEL, TARGETS, lora_model
from qwen_grpo.sentiment import load_splits
from research.train_multistep import BASE_FILE, SEED

from research.trl_reference.train import CONFIG, HERE, MANUAL_TRAINING, NAME, ROOT

MANUAL_PREDICTIONS = ROOT / "results" / "grpo_lora_last2_b2_balanced_step30_validation.json"
MANUAL_CKPT = ROOT / "checkpoints" / "grpo_lora_last2_b2_balanced_step30"
TRL_CKPT = HERE / "checkpoints" / f"{NAME}_step{CONFIG['grpo_config']['max_steps']}"


def shift_stats(base, trained, labels):
    """Delta s = s_trained - s_base per example (s = complete-label log-odds POSITIVE vs NEGATIVE)."""
    b, t = torch.tensor(base, dtype=torch.float64), torch.tensor(trained, dtype=torch.float64)
    d, pos = t - b, torch.tensor([label == "POSITIVE" for label in labels])
    return {"n": len(d), "mean": d.mean().item(), "sd": d.std().item(), "median": d.median().item(),
            "frac_positive": (d > 0).double().mean().item(), "pearson_base_trained": torch.corrcoef(torch.stack([b, t]))[0, 1].item(),
            "mean_POSITIVE_gold": d[pos].mean().item(), "mean_NEGATIVE_gold": d[~pos].mean().item(),
            "min": d.min().item(), "max": d.max().item()}


def flips(ref, new):
    """Paired prediction changes ref -> new, matched by id."""
    out = paired_changes(ref, new)
    out["NEGATIVE->POSITIVE"] = out["flips"].get("NEGATIVE->POSITIVE", 0)
    out["POSITIVE->NEGATIVE"] = out["flips"].get("POSITIVE->NEGATIVE", 0)
    out["incorrect->correct"] = sum(not r["correct"] and n["correct"] for r, n in zip(ref, new))
    out["correct->incorrect"] = sum(r["correct"] and not n["correct"] for r, n in zip(ref, new))
    return out


def load_adapter(model, path):
    weights = load_peft_weights(str(path))
    set_peft_model_state_dict(model, weights)
    live = get_peft_model_state_dict(model)
    assert live.keys() == weights.keys() and all(torch.equal(live[k], weights[k].to(live[k].device)) for k in live), path


def max_abs(a, b):
    return (torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)).abs().max().item()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    torch.manual_seed(SEED)
    model = lora_model(DTYPE, "cuda", TARGETS[CONFIG["lora"]["targets"]])
    assert all(p.abs().max() == 0 for n, p in model.named_parameters() if ".lora_B." in n)  # B = 0: exactly the base model

    val = [{"id": i, **r} for i, r in enumerate(load_splits()["validation"])]
    labels = [e["label"] for e in val]
    key = [(e["id"], e["text"], e["label"]) for e in val]
    saved = {"base": json.loads(BASE_FILE.read_text())["predictions"], "manual": json.loads(MANUAL_PREDICTIONS.read_text())["predictions"]}
    assert all([(p["id"], p["text"], p["label"]) for p in v] == key for v in saved.values())
    saved_scores = json.loads(MANUAL_TRAINING.read_text())["final"]["label_scores"]
    assert saved_scores["ids"] == [e["id"] for e in val]

    base_preds = evaluate(model, tok, val)
    scores = {"base": label_scores(model, tok, val)}
    load_adapter(model, MANUAL_CKPT)
    scores["manual"] = label_scores(model, tok, val)
    load_adapter(model, TRL_CKPT)
    trl_preds = evaluate(model, tok, val)
    scores["trl"] = label_scores(model, tok, val)
    preds = {"base": saved["base"], "manual": saved["manual"], "trl": trl_preds}

    consistency = {
        "base_predictions_mismatched_ids_vs_saved": [p["id"] for p, q in zip(base_preds, saved["base"]) if p["response"] != q["response"]],
        "base_scores_max_abs_vs_saved": max_abs(scores["base"], saved_scores["base"]),
        "manual_scores_max_abs_vs_saved": max_abs(scores["manual"], saved_scores["trained"]),
    }
    W = {k: win_matrix(v, labels) for k, v in scores.items()}
    result = {
        "model": MODEL, "split": "validation", "n": len(val), "trl_checkpoint": str(TRL_CKPT.relative_to(ROOT)),
        "manual_checkpoint": str(MANUAL_CKPT.relative_to(ROOT)), "consistency": consistency,
        "metrics": {k: metrics(v) for k, v in preds.items()},
        "auc": {k: w.mean().item() for k, w in W.items()},
        "score_shift": {"trl_vs_base": shift_stats(scores["base"], scores["trl"], labels),
                        "manual_vs_base": shift_stats(scores["base"], scores["manual"], labels),
                        "corr_shift_trl_vs_manual": torch.corrcoef(torch.tensor([[t - b for t, b in zip(scores[k], scores["base"])]
                                                                                for k in ("trl", "manual")], dtype=torch.float64))[0, 1].item()},
        "flips": {"base->trl": flips(preds["base"], preds["trl"]), "base->manual": flips(preds["base"], preds["manual"]),
                  "manual->trl": flips(preds["manual"], preds["trl"])},
        "bootstrap": {"resamples": BOOTSTRAP, "seed": SEED, "paired_by": "id", "auc_resampling": "within class",
                      "note": "evaluation-sample uncertainty only; one training seed, so training variability is not captured"},
        "label_scores": {"ids": [e["id"] for e in val], **scores},
        "trl_predictions": trl_preds,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    for ref in ("base", "manual"):
        diff, lo, hi = paired_bootstrap([p["correct"] for p in preds[ref]], [p["correct"] for p in trl_preds])
        result["bootstrap"][f"trl_vs_{ref}"] = {"accuracy_diff": diff, "accuracy_ci95": [lo, hi], **paired_auc_bootstrap(W["trl"], W[ref])}
    out = HERE / "results" / "evaluation.json"
    out.write_text(json.dumps(result, indent=1))

    print(f"consistency: {consistency['base_predictions_mismatched_ids_vs_saved']} mismatched base predictions; "
          f"score reproduction max |diff| base {consistency['base_scores_max_abs_vs_saved']:.2e}, manual {consistency['manual_scores_max_abs_vs_saved']:.2e}")
    for k in preds:
        print(f"{k:7} {summary(result['metrics'][k])}  AUC {result['auc'][k]:.4f}")
    for k, s in result["score_shift"].items():
        print(f"{k}: {s if isinstance(s, float) else {a: round(b, 4) for a, b in s.items()}}")
    for k, f in result["flips"].items():
        print(f"{k}: NEG->POS {f['NEGATIVE->POSITIVE']}  POS->NEG {f['POSITIVE->NEGATIVE']}  incorrect->correct {f['incorrect->correct']}  "
              f"correct->incorrect {f['correct->incorrect']}  invalid->valid {f['invalid_to_valid']}  valid->invalid {f['valid_to_invalid']}  all {dict(Counter(f['flips']))}")
    for k in ("trl_vs_base", "trl_vs_manual"):
        b = result["bootstrap"][k]
        print(f"{k}: acc diff {b['accuracy_diff']:+.4f} CI [{b['accuracy_ci95'][0]:+.4f}, {b['accuracy_ci95'][1]:+.4f}]  "
              f"AUC diff {b['auc_diff']:+.4f} CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]")
    print(f"peak reserved {result['peak_reserved_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
