"""Greedy baseline of untuned Qwen3-0.6B (non-thinking): predictions, the metrics GRPO + LoRA runs report, and label-score AUC.

Run from the repo root: .venv/bin/python -m research.baseline --split validation
Writes raw predictions to results/baseline/<split>.json and metrics to results/benchmarks/baseline.{json,md}
(one entry per split; running one split keeps the other's entry).
"""
import argparse
import json
import statistics

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from qwen_grpo.evaluation import (BENCHMARKS, MAX_NEW, RESULTS, confusion_table, label_scores, metrics, metrics_table,
                                  predict, predictions_path, summary, win_matrix)
from qwen_grpo.model import EOS_IDS, MODEL
from qwen_grpo.sentiment import load_splits

DECODING = {"do_sample": False, "max_new_tokens": MAX_NEW}


def write_report(bench):
    md = [f"# Baseline: {bench['model']} (untuned, non-thinking)", "",
          "Greedy decoding (settings per split below). AUC: complete-label scores s(x) = log P(POSITIVE<|im_end|>) − log P(NEGATIVE<|im_end|>).", "",
          metrics_table({s: {**e["metrics"], "auc": e["auc"]} for s, e in bench["splits"].items()})]
    for s, e in bench["splits"].items():
        g = e["generation"]
        md += ["", f"## {s}", "", f"Raw predictions: `{e['predictions']}`. Decoding {e['decoding']}. Truncated {g['truncated_rate']:.3f}, response "
               f"tokens min/median/max {g['length_min']} / {g['length_median']} / {g['length_max']}.", "",
               confusion_table(e["metrics"])]
    (BENCHMARKS / "baseline.json").write_text(json.dumps(bench, indent=1))
    (BENCHMARKS / "baseline.md").write_text("\n".join(md) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    split = ap.parse_args().split

    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    assert tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]) == EOS_IDS
    # Replace Qwen's shipped sampling defaults, which generate() would otherwise merge into our settings.
    model.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)

    examples = load_splits()[split]
    ids = list(range(len(examples)))
    predictions, lengths, truncated = predict(model, tok, examples, ids)
    m = metrics(predictions)
    scores = label_scores(model, tok, examples)
    auc = win_matrix(scores, [e["label"] for e in examples]).mean().item()
    generation = {"truncated_rate": sum(truncated) / len(truncated),
                  "length_min": min(lengths), "length_median": statistics.median(lengths), "length_max": max(lengths)}

    print(summary(m))
    print(f"ROC AUC (label scores) {auc:.4f}")
    print(f"truncated {generation['truncated_rate']:.3f}  length min/median/max "
          f"{generation['length_min']} / {generation['length_median']} / {generation['length_max']}")

    path = predictions_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "model": MODEL, "checkpoint": "base", "split": split, "decoding": DECODING,
        "label_scores": {"ids": ids, "scores": scores},
        "predictions": predictions,
    }, indent=1))

    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    bench_path = BENCHMARKS / "baseline.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {"splits": {}}
    bench.update(model=MODEL, checkpoint="base")
    bench["splits"][split] = {"predictions": str(path.relative_to(RESULTS.parent)), "decoding": DECODING, "metrics": m, "auc": auc, "generation": generation}
    bench["splits"] = {s: bench["splits"][s] for s in ("validation", "test") if s in bench["splits"]}
    write_report(bench)


if __name__ == "__main__":
    main()
