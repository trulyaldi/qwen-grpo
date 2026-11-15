"""Greedy baseline of untuned Qwen3-0.6B (non-thinking) on the sentiment test split."""
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from qwen_grpo.evaluation import MAX_NEW, predict
from qwen_grpo.model import EOS_IDS, MODEL
from qwen_grpo.sentiment import LABELS, load_splits

SPLIT = "validation"
OUT = Path(__file__).parents[1] / "results" / f"baseline_{SPLIT}.json"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    assert tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]) == EOS_IDS
    # Replace Qwen's shipped sampling defaults, which generate() would otherwise merge into our settings.
    model.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)

    examples = load_splits()[SPLIT]
    predictions, lengths, truncated = predict(model, tok, examples, list(range(len(examples))))

    mean = lambda xs: sum(xs) / len(xs)
    print(f"n = {len(predictions)}")
    print(f"accuracy        {mean([p['correct'] for p in predictions]):.3f}")
    for label in LABELS:
        print(f"recall {label:9} {mean([p['correct'] for p in predictions if p['label'] == label]):.3f}")
    print(f"truncated       {mean(truncated):.3f}")
    print(f"length min/median/max  {min(lengths)} / {statistics.median(lengths)} / {max(lengths)}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "model": MODEL,
        "checkpoint": "base",
        "split": SPLIT,
        "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW},
        "predictions": predictions,
    }, indent=1))


if __name__ == "__main__":
    main()
