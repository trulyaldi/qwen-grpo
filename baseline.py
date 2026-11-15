"""Greedy baseline of untuned Qwen3-0.6B (non-thinking) on the sentiment test split."""
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from grpo import response_mask
from sentiment import LABELS, load_splits, parse_label, prompt_text, reward

MODEL, SPLIT = "Qwen/Qwen3-0.6B", "validation"
MAX_NEW, BATCH = 16, 32
EOS_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>
OUT = Path(__file__).parent / "results" / f"baseline_{SPLIT}.json"


def build_prompts(tok, sentences):
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": prompt_text(s)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for s in sentences
    ]
    assert all(t.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n") for t in texts), texts[0]
    return texts


@torch.no_grad()
def generate(model, tok, sentences):
    """Greedy responses -> (text without EOS, token length incl. EOS, truncated) per sentence."""
    batch = tok(build_prompts(tok, sentences), return_tensors="pt", padding=True).to(model.device)
    ids = model.generate(**batch, do_sample=False, max_new_tokens=MAX_NEW)[:, batch["input_ids"].shape[1]:]
    mask = response_mask(ids, EOS_IDS)
    out = []
    for row, m in zip(ids.tolist(), mask):
        n = int(m.sum())
        truncated = row[n - 1] not in EOS_IDS
        out.append((tok.decode(row[: n if truncated else n - 1]), n, truncated))
    return out


def predict(model, tok, examples, ids):
    """Greedy prediction records in BATCH-sized chunks (order matters: bf16 outputs depend on batch shape)."""
    predictions, lengths, truncated = [], [], []
    for i in range(0, len(examples), BATCH):
        chunk = examples[i : i + BATCH]
        for k, ex, (response, n, trunc) in zip(ids[i : i + BATCH], chunk, generate(model, tok, [ex["text"] for ex in chunk])):
            predictions.append({
                "id": k,
                "text": ex["text"],
                "label": ex["label"],
                "response": response,
                "prediction": parse_label(response),
                "correct": reward(response, ex["label"]) == 1.0,
            })
            lengths.append(n)
            truncated.append(trunc)
    return predictions, lengths, truncated


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
