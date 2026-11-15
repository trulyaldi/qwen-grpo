import json
import random
import re
from pathlib import Path

LABELS = ("NEGATIVE", "POSITIVE")  # index = rotten_tomatoes label id
DATASET = "cornell-movie-review-data/rotten_tomatoes"
SPLITS_PATH = Path(__file__).parents[1] / "data" / "sentiment.json"
N_TRAIN, SEED = 1500, 0

PROMPT = (
    "Classify the sentiment of the following sentence as POSITIVE or NEGATIVE.\n"
    "ONE WORD ONLY.\n\n"
    "Sentence: {sentence}"
)


def prompt_text(sentence: str) -> str:
    return PROMPT.format(sentence=sentence)



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


def parse_label(response: str) -> str | None:
    """The predicted label, or None if the response is not exactly one valid label."""
    label = response.strip().upper()
    return label if label in LABELS else None


def reward(response: str, label: str) -> float:
    return float(parse_label(response) == label)


def _key(text: str) -> str:
    """Near-duplicate key: lowercase alphanumeric words, punctuation and spacing ignored."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _dedup(rows: list[dict], taken: set[str]) -> list[dict]:
    seen, out = set(taken), []
    for r in rows:
        k = _key(r["text"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _balanced(rows: list[dict], n_per_label: int, rng: random.Random) -> list[dict]:
    out = [r for label in LABELS for r in rng.sample([r for r in rows if r["label"] == label], n_per_label)]
    rng.shuffle(out)
    return out


def _min_label_count(rows: list[dict]) -> int:
    return min(sum(r["label"] == label for r in rows) for label in LABELS)


def make_splits(raw: dict, n_train: int = N_TRAIN, seed: int = SEED) -> dict:
    """raw: {"train" | "validation" | "test": [{"text", "label"}]} -> balanced, deduplicated splits.

    Test is claimed first, then validation, then train; each drops sentences whose key already
    appears in an earlier split or earlier in itself. Train is subsampled to n_train;
    validation and test keep as many as balance allows.
    """
    rng = random.Random(seed)
    test = _dedup(raw["test"], set())
    val = _dedup(raw["validation"], {_key(r["text"]) for r in test})
    train = _dedup(raw["train"], {_key(r["text"]) for r in test + val})
    return {
        "dataset": DATASET,
        "seed": seed,
        "train": _balanced(train, n_train // 2, rng),
        "validation": _balanced(val, _min_label_count(val), rng),
        "test": _balanced(test, _min_label_count(test), rng),
    }


def load_splits() -> dict:
    """Build the splits once from the original dataset labels, then reuse the cached file."""
    if not SPLITS_PATH.exists():
        from datasets import load_dataset

        ds = load_dataset(DATASET)
        assert ds["train"].features["label"].names == ["neg", "pos"]  # id order matches LABELS
        raw = {s: [{"text": r["text"], "label": LABELS[r["label"]]} for r in ds[s]] for s in ("train", "validation", "test")}
        SPLITS_PATH.parent.mkdir(exist_ok=True)
        SPLITS_PATH.write_text(json.dumps(make_splits(raw), indent=1))
    return json.loads(SPLITS_PATH.read_text())
