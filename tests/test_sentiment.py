import random

import pytest

from sentiment import LABELS, _dedup, _key, make_splits, parse_label, reward


@pytest.mark.parametrize(
    "response, label, expected",
    [
        ("POSITIVE", "POSITIVE", 1.0),
        ("NEGATIVE", "NEGATIVE", 1.0),
        ("Positive", "POSITIVE", 1.0),
        (" positive \n", "POSITIVE", 1.0),
        ("negative", "NEGATIVE", 1.0),
        ("POSITIVE", "NEGATIVE", 0.0),  # wrong label
        ("NEGATIVE", "POSITIVE", 0.0),
        ("positif", "POSITIVE", 0.0),  # invalid formats
        ("NEGATIVE.", "NEGATIVE", 0.0),
        ("**POSITIVE**", "POSITIVE", 0.0),
        ("The sentiment is POSITIVE", "POSITIVE", 0.0),
        ("POSITIVE NEGATIVE", "POSITIVE", 0.0),  # ambiguous
        ("", "NEGATIVE", 0.0),  # missing
        ("   ", "POSITIVE", 0.0),
    ],
)
def test_reward(response, label, expected):
    assert reward(response, label) == expected


def test_parse_label_invalid_is_none():
    assert parse_label("NEGATIVE.") is None
    assert parse_label(" negative ") == "NEGATIVE"


def _raw(seed=0):
    rng = random.Random(seed)
    split = lambda name, n: [{"text": f"{name} sentence {i}", "label": rng.choice(LABELS)} for i in range(n)]
    raw = {"train": split("train", 400), "validation": split("val", 60), "test": split("test", 60)}
    # near-duplicates of a test sentence planted in validation and train, and a repeat inside train
    raw["validation"].append({"text": "TEST sentence, 0!", "label": "POSITIVE"})
    raw["train"].append({"text": "test   sentence 0", "label": "NEGATIVE"})
    raw["train"].append({"text": "Train sentence 1.", "label": raw["train"][1]["label"]})
    return raw


def test_splits_are_balanced_and_sized():
    s = make_splits(_raw(), n_train=100)
    assert len(s["train"]) == 100
    for name in ("train", "validation", "test"):
        counts = [sum(r["label"] == label for r in s[name]) for label in LABELS]
        assert counts[0] == counts[1] > 0


def test_no_near_duplicates_within_or_across_splits():
    s = make_splits(_raw(), n_train=100)
    keys = [_key(r["text"]) for name in ("train", "validation", "test") for r in s[name]]
    assert len(keys) == len(set(keys))


def test_dedup_removes_taken_keys_and_internal_repeats():
    rows = [{"text": t, "label": "POSITIVE"} for t in ("A good film.", "a GOOD film", "Fine", "Boring!")]
    assert [r["text"] for r in _dedup(rows, {"boring"})] == ["A good film.", "Fine"]


def test_original_labels_preserved():
    raw = _raw()
    source = {r["text"]: r["label"] for name in raw for r in raw[name]}
    s = make_splits(raw, n_train=100)
    for name in ("train", "validation", "test"):
        assert all(source[r["text"]] == r["label"] for r in s[name])


def test_splits_deterministic_in_seed():
    assert make_splits(_raw(), n_train=100, seed=0) == make_splits(_raw(), n_train=100, seed=0)
    assert make_splits(_raw(), n_train=100, seed=0) != make_splits(_raw(), n_train=100, seed=1)
