"""Pure checks on data, reward, adapters and run artifacts: each returns a list of problems (empty = holds). Used by
the fault-injection tests (tests/silent_failures/), never by the training math."""
import re

from qwen_grpo.sentiment import LABELS, reward

# (response, label, expected reward): the reward contract the GRPO signal relies on (sentiment.reward / parse_label)
REWARD_CASES = [
    ("POSITIVE", "POSITIVE", 1.0), ("NEGATIVE", "NEGATIVE", 1.0), ("positive", "POSITIVE", 1.0),
    (" POSITIVE \n", "POSITIVE", 1.0),
    ("NEGATIVE", "POSITIVE", 0.0), ("", "POSITIVE", 0.0), ("   ", "NEGATIVE", 0.0),
    ("POSITIVE NEGATIVE", "POSITIVE", 0.0), ("POSITIVE.", "POSITIVE", 0.0), ("POSITIVE!", "POSITIVE", 0.0),
    ("The sentiment is POSITIVE", "POSITIVE", 0.0), ("POSITIVEPOSITIVE", "POSITIVE", 0.0),
    ("<think>POSITIVE</think>", "POSITIVE", 0.0), ("NEUTRAL", "POSITIVE", 0.0), ("POS", "POSITIVE", 0.0),
    ("POSITIVE\nNEGATIVE", "NEGATIVE", 0.0), ("**POSITIVE**", "POSITIVE", 0.0),
]


def _key(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def split_problems(splits):
    """Labels outside LABELS, duplicate texts inside a split, and texts shared between splits (leakage)."""
    problems, seen = [], {}
    for name in ("train", "validation", "test"):
        rows = splits[name]
        if bad := [i for i, r in enumerate(rows) if r.get("label") not in LABELS]:
            problems.append(f"{name}: {len(bad)} rows with a label outside {LABELS} (first index {bad[0]})")
        keys = [_key(r["text"]) for r in rows]
        if len(set(keys)) != len(keys):
            problems.append(f"{name}: {len(keys) - len(set(keys))} duplicate texts")
        for other, other_keys in seen.items():
            if shared := set(keys) & other_keys:
                problems.append(f"{name} and {other} share {len(shared)} texts (split leakage)")
        seen[name] = set(keys)
    return problems


def reward_problems(fn=reward, cases=REWARD_CASES):
    """Cases where the reward differs from the contract, or returns something other than 0.0 / 1.0."""
    problems = []
    for response, label, want in cases:
        got = fn(response, label)
        if not isinstance(got, float) or got not in (0.0, 1.0):
            problems.append(f"reward({response!r}, {label}) = {got!r}: not 0.0 / 1.0")
        elif got != want:
            problems.append(f"reward({response!r}, {label}) = {got} (expected {want})")
    return problems


def adapter_problems(initial, trained, min_relative_change=1e-4):
    """LoRA state dicts (step-0 vs trained, same keys): missing or extra tensors, non-finite values, an all-zero B after
    training (an inactive adapter: B starts at zero, so the adapter contributes nothing), and an adapter whose
    relative change from its initialization is below min_relative_change (unexpectedly unchanged)."""
    import torch
    problems = []
    if missing := sorted(set(initial) - set(trained)):
        problems.append(f"tensors missing from the trained adapter: {missing[:4]}")
    if extra := sorted(set(trained) - set(initial)):
        problems.append(f"unexpected tensors in the trained adapter: {extra[:4]}")
    common = sorted(set(initial) & set(trained))
    if bad := [k for k in common if not torch.isfinite(trained[k]).all()]:
        problems.append(f"non-finite values in {bad[:4]}")
    b_keys = [k for k in common if "lora_B" in k]
    if b_keys and all(trained[k].abs().max() == 0 for k in b_keys):
        problems.append("every lora_B is zero: the adapter is inactive (contributes exactly nothing)")
    num = sum((trained[k].float() - initial[k].float()).pow(2).sum().item() for k in common)
    den = sum(initial[k].float().pow(2).sum().item() for k in common) or 1.0
    if (num / den) ** 0.5 < min_relative_change:
        problems.append(f"relative change from initialization {(num / den) ** 0.5:.2e} < {min_relative_change:g}: the "
                        "adapter is unexpectedly unchanged")
    return problems


def rollout_record_problems(records, epoch_size=None):
    """On-disk rollout records of a streaming run, in commit order: duplicate group ids, policy versions that go
    backwards, and a (epoch, position) prompt occurrence used twice (a duplicated sample) or skipped. A skip inside an
    epoch shows as non-contiguous positions; a skip at the end of an epoch only with epoch_size (the number of training
    prompts): every epoch before the last must then hold exactly positions 0 .. epoch_size - 1."""
    problems, groups, occ = [], [], []
    for r in records:
        groups.append(r["group_id"])
        occ += [tuple(o[:2]) for o in r["occurrences"]]
    if len(set(groups)) != len(groups):
        problems.append(f"duplicate rollout groups {sorted({g for g in groups if groups.count(g) > 1})}")
    versions = [r["policy_version"] for r in records]
    if any(b < a for a, b in zip(versions, versions[1:])):
        problems.append(f"policy versions go backwards: {versions}")
    if len(set(occ)) != len(occ):
        problems.append("a prompt occurrence (epoch, position) was committed twice")
    by_epoch = {}
    for e, pos in occ:
        by_epoch.setdefault(e, []).append(pos)
    for e, positions in by_epoch.items():
        if sorted(positions) != list(range(len(positions))):
            problems.append(f"epoch {e}: positions are not contiguous from 0 (a skipped sample)")
        elif epoch_size is not None and e != max(by_epoch) and len(positions) != epoch_size:
            problems.append(f"epoch {e}: {len(positions)} of {epoch_size} samples before the next epoch began "
                            "(skipped samples)")
    return problems
