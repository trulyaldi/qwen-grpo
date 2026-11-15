"""GRPO training mechanics: prompt batching, rollout grouping, rewards, and one optimizer update per rollout batch."""
import itertools
import math
import random

import torch

from qwen_grpo.grpo import group_advantages, grpo_loss
from qwen_grpo.model import EOS_IDS
from qwen_grpo.policy import compute_logprobs
from qwen_grpo.sentiment import LABELS, reward

INIT_TOL = 1e-6  # max |current - old| log-prob allowed before an update (they are identical passes)


def masked_max(x, mask):
    return x[mask].abs().max().item()



def decode(tok, rollouts):
    """Response text without its terminating EOS, as evaluation.generate() does."""
    out = []
    for row, m in zip(rollouts["response_ids"], rollouts["completion_mask"]):
        ids = row[m].tolist()
        out.append(tok.decode(ids[:-1] if ids[-1] in EOS_IDS else ids))
    return out



def batches(train, B, mode, seed=42):
    """Deterministic, infinite stream of training batches. A zero-signal batch is consumed, never retried.

    cycle:         B=1; train[0], train[1], ... in subset order (the original policy)
    balanced:      B=2; [next POSITIVE, next NEGATIVE], each class reshuffled (own seeded RNG) after each pass
    unconstrained: B prompts from one stream over the whole subset, reshuffled after each pass
    """
    def stream(pool, rng):
        while True:
            order = list(pool)
            rng.shuffle(order)
            yield from order

    if mode == "cycle":
        assert B == 1
        return ([train[i % len(train)]] for i in itertools.count())
    if mode == "balanced":
        assert B == 2
        pos, neg = (stream([e for e in train if e["label"] == label], random.Random(seed + k))
                    for k, label in enumerate(("POSITIVE", "NEGATIVE")))
        return ([next(pos), next(neg)] for _ in itertools.count())
    assert mode == "unconstrained"
    s = stream(train, random.Random(seed))
    return ([next(s) for _ in range(B)] for _ in itertools.count())



def check_grouping(rollouts, prompt_ids, G):
    """Rows b*G .. b*G+G-1 must start with prompt b's left-padded tokens, the order group_rewards assumes."""
    P = rollouts["prompt_len"]
    assert torch.equal(rollouts["input_ids"][:, :P], prompt_ids.repeat_interleave(G, dim=0)), "rollout rows not grouped by prompt"



def balanced_subset(rows, n, seed):
    """n/2 rows per label, drawn and shuffled with a fixed seed; id = index in `rows`."""
    assert n % 2 == 0
    rng = random.Random(seed)
    picked = [i for label in LABELS for i in rng.sample([i for i, r in enumerate(rows) if r["label"] == label], n // 2)]
    rng.shuffle(picked)
    return [{"id": i, **rows[i]} for i in picked]



def group_rewards(responses, labels, G):
    """responses [B*G], rows b*G .. b*G+G-1 answer prompt b -> rewards [B, G]."""
    assert len(responses) == len(labels) * G, (len(responses), len(labels), G)
    return torch.tensor([reward(r, labels[i // G]) for i, r in enumerate(responses)]).reshape(len(labels), G)



def grpo_step(model, params, optimizer, rollouts, rewards, clip_eps=0.2):
    """One AdamW update from a fresh rollout group -> scalar logs, or None (no step) if no group has mixed rewards."""
    if (rewards == rewards[:, :1]).all():
        return None
    adv = group_advantages(rewards).flatten().to(rollouts["input_ids"].device)
    mask = rollouts["completion_mask"]
    with torch.no_grad():
        old_logp = compute_logprobs(model, rollouts)  # sampling policy of this group only
    new_logp = compute_logprobs(model, rollouts)
    init_diff = masked_max(new_logp.detach() - old_logp, mask)
    if init_diff > INIT_TOL:
        raise RuntimeError(f"old/current log-prob mismatch before the step: {init_diff:.3e}")
    loss = grpo_loss(new_logp, old_logp, adv, mask, clip_eps=clip_eps)
    if not (torch.isfinite(adv).all() and torch.isfinite(loss)):
        raise RuntimeError(f"non-finite advantages {adv.tolist()} or loss {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in params.items()}
    if not all(math.isfinite(g) for g in grad_norms.values()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"non-finite gradients {grad_norms}; no optimizer step taken")
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(), "init_diff": init_diff, "grad_norms": grad_norms,
        "grad_norm": math.sqrt(sum(g * g for g in grad_norms.values())),
        "update_norm": math.sqrt(sum((p.detach() - before[n]).pow(2).sum().item() for n, p in params.items())),
    }

