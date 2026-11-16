"""GRPO training mechanics: prompt batching, rollout grouping, rewards, and training.num_iterations optimizer updates per
rollout batch, all against the batch's sampling-time log-probs."""
import itertools
import math
import random

import torch

from qwen_grpo.grpo import clip_stats, group_advantages, grpo_loss
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



def shuffled_batches(train, B, seed=42):
    """Deterministic, infinite stream of B-prompt batches read from one sequence: the subset in an order shuffled by
    its own random.Random(seed), reshuffled after each full pass (epoch). Within an epoch every prompt appears once; a
    batch that straddles two epochs may repeat a prompt. No class balance is enforced per batch. A zero-signal batch is
    consumed, never retried. This is the historical batch_mode "unconstrained", element for element."""
    rng = random.Random(seed)

    def epochs():
        while True:
            order = list(train)
            rng.shuffle(order)
            yield from order

    s = epochs()
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



def prepare(model, rollouts, rewards):
    """Fixed targets of one rollout batch -> (advantages [B*G], old_logp [B*G, R]), or None if no group has mixed
    rewards. old_logp is the sampling policy's log-prob, computed once (no graph) and reused by every update."""
    if (rewards == rewards[:, :1]).all():
        return None
    adv = group_advantages(rewards).flatten().to(rollouts["input_ids"].device)
    with torch.no_grad():
        old_logp = compute_logprobs(model, rollouts)
    return adv, old_logp


def grpo_update(model, params, optimizer, rollouts, adv, old_logp, clip_eps=0.2, iteration=0):
    """One AdamW update of the clipped objective against the fixed old_logp -> scalar logs. Iteration 0 must see
    ratio 1 (same parameters as the sampling policy); later iterations recompute log-probs under the updated adapter."""
    mask = rollouts["completion_mask"]
    new_logp = compute_logprobs(model, rollouts)
    out = {"iteration": iteration}
    if iteration == 0:
        out["init_diff"] = masked_max(new_logp.detach() - old_logp, mask)
        if out["init_diff"] > INIT_TOL:
            raise RuntimeError(f"old/current log-prob mismatch before the step: {out['init_diff']:.3e}")
    loss = grpo_loss(new_logp, old_logp, adv, mask, clip_eps=clip_eps)
    if not (torch.isfinite(adv).all() and torch.isfinite(loss)):
        raise RuntimeError(f"non-finite advantages {adv.tolist()} or loss {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norms = {n: p.grad.norm().item() for n, p in params.items()}
    if not all(math.isfinite(g) for g in grad_norms.values()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"non-finite gradients {grad_norms}; no optimizer step taken")
    stats = clip_stats(new_logp.detach(), old_logp, adv, mask, clip_eps)  # ratios at the parameters being updated
    del new_logp  # frees the graph before the next forward
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        **out, "loss": loss.item(), "grad_norms": grad_norms,
        "grad_norm": math.sqrt(sum(g * g for g in grad_norms.values())),
        "update_norm": math.sqrt(sum((p.detach() - before[n]).pow(2).sum().item() for n, p in params.items())),
        **stats,
    }


def grpo_step(model, params, optimizer, rollouts, rewards, clip_eps=0.2, num_iterations=1):
    """num_iterations sequential updates on one rollout batch -> list of per-update logs, or None (no step) if no
    group has mixed rewards. trainer._train runs the same loop, interleaved with checkpoints."""
    targets = prepare(model, rollouts, rewards)
    if targets is None:
        return None
    return [grpo_update(model, params, optimizer, rollouts, *targets, clip_eps, k) for k in range(num_iterations)]
