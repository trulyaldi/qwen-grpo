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



def micro_batches(rollouts, size=None):
    """(row slice, rollouts restricted to those rows) in chunks of `size` completions, the last one possibly smaller;
    size None: the whole batch. Slices are views, and every chunk keeps the batch's padded width and prompt_len."""
    N = rollouts["input_ids"].shape[0]
    size = size or N
    for i in range(0, N, size):
        rows = slice(i, min(i + size, N))
        yield rows, {k: v[rows] if isinstance(v, torch.Tensor) else v for k, v in rollouts.items()}


def prepare(model, rollouts, rewards, micro_batch_size=None, logprobs=None):
    """Fixed targets of one rollout batch -> (advantages [B*G], old_logp [B*G, R]), or None if no group has mixed
    rewards. old_logp is the sampling policy's log-prob, computed once (no graph) and reused by every update, in the
    same micro-batches as the updates: bf16 outputs depend on batch shape, and iteration 0 needs ratio exactly 1.
    logprobs: (model, rollouts) -> [n, R], default compute_logprobs; grpo_update must be given the same one."""
    logprobs = logprobs or compute_logprobs
    if (rewards == rewards[:, :1]).all():
        return None
    adv = group_advantages(rewards).flatten().to(rollouts["input_ids"].device)
    with torch.no_grad():
        old_logp = torch.cat([logprobs(model, c) for _, c in micro_batches(rollouts, micro_batch_size)])
    return adv, old_logp


def grpo_update(model, params, optimizer, rollouts, adv, old_logp, clip_eps=0.2, iteration=0, micro_batch_size=None,
                logprobs=None):
    """One AdamW update of the clipped objective against the fixed old_logp -> scalar logs. Iteration 0 must see
    ratio 1 (same parameters as the sampling policy); later iterations recompute log-probs under the updated adapter.

    Micro-batches: the loss is -(1/N) sum_i J_i over all N = B*G completions (J_i: completion i's length-normalized
    clipped surrogate). grpo_loss of a chunk c of n_c completions is -(1/n_c) sum_{i in c} J_i, so (n_c / N) times it,
    summed over chunks, is exactly the full-batch loss; each chunk is backpropagated (and its graph freed) before the
    next, and the gradients accumulate to the full-batch gradient. A single chunk is not rescaled at all.

    logprobs: (model, rollouts) -> [n, R] with autograd, default compute_logprobs (e.g. output_head.streamed_logprobs
    for the L2L / chunked-head path); it must be the function prepare() used, or the INIT_TOL check fails."""
    logprobs = logprobs or compute_logprobs
    if not torch.isfinite(adv).all():
        raise RuntimeError(f"non-finite advantages {adv.tolist()}")
    N = adv.shape[0]
    out, loss, new_logp = {"iteration": iteration}, None, []
    optimizer.zero_grad(set_to_none=True)
    for rows, chunk in micro_batches(rollouts, micro_batch_size):
        logp = logprobs(model, chunk)
        mask = chunk["completion_mask"]
        if iteration == 0:
            out["init_diff"] = max(out.get("init_diff", 0.0), masked_max(logp.detach() - old_logp[rows], mask))
            if out["init_diff"] > INIT_TOL:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(f"old/current log-prob mismatch before the step: {out['init_diff']:.3e}")
        part = grpo_loss(logp, old_logp[rows], adv[rows], mask, clip_eps=clip_eps)
        if (n_c := mask.shape[0]) != N:
            part = part * (n_c / N)
        if not torch.isfinite(part):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"non-finite loss {part.item()} on completions {rows.start}..{rows.stop - 1}")
        part.backward()  # frees this chunk's graph before the next forward
        loss = part.detach() if loss is None else loss + part.detach()
        new_logp.append(logp.detach())
        del logp, part
    new_logp = torch.cat(new_logp)
    grad_norms = {n: p.grad.norm().item() for n, p in params.items()}
    if not all(math.isfinite(g) for g in grad_norms.values()):
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError(f"non-finite gradients {grad_norms}; no optimizer step taken")
    stats = clip_stats(new_logp, old_logp, adv, rollouts["completion_mask"], clip_eps)  # ratios at the updated parameters
    before = {n: p.detach().clone() for n, p in params.items()}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        **out, "loss": loss.item(), "grad_norms": grad_norms,
        "grad_norm": math.sqrt(sum(g * g for g in grad_norms.values())),
        "update_norm": math.sqrt(sum((p.detach() - before[n]).pow(2).sum().item() for n, p in params.items())),
        **stats,
    }


def grpo_step(model, params, optimizer, rollouts, rewards, clip_eps=0.2, num_iterations=1, micro_batch_size=None):
    """num_iterations sequential updates on one rollout batch -> list of per-update logs, or None (no step) if no
    group has mixed rewards. trainer._train runs the same loop, interleaved with checkpoints."""
    targets = prepare(model, rollouts, rewards, micro_batch_size)
    if targets is None:
        return None
    return [grpo_update(model, params, optimizer, rollouts, *targets, clip_eps, k, micro_batch_size)
            for k in range(num_iterations)]
