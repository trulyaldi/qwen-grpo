import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards [B, G] -> advantages [B, G], normalized within each group (population std)."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    return (rewards - mean) / (std + eps)


def response_mask(response_ids: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """response_ids [N, R] -> bool mask [N, R]: tokens up to and including the first EOS.

    Built from positions, not from `ids != pad_id`, because Qwen3's pad id is also an EOS id.
    A response with no EOS (truncated) is kept whole.
    """
    is_eos = torch.isin(response_ids, torch.tensor(eos_ids, device=response_ids.device))
    eos_before = torch.cumsum(is_eos.long(), dim=1) - is_eos.long()
    return eos_before == 0


def response_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """log pi(token_t | tokens_<t) for the response tokens.

    logits    [N, T, V]  model output for the full (left-padded prompt + response) sequence
    input_ids [N, T]     prompts occupy columns [0, prompt_len), responses [prompt_len, T)
    returns   [N, T - prompt_len]

    The logits at column t-1 score the token at column t.
    """
    pred = logits[:, prompt_len - 1 : -1]
    pred = pred.to(torch.promote_types(pred.dtype, torch.float32))  # bf16 -> fp32, fp64 stays fp64
    targets = input_ids[:, prompt_len:]
    return torch.log_softmax(pred, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped GRPO loss (beta = 0), per-sequence mean then mean over sequences.

    logp, old_logp [N, R]   current / sampling-time log-probs of the response tokens
    advantages     [N]      one advantage per sequence, shared by its tokens
    mask           [N, R]   bool, True for response tokens that count
    returns        scalar   -J, so minimizing it maximizes the GRPO objective
    """
    old_logp = old_logp.detach()
    log_ratio = torch.where(mask, logp - old_logp, torch.zeros_like(logp))
    ratio = torch.exp(log_ratio)
    adv = advantages.unsqueeze(1)
    surrogate = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
    surrogate = torch.where(mask, surrogate, torch.zeros_like(surrogate))
    lengths = mask.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("Empty response detected.")
    per_sequence = surrogate.sum(dim=1) / lengths
    return -per_sequence.mean()
