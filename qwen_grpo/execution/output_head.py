"""Memory-efficient output head (Ticket 05): selected-token log-probabilities without the [N, R, V] logits.

The reference (grpo.response_logprobs on Qwen3ForCausalLM logits, policy.compute_logprobs) computes, per response token,
    z = lm_head(h)                      [V] in the weight dtype
    log pi(t) = log_softmax(fp32(z))[t] = fp32(z)[t] - logsumexp(fp32(z))
over the whole output-head vocabulary: all V = weight.shape[0] rows (151,936 for Qwen3, more than the tokenizer's
151,669 tokens; the extra rows take part in the normalization exactly as in the HF forward, so they are kept here too).

_SelectedLogprobs computes the same quantity in vocabulary chunks of C rows, never holding more than [M, C] logits:
  forward   per chunk z_c = linear(h, W_c) in the weight dtype (the same per-element rounding as lm_head), upcast as
            response_logprobs does; a running maximum m and rescaled sum s give the exact full-vocabulary
            logsumexp = m + log s (online, numerically stable: every exponent is <= 0); the target's logit is picked
            from the chunk that holds it. Output (z_t - m) - log s, in fp32 (fp64 for an fp64 model).
            Saved for backward: h [M, H], the targets and the logsumexp [M]; no logits.
  backward  recomputes z_c chunk by chunk; dL/dz_c = g * (onehot_t - exp(z_c - lse)) in fp32, cast to the weight dtype
            as the reference's upcast backward does, and dL/dh = sum_c dL/dz_c @ W_c accumulated in fp32 and rounded
            to the hidden dtype once, as one matmul over V would be.
The head weight must be frozen (LoRA never targets lm_head or the tied embedding here); a weight that requires grad is
refused rather than given a silently missing gradient. Chunking changes the floating-point summation order of the
logsumexp and of dL/dh, so results agree with the reference within rounding, not bitwise (tolerances.yaml,
logprob_equivalence).

Shapes: N rows, T columns, R response columns, P = T - R prompt columns, M = N * R scored tokens, H hidden, V vocab.
"""
import torch
import torch.nn.functional as Fn

from qwen_grpo.execution.qwen3_forward import explicit_forward, explicit_hidden, unwrap
from qwen_grpo.grpo import response_logprobs


def _acc_dtype(dtype):
    return torch.promote_types(dtype, torch.float32)  # bf16/fp16 -> fp32, fp32 stays, fp64 stays (response_logprobs)


class _SelectedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        M, V = hidden.shape[0], weight.shape[0]
        acc = _acc_dtype(weight.dtype)
        m = torch.full((M,), -torch.inf, dtype=acc, device=hidden.device)
        s = torch.zeros(M, dtype=acc, device=hidden.device)
        picked = torch.zeros(M, dtype=acc, device=hidden.device)
        for lo in range(0, V, chunk):
            hi = min(lo + chunk, V)
            z = Fn.linear(hidden, weight[lo:hi]).to(acc)                          # [M, c]
            m_new = torch.maximum(m, z.amax(dim=1))
            s = s * torch.exp(m - m_new) + torch.exp(z - m_new[:, None]).sum(dim=1)
            m = m_new
            inside = (targets >= lo) & (targets < hi)
            z_t = z.gather(1, (targets - lo).clamp(0, hi - lo - 1)[:, None]).squeeze(1)
            picked = torch.where(inside, z_t, picked)
            del z
        lse = m + torch.log(s)
        ctx.chunk = chunk
        ctx.save_for_backward(hidden, weight, targets, lse)
        return (picked - m) - torch.log(s)

    @staticmethod
    def backward(ctx, g):                                                            # g [M]
        hidden, weight, targets, lse = ctx.saved_tensors
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        M, V, chunk = hidden.shape[0], weight.shape[0], ctx.chunk
        acc = lse.dtype
        grad_h = torch.zeros(hidden.shape, dtype=acc, device=hidden.device)
        rows = torch.arange(M, device=hidden.device)
        for lo in range(0, V, chunk):
            hi = min(lo + chunk, V)
            z = Fn.linear(hidden, weight[lo:hi]).to(acc)                          # [M, c]
            dz = torch.exp(z - lse[:, None]).mul_(-g[:, None])                    # -g * softmax
            inside = (targets >= lo) & (targets < hi)
            dz[rows[inside], targets[inside] - lo] += g[inside]                   # + g * onehot
            dz = dz.to(weight.dtype)                                             # as the reference's upcast backward
            grad_h += dz.to(acc) @ weight[lo:hi].to(acc)                          # fp32 accumulation over chunks
            del z, dz
        return grad_h.to(hidden.dtype), None, None, None


def selected_logprobs(hidden, weight, targets, chunk_size=None):
    """log softmax(fp32(hidden @ weight.T))[targets] -> [N, R] (fp32; fp64 for fp64 weights).

    hidden  [N, R, H] the final-norm hidden states at the columns that predict the targets
    weight  [V, H]    the output head (frozen)
    targets [N, R]    token ids in [0, V)
    chunk_size        None: the reference path (full [N, R, V] logits, grpo.response_logprobs' arithmetic);
                      C >= 1: _SelectedLogprobs over chunks of C vocabulary rows
    """
    N, R, H = hidden.shape
    V = weight.shape[0]
    if weight.requires_grad:
        raise RuntimeError("the output-head weight requires grad: the chunked head does not compute a weight gradient")
    if targets.numel() and (targets.min() < 0 or targets.max() >= V):
        raise ValueError(f"target ids outside [0, {V})")
    if chunk_size is None:
        logits = Fn.linear(hidden, weight).to(_acc_dtype(weight.dtype))         # [N, R, V]
        return torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if chunk_size < 1:
        raise ValueError(f"chunk_size {chunk_size} < 1")
    out = _SelectedLogprobs.apply(hidden.reshape(N * R, H), weight, targets.reshape(N * R), int(chunk_size))
    return out.view(N, R)


def output_head(model):
    """The LM head weight [V, H], after checking that it is what the HF forward uses: a bias-free Linear (not a LoRA
    layer), tied to the input embedding when the config says so."""
    decoder, lm_head = unwrap(model)
    if type(lm_head) is not torch.nn.Linear or lm_head.bias is not None:
        raise RuntimeError(f"unsupported output head {type(lm_head).__name__} (bias: {lm_head.bias is not None})")
    if model.config.tie_word_embeddings and lm_head.weight is not decoder.embed_tokens.weight:
        raise RuntimeError("tie_word_embeddings is set but lm_head.weight is not the embedding Parameter")
    return lm_head.weight


def streamed_logprobs(model, rollouts, vocab_chunk_size=None, layer_executor=None, layer_callback=None,
                      head_callback=None):
    """policy.compute_logprobs through explicit_forward (any layer executor, e.g. the L2L one) and the chunked head:
    log pi(o_t | q, o_<t) for the response columns -> [N, R]. Keeps autograd.

    The head scores only the columns P-1 .. T-2 (the logit at column t-1 scores the token at column t), the same R
    columns as compute_logprobs' logits_to_keep=R+1 minus its final column. head_callback(h) gets those hidden states
    [N, R, H] (the head's input; e.g. to retain their gradient)."""
    ids, mask, P = rollouts["input_ids"], rollouts["attention_mask"], rollouts["prompt_len"]
    if not 1 <= P < ids.shape[1]:
        raise ValueError(f"prompt_len {P}: need at least one prompt and one response column (T = {ids.shape[1]})")
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    weight = output_head(model)
    hidden = explicit_hidden(model, ids, mask, position_ids, layer_executor, layer_callback)    # [N, T, H]
    h = hidden[:, P - 1:-1]                                                                     # [N, R, H]
    if head_callback is not None:
        head_callback(h)
    return selected_logprobs(h, weight, ids[:, P:], vocab_chunk_size)


def full_logits_logprobs(model, rollouts, layer_executor=None):
    """The full-logits reference through explicit_forward: lm_head on the last R+1 columns, then response_logprobs
    (= qwen3_forward.explicit_logprobs, with a layer executor)."""
    ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
    R = ids.shape[1] - rollouts["prompt_len"]
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    logits = explicit_forward(model, ids, mask, position_ids, R + 1, layer_executor)            # [N, R+1, V]
    return response_logprobs(logits, ids, rollouts["prompt_len"])
