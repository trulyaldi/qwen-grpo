"""Explicit Qwen3 forward: HF's own modules (embeddings, decoder layers, final norm, LM head) called one by one.

Reproduces Qwen3ForCausalLM.forward (transformers 4.57, no KV cache) step by step, so the per-layer loop is ours and a
later step can decide where each layer's weights live before it runs. No transformer arithmetic is reimplemented: every
operation is an existing module or HF helper, and LoRA runs because PEFT's adapters sit inside the decoder layers.

Shapes: N rows, T columns, H hidden size, V vocabulary, K = kept logit columns.
"""
import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from qwen_grpo.grpo import response_logprobs


def unwrap(model):
    """(Qwen3Model, lm_head) of a Qwen3ForCausalLM, PEFT-wrapped or not."""
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    return causal_lm.model, causal_lm.lm_head


def attention_masks(decoder, embeds, attention_mask, position_ids):
    """{attention_type: mask} as Qwen3Model.forward builds it. An all-ones mask may come back as None (sdpa then uses
    is_causal); a padded one is a [N, 1, T, T] additive/boolean mask in the format of the configured attention."""
    kwargs = dict(config=decoder.config, input_embeds=embeds, attention_mask=attention_mask,
                  cache_position=torch.arange(embeds.shape[1], device=embeds.device),
                  past_key_values=None, position_ids=position_ids)
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if decoder.has_sliding_layers:
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    return masks


def explicit_forward(model, input_ids, attention_mask=None, position_ids=None, logits_to_keep=0):
    """Logits [N, K, V] of model(input_ids, attention_mask, position_ids, use_cache=False, logits_to_keep=...).

    input_ids      [N, T]
    attention_mask [N, T] 1 = token, 0 = padding (None: no padding)
    position_ids   [N, T] or None for 0..T-1 (HF's default); left-padded rows need cumsum(mask) - 1 as generate() uses
    logits_to_keep int (0 = all T columns, k = last k) or index tensor, sliced as Qwen3ForCausalLM does
    """
    decoder, lm_head = unwrap(model)
    T = input_ids.shape[1]

    hidden = decoder.embed_tokens(input_ids)                                     # [N, T, H]
    if position_ids is None:
        position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)      # [1, T]
    masks = attention_masks(decoder, hidden, attention_mask, position_ids)        # None or [N, 1, T, T]
    position_embeddings = decoder.rotary_emb(hidden, position_ids)                # (cos, sin), each [N|1, T, head_dim]

    for layer in decoder.layers[: decoder.config.num_hidden_layers]:             # the explicit layer loop
        hidden = layer(hidden, attention_mask=masks[layer.attention_type], position_ids=position_ids,
                       position_embeddings=position_embeddings)                   # [N, T, H] -> [N, T, H]

    hidden = decoder.norm(hidden)                                                 # [N, T, H]
    keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    return lm_head(hidden[:, keep, :])                                            # [N, K, V]


def explicit_logprobs(model, rollouts):
    """policy.compute_logprobs through explicit_forward: log pi(o_t | q, o_<t) for the response columns -> [N, R]."""
    ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
    R = ids.shape[1] - rollouts["prompt_len"]
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    logits = explicit_forward(model, ids, mask, position_ids, logits_to_keep=R + 1)  # [N, R+1, V]
    return response_logprobs(logits, ids, rollouts["prompt_len"])
