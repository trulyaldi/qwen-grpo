"""Streamed autoregressive generation (Ticket 06): prefill + token-by-token decode through a layer executor, with an
explicitly placed per-layer KV cache. Never calls model.generate().

The loop reproduces transformers' GenerationMixin._sample (4.57) for the settings policy.generate_rollouts uses:
  - the B prompts are expanded to B*G rows with repeat_interleave (num_return_sequences);
  - position_ids = cumsum(attention_mask) - 1, padding positions set to 1 (prepare_inputs_for_generation), sliced to
    the new columns; cache_position = 0..P-1 for the prefill, then P + step;
  - attention masks: create_causal_mask with this cache (the HF builder, the same mask format as with DynamicCache);
  - logits of the last column only (logits_to_keep=1), copied to fp32; the next token is
    multinomial(softmax(logits)) (temperature 1, no top-k / top-p: no logits processor changes them) or argmax
    (greedy);
  - a finished row receives pad_token_id; a row finishes when it emits any EOS id; the loop stops when every row has
    finished or max_new_tokens tokens were appended; the attention mask grows by a 1 per step for every row.
The layers are the unchanged HF/PEFT decoder layers (LoRA runs inside them, with the current adapter parameters),
called through a layer executor: ResidentLayerExecutor, or layer_stream.StreamingLayerExecutor for L2L, which
streams every layer's weights in for every prefill and decode step. GQA, RoPE and attention are HF's modules.

KV cache (KVCache): one preallocated [N, kv_heads, P + max_new, head_dim] key and value buffer per decoder layer, on
`kv_device`, chosen independently of where the weights live. A layer's update writes its new columns into its buffers
and returns the filled prefix as new contiguous tensors on the compute device (as DynamicCache's torch.cat would),
so attention never reads a view of a buffer and the cache may live on the CPU while layers run on the GPU (each
layer's prefix is then copied in for its attention only).

teacher forcing: `forced` [N, R] tokens replace the sampled ones (finished rows still get pad), so next-token
distributions of different execution paths can be compared along the same prefixes. step_logits keeps every step's
fp32 next-token logits [N, V].

Shapes: B prompts, G completions per prompt, N = B*G rows, P prompt columns, R new columns, V vocabulary.
"""
import torch
from transformers.masking_utils import create_causal_mask

from qwen_grpo.execution.qwen3_forward import ResidentLayerExecutor, unwrap


class KVCache:
    """Per-layer preallocated keys/values on `device` (duck-types the transformers Cache interface the Qwen3 attention
    and the mask builder use: update, get_seq_length, get_mask_sizes, is_sliding, is_compileable)."""

    is_compileable = False

    def __init__(self, num_layers, rows, kv_heads, capacity, head_dim, dtype, device, compute_device):
        shape = (rows, kv_heads, capacity, head_dim)
        self.keys = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.values = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.lengths = [0] * num_layers
        self.capacity, self.compute_device = capacity, torch.device(compute_device)
        self.is_sliding = [False] * num_layers

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        start, q = self.lengths[layer_idx], key_states.shape[-2]
        if start + q > self.capacity:
            raise RuntimeError(f"KV cache overflow at layer {layer_idx}: {start} + {q} > {self.capacity}")
        k, v = self.keys[layer_idx], self.values[layer_idx]
        k[:, :, start:start + q].copy_(key_states)
        v[:, :, start:start + q].copy_(value_states)
        self.lengths[layer_idx] = start + q
        end = start + q
        return (k[:, :, :end].to(self.compute_device, copy=True).contiguous(),
                v[:, :, :end].to(self.compute_device, copy=True).contiguous())

    def get_seq_length(self, layer_idx=0):
        return self.lengths[layer_idx]

    def get_mask_sizes(self, cache_position, layer_idx=0):
        return self.lengths[layer_idx] + cache_position.shape[0], 0


def _positions(mask):
    """prepare_inputs_for_generation's position ids: cumsum(mask) - 1, padding positions set to 1."""
    pos = mask.long().cumsum(-1) - 1
    return pos.masked_fill(mask == 0, 1)


@torch.no_grad()
def _step(model, ids, mask, position_ids, cache_position, cache, executor):
    """One forward over the new columns `ids` [N, q] -> fp32 next-token logits [N, V] (last column)."""
    decoder, lm_head = unwrap(model)
    hidden = decoder.embed_tokens(ids)                                              # [N, q, H]
    attn = create_causal_mask(config=decoder.config, input_embeds=hidden, attention_mask=mask,
                              cache_position=cache_position, past_key_values=cache, position_ids=position_ids)
    position_embeddings = decoder.rotary_emb(hidden, position_ids)
    for index, layer in enumerate(decoder.layers[: decoder.config.num_hidden_layers]):
        hidden = executor.execute(index, layer, hidden, attention_mask=attn, position_ids=position_ids,
                                  past_key_values=cache, use_cache=True, cache_position=cache_position,
                                  position_embeddings=position_embeddings)
    hidden = decoder.norm(hidden)
    return lm_head(hidden[:, -1:, :])[:, -1, :].to(dtype=torch.float32, copy=True)  # [N, V]


def check_model(model):
    config = unwrap(model)[0].config
    if getattr(config, "use_sliding_window", False) or "sliding_attention" in (getattr(config, "layer_types", None) or []):
        raise NotImplementedError("sliding-window attention layers are not supported by the streamed generator")
    if model.training:
        raise RuntimeError("generation needs eval mode (dropout off)")
    return config


@torch.no_grad()
def generate(model, input_ids, attention_mask, eos_token_ids, pad_token_id, max_new_tokens, num_return_sequences=1,
             do_sample=True, layer_executor=None, kv_device=None, forced=None, generator=None, keep_logits=False):
    """-> {"sequences": [N, P + R] ids, "step_logits": list of R fp32 [N, V] (if keep_logits), "kv_cache_bytes",
    "steps"}. input_ids/attention_mask: [B, P] left-padded prompts; rows are expanded to N = B*G. kv_device: where the
    KV cache lives (default: the input device). generator: torch.Generator for sampling (default: the global RNG,
    as HF uses)."""
    config = check_model(model)
    executor = ResidentLayerExecutor() if layer_executor is None else layer_executor
    device = input_ids.device
    eos = torch.tensor(list(eos_token_ids), device=device)
    ids = input_ids.repeat_interleave(num_return_sequences, dim=0)                 # [N, P]
    mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)
    N, P = ids.shape
    if forced is not None and forced.shape[0] != N:
        raise ValueError(f"forced has {forced.shape[0]} rows, expected {N}")
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dtype = unwrap(model)[0].embed_tokens.weight.dtype
    cache = KVCache(config.num_hidden_layers, N, config.num_key_value_heads, P + max_new_tokens, head_dim, dtype,
                    kv_device or device, device)
    unfinished = torch.ones(N, dtype=torch.long, device=device)
    step_logits = []
    new_cols = ids                                                                   # prefill: the whole prompt
    cache_position = torch.arange(P, device=device)
    for step in range(max_new_tokens):
        position_ids = _positions(mask)[:, -new_cols.shape[1]:]
        logits = _step(model, new_cols, mask, position_ids, cache_position, cache, executor)   # [N, V] fp32
        if keep_logits:
            step_logits.append(logits)
        if forced is not None:
            nxt = forced[:, step]
        elif do_sample:
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=generator).squeeze(1)
        else:
            nxt = logits.argmax(dim=-1)
        nxt = nxt * unfinished + pad_token_id * (1 - unfinished)
        ids = torch.cat([ids, nxt[:, None]], dim=-1)
        mask = torch.cat([mask, mask.new_ones((N, 1))], dim=-1)
        unfinished = unfinished & ~torch.isin(nxt, eos)
        if unfinished.max() == 0:
            break
        new_cols = nxt[:, None]
        cache_position = cache_position[-1:] + 1
    return {"sequences": ids, "step_logits": step_logits, "kv_cache_bytes": cache.nbytes, "steps": ids.shape[1] - P}


class StreamedGenerator:
    """The rollout generator policy.generate_rollouts uses instead of model.generate(): pure sampling (temperature 1,
    no top-k/top-p) through `layer_executor` with the KV cache on `kv_device`."""

    def __init__(self, layer_executor=None, kv_device=None):
        self.layer_executor, self.kv_device = layer_executor, kv_device
        self.last = None

    def __call__(self, model, input_ids, attention_mask, num_return_sequences, max_new_tokens, eos_token_ids,
                 pad_token_id):
        self.last = generate(model, input_ids, attention_mask, eos_token_ids, pad_token_id, max_new_tokens,
                             num_return_sequences, do_sample=True, layer_executor=self.layer_executor,
                             kv_device=self.kv_device)
        return self.last["sequences"]
