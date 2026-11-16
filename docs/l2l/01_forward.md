# Step 1: explicit Qwen3 forward

`qwen_grpo/execution/qwen3_forward.py` reproduces `Qwen3ForCausalLM.forward` (transformers 4.57.6, no KV cache) by
calling Hugging Face's own modules one at a time: embeddings, mask construction, rotary table, each decoder layer, final
norm, LM head. No transformer arithmetic is reimplemented. The result is **bitwise identical** to the HF forward, both
for the tiny test models and for the real Qwen3-0.6B, in fp32 and bf16, under eager and sdpa attention, with and
without LoRA.

## What HF actually executes

`Qwen3ForCausalLM.forward` → `Qwen3Model.forward` (`transformers/models/qwen3/modeling_qwen3.py`), with
`use_cache=False`:

| # | Operation | Module / helper | Output shape (Qwen3-0.6B: H=1024, V=151936, 28 layers) |
|---|-----------|-----------------|-------------------------------------------------------|
| 1 | token embedding | `model.embed_tokens` (`nn.Embedding [V, H]`) | `[N, T, H]` |
| 2 | positions | `position_ids` given, else `arange(T)[None]` | `[N, T]` or `[1, T]` |
| 3 | attention mask | `masking_utils.create_causal_mask` (+ sliding-window variant if `layer_types` has one; not for Qwen3-0.6B) | eager: float additive `[N, 1, T, T]`; sdpa: bool `[N, 1, T, T]` if padded, `None` if not (sdpa then uses `is_causal`) |
| 4 | RoPE table, shared by every layer | `model.rotary_emb(hidden, position_ids)` | `(cos, sin)`, each `[N or 1, T, head_dim=128]` |
| 5 | 28 × decoder layer | `model.layers[i](hidden, attention_mask=masks[layer.attention_type], position_ids, position_embeddings)` | `[N, T, H] → [N, T, H]` |
| 6 | final RMSNorm | `model.norm` | `[N, T, H]` |
| 7 | LM head on kept columns | `lm_head(hidden[:, -k:])` (`logits_to_keep=k`; 0 keeps all) | `[N, K, V]` |

Inside one `Qwen3DecoderLayer` (pre-norm, two residual branches):

```
x                                   [N, T, 1024]
h = input_layernorm(x)              RMSNorm over 1024
q = q_norm(q_proj(h).view(N,T,16,128)).transpose → [N, 16, T, 128]   q_proj: 1024 → 2048 (16 heads × 128)
k = k_norm(k_proj(h).view(N,T, 8,128)).transpose → [N,  8, T, 128]   k_proj: 1024 → 1024 (8 KV heads, GQA 2:1)
v =        v_proj(h).view(N,T, 8,128) .transpose → [N,  8, T, 128]
q, k = apply_rotary_pos_emb(q, k, cos, sin)
a = attention(q, k, v, mask, scale=128^-0.5)  → [N, T, 16·128 = 2048]   (eager or sdpa; KV heads repeated ×2)
x = x + o_proj(a)                   o_proj: 2048 → 1024
x = x + down_proj(silu(gate_proj(post_attention_layernorm(x))) * up_proj(...))   1024 → 3072 → 1024
```

`q_norm` / `k_norm` (per-head RMSNorm before RoPE) are what distinguish Qwen3 from Qwen2 attention.

Parameters per decoder layer: 15,730,944 (31.5 MB in bf16). Embedding: 155,582,464 (311 MB bf16), **tied** with
`lm_head` (`lm_head.weight is embed_tokens.weight`). Total: 596,049,920.

## How the explicit execution works

```python
decoder, lm_head = unwrap(model)                 # PeftModel → Qwen3ForCausalLM → (.model, .lm_head)
hidden = decoder.embed_tokens(input_ids)
masks = attention_masks(decoder, hidden, attention_mask, position_ids)   # HF's create_causal_mask
position_embeddings = decoder.rotary_emb(hidden, position_ids)
for layer in decoder.layers:                     # ← the loop we now own
    hidden = layer(hidden, attention_mask=masks[layer.attention_type],
                   position_ids=position_ids, position_embeddings=position_embeddings)
hidden = decoder.norm(hidden)
logits = lm_head(hidden[:, keep])
```

Important functions:

- `unwrap(model)`: returns `(Qwen3Model, lm_head)` for a plain or PEFT-wrapped model. LoRA needs no special handling:
  PEFT replaces `q_proj`/`v_proj`/`o_proj` inside the decoder layers with `lora.Linear` modules, so calling
  `layer(...)` runs base weight plus adapter. The tests show the adapter really runs: disabling it changes the
  explicit output.
- `attention_masks(...)`: calls `create_causal_mask` with exactly the arguments `Qwen3Model.forward` uses when there
  is no cache (`cache_position = arange(T)`, `past_key_values=None`). It returns a dict keyed by `layer.attention_type`
  so sliding-window configs also work.
- `explicit_forward(model, input_ids, attention_mask, position_ids, logits_to_keep)`: the loop above. It is a drop-in
  replacement for `model(..., use_cache=False, logits_to_keep=...).logits`.
- `explicit_logprobs(model, rollouts)`: `policy.compute_logprobs` routed through `explicit_forward`. It uses the
  same left-padding positions `cumsum(mask) - 1`, `logits_to_keep = R + 1` and `grpo.response_logprobs`.

**Padding.** The repo left-pads prompts. A left-padded row needs `position_ids = cumsum(mask) - 1`, the convention
`generate()` and `compute_logprobs` use, so the real tokens get positions 0, 1, …. Right padding works with the
default positions. One subtlety of HF 4.57 that the explicit path inherits: when `attention_mask is None`,
`create_causal_mask` infers *packed sequences* from `position_ids` (every restart at 0 starts a new sequence). Left
padding with clamped cumsum positions and no mask is therefore still isolated correctly. The tests don't rely on this;
they always pass the mask.

## Tests (`tests/test_qwen3_forward.py`)

Tiny random Qwen3 models (4 layers, H=32, 4 heads / 2 KV heads, head_dim 8, V=64, tied embeddings), because
`tests/conftest.py` forbids pretrained weights under pytest. Every test is parametrized over `eager` and `sdpa`, and
over with and without LoRA (`last2-attention` preset, `lora_B ~ N(0, 0.5)`).

1. `test_explicit_forward_equals_hf_logits` (fp32 and bf16): unpadded (no mask and an all-ones mask), left-padded,
   `logits_to_keep=4`, and an index-tensor `logits_to_keep`. Asserts `torch.equal`.
2. `test_padded_rows_match_unpadded_sequences` (fp32): each row of a left-padded batch (lengths 9, 5, 1, 7), and the
   same rows right-padded, at its real-token columns, against the same tokens run alone. Also asserts the logits at
   real-token columns are finite, and includes a control: without mask and positions, the padded row differs by
   0.34, more than 1000× the tolerance.
3. `test_selected_token_logprobs_equal_compute_logprobs` (fp32 and bf16): a hand-built rollout batch (left-padded
   prompts, responses with a post-EOS tail). Asserts `explicit_logprobs == policy.compute_logprobs` bitwise. In
   fp32 it also checks them against `log_softmax(full HF logits).gather(selected tokens)` to 1e-5, a tolerance rather
   than bitwise, because `lm_head` then runs on T columns instead of R + 1, a different GEMM shape.

### Tolerances and why

Errors are reported as max absolute error and relative error = `max|a − b| / max|b|`. An elementwise ratio is
meaningless for logits near 0.

| Comparison | Tolerance | Justification |
|------------|-----------|---------------|
| explicit vs HF, same input tensors | **0 (bitwise)** | Same modules, same order, same kernels, same arguments. Any difference is a bug (a wrong mask, positions, or a cache), never rounding. |
| selected-token log-probs vs `compute_logprobs` | **0 (bitwise)** | Follows from the above: identical logits through the identical `response_logprobs`. |
| padded row vs unpadded sequence, fp32 | **1e-5 abs** | Different shapes take different reduction orders and sdpa paths, so only rounding agreement is possible. Measured ≤ 2.7e-7 on the tiny models. A wrong mask or position moves logits by O(0.1). |
| padded vs unpadded, bf16 | not tested | bf16 logits are shape-dependent by up to ~1 logit (see below). A bf16 test would only measure bf16, not the code. |

### Results

Tiny models (pytest, CPU): all 20 parametrizations pass. Explicit vs HF and log-probs: max abs error 0. Padded vs
unpadded (fp32):

| attention | LoRA | max abs | rel |
|-----------|------|---------|-----|
| eager | no | 1.8e-7 | 3.3e-7 |
| eager | yes | 2.7e-7 | 7.4e-7 |
| sdpa | no | 1.2e-7 | 1.9e-7 |
| sdpa | yes | 2.1e-7 | 6.5e-7 |

Real Qwen3-0.6B on an RTX 3050 Ti, 4 chat-formatted prompts left-padded to T=23 (row lengths 18, 11, 18, 23).
Produced by `python -m tests.test_qwen3_forward`:

| dtype | attention | explicit vs HF logits (abs / rel) | log-probs (abs / rel) | padded vs unpadded logits (abs / rel) |
|-------|-----------|-----------------------------------|-----------------------|---------------------------------------|
| fp32 | eager | 0 / 0 (bitwise) | 0 / 0 | 9.1e-5 / 2.5e-6 |
| fp32 | sdpa | 0 / 0 (bitwise) | 0 / 0 | 7.9e-5 / 2.1e-6 |
| bf16 | eager | 0 / 0 (bitwise) | 0 / 0 | 1.19 / 3.4e-2 |
| bf16 | sdpa | 0 / 0 (bitwise) | 0 / 0 | 0.75 / 2.0e-2 |

The padded-vs-unpadded column is a property of HF itself, not of the explicit path, which is bitwise equal to HF. In
fp32 it is rounding: logits reach ~36, so 8e-5 abs is about 2e-6 relative. In bf16 it is the known shape-dependence
of bf16 kernels (8 mantissa bits, ulp 0.25 at |logit| ≈ 32). That is why padding equivalence is asserted in fp32 only.
For each dtype/attention pair, the reported rel is the one from the row with the largest abs error.

## Why this is the foundation for streaming weights

Layer-to-layer (L2L) execution keeps only some decoder layers' weights on the GPU and moves each layer in just before
it runs. That requires owning the loop over layers, which `model(...)` hides inside `Qwen3Model.forward`. After this
step:

- **The loop is ours.** A later step can put "make layer *i* resident" before `layer(...)` and "release it" after,
  without touching HF code. Nothing else in the forward is per-layer.
- **The state carried between layers is small and explicit**: `hidden [N, T, 1024]`, the mask (`None` or
  `[N, 1, T, T]`), `(cos, sin)` `[N, T, 128]`, and `position_ids`. It is built once before the loop and is the only
  thing a scheduler must keep on the GPU across layers.
- **Per-layer weights are uniform**: 15.7 M parameters (31.5 MB bf16) per layer, 28 layers. The always-needed pieces
  are `embed_tokens` and `lm_head`, which are **one tied tensor** (155.6 M parameters, 311 MB bf16) used at both ends
  of the forward. It must be treated as a single resident (or twice-fetched) weight. The final `norm` is negligible.
- **LoRA stays inside the layer modules**, so streaming a layer streams its adapters with it, and the trainable fp32
  LoRA parameters (a few layers only) can stay resident.
- **Correctness is pinned bitwise.** Any later change to the loop (prefetching, offloading, recomputation) can be
  checked against HF and `compute_logprobs` with tolerance 0 using these tests.

Not done in this step: offloading, scheduling, activation checkpointing, CUDA prefetching, and backward through the
explicit loop. The explicit forward only calls modules, so autograd is not cut, but gradients are not tested here and
it is not wired into training.
