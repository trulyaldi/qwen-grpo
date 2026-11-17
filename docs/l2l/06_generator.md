# Ticket 06: streamed autoregressive generation

## Code

- `qwen_grpo/execution/generator.py`: `generate`, `KVCache` and `StreamedGenerator`.
- `policy.generate_rollouts(..., generator=None)`: with a generator, `model.generate()` is never called.
- `qwen_grpo/gen_lab.py`: lab case `generation_equivalence`.
- Tests: `tests/test_generator.py`, plus two cases in `tests/test_lab.py`.

**Not wired into the trainer.** Training still refuses `l2l_sync` and `kv_cache: cpu` (`config.unimplemented`). The
generator is selected by passing it to `generate_rollouts`.

## Design

`generate` reproduces `GenerationMixin._sample` (transformers 4.57) for the settings the rollouts use.

**Prompt expansion and positions**
- Prompts are expanded with `repeat_interleave` (`num_return_sequences`).
- Position ids are `cumsum(mask) − 1`, with padding positions set to 1, sliced to the new columns. This is
  `prepare_inputs_for_generation`. It handles left padding.
- `cache_position` is `0..P−1` for the prefill, then `P + step` for each decode step.

**Masks.** HF's `create_causal_mask` builds them. The cache reports the same sizes as `DynamicCache`, so the masks are
identical in eager and SDPA.

**Layers.** HF's own embedding, rotary embedding, decoder layers (so GQA and the current LoRA adapters run inside
them), final norm and head are called through a layer executor: plain, or L2L (`StreamingLayerExecutor`). Under L2L
every layer's weights are streamed in for the prefill and for every decode step.

**Sampling semantics**
- Only the last column goes through the head (`logits_to_keep=1`), and its logits are copied to fp32.
- The next token is `multinomial(softmax(logits))`: temperature 1, no top-k or top-p. Greedy uses `argmax`.
- A finished row receives the pad id; a row finishes on any EOS id.
- The loop stops when every row has finished, or after `max_new_tokens`.

**KV cache residency (`KVCache`).** This is managed separately from weight residency.
- Each layer gets a preallocated key and value buffer, `[N, kv_heads, P + max_new, head_dim]`, on `kv_device`. Its
  size is known up front, and the lab checks it against the analytic formula.
- An update writes the new columns into the buffer. It hands attention a contiguous copy of the filled prefix on the
  compute device, just as `DynamicCache`'s `torch.cat` would.
- So the cache can sit on the CPU while layers run on the GPU (FlexGen-style). Each layer's prefix is copied in only
  for its own attention.

**Teacher forcing.** `forced` tokens replace sampled ones, and `keep_logits` keeps every step's fp32 logits. This
lets distributions of different paths be compared along the same prefixes.

**Fail-closed.** Training mode, sliding-window layers and KV-cache overflow each raise.

**No resident fallback.** The test makes `model.generate` fail on both models, then runs the L2L rollouts
successfully. It also confirms that the default path really does call the guarded `generate`.

## Tolerances (`tolerances.yaml`, `generation_equivalence`)

**Cached paths vs HF `generate`: 0.** This covers greedy sequences, step logits, and probabilities along sampled
trajectories. The modules, shapes, kernels and sampling ops are the same, and L2L moves the same bytes.

**Cached decode vs the full-prefix forward: a derived rounding bound.** The two compute the same function with
different query shapes.

| quantity | fp32 | bf16 | derivation |
|---|---|---|---|
| `prefix_logits` | rtol 1e-5 | rtol 2⁻⁶ | bf16: 4 ulps at the largest logit |
| `prefix_probs` | rtol 4e-4 | rtol 0.65 | from `|Δp_i| ≤ p_i(e^{2δ} − 1)` with `|logit| ≤ 16` (checked) |

The bf16 probability bound is loose by construction. The bitwise cached comparisons carry the power. Greedy
agreement against the full-prefix forward is required wherever the top-2 margin exceeds twice the logit bound;
near-ties are counted.

**Disclosure.** A scratch pilot of the case ran before the thresholds and the H04 criterion were recorded:
- every cached comparison was 0;
- same-seed HF and streamed samples were identical;
- the full-prefix differences were at most 9e-7 (fp32) and 2e-3 (bf16) in logits.

## Experiment (`generation_equivalence`)

**Grid**
- Both shapes × eager/SDPA × fp32/bf16, with LoRA on the first and last layers.
- Left-padded prompts of lengths {9, 5, 9, 2} and {17, 3, 11, 1}.
- G = 2, 8 new tokens.
- The KV cache on the compute device and on the CPU.
- An EOS-boosted model: rows ending in EOS with a padded tail are required.
- A real-vocabulary (151,936) bf16 model with non-thinking Qwen3 chat prompts from `sentiment.build_prompts` and
  Qwen3's real EOS and pad ids.

**Measurements**
- same-seed sample identity;
- near-tie counts;
- weight loads per token (one per layer per step);
- KV-cache bytes vs analytic;
- seconds per token, resident vs streamed;
- the CUDA allocator peak, on a GPU.

**Faults:** `stale_position` (decode never advances the position id) and `no_padding_mask` (padding is attended).

## Evidence

Commit `04e56b9`, tracked tree clean, CPU only. Commands: `research/l2l/experiments/ticket06_commands.log`.

| run | result |
|---|---|
| `t06_generation_equivalence_cpu` (H04) | **PASS, H04 SUPPORTED.** 950 comparisons. Every cached and sampled comparison is exactly 0. The full-prefix differences use at most 42 % (logits) and 1.4 % (probabilities) of their bounds |
| `t06_generation_equivalence_stale_position` (fault) | **FAIL as expected,** execution PASS: 877 of 1,054 comparisons fail |
| `t06_generation_equivalence_no_padding_mask` (fault) | **FAIL as expected,** execution PASS: 1,003 of 1,075 comparisons fail |
| CPU suite (CUDA hidden), `ticket06_tests.log` | 570 passed, 99 skipped (CUDA), 23 failed, 1 collection error. The failing set is identical to before Ticket 04, on this host |

Measured:
- **Full-prefix differences:** at most 5.7e-6 (fp32) and 2.0e-3 (bf16) in logits, and 3.2e-7 and 4.0e-5 in
  probabilities.
- **Same-seed samples:** the streamed generator draws the same tokens as HF `generate` in every model.
- **Near-ties:** 70 bf16 positions were excluded from the full-prefix greedy check as near-ties.
- **EOS model:** 2 rows per prompt set end in EOS with a padded tail.
- **CPU time per token** (tiny models; wall clock, one run, diagnostic only): 15 ms resident vs 18 ms streamed for
  the 6-layer fp32 model, and 17 vs 16 ms for the real-vocabulary model.

## Not established here

- **CUDA.** Not run on this machine, by request; the commands are in the hand-off.
- **Early stopping inside the lab grid.** The lab requires padded EOS tails. Stopping once every row has finished
  is checked against HF `generate` by `test_rows_that_emit_eos_get_padding_and_the_loop_stops_when_all_have_finished`.
- Real pretrained weights.
- Trainer wiring.
- Generation throughput optimizations: batching across layers and overlapped transfers (Ticket 08).
