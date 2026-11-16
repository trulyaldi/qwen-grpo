# Step 0 — theoretical memory and PCIe budget

Predictions only. Nothing here is measured. The calculator is `qwen_grpo/budget/`, the configs are `configs/budget/*.yaml`,
and `python scripts/budget.py --config configs/budget/qwen3-4b.yaml` writes `results/budget/predictions/<run>.{json,md}`.
It never overwrites earlier files, which stay available for comparison with measurements.

## Sources of numbers

| kind | what | source |
|---|---|---|
| exact | every parameter tensor's shape, per-layer bytes, tied duplicates, shard sizes | `config.json` + safetensors headers (local cache, else Hub header metadata pinned to the cached config revision); no weights loaded |
| exact | LoRA targets | the training code's own `qwen_grpo.model.target_modules` on a meta-device skeleton |
| exact | KV cache, checkpoints, logits, LoRA/AdamW state | shapes from the config: B, G, P, R, m |
| estimate | layer intermediates, generate() sampling buffers, autograd ordering in the log_softmax backward | formulas below |
| estimate | CUDA context (0.5 GiB), allocator overhead (10 %), CPU process baseline (2 GiB), PCIe 12 GB/s | YAML assumptions, to be replaced by measurements |
| unknown | DFlash draft KV / feature buffers; HF generate() internals | covered only by the 10 % safety margin |

`P = 100` is the longest tokenized prompt over all three splits of `data/sentiment.json` (mean 58). It is a data property,
not a runtime measurement.

## Memory equations

Notation: L layers, h hidden, I intermediate, n_q / n_kv heads of size d, V vocabulary, w bytes per weight (bf16: 2).
Also N = B·G sequences, m sequences per micro-batch, and T = P + R tokens.

- **Weights.** `bytes(layer i) = w · Σ_{tensors in layer i} numel`, computed per layer from the headers. The embedding
  `V·h·w` is counted once when `tie_word_embeddings`. Qwen3-0.6B stores a duplicate `lm_head.weight` (311 MB) on disk,
  which counts toward the loading copy but not toward live memory. Qwen3-4B stores none.
- **LoRA.** `n_lora = Σ_targets r · (in + out)`. fp32 parameters, gradients and AdamW `exp_avg`/`exp_avg_sq` cost
  `4·n_lora` bytes each, so the 4B total is 4 × 4.1 MiB.
- **KV cache (GQA).** `2 · N · T · L · n_kv · d · w`. Because n_kv = 8, not n_q = 32 (4B), this is 4× smaller than MHA.
- **Boundary checkpoints.** `(L − a + 1) · n · T · h · w`, where a is the first adapted layer. Layers below a hold no
  trainable parameter, so they run under no_grad and need no checkpoint and no backward sweep. With `last2-attention`
  only 3 boundaries are stored. The schedule changes n: layer-major keeps n = N (all micro-batches pass a layer per load),
  micro-batch-major keeps n = m.
- **Logits** (the repo's path, `grpo.response_logprobs`): bf16 `[m, K, V]` with K = R+1 (`logits_to_keep`) or T (full
  sequence), then an fp32 copy and an fp32 `log_softmax` of the R scored columns. That gives a forward peak of
  `m·K·V·w + 2·m·R·V·4` and a backward peak of `3·m·R·V·4 + m·K·V·w`. For 4B with m = 6, response-only is 36.5 MiB.
  Full-sequence logits would hold `6·104·151936·2` = 181 MiB for the bf16 tensor alone.
- **Recomputed layer (estimate).** Per token, `(4h + 4·n_q·d + 3·n_kv·d + 4I)·w`. This covers the norm inputs and outputs,
  q/k before and after the q/k-norm and RoPE, v, the attention output, and gate, up, act and act·up. It adds PEFT's
  fp32 LoRA-A inputs and assumes SDPA, so there is no `[T, T]` score matrix.
- **Pinned rounding.** `CachingHostAllocator` (torch 2.14.1, `CachingHostAllocator.h:340`) rounds every pinned block up
  to a power of two. Both cut-offs, `pinned_max_round_threshold` and `pinned_max_cached_size`, default to `SIZE_MAX`. For 4B, one 192.5 MiB layer occupies 256 MiB, so 36 layers take 9.0 GiB instead of 6.77 GiB.
- **Peak.** Each phase is a few moments of simultaneously live tensors, and the phase peak is the max over those moments.
  `allocated` ≈ `max_memory_allocated`, `reserved = allocated·1.1` ≈ `max_memory_reserved`, and
  `device = reserved + context` ≈ nvidia-smi.

## Results (configured baseline: 0 resident layers, 2 buffers, pinned CPU weights)

| | Qwen3-0.6B / 4 GiB | Qwen3-4B / 8 GiB |
|---|---|---|
| GPU peak (device), phase | 1.38 GiB, evaluation prefill | 2.47 GiB, evaluation prefill |
| GPU peak, training | 1.00 GiB | 1.94 GiB |
| CPU peak (init / steady) | 4.28 / 2.88 GiB | 14.71 / 11.00 GiB |
| verdict | **PASS**\* | **FAIL** (CPU) |

\* The 0.6B PASS is measured against total RAM (WSL2 MemTotal, 6.7 GiB). Only about 3.1 GiB was free when that was
read, which is below the 4.28 GiB initialization peak (the whole-shard loading copy dominates it). It would fit after
freeing about 1.2 GiB, or with `loading_copy: tensor`. Whether the limit should be total or available RAM is still open.

**Placement.** Embedding (= tied LM head) and final norm are GPU-resident. They are used at both ends of every forward
and every decode step. Streaming them would add 0.78 GB (4B) per sweep, and keeping them costs the same 0.72 GiB once.

**Why 4B fails here, and what would fit.** The development host's WSL2 has 6.7 GiB of RAM in total. The streamed
weights need 9.0 GiB of pinned RAM. Initialization adds a shard-sized pageable copy of 3.7 GiB. On the GPU side,
residency is cheap: up to 22 decoder layers can stay resident with the GPU still PASS at 7.0 GiB device. That leaves
14 streamed layers at 3.5 GiB pinned, plus the 2 GiB baseline, plus the init copy, so CPU RAM is still over 6.7 GiB.
No residency count passes both limits on this host.

Two things to settle before Step 1:
- the assignment machine's CPU RAM, or a larger WSL `memory=` in `.wslconfig`;
- whether to load per tensor (`loading_copy: tensor`) and use one pinned slab instead of power-of-two blocks.

For Qwen3-0.6B, all 28 layers fit resident in 4 GiB. Streaming it is only for verifying L2L, not for memory.

## PCIe transfers

`T_transfer = bytes / B_eff` with B_eff = 12 GB/s, an unmeasured assumption. Each phase moves `bytes = sweeps ·
Σ_streamed layer bytes + activation traffic`. These are bandwidth-only service times. With copy/compute overlap the
critical path is `max(T_transfer, T_compute)`, and `T_transfer + T_compute` without it. Compute time is not estimated,
so every critical path is reported only as `≥ T_transfer`.

Qwen3-4B, configured baseline, per GRPO step (B = 5, G = 6, R = 4, K = 1), one sweep = 7.27 GB:

| stage | sweeps | GB | ≥ s | tokens per sweep |
|---|---|---|---|---|
| rollout prefill | 1 | 7.27 | 0.61 | 3000 |
| rollout decode | R − 1 = 3 | 21.80 | 1.82 | 30 |
| old log-probs forward | 1 | 7.27 | 0.61 | 3120 |
| training forward | 1 | 7.27 | 0.61 | 3120 |
| backward (layers 34–35 only) | 1 | 0.40 | 0.03 | 3120 |
| **step total** | | **44.0** | **≥ 3.67** | |
| evaluation, 512 examples | 16·4 + 64 | 930 | ≥ 77.5 | 32 / 8 |

## Predicted bottlenecks

- **Rollout decoding.** Every generated token re-streams all weights while computing on only N = 30 tokens. That is
  about 30 tokens per 7.3 GB, which is certainly transfer-bound. It costs half the step's traffic even at R = 4 and
  grows linearly in R. Training sweeps compute on 3,120 tokens per load, so they may be compute-bound.
- **Evaluation.** At batch 32 for generation and 8 for scoring, one validation pass moves 21× a training step. The
  evaluation batch sizes, not the model, set its cost.
- **Merge** is negligible: 6 adapted matrices, with a peak work set of 124 MiB for 4B.
- **Upper bounds a Step-1 design can cut.**
  - Layers 0–33 are frozen with no adapters, so their output is the same for the old-log-prob forward and all K
    training forwards. Caching the input of layer 34 once per rollout batch would save one 7.27 GB sweep per step.
  - With 2 buffers, layers 34–35 are still on the GPU when the forward ends, so the 0.40 GB backward reload is an
    upper bound.
- **Layer-major vs micro-batch-major.** Micro-batch-major would multiply the training sweeps by ⌈N/m⌉ = 5. Layer-major
  costs only N·T·h·w = 15 MiB per extra checkpoint at this size.

## What measurements must verify

1. `max_memory_allocated` / `max_memory_reserved` per phase against `gpu_allocated` / `gpu_reserved`, with raw
   nvidia-smi against `gpu_device`. This tests the 10 % overhead and the 0.5 GiB context assumptions.
2. Process RSS and pinned bytes against `cpu`, in particular the power-of-two rounding and the loading copy. Also check
   how much memory can actually be pinned under WSL2: NVIDIA lists limited pinned system memory as a known limitation of
   CUDA on WSL.
3. Effective H2D bandwidth from pinned memory under WSL2, which replaces the 12 GB/s assumption.
4. Measured rollout and training wall time against the `≥ T_transfer` bounds, to see whether each stage is
   transfer- or compute-bound.
5. Which moment is actually the peak in each phase (the memory profiler's snapshot).
