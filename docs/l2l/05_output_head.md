# Ticket 05: exact GRPO log-probabilities with a memory-efficient output head

## Code

- `qwen_grpo/execution/output_head.py`: the chunked head (`selected_logprobs`, `_SelectedLogprobs`), the model-level
  `streamed_logprobs` (any layer executor, full or chunked head), and `output_head` (head and tie checks).
- `qwen3_forward.explicit_hidden`: the final-norm hidden states. `explicit_forward(..., logits_to_keep=None)` returns
  them. The logits path is unchanged.
- `training.prepare` and `training.grpo_update` take an optional `logprobs` function. The default is
  `compute_logprobs`, looked up at call time. The GRPO arithmetic (`grpo.py`) and the `INIT_TOL` check are unchanged.
- `qwen_grpo/head_lab.py`: lab case `logprob_equivalence`.
- Tests: `tests/test_output_head.py`, plus two cases in `tests/test_lab.py`.

**Not wired into the trainer.** `config.unimplemented` still refuses `execution.vocab_chunk_size` and `l2l_sync`. A
run selects the head through the `logprobs` argument (lab and tests), not through a config yet.

## Design

The reference is `policy.compute_logprobs`, i.e. `grpo.response_logprobs` on the HF logits. It scores token `t` as
`fp32(z)[t] − logsumexp(fp32(z))`, where `z = lm_head(h)` is in the weight dtype.

**The whole head is normalized over.** The normalization runs over all `V = weight.shape[0]` head rows: 151,936 for
Qwen3. That is more than the tokenizer's 151,669 tokens, and the extra rows take part in HF's softmax, so they take
part here too.

**`_SelectedLogprobs` never materializes `[N, R, V]`.**

Forward, per chunk of `C` vocabulary rows:
1. Compute `z_c = linear(h, W_c)` in the weight dtype. This gives each logit the same bf16 rounding as `lm_head`.
2. Upcast `z_c` as `response_logprobs` does.
3. Update a running maximum `m` and a rescaled sum `s`. This is an online log-sum-exp: every exponent is ≤ 0, so
   nothing overflows (tested with logits around 1e3).
4. Pick the target's logit from the chunk that holds it.

The output is `(z_t − m) − log s`. It never uses the raw logit alone: the fault-injection run shows that doing so is
caught.

The forward saves only `h`, the targets and `lse`.

Backward recomputes `z_c` chunk by chunk:
- `dL/dz_c = g·(onehot − exp(z_c − lse))`, in fp32;
- this is cast to the weight dtype, as the reference's upcast backward does;
- `dL/dh = Σ_c dL/dz_c · W_c` is accumulated in fp32 and rounded once.

**Checks.**
- An fp64 `gradcheck` passes.
- The head weight (tied to the embedding, frozen) must not require grad, so a missing weight gradient cannot pass
  silently.
- `output_head` checks that `lm_head` is a bias-free plain `Linear` and is the embedding `Parameter` when
  `tie_word_embeddings` is set.

**Integration.** `streamed_logprobs` scores hidden columns `P−1 … T−2`. These are the same `R` columns as
`compute_logprobs`' `logits_to_keep=R+1` minus its last column. With `vocab_chunk_size=None` (the full head),
`streamed_logprobs` equals `compute_logprobs` bitwise; the test and the lab's reference-integrity check confirm it.

`prepare` and `grpo_update` must be given the same function. Otherwise the unchanged iteration-0 check
(`INIT_TOL = 1e-6`) raises, and a test confirms this. With the chunked head, `init_diff` is exactly 0.

## Tolerances (`tolerances.yaml`, `logprob_equivalence`)

L2L itself is exact (Ticket 04). The chunked head changes only the fp32 summation order of the log-sum-exp and of
`dL/dh`, so it cannot be bitwise. The thresholds follow from that:

| quantity | fp32 | bf16 | derivation |
|---|---|---|---|
| logp | atol 1e-4 | atol 1e-4 | Both dtypes compute the log-softmax in fp32. The fp32 reference's own error against fp64 is about 2e-5 at V = 151,936. 1e-4 is 1/2000 of the clip range ε |
| loss | atol 4e-4 | atol 4e-4 | `|Δratio| ≤ e^{2·1e-4} − 1`, times `|A| ≤ √2`, times `(1 + ε)` |
| LoRA, head-input and layer-output gradients | rtol 1e-4 | rtol 2⁻⁶ | fp32: summation-order level. bf16: the cast of `dL/dz` and the single rounding of `dL/dh` can each flip one ulp (≤ 2⁻⁷); allowing 2 ulps gives 2⁻⁶ |

**Disclosure.** Before these thresholds were recorded, a scratch pilot of this case was run with non-binding
thresholds. Measured worst cases:

| quantity | fp32 | bf16 |
|---|---|---|
| logp | 1.9e-6 | 5.7e-6 |
| loss | 2.4e-7 | 3.9e-7 |
| gradients (fraction of max) | 1.0e-5 | 5.7e-3 |
| L2L + full head | 0 | 0 |

The thresholds come from the derivations above, not from these values.

## Experiment (`logprob_equivalence`)

**Grid**
- V ∈ {64, 1000, 151,936 (Qwen3)} × fp32/bf16 × response length R ∈ {1, 4, 16} × whole batch or micro-batches of 2.
- Two chunk sizes per V; one of them leaves a remainder chunk.
- Tiny Qwen3, with LoRA on every projection of the first and last layers.

**Sides:** `full` (the reference), `chunked<C>`, `streamed_full` (L2L + full head), `streamed<C>` (L2L + chunked
head).

**Optimizer iterations**
- The reference runs 4 AdamW iterations through the real `prepare` / `grpo_update`, with lr 5e-2.
- At least one later iteration must clip a token (power check).
- At each reference snapshot, every side runs `grpo_update(iteration=k)` against its own old log-probs. A zero-LR
  probe optimizer captures the gradients.
- Each side's own 4-step training run is recorded as a measurement, not a threshold. AdamW's normalization amplifies
  tiny gradient differences in near-zero gradient elements, so trajectories drift.

**Memory, on CPU**
- Largest tensor allocated during head forward + backward, found with a dispatch mode.
- Non-parameter bytes saved for backward.
- The CUDA allocator peak is recorded only on a GPU.
- Both are measured for the tiny models at V = 151,936 and for the real Qwen3-0.6B head shape: H 1024, V 151,936,
  4 × 128 tokens.

**Time:** interleaved trials of head forward + backward, full vs chunked, with a bootstrap CI of the ratio.

**Fault injection:** `raw_logit` (unnormalized selected logit) applied to every candidate side.

## Evidence

Recorded after the code commit; see `research/l2l/experiments/t05_*` and the table appended below.

## Not established here

- **GPU peak memory and GPU time.** The local GPU is unusable with this PyTorch build. CPU allocation figures are
  allocator-free tensor accounting.
- **Real pretrained weights.** Only the real head *shape* with random weights was run.
- **Trainer and config wiring** (`vocab_chunk_size`, `l2l_sync`).
- **Generation.** The head here is teacher-forced scoring only (Ticket 06).
