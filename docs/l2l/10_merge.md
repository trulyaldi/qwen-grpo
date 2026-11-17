# Ticket 10: streaming LoRA merge and streamed evaluation

## Code

- `qwen_grpo/export/merge.py`:
  - `stream_merge(base, adapter, out)`: the tensor-by-tensor merge.
  - `load_merged_l2l(path, device)`: the merged checkpoint as a plain HF model, run by the L2L engine.
- `scripts/merge.py`: merge a run's adapter, optionally with `--evaluate` (greedy validation predictions through L2L).
- `qwen_grpo/merge_lab.py`: lab case `merge_equivalence`. Tolerances: `merge_equivalence` in `tolerances.yaml`.
- Tests: `tests/test_merge.py`, plus one case in `tests/test_lab.py`. `tests/conftest.py` now lets tests load
  checkpoints they wrote to a local directory; hub names are still refused.

## The merge

`W_merged = round_to_base_dtype(fp32(W0) + (α/r) · fp32(B) @ fp32(A))`, computed **one tensor at a time on the CPU**:

**Reading.** Base tensors are read lazily from the base checkpoint's safetensors (single file or sharded index). Only
the current tensor and the small adapter are materialized. No model is built.

**Merging.**
- Adapted tensors are rounded **once**, from the exact fp32 sum. PEFT's `merge_and_unload` rounds the delta to the base
  dtype and then adds in that dtype, which is two roundings.
- Every element is checked to lie within half an ulp of the exact sum.
- The realized delta, the intended delta and the number of changed elements are recorded, so an adapter that didn't
  survive the merge is visible.

**Everything else is copied unchanged:** same name, dtype and bytes. Tied embedding / LM head relationships stay as the
base stores them; an adapter on `embed_tokens` or `lm_head` is refused.

**Output.**
- Shards of at most `shard_bytes` (plus one tensor), each released after writing, with
  `model.safetensors.index.json`.
- `config.json`, `generation_config.json` and the tokenizer files are copied, and `merge_report.json` is written.
- The whole directory is written to a temporary sibling and renamed into place. An existing output is never
  overwritten.

**Refused, rather than merged wrong:** rsLoRA, DoRA, `fan_in_fan_out`, bias, per-module rank or alpha patterns, adapter
modules without a matching base tensor, shape mismatches, and unused adapter modules.

**Memory.** At most one shard buffer, one tensor and the adapter at a time; never a duplicate full model, and never a
GPU. When the L2L backend is selected, checkpoint verification (`run.py verify`, which uses PEFT's whole-model merge)
refuses the config (Ticket 07), so the whole-model GPU merge path is never used under L2L.

## Streamed evaluation of the merged model

`load_merged_l2l` loads the merged checkpoint on the CPU as a plain `AutoModelForCausalLM`, with no PEFT. An
`L2LBackend` then:
- adopts its frozen decoder weights into the `WeightStore`;
- moves only the embedding / tied head, final norm and rotary buffers to the device;
- computes logits through the streaming executor, and runs greedy generation with `execution.generator`, never
  `model.generate`.

`scripts/merge.py --evaluate` runs the standard `evaluation.evaluate` path with that generator.

## Experiment (`merge_equivalence`)

**Variants**
- tiny random Qwen3 (real vocabulary) in fp32, and in bf16, with LoRA B ~ N(0, 0.5) so the adapter matters;
- with `--real`: Qwen3-0.6B in bf16 with `runs/qwen/checkpoints/step90`.

**Merge, in a spawned child process.** Its peak anonymous and total RSS are sampled every 10 ms, it must never
initialize CUDA, and the parent never blocks on a child that died.

**Evaluation.** Adapter-attached (PEFT), base (`disable_adapter`) and merged models all run through L2L on identical
validation prompts:
- full logits per prompt at real positions;
- batched greedy generation with left padding;
- peak GPU memory of the merged evaluation.

**Compared**
- **Merged tensor vs exact fp32 sum:** rtol is half an ulp of the dtype, fp32 2⁻²⁴ and bf16 2⁻⁸.
- **Merged vs adapter logits:** fp32 rtol 1e-4 (the existing `FP32_MERGE_RTOL`); bf16 rtol 2⁻⁴ (8 bf16 ulps).
- **Greedy tokens:** must match except at near-ties.

**Requirements**
- **Power:** the adapter's effect must be at least 100× (fp32) or 10× (bf16) the merge discrepancy.
- **Adapter survives:** every adapted tensor actually changed.
- **Loads independently:** the merged checkpoint loads with no LoRA tensors, the index holds exactly the base tensors,
  and the head stays tied.
- **Never fully resident on the GPU:** the merged evaluation's peak allocated GPU memory stays below the model's size.

## Pilot and disclosure (CPU, tiny models, before the evidence runs)

**fp32**
- merged = exact rounding everywhere;
- logits within 5.8e-7 absolute of the adapter's;
- power ratio about 1.6 × 10⁶;
- greedy tokens identical.

**bf16**
- Logits within 8.8e-3 absolute (about 1.8e-2 relative in a first standalone pilot); power ratio about 109; greedy
  tokens identical.
- **A threshold error was found and fixed.** The first draft of the bf16 tolerance used 2⁻⁹ as half a bf16 ulp. bf16
  has 7 explicit mantissa bits, so machine epsilon is 2⁻⁷ and half an ulp is 2⁻⁸. The pilot's weight comparisons
  exposed this; the merge's own element-by-element half-ulp check, which uses `torch.finfo`, had passed. The
  tolerance was corrected to 2⁻⁸ before any recorded run. The logit bound was restated in correct ulps.

## Evidence

Pending: `scripts/laptop_checks.sh` records `t10_merge_equivalence_tiny` (CPU) and `t10_merge_equivalence_qwen3_0.6b`
(GPU, real model).

## Not established here

- **Qwen3-4B.** The ticket's target model is not cached on the development host, and the laptop GPU (4 GB) cannot
  hold its non-streamed parts plus buffers comfortably. The code path is the same: shards, the store and streamed
  evaluation.
- **Adapters on embeddings or the LM head** (refused).
