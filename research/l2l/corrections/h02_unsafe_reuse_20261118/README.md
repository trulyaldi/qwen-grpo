# Correction: H02.fault_unsafe_reuse (2026-11-18)

The original FAIL records (assignment runs, `evidence/`, `research/l2l/final_report.md`, the archive under
`qwen-grpo-archive/2026-11-17_pre-cleanup`) are left exactly as recorded. This folder adds a CPU reproduction of
that FAIL on the unfixed code and the results after the correction.

## What failed

`gradient_equivalence` under `l2l_async` with `unsafe_reuse=2`: 80 injected configurations, made of 2 shapes ×
2 attention implementations × 2 dtypes × 5 LoRA placements × 2 buffer counts.

`before_fix/` was produced by commit `e8440e4` with a clean tree on CPU (`async_cpu.yaml`). It shows the following:
- status FAIL, execution FAIL, and 1,360 failed comparisons;
- 48 configurations were refused by backward.py's recomputation checksum (`layer 2: the recomputed output differs
  from the forward's`). These were recorded as errors, so the `lab_fault` judge reported the fault as "NOT detected";
- 32 configurations ran to completion with wrong logits, losses and gradients, and nothing raised at runtime. Only
  the offline comparison against a resident model caught them. These are `final` on both shapes, plus `last2` and
  `middle` on the 6-layer shape.

## Root cause

1. `PrefetchMixin` enforced R1 and R3 only through the slot-selection policy (`_victim`). The copy itself (`_copy`)
   trusted whatever slot `_target` returned. `UnsafeReuse._target` returns the slot that layer 2 is reading and
   drops its `free` event. Layer 3's prefetch then overwrote layer 2's weights before layer 2 ran, and the compute
   stream waited for that copy (R2), so layer 2 computed on layer 3's weights. The diagnostic checksum (`verify=True`)
   would have caught it, but it is off in production and in the fault run.
2. The only runtime guard left was the checksum that backward.py takes of the recomputation. It compares the
   backward recomputation with the forward output, so it can only fire for layers that are recomputed: layers at or
   above the lowest layer that has a LoRA adapter (`_graph_floor`).
   - With `final` placement (layer n−1 only), layer 2 is below the floor. It has no autograd node and is never
     recomputed, so its corrupted forward output flows into the logits, the loss and every gradient without a check.
   - With `last2` and `middle` on the 6-layer model, the floor is 4 or 3, which is above layer 2, so the same applies.
   - Where layer 2 is recomputed, backward's prefetch of layer 1 overwrote it again. The forward had used layer 3's
     weights and the recomputation used layer 1's, so the checksum differed and refused.
3. The harness put that refusal in `errors`, which made the execution FAIL. The `lab_fault` judge requires status
   FAIL and execution PASS, so a correct refusal was scored as "not detected".

## Correction

- `prefetch.BufferReuseError(RuntimeError)`.
- `PrefetchMixin._claim`, called by every `_copy` (prefetch and on-demand), plus a check of the prefetch window in
  `_prefetch`. It refuses a copy into the slot bound to the running layer (R3), into a slot of the prefetch window
  (R3), or on CUDA into a used slot that has no readers' `free` event (R1). The check runs always, independently of
  `verify`. It is host-side bookkeeping with no synchronization. It raises before anything is enqueued, so the
  buffer and the slot state are untouched.
- `_acquire` releases its binding if the prefetch is refused. `bound()` has not entered its `try` block yet, so its
  `finally` would not run.
- Harness (`lab.record_refusal`, `lab.check_fault_detected`, used by `forward_lab` and `backward_lab`):
  - A `BufferReuseError` under `unsafe_reuse=<i>` is a **detection**. It goes in `result["detections"]` and makes
    the status FAIL without making the execution FAIL.
  - Any other exception, including the recomputation-checksum refusal, stays an error.
  - A `BufferReuseError` in a normal run is an error.
  - Detections recorded in a non-fault result make the execution FAIL.
  - A configuration that completes with every comparison within its threshold is an error: either "undetected
    corruption", or "the fault was never injected (no power)" when `UnsafeReuse.injected == 0`.
- Tolerances, the GRPO loss, the LoRA config, hyperparameters, prefetch depth and order, and memory layout are
  unchanged. The `lab_fault` judge and the `WAIVED` entry in `scripts/assignment.py` are unchanged.

## After the correction (`after_fix/`, CPU, `async_cpu.yaml`)

The tree was dirty at commit `e8440e4` because these runs were made before the correction was committed.

| run | status | execution | detail |
|---|---|---|---|
| `ok`: H02, no fault | PASS | PASS | 5,296 comparisons, max_abs 0 |
| `fault`: `unsafe_reuse=2` | FAIL | PASS | 80/80 configurations refused by `BufferReuseError`, no errors |
| `fault0`, `fault1`: `unsafe_reuse=0`, `unsafe_reuse=1` | FAIL | PASS | 80/80 refused each |
| `fault3`: `unsafe_reuse=3` | FAIL | FAIL | 48 refused. In 32 configurations the fault never fires (layer 3 is the last layer of the 4-layer shape, so its window is already resident). These are reported as "never injected (no power)", not as detected |
| `h05_async`: H05 `update_equivalence` | PASS | PASS | 6,528 comparisons |
| `h05_sync`: H05 `update_equivalence` (`store_cpu.yaml`) | PASS | PASS | 6,528 comparisons |

Re-deriving the `before_fix` result with the corrected `lab.py` still gives FAIL/FAIL, the same as recorded. Old
results have no `detections` key, so their derivation is unchanged.

The GPU entries (`H02.fault_unsafe_reuse` on CUDA) still have to be rerun on a GPU host. This host cannot run CUDA
kernels.
