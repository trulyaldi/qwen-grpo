# Ticket 11: memory and performance validation

## Code

- `qwen_grpo/execution/profiler.py`: `Profiler` (sampler thread, phase windows, memory guard, evidence grade) and
  `h2d_bandwidth`.
- `scripts/profile_run.py`: every phase of the workflow on a real model, under the profiler and the guard, compared
  with the budget calculator's predictions.
- `qwen_grpo/residency_lab.py`: lab case `depth_residency` (H07).
- Config: `configs/experiments/qwen3_4b_l2l.yaml` (H11).
- Tests: `tests/test_profiler.py`.
- `scripts/laptop_checks.sh` step 4c runs everything above.

## Predictions (before measuring)

The expected use is computed by the Ticket 00 budget calculator (`qwen_grpo/budget/calculator.py`, `scripts/budget.py`),
from the checkpoint's own tensor shapes, before any full-scale run. It covers frozen weights, layer and prefetch buffers,
activation checkpoints, logits, KV cache, LoRA, optimizer state, CUDA context, the DFlash drafter, merge buffers, CPU RAM,
pinned memory and transfer time. The saved predictions are `results/budget/predictions/qwen3-0.6b_4gb.json` and
`qwen3-4b_8gb.json`, documented in `docs/l2l/00_budget.md`.

Two points from the 4B prediction:
- **GPU fits:** about 2.65 GB peak with every layer streamed.
- **CPU RAM is predicted to FAIL:** about 15.8 GB peak.

`profile_run.py --prediction` maps each measured phase to its predicted one and records them side by side:
- allocated vs predicted allocated;
- reserved vs predicted reserved;
- device-wide and per-process vs predicted device (CUDA context included);
- CPU RSS vs predicted CPU.

The prediction's batch and token settings can differ from the run's; both configs are kept in the result.

## Instrumentation

**What the sampler records**, every 0.25 s for the whole run:
- raw `nvidia-smi` device and per-process output (kept verbatim);
- `cudaMemGetInfo`;
- allocator allocated and reserved;
- CPU VmRSS and RssAnon;
- the current phase.

**Phase windows:** setup, rollout, old log-probs, differentiable forward, backward (with recomputation), optimizer, merge,
evaluation. Each window records wall time, allocator peaks, and the maxima of the samples taken inside it.

**Transfers:** per phase, the weight bytes the executor actually copied, and the transfer time the *measured* pinned
H2D bandwidth predicts. Comparing that predicted time with the phase's wall time shows whether streaming bounds the
phase, rather than assuming it.

**Memory guard.** When process-level GPU memory exceeds the budget, the guard records the violation and interrupts the
run. It uses per-process `nvidia-smi` when that is numeric, and otherwise the device-wide `cudaMemGetInfo`, which is
larger. The PyTorch allocator's limit is never treated as the budget.

**Evidence grade.** "assignment" requires a numeric per-process `nvidia-smi` reading in every sample and a host that is
not WSL2; anything less is "diagnostic". A run is reported PASS only with no violation and assignment-grade evidence;
otherwise it is FAIL (the guard fired) or INCOMPLETE_EVIDENCE.

## Experiments (`scripts/laptop_checks.sh`, step 4c)

1. **`depth_residency` (H07):** tiny models of one layer shape at depths 2–32; resident vs synchronous L2L vs async L2L;
   no-grad forward and training peaks; analytic checkpoint growth.
2. **Qwen3-0.6B profiles,** budget 4 GiB:
   - synchronous vs asynchronous streaming;
   - micro-batches of 5 (a different micro-batch configuration);
   - KV cache in CPU RAM (a different residency policy).

   `resident_layers > 0` is not implemented, so the safe residency options are buffers, prefetch depth, KV placement
   and micro-batching.
3. **Qwen3-4B,** async, budget 8 GiB, one step (H11).

## Expected limits of the evidence

- **On WSL2, nothing can PASS a budget.** `nvidia-smi` reports per-process memory as `[N/A]`, so every run on the laptop
  under WSL2 is at most diagnostic, and the budget verdicts are INCOMPLETE_EVIDENCE by construction. H11 needs a native
  Linux host for SUPPORTED.
- **Qwen3-4B may fail on the CPU first,** as predicted (pinned store about 7.5 GB plus the CPU model copy during
  loading). That outcome is recorded, not hidden.

## Evidence

Pending (laptop): `t11_depth_residency`, `t11_profile_qwen3_0.6b_{sync,async,async_mb5,async_kvcpu}` and
`t11_profile_qwen3_4b_async`.
