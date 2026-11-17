# Ticket 07: complete resident-versus-L2L GRPO update

## Code

- `qwen_grpo/execution/backend.py`: the policy-execution interface, with `ResidentBackend`, `L2LBackend` and
  `make_backend(cfg, model, device)`.
- `qwen_grpo/trainer.py`: rollouts, old/current log-probs, in-training evaluation and label scores all go through the
  backend. The GRPO loss, the micro-batch accumulation and the AdamW step are the unchanged `training.prepare` /
  `training.grpo_update` (via their `logprobs` argument). There is no separate L2L GRPO implementation.
- `evaluation.generate` / `predict` / `evaluate` take an optional greedy `generator`. `label_scores` takes an
  optional `logprobs`. The defaults are the old calls.
- `config.unimplemented`: `l2l_sync` now trains, including `kv_cache: cpu`. Still refused: `resident_layers > 0`,
  `vocab_chunk_size` (H03 unresolved) and `l2l_async`.
- `scripts/run.py verify` / `validate` / `test` refuse a non-resident config, rather than evaluating the checkpoint
  resident in silence.
- `qwen_grpo/update_lab.py`: lab case `update_equivalence`.
- Tests: `tests/test_backend.py`, plus two cases in `tests/test_lab.py`.

## Interface

| need | `ResidentBackend` | `L2LBackend` |
|---|---|---|
| `generate_rollouts` | `model.generate` (unchanged) | `execution.generator` (sampling); never `model.generate` |
| `logprobs(model, rollouts)` | `policy.compute_logprobs` | `output_head.streamed_logprobs` through `CheckpointedLayerExecutor`, full head |
| evaluation (`greedy`) | `model.generate` | `StreamedGenerator(do_sample=False)` |
| state (`trainable`, `frozen_digests`, `adapter_state`) | the model's own `Parameter`s | the same objects; frozen decoder weights are views of CPU slabs |

**How the L2L model is loaded.** The trainer loads it on the CPU. The backend builds the `WeightStore` and moves only
the non-streamed tensors (embedding = tied head, final norm, rotary, LoRA) to the device. Checkpointing (PEFT
save/load) and `disable_adapter` work unchanged.

## Evidence that the integration is the real pipeline

`tests/test_backend.py` runs the real trainer on CPU, using the `test_trainer` sandbox: a tiny Qwen3, the real
tokenizer and data, 4 optimizer steps. It runs once with `resident` and once with `l2l_sync` (pageable store, CPU
buffers), and checks the following bitwise:
- rollout hashes, losses, clipping statistics, gradient and update norms;
- in-training evaluations;
- final adapters and AdamW state.

It does this with the default settings and with micro-batches of 4 over 2 GRPO iterations. During the L2L run,
`GenerationMixin.generate` is patched to fail the test. An interrupted L2L run resumes bitwise.

## Tolerance

0 for every quantity (`tolerances.yaml`, `update_equivalence`). Both sides run the same `grpo_update`. The L2L
log-prob function gives bitwise-equal log-probs and gradients (Tickets 03–05), so AdamW applies the same kernels to
the same tensors.

**Disclosure.** A scratch pilot of the case ran before the threshold and the H05 criterion were recorded. All 6,528
comparisons were 0, and the advantage fault failed 5,216.

## Experiment (`update_equivalence`)

**Fixture.** `fixtures.build(shape)`, the Ticket 00 batch built from code and seeds: input ids, response ids,
completion masks, rewards, advantages and old log-probs. Its integer tensors and rewards are checked against the
saved Ticket 00 fixture (reference integrity). For the bf16-base model and on a GPU, the advantages and old log-probs
are recomputed by the resident `prepare()`, as training would.

**Grid:** both shapes × {fp32, bf16 base with fp32 adapters} × micro-batch size {all, 1, 2, 4} × AdamW state {fresh,
seeded warm state at step 3} × 3 iterations, with LR 5e-2. Tokens clip after iteration 0; this is checked as a power
requirement.

**Compared per iteration:** log-probs, loss, `ratio_min`/`ratio_max`/`clip_frac`/`approx_kl`/`grad_norm`/
`update_norm`, every gradient, every update, and every AdamW `exp_avg`/`exp_avg_sq`/`step`.

**Execution requirements**
- frozen hashes are unchanged and equal on both sides;
- old log-probs are unchanged across iterations;
- gradients are finite;
- iteration-0 `init_diff` is 0;
- `Parameter` identities are kept.

**Fault:** `advantage=<i>` sets the candidate's advantage `i` to `1 − A_i`.

## Evidence

Commit `3cbd70c`, tracked tree clean, CPU. Commands: `research/l2l/experiments/ticket07_commands.log`.

| run | result |
|---|---|
| `t07_update_equivalence_cpu` (H05) | **PASS, H05 SUPPORTED.** 6,528 comparisons, all exactly 0. Reference integrity clean, and frozen hashes unchanged on both sides |
| `t07_update_equivalence_advantage` (fault: advantage 2 corrupted) | **FAIL as expected,** execution PASS: 5,216 of 6,528 comparisons fail |
| CPU suite (CUDA hidden), `ticket07_tests.log` | 578 passed, 99 skipped (CUDA), 23 failed, 1 collection error. The failing set is identical to before Ticket 04, on this host |

In the fault run, the iteration-0 log-probs still match, as they should: no update has happened yet. From the loss
onward every quantity fails.

## Change after `t07_update_equivalence_cuda` (recorded FAIL)

**What failed.** Your GPU run `t07_update_equivalence_cuda` (commit `2c2c585`, RTX 3050 Ti) ended with execution
FAIL. All 6,528 comparisons passed, with every difference exactly 0, resident vs L2L. The failure was the lab's own
requirement that iteration 0 have `init_diff == 0`, which was violated 16 times. Every violation was fp32 with
micro-batches of 1, 2 or 4, with `init_diff` = 4.8e-7, identical on both sides.

**Cause: a flaw in the case's design, not in L2L.**
- The fixture's old log-probs were computed on the whole batch, while the updates computed current log-probs in
  micro-batches.
- GPU kernels depend on batch shape, so the two differ in the last bits.
- On the CPU they were bitwise equal, so the CPU pilot could not show this.
- The trainer itself avoids this case: `prepare()` uses the update's micro-batches. The difference is also within the
  trainer's `INIT_TOL = 1e-6`.

**Fix.** For each micro-batch size, the case now computes the old log-probs with the resident
`prepare(..., micro_batch_size)`, as training does. Nothing else changed: the thresholds stay at 0, and the
`init_diff == 0` requirement stays. The failed result stays in the repository. The fixed case needs a new CUDA run
under a new name.

The fixed case on CPU, `t07_update_equivalence_cpu_v2` (commit `ce0b4d4`, clean tree): **PASS, H05 SUPPORTED**, 6,528
comparisons, all exactly 0.

The fixed case on CUDA, `t07_update_equivalence_cuda_v2` (commit `a8aab7f`, clean tree, RTX 3050 Ti Laptop GPU,
torch `2.14.1+cu126`): **PASS, H05 SUPPORTED.**
- 6,528 comparisons, all exactly 0. There were no errors, and the iteration-0 `init_diff` was 0 everywhere.
- All 32 configurations clip tokens after iteration 0.
- Gradient norms range from 2.6e-5 to 0.42.

## A real Qwen3-0.6B step (`scripts/l2l_real_step.py`)

Two runs at commit `3106101`, clean tree, RTX 3050 Ti Laptop GPU (4 GB):
- `real_step_20261117T063325Z.json`: one iteration.
- `real_step_20261117T063548Z.json`: two iterations.

**Setup**
- Real Qwen3-0.6B weights, bf16, SDPA, head_dim 128, 28 layers.
- LoRA from `configs/qwen.yaml`: rank 32 on the q/v/o projections of layers 26–27, with B set to a seeded
  N(0, 1e-2) so A's gradients are nonzero.
- 5 prompts × 6 completions from the training split, the real reward, pinned store, 2 weight buffers.
- One script run does the L2L side, then the resident side on the same GPU.

**Correctness, L2L vs resident, all exactly equal**
- The sampled rollouts are identical with the same seed. L2L generation draws exactly the tokens HF `generate` draws.
- Old log-probs, current log-probs, loss and `clip_frac` are identical.
- All 12 LoRA gradient tensors and all 12 updates are identical, in both iterations.
- Iteration 1 is a non-trivial check: ratios span [0.077, 2.65] and 23.6 % of tokens clip.
- So the recomputation checksum holds at real size: SDPA did not pick a different kernel for the
  grad-enabled recomputation.

**Resources** (one run each, wall clock, diagnostic grade, synchronous transfers, before Ticket 08)

| | L2L | resident |
|---|---|---|
| rollout generation, 30 rows (4 steps: every row ended with EOS after the 2-token label) | 2.7 s / 4.2 s (0.68 / 1.06 s per step) | 0.51 s / 0.53 s |
| effective weight streaming | 0.84 GB per pass (840 MiB), 1.3 / 0.83 GB/s | — |
| `prepare` (old log-probs) | 0.56 s | 0.35 s |
| one `grpo_update` | 0.93 s (iteration 1: 0.67 s) | 0.47 s |
| weight loads in the step | forward: 28 per pass; backward: 2 per iteration (layers 26–27 only) | — |
| peak allocated, generation / step | 799 / 604–609 MiB | 1,481 / 1,688–1,692 MiB |

**Reading the numbers**
- **Memory:** L2L uses about 2.1–2.8× less peak allocated memory than resident.
- **Generation is the bottleneck:** it is 5–8× slower than resident, because all 840 MiB of weights stream
  synchronously for every token. The effective bandwidth (0.8–1.3 GB/s) is far below PCIe. Each layer's copy waits
  for the GPU and does not overlap compute; that is Ticket 08.
- **Training is fine:** the training step costs about 1.4–2× resident.
- **The KV cache is oversized:** it is preallocated for `max_new = 32` (354 MiB) even though every row stopped after 4
  tokens.
- **Caveats:** the two runs differ by 1.6× in generation time, and there is one trial each, so the timings are
  indicative only.

## Full real training runs: resident vs L2L (`scripts/compare_runs.py`)

**Setup.** The real trainer (`scripts/run.py train`) ran Qwen3-0.6B in bf16 on the RTX 3050 Ti Laptop GPU (4 GB),
with configs that differ only in `execution` and `run.name`:
- `configs/experiments/qwen_resident.yaml` vs `qwen_l2l.yaml`: `configs/qwen.yaml` (100 steps, B = 5, G = 6, LoRA
  rank 32 on q/v/o of layers 26–27, subset evaluation every 10 steps, full validation);
- `qwen_resident_short.yaml` vs `qwen_l2l_short.yaml`: 10 steps, no full validation.

L2L used a pinned store, 2 weight buffers, checkpointed backward, the streamed generator for rollouts and evaluation,
and the full output head. All four runs ran at commit `5300146`, with a clean tree, on the RTX 3050 Ti Laptop GPU.
This is recorded in each run's `training.json` (`environment`). The committed run folders under `runs/qwen_*`
reproduce both comparisons. The comparison JSONs themselves predate the provenance field that `compare_runs.py` now
writes.

**These runs predate the head-shape fix (`bc5a05d`).** An earlier version of this section said the opposite, which
was wrong.

| comparison | result |
|---|---|
| `compare_qwen_resident_short_vs_qwen_l2l_short.json` | **IDENTICAL.** 10/10 step logs, 3/3 subset evaluations, 3/3 checkpoints (adapter and AdamW state, bitwise) |
| `compare_qwen_resident_vs_qwen_l2l.json` | **IDENTICAL.** 100/100 step logs, 11/11 subset evaluations, 11/11 checkpoints, final full validation (1,066 predictions and label scores) |

**What "identical" covers.** Rollout hashes, rewards, loss, clipping statistics, gradient and update norms,
`init_diff`, zero-signal batches, every evaluation's predictions, and every checkpoint. Frozen weights were unchanged
in all four runs.

**Resources** (one run each, wall clock including evaluations; diagnostic-grade memory, from the PyTorch allocator)

| run | train time | s/step | peak allocated | peak reserved |
|---|---|---|---|---|
| `qwen_resident` | 201 s | 2.0 | 1.69 GiB | 2.19 GiB |
| `qwen_l2l` | 360 s | 3.6 | 0.81 GiB | 1.37 GiB |
| `qwen_resident_short` | 27 s | 2.7 | 1.63 GiB | 2.15 GiB |
| `qwen_l2l_short` | 51 s | 5.1 | 0.81 GiB | 1.36 GiB |

So L2L costs about 1.8–1.9× the time for half the allocated memory. Synchronous weight streaming in generation
dominates the difference (Ticket 08).

**Not an L2L effect.** The rerun of `configs/qwen.yaml` does not reproduce the historical accuracy noted in that
file, 0.81 at step 90. Subset accuracy peaked at 0.79 (steps 40–50), fell to 0.55–0.63 (steps 60–90), and ended at
0.73; final full validation was 0.717. Both backends are identical, so this concerns the base GRPO run on this
hardware and software, not L2L.

**History of this result.** At `5300146`, the streamed head still computed on R rows and the reference on R+1.
- **On the GPU:** this made no difference, so these full runs are identical.
- **On the laptop CPU:** the same comparison in `tests/test_backend.py` drifted by about 1e-7 in gradient norm from
  step 1. `bc5a05d` makes the shapes identical (`docs/l2l/05_output_head.md`).
- **Not yet verified:** whether the laptop CPU tests pass after `bc5a05d`, and whether GPU runs after it stay
  identical. The shape test passes on the development host.

## Not established here

- **CUDA fault injection** (the advantage fault was run on CPU only).
- **Checkpoint verification and evaluation under L2L** (`run.py verify` / `validate` / `test`). These refuse the
  config and stay resident.
- **The chunked head in training** (H03).
- **The 8 GB budget** (Ticket 11). The real-model figures are diagnostic-grade (PyTorch allocator under WSL2/laptop), not
  assignment-grade evidence.
