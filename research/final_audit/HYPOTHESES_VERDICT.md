# Hypotheses verdict (audit, 2026-11-19)

**Source of definitions:** `research/l2l/hypotheses.yaml` (H01–H12; each criterion is marked "written before" its ticket's run). The tolerances live in `research/l2l/tolerances.yaml`. Neither file has changed since commit 9ca0390, so every criterion and bound used below predates both the 2026-11-18 and the 2026-11-19 evidence.

**Evidence judged:** the 150-update run of 2026-11-19 at commit `29852d1` on the Soup host (RTX 3060 12 GB): the staged runner's correctness stage (`assignment_results/final150_v2/state.json`, evidence in `evidence/20261119T05*Z_H*/`), the guarded manual stages (`assignment_results/manual/*/`), the held-out evaluation `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json` ("t13"), and the real-model kill/resume `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json`. The 2026-11-18 60-update evidence was removed from the repo and is not used, except where a change of verdict is explained.

**Rule of this audit:** each hypothesis is judged against its **pre-registered criterion**. Interpretations that go beyond the criterion are labelled INFERENCE. No criterion was changed.

**Recorded status:** the status recorded in `hypotheses.yaml` was not updated after the 2026-11-18 or 2026-11-19 runs (e.g. H11 and H12 still read `untested`, H08 still cites only `t09_resume_tiny_cpu`).

| H | kind | statement (abridged) | final status | one-line reason |
|---|---|---|---|---|
| H01 | correctness | sync (and, under Ticket 08, async) L2L logits = resident | **SUPPORTED** (sync and async, tiny models) | async t08 on the Soup host: 2690/2690 comparisons bitwise 0, including the same-host fixture tie-backs. Changed from "NOT SUPPORTED by its rule" on 2026-11-18 |
| H02 | correctness | checkpointed L2L backward = resident LoRA grads | **SUPPORTED** (tiny models) | 5296 comparisons bitwise 0 on the RTX 3060, async; earlier sync CPU/CUDA |
| H03 | correctness | memory-efficient output head preserves log-probs and grads | **SUPPORTED (historical; not exercised by the final run)** | t05 CPU; the final run used the full head (`vocab_chunk_size: null`) |
| H04 | correctness | streamed generation follows the same policy distribution | **SUPPORTED** (tiny models) | 1292 comparisons within bounds |
| H05 | correctness | complete streamed GRPO update = resident update | **SUPPORTED** (tiny models + real Qwen3-0.6B sync) | 6528 bitwise on the 3060 async; 0.6B resident vs l2l_sync 100 steps bitwise |
| H06 | performance | async prefetch reduces time without changing numerics | **NOT TESTED** | no prefetch_timing run, no sync-vs-async run, no overlap timeline |
| H07 | memory | bounded residency makes GPU weight memory independent of depth | **SUPPORTED (Ticket 02, allocator-only, diagnostic grade); Ticket 11 variant NOT TESTED** | `t02_store_depth` |
| H08 | correctness | interrupted and resumed run = uninterrupted run | **SUPPORTED** (criterion: tiny CPU, 6 crash points); **also shown on Qwen3-4B with a real SIGKILL** (outside the criterion's protocol) | t09 real-model kill/resume `pass: true`. Changed from "tiny CPU only, no real kill" |
| H09 | correctness | tensor-by-tensor merge preserves behaviour | **INCONCLUSIVE** under its criterion (the real Qwen3-0.6B variant was not run); 4B evidence outside the protocol is favourable | tiny CPU 30/30 PASS; t13 4B predictions 1066/1066 equal |
| H10 | performance | DFlash reduces target overhead while preserving the policy | **NOT TESTED** | never run on a real model; DFlash was off in this run |
| H11 | memory ("performance" in the yaml) | the complete Qwen3-4B system respects the 8 GB budget | **SUPPORTED by its own rule** (profile completed, no violation, numeric per-process samples), now with every phase also measured in separate guarded stages | manual profile + guarded train/merge/eval stages; budget 8e9 B (stricter than the criterion's 8 GiB) |
| H12 | learning | the trained policy meaningfully learned the intended task | **NOT SUPPORTED** | trained − base accuracy CI [−0.38, +2.25] pp includes 0; the criterion requires the CI entirely above 0. (AUC gain is significant, but the criterion is on accuracy) |

---

## H01: forward equivalence

- **Original hypothesis:** "Synchronous L2L produces the same logits as resident Qwen3 execution". The criterion requires, "on the same device, with the same weights and inputs", 0 difference for every quantity and dtype. Ticket 08 (`tolerances.yaml: l2l_async: same_as l2l_sync`) applies the same bound to async.
- **Motivation:** streaming must not change arithmetic. The bound is bitwise because the same device, kernels and bytes should give identical results (`tolerances.yaml:44-50`).
- **Implementation:** `qwen_grpo/forward_lab.py` (grid over the tiny fixture shapes × eager/sdpa × fp32/bf16 × LoRA variants × buffers × padding), plus the fixture tie-back. Since commit 5680e71 the tie-back has two parts (`forward_lab.py:233-281`):
  - **gating:** `logits.fixture.<shape>.cpu.same_host`. The L2L output must equal a resident forward computed **on the same host**, bitwise (atol = rtol = 0);
  - **non-gating diagnostic:** the comparison against the logits saved in the Ticket-00 fixture (produced on another host), reported as `cross_host_fixture_drift`.
- **Evidence:**
  - sync: `t03_forward_equivalence` PASS (laptop).
  - async on the Soup host, `research/l2l/experiments/t08_forward_equivalence_async_r20261119T051124Z/result.json` (evidence `evidence/20261119T051124Z_H01.lab_async_cuda/`):
    - **2690 comparisons, all bitwise 0**, including both same-host fixture tie-backs; `hypothesis_result: SUPPORTED`; staged-runner item `H01.lab_async_cuda` PASS, memory PASS.
    - Non-gating diagnostics on this host (Xeon E5-2697 v4): the saved fixture differs from this host's output by 1.341e-7 (0.6b-like) and 2.347e-7 (4b-like), and `l2l_drift_equals_resident_drift: true`, i.e. the **resident** forward shows exactly the same drift as L2L.
- **What changed since 2026-11-18:**
  - On 2026-11-18 the same two CPU fixture comparisons (1.34e-7 and 2.35e-7 against a bound of 0) were gating and failed the run. The audit then attributed them, by inference, to cross-host CPU reproducibility, and left one item NOT VERIFIED: that the resident forward drifts from the fixture on the GPU host itself.
  - That item is now **verified**: on the Soup host the resident forward drifts from the fixture by the same amount as L2L.
- **Validity of the gate change:**
  - The bound was not loosened: the gating comparison is still bitwise, and `tolerances.yaml` is unchanged.
  - The gate now compares on the same host, which is what the criterion's text ("on the same device") asks for. The cross-host comparison is still computed and reported, not hidden.
  - The change was made after the 2026-11-18 failure (commit 5680e71). It is disclosed here and in the commit message.
- **Supported conclusion:** SUPPORTED for sync and async L2L on the tiny fixture models. Not shown on a real model for async (stage C, Qwen3-0.6B, was not rerun).

## H02: checkpointed backward gradients

- **Evidence:** `t08_gradient_equivalence_async_r20261119T051240Z`, 5296 comparisons (loss, logits, hidden grads, LoRA grads with require_nonzero), all bitwise 0 on the RTX 3060. Earlier `t04_*` sync CPU/CUDA PASS.
- **Fault power:** `H02.fault_unsafe_reuse` (`evidence/20261119T051321Z_H02.fault_unsafe_reuse/`): all 80 unsafe-reuse configurations were refused at runtime with `BufferReuseError`; none went unnoticed.
- **Confounders:**
  - tiny random models (hidden 32/40, 4–6 layers, vocab 64);
  - `verify=True` mode, which host-synchronizes each layer use, whereas production uses `verify=False`.
- **Status:** SUPPORTED for the tested configurations. Not shown at real-model scale for async. Real-model scale is shown for sync (0.6B, 100 steps).

## H03: output head

- **Evidence:** `t05_logprob_equivalence_*` (CPU), all within tolerance. The final run did not use a chunked head.
- **Status:** SUPPORTED historically; irrelevant to the final configuration.

## H04: generation

- **Evidence:** `t08_generation_equivalence_async_r20261119T051351Z`: 1292 comparisons within bounds. 1130 are bitwise 0. The 162 nonzero ones are prefix logits (max 9.77e-4 against a bound of 5.1e-3) and prefix probabilities (max 1.3e-5).
- **Status:** SUPPORTED (tiny). The final run's rollouts used this generator (`training.json: rollout_backend.generator = StreamedGenerator`).

## H05: full update

- **Evidence:**
  - `t08_update_equivalence_async_r20261119T051445Z`: 6528 comparisons bitwise, including AdamW state;
  - `compare_qwen_resident_vs_qwen_l2l.json`: Qwen3-0.6B resident vs l2l_sync, 100 steps, every training row, adapter and optimizer state equal.
- **Confounder:** the 0.6B real-model comparison is **sync**, on another GPU, and the run had a dirty tree.
- **Status:** SUPPORTED.

## H06: async prefetch performance

- **Criterion:** requires a prefetch_timing case on the real Qwen3-0.6B with a bootstrap CI of the async/sync median ratio, and "overlap … only as measured from CUDA event timestamps".
- **Evidence:** none. Commit 5680e71 added timeline instrumentation and a Qwen3-4B prefetch_timing entry, but no prefetch_timing result exists in `research/l2l/experiments/`.
- **Indirect observation (INFERENCE, not admissible under the criterion):** the final-run mean step (9.55 s) is about 1.2× the pure transfer time at the 12.23 GB/s measured on 2026-11-18 (7.72–8.32 s). That fits a transfer-dominated step but is not a sync-vs-async comparison. The new profile's bandwidth figure (7.25 GB/s) is inconsistent with the step times (RESULTS_ANALYSIS §6.2), so it cannot be used either.
- **Status:** NOT TESTED.

## H07: depth-independent residency

- **Evidence:** `t02_store_depth` (allocator only).
- **Independent observation in the 4B run:** PyTorch allocated memory peaked at 1.81e9 B, against 8.04e9 B of bf16 weights (`t11_profile_manual.json: merge.total_bytes`). This confirms the model was never resident.
- **Status:** SUPPORTED (Ticket 02 criterion). The Ticket-11 execution variant was not run.

## H08: deterministic resume

- **Criterion:** `scripts/resume_experiment.py` with six injected crash points; committed rows, zero-signal attempts, subset evaluations, counters, rollout records and every common checkpoint's adapter and AdamW state must equal the uninterrupted run exactly, with no duplicated or missing samples.
- **Evidence within the criterion:** `t09_resume_tiny_cpu_r20261119T051532Z.json` (staged runner, Soup host, `evidence/20261119T051532Z_H08.resume_tiny_cpu/`): 6 crash points (`prompt_read`, `rollouts_generated`, `update`, `files_written`, `committed_before_rename`, `after_checkpoint`), `pass: true`.
- **New evidence beyond the criterion (real process kill, real model):**
  - `t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json` (config `configs/experiments/qwen3_4b_resume_kill.yaml`, 6 steps, checkpoint every 2; runs `runs/qwen3_4b_resume_kill_killA/`, `…_killB/`).
  - Run B was **SIGKILLed from outside** after 3 step lines, with checkpoints 0 and 2 committed. A **new process** resumed it from the latest checkpoint and finished 6/6.
  - Every check equals the uninterrupted run A: committed training rows, zero-signal attempts, subset predictions, counters, rollout records on disk, the (epoch, sample) multiset with no gap or duplicate, the adapter and AdamW state at every common checkpoint, the final torch (and CUDA) RNG, and the Python RNG restored from the resume checkpoint. `pass: true`.
  - All three processes stayed under the 8e9 B cap (nvidia-smi peaks 2364, 2220 and 2126 MiB).
  - The tiny CPU external-kill test also passes (`t09_resume_kill_tiny_cpu_r20261119T051046Z.json`, Soup host). An earlier local attempt (`…_r20261119T040250Z.json`) failed on `final_rng_state_equal` before the fix; the rerun `…_r20261119T040414Z.json` passed.
- **Remaining limits:**
  - One kill at one point of one short (6-step) run.
  - Per-token log-probs and gradient tensors are not recorded, so they are not compared (`not_compared`).
- **Status:** SUPPORTED. The assignment's "kill the job mid-epoch" is now **demonstrated on Qwen3-4B**. This was a gap on 2026-11-18.

## H09: merge

- **Evidence:**
  - tiny CPU: `t10_merge_equivalence_tiny_r20261119T051649Z`, 30/30 within bounds (weights max 3.8e-3, bound 4.1e-3; logits max 1.1e-2, bound 3.7e-2).
  - Real-model variant required by the criterion (Qwen3-0.6B bf16 with the step-90 adapter): **not run** in this study.
  - Outside the protocol, Qwen3-4B (t13 and `assignment_results/manual/merge_20261119T070347Z/`):
    - the merge ran in its own CPU process with **no CUDA context** (no GPU process seen by nvidia-smi), as the criterion requires;
    - all 6 adapted tensors are a correct rounding within half an ulp (max residual 8.8e-4);
    - predictions 1066/1066 equal; label-score difference median 9.5e-7, max 0.375 nats;
    - AUC difference +8.8e-6, CI [−2.8e-4, +3.1e-4];
    - the merged evaluation's allocator peak (1,563,545,088 B) is below the model size (8.04e9 B).
- **Status:** INCONCLUSIVE by the criterion. The 4B evidence is favourable at the prediction level (INFERENCE: score differences are within bf16 merge rounding; the mean adapter effect, 6.01 nats, is 16× the largest merge difference, though the criterion's ratio is defined on logits, not label scores). The merged checkpoint was not kept (`assignment_results/final150_v2/merged/` is empty).

## H10: DFlash

- **Status:** NOT TESTED on any real model. DFlash was disabled in this run (preflight: `dflash.enabled = false`). CPU unit tests with tiny/oracle drafters exist (`tests/test_dflash.py`).

## H11: 8 GB budget

- **Criterion:** `scripts/profile_run.py` (setup, rollout, old log-probs, forward, backward, optimizer, streaming merge and evaluation) completes with the guard armed, no violation, and a numeric per-process nvidia-smi reading in every sample, not under WSL2. The criterion names `configs/experiments/qwen3_4b_l2l.yaml` and a budget of 8 GiB.
- **Evidence:** `assignment_results/manual/profile_20261119T054235Z/results/t11_profile_manual.json`, run under the guarded wrapper:
  - config `configs/experiments/qwen3_4b_final_plain_150_v2.yaml` (the final run's own config), 2 steps;
  - budget **8,000,000,000 B**, stricter than the criterion's 8 GiB;
  - completed (`verdict: PASS`), 0 violations, 334 samples with 0 missing per-process readings, `wsl2: false`, `evidence: assignment`;
  - in-process peak 2,208,301,056 B; guarded external peak 2106 MiB.
- **Deviations from the criterion (disclosed):** a different config file (INFERENCE: more relevant, since it is the evaluated run's config) and a stricter budget. The profile was run by hand, because the staged runner had blocked it after the correctness FAIL.
- **Caveats inside the profile:**
  - The optimizer phases received 0 samples, and only 2 of 10 backward windows got a sample.
  - The profile's merge ran inside a process that already held the model on the GPU.
  - The profile's evaluation used the adapter model on 32 examples.
- **Supplementary evidence that closes those caveats:** each real phase also ran as its own guarded process with 200 ms nvidia-smi sampling and 0 violations:
  - training 2732 MiB (150 steps; includes backward and optimizer, unlabelled);
  - the merge of the evaluated adapter: no GPU process at all;
  - the evaluation of base, trained, **merged** and control: 1912 MiB.
- **Not shown:** that the guard would actually stop an over-budget run on this host. The two enforcement unit tests failed there (they pass locally). The criterion only requires the guard to be armed, so this does not change the verdict, but it limits the claim "strictly enforced".
- **Status:** SUPPORTED by its rule. The old gap (merged-model evaluation not measured) is closed.

## H12: meaningful learning

- **Criterion (pre-registered):**
  - trained accuracy > base with the paired-bootstrap 95% CI **entirely above 0**;
  - merge agreement;
  - invalid rate does not rise;
  - no label collapse;
  - LR=0 control unchanged.
- **Evidence (t13):**

  | sub-criterion | observed | met? |
  |---|---|---|
  | Δ accuracy CI entirely above 0 | +0.938 pp, CI [−0.375, +2.251] | **no** |
  | merge agreement | 1066/1066 | yes |
  | invalid rate does not rise | 0.47% → 0.09% (5 → 1) | yes |
  | no label collapse | predicted POSITIVE 0.489 | yes |
  | LR=0 control unchanged | exactly unchanged | yes |

- **Status:** **NOT SUPPORTED**, because the first sub-criterion fails.
- **Interpretation (INFERENCE):**
  - The adapter is active and changed the model: label scores shifted +6.01 nats on average (SD 0.88) and 56 predictions changed.
  - Most of the change is a uniform POSITIVE bias shift (Spearman 0.9985 with base).
  - Unlike the 60-update run, there is now a **statistically significant ranking gain**: AUC +0.0012, CI [+0.0002, +0.0022]. Recall is more balanced (POS 0.816 → 0.876, NEG 0.938 → 0.897).
  - The accuracy gain is not significant (McNemar 22 vs 32, p = 0.22).
  - Rollout reward on all attempts stayed flat (0.856–0.916 per 50-attempt bin).
  - Summary wording: learning is **small and partly demonstrated** (ranking yes, accuracy no). The pre-registered criterion is not met.

---

## Separation by type of claim

| type | what is established | what is not |
|---|---|---|
| correctness | L2L forward, backward, generation and update equal resident bitwise on tiny models (async, same host, all H01 comparisons now pass) and on Qwen3-0.6B (sync, 100 steps); GRPO math verified by code reading and CPU tests; deterministic resume after a real SIGKILL on Qwen3-4B | async on a real model (stage C not rerun); production `verify=False` mode bitwise |
| memory efficiency | every phase of the Qwen3-4B run (8.04 GB bf16 weights) at ≤ 2.86e9 B nvidia-smi per process, including the merged-model evaluation (1912 MiB); budget 8e9 B exactly | that the guard stops an over-budget run on the Soup host (enforcement tests failed there) |
| transfer overlap | async API and event ordering implemented; timeline instrumentation added | overlap measured: **no** |
| throughput | 2.80 updates/min, 9.55 s per successful step, 40.1 rollout tok/s; transfer-dominated (INFERENCE) | any speed-up vs sync; DFlash speed-up; a consistent bandwidth figure for this host |
| training/learning | adapter changes the model; small significant AUC gain on 4B; pipeline learns on 0.6B (+6.4 pp, CI [4.2, 8.5]; different setting) | significant accuracy improvement on 4B |
| merge | prediction-level equality on 4B; CPU merge with no CUDA context | pre-declared logit tolerance for the 4B merge; the real 0.6B variant of H09 |
| reproducibility | commit, configs, dataset hash, model revision, package versions, adapter sha256 recorded; bitwise checkpoint reload; final-eval command and logs recorded; evidence manifest `export_manifest.sha256` verifies | the merged checkpoint (not kept); a single orchestrated verdict (the staged runner's correctness gate failed and later stages were run by hand) |
