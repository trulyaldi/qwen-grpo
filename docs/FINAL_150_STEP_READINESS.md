# Readiness for the 150-update Qwen3-4B experiment

Written 2026-11-19, after comparing the original assignment (`Soup-Task-01-AI-Engineer-v5.pdf`) with the repository,
the 60-update Vast.ai run of 2026-11-18 (`docs/results_20261118T171404Z.md`) and the audit in
`research/final_audit/RISKS_AND_GAPS.md`.

Nothing in this document comes from a new GPU run. This host is a Quadro P1000 under WSL2 with a torch build that has
no kernels for it, so every GPU claim below is a **prepared check**, not a result. Each row says which command
produces the evidence.

**Verdict: READY WITH GPU-ONLY CHECKS OUTSTANDING.** The driver (`scripts/final150.py`) will not start the 150-update
run until the GPU-only gates below have passed on the rented card.

## Requirement by requirement

| # | original requirement | previous gap | code change | new test (CPU, run here) | local result | remaining limitation | GPU run needed? |
|---|---|---|---|---|---|---|---|
| 1 | "within 8 GB of GPU memory as shown by nvidia-smi (CUDA context included)" | The in-process guard enforced 8 GiB = 8,589,934,592 B. Only the orchestrator's watchdog used 8e9. The final runs were started by hand, outside that watchdog. | `memory.budget_bytes` (exact bytes) in `qwen_grpo/config.py`, with `budget_bytes(cfg)` as the single source. The trainer guard, the allocator cap and `profile_run.py` all use it. New config sets 8,000,000,000. An exact budget without `memory.guard` is refused. | `tests/test_final150.py` (bytes ≠ GiB, guard required, historical config hashes unchanged); `tests/test_guarded_run.py` (7629 MiB passes, 7630 MiB stops the workload) | pass | nvidia-smi samples every 200 ms, so a spike shorter than that is invisible (see below) | yes: smoke + every stage |
| 2 | raw nvidia-smi output sampled during the whole run | The merge and the merged-model evaluation had no nvidia-smi evidence (P0-1). Telemetry did not detect missing samples, a sampler that died, hidden PIDs, or a window it failed to cover. | `qwen_grpo/telemetry.py`: `run_guarded()`, the one supervisor (the orchestrator's `run_item_quiet` now delegates to it). It starts monitoring before the workload exists, records workload start/end UTC and nvidia-smi's UTC offset, and checks sampling gaps and coverage. It fails closed when a sampler dies, when no attributed per-process sample appears within 900 s, or when nvidia-smi is missing (the workload is not started). `phase_peaks()` slices the raw CSV per phase. `scripts/guarded_run.py` exposes it as a CLI. | `tests/test_guarded_run.py` (10 tests: raw files, attribution, cap stop, missing smi, dying sampler, hidden PIDs, `[N/A]`, CPU-only workload, gaps/coverage, phase slicing) | pass | per-process attribution needs PIDs visible to nvidia-smi (true on the Vast host on 2026-11-18; WSL2 and some containers: INCONCLUSIVE, never PASS) | yes: `final150.py smoke` must show the four raw files with attributed samples |
| 3 | merge and merged-model evaluation under the cap | `final_eval.py` merged in a temp dir inside the eval process. The merged checkpoint was deleted, and the eval had no sampler. | `scripts/merge.py` refuses to overwrite and writes `merge_provenance.json` (adapter sha256, shard sha256, max RSS). `final_eval.py --merged-dir` evaluates a merged checkpoint written by a separate monitored process. It checks that the adapter sha256 matches and records per-model UTC phase windows. The driver runs merge and eval as separate guarded items (merge with `--no-gpu-expected`) and writes `phase_memory.json`. A phase with no sample is INCONCLUSIVE. | `tests/test_guarded_run.py::test_a_cpu_only_workload_passes_at_zero_bytes_only_when_declared`, `::test_phase_peaks_*` | pass | the CPU merge's "0 bytes" is evidence that nvidia-smi never listed the process, not a measured peak | yes: `final150.py merge`, `eval` |
| 4 | "Prove equivalence: … logits match a reference non-L2L run … within a tolerance you justify" | H01 failed on Vast.ai only in the two comparisons against stored CPU fixtures (1.34e-7, 2.35e-7, bound 0). All 2,688 same-GPU comparisons had max_abs 0. | `forward_lab.fixture_tie_back`: the resident reference is recomputed on the same host with the unchanged resident code, and the L2L logits must match it **bitwise** (`…cpu.same_host`, gating, atol = rtol = 0). The stored-fixture comparison keeps its zero bound. It stays gating when the stored fixture's packages and recorded CPU match this host. Otherwise its rows are kept under `result["diagnostics"]` (`cross_host_fixture_drift`, `gating: false`), next to the resident drift. That drift must equal the L2L drift bitwise. No tolerance was widened. | `tests/test_h01_same_host.py` (bitwise same-host rows; a simulated 2e-7 cross-host drift is recorded and fails as a diagnostic but does not gate; a wrong-layer fault still fails the gating rows; the historical Vast.ai FAIL re-derives as FAIL) | pass (4/4) | the same-host methodology has not yet run on a GPU host. The historical FAIL stays in `research/l2l/experiments/t08_forward_equivalence_async/` | yes: `final150.py correctness` (`H01.lab_async_cuda --again`) |
| 5 | copy and compute overlap on a separate CUDA stream (and "was your prediction right?") | H06 never ran. `profile_run.py` under-counted async H2D bytes, because prefetched layers were hits and counted 0. | `prefetch.executor_bytes()` counts every issued copy. `profile_run.py` and the trainer use it. `timeline_report` and the pure `timeline_metrics` add bytes, bandwidth, compute idle, stall-on-copy, and submit-to-ready latency (a submit event on the compute stream), all from CUDA event timestamps of completed work. New Qwen3-4B lab config and study entry `H06.prefetch_timing_4b`: sync vs async on the same model, store, inputs and pools, `verify` off on both sides, bitwise equality of the first trial checked. | `tests/test_h2d_accounting.py` (prefetches counted; the old count missed them; phase deltas sum to the total; hand-made timelines give the expected overlap / stall / idle / bandwidth / latency) | pass (6/6) | event recording itself costs a little on both sides. No `nsys` timeline is wired in (see the experiment doc for the optional command) | yes: `final150.py profile` |
| 6 | "kill the job mid-epoch, resume, and show the same samples with no duplicates and no gaps" | Only in-process `InjectedCrash` evidence existed, resumed in the same process. | `scripts/resume_kill_test.py`: separate processes. B1 is **SIGKILLed** from outside after its first resumable checkpoint, with an uncommitted rollout on disk (`QWEN_GRPO_STALL`, a print-and-sleep hook that raises nothing). B2 is a new process started with `--resume latest`. A vs B checks: committed rows, rollout records, (epoch, sample) multiset, epochs complete, LoRA + AdamW at every checkpoint, torch RNG, restoration of the Python RNG, no leftover partial checkpoints. GPU mode: `configs/experiments/qwen3_4b_resume_kill.yaml`, each process under the cap. | CPU tiny mode run here: `research/l2l/experiments/t09_resume_kill_tiny_cpu_r20261119T040414Z.json` (PASS, 12/12). The first attempt `…r20261119T040250Z.json` is kept: it FAILed on a wrong check, explained below | PASS (CPU) | the Qwen3-4B GPU mode has not run (costs ~10 min) | optional: `scripts/resume_kill_test.py --config configs/experiments/qwen3_4b_resume_kill.yaml --device cuda` |
| 7 | Soup's checks (data doctor, pre-flight, soup ship) | The cleanup (9ca0390) removed the Soup integration. The 60-update run's Soup items were NOT_TESTED. | Restored `qwen_grpo/integrations/soup.py` (verified against 0.77.2) and the archived 0.77.2 evidence `research/soup/0.77.2_cpu/` (byte-identical, 27 files). New `scripts/soup_checks.py` adds discovery (`--help` of every command used), six controlled invalid data-doctor fixtures, adapter checks on the real 4B checkpoints, `soup ship` on the real held-out numbers, and explicit NOT_RUN for pre-flight | the CLI options were checked statically against the soup-cli 0.78.0 wheel source | **not executed here**: installing Soup was not permitted on this host | `docs/SOUP_CHECK_COVERAGE.md` | needs Soup installed (CPU is enough): `final150.py soup` |
| 8 | memory and time budget before running, compared with measurement | The predictions existed (frozen 2026-11-18). The comparison was spread over reports, and "researcher to explain" gaps were left open. | `scripts/budget_residuals.py` → `results/budget/residuals/qwen3-4b-final_150_v2.{json,md}`. ANALYTICAL, MEASURED and CALIBRATED numbers are kept apart, with residuals. The 150-update forecast is labelled as calibrated on the 60-update run, not as an independent prediction. | generated here from saved files | done | the calibrated forecast inherits the 60-update run's acceptance rate | no |
| 9 | 150 successful updates, validation every 10, checkpoints at each | n/a (new protocol) | `configs/experiments/qwen3_4b_final_plain_150_v2.yaml` (and `_lr0`, `_smoke`). The trainer now records a timestamp, settings, label scores, AUC and adapter sha256 for each subset evaluation; advantage statistics and an nvidia-smi peak for each update; timing, bytes and nvidia-smi peak for each zero-signal attempt. | `tests/test_final150.py` (exact protocol values; only the intended keys differ from the 60-update run; run names new; evaluations at 0/2/4/6 with label scores; one zero-signal timing per zero-signal attempt) | pass | `max_attempts` 1200 may still run out if acceptance falls below 12.5%. The run then says so and the driver marks the stage FAIL | yes: the run itself |
| 10 | orchestration with gates, no silent bypass | The 60-update gate blocked on H01, and the runs were then started by hand outside it | `scripts/final150.py`: dry-run, stages, gates, a PASS-only policy, `--retry` for expensive stages, explicit `--waive` (reported as WAIVED, never PASS), never overwriting, `export` with a sha256 manifest | `tests/test_final150.py` (dry run writes nothing; gates block; a PASSed stage is not re-run; a failed expensive stage needs `--retry`; waivers need a reason) | pass | it is a separate driver from `run_assignment.sh` (reason in the experiment doc) | yes |

## The kill/resume finding on the "RNG" check

The first CPU kill/resume run compared the final checkpoint's whole RNG state between A and B and failed. Only
Python's global `random` state differed, and it already differed at the step-2 checkpoint, which B1 wrote **before**
it was killed. The trainer never seeds or draws from that generator. Batch order comes from dedicated
`random.Random(seed)` streams whose state is part of the compared sampler and rollout records. Two independent processes
therefore start from different OS-entropy states, so the check was wrong, not the resume.

The check was replaced by two checks that do test restoration:

- the torch RNG state, which drives rollout sampling, must match A exactly;
- B's Python state must equal the state saved in the checkpoint B resumed from.

The failing record is kept on disk.

## The sampling limitation, stated once

nvidia-smi sampled every 200 ms sees allocations that last at least that long. A peak it reports is a lower bound on
the true peak. The second line of defence is in-process:

- `memory.guard` samples every 0.5 s and stops at the first sample above 8e9 B;
- the allocator cap (8e9 B as a fraction of the card) makes an over-cap PyTorch allocation fail immediately.

The allocator cap excludes the CUDA context, so it is never used as evidence of compliance. The 60-update run's
backward and optimizer windows (< 0.25 s) had no sample. Neither layer can prove that no sub-200 ms spike outside the
PyTorch allocator crossed the cap.

## Blocking items before spending GPU credit

1. `final150.py correctness` must PASS on the rented GPU:
   - H01 with the same-host methodology;
   - H02, H04 and H05 async;
   - both unsafe-reuse fault controls, the H02 one with the 2026-11-18 runtime refusal;
   - tiny resume and merge;
   - the CPU suite.
2. `final150.py smoke` must PASS: the raw `nvidia_smi_apps.csv`, `nvidia_smi_gpu.csv`, `pids.csv` and
   `telemetry.json` exist, have attributed samples and no gaps, and checkpoints step0/2/4 publish.
3. `final150.py profile` must PASS on memory. H06 may come out "no speed-up", which is a result, not a failure.

Without these, `final150.py train` refuses to start. That is intended.
