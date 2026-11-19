# Recommended next actions (audit, 2026-11-19)

Scope: the 150-update run `qwen3_4b_final_plain_150_v2` at commit `29852d1` (artifacts listed in FINAL_VERDICT.md). This replaces the action list written for the 60-update run.

**Guiding rule:** no new GRPO training. Every proposal names the decision it affects. Commands use only flags that exist in the repository; they were checked against the argparse definitions.

## Done since the 60-update audit

| former action | status | evidence |
|---|---|---|
| B1: merged-model evaluation under external nvidia-smi at 8e9 B | **DONE.** Merge as its own CPU process (no CUDA context, device 13 MiB, 32 s); eval of base, trained, merged and control under the guarded wrapper: per-process peak 1912 MiB over 7295 samples, 0 violations; command and logs kept | `assignment_results/manual/merge_20261119T070347Z/`, `assignment_results/manual/eval_20261119T070439Z/` |
| B2: attribute the H01 failure | **DONE**, by a changed gate rather than a separate CPU check: H01 now gates on the same-host comparison (`5680e71`) and passes 2690/2690 on the RTX 3060; the cross-host fixture drift is recorded as a non-gating diagnostic | `research/l2l/experiments/t08_forward_equivalence_async_r20261119T051124Z/` |
| B4: real process kill/resume | **DONE**, on the real Qwen3-4B (SIGKILL of the process group, resume in a new process, `pass: true`), plus the tiny CPU external-kill test | `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json` (+ `_raw/`); `t09_resume_kill_tiny_cpu_r20261119T051046Z.json` |
| guard at 8 GiB instead of 8 GB | **DONE.** `memory.budget_bytes` = 8,000,000,000 B, checked by the preflight | `assignment_results/final150_v2/state.json` |
| run Soup's checks on the final artifact | **PARTLY DONE** (see B2 below) | `research/soup/0.78.0_20261119T073222Z/` |

## A. Mandatory before finishing the report (no GPU)

1. **Adopt DON'T SHIP as the verdict and report H12 as NOT SUPPORTED, with the AUC result beside it.**
   - Use the pre-registered rule: accuracy +0.938 pp, CI [−0.375, +2.251]; McNemar p = 0.22.
   - Report the secondary result: AUC +0.0012, CI [+0.0002, +0.0022]; invalid answers 5 → 1; predicted-POSITIVE 0.436 → 0.489.
   - Explain the bias-shift finding (+6.01 nats, Spearman 0.9985; RESULTS_ANALYSIS §4.2).
   - Say that the 150-update run followed the 60-update null (forking paths), and give the 60-update numbers as the earlier run.
   - *Decision affected:* the verdict.
   - *Cost:* writing only.
2. **Present the execution history truthfully.**
   - The staged runner (`scripts/final150.py`) passed preflight and every correctness lab, but its correctness gate FAILED on `pytest_cpu` (2 guard-enforcement tests of 360), so it BLOCKED smoke and profile.
   - Smoke, profile, training, the control, the merge and the eval were then run by hand, each under `scripts/guarded_run.py` with the 8e9 B cap and 200 ms nvidia-smi sampling.
   - There is no generated requirements matrix or verdict for this run.
3. **Disclose every unmet requirement as such.** These are:
   - DFlash never run on the real model;
   - overlap not measured (H06 was part of the blocked profile stage);
   - Soup `data doctor` and `adapters check` did not start (missing `soup-cli[train]` dependencies), and `doctor` ran in another interpreter;
   - the in-process guard's stop-on-violation tests failed on the Soup host;
   - the residency choice of 0 versus the calculator's recommendation of 20;
   - the merged checkpoint was not kept (shard sha256s are in t13).
4. **Use only the numbers in REPORT_READY_FACTS.md.** Do not use the profile's bandwidth (7.25 GB/s) or its transfer shares as fact, nor the successful-step reward curve.
5. ~~Fix the existing draft report.~~ **Done 2026-11-19:** `reports/Soup_Final_Research_Report.{html,pdf}` rewritten for the 150-update run (no host-RAM figure; contains "what the check does not detect"); `reports/REPORT_EVIDENCE_MAP.md` regenerated; `reports/verify_report_numbers.py` 93/93.

## B. Strongly recommended (small, targeted; ordered by value per cost)

| # | unresolved question | why existing evidence is insufficient | hardware | workload | command (exists in repo) | evidence artifact | decision criterion | essential? |
|---|---|---|---|---|---|---|---|---|
| B1 | Does the async prefetch overlap copies with compute, and does it beat sync? | no sync baseline or timeline exists; the profile's bandwidth figure (7.25 GB/s) contradicts the step times, so "transfer-bound" is uncertain | RTX 3060 | about 10–20 min | `.venv/bin/python scripts/guarded_run.py --out assignment_results/manual/h06_<UTC> -- .venv/bin/python scripts/experiment.py run --entry H06.prefetch_timing_4b` (sync vs async on Qwen3-4B, `research/l2l/configs/async_qwen3_4b.yaml`, verify off, first trial checked bitwise) | the lab result JSON, CUDA-event overlap and stall metrics, guarded telemetry | async/sync median step ratio with a bootstrap CI; overlap from event timestamps. INFERENCE predicts a ratio near 1 if the step is transfer-bound | recommended (live-discussion question: "Where is your L2L bottleneck?"); otherwise disclose as unmeasured |
| B2 | What do Soup's `data doctor` and `adapters check` catch? | both failed to start on missing dependencies; that is an installation fact, not a finding | CPU only | minutes | install `soup-cli[train]` in the environment Soup runs in, then `.venv/bin/python scripts/soup_checks.py --out research/soup/<version>_<UTC>` (or `scripts/final150.py soup --retry`) | `summary.json` plus one log per check | each of the 6 injected data faults detected or not; adapter check verdicts on step 0 / step 150 / control | recommended; closes most of F2 |
| B3 | Why do the two guard-enforcement tests fail on the Soup host? | they pass locally at the same code; on the Soup host they did not raise | CPU only (the Soup host image) | minutes | `.venv/bin/python -m pytest -x -vv tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run tests/test_profiler.py::test_the_guard_records_a_violation_and_interrupts_the_main_thread` | pytest log | if they pass on a re-run, the failure was flaky (disclose); if they fail again, find the host-specific cause before claiming in-process enforcement | recommended; otherwise disclose (the external supervisor is the cap of record) |

## C. Optional (only if credit remains; each answers a scored but non-blocking item)

| # | question | hardware / cost | command | criterion |
|---|---|---|---|---|
| C1 | DFlash on the real model: greedy equality, acceptance length, step time with and without, drift | RTX 3060 plus a ~1 GB drafter download; about 30–60 min | `.venv/bin/python scripts/guarded_run.py --out assignment_results/manual/dflash_<UTC> -- .venv/bin/python scripts/dflash_bench.py --config configs/experiments/qwen3_4b_l2l_dflash.yaml --adapters runs/qwen3_4b_final_plain_150_v2/checkpoints/step150 --max-new 4 32 --budget-gib 8` | greedy tokens equal; passes per token < 1; GRPO step time ratio CI. **Prediction (INFERENCE):** at 4 new tokens and B×G = 30 rows with row-min acceptance, little or no speed-up; rollout is only 23.5% of a timed step |
| C2 | Shuffled-reward negative control | about 15 min training + 25 min eval (from the control and eval stage times) | needs a config flag; none exists (`docs/REMAINING_LIMITATIONS.md`: "not implemented") | trained − control CI; separates reward learning from drift |
| C3 | Async equivalence with `verify=False` on GPU | minutes | lab cases H02/H05 with verify off (if exposed by config) | bitwise 0 |
| C4 | A forgetting baseline for `soup ship` | depends on the benchmark chosen | `soup ship` with `--baseline` on base and trained (see `research/soup/0.78.0_20261119T073222Z/help_ship.log`) | Soup's regression rule evaluated instead of `missing_baseline` |

## D. Not recommended

- **Longer GRPO training (another run beyond 150 updates).**
  - Only 4.85% of groups are mixed and 77.7% of rollout batches carry no signal; going from 60 to 150 updates moved accuracy from +0.375 to +0.938 pp, still with a CI that includes 0.
  - The bias-shift pattern suggests diminishing returns (INFERENCE).
  - It would not change the verdict without a pre-registered stopping rule.
  - A third look to chase significance would make the forking-paths problem worse.
- **Re-running the 150-update training, the LR=0 control or the eval.** The existing telemetry already covers them.
- **Waiving the failed `pytest_cpu` item to make the staged runner "pass".** Report the failure; the runner's `--waive` accepts only correctness lab entries in any case.
- **Loosening H01's bitwise tolerance retroactively.** The same-host gate is the rule now; keep the cross-host drift as a diagnostic.
- **Editing anything under `assignment_results/`, `evidence/`, `runs/`, or `research/l2l/experiments/`.**

## Future research (beyond this assignment)

- Address the zero-signal waste:
  - harder or curriculum prompts (filter out prompts the base always gets right);
  - dynamic resampling until a group is mixed;
  - a larger G only for uncertain prompts, or a higher sampling temperature;
  - skipping the backward pass for uniform groups.
- Make the micro-batch loop layer-major, cutting 10 of 14 sweeps per step (INFERENCE: about 50% wall time).
- Keep up to 20 layers resident (the calculator predicts 6.56 GiB, under the cap).
- Use per-row ragged acceptance for DFlash.
