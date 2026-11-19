# Experiment inventory (audit, 2026-11-19)

Each experiment carries one of four states, which are kept separate:
- **planned:** in a config or protocol, never executed;
- **executed:** the process ran;
- **passed:** it met its own pass rule;
- **supports:** the outcome supports the research claim.

All times are UTC. "Soup host" = the rented RTX 3060 12 GB (sm_86), Linux 7.0.0-38-generic, 72 CPUs (Xeon E5-2697 v4), 135,040,708,608 B RAM, torch 2.14.1+cu126 (`runs/qwen3_4b_final_plain_150_v2/training/training.json: environment`; `assignment_results/final150_v2/state.json`, preflight items `cuda_bf16` and `host_ram`).

This inventory covers the **150-update run `qwen3_4b_final_plain_150_v2`** at commit `29852d1` (clean tree). The earlier 60-update run (`assignment_results/20261118T171404Z`, `runs/qwen3_4b_final_plain`, `runs/qwen3_4b_final_plain_lr0`, `runs/qwen3_4b_smoke`, `evidence/20261118T17*`) was removed from the repo and is archived in `~/soup-assignment-results.tar.gz`. It appears below only as a superseded timeline row.

## 0. Reconstructed timeline

| when (UTC) | what | where | outcome |
|---|---|---|---|
| 2026-11-16 | Step 0: theoretical memory/PCIe budget | commit 21ea586 | written |
| 2026-11-16 07:54 | Qwen3-0.6B GRPO, 100 steps, resident backend, RTX 3050 Ti laptop (WSL2), dirty tree | `runs/qwen/` | completed. Test (step 90): 0.733 → 0.796, Δ +6.4 pp, CI [+4.2, +8.5] (`runs/qwen/evaluation/step90/test.json`) |
| 2026-11-17 07:03–07:11 | Qwen3-0.6B resident vs l2l_sync, 10 and 100 steps, laptop | `runs/qwen_resident*`, `runs/qwen_l2l*`, `research/l2l/experiments/compare_qwen_resident_vs_qwen_l2l.json` | all training rows, adapters (steps 0–90) and optimizer states **bitwise equal** across backends |
| 2026-11-16/17 | Tickets 00–10 lab cases (tiny random Qwen3 shapes; CPU + RTX 3050 Ti) | `research/l2l/experiments/t01_…t10_*`, `ticket0*_tests.*` | H01–H05, H07, H08 recorded SUPPORTED in `research/l2l/hypotheses.yaml` |
| 2026-11-17 | Kaggle T4 fp32 runs (Qwen3-0.6B) | archived (`~/qwen-grpo-archive/2026-11-17_pre-cleanup/kaggle_outputs.tar.gz`) | not re-audited here |
| 2026-11-17 13:42 | Soup v0.77.2 checks on CPU / Qwen3-0.6B | `research/soup/0.77.2_cpu/` | historical; restored to the repo in 5680e71 |
| 2026-11-17 17:52 | H08 tiny CPU resume (local Xeon) | `evidence/20261117T175211Z_H08.resume_tiny_cpu/` | PASS |
| 2026-11-18 17:14–18:37 | **superseded** 60-update run: orchestrator `20261118T171404Z` (H01 FAIL blocked training), then manual training, control and evaluation | removed from the repo; `~/soup-assignment-results.tar.gz` | +0.375 pp, CI [−0.563, +1.313]; not used by this audit except for labelled comparisons |
| 2026-11-19 04:02–04:05 | tiny CPU external-kill resume test (local laptop) | `research/l2l/experiments/t09_resume_kill_tiny_cpu_r20261119T040250Z.json`, `…_r20261119T040414Z.json` | first attempt FAIL (`final_rng_state_equal: false`); rerun after the fix PASS |
| 2026-11-19 04:08:54 | budget residuals and 150-update forecast, from the 60-update measurements | `results/budget/residuals/qwen3-4b-final_150_v2.{md,json}` | forecast 665 attempts, 0.83 h |
| 2026-11-19 (commits 5680e71, 29852d1) | exact 8e9-byte cap, guarded wrapper, same-host H01 gate, H2D accounting fix, real kill/resume script, Soup checks, `_150_v2` configs, `scripts/final150.py` | git | — |
| **05:05:05–05:05:13** | staged runner preflight | `assignment_results/final150_v2/preflight/20261119T050505Z/` | PASS |
| **05:05:22–05:17:53** | staged runner correctness stage | `assignment_results/final150_v2/correctness/20261119T050522Z/` | **FAIL** (only `pytest_cpu`: 2 of 360 tests); every lab item PASS |
| 05:36:15 | staged smoke and profile | `state.json` | **BLOCKED** ("correctness: FAIL") |
| **05:37:34–05:41:40** | guarded manual smoke (4 steps) | `assignment_results/manual/smoke_20261119T053734Z/`, `runs/qwen3_4b_final_plain_150_v2_smoke/` | exit 0, 2362 MiB, 0 violations |
| **05:42:35–05:44:22** | guarded manual profile (2 steps + merge + eval) | `assignment_results/manual/profile_20261119T054235Z/` | exit 0, 2106 MiB, profile verdict PASS |
| **05:45:19–06:41:11** | guarded manual training, 150 updates | `assignment_results/manual/train_20261119T054519Z/`, `runs/qwen3_4b_final_plain_150_v2/` | exit 0, 2732 MiB, 0 violations |
| **06:45:30–07:01:03** | guarded manual LR=0 control, 30 updates | `assignment_results/manual/control_20261119T064530Z/`, `runs/qwen3_4b_final_plain_150_v2_lr0/` | exit 0, 2732 MiB, 0 violations |
| **07:03:47–07:04:19** | guarded manual CPU merge of step150 | `assignment_results/manual/merge_20261119T070347Z/` | exit 0, no GPU process |
| **07:04:39–07:29:13** | guarded manual final evaluation (base / trained / merged / control) | `assignment_results/manual/eval_20261119T070439Z/`, `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json` | exit 0, 1912 MiB, 0 violations |
| 07:30:27 | Soup checks, first attempt | `research/soup/unavailable_20261119T073027Z/` | UNAVAILABLE (Soup not found) |
| **07:32:22–07:32:39** | Soup v0.78.0 checks | `research/soup/0.78.0_20261119T073222Z/` | executed (see §1.1) |
| **07:38:42–07:46:26** | real-model kill/resume (Qwen3-4B, external SIGKILL) | `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z{.json,_raw/}`, `runs/qwen3_4b_resume_kill_killA/`, `…_killB/` | `pass: true` |
| 2026-11-19 | this audit: CPU-only reanalysis and tests | `research/final_audit/` | — |

## 1. Staged run `assignment_results/final150_v2` (driver `scripts/final150.py`; state: `state.json`)

| id | purpose | config / command | evidence | result | status | interpretation / limits |
|---|---|---|---|---|---|---|
| preflight.config_150_v2 / config_lr0 | the configs hold the intended values | `configs/experiments/qwen3_4b_final_plain_150_v2{,_lr0}.yaml` | `state.json: stages.preflight` | run name; 150 (lr0: 30) steps; eval every 10; budget 8,000,000,000 B; guard on; DFlash off; `l2l_async`; Qwen3-4B; B=5, G=6; 4 new tokens; r=32, α=64; lr 2e-4 (lr0: 0); clip 0.2; streaming on; max_attempts 1200 | PASS | confirms DFlash was **intentionally off** and the cap is the literal 8e9 B |
| preflight: run dirs, nvidia-smi, cuda_bf16, host_ram, disk, budget files | environment | — | `state.json` | run dirs absent (new run); RTX 3060 with bf16; 135,040,708,608 B RAM; 58.6 GB free disk; prediction and residuals files present | PASS | environment checks only |
| correctness.pytest_cpu | CPU unit tests (17 test files/dirs) | `.venv/bin/python -m pytest -q …` (argv in `state.json`) | `correctness/20261119T050522Z/pytest_cpu.xml` | 360 tests: 341 passed, **2 failed**, 17 skipped | **FAIL** | failures: `tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run` (DID NOT RAISE SystemExit) and `tests/test_profiler.py::test_the_guard_records_a_violation_and_interrupts_the_main_thread` (DID NOT RAISE KeyboardInterrupt). Both pass locally at the same code (re-run by this audit: 2 passed). So the guard's ability to **stop** an over-budget run was not demonstrated on the Soup host |
| correctness.resume_kill_tiny_cpu | external-kill resume, tiny CPU model | `scripts/resume_kill_test.py --tiny` | `research/l2l/experiments/t09_resume_kill_tiny_cpu_r20261119T051046Z{.json,_raw/}` | equal to the uninterrupted run | PASS | tiny CPU model |
| H01.lab_async_cuda | async L2L logits/layers = resident | forward_equivalence | `evidence/20261119T051124Z_H01.lab_async_cuda/`, `research/l2l/experiments/t08_forward_equivalence_async_r20261119T051124Z/` | **2690/2690 bitwise equal**, including the same-host fixture tie-backs; cross-host fixture drift (1.34e-7, 2.35e-7) kept as non-gating diagnostics | PASS | the 2026-11-18 fixture-only failure is gone (gate fixed in 5680e71); see HYPOTHESES_VERDICT H01 |
| H01.fault_unsafe_reuse | fault injection has power | same, unsafe reuse | `evidence/20261119T051215Z_H01.fault_unsafe_reuse/`, `t08_forward_equivalence_async_unsafe_reuse_r20261119T051215Z/` | 66/66 configurations refused at runtime with `BufferReuseError`; 0 unnoticed | PASS (detected) | the lab result itself reads FAIL, which is the expected outcome for an injected fault; tests the host-side `_claim` guard only, not a real stream race |
| H02.lab_async_cuda | async checkpointed backward grads = resident | gradient_equivalence | `evidence/20261119T051240Z_H02.lab_async_cuda/` | 5296 comparisons bitwise 0 | PASS | tiny random models, `verify=True` |
| H02.fault_unsafe_reuse | fault power | — | `evidence/20261119T051321Z_H02.fault_unsafe_reuse/` | 80/80 refused, 0 unnoticed | PASS (detected) | same limits as the H01 fault |
| H04.lab_async_cuda | streamed generation = HF generate | generation_equivalence | `evidence/20261119T051351Z_H04.lab_async_cuda/` | 1292 comparisons; prefix logits max 9.8e-4 (bound 5.1e-3) | PASS | tiny models |
| H05.lab_async_cuda | full GRPO update = resident | update_equivalence | `evidence/20261119T051445Z_H05.lab_async_cuda/` | 6528 comparisons bitwise 0 incl. AdamW state | PASS | tiny models |
| H08.resume_tiny_cpu | crash/resume determinism | `scripts/resume_experiment.py` | `evidence/20261119T051532Z_H08.resume_tiny_cpu/`, `research/l2l/experiments/t09_resume_tiny_cpu_r20261119T051532Z.json` | 6 crash points, all equal to the uninterrupted run | PASS | tiny CPU model, in-process crashes (the real kill is §2 M6) |
| H09.merge_tiny_cpu | tensor-by-tensor merge equivalence | merge_equivalence `--device cpu` | `evidence/20261119T051649Z_H09.merge_tiny_cpu/`, `t10_merge_equivalence_tiny_r20261119T051649Z/` | 30 comparisons; weights max 3.8e-3 (bound 4.1e-3), logits 1.1e-2 (bound 3.7e-2) | PASS | tiny CPU only; the real Qwen3-0.6B variant was not run |
| smoke, profile | memory smoke and per-phase profile | — | `state.json` | gate: "correctness: FAIL" | **BLOCKED** | run by hand instead (§2) |
| soup (first attempt) | Soup checks | `scripts/soup_checks.py` | `assignment_results/final150_v2/soup/20261119T073027Z/`, `research/soup/unavailable_20261119T073027Z/` | Soup not installed / not found | UNAVAILABLE | — |
| soup (second attempt) | Soup checks | same | `assignment_results/final150_v2/soup/20261119T073222Z/`, `research/soup/0.78.0_20261119T073222Z/` | checks executed; findings in §1.1 | PASS (= executed) | PASS here means "the checks ran", not a verdict |
| stage C (real-model Qwen3-0.6B equivalence), DFlash stage | — | — | not part of this driver | — | NOT RUN | — |

- The staged runner has no training or evaluation stage of its own after the block, and it generates no requirements matrix or final verdict. **There is no orchestrated verdict for this run.** Any requirement status in this audit is the auditor's judgement from the artifacts.
- Integrity: `assignment_results/final150_v2/export_manifest.sha256` lists 1298 files; `sha256sum -c` passes for all of them (re-run by this audit).

### 1.1 Soup v0.78.0 checks (`research/soup/0.78.0_20261119T073222Z/summary.json`)

| check | executed? | result | what it shows |
|---|---|---|---|
| `soup doctor` | yes (exit 0) | ran in a **different interpreter** from training | describes Soup's own environment, not the training venv |
| `soup data doctor`, clean + 6 injected faults (no assistant, empty assistant, unknown role, truncation, non-canonical label, label leak) | ran, exit 1 on all 7 | "needs transformers to load a tokenizer — install soup-cli[train]" | **no fault was tested** |
| `soup adapters check` (150_v2 step0 / step150; lr0 step0 / step30) | ran, exit 1 on all 4 | "numpy package required" | **not tested** |
| reward stress, real reward | yes | robust: 0 of 4600 attacks accepted; reference accepted 100% | the real reward is not trivially gameable by these attacks |
| reward stress, deliberately permissive verifier (positive control) | yes | gameable: answer spray accepted 800/800 | Soup detects a gameable reward |
| `soup ship`, 150-update model | yes (exit 2) | DON'T SHIP, `failed_rule: missing_baseline` | task win recognised (0.8771 → 0.8865), but no general benchmark suite or `--baseline`; no paired CI used |
| `soup ship`, LR=0 control | yes (exit 2) | DON'T SHIP | no task win; same rule |
| `advise`, `plan`, `train` hardware-fit | not run | NOT_RUN by design | they plan Soup's own trainer from a `soup.yaml`; this engine is not a Soup trainer |

## 2. Guarded manual runs (outside the staged runner's gate, but each under `scripts/guarded_run.py` with 200 ms nvidia-smi sampling and cap 8e9 B)

| id | purpose | config | command | evidence | result | status | limits |
|---|---|---|---|---|---|---|---|
| M0 smoke | 4-step Qwen3-4B smoke | `configs/experiments/qwen3_4b_final_plain_150_v2_smoke.yaml` | `python scripts/run.py train --config …_smoke.yaml` (recorded in `guarded_run.json: argv`) | `assignment_results/manual/smoke_20261119T053734Z/`, `runs/qwen3_4b_final_plain_150_v2_smoke/` | 4 steps / 14 attempts; mean step 10.8 s; process peak 2362 MiB; 246 s | executed and completed | — |
| M1 profile | per-phase memory profile, 2 steps + merge + eval (32 ex.) | `configs/experiments/qwen3_4b_final_plain_150_v2.yaml` | `python scripts/profile_run.py --config … --prediction results/budget/predictions/qwen3-4b-final_8gb.json --steps 2 …` | `assignment_results/manual/profile_20261119T054235Z/` (result `results/t11_profile_manual.json`) | in-process peak 2,208,301,056 B; guarded peak 2106 MiB; H2D 7.25 GB/s; verdict PASS | executed and passed its own rule | optimizer phases 0 samples, backward 2 of 10; merge ran inside the GPU process; **its H2D figure is inconsistent with the training step times** (RESULTS_ANALYSIS §6.2) |
| M2 `qwen3_4b_final_plain_150_v2` | primary GRPO+LoRA training | `configs/experiments/qwen3_4b_final_plain_150_v2.yaml` (sha256 568afd24…, same as preflight and t13) | `python scripts/run.py train --config …150_v2.yaml` | `assignment_results/manual/train_20261119T054519Z/`, `runs/qwen3_4b_final_plain_150_v2/` | 150/150 steps, 672 attempts, 522 zero-signal, 3211 s, ‖Δθ‖ 1.9963, frozen 0/398 changed, nvidia-smi peak 2,864,709,632 B (2732 MiB), 0 violations | executed and completed; **not a staged-runner PASS** | run by hand after the correctness FAIL; guard enforcement tests failed on this host |
| M3 `qwen3_4b_final_plain_150_v2_lr0` | negative control | `…_150_v2_lr0.yaml` (sha256 9416f07a…) | `python scripts/run.py train --config …_lr0.yaml` | `assignment_results/manual/control_20261119T064530Z/`, `runs/qwen3_4b_final_plain_150_v2_lr0/` | 30/30 steps (configured 30, not 150), 208 attempts, adapter == init, subset predictions unchanged, 2732 MiB | executed and completed | equal to base by construction; not "the same run" |
| M4 merge | CPU merge of the step150 adapter | — | `python scripts/merge.py --adapter runs/qwen3_4b_final_plain_150_v2/checkpoints/step150 --out assignment_results/final150_v2/merged/qwen3_4b_final_plain_150_v2_step150` | `assignment_results/manual/merge_20261119T070347Z/`; provenance in t13 `merged_dir` | no GPU process; device 13 MiB; 32 s; max RSS 8,602,054,656 B; 6 tensors within half an ulp | executed and completed | **merged weights not kept**: `assignment_results/final150_v2/merged/` is empty; shard sha256 recorded in t13 |
| M5 t13 final evaluation | held-out test: base / trained / merged / control | config of M2; control = M3 step30 | `python scripts/final_eval.py --run qwen3_4b_final_plain_150_v2 --control qwen3_4b_final_plain_150_v2_lr0 --merged-dir … --out research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json` | `assignment_results/manual/eval_20261119T070439Z/` (stdout/stderr logs, telemetry), t13 | trained − base +0.938 pp, CI [−0.375, +2.251]; AUC +0.0012, CI [+0.0002, +0.0022]; merged = trained on 1066/1066; control = base; process peak 1912 MiB | executed; H12 rule **not met** | — |
| M6 real-model kill/resume | deterministic resume after an external SIGKILL, Qwen3-4B | `configs/experiments/qwen3_4b_resume_kill.yaml` (6 steps, checkpoint every 2) | `scripts/resume_kill_test.py` (child argv in the t09 file) | `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z{.json,_raw/}`, `runs/qwen3_4b_resume_kill_killA/`, `runs/qwen3_4b_resume_kill_killB/` | B killed after 3 step lines (checkpoints 0 and 2 committed), resumed by a new process, 6/6; all checks equal to A; nvidia-smi peaks A 2364, B1 2220, B2 2126 MiB | executed and **passed** (`pass: true`) | one kill point, short run; per-token log-probs and gradients not compared |

## 3. Earlier evidence relevant to claims (not re-executed)

| id | what | evidence | outcome | relevance |
|---|---|---|---|---|
| t03_forward_equivalence | sync L2L vs resident logits (H01) | `research/l2l/experiments/t03_forward_equivalence/` | PASS, laptop | H01 recorded SUPPORTED (sync) |
| t04_gradient_equivalence_cpu/cuda | sync checkpointed backward (H02) | `research/l2l/experiments/t04_*` | PASS | |
| t05_logprob_equivalence_* | output head (H03) | `research/l2l/experiments/t05_*` | PASS | not re-run on the Soup host |
| t06_generation_equivalence_cpu/cuda | streamed generation (H04) | `research/l2l/experiments/t06_*` | PASS | |
| t07_update_equivalence_* | full update (H05) | `research/l2l/experiments/t07_*` | PASS | |
| real_step_2026111{7}T0633*Z | real Qwen3-0.6B one-step sync L2L vs resident | `research/l2l/experiments/real_step_*.json` | equal | real-model equivalence evidence exists **only for l2l_sync on 0.6B** |
| t02_store_depth | allocator peak independent of depth (H07) | `research/l2l/experiments/t02_store_depth/` | SUPPORTED (allocator only) | |
| t10_merge_equivalence_tiny | merge (H09), earlier run | `research/l2l/experiments/t10_merge_equivalence_tiny/` | tiny only | rerun on 2026-11-19 (§1) |
| Soup 0.77.2 on CPU / Qwen3-0.6B (2026-11-17, laptop WSL2) | `soup data doctor`, `soup adapters check`, `soup ship` ("DON'T SHIP") | `research/soup/0.77.2_cpu/` | historical | not on the final artifact; the 0.78.0 checks (§1.1) are |

## 4. Planned but never executed

| item | where planned | status |
|---|---|---|
| DFlash Qwen3-4B training / bench (`configs/experiments/qwen3_4b_l2l_dflash.yaml`, `qwen3_4b_final.yaml`, `scripts/dflash_bench.py`), H10 | Ticket 12 | **NOT RUN** on any real model; disabled in this run. CPU unit tests exist (`tests/test_dflash.py`, `tests/test_dflash_training.py`) |
| H06 prefetch_timing (sync vs async speed; a Qwen3-4B entry was added in 5680e71) | Ticket 08 | NOT RUN |
| 210-step continuation (`configs/experiments/qwen3_4b_final_ext210.yaml`) | commit 0cc2715 | NOT RUN |
| H08 / H09 on real Qwen3-0.6B (stage C) | `configs/studies/assignment.yaml` | NOT RUN (H08 now has a real-model kill/resume on Qwen3-4B, §2 M6; H09's 0.6B variant is still missing) |
| shuffled-reward control | assignment option | NOT PLANNED |
| Soup data doctor and adapters check on the final artifacts | `scripts/soup_checks.py` | executed but **did not run their checks** (missing `soup-cli[train]` dependencies; §1.1) |
| copy/compute overlap measurement | Ticket 08 | NOT RUN |

## 5. CPU test suites

- **Staged runner `pytest_cpu` on the Soup host** (`assignment_results/final150_v2/correctness/20261119T050522Z/pytest_cpu.xml`, 324.5 s): 360 tests from 16 test files plus `tests/silent_failures`; 341 passed, **2 failed**, 17 skipped. The two failures are the guard-enforcement tests listed in §1.
- **Re-run of the two failing tests by this audit** (local laptop, `CUDA_VISIBLE_DEVICES=""`, `-p no:cacheprovider`, 2026-11-19): 2 passed. The failure is specific to the Soup host or container (INFERENCE: environment-dependent, e.g. signal or timing behaviour; not investigated).
- **Earlier local runs by this audit (2026-11-19, local Xeon W-2145 / WSL2, before the 150-update run's commit):**
  - Full suite: `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -p no:cacheprovider` was killed by the audit's 1700 s timeout before printing a summary. **Full-suite status: NOT VERIFIED.**
  - Targeted (GRPO sub-audit), completed: `tests/test_grpo.py`, `test_multistep.py`, `test_backward.py`, `test_sentiment.py`, `test_policy.py`: 142 passed, 42 skipped (CUDA-parametrized); `tests/test_trainer.py`: 10 passed.
- **To get a full-suite result:** rerun with a longer timeout, or per file, e.g. `pytest -q tests/<file>.py`.
