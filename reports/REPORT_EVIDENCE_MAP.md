# Evidence index for `Soup_Final_Research_Report.pdf`

This index sits outside the three-page report. It maps every important claim in the report to the files that support it.

- **Report:** `reports/Soup_Final_Research_Report.pdf` (3 pages). The editable source is `reports/Soup_Final_Research_Report.html`.
- **Build:** `.venv/bin/python -I reports/figures/make_figures.py && bash reports/build_pdf.sh`
- **Number check:** `python3 -I reports/verify_report_numbers.py` recomputes the report's numbers from the raw files and asserts them. It uses the CPU and the standard library only. On 2026-11-19 it passed **93/93 checks**.
- **Audit:** the longer audit with per-requirement verdicts is in `research/final_audit/`. Regenerate its numbers with `python3 -I research/final_audit/analyze.py`.

> **Submission note.** `runs/` and `assignment_results/` are gitignored (`.gitignore`). The raw logs, nvidia-smi samples and checkpoints listed below exist only in this working tree. They must be archived together with the repository when the work is submitted.

Abbreviations:

| short | path |
|---|---|
| RUN | `runs/qwen3_4b_final_plain_150_v2/` (final run, 150 updates, plain L2L) |
| CTL | `runs/qwen3_4b_final_plain_150_v2_lr0/` (LR=0 control, 30 updates) |
| T13 | `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json` (held-out evaluation) |
| MAN | `assignment_results/manual/<stage>_<UTC>/` (guarded stages: smoke, profile, train, control, merge, eval) |
| STG | `assignment_results/final150_v2/` (staged runner: preflight, correctness, Soup; `state.json`) |
| T11 | `MAN/profile_20261119T054235Z/results/t11_profile_manual.json` |
| PRED | `results/budget/predictions/qwen3-4b-final_8gb.md` (analytical budget, written before any 4B run) |
| RES | `results/budget/residuals/qwen3-4b-final_150_v2.md` (predicted vs measured; 150-update forecast) |
| SOUP | `research/soup/0.78.0_20261119T073222Z/` (Soup v0.78.0 run on these artifacts) |

Each claim carries a status: **measured** (read from an artifact or recomputed from one), **code** (established by reading the source), or **inference** (an interpretation, not a measurement).

## Raw logs and nvidia-smi (deliverables)

| deliverable | path |
|---|---|
| timestamped training logs | `RUN/training/training.log`, `CTL/training/training.log` (every line UTC-stamped) |
| per-step training records | `RUN/training/training.json` (steps, zero-signal batches, timings, evaluations, config, environment) |
| in-process nvidia-smi samples, whole run | `RUN/training/memory/20261119T054525Z.json` (5947 samples, raw `smi_apps_raw` / `smi_gpu_raw` per sample); `CTL/training/memory/20261119T064535Z.json` |
| external nvidia-smi per stage (CSV) | `MAN/*/telemetry/nvidia_smi_apps.csv`, `nvidia_smi_gpu.csv`, `pids.csv`; summaries in `MAN/*/guarded_run.json` |
| stage stdout/stderr (command lines included) | `MAN/*/stdout.log`, `MAN/*/stderr.log`; argv in `MAN/*/guarded_run.json` |
| staged-runner record | `STG/state.json`, `STG/preflight/*/`, `STG/correctness/*/`; integrity: `sha256sum -c STG/export_manifest.sha256` |
| verification script (part 3) | `scripts/final_eval.py` → T13; independent recomputation: `reports/verify_report_numbers.py`, `research/final_audit/analyze.py` |
| L2L equivalence tests | `scripts/l2l_lab.py` (cases `forward/gradient/generation/update_equivalence`); results in `research/l2l/experiments/t08_*_r20261119T05*Z/`; criteria in `research/l2l/hypotheses.yaml` and `research/l2l/tolerances.yaml`; unit tests in `tests/test_backward.py`, `tests/test_layer_stream.py`, `tests/test_prefetch.py`, `tests/test_multistep.py` |
| code | `qwen_grpo/` (engine: `qwen_grpo/execution/`), `scripts/`, `configs/experiments/qwen3_4b_final_plain_150_v2*.yaml` |

## Header and verdict

| claim | source | status |
|---|---|---|
| commit `29852d1`, clean tree; torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3; RTX 3060 | `RUN/manifest.json: git_commit`; T13 `provenance.environment` | measured |
| final run is plain L2L, DFlash off; `l2l_async`, 0 resident layers, 2 buffers, prefetch depth 1 | `RUN/config.yaml`; preflight rows `dflash.enabled: false`, `execution.backend: l2l_async` in `STG/state.json` | measured |
| 150 updates | `RUN/training/training.json: successful_steps` | measured |
| highest per-process peak 2,864,709,632 B = 35.8% of 8×10⁹ B | `RUN/training/memory/20261119T054525Z.json: peak.process_used` | measured |
| Δaccuracy +0.94 pp, CI [−0.38, +2.25] pp; ΔAUC +0.0012, CI [+0.0002, +0.0022] | T13 `paired.trained_minus_base`, `paired_auc.trained_minus_base` | measured |
| DON'T SHIP | sections 3–5 below; Soup agrees for its own reason (`SOUP/ship_t13_150_update_verdict.json`) | inference |

## Task and choices

| claim | source | status |
|---|---|---|
| reward = exact canonical label | `qwen_grpo/sentiment.py` (`reward`) | code |
| non-thinking template asserted on every prompt | `qwen_grpo/sentiment.py` (`build_prompts`); `tests/test_sentiment.py` | code |
| 1024 training prompts; 1066 test and 1066 validation examples; 0 shared texts | T13 `data_separation`; `research/final_audit/analysis_output.json: evaluation.overlap_*` | measured |
| Qwen3-0.6B moved 0.733 → 0.796 in 100 steps (earlier development run) | `runs/qwen/report.md`, `runs/qwen/evaluation/step90/test.json` | measured (historical) |
| base Qwen3-4B accuracy 0.877 | T13 `models.base.accuracy` | measured |

## 1. Memory and time budget

| claim | source | status |
|---|---|---|
| 4,022,468,096 parameters, 8.045 GB of bf16 weights; layers of 192.5 MiB, 6.768 GiB in total; embedding 741.9 MiB; vocab 151,936 | PRED "Model metadata" | measured (safetensors headers) |
| eq. (1) components: 2 buffers 385 MiB; KV cache 438.8 MiB (30 × 104 tokens); response-only logits 36.5 MiB; LoRA + AdamW 12.3 MiB | PRED "rollout" and "training" sections | measured (exact sizes) |
| ≈0.74 GiB for logits at all 104 positions | 36.5 MiB × 104 / 5 response positions | inference (arithmetic) |
| no reference log-probs (β = 0), no drafter loaded; drafter predicted at +0.84 GiB allocated | `qwen_grpo/grpo.py`; PRED "dflash (optional estimate)" row: 2.588 vs 1.750 GiB | code / predicted |
| 20 resident layers allowed (6.56 GiB); 0 chosen; 7.27 vs 3.23 GB per sweep | PRED residency section; `RUN/config.yaml: execution.resident_layers` | predicted / measured |
| Table 1: allocated 1.682 / reserved 2.506 / per-process 2.668 / device 2.699 GiB | `RUN/training/memory/20261119T054525Z.json: peak` | measured |
| Table 1: evaluation of all 4 models, 1.867 GiB per process (1912 MiB) | `MAN/eval_20261119T070439Z/guarded_run.json: telemetry.process_peak_mib`; CSV in `…/telemetry/` | measured |
| Table 1: merge opened no CUDA context; host max RSS 8,602,054,656 B; 32 s | `MAN/merge_20261119T070347Z/guarded_run.json: telemetry.memory_reason, children_max_rss_bytes, seconds`; output log `…/stdout.log` | measured |
| Table 1: host RSS 14.39 GiB (VmHWM) during training | `RUN/training/memory/*.json: samples[].VmHWM` | measured |
| Table 1: 94.8 GB mean H2D per update; 13–14 sweeps × 7,267,018,752 B | `RUN/training/training.json: timings[].weight_bytes_copied`; T11 `streamed_bytes_per_pass` | measured |
| Table 1: generation 2.24 s, old log-probs 3.52 s, update 3.76 s, 9.55 s mean per update | `RUN/training/training.json: timings` | measured |
| predicted times 2.42 / 6.23 / 8.65 s at 12.0 GB/s | RES "Time per GRPO step"; PRED | predicted |
| control 2.668, smoke 2.307, profile 2.057 GiB per process | `MAN/{control,smoke,profile}_*/guarded_run.json` | measured |
| 5947 samples, median gap 0.56 s; backward and optimizer windows shorter than one sample | `RUN/training/memory/*.json`; RES (backward and optimizer rows with no sample) | measured |
| CUDA context 0.17 GB = per-process minus reserved at peak | `RUN/training/memory/*.json: peak` | measured |
| profiled merge: anonymous growth 0.32 GB | T11 `profile.phases[merge].cpu_anon_peak` minus the preceding phase | measured |
| most of the merge RSS is file-backed mmap pages | from the two rows above | **inference** |
| 12.23 GB/s measured on 2026-11-18; 7.25 GB/s on 2026-11-19, **on different hosts** (2026-11-18: Linux 5.4, 540.9 GB RAM; 2026-11-19: Linux 7.0, 135.0 GB RAM) | RES (2026-11-18 measurement); T11 `h2d_bandwidth_bytes_per_s`; T11 `transfers.*.transfer_share_of_wall` > 1; `RUN/training/training.json: environment` | measured; the cause is unresolved (host difference is one untested candidate) |
| guard uses exactly 8×10⁹ B | `RUN/training/memory/*.json: budget_bytes`; `RUN/config.yaml: memory.budget_bytes` | measured |
| **two guard tests failed on the rented host** (`test_a_sample_above_the_budget_stops_the_run`, `test_the_guard_records_a_violation_and_interrupts_the_main_thread`): 2 failed / 341 passed / 17 skipped of 360 | `STG/correctness/20261119T050522Z/pytest_cpu.xml`, `…/pytest_cpu/stdout.log`; `STG/state.json` (correctness FAIL) | measured |
| both tests pass on the author's laptop at the same code | rerun on 2026-11-19 (`.venv/bin/python -m pytest tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run tests/test_profiler.py::test_the_guard_records_a_violation_and_interrupts_the_main_thread` → 2 passed); not archived as a file | measured, not archived |

## 2. L2L, streaming and DFlash

| claim | source | status |
|---|---|---|
| pinned slabs, side copy stream, ready/free events, buffer ownership check | `qwen_grpo/execution/weight_store.py`, `prefetch.py` (`_claim`), `layer_stream.py` | code |
| backward re-streams layers 35 and 34 with an exact recompute checksum | `qwen_grpo/execution/backward.py` | code |
| 10 training sweeps per update; layer-major would need 2 | `RUN/training/training.json: timings` (13–14 sweeps); micro-batch loop in `qwen_grpo/training.py` | measured / inference |
| H01 2690, H02 5296, H05 6528 comparisons, all bitwise; H04 1292 within bound | `evidence/20261119T051124Z_H01.lab_async_cuda/results.json`, `…051240Z_H02…`, `…051445Z_H05…`, `…051351Z_H04…`; `research/l2l/experiments/t08_*_r20261119T05*Z/summary.md` | measured |
| H04 bound 8.7×10⁻³, largest error 9.8×10⁻⁴ | `research/l2l/experiments/t08_generation_equivalence_async_r20261119T051351Z/summary.md` | measured |
| Qwen3-0.6B, 100 steps: resident and synchronous L2L give identical rows | `research/l2l/experiments/compare_qwen_resident_vs_qwen_l2l.json` | measured (historical, other GPU) |
| H01 history: 2 cross-host fixture comparisons failed (1.3e-7, 2.3e-7) on 2026-11-18; the gate became same-host | commit `5680e71` message; the fixture drift is still reported as a diagnostic in the `t08_forward_equivalence_async_r20261119T051124Z` summary | measured |
| 146 injected unsafe buffer reuses refused (66 + 80) | `research/l2l/experiments/t08_forward_equivalence_async_unsafe_reuse_r20261119T051215Z/summary.md`, `t08_gradient_equivalence_async_unsafe_reuse_r20261119T051321Z/summary.md` | measured |
| sync vs async speed **not measured** on 4B (H06 untested) | no H06 result in `research/l2l/experiments/`; `STG/state.json` (profile stage BLOCKED) | measured (absence) |
| 1.15× the transfer bound (this run); 1.01× (60-update run) | 9.55 s / 8.32 s; RES | measured |
| 672 rollout batches written to disk with an index | `RUN/streaming/rollouts/index.jsonl`; `RUN/training/training.json: groups_generated` | measured |
| tiny CPU resume: 6 crash points and an external SIGKILL pass | `research/l2l/experiments/t09_resume_tiny_cpu_r20261119T051532Z.json`, `t09_resume_kill_tiny_cpu_r20261119T051046Z.json` (+ `_raw/`) | measured |
| **earlier failed attempt:** the first local external-kill test failed on `final_rng_state_equal` (all other checks passed); the rerun at 040414Z passed | `research/l2l/experiments/t09_resume_kill_tiny_cpu_r20261119T040250Z.json` (committed), `t09_resume_kill_tiny_cpu_r20261119T040414Z.json` | measured |
| **real Qwen3-4B SIGKILL/resume** matches the uninterrupted run (rows, records, samples, adapter, AdamW, RNG) | `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json: checks, pass`; raw logs and nvidia-smi in `…_raw/{A,B1,B2}/`; runs `runs/qwen3_4b_resume_kill_kill{A,B}/` | measured |
| merge: 6 adapted tensors within half a bf16 ulp; merged checkpoint not kept | `MAN/merge_*/stdout.log` ("all within half an ulp: True"); T13 `merge`; `STG/merged/` is empty | measured |
| DFlash implemented, CPU tests only; not run on 4B; `dflash.enabled: false` | `qwen_grpo/execution/dflash.py`, `tests/test_dflash.py`, `tests/test_dflash_training.py`; `RUN/config.yaml`; `STG/state.json` preflight | code / measured |
| DFlash could save at most 2–3 of 13–14 sweeps (14–21% of transfer time) | sweep counts above | inference |

## 3. Evidence that training occurred

| claim | source | status |
|---|---|---|
| ‖θ − θ₀‖ = 2.00; adapter sha256 differs from initialization | `RUN/training/training.json: training[-1].change_from_init`; sha256 of `RUN/checkpoints/step{0,150}/adapter_model.safetensors` | measured |
| 56 changed test predictions | T13 `predictions.base` vs `predictions.trained` | measured |
| 0 of 398 frozen tensors changed | `RUN/training/training.json: frozen_changed` | measured |
| \|log π_θ − log π_old\| = 0 before all 150 updates; ratio ≡ 1; clip 0; \|L\| ≤ 3.2×10⁻⁸ | `RUN/training/training.json: training[].init_diff, ratio_min/max, clip_frac, loss` | measured |
| Table 2: the four policies (accuracy, macro-F1, AUC, invalid count, POS rate, recall) | T13 `models`; recomputed from T13 `predictions`, `label_scores` and `data/sentiment.json` | measured |
| McNemar p = 0.22 (22 vs 32); detectable effect ≈1.35 pp | recomputed from T13 `predictions` | measured |
| merged vs trained: 1066/1066 identical predictions; scores within 4×10⁻⁵ on 633 (exactly equal on 400); 433 differ by 0.0625–0.375 nats, multiples of 1/16 | T13 `label_scores.trained`, `label_scores.merged` | measured |
| the merge differences are consistent with bf16 rounding | 1/16 nat is one bf16 spacing for \|s\| in [8, 16) and half of one in [16, 32); max \|s\| ≈ 26 | **inference**; the cause is not isolated |
| control: step-30 adapter byte-identical to its step 0 and to the trained run's step 0 | sha256 of `CTL/checkpoints/step{0,30}/adapter_model.safetensors`, `RUN/checkpoints/step0/…`; T13 `lora_change.control`, `control_log` | measured |
| score shift +6.01 ± 0.88 nats; Spearman 0.9985 | recomputed from T13 `label_scores` | measured |
| 60-update comparison: +0.38 pp | historical; the 60-update artifacts were removed from this tree and are archived in `~/soup-assignment-results.tar.gz` (`research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain.json`) | measured (historical) |
| logged reward 0.86 → 0.73 (successful steps); all-rollout reward 0.86–0.92 per 50-batch bin over 20,160 rollouts | `RUN/training/training.json: training[].mean_reward`; `training` + `zero_signal` rewards | measured |
| 163 of 3360 groups mixed; 522 of 672 batches with zero signal; 55% of 3211 s outside updates | `RUN/training/training.json` | measured |
| Figure 2 | `reports/figures/make_figures.py` → `reports/figures/fig_memory_scores.svg` (from the RUN memory file and T13) | measured |

## 4. Silent failures and Soup coverage

| claim | source | status |
|---|---|---|
| reward stress: 0 of 4600 attacks accepted; the planted permissive verifier was flagged as gameable (800/800 answer spray) | `SOUP/reward_stress.json`, `SOUP/reward_stress_permissive.json`, `SOUP/summary.json` | measured |
| `soup ship`: DON'T SHIP, `failed_rule: missing_baseline`, for both the trained model and the control | `SOUP/ship_t13_150_update_verdict.json`, `SOUP/ship_t13_150_update_lr0_control_verdict.json` | measured |
| 0.77.2: `soup ship` said SHIP on +0.09 pp once a benchmark was supplied; data doctor rated "no assistant turn" the same as clean data; adapters check flagged an all-zero adapter | `research/soup/0.77.2_cpu/ship_insignificant_gain_verdict.json`, `data_doctor_fault_no_assistant.json`, `adapter_check_step0.log`; `docs/SOUP_CHECK_COVERAGE.md` | measured (historical, CPU, Qwen3-0.6B artifacts) |
| data doctor (7 fixtures) and adapters check (4 checkpoints) did not run: missing `soup-cli[train]` (transformers / numpy) | `SOUP/data_doctor_*.log`, `SOUP/adapters_check_*.log` | measured |
| Soup pre-flight (`advise`, `plan`, `train` hardware-fit) was not run | `SOUP/summary.json` (status NOT_RUN) | measured |
| `soup doctor` ran in a different interpreter from the training environment | `SOUP/summary.json: checks[doctor].same_interpreter_as_training` | measured |
| stale old log-prob assertion; micro-batch scaling | `qwen_grpo/training.py`; `tests/test_grpo.py`, `tests/test_multistep.py`; H05 above | code / measured |
| external supervisor tests passed on the host | `STG/correctness/*/pytest_cpu.xml` (`tests/test_guarded_run.py`) | measured |

## 5. Verdict, surprises, provenance

| claim | source | status |
|---|---|---|
| staged runner: preflight PASS, correctness FAIL (only `pytest_cpu`), smoke and profile BLOCKED; stages then run by hand under the wrapper | `STG/state.json`; `MAN/` | measured |
| 119 of 137 commits co-authored by Claude | `git log --format=%B \| grep -c "Co-Authored-By: Claude"`; `git log --oneline \| wc -l` | measured |
| every number recomputed by a separate script | `reports/verify_report_numbers.py` (93/93) | measured |

## Known corrections relative to the earlier drafts

- The 60-update report said "host 504 GiB RAM". The 150-update run's host reports `mem_total_bytes` 135,040,708,608 (`RUN/training/training.json: environment.host`), so the header no longer states host RAM.
- T13's top-level `peak_gpu_allocated_bytes_eval` (1,195,164,160 B) is not a peak over the evaluation: `scripts/final_eval.py` resets the peak counter after each model's pass, so the field reflects a single pass. The per-model peaks are in T13 `phases` (1,567,870,464 B; 1,563,545,088 B for the merged model). The report quotes the per-process nvidia-smi figure for evaluation (1912 MiB).
- `research/final_audit/analyze.py` reproduces the accuracy CI with a different seed and resample count ([−0.47, +2.35] pp vs T13's [−0.38, +2.25] pp). The conclusion (the CI includes 0) does not change.
