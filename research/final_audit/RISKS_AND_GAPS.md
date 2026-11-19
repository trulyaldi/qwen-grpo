# Risks and gaps (audit, 2026-11-19)

Scope: the 150-update run `qwen3_4b_final_plain_150_v2` at commit `29852d1` (artifacts listed in FINAL_VERDICT.md). Items closed since the audit of the 60-update run (`20261118T171404Z`) are listed at the end.

Priorities:
- **P0:** invalidates a mandatory requirement or a critical claim as currently evidenced.
- **P1:** weakens the main conclusions or reproducibility.
- **P2:** an important limitation.
- **P3:** minor.

Each item gives the shortest action that resolves it. "Report-only" means the issue is fixed by stating it accurately, without new runs.

Abbreviations: FS = `assignment_results/final150_v2`, MAN = `assignment_results/manual`, t13 = `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json`.

## P0: blocking for the claims they touch

| id | issue | evidence | implication | shortest resolution |
|---|---|---|---|---|
| P0-1 | **DFlash rollouts (task 2c) were never run on the real model.** Excluded on purpose (`a1d59ea`); `dflash.enabled: false` in every `*_150_v2` config, checked by the preflight. | FS `state.json` preflight `config_150_v2`; `docs/REMAINING_LIMITATIONS.md`; `docs/l2l/12_dflash.md` "Not established here" | a full scored sub-task (part of the 30% engineering score) is not demonstrated | Either run `scripts/dflash_bench.py` once (greedy equality, acceptance, step time with and without DFlash, drift at B scales) or disclose it with the Amdahl bound from RESULTS_ANALYSIS §6.1 / REQUIREMENTS D5 (rollouts are 23.5% of a timed step). The assignment accepts honest non-completion. |
| P0-2 | **H12 (the learning claim) is NOT SUPPORTED by its pre-registered criterion.** Δ accuracy +0.938 pp, CI [−0.375, +2.251]; McNemar p = 0.22 (22 vs 32 discordant). | t13 `paired`; `hypotheses.yaml: H12` | any statement that GRPO made the 4B model more *accurate* is unjustified; the SHIP case fails. A small ranking gain (AUC +0.0012, CI [+0.0002, +0.0022]) can be claimed, as a secondary result | Report-only: state the accuracy null, the AUC gain with its CI, and a DON'T SHIP verdict. Do **not** train longer just to chase significance. |

## P1: major

| id | issue | evidence | implication | resolution |
|---|---|---|---|---|
| P1-1 | **The staged runner's correctness gate FAILED, so smoke, profile, training, control, merge and eval were run by hand.** The only failing item was `pytest_cpu` (2 of 360). Each manual stage ran under the guarded wrapper with nvidia-smi telemetry and the 8e9 B cap. | FS `state.json` (correctness FAIL; smoke and profile BLOCKED); MAN `*/guarded_run.json` (argv, exit 0, violation None) | no single orchestrated requirements matrix or verdict exists for this run; every status is the auditor's judgement | Report-only: say plainly that the gate failed, on what, and that the remaining stages were run by hand at the same commit and config hashes (config sha256 `568afd24…` in both the preflight and t13) |
| P1-2 | **The in-process guard's stop-on-violation was not demonstrated on the Soup host.** `tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run` (DID NOT RAISE SystemExit) and `tests/test_profiler.py::test_the_guard_records_a_violation_and_interrupts_the_main_thread` (DID NOT RAISE KeyboardInterrupt) failed there; both pass locally at the same code. The external supervisor's tests (`tests/test_guarded_run.py`, 10) passed on the Soup host. | `FS/correctness/20261119T050522Z/pytest_cpu/stdout.log`, `pytest_cpu.xml` | "the cap is enforced" rests on the external supervisor on that host; the real runs never needed enforcement (0 violations, peak ≈36% of the cap) | CPU, minutes: re-run the two tests on the Soup host image with `-x -vv` and find the host-specific cause (INFERENCE: a timing or signal-delivery difference in the container). Otherwise disclose |
| P1-3 | **Copy/compute overlap and any async speed-up are not demonstrated.** H06 untested; no event timeline. The new profile's H2D bandwidth (7.25 GB/s) contradicts the training step times (14 sweeps would take 14.03 s vs a 9.55 s mean step); the 2026-11-18 figure was 12.23 GB/s. | `hypotheses.yaml: H06`; `MAN/profile_20261119T054235Z/results/t11_profile_manual.json` (`h2d_bandwidth_bytes_per_s`, `transfers`); `results/budget/residuals/qwen3-4b-final_150_v2.md` | the assignment says copy and compute "overlap"; only the mechanism exists. "Transfer-bound" is an INFERENCE that depends on which bandwidth figure is right | Small GPU run: a few steps of `qwen3_4b_l2l_sync.yaml` vs `qwen3_4b_l2l.yaml`, or the H06 `prefetch_timing` case (`research/l2l/configs/async_qwen3_4b.yaml`), with a repeated bandwidth measurement. Or report as unmeasured, giving both bandwidth figures without choosing |
| P1-4 | **Soup's checks ran only in part.** `data doctor` (clean + 6 faults) failed 7/7 with "needs transformers to load a tokenizer — install soup-cli[train]"; `adapters check` failed 4/4 with "numpy package required"; `doctor` ran in a different interpreter from training. Reward stress and `soup ship` did run. | `research/soup/0.78.0_20261119T073222Z/summary.json` and per-check logs | F2 ("whether Soup's built-in checks catch it") is answered for reward hacking and the ship gate only; nothing is known about what `data doctor` and `adapters check` detect | CPU, minutes: install `soup-cli[train]` in the environment Soup runs in and re-run `scripts/soup_checks.py` (or `scripts/final150.py soup`). Otherwise report the missing dependency as the cause, not as a result |
| P1-5 | **The reward / task choice gave almost no signal on 4B.** 4.85% of groups mixed (163 of 3360); 77.7% of rollout batches zero-signal; rollout reward flat; the base is already 87.7% accurate. | `analysis_output.json: trained`; RESULTS_ANALYSIS §1.3, §2 | the assignment asks to "pick a reward that can move within a few dozen steps and justify it"; on 4B it barely did | Report-only: justify the choice (it moved on 0.6B, +6.4 pp CI [4.2, 8.5]) and present the small 4B effect as the finding |
| P1-6 | **The 150-update run is a second look after the 60-update null.** Its evaluation plan was written before it ran (`docs/FINAL_150_STEP_EXPERIMENT.md`, commit `5680e71`), but the decision to run it followed the first result, and the AUC gain is not the H12 criterion. | `git log` (`5680e71` after `1dc2be8`); `hypotheses.yaml: H12` | a reader could see the significant AUC as the result of searching (forking paths) | Report-only: state both runs, which was planned when, and that accuracy (the pre-registered criterion) is still not significant |
| P1-7 | ~~The draft report has a wrong host figure and a stale evidence map.~~ **Closed 2026-11-19:** the report was rewritten for the 150-update run (no host-RAM figure in the header), `reports/REPORT_EVIDENCE_MAP.md` was regenerated against the `*_150_v2` paths, and `reports/verify_report_numbers.py` passes 93/93. | `reports/Soup_Final_Research_Report.html`; `reports/REPORT_EVIDENCE_MAP.md` | — | Done |

## P2: moderate

| id | issue | evidence | resolution |
|---|---|---|---|
| P2-1 | **The trained model's change is mostly a uniform bias shift** (+6.01 nats toward POSITIVE, SD 0.88; Spearman 0.9985 with base), with a small real ranking gain (AUC +0.0012, CI excludes 0). A naive "it learned sentiment much better" reading is wrong. | t13 `label_scores`; RESULTS_ANALYSIS §4.2 | Report-only; this is a good "what the check would not detect" point: accuracy gains can come from threshold shifts |
| P2-2 | **The LR=0 control equals base by construction** (B = 0, lr = 0). It ran 30 rather than 150 updates. It shows eval determinism, not reward causality. | t13 `control_log`; RC config | Report-only. Optional: a shuffled-reward run (not implemented) |
| P2-3 | **PPO diagnostics are structurally uninformative.** ratio = 1, clip 0, approx-KL 0 at every one of the 150 steps, because they are measured before `optimizer.step()` with `num_iterations = 1`. | `training.py:132,139-141`; `analysis_output.json: trained.ratio_*`, `clip_frac_*`; RESULTS_ANALYSIS §1.1 | Report-only: do not cite them as evidence of small policy drift. Use ‖Δθ‖ (2.00) and the label-score shift instead |
| P2-4 | **The profile's bandwidth-derived transfer shares are not credible.** With the 7.25 GB/s figure, predicted transfer time exceeds wall time for old log-probs (share 1.40), forward (1.38) and evaluation (1.19). The byte counter itself is now correct (prefetches counted since `5680e71`; `tests/test_h2d_accounting.py` passes). | `t11_profile_manual.json: transfers`; `scripts/profile_run.py:51-55` | Do not cite the profile's transfer shares or its bandwidth as fact. Use `training.json: timings` (13–14 sweeps per step) and give both bandwidth figures (P1-3) |
| P2-5 | **Async GPU equivalence was tested only with `verify=True`** (host sync per layer use). Production uses `verify=False`. The fault injection only exercises the host-side `_claim` guard, not a missing event wait. | L2L sub-audit; `prefetch.py:89-106,188-192` | Report-only. Optional: run H02/H05 with `verify=False` on GPU (minutes) |
| P2-6 | **Async equivalence only on tiny random models.** Real-model bitwise evidence exists only for `l2l_sync` on Qwen3-0.6B (stage C was not re-run for this run). | inventory §3; `research/l2l/experiments/compare_qwen_resident_vs_qwen_l2l.json` | Report-only |
| P2-7 | **The merge check `all_within_half_ulp` is a per-tensor rounding self-check.** The model-level match rests on predictions (1066/1066) and score diffs (median 9.5e-7, max 0.375 nats) with no pre-declared tolerance. | `export/merge.py:132-138`; t13 `merged_vs_trained_score_abs_diff` | Report-only: state the observed score-diff distribution and that it is bf16-rounding-scale (INFERENCE) |
| P2-8 | **Telemetry has no phase labels, and the planned per-model eval slice was not written.** The training memory file has `phases_labelled` 0; backward and optimizer phases (< 0.1 s in the profile) mostly got no sample. No `phase_memory.json` exists for the eval stage; the merged-pass peak (1912 MiB) is the auditor's own slice of `MAN/eval_20261119T070439Z/telemetry/nvidia_smi_apps.csv` using t13 `phases`. All nvidia-smi peaks are lower bounds (200 ms sampling). | `analysis_output.json: memory_trained.phases_labelled`; MAN eval dir | Report-only; the allocator plateau argument (RESULTS_ANALYSIS §5.1) |
| P2-9 | **Residency choice unexplained.** The calculator recommends 20 resident layers; the run used 0. That makes the run maximally transfer-bound (13–14 × 7.27 GB per step). | prediction md "recommended: 20" | Report-only: justify the choice, or present it as the main throughput lever |
| P2-10 | **Validation subset n = 128 cannot resolve trends.** SE ≈ 0.028; 0.859 at step 0 → 0.898 at step 150 is 1 vs 6 discordant (exact p = 0.125); the curve plateaus around 0.89 from step 30. | `analysis_output.json: trained.val_subset`, `val_subset_step0_vs_last_discordant` | Report-only |
| P2-11 | **The merged checkpoint was not kept.** `FS/merged/` is empty. The adapter sha256 (t13 `identity`) and the sha256 of every merged shard (t13 `merged_dir.provenance.shards_sha256`) are recorded, and the merge is deterministic from them (INFERENCE). | `ls assignment_results/final150_v2/merged/`; t13 | Report-only: say it was not kept and cite the hashes |
| P2-12 | **Training steps are ~1 s slower than in the 60-update run** (mean timed step 9.55 s vs 8.57 s), unexplained. 55% of train time (1779 of 3211 s) is outside timed successful steps (zero-signal batches). | `analysis_output.json: trained.mean_step_s`, `seconds_not_in_timed_steps`; the 60-update value from the previous `analysis_output.json` (not kept in the repo) | Report-only: give both numbers; do not attribute the difference without a measurement |

## P3: minor

| id | issue | resolution |
|---|---|---|
| P3-1 | `training.json` field names: `groups_generated` and `zero_signal_groups` actually count batches | note in the report |
| P3-2 | `evaluation.bootstrap_samples` config ignored (hard-coded 10,000, same value). `analyze.py`'s own resampling gives a slightly different accuracy CI ([−0.469, +2.345] pp vs the reported [−0.375, +2.251]); same conclusion | note; cite the t13 CI |
| P3-3 | Tied embedding/LM head (~0.78 GB) is permanently GPU-resident, beyond "current + prefetched layer" | disclose |
| P3-4 | `hypotheses.yaml` statuses were not updated after the runs (H06, H11, H12 still `untested`) | leave the raw file; report the statuses from this audit |
| P3-5 | The successful-step reward (first 10 = 0.86, last 10 = 0.73) is selection-biased: it is the reward of batches that had a mixed group | use the unbiased all-attempt bins (0.856–0.916, flat) |
| P3-6 | Possible base-model pretraining exposure to Rotten Tomatoes | mention as a limitation; it affects all four models equally |
| P3-7 | No importance correction for sampler-vs-learner numeric mismatch (30-row KV decode vs 6-row teacher-forced pass); standard practice | mention if asked |
| P3-8 | Uniform groups are still forwarded and back-propagated (wasted compute: 4 of 5 groups on 137 of 150 steps, 3 of 5 on the other 13) | optimization idea |
| P3-9 | `docs/SOUP_CHECK_COVERAGE.md` and `docs/REMAINING_LIMITATIONS.md` describe the state before the 150-update run (60-update numbers; merged eval, kill/resume and Soup "not yet run") | update or mark as pre-run; cite `docs/results_20261119_150_v2.md` instead |

## Silent-failure checks for the specific failure modes the brief listed

| risk | found? | basis |
|---|---|---|
| numerical errors masked by reasonable metrics | not found | bitwise equivalence on the RTX 3060 (tiny async) and 0.6B sync; frozen digest 0/398; backward checksum |
| H01 unresolved | closed | 2690/2690 same-host on the RTX 3060 (`evidence/20261119T051124Z_H01.lab_async_cuda`); cross-host drift kept as a non-gating diagnostic |
| incorrect gradient scaling | not found | `training.py:125-126`, n_c/N; CPU test that micro-batch equals full batch |
| detached computations | not found | `allow_unused=False` in the checkpointed backward (`backward.py:76`); grad norm > 0 at every step (min 0.140) |
| incorrect sampling-policy probabilities | not found | old_logp equals the new logp at iteration 0 (assert at 1e-6; ratio 1.0 at every step) |
| incorrect masks | not found | `grpo.py:11-19`; tests |
| reward defects | not found | `sentiment.py:35-42`; 10/20160 invalid rollout answers; Soup reward stress 0/4600 attacks accepted |
| biased or leaking data | not found | 0 overlap (train/val/test), normalized; balanced splits |
| silent eval config change | not found | identical prompts, decoding, backend (`l2l_async`) for all four models |
| false memory claims | low risk: P1-2, P2-8 | all four named phases measured under nvidia-smi at 8e9 B |
| incomplete telemetry | partial: P2-8 | — |
| misread zero-signal batches | no defect; expected (RESULTS_ANALYSIS §2) | observed 0.777 vs 0.780 expected if groups were independent |
| incorrect checkpoint selection | not found | step 150 = max_steps; no selection on test or validation |
| merged-model differences | small, bf16-scale (P2-7) | — |
| missing Soup integration | partial (P1-4) | — |
| unexecuted mandatory experiments | yes: DFlash, overlap | — |
| provenance inconsistencies | staged gate failed, stages run by hand (P1-1); stale docs (P3-9) | — |

## Closed since the 60-update audit

| former id | issue | how it was closed | evidence |
|---|---|---|---|
| P0-1 | merged-model evaluation had no nvidia-smi evidence | the eval (base, trained, merged, control) ran as its own guarded process: per-process peak 1912 MiB over 7295 samples, 0 violations; the merge ran as a separate CPU process with no CUDA context (device 13 MiB) | `MAN/eval_20261119T070439Z/`, `MAN/merge_20261119T070347Z/` |
| P1-1 (part) | manual runs were outside the 8e9 B watchdog | every stage ran under `scripts/guarded_run.py` with cap 8e9 B and 200 ms nvidia-smi sampling | `MAN/*/guarded_run.json` |
| P1-2 | H01 gate failure not closed | H01 PASS on the RTX 3060 host, 2690/2690 (fixed in `5680e71`: same-host gate) | `research/l2l/experiments/t08_forward_equivalence_async_r20261119T051124Z/` |
| P1-3 | in-process guard enforced 8 GiB, not 8 GB | `memory.budget_bytes` = 8,000,000,000 B, checked by the preflight; the training memory file records `budget_bytes` 8000000000 | FS `state.json`; `runs/qwen3_4b_final_plain_150_v2/training/memory/20261119T054525Z.json` |
| P1-5 | kill/resume only on a tiny CPU model, in-process | real Qwen3-4B SIGKILL of the process group, resumed in a new process, all checks equal, `pass: true` | `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json` |
| P1-6 (part) | final-eval command and log not recorded | argv, stdout and stderr recorded | `MAN/eval_20261119T070439Z/guarded_run.json`, `stdout.log`, `stderr.log` |
| P2-4 (part) | profile under-counted async H2D bytes | `executor_bytes` counts prefetches | `scripts/profile_run.py:51-55`; `tests/test_h2d_accounting.py` |
