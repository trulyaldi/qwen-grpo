# Hard Track v5: requirements to implementation

This matrix comes from an audit of the repository at `8859e53`, made before `scripts/run_assignment.sh` existed. It maps
every clause of `Soup-Task-01-AI-Engineer-v5.pdf` to the code that implements it, the test or experiment that checks
it, the command that runs it, where its output lands, and the gap that remained. `scripts/assignment_report.py` judges
each row as `requirements.json` / `reports/requirements_matrix.md` in every `assignment_results/<run-id>/`.

Abbreviations:
- **study:** `configs/studies/assignment.yaml`.
- **X:** `scripts/experiment.py run --entry`.
- **exp/:** `research/l2l/experiments/`.

| id | assignment clause | implementation | test / experiment | command | output | gap at audit → closed by |
|---|---|---|---|---|---|---|
| T1 | Qwen3-4B, non-thinking, no SFT, GRPO + LoRA, verifiable reward, 1–2k prompts | `configs/experiments/qwen3_4b_final.yaml` (1024 prompts); `sentiment.build_prompts` (empty think block); `sentiment.reward` (exact label) | `tests/test_sentiment.py`, `tests/test_policy.py`, `tests/test_profiles.py` | `run_assignment.sh` stage A (static config check) | `requirements.json` T1 | not checked against the resolved config at run time → stage A checks it |
| M1 | training ≤ 8 GB as shown by nvidia-smi, CUDA context included | `memory.guard` (`execution/profiler.py`, 8 **GiB**, per-process nvidia-smi else cudaMemGetInfo) | `tests/test_memory_guard.py`, `tests/test_profiler.py`; study H11.smoke, H12.train_final | X H11.smoke, H12.train_final | `runs/<run>/training/memory/*.json` | the guard compares with 8 GiB (8.59e9 B), not 8 GB (8e9 B); final_eval has no guard; per-process attribution only for the guard's own PID → external 200 ms nvidia-smi telemetry with process-group attribution and an 8e9 B watchdog (`qwen_grpo/telemetry.py`) |
| M2 | rollout generation ≤ 8 GB | same guard (rollouts happen inside training); `profile_run.py` rollout phase | H11.profile | X H11.profile | `exp/t11_profile_qwen3_4b_async.json` | same as M1 |
| M3 | merge ≤ 8 GB | `export/merge.py` (CPU, tensor by tensor) | H09 lab merge; profile merge phase | X H09.*, H11.profile | `exp/t10_*`, profile JSON | no nvidia-smi evidence for the final adapter's merge → stage G telemetry covers `final_eval.py` (merge + merged evaluation) |
| M4 | merged-model evaluation ≤ 8 GB | `final_eval.py` → `load_merged_l2l` | — | X H12.final_eval | `exp/t13_final_eval_*.json` | no guard, no nvidia-smi → stage G telemetry + watchdog |
| P1 | memory and time budget *before* running | `qwen_grpo/budget/`, `scripts/budget.py` | `tests/test_budget.py`, `tests/test_assignment_provenance.py` | `budget.py --config configs/budget/qwen3-4b_final.yaml` | `results/budget/predictions/qwen3-4b-final_8gb.*` | not tied to a run, and the Step-0 prediction `qwen3-4b_8gb` described another configuration (micro-batch 6 vs 30, GPU vs CPU merge, 512 vs 128 eval examples, 6.7 GiB host) → versioned `qwen3-4b_final.yaml` checked against the resolved final config; historical file kept unchanged; stage A copies and recomputes both into a read-only `predictions/` |
| P2 | predicted vs measured peak: allocated, reserved, nvidia-smi | `profile_run.py --prediction` | H11.profile | X H11.profile | profile JSON `prediction.rows` | not tabulated → `comparisons`/`measurements/memory.csv`, `reports/memory_analysis.md` |
| P3 | step time from PCIe bandwidth vs measured | `profile_run.py` transfers + `h2d_bandwidth` | H11.profile | same | profile JSON `transfers` | same → `measurements/throughput.csv` |
| P4 | resident vs streamed layer split, justified | budget report; `resident_layers: 0` (more is refused as unimplemented) | `tests/test_config.py` | — | `predictions/` | documentation only (not measurable) |
| L1 | own L2L: pinned CPU weights, current + prefetched layer on the GPU, boundary checkpoints, copy/compute overlap on a side stream | `execution/weight_store.py`, `prefetch.py`, `backward.py`, `qwen3_forward.py` | tiny lab cases H01–H05, H07 | X H0x.* | `exp/t0*` | — |
| L2 | logits equal a non-L2L reference (justified tolerance) | lab `forward_equivalence`; tolerances 0 (bitwise) | H01.lab_sync, H01.lab_async_cuda + faults | X H01.* | `exp/t03_*`, `t08_forward_*` | CUDA evidence predates later executor changes; async on CUDA never run → stage B re-runs all of them on the target GPU |
| L3 | gradients equal the reference | lab `gradient_equivalence` | H02.* | X H02.* | `exp/t04_*`, `t08_gradient_*` | same |
| L4 | L2L also for rollouts and evaluation | `execution/generator.py`, `make_evaluation_backend` | H04.*; H05.real_0.6b_bf16 (whole runs bitwise) | X H04.*, H05.real_0.6b_bf16 | `exp/t06_*`, evidence compare JSON | 0.6B bf16 comparison never run on an Ampere GPU → stage C |
| L5 | full GRPO step equivalence, sync vs async | lab `update_equivalence`, `compare_runs.py` | H05.* | X H05.* | `exp/t07_*` | async never run → stage B/C |
| L6 | the GRPO step's split between generation and training, and why | trainer `timings` (rollout_s, prepare_s, update_s); profile phases | — | — | `training.json` | not summarised → `measurements/timing.csv` |
| L7 | prefetch performance / bounded residency | lab `prefetch_timing`, `depth_residency` | H06, H07 | X H06.*, H07.* | `exp/t08_prefetch_timing`, `t11_depth_residency` | never run → stages B/C |
| S1 | streaming from disk, deterministic resume: kill, resume, no duplicates, no gaps | `streaming/`, `resume_experiment.py` (six crash points) | H08.resume_tiny_cpu (PASS), H08.resume_qwen3_0.6b | X H08.* | `exp/t09_resume_*.json` | 0.6B on GPU never run → stage C; the final run's own resumes are audited from its rollout index (stage report) |
| S2 | merge tensor- or shard-by-shard, never the full model on the GPU, show the peak | `export/merge.py` | H09.merge_tiny_cpu, H09.merge_qwen3_0.6b | X H09.* | `exp/t10_*` | real-model variant never run; peak at 4B only in the profile → stages C, D, G |
| D1 | drafter `z-lab/Qwen3-4B-DFlash-b16`, resident, hidden features from target layers inside the L2L loop | `execution/dflash.py` | `tests/test_dflash.py` (tiny) | X H10.dflash_bench | `exp/t12_dflash_bench.json` | never run at 4B → stage E |
| D2 | verifier runs the current LoRA; greedy token-identical to plain decoding | `DFlashGenerator` | dflash_bench `greedy_equal` per policy | same | same | — |
| D3 | sampling distribution check | exact law test on tiny models (`tests/test_dflash.py`) | — | — | — | no 4B check → dflash_bench keeps the sampled responses of both arms (`--repeats`), and the report runs a two-sample permutation test |
| D4 | acceptance length | `accepted_per_pass` | dflash_bench | same | same | single measurement → `--repeats` with bootstrap CIs |
| D5 | GRPO step time with and without DFlash | dflash_bench `grpo_step_*` (one step each) | — | — | — | no end-to-end DFlash GRPO *run* → study entry `H10.smoke_dflash` (`qwen3_4b_smoke_dflash.yaml`, same batch as `qwen3_4b_smoke`) compared step by step |
| D6 | acceptance vs policy drift | `--b-scales` (synthetic drift) / `--adapters` | — | — | — | no real checkpoints → study entry `H10.dflash_drift` on the final run's checkpoints |
| V1 | held-out improvement with CIs vs base | `final_eval.py` paired bootstrap | `tests/test_verification.py` | X H12.final_eval | `exp/t13_final_eval_qwen3_4b_final.json` | — |
| V2 | merged matches adapter up to numerical noise | `final_eval.py`: prediction agreement, merged−trained CI, half-ulp merge check | — | same | same | label-score discrepancy not reported → report computes max/mean \|Δscore\| |
| V3 | negative control (lr = 0) | `qwen3_4b_final_lr0.yaml` (30 steps, not 210) | trainer LR=0 control | X H12.train_final | `runs/qwen3_4b_final_lr0` | shorter schedule must be stated → the report states it |
| V4 | trained vs only reward gaming; what the check would not detect | invalid rate, predicted-POSITIVE fraction, per-class recall, AUC | — | — | same | not judged → V4 row; the limitations are written into the report |
| F1 | silent failures along data → … → inference: what, how checked, evidence | `diagnostics/invariants.py`, `tests/silent_failures/`, lab `--perturb` faults, `final_report.AUDIT` | fault-injection entries in the study | stages B, H | `reports/silent_failures.md` | the matrix was static text → each row is now joined to that run's fault-control outcomes |
| F2 | whether Soup's data doctor, pre-flight and `soup ship` catch each failure | Soup integration was removed in `9ca0390` (archived run 0.77.2 CPU) | — | stage H, if `soup` is installed | `raw/soup_checks/` | → stage H runs `soup data doctor` / `soup adapters check` when Soup is installed; otherwise NOT_TESTED with the reason |
| X1 | verdict SHIP / DON'T SHIP with what must change | `final_report.py` (repository-wide) | — | — | `research/l2l/final_report.md` | → `reports/final_verdict.md`, derived from this run's evidence |
| X2 | raw nvidia-smi during the whole run | `experiment.py` SmiSampler (1 s), profiler samples | — | — | `evidence/*/raw/` | 1 s, not per phase, no attribution → 200 ms per item, raw CSV, process-group attribution |
| X3 | raw timestamped logs, report ≤ 3 pages, AI-use note, surprise/concern paragraph | trainer `tee` timestamps | — | — | — | → `reports/summary.md` (three-page draft) with sections the researcher must write |

## Output of one run

```text
assignment_results/<run-id>/            run-id = UTC start time; never reused, never overwritten
  manifest.json                         commit, tracked-diff hash, untracked files, study and config hashes, the cap
  state.json                            every stage attempt and item: argv, UTC times, exit, status, telemetry summary
  requirements.json                     every clause above: status, detail, evidence, measured values; the verdict
  resolved_configs/                     each config resolved by the trainer's loader, with its sha256
  environment/                          capabilities probe, host RAM/disk, nvidia-smi identity, blockers, models/data/seeds
  predictions/                          budget predictions (committed copy + recomputed), read-only before stage C
  raw/<preflight|correctness|l2l|memory|dflash|training|evaluation|soup_checks>/stage<S>_attempt<n>/<item>/
                                        stdout.log, stderr.log, telemetry/{nvidia_smi_apps.csv, nvidia_smi_gpu.csv,
                                        *.stderr, pids.csv, telemetry.json}
  measurements/                         memory.csv, memory_summary.json, timing.csv, throughput.csv,
                                        correctness.json, dflash.json, learning.json
  comparisons/                          l2l_equivalence.csv, backend_performance.csv, dflash_vs_plain.csv, model_artifacts.csv
  reports/                              summary.md (three-page draft), requirements_matrix.md, silent_failures.md,
                                        memory_analysis.md, experimental_results.md, report_tables.md, limitations.md,
                                        final_verdict.md
  logs/orchestrator.log                 every line UTC-stamped
  checksums.sha256
```

Raw directories are named per stage. Some names differ from the task's suggested layout:
- `memory/` is stage D, the Qwen3-4B qualification.
- The 0.6B streaming/resume and merge cases are in `l2l/` (stage C) and `correctness/` (stage B).
- The LR=0 control is trained by the same dispatcher entry as the final run, so its logs are in `training/`.
- The final adapter's merge happens inside the held-out evaluation process, so its logs are in `evaluation/`.

There is one telemetry stream per process, so a merge or control item cannot be separated from its parent. Large
outputs are not copied. Training runs stay in `runs/<name>/` and lab results in
`research/l2l/experiments/`; the dispatcher's own record of each execution is in `evidence/<UTC>_<entry>/`. The state
links to all of them.

## Provenance rules

- **Experiment identity.** Resume requires the same commit, study and resolved configs, plus the same
  `experiment_code_sha256`. That hash covers the content of every tracked and untracked, not ignored, file under
  `qwen_grpo/`, `scripts/`, `configs/`, `data/`, `tests/`, the hypotheses and tolerances, and the requirements
  (`qwen_grpo/evidence.py`). A resume against changed experiment code is refused.
- **Report identity.** `report_code_sha256` covers `scripts/assignment_report.py` alone. When only it changes, a resume
  re-runs no experiment and regenerates the report. `--status` flags a report written by other report code.
- **Reuse of existing outputs** that this run did not produce. This applies to the fixed-name Qwen3-4B results and
  training runs:
  - Reuse needs the evidence's own record of its GPU model, its experiment-code hash and, for runs, its config hash.
  - The held-out evaluation must also match field by field: the adapter sha256 of the trained and control
    checkpoints, the final config hash, the data file sha256, the split and its size, the code hash, the cached
    base-model revision, and no overrides (`final_eval.py` writes them under `identity`).
  - Otherwise the item is INCONCLUSIVE and nothing depending on it starts.
- **Resume of training runs.**
  - Only streaming runs are resumed: `--resume latest` from their crash-consistent (`COMMIT.json`) checkpoints, and
    only if this assignment run started them.
  - Non-streaming runs (the three Qwen3-0.6B equivalence runs) publish no `COMMIT.json` (`trainer.save`). Their
    resume is not crash-consistent and not covered by H08. An interrupted one is BLOCKED and must be moved aside and
    retrained.
