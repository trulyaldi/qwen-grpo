# The 150-update Qwen3-4B experiment: how to run it

Run name: **`qwen3_4b_final_plain_150_v2`**. Config: `configs/experiments/qwen3_4b_final_plain_150_v2.yaml`
(sha256 of the resolved config `568afd24d338…`; `python scripts/final150.py dry-run` prints the full hash).

## Configuration

The config extends `qwen3_4b_final_plain.yaml`, the 60-update run of 2026-11-18. Only six resolved keys differ from
that run, and `tests/test_final150.py` pins them: `run.name`, `training.max_steps`, `training.max_attempts`,
`evaluation.every`, `evaluation.subset_label_scores` and `memory.budget_bytes`.

| parameter | value |
|---|---|
| model | Qwen/Qwen3-4B, bf16, non-thinking chat template (asserted by `sentiment.build_prompts`) |
| algorithm | GRPO: clipped surrogate with clip 0.2, β = 0, 1 update per rollout batch |
| adapter | LoRA r 32, α 64, q/v/o projections of layers 34–35, fp32 |
| B × G | 5 prompts × 6 completions; micro-batch 6; 4 generated tokens; temperature 1 |
| learning rate | 2e-4, AdamW, weight decay 0 |
| successful updates | **150**; rollout-batch budget `max_attempts` 1200 (see below) |
| validation | the same 128-prompt balanced validation subset at steps **0, 10, 20, …, 150** (16 points): accuracy, macro-F1, recall per class, invalid rate, prediction balance, confusion matrix, label scores, AUC, adapter sha256, UTC timestamp, settings |
| checkpoints | crash-consistent, at every validation step: adapter, AdamW state, RNG, sampler/rollout cursor, counters, the log so far, config and provenance |
| execution | `l2l_async`: pinned store, 0 resident layers, 2 GPU buffers, prefetch depth 1, checkpointed backward |
| streaming | on (`runs/<name>/streaming/`); resume from the latest valid checkpoint |
| GPU budget | **8,000,000,000 B**, CUDA context included: in-process guard + allocator cap, plus the external nvidia-smi supervisor |
| DFlash | **disabled** (out of scope, `docs/REMAINING_LIMITATIONS.md`) |
| test split | used only by the final evaluation, never to select a checkpoint or a hyperparameter |

**max_attempts = 1200.** The 60-update run accepted 60 of 266 rollout batches (22.6%; Wilson 95% interval
17.9–27.9%), so 150 updates need about 665 batches at the measured rate and about 836 at the lower bound. 1200
batches still reach 150 updates at an acceptance of 12.5%. The rate can fall as the policy is right more often on all
six samples, which makes more groups uniform.

If the budget runs out, the run stops with fewer updates and `training.json` says so; the driver then marks the stage
FAIL. Nothing resamples prompts or changes rewards to manufacture updates.

**LR = 0 control: 30 updates.** Config `…_150_v2_lr0.yaml`, with evaluations at 0, 10, 20 and 30. LoRA B starts at 0
and lr 0 never changes a parameter, so the control adapter is bitwise the initial one at any step count. More updates
would cost GPU time without producing a different artifact.

The control tests that the training loop, the checkpoint round trip and the evaluation pipeline attribute no change
to an unchanged model. It does not test reward causality (`docs/REMAINING_LIMITATIONS.md`).

## Estimated memory and transfer workload

Source: `results/budget/residuals/qwen3-4b-final_150_v2.md`. Analytical figures are from the frozen pre-run budget;
calibrated figures use the 60-update measurements.

- **GPU, nvidia-smi per process:**
  - 60-update run peak: 2.67 GiB (2,864,709,632 B).
  - Analytical maximum: 2.43 GiB.
  - Cap: 7.45 GiB (8e9 B), leaving 4.78 GiB of headroom.
  - The per-step shapes are unchanged. The new subset label scores add at most one fp32 [8, ≤102, 151,936] logits
    tensor (~0.5 GB) during evaluations.
- **Host RAM:** peak RSS 14.0 GiB in the 60-update run (analytical 14.7 GiB). The host needs at least 32 GB.
- **Time per successful update:** 8.5 s measured, against a transfer bound of 8.5 s at the measured 12.23 GB/s. The
  step is transfer-bound.
- **Time per zero-signal attempt:** about 2.1 s (rollout only).
- **Forecast for the whole run (calibrated):**

  | scenario | rollout batches | time | host→device transfer |
  |---|---|---|---|
  | point estimate | ~665 | ~0.8 h | ~37 TB |
  | Wilson lower bound | ~836 | ~0.9 h | — |
  | all 1200 batches | 1200 | ~1.1 h | ~52 TB |

  Add the LR = 0 control (~10 min), the profile (~10 min) and the final evaluation of 4 models × 1066 test prompts
  (estimated 1–1.5 h from transfer bounds; the 60-update evaluation's wall time was not logged).

## Commands (run from the repository root, inside `tmux`, on the rented GPU host)

```bash
source .venv/bin/activate
python scripts/final150.py dry-run          # checks configs, run dirs, nvidia-smi, gates; runs and writes nothing
python scripts/final150.py preflight        # environment, CUDA/bf16, RAM >= 32e9, disk >= 40e9, budget files
python scripts/final150.py correctness      # CPU suite + kill/resume (tiny) + async L2L lab cases on this GPU, capped
python scripts/final150.py smoke            # 4 updates; must show the raw nvidia-smi files with attributed samples
python scripts/final150.py profile          # phase profile (2 steps) + H06 sync-vs-async prefetch timing on Qwen3-4B
python scripts/final150.py train            # the 150-update run (refuses unless the four stages above PASSed)
python scripts/final150.py control          # the LR = 0 control
python scripts/final150.py merge            # CPU tensor-by-tensor merge of step150 (monitored: expects no GPU use)
python scripts/final150.py eval             # base / trained / merged / control on the 1066 test prompts (monitored)
python scripts/final150.py soup             # Soup checks (needs Soup installed; otherwise UNAVAILABLE)
python scripts/final150.py report           # assignment_results/final150_v2/compliance.md
python scripts/final150.py export --to ~    # ~/final150_v2_evidence_<UTC>.tar.gz + sha256 manifest
# or everything in order, stopping at the first failed gate:
python scripts/final150.py all
```

Every GPU workload runs as
`qwen_grpo.telemetry.run_guarded(cap_bytes=8_000_000_000)`. Monitoring starts before the process exists. The
workload is stopped (SIGINT, then SIGTERM, then SIGKILL) on a sample above the cap, a dead sampler, or no attributed
sample within 900 s. A one-off command can be wrapped the same way:

```bash
python scripts/guarded_run.py --out assignment_results/manual/<UTC>_<name> -- python scripts/run.py train --config <yaml>
```

**Resume after an interruption** (crash, SIGKILL, lost SSH): the run continues from its highest-step valid checkpoint;
partial checkpoints are rejected.

```bash
python scripts/final150.py resume
#   = guarded: python scripts/run.py train --config runs/qwen3_4b_final_plain_150_v2/config.yaml --resume latest
```

**Final held-out evaluation, by hand** (what the `merge` and `eval` stages run):

```bash
python scripts/guarded_run.py --out assignment_results/final150_v2/manual_merge --no-gpu-expected -- \
  python scripts/merge.py --adapter runs/qwen3_4b_final_plain_150_v2/checkpoints/step150 \
                          --out assignment_results/final150_v2/merged/qwen3_4b_final_plain_150_v2_step150
python scripts/guarded_run.py --out assignment_results/final150_v2/manual_eval -- \
  python scripts/final_eval.py --run qwen3_4b_final_plain_150_v2 --control qwen3_4b_final_plain_150_v2_lr0 \
    --merged-dir assignment_results/final150_v2/merged/qwen3_4b_final_plain_150_v2_step150 \
    --out research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json
```

The evaluation reports, for each of base, trained, merged and control:

- accuracy, macro-F1, AUC, invalid rate, recall per class and prediction balance;
- changed predictions (indices) against base;
- paired bootstrap 95% CIs of the accuracy and AUC differences (trained − base, merged − trained, control − base);
- merged-vs-trained prediction agreement and label-score difference (max, median, mean);
- the merge's residuals;
- a UTC window for each model, sliced against the raw nvidia-smi CSV into `phase_memory.json`.

A CI that includes 0 means "no significant improvement", whatever the point estimate.

**Optional: kill/resume on the real model** (~10 min, never run by the driver):

```bash
python scripts/resume_kill_test.py --config configs/experiments/qwen3_4b_resume_kill.yaml --device cuda
```

**Optional: CUDA timeline** (if Nsight Systems is installed; the H06 lab's CUDA-event intervals are the primary
evidence):

```bash
nsys profile -t cuda,nvtx -o assignment_results/final150_v2/h06_timeline \
  python scripts/l2l_lab.py run --name t08_prefetch_timing_qwen3_4b_nsys --case prefetch_timing \
    --config research/l2l/configs/async_qwen3_4b.yaml --device cuda --trials 3 --warmup 1 --max-new 4
```

## Where things are written

| what | where |
|---|---|
| driver state, per-stage attempts | `assignment_results/final150_v2/state.json`, `<stage>/<UTC>/stage.json` |
| raw telemetry of each workload | `assignment_results/final150_v2/<stage>/<UTC>/<item>/telemetry/{nvidia_smi_apps.csv, nvidia_smi_gpu.csv, nvidia_smi_*.stderr, pids.csv, telemetry.json}`, plus `guarded_run.json`, `stdout.log`, `stderr.log` |
| training run | `runs/qwen3_4b_final_plain_150_v2/`: `config.yaml`, `manifest.json`, `training/{training.json, training.log, memory/<session>.json}`, `checkpoints/step{0,10,…,150}/`, `streaming/` |
| per-update metrics | `training.json` `training[]`: step, attempt, prompt ids, rewards, mixed groups, advantage stats, loss, gradient norms (all/A/B), update norm, effective adapter norm, change from init, learning rate, clipping, allocator memory. `timings[]`: rollout / old-log-prob / update / step seconds, rollout tokens, weight bytes, RSS, nvidia-smi peak |
| zero-signal attempts | `training.json` `zero_signal[]` (prompts, rewards, rollout hash) and `zero_signal_timings[]` (seconds, bytes, nvidia-smi peak) |
| validation curve | `training.json` `evaluations[]` (metrics incl. AUC, predictions, label scores, adapter sha256) and `evaluation_times[]` |
| LR = 0 control | `runs/qwen3_4b_final_plain_150_v2_lr0/` |
| merged model | `assignment_results/final150_v2/merged/qwen3_4b_final_plain_150_v2_step150/` (+ `merge_provenance.json`, `merge_report.json`) |
| held-out result | `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json`; per-model memory in `assignment_results/final150_v2/eval/<UTC>/phase_memory.json` |
| H06 / profile | `evidence/<UTC>_H06.prefetch_timing_4b/`, `research/l2l/experiments/t08_prefetch_timing_qwen3_4b*/`; `assignment_results/final150_v2/profile/<UTC>/phase_profile_result/` |
| Soup | `research/soup/<version>_<UTC>/` |

## Safe failure recovery

- **Cap violation or monitoring failure.** The workload was stopped. The stage is FAIL or INCONCLUSIVE, with the reason
  in `stage.json` and the raw CSVs kept. Read `guarded_run.json` before doing anything else. Do not relax the cap.
- **The training process died** (OOM, SSH loss, preemption). Run `python scripts/final150.py resume`. The run continues
  from the latest valid checkpoint, and all updates up to it are kept. Partial checkpoints are quarantined, never used.
- **The stage failed for another reason.** Fix the cause, then rerun with `python scripts/final150.py <stage> --retry`.
  Nothing is overwritten: every attempt gets a new `<UTC>` directory.
- **The smoke run must be rerun.** `runs/qwen3_4b_final_plain_150_v2_smoke/` must first be moved aside by hand (the
  driver never deletes a run):
  `mv runs/qwen3_4b_final_plain_150_v2_smoke runs/qwen3_4b_final_plain_150_v2_smoke.archived-$(date -u +%Y%m%dT%H%M%SZ)`.
- **The attempt budget ran out before 150 updates.** That is a result: report it. Do not raise `max_attempts` and
  continue silently. If more updates are wanted, run a new continuation with its own name
  (`scripts/run.py train --continue-from`) and report it as such.

## Back up artifacts (before the rented host is destroyed)

```bash
python scripts/final150.py export --to ~          # tar.gz without the ~8 GB merged weights (their shard sha256 are kept)
sha256sum ~/final150_v2_evidence_*.tar.gz > ~/final150_v2_evidence.sha256
# from your own machine:
scp -P <port> root@<host>:~/final150_v2_evidence_*.tar.gz* .
tar -xzf final150_v2_evidence_<UTC>.tar.gz -C /path/to/qwen-grpo   # the same layout as the repository
sha256sum -c <(cd /path/to/qwen-grpo && cat assignment_results/final150_v2/export_manifest.sha256)
```

## Why a separate driver

`scripts/run_assignment.sh` / `scripts/assignment.py` (stages A–H) is the 60-update protocol: fixed item IDs,
DFlash-aware gates, and a report generator bound to those run names. It stays unchanged for reproducing that
protocol; only its item runner now uses the shared supervisor.

`scripts/final150.py` reuses the same components: the study dispatcher for the lab cases, `qwen_grpo.telemetry` for
monitoring, and the trainer, merge and evaluation scripts. What is new is only the stage list, the gates and the
paths of the 150-update protocol.
