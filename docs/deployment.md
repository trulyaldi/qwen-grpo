# Running on a rented Linux GPU

Only after the Qwen3-0.6B evidence from Kaggle (`kaggle/README.md`) is in. You rent the machine, connect and clone;
nothing here provisions cloud resources. The steps go cheapest first: each can reveal the problem the next one would
waste money on. Use `tmux` (or `nohup ... &`) so a disconnect does not kill a run.

## Choosing hardware

| experiment | GPU | host RAM | disk |
|---|---|---|---|
| Qwen3-0.6B, all three backends (bf16) | Ampere or newer (native bf16), ≥ 8 GB | ≥ 16 GB | ≥ 20 GB |
| Qwen3-4B L2L, 8 GiB target (H10–H12) | Ampere or newer, ≥ 8 GB (larger is fine: the guard enforces the budget) | ≥ 32 GB (estimated 11.7–18.5 GiB incl. a ~7.3 GiB pinned store) | ≥ 40 GB |
| Qwen3-4B resident reference (optional) | ≥ 16 GB (estimated peak ~9.5 GiB) | ≥ 16 GB | ≥ 40 GB |

Estimates are from the Step-0 calculator (`scripts/budget.py`), not measurements. **Budget evidence (H11) counts only
if `nvidia-smi` lists the training process with a number**: many containers hide host PIDs. Step 2 checks this.

## One command

After step 1 (environment), the whole protocol below (steps 2–8) runs unattended:

```bash
tmux new -s assignment
bash scripts/run_assignment.sh --dry-run     # first: capabilities, blockers, the commands each stage will run
bash scripts/run_assignment.sh               # then the run; after a disconnect: tmux attach -t assignment
bash scripts/run_assignment.sh --resume      # after an interruption (refused if the commit, diff or configs changed)
```

Preflight lists blockers in the first minutes. These are the ones to act on before paying for hours:
- `per-process nvidia-smi` False: the 8 GB qualification is INCONCLUSIVE and the final run will not start. Use
  another host.
- `fixture packages` False: the lab equivalence cases are BLOCKED. Install the pinned torch build.
- Host RAM or disk below the Qwen3-4B requirement.

Results land in `assignment_results/<run-id>/`. The files to read first are `reports/summary.md` (the three-page
draft) and `reports/requirements_matrix.md`. Copy the directory off the host (plus `evidence/`,
`research/l2l/experiments/` and the small files of `runs/`) before destroying it. The steps below are what the script
does, for running a part by hand.

## Steps

1. **Environment** (the lab fixtures require exactly these packages):
   ```bash
   git clone https://github.com/trulyaldi/qwen-grpo.git && cd qwen-grpo && git checkout <pushed commit>
   python3.12 -m venv .venv && source .venv/bin/activate
   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   export HF_HOME=/workspace/hf        # a persistent volume, so models download once
   ```
2. **What can run here (≈ 1 min, no model):**
   ```bash
   python scripts/experiment.py plan --target local
   ```
   The header must show a usable GPU, `bf16 True`, `fixture packages True`, and `per-process nvidia-smi True`. If the
   last is False, H11 can only be INCONCLUSIVE (its entry is blocked): use another host for H11.
3. **Tests (CPU ~11 min, then the CUDA variants on tiny models):**
   ```bash
   CUDA_VISIBLE_DEVICES="" python -m pytest -q
   python -m pytest -q tests/test_prefetch.py tests/test_backward.py tests/test_generator.py tests/test_layer_stream.py \
     tests/test_backend.py tests/test_output_head.py tests/test_weight_store.py tests/test_dflash.py
   ```
4. **Correctness on this GPU (tiny models, minutes):** the async cases, then the sync CUDA cases again, because their
   recorded evidence predates later executor changes (`research/l2l/final_report.md`, evidence freshness):
   ```bash
   python scripts/experiment.py run --entry H01.lab_async_cuda H01.fault_unsafe_reuse H02.lab_async_cuda \
     H02.fault_unsafe_reuse H04.lab_async_cuda H05.lab_async_cuda
   python scripts/experiment.py run --again --entry H01.lab_sync H02.lab_sync_cuda H04.lab_sync_cuda H05.lab_sync_cuda
   python scripts/experiment.py run --entry H07.depth_residency_execution
   ```
5. **Qwen3-0.6B in bf16** (downloads ~1.5 GB): the three backends on identical inputs and their comparison, prefetch
   timing (needs step 4's async evidence), the real-model merge and resume:
   ```bash
   python scripts/experiment.py run --entry H05.real_0.6b_bf16 H06.prefetch_timing H09.merge_qwen3_0.6b H08.resume_qwen3_0.6b
   ```
6. **Qwen3-4B** (downloads ~8 GB): the guarded plain smoke run first (4 steps, the final batch shape), then the
   profile, the DFlash benchmark and the 10-step DFlash smoke, and only then the 60-step primary run, its LR=0 control
   and the held-out evaluation:
   ```bash
   python scripts/experiment.py run --entry H11.smoke
   python scripts/experiment.py run --entry H11.profile H10.dflash_bench H10.smoke_dflash   # the final run uses DFlash
   python scripts/experiment.py run --entry H12.train_final       # hours; resumable
   python scripts/experiment.py run --entry H12.final_eval H10.dflash_drift
   # optional, opt-in only (hours): continue the primary run to 210 updates as the new run qwen3_4b_final_ext210
   python scripts/experiment.py run --entry H12.train_extended H10.dflash_drift_extended
   ```
   Go on from the smoke run only if it completed (its memory guard stops it at the first nvidia-smi sample above
   8 GiB). On a guard failure, change one measurable setting (e.g. `training.micro_batch_size`) in a new config and
   profile that; nothing is reduced automatically.
7. **Resume after an interruption** (the run's status is `interrupted` or `failed`):
   ```bash
   python scripts/run.py train --config runs/<run>/config.yaml --resume latest              # streaming runs
   python scripts/run.py train --config runs/<run>/config.yaml --resume runs/<run>/checkpoints/step<N>   # others
   python scripts/experiment.py run --entry <id>        # then: completed runs are skipped, the rest continues
   ```
8. **Report, collect, then stop paying:**
   ```bash
   python scripts/experiment.py report      # evidence/study_report.md, research/l2l/final_report.md
   tar czf results_$(date -u +%Y%m%dT%H%MZ).tgz evidence/ research/l2l/ runs/
   ```
   Copy the archive off the machine (`scp` / `rsync`) and check the copy before destroying the instance. Statuses in
   `research/l2l/hypotheses.yaml` are then updated by hand, citing the new evidence names.

## What each run records

| quantity | where |
|---|---|
| GPU, CUDA, driver, packages, git commit (dirty flag), config sha256, model revision, dataset sha256, host RAM | `runs/<run>/training/training.json` → `environment`, `config_sha256` |
| allocator allocated / peak allocated / peak reserved (exclude the CUDA context) | `training.json` → `training[*]`, `memory` |
| process / device GPU memory incl. CUDA context | `training/memory/*.json` (with `memory.guard`), `evidence/<id>/raw/nvidia_smi_*.csv` |
| rollout / old-log-prob / update time, rollout tokens, CPU RSS, weight bytes copied | `training.json` → `timings` |
| phase profile, H2D bandwidth | `scripts/profile_run.py` output (H11 entry) |
| model quality with CIs | `evaluation/step<N>/<split>.json`, `final_eval` output |
| every dispatcher execution: hardware probe, commands, UTC times, exit codes, raw logs | `evidence/<UTC>_<entry>/` |
