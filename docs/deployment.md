# Deployment on a rented GPU (Vast.ai)

Nothing here rents, starts or stops an instance: every step is a command you run. Order matters. Each step is the
cheapest one that can reveal the next problem, and none needs the full run. Commands assume the repository root with
the virtualenv active.

## What runs where

| where | what | cost |
|---|---|---|
| any CPU machine (the 64 GB desktop) | Tier 1–2 tests, offline preflight, fault-injection tests, Soup diagnostics, CPU lab cases (H03, H08 tiny, H09 tiny), the final report | free |
| laptop GPU (RTX 3050 Ti, 4 GB, WSL2) | CUDA tests on tiny models, Qwen3-0.6B cases (H06, H08 real, H09 real) | free; no assignment-grade memory evidence under WSL2 |
| rented Ampere GPU, native Linux | everything Qwen3-4B: GPU preflight, smoke, DFlash, final run, verification, merge, H10–H12 | paid |

## Choosing the instance

- **GPU:** Ampere or newer (compute capability ≥ 8.0). A card with more than 8 GB is fine: the budget is enforced by
  `memory.guard`, not by the hardware. Exclusive use, so no other process shares the GPU.
- **Host RAM:** ≥ 32 GB recommended. Offline preflight estimates **11.7–18.5 GiB** for the Qwen3-4B run: a 9 GiB
  pinned store, plus the loader's transient copy, plus the process. On a 16 GB host the result is inconclusive
  (`cpu.estimate` NOT TESTED). The estimate is a conservative upper bound until GPU preflight measures it.
- **Disk:** ≥ 40 GB, ideally on a persistent volume. The Qwen3-4B checkpoint is 7.5 GiB and the drafter 1 GiB. One
  merged checkpoint is another 7.5 GiB, plus checkpoints, streaming records and the pip cache.
- **Per-process GPU memory accounting:**
  - Budget evidence is assignment-grade only if `nvidia-smi --query-compute-apps` reports **this** process's memory.
  - In some containers PID namespaces hide it. The guard then falls back to the device-wide figure, which can only stop
    a run early, never let one through. But the evidence grades as diagnostic, and `gpu.measured` stays BLOCKED.
  - Step 4 checks this in about a minute. If it is BLOCKED, the instance cannot produce the required evidence:
    destroy it and pick another template or host before downloading anything.

## Before renting (CPU, free)

```bash
python -m pytest -q                                            # Tier 2: full CPU suite (~11 min)
qwen-grpo faults test                                          # fault-injection tests (~25 s)
qwen-grpo preflight --config configs/experiments/qwen3_4b_smoke.yaml --allow-metadata-download --host-ram-gib 32
qwen-grpo research run H03 --execute                           # CPU lab cases that are still missing
qwen-grpo research run H09 --execute                           # (cuda entries report BLOCKED here)
```

The preflight must show no FAIL. BLOCKED items are the ones that need the GPU host, such as the tokenizer once the
model is downloaded.

## On the instance

1. **Environment.** Pin the same versions as the evidence:
   ```bash
   git clone <repo> && cd qwen-grpo && git checkout <commit>
   uv venv .venv && source .venv/bin/activate
   uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
   uv pip install -e . --no-deps                                 # the qwen-grpo command
   export HF_HOME=/workspace/hf                                  # persistent volume: download once
   ```
2. **Check the environment before downloading anything:**
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
   nvidia-smi
   ```
3. **Models.** About 8.5 GB, once:
   ```bash
   hf download Qwen/Qwen3-4B
   hf download z-lab/Qwen3-4B-DFlash-b16 config.json model.safetensors
   ```
   The dataset is `data/sentiment.json`, which is in the repository. Its sha256 is checked by preflight and recorded
   in every run.
4. **GPU preflight, hardware only (≈ 1 min):**
   ```bash
   qwen-grpo preflight --config configs/experiments/qwen3_4b_smoke.yaml --gpu --hardware-only
   ```
   Must PASS: `gpu.architecture`, `gpu.capacity`, `gpu.accounting` (per-process nvidia-smi), `gpu.exclusive`,
   `cpu.pinned` and `pcie.bandwidth`. Then rerun the estimate with the measured bandwidth (it is printed and stored
   under `estimates.resources_measured_pcie`).
5. **GPU preflight with the model (minutes).** One profiled step of every phase under the 8 GiB guard (setup, rollout,
   old log-probs, forward, backward, optimizer, merge, evaluation):
   ```bash
   qwen-grpo preflight --config configs/experiments/qwen3_4b_smoke.yaml --gpu
   ```
   `gpu.measured` PASS is the first real 8 GB evidence. It also unlocks `train` for every config with the same memory
   signature: the smoke, final and LR=0 runs.
   - On FAIL, do **not** start training. Read the per-phase peaks in `runs/_preflight/<run>/<UTC>_gpu/profile.json`.
   - Change one measurable setting (for example `training.micro_batch_size`) in a new experiment file and preflight
     that.
   - Nothing changes batch size, sequence length, precision or DFlash on its own.
6. **Correctness on the GPU (tiny models, minutes):**
   ```bash
   python -m pytest -q tests/test_prefetch.py tests/test_backward.py tests/test_generator.py tests/test_layer_stream.py \
     tests/test_backend.py tests/test_output_head.py tests/test_weight_store.py
   for h in H01 H02 H04 H05 H07; do qwen-grpo research run $h --execute; done
   ```
7. **Smoke run with a crash and a resume (4 steps):**
   ```bash
   QWEN_GRPO_CRASH=update@3 qwen-grpo train --config configs/experiments/qwen3_4b_smoke.yaml  # dies at update 3
   qwen-grpo resume --run qwen3_4b_smoke                         # continues from the latest valid checkpoint
   ```
   - Read `runs/qwen3_4b_smoke/report.md`.
   - Note the step time from `training/training.json` (`step_seconds`). The final run takes about 210 × that, plus
     evaluation.
   - `training/memory/` holds one nvidia-smi sample file per session.
8. **Bitwise resume on a real model:** `qwen-grpo research run H08 --execute` (Qwen3-0.6B, six crash points).
9. **DFlash:**
   - `qwen-grpo preflight --config configs/experiments/qwen3_4b_smoke_dflash.yaml --gpu` checks greedy equality and the
     budget with the drafter resident.
   - `qwen-grpo research run H10 --execute` measures acceptance and GRPO step time with and without DFlash.
   - Decide here whether the final run uses DFlash: `configs/experiments/qwen3_4b_final.yaml` does not. See
     `docs/status.md`.
10. **Budget profile (H11):** `qwen-grpo research run H11 --execute`.
11. **Full run.** Resumable. Each script stops at the first failure:
    ```bash
    bash scripts/final_run.sh       # gated train of the final run + LR=0 control, test-split evaluation, report
    ```
12. **Verification and merge.** `final_run.sh` already ran `verify`; to run the steps by hand:
    ```bash
    qwen-grpo verify --run qwen3_4b_final --control qwen3_4b_final_lr0  # base / trained / merged / LR=0 on test, CIs
    qwen-grpo merge  --run qwen3_4b_final                                 # runs/qwen3_4b_final/merged/step<N>/
    qwen-grpo ship                                                        # evidence-derived verdict
    qwen-grpo soup all --soup-bin <soup>                                  # optional; informational only
    ```
13. **Export, then stop paying.**
    - Commit the evidence (`research/l2l/experiments/`, `research/l2l/final_*`), which is small.
    - Copy `runs/` (checkpoints, raw nvidia-smi samples in `training/memory/`, timestamped `training.log`) to persistent
      storage or `rsync` it home.
    - Check the copies, then destroy the instance from the Vast console or CLI.

## Credit discipline

- Steps 4–5 exist so that an unusable instance costs minutes, not hours. Never skip them for a new instance type.
- Keep `HF_HOME` and the pip cache on the persistent volume, so a new instance does not download again.
- Smoke before final; one change at a time; resume instead of restarting (`qwen-grpo resume`).
- Do not rerun lab cases whose evidence exists: `research run` skips them.
