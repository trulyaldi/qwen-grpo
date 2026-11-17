# Running on a rented Linux GPU

You rent the machine, connect and clone. This page lists the commands, cheapest first: each step can reveal the
problem the next one would waste money on. Nothing here provisions or manages cloud resources.

## Choosing hardware

| experiment | GPU | host RAM | disk |
|---|---|---|---|
| Qwen3-0.6B, all three backends | any CUDA GPU ≥ 4 GB with bf16 (compute capability ≥ 8.0 recommended) | ≥ 16 GB | ≥ 20 GB |
| Qwen3-4B resident | ≥ 16 GB (estimated peak ~9.5 GiB, unmeasured) | ≥ 16 GB | ≥ 40 GB |
| Qwen3-4B L2L, 8 GiB target | ≥ 8 GB | ≥ 32 GB (estimated 11.7–18.5 GiB, incl. a ~7.3 GiB pinned store) | ≥ 40 GB |

The estimates come from `qwen-grpo preflight` (the budget calculator, no measurement). A card larger than 8 GB is fine
for the 8 GiB target: `memory.guard` enforces the budget during training, not the hardware.

Budget evidence counts only if `nvidia-smi --query-compute-apps=pid,used_memory` lists the training process with a
number. Some containers hide host PIDs. The guard then falls back to the device-wide figure, which can only stop a
run early, but the evidence is graded diagnostic. Step 3 checks this in a minute.

## Steps

1. **Environment:**
   ```bash
   git clone --branch feat/l2l-streaming-research https://github.com/trulyaldi/qwen-grpo.git && cd qwen-grpo
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # cuXXX matching the driver
   pip install -r requirements.txt && pip install -e . --no-deps
   export HF_HOME=/workspace/hf        # a persistent volume, so models download once
   python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.is_bf16_supported())"
   nvidia-smi
   ```
2. **Lightweight tests (CPU, ~10 min), then the CUDA variants on tiny models (minutes):**
   ```bash
   CUDA_VISIBLE_DEVICES="" python -m pytest -q
   python -m pytest -q tests/test_prefetch.py tests/test_backward.py tests/test_generator.py tests/test_layer_stream.py \
     tests/test_backend.py tests/test_output_head.py tests/test_weight_store.py
   ```
   The CUDA variants are the first GPU check of the current L2L code: the saved CUDA evidence predates later changes.
3. **Hardware preflight (≈ 1 min, no model):**
   ```bash
   qwen-grpo preflight --config configs/gpu/qwen3_4b_l2l_async.yaml --gpu --hardware-only --allow-metadata-download
   ```
   It checks the architecture, that this torch build has kernels for the GPU, per-process `nvidia-smi` accounting,
   that the GPU is exclusive, the planned pinned allocation, and the measured H2D bandwidth.
4. **Qwen3-0.6B, one run per backend** (downloads ~1.5 GB):
   ```bash
   for b in resident l2l_sync l2l_async; do
     python scripts/run.py train --config configs/gpu/qwen3_0.6b_$b.yaml
   done
   python scripts/compare_runs.py qwen3_0.6b_resident qwen3_0.6b_l2l_sync   # identical, or every difference listed
   ```
   These are the full 100-step comparison runs.
   - **Timing:** read `runs/<name>/training/training.json` → `timings` after the first steps. To stop early,
     Ctrl-C; the run is `interrupted` and resumable (below).
   - **Determinism:** bitwise identity resident vs L2L was observed on the laptop GPU. On another GPU, kernels may
     differ between the two backends; the comparison script reports any difference.
5. **Qwen3-4B, with a guarded profile first.** One step of every phase under the 8 GiB guard; it writes a JSON with
   raw samples:
   ```bash
   python scripts/profile_run.py --config configs/gpu/qwen3_4b_l2l_async.yaml --budget-gib 8 --steps 1 --eval-n 16 \
     --out-dir runs/_profile
   ```
   Go on only if its verdict is not FAIL.
   - **On FAIL:** change one measurable setting (e.g. `training.micro_batch_size`) in a new config, and profile that.
     Nothing is reduced automatically.
   - **Then train:**
     ```bash
     python scripts/run.py train --config configs/gpu/qwen3_4b_l2l_async.yaml   # 8 GiB target, guard on
     python scripts/run.py train --config configs/gpu/qwen3_4b_l2l_sync.yaml    # prefetching's effect
     python scripts/run.py train --config configs/gpu/qwen3_4b_resident.yaml   # on a >= 16 GB GPU
     ```
6. **Resume after an interruption** (a dropped SSH session, a preemption):
   ```bash
   ls runs/qwen3_4b_l2l_async/checkpoints/
   python scripts/run.py train --config runs/qwen3_4b_l2l_async/config.yaml \
     --resume runs/qwen3_4b_l2l_async/checkpoints/step<N>
   ```
   - Use `tmux` or `nohup ... &` so a disconnect does not kill the run.
   - Checkpoints are written every `evaluation.every` steps (10). Resume continues bitwise as the uninterrupted run
     would, tested on CPU.
7. **Evaluate and compare:**
   ```bash
   python scripts/run.py validate --run qwen3_4b_l2l_async
   python scripts/final_eval.py --run qwen3_4b_l2l_async      # test split: base / adapter / streamed-merged
   python scripts/compare_runs.py qwen3_4b_resident qwen3_4b_l2l_async
   ```
8. **Collect results, then stop paying:**
   ```bash
   tar czf results_$(date -u +%Y%m%dT%H%MZ).tgz runs/ research/l2l/experiments/
   ```
   - The archive holds the reports, logs with UTC timestamps, raw memory samples, checkpoints and evidence.
   - Copy it off the machine (`scp` / `rsync`) and check the copy before destroying the instance.

## What to record from each run

| quantity | where |
|---|---|
| GPU, CUDA, driver, packages, git commit (dirty flag), config sha256, model revision, dataset sha256, host RAM | `training/training.json` → `environment`, `config_sha256` |
| allocator allocated / peak allocated / peak reserved | `training/training.json` → `training[*]`, `memory` |
| process / device GPU memory incl. CUDA context | `training/memory/*.json` (with `memory.guard`) |
| rollout / old-log-prob / update time, rollout tokens, CPU RSS, weight bytes copied | `training/training.json` → `timings` |
| phase profile, H2D bandwidth, transfer time estimate | `scripts/profile_run.py` output |
| model quality with CIs | `evaluation/step<N>/<split>.json`, `final_eval` output |

## Gated alternative

`qwen-grpo train --config <yaml>` runs the same trainer, but first runs an offline preflight. For configs with
`memory.guard` (the Qwen3-4B ones), it also requires that a GPU preflight with the same memory settings has passed:

```bash
qwen-grpo preflight --config configs/gpu/qwen3_4b_l2l_async.yaml --gpu
```

The guard itself is active in both entry points.

Assignment-specific runs, kept from the earlier work, are in `configs/experiments/qwen3_4b_*` and
`scripts/final_run.sh`: smoke, final 1024-prompt run, LR=0 control, held-out evaluation and report.
