# Kaggle: free preliminary test runs

Kaggle runs the repository's own tests and a short training run on free CPU and T4 GPU machines, before any Vast.ai
credit is spent. **It is not the assignment's hardware validation:** a T4 is Turing (compute capability 7.5), has no
native bfloat16 and 16 GB per GPU, and the 8 GB cap is not enforced here. Passing runs say nothing about Qwen3-4B in
bf16 on an Ampere card under 8 GB.

## How it works

- `kaggle/run.py` is the kernel source (standard library only). `scripts/kaggle_submit.py` writes the submission into
  it (pinned commit, mode, kernel id, a submission UUID) and pushes it with `kaggle kernels push`. Only `run.py` and
  `kernel-metadata.json` are uploaded: no credentials, checkpoints, caches or data.
- On Kaggle the runner clones the **public** GitHub repository at the pinned commit (`git rev-parse HEAD` is checked).
  It then builds a fresh venv from `requirements.txt` (torch from the CPU or cu126 index), records the environment and
  runs the mode's steps. Internet is enabled for the clone, pip and the Hugging Face model download.
- One kernel per mode: `<username>/qwen-grpo-cpu`, `-gpu-tests`, `-gpu-short`. Each push is a new version, so Kaggle
  keeps the history.
- The local ledger `kaggle/submissions.jsonl` is not committed. It refuses a push while the kernel is still
  queued/running, and refuses to repeat a mode + commit + scope that was already submitted (`--force` overrides).

## Commands (from the repository root, WSL)

```bash
python scripts/kaggle_submit.py submit   --mode cpu                 # quick CPU subset (default --scope quick)
python scripts/kaggle_submit.py submit   --mode cpu --scope full    # the whole CPU suite
python scripts/kaggle_submit.py submit   --mode gpu-tests           # T4: CUDA correctness tests
python scripts/kaggle_submit.py submit   --mode gpu-short           # T4: short GRPO + LoRA runs, Qwen3-0.6B
python scripts/kaggle_submit.py status   --mode cpu                 # kaggle kernels status <id>
python scripts/kaggle_submit.py logs     --mode cpu                 # kaggle kernels logs <id>
python scripts/kaggle_submit.py download --mode cpu                 # kaggle kernels output -> kaggle/outputs/cpu/<submission>/
python scripts/kaggle_submit.py summary  --mode cpu                 # pass / fail / skip, attribution check, exit code
```

- `--commit <sha>` tests another commit. The commit must already be on GitHub; submit checks this.
- `--dry-run` renders and validates `kaggle/build/<mode>/` without pushing.
- Raw CLI equivalents: `kaggle kernels status|logs|output aldiyarmoldazhanov/qwen-grpo-<mode>`.

## Modes

| mode | machine | what runs |
|---|---|---|
| `cpu` | CPU | Import checks. Every repository config loaded and validated. pytest: config, sentiment data and reward, GRPO maths, policy, explicit forward, L2L layer streaming, checkpointed backward, trainer (checkpoints, bitwise resume), disk streaming and crash-consistent resume, correctness audit, run reports. `--scope full`: the whole suite. Qwen weights are never downloaded (`tests/conftest.py` forbids it). |
| `gpu-tests` | T4 (device 0) | pytest on every file with CUDA-parametrized tests (resident vs L2L forward, checkpointed backward and LoRA gradients, sync vs async prefetch, buffer reuse and synchronization, pinned weight store, streamed generation, DFlash exactness, profiler). The lab cases (`forward / gradient / generation / update_equivalence`, async, CUDA) run only on a GPU with native bf16, because each lab grid mixes fp32 and bf16. |
| `gpu-short` | T4 (device 0) | Three 8-step GRPO + LoRA runs of Qwen3-0.6B in **float32** (diagnostic), one per backend: `configs/experiments/kaggle_06b_fp32_resident.yaml`, `_l2l_sync.yaml`, `_l2l_async.yaml` (identical except `execution` and `run.name`). Then `scripts/compare_runs.py` compares sync and async with resident (bitwise rule of `tolerances.yaml` `update_equivalence`). They cover model loading, rollouts, reward, advantages, LoRA gradient flow, optimizer steps, checkpoint save/reload and subset evaluation. |
| `gpu-train` | T4 (device 0) | One full training run, `scripts/run.py train --config <yaml>`, of a YAML committed at the pinned commit (`--config`). See below. |

## gpu-train: full Qwen3-0.6B training runs

```bash
python scripts/kaggle_submit.py submit --mode gpu-train --config configs/experiments/kaggle_06b_fp32_pilot.yaml --dry-run
python scripts/kaggle_submit.py submit --mode gpu-train --config configs/experiments/kaggle_06b_fp32_pilot.yaml [--max-hours 11]
python scripts/kaggle_submit.py status|logs|download|summary --mode gpu-train [--submission <id>]
python scripts/kaggle_submit.py submit --mode gpu-train --config <same yaml> --resume-from <submission id>   # next version
```

- **Configs** (`configs/experiments/`, all float32 diagnostics): `kaggle_06b_fp32_train_{resident,l2l_sync,l2l_async}.yaml`
  (1024 prompts, 210 updates, evaluation and checkpoint every 30, full validation, memory guard),
  `kaggle_06b_fp32_train_streaming.yaml` (l2l_async + disk streaming), `kaggle_06b_fp32_pilot.yaml` (the streaming run,
  20 updates). Other steps, intervals or seeds: commit a YAML that extends one of them, push, submit it.
- **Checks before anything is pushed:** the config is resolved from the pinned commit's tree (not the working copy);
  bf16, models other than Qwen3-0.6B and unimplemented settings are refused; the resolved sha256 goes into the
  submission and the kernel refuses to train if its own resolution differs. A submission repeating the same commit,
  config and resume source is refused, as is a new version while the previous gpu-train output is not downloaded
  (`--force` overrides both), and while the kernel is queued or running.
- **Time and storage:** training stops at `--max-hours` (default 11, measured from kernel start) with SIGINT, so the
  run ends `interrupted` with its saved checkpoints; then the whole run directory (checkpoints included) is copied to
  `/kaggle/working/evidence/artifacts/runs/<run>/`. `/kaggle/working` disappears with the session, so download each
  version's output (by its own kernel version) before submitting the next.
- **Resume across versions:** `--resume-from <id>` needs that submission's downloaded output and the same config and
  commit. The run directory is packed (`run.tar` + `qwen_grpo_resume.json` with every file's sha256) and uploaded as a
  **private** Kaggle dataset `<user>/qwen-grpo-resume-<run>`, attached to the new version. The kernel restores it,
  verifies every file and the marker's sha256 (a stale dataset version is refused), and runs
  `scripts/run.py train --config runs/<run>/config.yaml --resume <latest | runs/<run>/checkpoints/step<N>>`: the
  continued run equals an uninterrupted one bitwise on the same hardware and software (tests/test_kaggle_train.py, CPU).
- **Evidence** (beyond the table below): `resolved_config.yaml`, `raw/train.log`, `performance_summary.json` (status,
  steps, s/step, per-phase timings, weight bytes copied host→GPU, allocator peaks, nvidia-smi device and per-process
  maxima, host RAM, peak RSS, free disk, reward curve, accuracy and predicted-POSITIVE rate per evaluation, final
  validation vs base, git commit, packages, model revision), `host_resources.csv`, `gpu_apps.csv`, and the run itself.
- **Outcome:** PASS only for a completed run (every step, frozen weights unchanged, final checkpoint). INCOMPLETE (time
  budget, resumable) and FAIL exit non-zero; `summary` passes only PASS, and prints the resume command when INCOMPLETE.

### bfloat16 on T4 (never silently replaced)

- The runner records `torch.cuda.is_bf16_supported(including_emulation=False)`. Where it is False:
  - CUDA × bfloat16 pytest items are **SKIPPED**, with the GPU name and compute capability as the reason.
  - The bf16 lab cases are **SKIPPED**.
  - The bf16 backend comparison (`compare_backends_bf16`) is **SKIPPED**; it is study entry `H05.real_0.6b_bf16`.
- `gpu-short` runs the committed, separately labelled float32 diagnostics `configs/experiments/kaggle_06b_fp32_*.yaml`
  (their resolved form and sha256 are saved in the evidence). float32 results are never evidence for a bf16 claim.
- Every mode also records `python scripts/experiment.py plan --target local` (`raw/study_plan.log`): which study entries
  this machine could run, and why the others are blocked.
- The production configs, tolerances and training code are not changed.
- Only device 0 is used (`CUDA_VISIBLE_DEVICES=0`). Kaggle's two T4s are two separate 16 GB devices, not a pool.

## Evidence (`/kaggle/working/evidence/`, downloaded to `kaggle/outputs/<mode>/<submission_id>/`)

| file | content |
|---|---|
| `environment.json` | submission (id, kernel, mode, commit), UTC start, host CPU / RAM / disk, nvidia-smi query, python / torch / CUDA / cuDNN / transformers / peft, every GPU's name, capability, memory and bf16 support |
| `git_commit.txt` | the commit actually checked out on Kaggle |
| `execution.log` | every command with UTC timestamps and exit codes |
| `raw/` | raw output of every step, pytest JUnit XML, `pip freeze` |
| `gpu_memory.csv` | `nvidia-smi` sampled every second for the whole run (a note if no GPU) |
| `resolved_config.yaml` | `gpu-short`: the three diagnostic configs and their resolved form and sha256 |
| `artifacts/` | `gpu-short`: the runs' directories (training.json with per-step allocator peaks, checkpoints, reports), the comparison |
| `results.json`, `test_results.txt` | derived: per-step status and test counts, skipped tests with reasons, overall |

**Overall status:**
- **PASS** only if no step failed or errored and nothing was skipped.
- **PASS_WITH_SKIPS** lists every skip.
- **FAIL** makes the kernel exit non-zero, so Kaggle marks the version as an error.
- `summary` exits non-zero on FAIL, or if the evidence does not name the ledger's submission id, kernel, mode and
  commit.

## Still needs Vast.ai (Ampere or newer)

- Qwen3-4B in bf16 under the 8 GB cap, with per-process nvidia-smi.
- The bf16 lab cases (H01, H02, H04, H05 async).
- H06 prefetch timing, H09 real, H10 DFlash, H11 budget (`python scripts/experiment.py plan --target ampere`).
- H07's execution-level case (`t11_depth_residency`, float32) can run on the T4.
- The final run with its LR=0 control and the held-out evaluation.
