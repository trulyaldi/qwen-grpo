# qwen-grpo

GRPO + LoRA fine-tuning of Qwen3 in non-thinking mode, without an SFT stage, written by hand and checked numerically.
The task is binary sentiment classification on Rotten Tomatoes. The reward is automatic: 1 if the completion is
exactly the correct label, else 0.

Qwen3-4B's bf16 weights alone are ~8 GB, so under an 8 GB GPU cap the model cannot be resident. This repository
implements, without external streaming libraries:
- **L2L execution:** weights in pinned CPU memory, a bounded set of GPU layer buffers, asynchronous prefetch,
  layer-boundary activation checkpointing.
- **Disk-backed streaming** of prompts and rollouts, with crash-consistent checkpoints and deterministic resume.
- **DFlash-drafted rollouts** verified by the current LoRA policy.
- **A streamed tensor-by-tensor LoRA merge.**
- **Evidence tooling:** a held-out comparison of base, adapter, merged model and LR=0 control with confidence
  intervals, and a verdict derived from recorded evidence.

## Status

| | |
|---|---|
| Qwen3-0.6B, resident | trained and verified: validation accuracy 0.7486 → 0.8096, ROC AUC 0.8277 → 0.8929 (`runs/qwen`) |
| L2L vs resident | bitwise-equal forward, gradients, generation and updates on tiny models (CPU + CUDA) and on Qwen3-0.6B training runs; the CUDA evidence predates later executor changes |
| async prefetch, DFlash, streamed merge, disk streaming / resume, chunked output head | implemented and tested on CPU with tiny models; real-model GPU evidence pending |
| Qwen3-4B under 8 GB | **not run yet**: no GPU evidence |
| verdict (`qwen-grpo ship`) | **DON'T SHIP**: 6 of 15 requirements PASS, 9 NOT TESTED (all need a GPU) |

Details, the audit behind this table, open decisions and limitations: [`docs/status.md`](docs/status.md).

## Hardware

| use | hardware |
|---|---|
| CPU tests, offline preflight, Soup diagnostics, CPU research cases | any x86-64 Linux, ≥ 16 GB RAM (no GPU) |
| local training | Qwen3-0.6B on an RTX 3050 Ti (4 GB), ~14 GB usable RAM |
| the assignment | Qwen3-4B on a rented Ampere-or-newer GPU, capped at 8 GB (CUDA context included), ≥ 32 GB host RAM recommended, native Linux with per-process `nvidia-smi` |

## Installation

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126   # pinned versions
uv pip install -e . --no-deps          # the `qwen-grpo` command (or: python -m qwen_grpo)
```

## Quick start: Qwen3-0.6B

```bash
qwen-grpo preflight --config configs/experiments/qwen_local.yaml       # offline checks, no GPU
qwen-grpo train     --config configs/experiments/qwen_local.yaml       # L2L async + streaming, 10 steps
qwen-grpo resume    --run qwen_local                                    # after an interruption
qwen-grpo report    --run qwen_local                                    # runs/qwen_local/report.md
```

`configs/profiles/qwen3_0_6b.yaml` is the resident reference setup (100 steps), and `qwen-grpo validate | test |
verify --run <name>` evaluates resident runs.

## Qwen3-4B: smoke, then the final run

On the rented GPU, after the steps in [`docs/deployment.md`](docs/deployment.md):

```bash
qwen-grpo preflight --config configs/experiments/qwen3_4b_smoke.yaml --gpu --hardware-only   # ~1 min
qwen-grpo preflight --config configs/experiments/qwen3_4b_smoke.yaml --gpu                   # one guarded step of every phase
qwen-grpo train     --config configs/experiments/qwen3_4b_smoke.yaml                         # 4 steps, final batch shape
bash scripts/final_run.sh                    # final run + LR=0 control + held-out evaluation + report (resumable)
qwen-grpo verify --run qwen3_4b_final --control qwen3_4b_final_lr0   # base / trained / merged / control, CIs
qwen-grpo merge  --run qwen3_4b_final        # streamed merge -> runs/qwen3_4b_final/merged/step<N>/
qwen-grpo ship                               # SHIP / DON'T SHIP from the evidence (exit 0 / 2)
```

`train` refuses a config with `memory.guard` until a GPU preflight with the same memory signature has passed. A
smoke run's preflight covers the final run and its control. During training the guard samples nvidia-smi and stops
the run at the first sample above 8 GiB; the raw samples go to `runs/<run>/training/memory/`.

## Commands

| command | does |
|---|---|
| `preflight --config Y [--gpu [--hardware-only]] [--allow-metadata-download]` | PASS / FAIL / BLOCKED / NOT TESTED per check; estimates never count as measurements |
| `train --config Y [--skip-preflight]` | new run in `runs/<run.name>/`, gated by preflight |
| `resume --run R [--checkpoint P]` | continue from the latest valid checkpoint |
| `validate`, `test --run R [--step N]` | held-out evaluation of a resident run's checkpoint |
| `verify --run R [--control C]` | resident: `runs/R/verification/`; L2L: test split through L2L incl. the streamed merge (`research/l2l/experiments/t13_final_eval_R.json`) |
| `merge --run R [--step N] [--evaluate]` | tensor-by-tensor merge on the CPU (no GPU, no full model in memory); `--evaluate` runs the merged model through L2L |
| `report`, `remove --run R [--confirm]` | rebuild a run's report; delete a run (dry run by default) |
| `research list`, `research run H.. [--execute]` | hypotheses, their evidence and the commands that test them (dry run by default) |
| `ship [--soup]` | the evidence-derived verdict (`research/l2l/final_report.md`); Soup's ship beside it, informational |
| `soup all --soup-bin S` | Soup diagnostics on our artifacts, raw outputs under `runs/_soup/` |
| `faults list / doc / test` | the silent-failure registry, its document and fault-injection tests |

Exit codes: 0 success, 1 error, 2 a check failed or DON'T SHIP, 3 usage. Every call is logged to
`runs/_cli/history.jsonl`. The older entry points still work unchanged: `scripts/run.py`, `scripts/l2l_lab.py`,
`scripts/merge.py`, `scripts/final_eval.py`, `scripts/profile_run.py`, `scripts/dflash_bench.py`.

## Configuration

One schema and layered YAML. The model-independent defaults are in `configs/profiles/shared.yaml`, which is abstract.
There is one profile per model (`qwen3_0_6b.yaml`, `qwen3_4b.yaml`), and experiments override a few keys:

```yaml
extends: ../profiles/qwen3_4b.yaml      # paths are relative to this file
run:
  name: my_experiment                   # required; names runs/my_experiment/
training:
  micro_batch_size: 6                   # floats need a decimal point: 1.0e-5
```

Unknown keys, wrong types and unsupported combinations are errors. Implemented-but-refused settings stop `train`
instead of falling back. A run saves its fully resolved config with its sha256, the git commit, packages, the dataset
hash, the model revision, the GPU and the host. Selection guide and full schema:
[`docs/configuration.md`](docs/configuration.md).

## Architecture

```
qwen_grpo/
  grpo.py training.py policy.py      GRPO math (advantages, masks, clipped loss), rollout grouping, one update
  sentiment.py model.py               task (prompt, strict reward, splits); bf16 base + fp32 LoRA
  trainer.py runs.py checkpointing.py a training run: checkpoints, resume, memory guard, provenance
  execution/                          backends: resident | L2L (weight_store, layer_stream, backward, prefetch,
                                      generator, output_head, dflash), profiler, memory plan
  streaming/                          disk-backed prompts and rollout records, crash-consistent publication
  export/merge.py                     streamed LoRA merge
  evaluation.py verification.py reporting.py   metrics and CIs, base/adapter/merged/control checks, reports
  budget/                             Step-0 memory/PCIe calculator (no weights loaded)
  diagnostics/                        preflight, invariants, silent-failure registry
  integrations/soup.py                optional Soup CLI diagnostics
  cli.py                              qwen-grpo
  *_lab.py, lab.py                    research lab cases (resident vs candidate; imported only by scripts/l2l_lab.py and tests)
```

The GRPO mathematics lives in `grpo.py` and `training.py`. A backend decides only *how* the model is executed. The
L2L backend streams every decoder layer through two GPU buffers for every micro-batch, decode step and backward
sweep: the schedule is micro-batch-major. One update, tensor by tensor, and the execution design are in
[`docs/architecture.md`](docs/architecture.md).

## Runs

```
runs/<run>/
  report.md summary.json manifest.json config.yaml   report, the same as data, identity/status, resolved config
  training/   training.json (per-step log), training.log (UTC timestamps), memory/ (raw nvidia-smi samples)
  checkpoints/step<N>/   adapter + training_state.pt; step0 is the initial adapter
  streaming/             prompt shards, rollout records (streaming runs)
  evaluation/, verification/, merged/
```

A run's status is one of `running`, `completed`, `failed` (an exception, a budget violation, changed frozen weights)
or `interrupted`. `train` refuses an existing run name unless `run.overwrite: true`. Training runs are not committed;
`runs/qwen` is the tracked reference run.

## Testing

| tier | command | runs |
|---|---|---|
| 1 | `python -m pytest -q tests/test_config.py tests/test_profiles.py tests/test_cli.py tests/silent_failures tests/test_memory_guard.py` | config, CLI, preflight, fault injection (~1–2 min) |
| 2 | `CUDA_VISIBLE_DEVICES="" python -m pytest -q` | the full CPU suite, incl. resident-vs-L2L bitwise tests (~11 min) |
| 3 | `python -m pytest -q tests/test_prefetch.py tests/test_backward.py tests/test_generator.py tests/test_layer_stream.py tests/test_backend.py tests/test_output_head.py tests/test_weight_store.py` | CUDA tests on tiny models (local GPU) |
| 4 | `docs/deployment.md` | Qwen3-4B on the rented GPU; never automatic |

Tests never load pretrained weights, and they fail the session if `runs/`, `results/` or `checkpoints/` change.

## Evidence

| what | where |
|---|---|
| hypotheses H01–H12, protocol, tolerances, runbook | `research/l2l/` |
| raw lab results (each with commit, environment, measurements, status) | `research/l2l/experiments/` |
| final report and requirement statuses (≤ 3 pages, generated) | `research/l2l/final_report.md`, `final_results.json` |
| silent failures, Soup coverage | `docs/silent_failures.md`, `research/soup/` |
| budget predictions | `results/budget/predictions/` |
| development history (tickets 00–13, older run layouts) | `docs/development_history.md` |

## Reproducibility

- Pin the packages in `requirements.txt`. The dataset is `data/sentiment.json`, whose sha256 is recorded in every run.
- Seeds come from the config: `training.seed` for LoRA init, sampling and batch order; `data.subset_seed` for the
  subsets.
- Resuming from a checkpoint reproduces the uninterrupted run bitwise on CPU (tests). Streaming runs also persist the
  sampler state and the rollout records.
- Lab results record the commit and whether the tree was dirty. `qwen-grpo ship` lists the engine files changed since
  each PASS.
- bf16 results depend on batch shape and hardware: compare runs only on the same machine and software (Qwen3-0.6B
  resident vs L2L: `scripts/compare_runs.py`).
