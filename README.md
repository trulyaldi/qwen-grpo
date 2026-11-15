# qwen-grpo

GRPO + LoRA fine-tuning of Qwen3 in non-thinking mode, implemented by hand and checked numerically. Current task: binary sentiment classification on Rotten Tomatoes with Qwen3-0.6B (RTX 3050 Ti, 4 GB). The code accepts any dense Qwen3 model (`configs/experiments/qwen3_4b.yaml`), but only 0.6B has been run. L2L streaming is not implemented yet.

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # torch: cu126 wheel index
.venv/bin/python -m pytest -q                                                     # CPU, tiny models, no weights

.venv/bin/python scripts/run.py train    --config configs/experiments/grpo_32.yaml   # writes runs/<run>/
.venv/bin/python scripts/run.py validate --run grpo_32          # held-out validation split, latest checkpoint
.venv/bin/python scripts/run.py verify   --run grpo_32          # base / adapter / merged / LR=0 evidence
.venv/bin/python scripts/run.py report   --run grpo_32          # rebuild plots, summary.json, report.md (no GPU)
.venv/bin/python scripts/run.py remove   --run grpo_32          # dry run: what would be deleted, and its size
```

`--step N` picks a checkpoint for `validate`, `test` and `verify` (default: the latest). `test` evaluates the test split;
keep it for the final report. Each command prints where it wrote; open `runs/<run>/report.md` to read a run.

## One run, one directory

Everything a run produces is under `runs/<run>/` (`<run>` = `run.name` of its YAML):

```
runs/<run>/
  report.md            the report to read: configuration, progress, diagnostics, base vs trained, drift, verification, memory
  summary.json         the same as data (metrics, statuses, paths relative to the run directory)
  manifest.json        identity: unique id, creation time, config sha256, git commit, status history, list of files
  config.yaml          the fully resolved config (train --config accepts it)
  plots/               reward, validation_accuracy, prediction_bias, gradient_norm, mixed_reward_fraction, loss
  training/            training.json (per-step log, subset evaluations, final full-validation comparison, environment),
                       training.log (console output), validation.json (final full-validation predictions)
  checkpoints/step<N>/ adapter + training_state.pt (resumable); step0 is the initial adapter
  evaluation/step<N>/  validation.json, test.json (from validate / test)
  verification/step<N>/  report.md, plots/, evidence/ (raw evaluations and every check as JSON)
```

`train` writes `config.yaml` and `manifest.json` first, then the checkpoints and log, then plots, `summary.json` and
`report.md`. `validate`, `test` and `verify` add their results under the checkpoint's step and rebuild the summary and
report; results of other checkpoints are kept. `report --run` rebuilds them from the saved files alone. A section whose
measurement does not exist says so ("not run", "not recorded", `null` in the summary); nothing is filled in.

**Status** (`manifest.json`, top of `report.md`): `running`, `completed`, `failed` (an exception, e.g. frozen weights
changed), or `interrupted` (Ctrl-C, or a run recorded as running whose process is gone, e.g. killed). An interrupted run
keeps its resumable checkpoints, and its report shows the progress saved with the latest one.

**Names.** `train` refuses a `run.name` that `runs/` already has (set `run.overwrite: true` to delete that run directory
first) and a name used by a run from before `runs/` existed (below), so two runs never share a name.

**Compare runs:** `summary.json` has the same keys for every run, e.g.
`jq '{run, status, full_validation, verification}' runs/*/summary.json`.

## Code layout

```
qwen_grpo/          implementation (no experiment hyperparameters)
  grpo.py           group advantages, response masks, token log-probs, clipped GRPO loss
  policy.py         rollout generation, teacher-forced log-probs
  training.py       prompt batching, rollout grouping, rewards, one GRPO optimizer update
  sentiment.py      dataset splits, prompt, strict parser, reward
  model.py          bf16 base + fp32 LoRA, decoder-layer discovery and LoRA target resolution
  config.py         YAML configuration: defaults, `extends`, strict validation, hashing
  trainer.py        a multi-step GRPO + LoRA run: train(cfg, resume=None)
  evaluation.py     greedy predictions, metrics, label scores, ROC AUC, paired bootstraps
  verification.py   base / adapter / merged / LR=0 comparisons, merge equivalence, parameter deltas, reward integrity
  reporting.py      plots, verification reports, the run's summary.json and report.md, from saved files
  runs.py           where every artifact lives; checkpoint lookup; run identity and status
  checkpointing.py  adapter load/checks, resumable training state, training lock
  cleanup.py        `remove`: deletes a run's artifacts only when ownership is certain
configs/            default.yaml (parent of every experiment), experiments/*.yaml (small overrides)
scripts/run.py      the command line: train | validate | test | verify | report | remove
research/           historical experiments (baseline.py writes the shared results/baseline/), TRL reference
tests/              unit and integration tests; tests/conftest.py forbids loading weights and touching runs/, results/, checkpoints/
docs/architecture.md  one GRPO + LoRA step tensor by tensor: shapes, objective, gradient flow, precision
```

## Configuration

`configs/default.yaml` holds the settings experiments vary; every other key has a Python default in
`qwen_grpo/config.py`. An experiment is a file in `configs/experiments/` that names its parent and overrides a few keys:

```yaml
extends: ../default.yaml     # path relative to this file; the parent may itself extend another file
run:
  name: grpo_low_lr          # required in every file that extends another, and different from the parent's
training:
  learning_rate: 1.0e-5      # write floats with a decimal point: PyYAML reads 1e-5 as a string
  max_steps: 200
```

Mappings merge key by key; any other value (scalar, list, `null`, a `lora.targets` preset) replaces the parent's.
Omitted keys take the defaults below. The result is validated as a whole: unknown keys, missing required keys, wrong
types, unsupported values and inconsistent combinations are errors, as are a missing parent, a cycle, or a child
without its own `run.name`. Full YAMLs with every key spelled out also load.

| section | keys (default) | notes |
|---|---|---|
| `run` | `name`, `gpu_budget_gib` (4.0), `overwrite` (false) | `name` names the run directory; the budget is reported, not enforced; `overwrite: true` makes `train` delete `runs/<name>/` first (never with `--resume`) |
| `model` | `name`, `dtype` (bfloat16), `thinking` (false) | `name`: any `Qwen/Qwen3-<size>B`; `dtype`: `bfloat16` or `float32` (base weights); `thinking`: `false` only |
| `lora` | `rank`, `alpha`, `dropout` (0.0), `targets` | a preset (`last2-attention`, `last-o-proj`) or `{modules: [q_proj, ...], num_layers: N}`: those Linear layers in each of the last N decoder layers of the loaded model |
| `data` | `train_samples`, `val_subset`, `subset_seed` (42) | balanced subsets of the cached splits |
| `training` | `batch_size`, `batch_mode`, `group_size`, `learning_rate`, `max_steps`, `max_attempts`, `clip_eps` (0.2), `temperature` (1.0), `max_new_tokens`, `seed`, `eval_every`, `full_val` | `learning_rate: 0.0` is the negative control |
| `evaluation` | `max_new_tokens` (4), `batch_size` (32), `score_batch_size` (8), `bootstrap_samples` (10000), `bootstrap_seed` (42) | greedy evaluation in `train`, `validate`, `test`; the defaults are the saved baseline's settings |
| `verification` | `split` (validation), `merge_subset` (8), `negative_control` (`checkpoint: null`, `train: false`), `statistics` (`confidence_level: 0.95`, `bootstrap_samples: 2000`, `seed: 42`) | `verify` only |

Every default is the value all runs so far used. The SHA-256 of the resolved config is in `config.yaml`'s first line,
`manifest.json`, `training.json`, every checkpoint's training state, and every `validate`/`test`/`verify` output.

## Resuming

Every checkpoint after step 0 holds `training_state.pt` (AdamW state, counters, the log so far, RNG states). Resume with
the run's own saved config:

```bash
.venv/bin/python scripts/run.py train --config runs/<run>/config.yaml --resume runs/<run>/checkpoints/step<N>
```

The result equals the uninterrupted run bitwise (checked on CPU by `tests/test_trainer.py`). The run keeps its id; the
manifest's status history records the interruption and the resume, and `training.log` is appended to. Resume refuses a
different config (naming the differing keys; only `verification` and `run.overwrite` may differ), a checkpoint of
another run, and anything that would overwrite later checkpoints.

## Verification

`verify` never retrains. On `verification.split` it compares the base model (the saved baseline is reused after an
in-process drift check), the adapter-attached model, the merged model (PEFT `merge_and_unload`, also checked in fp32 on
CPU) and, if configured, an LR=0 control, and records parameter deltas, policy change, merge equivalence, paired
bootstrap CIs, reward integrity and training diagnostics. LR=0 control: set `verification.negative_control.checkpoint` to
an existing control checkpoint of the same config (its training log is checked), or `train: true` to train one first as
the separate run `<run>_lr0`.

## Removing a run

```bash
.venv/bin/python scripts/run.py remove --run <run>             # dry run: lists runs/<run>/ with sizes and the total
.venv/bin/python scripts/run.py remove --run <run> --confirm   # deletes it
```

It deletes `runs/<run>/` and a stale `runs/.<run>.lock`, nothing else (no config, dataset, baseline or other run), and
refuses, listing every reason, when: the name is not a run name; `manifest.json` is missing or names another run; the
run is training (`train` holds `runs/.<run>.lock` with its PID); a path is or contains a symbolic link, or `runs/`
passes through one; a file is committed to git; or another file (another run, `configs/`, `README.md`, `docs/`,
`research/`, `results/`) mentions the run's paths, e.g. a run whose `verify` uses this run as its LR=0 control.
`validate`, `test` and `verify` hold no lock: do not remove a run while evaluating it.

## Runs from before `runs/`

Earlier runs keep their original layout and are never moved or rewritten by the new commands: `checkpoints/<run>_step<N>/`,
`results/<run>/` (training log, predictions, plots), `results/benchmarks/<run>.{json,md}`, `results/eval/`,
`results/verification/`. Use the path form for them; their outputs keep going to `results/eval/` and `results/verification/`:

```bash
.venv/bin/python scripts/run.py verify --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
.venv/bin/python scripts/run.py report --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
.venv/bin/python scripts/run.py remove --config <its yaml> --checkpoint checkpoints/<run>_step<N> [--confirm]
```

They cannot be resumed into `runs/`. The shared base-model baseline is `results/baseline/` (written by
`python -m research.baseline --split validation`).

| Question | Where |
|---|---|
| Base model metrics and AUC | `results/benchmarks/baseline.{json,md}`; raw predictions `results/baseline/{validation,test}.json` |
| Earlier multi-step runs | `results/benchmarks/<run>.{json,md}`, `results/<run>/`, `results/verification/<run>_step<N>/report.md` |
| Implementation audit and the POSITIVE-drift analysis | `results/grpo_lora_audit.md` |
| Independent TRL reference | `research/trl_reference/findings.md` and `research/trl_reference/results/` |

`results/grpo_lora_audit.md` and `research/trl_reference/findings.md` cite file paths and line numbers from before the `qwen_grpo/` + `research/` reorganization. Read them against tag `pre-refactor` (commit `a038e17`). The code they describe was moved, not changed.
