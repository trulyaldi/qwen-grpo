# qwen-grpo

GRPO + LoRA fine-tuning of Qwen3 in non-thinking mode, implemented by hand and checked numerically. Current task: binary sentiment classification on Rotten Tomatoes with Qwen3-0.6B (RTX 3050 Ti, 4 GB). The code accepts any dense Qwen3 model (`configs/experiments/qwen3_4b.yaml`), but only 0.6B has been run. L2L streaming is not implemented yet.

## Layout

```
qwen_grpo/          reusable implementation (no scripts, no experiment hyperparameters)
  grpo.py           group advantages, response masks, token log-probs, clipped GRPO loss
  policy.py         rollout generation, teacher-forced log-probs
  sentiment.py      dataset splits, prompt, strict parser, reward
  model.py          EOS ids, bf16 base + fp32 LoRA, decoder-layer discovery and LoRA target resolution
  config.py         YAML run configuration: defaults, `extends` inheritance, strict validation, hashing, saved configs
  trainer.py        multi-step GRPO + LoRA run: train(cfg, resume=None); logs, checkpoints, final evaluation
  checkpointing.py  adapter save/load/config checks, resumable training state (optimizer, progress, RNG), training lock
  cleanup.py        `remove`: find a run's generated artifacts and delete them only when ownership is certain
  verification.py   base / adapter / merged / LR=0 comparisons, merge equivalence, parameter deltas, reward integrity
  reporting.py      verification JSON, plots and report.md from saved files
  training.py       prompt batching, rollout grouping, rewards, one GRPO optimizer update
  evaluation.py     greedy predictions, metrics, complete-label scores, ROC AUC, paired bootstraps
configs/            default.yaml (the historical last2 B=2 run, parent of every experiment)
  experiments/      one small override file per experiment (qwen3_4b.yaml: untested example)
scripts/run.py      entry point: train [--resume] | validate | test | verify | report | remove --config <yaml>
research/           historical experiments: each script has a main() and writes to results/ or checkpoints/
  baseline.py       greedy base-model evaluation
  train_grpo.py     historical, Stage 1: one full-rank GRPO step on one fp32 matrix, LR=0 control
  train_lora.py     historical, Stage 2: one LoRA GRPO step, precision and merge checks (shares train_grpo.py's constants)
  trl_reference/    independent reference with Hugging Face TRL GRPOTrainer (own results/, checkpoints/, findings.md)
tests/              unit and small integration tests (tiny models; no weights, only tokenizer/config files)
  test_historical.py  saved results/checkpoints still pass the current checks (baseline numbers, provenance, targets)
data/               cached dataset splits (data/sentiment.json)
results/            baseline/<split>.json and <run>/{validation,training}.json: raw predictions and training logs
  benchmarks/       <name>.json (metrics) and <name>.md (report) for the baseline and every GRPO + LoRA run
checkpoints/        LoRA adapters saved by the runs
docs/architecture.md  one GRPO + LoRA step tensor by tensor: shapes, objective, gradient flow, precision
```

## Setup and tests

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt   # torch: cu126 wheel index
.venv/bin/python -m pytest -q
```

## Running experiments

Training and evaluation are driven by one YAML file per experiment. Run from the repository root:

```bash
.venv/bin/python scripts/run.py train    --config configs/experiments/grpo_low_lr.yaml
.venv/bin/python scripts/run.py train    --config configs/default.yaml
.venv/bin/python scripts/run.py train    --config configs/default.yaml --resume checkpoints/default_step20
.venv/bin/python scripts/run.py validate --config configs/default.yaml --checkpoint checkpoints/default_step30
.venv/bin/python scripts/run.py test     --config configs/default.yaml --checkpoint checkpoints/default_step30
.venv/bin/python -m research.baseline --split validation    # base model; writes results/baseline/ and results/benchmarks/baseline.*
.venv/bin/python -m research.trl_reference.train            # then .compare_gradients, .evaluate
```

### YAML configuration

`configs/default.yaml` holds the settings that experiments vary; every other key has a Python default in
`qwen_grpo/config.py`. An experiment is a file in `configs/experiments/` that names its parent and overrides a few keys:

```yaml
extends: ../default.yaml     # path relative to this file; the parent may itself extend another file
run:
  name: grpo_low_lr          # required in every file that extends another, and different from the parent's
training:
  learning_rate: 1.0e-5      # write floats with a decimal point: PyYAML reads 1e-5 as a string
  max_steps: 200
```

Resolution: the parent chain is loaded first; mappings merge key by key, and any other value (scalar, list, `null`, or a
`lora.targets` preset replacing a `{modules, num_layers}` mapping) replaces the parent's. Omitted keys then take the
defaults below. The result is validated as a whole: unknown keys, missing required keys, wrong types (`1` for a float is
fine, `True` for an int is not), unsupported values and inconsistent combinations (e.g. `batch_mode: balanced` with
`batch_size: 3`) are errors, as are a missing parent, a cycle, or a child without its own `run.name`. A file without
`extends` must contain all required keys; full YAMLs written before this scheme (every key spelled out) still load.

To create an experiment: copy one in `configs/experiments/`, give it a new `run.name`, change the keys under study.
`train` refuses a `run.name` that already has outputs (unless `run.overwrite: true`).

| section | keys (default) | notes |
|---|---|---|
| `run` | `name`, `gpu_budget_gib` (4.0), `overwrite` (false) | `name` names every output; the budget is reported, not enforced; `overwrite: true` makes `train` delete `results/<name>/`, `results/benchmarks/<name>.*` and `checkpoints/<name>_step*` first (not with `--resume`) |
| `model` | `name`, `dtype` (bfloat16), `thinking` (false) | `name`: any `Qwen/Qwen3-<size>B`; `dtype`: `bfloat16` or `float32` (base weights); `thinking`: `false` only |
| `lora` | `rank`, `alpha`, `dropout` (0.0), `targets` | `targets`: a preset (`last2-attention`, `last-o-proj`) or `{modules: [q_proj, ...], num_layers: N}`: those Linear layers in each of the last N decoder layers, found on the loaded model |
| `data` | `train_samples`, `val_subset`, `subset_seed` (42) | balanced subsets of the cached splits |
| `training` | `batch_size`, `batch_mode`, `group_size`, `learning_rate`, `max_steps`, `max_attempts`, `clip_eps` (0.2), `temperature` (1.0), `max_new_tokens`, `seed`, `eval_every`, `full_val` | `learning_rate: 0.0` is the negative control |
| `evaluation` | `max_new_tokens` (4), `batch_size` (32), `score_batch_size` (8), `bootstrap_samples` (10000), `bootstrap_seed` (42) | greedy evaluation in `train`, `validate`, `test`; the defaults are the saved baseline's settings |
| `verification` | `split` (validation), `merge_subset` (8), `negative_control` (`checkpoint: null`, `train: false`), `statistics` (`confidence_level: 0.95`, `bootstrap_samples: 2000`, `seed: 42`) | `verify` only; its bootstrap is `statistics`, not `evaluation` |

Keys without a default are required. Every default is the value all runs so far used, so applying defaults changes no
run, and the `yaml_config` saved in older training logs still loads and compares as before
(`tests/test_historical.py` checks that `default.yaml` resolves to the config `default_rerun` was trained with).

**Saved configuration.** `train` writes the fully resolved config (parents merged, defaults filled in) to
`results/<name>/config.yaml`, with its SHA-256 in the first line; the same hash is `config_sha256` in
`results/<name>/training.json` (next to `yaml_config`), in every checkpoint's `training_state.pt` (which also holds the
config) and in the outputs of `validate`/`test` and `verify`. The hash covers the whole resolved config and does not
depend on key order or on how the YAML was split across files. `--config results/<name>/config.yaml` resumes the
run with exactly its original settings even if the files in `configs/` have changed since.

`train` writes `results/<name>/training.json` (with the resolved config), `checkpoints/<name>_step<N>/`, and, with
`training.full_val`, `results/<name>/validation.json` and `results/benchmarks/<name>.{json,md}`, plus
`results/<name>/plots/` (rollout reward, validation-subset accuracy, mixed-reward fraction, gradient norm, loss). The training log also
records an `environment` block: git commit and dirty flag, Python and package versions, dataset name, split seed and
SHA-256 of `data/sentiment.json`, device and GPU. `validate`/`test` check that the
adapter's saved config matches the YAML, evaluate it on the full split with the baseline's prompt, decoding and parser, and
write `results/eval/<checkpoint>/<split>.json` with a paired comparison against the saved baseline predictions.

### Resuming a run

Every checkpoint `checkpoints/<name>_step<N>/` written by `train` (steps that are multiples of `training.eval_every`, and
the last step) holds the adapter plus `training_state.pt`: AdamW state, step and attempt counters, the training log so
far and the CPU/CUDA/Python RNG states. `train --resume checkpoints/<name>_step<N>` with the same YAML reloads all of it,
replays the prompt stream to the saved attempt, and continues; the result equals the uninterrupted run bitwise (checked on
CPU by `tests/test_trainer.py` and on the GPU for the default config). Resume refuses a different config, naming the
keys that differ (only the `verification` section and `run.overwrite` may differ), and refuses to overwrite checkpoints later than `<N>`. Checkpoints from before this
feature (all those committed so far) hold the adapter only and cannot resume; they evaluate as before.

### Verifying a checkpoint

```bash
.venv/bin/python scripts/run.py verify --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>
.venv/bin/python scripts/run.py report --config configs/default.yaml --checkpoint checkpoints/<run>_step<N>   # no model loaded
```

`verify` never retrains the checkpoint. On `verification.split` it compares the base model (the saved baseline is reused
after an in-process drift check), the adapter-attached model, the merged model (PEFT `merge_and_unload`, also checked in fp32
on CPU) and, if configured, an LR=0 control. It writes `results/verification/<checkpoint>/`: `evaluation/<model>.json` (raw
predictions, label scores, metrics), `verification/*.json` (parameter deltas, policy change, merge equivalence, negative
control, paired bootstrap CIs, reward integrity, training diagnostics), `plots/` and `report.md`. `report` rebuilds `plots/`
and `report.md` from those files.

LR=0 control: set `verification.negative_control.checkpoint` to an existing control of the same config (its training log is
checked), or `train: true` to run one first as `<run.name>_lr0`. A second `verify` with `train: true` refuses because
`<run.name>_lr0` then exists; point `checkpoint:` at the control it produced instead. Training plots need the run's
`results/<run.name>/training.json`; runs from this version on also save their initial adapter as `checkpoints/<run>_step0`.

Historical runs (before YAML configs) were launched with command-line flags; their exact settings are in the `config` block
of each training log.

### Removing a run's artifacts

```bash
.venv/bin/python scripts/run.py remove --config configs/qwen.yaml --checkpoint checkpoints/qwen_step30            # dry run
.venv/bin/python scripts/run.py remove --config configs/qwen.yaml --checkpoint checkpoints/qwen_step30 --confirm  # delete
```

Without `--confirm` it only lists what belongs to the run, with sizes and the total:

```
run 'qwen' (dry run):
   519.9 KiB  checkpoints/qwen_step0/
     1.8 MiB  checkpoints/qwen_step10/
     ...
     1.3 MiB  results/verification/qwen_step30/
     8.1 MiB  total, 9 paths
dry run: nothing deleted; add --confirm to delete these paths
```

A run `<N>` owns exactly `checkpoints/<N>_step<k>/`, `results/<N>/` (logs, predictions, plots),
`results/benchmarks/<N>.{json,md}`, `results/eval/<N>_step<k>/` and `results/verification/<N>_step<k>/`, matched by full
name (so `qwen` never matches `qwen_lr0`, `qwen.v2` or `qwen_step3_step10`). The whole run is removed; the checkpoint
argument identifies it. Nothing else is touched: no parent directory, config, dataset or baseline. The command refuses,
listing every reason, when:

- the checkpoint is not `checkpoints/<run.name>_step<k>`, or the YAML differs from the one in `results/<N>/training.json`;
- a checkpoint does not match the YAML's LoRA settings, or is neither listed in the training log nor resumable;
- an evaluation or verification output names a checkpoint of another run;
- another file (other runs' results, `configs/`, `README.md`, `docs/`, `research/`) mentions one of the paths, e.g. an LR=0
  control used by another run's `verify`;
- a path is, contains, or passes through a symbolic link;
- a path is committed to git (deleting it would free no space; use `git rm` deliberately instead);
- the run is training: `train` holds `results/.<N>.lock` with its PID for the whole run (a lock left by a dead process is
  removed with the run).

`validate`, `test` and `verify` hold no lock, so do not run `remove` on a run while evaluating it.

## Evidence

| Question | Where |
|---|---|
| Base model metrics and AUC | `results/benchmarks/baseline.{json,md}`; raw predictions `results/baseline/{validation,test}.json` |
| Multi-step runs: full-validation metrics, AUC, bootstraps vs base | `results/benchmarks/<run>.{json,md}`; per-step logs and raw predictions in `results/<run>/` |
| Implementation audit and the POSITIVE-drift analysis | `results/grpo_lora_audit.md` |
| Independent TRL reference: gradient equivalence, training, evaluation | `research/trl_reference/findings.md` and `research/trl_reference/results/` |
| Adapters | `checkpoints/<run>_step<N>/`, `research/trl_reference/checkpoints/` |

`results/grpo_lora_audit.md` and `research/trl_reference/findings.md` cite file paths and line numbers from before the `qwen_grpo/` + `research/` reorganization. Read them against tag `pre-refactor` (commit `a038e17`). The code they describe was moved, not changed.
