# qwen-grpo

GRPO + LoRA fine-tuning of Qwen3 in non-thinking mode, implemented by hand and checked numerically. Current task: binary sentiment classification on Rotten Tomatoes with Qwen3-0.6B (RTX 3050 Ti, 4 GB). The code accepts any dense Qwen3 model (`configs/qwen3_4b.yaml`), but only 0.6B has been run. L2L streaming is not implemented yet.

## Layout

```
qwen_grpo/          reusable implementation (no scripts, no experiment hyperparameters)
  grpo.py           group advantages, response masks, token log-probs, clipped GRPO loss
  policy.py         rollout generation, teacher-forced log-probs
  sentiment.py      dataset splits, prompt, strict parser, reward
  model.py          EOS ids, bf16 base + fp32 LoRA, decoder-layer discovery and LoRA target resolution
  config.py         YAML run configuration: strict loading, validation, normalization of saved configs
  trainer.py        multi-step GRPO + LoRA run: train(cfg, resume=None); logs, checkpoints, final evaluation
  checkpointing.py  adapter save/load/config checks, resumable training state (optimizer, progress, RNG)
  reporting.py      training diagnostics plots from a training log
  training.py       prompt batching, rollout grouping, rewards, one GRPO optimizer update
  evaluation.py     greedy predictions, metrics, complete-label scores, ROC AUC, paired bootstraps
configs/            YAML run configurations: default.yaml (the historical last2 B=2 run), qwen3_4b.yaml (untested example)
scripts/run.py      entry point: train [--resume] | validate | test --config <yaml>
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

Training and evaluation are driven by one YAML file. `configs/default.yaml` lists every setting; `qwen_grpo/config.py`
rejects unknown, missing or ill-typed keys and values the code does not support. Run from the repository root:

```bash
.venv/bin/python scripts/run.py train    --config configs/default.yaml
.venv/bin/python scripts/run.py train    --config configs/default.yaml --resume checkpoints/default_step20
.venv/bin/python scripts/run.py validate --config configs/default.yaml --checkpoint checkpoints/default_step30
.venv/bin/python scripts/run.py test     --config configs/default.yaml --checkpoint checkpoints/default_step30
.venv/bin/python -m research.baseline --split validation    # base model; writes results/baseline/ and results/benchmarks/baseline.*
.venv/bin/python -m research.trl_reference.train            # then .compare_gradients, .evaluate
```

### YAML options

| section | keys | notes |
|---|---|---|
| `run` | `name`, `gpu_budget_gib` (optional, 4.0), `overwrite` (optional, false) | `name` names every output; the budget is reported, not enforced; `overwrite: true` makes `train` delete `results/<name>/`, `results/benchmarks/<name>.*` and `checkpoints/<name>_step*` first (not with `--resume`) |
| `model` | `name`, `dtype`, `thinking` | `name`: any `Qwen/Qwen3-<size>B`; `dtype`: `bfloat16` or `float32` (base weights); `thinking`: `false` only |
| `lora` | `rank`, `alpha`, `dropout`, `targets` | `targets`: a preset (`last2-attention`, `last-o-proj`) or `{modules: [q_proj, ...], num_layers: N}`: those Linear layers in each of the last N decoder layers, found on the loaded model |
| `data` | `train_samples`, `val_subset`, `subset_seed` | balanced subsets of the cached splits |
| `training` | `batch_size`, `batch_mode`, `group_size`, `learning_rate`, `max_steps`, `max_attempts`, `clip_eps`, `temperature`, `max_new_tokens`, `seed`, `eval_every`, `full_val` | `learning_rate: 0.0` is the negative control |
| `evaluation` (optional) | `max_new_tokens` (4), `batch_size` (32), `score_batch_size` (8), `bootstrap_samples` (10000), `bootstrap_seed` (42) | greedy evaluation in `train`, `validate`, `test`; the defaults are the saved baseline's settings |

Optional keys were added after the first YAML runs. Their defaults are the values those runs used, so older configs and
the `yaml_config` saved in older training logs still load and compare as before.

For a new experiment, copy the YAML and change `run.name` (training refuses to overwrite existing outputs of that name unless `run.overwrite: true`).
`train` writes `results/<name>/training.json` (with the exact YAML config), `checkpoints/<name>_step<N>/`, and, with
`training.full_val`, `results/<name>/validation.json` and `results/benchmarks/<name>.{json,md}`. The training log also
records an `environment` block: git commit and dirty flag, Python and package versions, dataset name, split seed and
SHA-256 of `data/sentiment.json`, device and GPU. `validate`/`test` check that the
adapter's saved config matches the YAML, evaluate it on the full split with the baseline's prompt, decoding and parser, and
write `results/eval/<checkpoint>/<split>.json` with a paired comparison against the saved baseline predictions.

### Resuming a run

Every checkpoint `checkpoints/<name>_step<N>/` written by `train` (steps that are multiples of `training.eval_every`, and
the last step) holds the adapter plus `training_state.pt`: AdamW state, step and attempt counters, the training log so
far and the CPU/CUDA/Python RNG states. `train --resume checkpoints/<name>_step<N>` with the same YAML reloads all of it,
replays the prompt stream to the saved attempt, and continues; the result equals the uninterrupted run bitwise (checked on
CPU by `tests/test_trainer.py` and on the GPU for the default config). Resume refuses a different config (only
`run.overwrite` may differ) and refuses to overwrite checkpoints later than `<N>`. Checkpoints from before this
feature (all those committed so far) hold the adapter only and cannot resume; they evaluate as before.

### Validation and negative control

`train` evaluates the validation subset at step 0 (the zero adapter, i.e. the base model) and at every checkpoint, and with
`training.full_val` compares the final checkpoint with the base model on the full validation split (accuracy, macro-F1,
ROC AUC, paired bootstrap CIs, prediction flips). It also writes `results/<name>/plots/`: rollout reward, validation-subset
accuracy, mixed-reward fraction, gradient norm and loss per step. `validate` repeats the base-vs-trained comparison for
any saved checkpoint. For an LR=0 negative control, train a copy of the config with `learning_rate: 0.0` and its own
`run.name`; the training log then records whether parameters and subset predictions stayed unchanged.

A separate `verify` workflow (adapter vs merged model, merge equivalence, reward integrity) existed until it was removed in
the cleanup after commit `f403e12`; its outputs remain in `results/verification/` as historical evidence and can be
regenerated by checking out that commit.

Historical runs (before YAML configs) were launched with command-line flags; their exact settings are in the `config` block
of each training log.

## Evidence

| Question | Where |
|---|---|
| Base model metrics and AUC | `results/benchmarks/baseline.{json,md}`; raw predictions `results/baseline/{validation,test}.json` |
| Multi-step runs: full-validation metrics, AUC, bootstraps vs base | `results/benchmarks/<run>.{json,md}`; per-step logs and raw predictions in `results/<run>/` |
| Implementation audit and the POSITIVE-drift analysis | `results/grpo_lora_audit.md` |
| Independent TRL reference: gradient equivalence, training, evaluation | `research/trl_reference/findings.md` and `research/trl_reference/results/` |
| Adapters | `checkpoints/<run>_step<N>/`, `research/trl_reference/checkpoints/` |

`results/grpo_lora_audit.md` and `research/trl_reference/findings.md` cite file paths and line numbers from before the `qwen_grpo/` + `research/` reorganization. Read them against tag `pre-refactor` (commit `a038e17`). The code they describe was moved, not changed.
