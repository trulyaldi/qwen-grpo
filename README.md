# qwen-grpo

GRPO + LoRA fine-tuning of Qwen3 in non-thinking mode, implemented by hand and checked numerically. Current task: binary sentiment classification on Rotten Tomatoes with Qwen3-0.6B. L2L streaming and Qwen3-4B are not implemented yet.

## Layout

```
qwen_grpo/          reusable implementation (no scripts, no experiment hyperparameters)
  grpo.py           group advantages, response masks, token log-probs, clipped GRPO loss
  policy.py         rollout generation, teacher-forced log-probs
  sentiment.py      dataset splits, prompt, strict parser, reward
  model.py          base model id, EOS ids, bf16 base + fp32 LoRA (config, targets, inspection)
  config.py         YAML run configuration: strict loading and validation
  verification.py   base / adapter / merged / LR=0 comparisons, merge equivalence, parameter deltas, reward integrity
  reporting.py      verification JSON, plots and report.md from saved files
  training.py       prompt batching, rollout grouping, rewards, one GRPO optimizer update
  evaluation.py     greedy predictions, metrics, complete-label scores, ROC AUC, paired bootstraps
configs/            YAML run configurations (default.yaml)
scripts/run.py      entry point: train | validate | test | verify | report --config <yaml>
research/           experiments: each script has a main() and writes to results/ or checkpoints/
  baseline.py       greedy base-model evaluation
  train_multistep.py  multi-step GRPO + LoRA training loop, train(cfg); run via scripts/run.py
  train_grpo.py     historical, Stage 1: one full-rank GRPO step on one fp32 matrix, LR=0 control
  train_lora.py     historical, Stage 2: one LoRA GRPO step, precision and merge checks (shares train_grpo.py's constants)
  trl_reference/    independent reference with Hugging Face TRL GRPOTrainer (own results/, checkpoints/, findings.md)
tests/              unit and small integration tests (tiny models; no 0.6B weights except the tokenizer)
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
.venv/bin/python scripts/run.py validate --config configs/default.yaml --checkpoint checkpoints/default_step30
.venv/bin/python scripts/run.py test     --config configs/default.yaml --checkpoint checkpoints/default_step30
.venv/bin/python -m research.baseline --split validation    # base model; writes results/baseline/ and results/benchmarks/baseline.*
.venv/bin/python -m research.trl_reference.train            # then .compare_gradients, .evaluate
```

For a new experiment, copy the YAML and change `run.name` (training refuses to overwrite existing outputs of that name).
`train` writes `results/<name>/training.json` (with the exact YAML config), `checkpoints/<name>_step<N>/`, and, with
`training.full_val`, `results/<name>/validation.json` and `results/benchmarks/<name>.{json,md}`. `validate`/`test` check that the
adapter's saved config matches the YAML, evaluate it on the full split with the baseline's prompt, decoding and parser, and
write `results/eval/<checkpoint>/<split>.json` with a paired comparison against the saved baseline predictions.

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

## Evidence

| Question | Where |
|---|---|
| Base model metrics and AUC | `results/benchmarks/baseline.{json,md}`; raw predictions `results/baseline/{validation,test}.json` |
| Multi-step runs: full-validation metrics, AUC, bootstraps vs base | `results/benchmarks/<run>.{json,md}`; per-step logs and raw predictions in `results/<run>/` |
| Implementation audit and the POSITIVE-drift analysis | `results/grpo_lora_audit.md` |
| Independent TRL reference: gradient equivalence, training, evaluation | `research/trl_reference/findings.md` and `research/trl_reference/results/` |
| Adapters | `checkpoints/<run>_step<N>/`, `research/trl_reference/checkpoints/` |

`results/grpo_lora_audit.md` and `research/trl_reference/findings.md` cite file paths and line numbers from before the `qwen_grpo/` + `research/` reorganization. Read them against tag `pre-refactor` (commit `a038e17`). The code they describe was moved, not changed.
