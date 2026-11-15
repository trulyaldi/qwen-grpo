# qwen-grpo

GRPO + LoRA fine-tuning of Qwen3 in non-thinking mode, implemented by hand and checked numerically. Current task: binary sentiment classification on Rotten Tomatoes with Qwen3-0.6B. L2L streaming and Qwen3-4B are not implemented yet.

## Layout

```
qwen_grpo/          reusable implementation (no scripts, no experiment hyperparameters)
  grpo.py           group advantages, response masks, token log-probs, clipped GRPO loss
  policy.py         rollout generation, teacher-forced log-probs
  sentiment.py      dataset splits, prompt, strict parser, reward
  model.py          base model id, EOS ids, bf16 base + fp32 LoRA (config, targets, inspection)
  training.py       prompt batching, rollout grouping, rewards, one GRPO optimizer update
  evaluation.py     greedy predictions, metrics, complete-label scores, ROC AUC, paired bootstraps
research/           experiments: each script has a main() and writes to results/ or checkpoints/
  baseline.py       greedy base-model evaluation
  train_multistep.py  current training entry point: multi-step GRPO + LoRA runs (CLI)
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

Run from the repository root as modules:

```bash
.venv/bin/python -m research.baseline
.venv/bin/python -m research.train_multistep                                   # last o_proj, B=1
.venv/bin/python -m research.train_multistep --lora-targets last2-attention    # q/v/o of layers 26-27, B=1
.venv/bin/python -m research.train_multistep --lora-targets last2-attention --batch-size 2 --batch-mode balanced
.venv/bin/python -m research.train_multistep --lora-targets last2-attention --batch-size 2 --batch-mode unconstrained
.venv/bin/python -m research.trl_reference.train            # then .compare_gradients, .evaluate
```

`python -m research.train_multistep --help` lists all options. The exact configuration of every saved run, including smoke and LR=0 runs, is in the `config` block of its `results/<run>/training.json`.

## Evidence

| Question | Where |
|---|---|
| Base model metrics and AUC | `results/benchmarks/baseline.{json,md}`; raw predictions `results/baseline/{validation,test}.json` |
| Multi-step runs: full-validation metrics, AUC, bootstraps vs base | `results/benchmarks/<run>.{json,md}`; per-step logs and raw predictions in `results/<run>/` |
| Implementation audit and the POSITIVE-drift analysis | `results/grpo_lora_audit.md` |
| Independent TRL reference: gradient equivalence, training, evaluation | `research/trl_reference/findings.md` and `research/trl_reference/results/` |
| Adapters | `checkpoints/<run>_step<N>/`, `research/trl_reference/checkpoints/` |

`results/grpo_lora_audit.md` and `research/trl_reference/findings.md` cite file paths and line numbers from before the `qwen_grpo/` + `research/` reorganization. Read them against tag `pre-refactor` (commit `a038e17`). The code they describe was moved, not changed.
