# Run `qwen` — completed

id `qwen-20261116T075418Z-73f5ff3f`, created 2026-11-16T07:54:18+00:00, config sha256 `73f5ff3f4ec175b7…`, git `53bc8b2843d8` (uncommitted changes).

## 1. Configuration

- Model Qwen/Qwen3-0.6B (bfloat16 base, fp32 LoRA), thinking False.
- LoRA r=32, α=64, dropout 0.0, targets `last2-attention`.
- GRPO: B=5 (unconstrained), G=6, lr 0.0005, max 100 steps / 1000 attempts, clip ε 0.2, 1 update(s) per rollout batch, temperature 1.0, rollout max_new_tokens 4, seed 42.
- Data: 512 training prompts, validation subset 512 (subset seed 42). Full settings: `config.yaml`.

## 2. Training progress

- 100/100 optimizer steps from 142 rollout attempts (42 zero-signal batches skipped); training time 237 s. Source: `runs/qwen/training/training.json`.
- Checkpoints: step 0, step 10 (resumable), step 20 (resumable), step 30 (resumable), step 40 (resumable), step 50 (resumable), step 60 (resumable), step 70 (resumable), step 80 (resumable), step 90 (resumable), step 100 (resumable); latest `checkpoints/step100/`.

## 3. Reward and gradient diagnostics

- Frozen base tensors changed: 0; adapter equal to its initialization: False.
- Rising rollout reward is measured on the policy's own samples of training prompts: it is not held-out evidence.
- Plots: [reward.png](plots/reward.png), [validation_accuracy.png](plots/validation_accuracy.png), [mixed_reward_fraction.png](plots/mixed_reward_fraction.png), [gradient_norm.png](plots/gradient_norm.png), [loss.png](plots/loss.png), [prediction_bias.png](plots/prediction_bias.png)

## 4. Base vs trained (full validation, at the end of training)

| metric | base | trained |
|---|---:|---:|
| n | 1066 | 1066 |
| accuracy | 0.7486 | 0.5281 |
| macro_f1 | 0.7493 | 0.3958 |
| recall_NEGATIVE | 0.7824 | 0.0600 |
| recall_POSITIVE | 0.7148 | 0.9962 |
| predicted_positive_rate | 0.4662 | 0.9681 |
| invalid_rate | 0.0028 | 0.0009 |
| auc | 0.8277 | 0.8900 |

- Accuracy trained − base -0.2205, paired bootstrap 95% CI [-0.2608, -0.1801]; AUC diff +0.0623, CI [+0.0492, +0.0763]. One training seed: CIs cover evaluation sampling only.

## 5. Held-out evaluations (`validate` / `test`)

| checkpoint | split | n | accuracy | macro-F1 | recall NEG | recall POS | pred. POS rate | AUC | Δ acc vs base [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| step90 | [test](evaluation/step90/test.json) | 1066 | 0.7964 | 0.7952 | 0.8856 | 0.7073 | 0.4109 | 0.8874 | +0.0638 [+0.0422, +0.0854] |
| step90 | [validation](evaluation/step90/validation.json) | 1066 | 0.8096 | 0.8083 | 0.9024 | 0.7167 | 0.4071 | 0.8929 | +0.0610 [+0.0394, +0.0826] |

## 6. POSITIVE prediction rate and classification drift

- Validation subset (n=512): predicted-POSITIVE rate step 0 0.469, step 10 0.359, step 20 0.393, step 30 0.389, step 40 0.373, step 50 0.598, step 60 0.504, step 70 0.758, step 80 0.371, step 90 0.422, step 100 0.959; accuracy step 0 0.736, step 10 0.738, step 20 0.764, step 30 0.768, step 40 0.775, step 50 0.797, step 60 0.793, step 70 0.707, step 80 0.797, step 90 0.805, step 100 0.537. Step 0 is the base model (zero adapter).
- Full validation: predicted-POSITIVE rate base 0.4662 → trained 0.9681; recall NEGATIVE 0.7824 → 0.0600, POSITIVE 0.7148 → 0.9962.

## 7. Verification

- Not run yet (`scripts/run.py verify --run <name>`).

## 8. Hardware and memory

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU; peak reserved 6.30 GiB (budget 4.0 GiB, within: False); torch allocator statistics, excluding the CUDA context.

## 9. Files

- `config.yaml` (resolved config), `manifest.json` (identity, status history, artifacts), `summary.json` (this report as data)
- `training/` (training.json, training.log, validation.json), `checkpoints/step<N>/`, `evaluation/step<N>/`, `verification/step<N>/` (report.md, evidence/)
