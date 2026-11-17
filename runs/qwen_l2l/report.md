# Run `qwen_l2l` — completed

id `qwen_l2l-20261117T071109Z-a13007a6`, created 2026-11-17T07:11:09+00:00, config sha256 `a13007a61c0599df…`, git `5300146b4d98`.

## 1. Configuration

- Model Qwen/Qwen3-0.6B (bfloat16 base, fp32 LoRA), non-thinking.
- LoRA r=32, α=64, targets `last2-attention`.
- GRPO: B=5 (shuffled sampler), G=6, lr 0.0005, max 100 steps / 1000 attempts, clip ε 0.2, 1 update(s) per rollout batch, micro-batch whole batch, temperature 1, rollout max_new_tokens 4, seed 42.
- Data: 512 training prompts, validation subset 512 (subset seed 42). Full settings: `config.yaml`.

## 2. Training progress

- 100/100 optimizer steps from 111 rollout attempts (11 zero-signal batches skipped); training time 360 s. Source: `runs/qwen_l2l/training/training.json`.
- Checkpoints: step 0, step 10 (resumable), step 20 (resumable), step 30 (resumable), step 40 (resumable), step 50 (resumable), step 60 (resumable), step 70 (resumable), step 80 (resumable), step 90 (resumable), step 100 (resumable); latest `checkpoints/step100/`.

## 3. Reward and gradient diagnostics

- Frozen base tensors changed: 0; adapter equal to its initialization: False.
- Rising rollout reward is measured on the policy's own samples of training prompts: it is not held-out evidence.
- Plots: [reward.png](plots/reward.png), [validation_accuracy.png](plots/validation_accuracy.png), [mixed_reward_fraction.png](plots/mixed_reward_fraction.png), [gradient_norm.png](plots/gradient_norm.png), [loss.png](plots/loss.png), [prediction_bias.png](plots/prediction_bias.png)

## 4. Base vs trained (full validation, at the end of training)

| metric | base | trained |
|---|---:|---:|
| n | 1066 | 1066 |
| accuracy | 0.7486 | 0.7167 |
| macro_f1 | 0.7493 | 0.7014 |
| recall_NEGATIVE | 0.7824 | 0.4878 |
| recall_POSITIVE | 0.7148 | 0.9456 |
| predicted_positive_rate | 0.4662 | 0.7289 |
| invalid_rate | 0.0028 | 0.0009 |
| auc | 0.8277 | 0.8943 |

- Accuracy trained − base -0.0319, paired bootstrap 95% CI [-0.0629, -0.0009]; AUC diff +0.0666, CI [+0.0530, +0.0811]. One training seed: CIs cover evaluation sampling only.

## 5. Held-out evaluations (`validate` / `test`)

- None run yet (`scripts/run.py validate --run <name>`).

## 6. POSITIVE prediction rate and classification drift

- Validation subset (n=512): predicted-POSITIVE rate step 0 0.469, step 10 0.359, step 20 0.375, step 30 0.367, step 40 0.434, step 50 0.381, step 60 0.145, step 70 0.947, step 80 0.117, step 90 0.127, step 100 0.717; accuracy step 0 0.736, step 10 0.738, step 20 0.758, step 30 0.762, step 40 0.793, step 50 0.791, step 60 0.633, step 70 0.547, step 80 0.605, step 90 0.611, step 100 0.729. Step 0 is the base model (zero adapter).
- Full validation: predicted-POSITIVE rate base 0.4662 → trained 0.7289; recall NEGATIVE 0.7824 → 0.4878, POSITIVE 0.7148 → 0.9456.

## 7. Verification

- Not run yet (`scripts/run.py verify --run <name>`).

## 8. Hardware and memory

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU; peak reserved 1.37 GiB (budget 4.0 GiB, within: True); torch allocator statistics, excluding the CUDA context.

## 9. Files

- `config.yaml` (resolved config), `manifest.json` (identity, status history, artifacts), `summary.json` (this report as data)
- `training/` (training.json, training.log, validation.json), `checkpoints/step<N>/`, `evaluation/step<N>/`, `verification/step<N>/` (report.md, evidence/)
