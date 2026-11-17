# Run `qwen_resident_short` — completed

id `qwen_resident_short-20261117T070315Z-dceb069b`, created 2026-11-17T07:03:15+00:00, config sha256 `dceb069be6183565…`, git `5300146b4d98`.

## 1. Configuration

- Model Qwen/Qwen3-0.6B (bfloat16 base, fp32 LoRA), non-thinking.
- LoRA r=32, α=64, targets `last2-attention`.
- GRPO: B=5 (shuffled sampler), G=6, lr 0.0005, max 10 steps / 1000 attempts, clip ε 0.2, 1 update(s) per rollout batch, micro-batch whole batch, temperature 1, rollout max_new_tokens 4, seed 42.
- Data: 512 training prompts, validation subset 512 (subset seed 42). Full settings: `config.yaml`.

## 2. Training progress

- 10/10 optimizer steps from 10 rollout attempts (0 zero-signal batches skipped); training time 27 s. Source: `runs/qwen_resident_short/training/training.json`.
- Checkpoints: step 0, step 5 (resumable), step 10 (resumable); latest `checkpoints/step10/`.

## 3. Reward and gradient diagnostics

- Frozen base tensors changed: 0; adapter equal to its initialization: False.
- Rising rollout reward is measured on the policy's own samples of training prompts: it is not held-out evidence.
- Plots: [reward.png](plots/reward.png), [validation_accuracy.png](plots/validation_accuracy.png), [mixed_reward_fraction.png](plots/mixed_reward_fraction.png), [gradient_norm.png](plots/gradient_norm.png), [loss.png](plots/loss.png), [prediction_bias.png](plots/prediction_bias.png)

## 4. Base vs trained (full validation, at the end of training)

- Not available: evaluation.full_validation is false or training did not finish. See `validate` results (section 5).

## 5. Held-out evaluations (`validate` / `test`)

- None run yet (`scripts/run.py validate --run <name>`).

## 6. POSITIVE prediction rate and classification drift

- Validation subset (n=512): predicted-POSITIVE rate step 0 0.469, step 5 0.623, step 10 0.359; accuracy step 0 0.736, step 5 0.725, step 10 0.738. Step 0 is the base model (zero adapter).

## 7. Verification

- Not run yet (`scripts/run.py verify --run <name>`).

## 8. Hardware and memory

- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU; peak reserved 2.15 GiB (budget 4.0 GiB, within: True); torch allocator statistics, excluding the CUDA context.

## 9. Files

- `config.yaml` (resolved config), `manifest.json` (identity, status history, artifacts), `summary.json` (this report as data)
- `training/` (training.json, training.log, validation.json), `checkpoints/step<N>/`, `evaluation/step<N>/`, `verification/step<N>/` (report.md, evidence/)
