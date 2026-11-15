# default_rerun: GRPO + LoRA vs baseline

Qwen/Qwen3-0.6B, LoRA r=8 α=16 on last2-attention, lr 0.0001, 30 steps, B=2 (balanced), G=6, seed 42. Full validation (n=1066), greedy, max_new_tokens = 4. One training seed: CIs cover evaluation sampling only.

| metric | base | trained |
|---|---|---|
| n | 1066 | 1066 |
| accuracy | 0.7486 | 0.7261 |
| macro_f1 | 0.7493 | 0.7222 |
| recall_NEGATIVE | 0.7824 | 0.5966 |
| recall_POSITIVE | 0.7148 | 0.8555 |
| predicted_positive_rate | 0.4662 | 0.6295 |
| invalid_rate | 0.0028 | 0.0019 |
| auc | 0.8277 | 0.8366 |

Accuracy trained − base: -0.0225, paired bootstrap 95% CI [-0.0469, +0.0019].
AUC trained − base: +0.0089, class-stratified paired bootstrap 95% CI [+0.0063, +0.0115].

## Paired changes vs base

```
{"NEGATIVE": {"base_wrong_now_right": 0, "base_right_now_wrong": 99}, "POSITIVE": {"base_wrong_now_right": 75, "base_right_now_wrong": 0}, "flips": {"NEGATIVE->POSITIVE": 173, "None->POSITIVE": 1}, "invalid_to_valid": 1, "valid_to_invalid": 0}
```

## Trained confusion

| predicted \ gold | NEGATIVE | POSITIVE |
|---|---|---|
| NEGATIVE | 318 | 75 |
| POSITIVE | 215 | 456 |
| None | 0 | 2 |

## Trained vs saved runs

| reference | Δ accuracy | 95% CI | Δ AUC | 95% CI |
|---|---|---|---|---|
| base | -0.0225 | [-0.0469, +0.0019] | +0.0089 | [+0.0063, +0.0115] |
| last2 B=1 | — | — | +0.0053 | [+0.0031, +0.0076] |
| last2 B=2 balanced | — | — | +0.0000 | [+0.0000, +0.0000] |
| last2 B=2 unconstrained | — | — | +0.0038 | [+0.0018, +0.0058] |

Peak reserved 2.44 GiB of 4.0 GiB budget. Frozen weights changed: 0. Raw predictions: `results/default_rerun/validation.json`; training log: `results/default_rerun/training.json`.
