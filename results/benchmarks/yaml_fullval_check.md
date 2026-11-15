# yaml_fullval_check: GRPO + LoRA vs baseline

Qwen/Qwen3-0.6B, LoRA r=8 α=16 on last2-attention, lr 0.0001, 1 steps, B=2 (balanced), G=6, seed 42. Full validation (n=1066), greedy, max_new_tokens = 4. One training seed: CIs cover evaluation sampling only.

| metric | base | trained |
|---|---|---|
| n | 1066 | 1066 |
| accuracy | 0.7486 | 0.7523 |
| macro_f1 | 0.7493 | 0.7530 |
| recall_NEGATIVE | 0.7824 | 0.7936 |
| recall_POSITIVE | 0.7148 | 0.7111 |
| predicted_positive_rate | 0.4662 | 0.4587 |
| invalid_rate | 0.0028 | 0.0028 |
| auc | 0.8277 | 0.8275 |

Accuracy trained − base: +0.0038, paired bootstrap 95% CI [-0.0028, +0.0103].
AUC trained − base: -0.0003, class-stratified paired bootstrap 95% CI [-0.0018, +0.0014].

## Paired changes vs base

```
{"NEGATIVE": {"base_wrong_now_right": 6, "base_right_now_wrong": 0}, "POSITIVE": {"base_wrong_now_right": 2, "base_right_now_wrong": 4}, "flips": {"POSITIVE->NEGATIVE": 10, "NEGATIVE->POSITIVE": 2}, "invalid_to_valid": 0, "valid_to_invalid": 0}
```

## Trained confusion

| predicted \ gold | NEGATIVE | POSITIVE |
|---|---|---|
| NEGATIVE | 423 | 151 |
| POSITIVE | 110 | 379 |
| None | 0 | 3 |

## Trained vs saved runs

| reference | Δ accuracy | 95% CI | Δ AUC | 95% CI |
|---|---|---|---|---|
| base | +0.0038 | [-0.0028, +0.0103] | -0.0003 | [-0.0018, +0.0014] |
| last2 B=1 | — | — | -0.0038 | [-0.0059, -0.0019] |
| last2 B=2 balanced | — | — | -0.0092 | [-0.0118, -0.0066] |
| last2 B=2 unconstrained | — | — | -0.0054 | [-0.0076, -0.0033] |

Peak reserved 1.99 GiB of 4.0 GiB budget. Frozen weights changed: 0. Raw predictions: `results/yaml_fullval_check/validation.json`; training log: `results/yaml_fullval_check/training.json`.
