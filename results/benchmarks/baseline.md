# Baseline: Qwen/Qwen3-0.6B (untuned, non-thinking)

Greedy decoding (settings per split below). AUC: complete-label scores s(x) = log P(POSITIVE<|im_end|>) − log P(NEGATIVE<|im_end|>).

| metric | validation | test |
|---|---|---|
| n | 1066 | 1066 |
| accuracy | 0.7486 | 0.7326 |
| macro_f1 | 0.7493 | 0.7337 |
| recall_NEGATIVE | 0.7824 | 0.7674 |
| recall_POSITIVE | 0.7148 | 0.6979 |
| predicted_positive_rate | 0.4662 | 0.4653 |
| invalid_rate | 0.0028 | 0.0038 |
| auc | 0.8277 | 0.8166 |

## validation

Raw predictions: `results/baseline/validation.json`. Decoding {'do_sample': False, 'max_new_tokens': 4}. Truncated 0.000, response tokens min/median/max 3 / 3.0 / 4.

| predicted \ gold | NEGATIVE | POSITIVE |
|---|---|---|
| NEGATIVE | 417 | 149 |
| POSITIVE | 116 | 381 |
| None | 0 | 3 |

## test

Raw predictions: `results/baseline/test.json`. Decoding {'do_sample': False, 'max_new_tokens': 4}. Truncated 0.001, response tokens min/median/max 2 / 3.0 / 4.

| predicted \ gold | NEGATIVE | POSITIVE |
|---|---|---|
| NEGATIVE | 409 | 157 |
| POSITIVE | 124 | 372 |
| None | 0 | 4 |
