# Architecture

One GRPO + LoRA update, tensor by tensor. Shapes are read from the code, not from a template.

## Modules

```
qwen_grpo/grpo.py        pure tensor math, CPU-testable: advantages, response mask, token log-probs, loss
qwen_grpo/policy.py      model calls: sampling rollouts, teacher-forced log-probs
qwen_grpo/sentiment.py   task: splits, prompt, strict parser, reward
qwen_grpo/model.py       Qwen3-0.6B id, EOS ids, bf16 base + fp32 LoRA wrapping
qwen_grpo/training.py    prompt batches, reward grouping, one optimizer update (grpo_step)
qwen_grpo/evaluation.py  greedy predictions, metrics, label scores, AUC, paired bootstraps
research/                experiment entry points (python -m research.<name>); they write results/ and checkpoints/
```

Imports point downward only: `grpo` ← `policy` ← `training`, `evaluation`; `sentiment` and `model` hold no GRPO logic.

## Symbols

```
B  prompts per optimizer step         (1 or 2 in the saved runs)
G  completions per prompt             (6)
N  = B * G rollout rows; row b*G + g answers prompt b
P  prompt length after left padding to the longest prompt in the batch
R  response length = longest generated response in the batch (<= max_new_tokens = 8)
V  vocabulary size (151 936)
```

## One step

| # | Where | Operation | Output |
|---|---|---|---|
| 1 | `sentiment.build_prompts` | chat template, non-thinking (`<think>\n\n</think>` appended) | B strings |
| 2 | `policy.generate_rollouts` | `generate`, temperature 1, top-p 1, top-k 0, `num_return_sequences=G`, no grad | `input_ids` [N, P+R] |
| 3 | `grpo.response_mask` | tokens up to and including the first EOS (Qwen's pad id is an EOS id, so `ids != pad` is wrong) | `completion_mask` [N, R] bool |
| 4 | `training.decode`, `training.group_rewards` | strict parse; reward 1 if exactly the gold label | rewards [B, G] |
| 5 | `grpo.group_advantages` | `(r - mean_g) / (std_g + 1e-4)`, population std (`unbiased=False`) | [B, G] → flattened [N] |
| 6 | `policy.compute_logprobs`, no grad | teacher-forced forward; `position_ids = cumsum(mask) - 1` as `generate` uses | `old_logp` [N, R] |
| 7 | `policy.compute_logprobs`, with grad | same forward, autograd on | `logp` [N, R] |
| 8 | `grpo.grpo_loss` | clipped surrogate, masked | scalar |
| 9 | `training.grpo_step` | `backward`, finiteness checks, `AdamW.step` on LoRA params only | logs |

Steps 6–7 inside `compute_logprobs`: logits [N, P+R, V] in bf16. `response_logprobs` keeps the R columns that predict response tokens (`logits[:, P-1:-1]`, since column t-1 scores token t), casts that [N, R, V] slice to fp32, and applies `log_softmax` + `gather`.

If every group has constant rewards, step 5 gives all-zero advantages and `grpo_step` returns `None` without an update. With B = 2, a constant group contributes zero to the loss but still counts in the 1/N.

## Objective

```
rho_it = exp(logp_it - old_logp_it)
L = -(1/N) sum_i (1/|o_i|) sum_{t in o_i} min(rho_it A_i, clip(rho_it, 1-0.2, 1+0.2) A_i)
```

There is no KL term (beta = 0). Each sequence's tokens are averaged first, then the sequence means are averaged; it is not a token-sum over total tokens. Each rollout batch gets exactly one update and `old_logp` comes from the same weights. So rho = 1 at the only point where it is evaluated, the clip never binds, and the gradient is the REINFORCE form

```
grad L = -(1/N) sum_i (A_i / |o_i|) sum_t grad log pi(o_it | q, o_<t)
```

`grpo_step` asserts `max |logp - old_logp| <= 1e-6` over masked tokens before the update.

## Gradients and precision

- Base weights: bf16, `requires_grad=False`. Their digests are checked to be unchanged before and after training.
- LoRA: fp32 A [r, in] and B [out, r], r = 8, alpha = 16, no dropout, on `layers.27.o_proj` (`last-o-proj`) or q/v/o of layers 26–27 (`last2-attention`). PEFT runs the adapter path in fp32 and adds it to the bf16 base output, cast back to bf16.
- Backward runs from the loss through `lm_head`, the final norm and every layer above an adapter. Gradients pass through frozen bf16 weights to reach the layer-26 adapters, but those weights get no `.grad`. Below the lowest adapter no input requires grad, so autograd records nothing there.
- At init B = 0, so dL/dA = (alpha/r) B^T (dL/dh) x^T = 0 exactly; the first step moves B only (`tests/test_multistep.py`).
- Neither the [N, P+R, V] bf16 logits nor the fp32 [N, R, V] slice is chunked. Their sizes are 2·N·(P+R)·V and 4·N·R·V bytes, e.g. 12·8·151 936·4 B ≈ 58 MB for the fp32 slice at N = 12, R = 8. That is arithmetic, not a profiled peak.

## Evaluation

`evaluation.predict`: greedy decoding, `max_new_tokens = 16`, chunks of 32. bf16 outputs depend on batch shape, so chunking is fixed. `label_scores` computes s(x) = log P(POSITIVE<|im_end|>|x) − log P(NEGATIVE<|im_end|>|x) with teacher forcing, for ROC AUC. Model comparisons are paired by example id (bootstrap CIs for accuracy and AUC).
