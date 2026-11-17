# Architecture

The system end to end, then one GRPO + LoRA update tensor by tensor. Details per component, with measurements and
design decisions, are in `docs/l2l/NN_*.md` (index: `docs/development_history.md`).

## Data flow of one training step

```
data/sentiment.json --balanced_subset--> PromptStream (disk, sampler state)           streaming/dataset.py
   B prompts --build_prompts (non-thinking template)--> token ids                      sentiment.py
   backend.generate_rollouts: B x G completions (T = 1)                                execution/generator.py | dflash.py
   group_rewards: exact-label reward [B, G] -> group advantages                        sentiment.py, grpo.py
   RolloutStore.write (record: ids, masks, rewards, advantages, old log-probs)          streaming/rollouts.py
   prepare: old log-probs (no grad)  ->  grpo_update: clipped loss, backward, AdamW     training.py
   checkpoint (adapter + optimizer + RNG + sampler), published crash-consistently      checkpointing.py, streaming/resume.py
```

The GRPO mathematics, shown in the sections below, lives in `grpo.py` and `training.py` and is identical for every
backend. A backend (`execution/backend.py`) supplies only three things:
- `generate_rollouts`;
- `logprobs`, the differentiable log-prob function;
- `greedy`, the evaluation generator.

## Execution backends

**`resident`:**
- the whole bf16 model on the GPU;
- HF `generate` for rollouts, `policy.compute_logprobs` for log-probs.

**`l2l_sync` / `l2l_async`:**
- **Placement.** `WeightStore` (`execution/weight_store.py`) adopts every frozen decoder weight into one pinned CPU slab
  per layer (`cudaHostRegister`); the Parameters are re-pointed and their values are unchanged. The embedding (tied LM
  head), final norm, LoRA tensors, AdamW state and the KV cache stay on the GPU.
- **Forward.** `execution/qwen3_forward.py` owns the layer loop. A layer executor copies layer *i* into one of the
  `weight_buffers` GPU buffers, runs it, and releases the buffer. Under `l2l_async`, `execution/prefetch.py` copies the
  next `prefetch_depth` layers on a separate CUDA stream, and buffer ownership is tracked with events.
- **Backward.** `execution/backward.py` keeps only the layer-boundary hidden states. In backward it streams the layers
  again in reverse, from the last layer down to the first adapted one, and recomputes each layer under autograd.
- **Output head.** `execution/output_head.py` evaluates the full LM head on the response columns only (R + 1). The
  chunked-vocabulary head exists but is refused for training: H03 is not supported.
- **Generation.** `execution/generator.py` does prefill and decode with an explicit KV cache and HF sampling semantics,
  and never calls `model.generate`.

**Scheduling is micro-batch-major.** Each micro-batch of the old-log-prob pass and of the update, each prefill or
decode step, and each backward sweep streams the layers again. The weight traffic per step is roughly
(1 + new tokens + 2 + backward fraction) × model size; preflight reports the estimate. Bitwise equality with the
resident backend is the acceptance criterion (`research/l2l/tolerances.yaml`: 0 for these cases).

**DFlash** (`execution/dflash.py`, `dflash.enabled`):
- **Drafter.** `z-lab/Qwen3-4B-DFlash-b16` is GPU-resident. It drafts a 16-token block from hidden features of five
  target layers.
- **Verification.** One streamed target pass verifies the whole block with the *current LoRA policy*. The target draws
  its own token at every position, and the matching prefix is accepted.
- **Exactness.** Greedy outputs equal plain decoding, and sampling follows the target's law.
- **Batching.** Rows advance by the batch's minimum acceptance.

## Around training

- **Streaming and resume** (`streaming/`):
  - prompts are written once, with a dataset hash and the sampler state saved in every checkpoint;
  - one rollout record per group, holding the policy version;
  - checkpoints are published by temp directory, `COMMIT.json` and rename;
  - `resume` takes the highest valid checkpoint, quarantines torn ones and replays nothing.
- **Merge** (`export/merge.py`): `W = round(fp32(W0) + (alpha/r)·B·A)`, one tensor at a time on the CPU, into a sharded
  HF checkpoint. The merged model is evaluated through L2L.
- **Verification:**
  - resident runs use `verification.py`: base, adapter, merged, LR=0 control, paired bootstrap CIs, policy change,
    reward integrity;
  - L2L runs use `scripts/final_eval.py`: the same four models on the test split, through L2L.
- **Resources.** `execution/profiler.py` samples nvidia-smi, `cudaMemGetInfo`, the allocator and RSS, with a guard
  that stops the run. `memory.guard` turns it on for training. `budget/` estimates memory and PCIe time from config and
  safetensors headers.
- **Diagnostics** (`diagnostics/`): preflight, invariants, and the silent-failure registry. Soup is called only through
  `integrations/soup.py`.

Imports point downward: `grpo` ← `policy` ← `training` ← `trainer`, and `execution/*` never imports the trainer.
Research lab modules (`*_lab.py`, `lab.py`) and `diagnostics/` / `integrations/` are not imported by the training path.

# One GRPO + LoRA update

Shapes are read from the code, not from a template.

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

`evaluation.predict`: greedy decoding, `max_new_tokens = 4` and chunks of 32 by default (YAML `evaluation` section; defaults `evaluation.MAX_NEW`, `evaluation.BATCH`). bf16 outputs depend on batch shape, so chunking is fixed. `label_scores` computes s(x) = log P(POSITIVE<|im_end|>|x) − log P(NEGATIVE<|im_end|>|x) with teacher forcing, for ROC AUC. Model comparisons are paired by example id (bootstrap CIs for accuracy and AUC).
