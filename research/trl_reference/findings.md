# TRL GRPOTrainer as an independent reference for the manual GRPO + LoRA run

Date: 2026-11-15. Comparison target: the manual **expanded-LoRA (q/v/o, layers 26–27), balanced B=2, 30-step** run (`results/grpo_lora_last2_b2_balanced_*`). All numbers below are measured unless marked as a hypothesis. There is one training seed per implementation.

## A. Implementation

**Packages.** `trl==1.18.0` was installed into the existing `.venv`. A dry run showed that it added only this one package. torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, accelerate 1.15.0 and datasets 5.0.1 are unchanged. `requirements.txt` does **not** list trl yet, and I did not edit it.

**Files** (nothing in the manual implementation or previous results was modified):

| File | Role |
|---|---|
| `research/trl_reference/config.json` | the single source of the run configuration; all `GRPOConfig` arguments |
| `research/trl_reference/train.py` | TRL training run, instrumentation, adapter save |
| `research/trl_reference/compare_gradients.py` | fixed-rollout gradient comparison on the real model (bf16 and fp32) |
| `research/trl_reference/evaluate.py` | full validation, scores, flips, bootstrap (separate process, adapter loaded from disk) |
| `tests/test_trl_reference.py` | 9 tests; a tiny fp32 Qwen3 with the real tokenizer, no 0.6B weights |
| `research/trl_reference/results/{training,gradient_comparison,evaluation}.json` | outputs |
| `research/trl_reference/checkpoints/trl_grpo_last2_b2_balanced_step30/` | adapter |

Run everything from the repo root with `.venv/bin/python -m research.trl_reference.<train|compare_gradients|evaluate>`. Running as a module matters: `evaluate.py` would otherwise shadow the HF `evaluate` package.

**What is reused from the manual implementation.** The 32-example subset (`balanced_subset`), the chat-templated prompt (`baseline.build_prompts`), the reward (`sentiment.reward` on the manually decoded completion ids), the LoRA model (`train_lora.lora_model` with the same seed, so the initial adapter is identical), the balanced prompt stream (`train_multistep.batches`), and all evaluation code (`baseline.predict`, `label_scores`, `metrics`, the bootstraps).

**TRL owns** generation, reward bookkeeping, advantages, masks, log-probs (its fused LM head), the loss, backward, and the optimizer and Trainer loop.

**The one TRL override** is `_get_train_sampler`. It returns the manual balanced stream with each prompt repeated G=6 times in a row, instead of `RepeatSampler`'s random order.

**Model configuration.**
- Qwen3-0.6B, bf16 frozen base.
- LoRA r=8, α=16, dropout 0 on exactly `model.layers.{26,27}.self_attn.{q,v,o}_proj`, asserted.
- 131,072 trainable fp32 parameters, asserted before and after `GRPOTrainer.__init__`.
- The trainer left the adapters bitwise equal to their initial values and did not change their dtypes.
- `bf16=False`, so there is no autocast and the adapter matmuls run in fp32 exactly as in the manual code. TRL's default `bf16=True` would have run them in bf16.

**TRL defaults that had to be overridden.** The full list is in `config.json`:

| Setting | TRL default | Used |
|---|---|---|
| `loss_type` | `"dapo"` | `"grpo"` |
| `max_grad_norm` | 1.0 | 0 (no clipping) |
| `lr_scheduler_type` | linear | constant |
| `optim` | fused AdamW | `adamw_torch` |
| `gradient_checkpointing` | True | False |
| `bf16` | True | False |
| `num_generations` | 8 | 6 |
| `max_completion_length` | 512 | 8 |

The clipping override matters: the manual maximum grad norm was 0.997, so a cap of 1.0 is not inert.

**Batch semantics** (verified in the trainer, not assumed): `per_device_train_batch_size=12`, `steps_per_generation=1`, `gradient_accumulation_steps=1`. That gives `generation_batch_size = 12 = 2 prompts × 6`, one generation and one optimizer step. Checks:
- Every reward call asserted 12 rows in two 6-blocks, POSITIVE then NEGATIVE, equal to manual batch *k*.
- Each row's prompt and ground truth belong to its example id.
- After training, generations = optimizer steps = 30, checked at every step.

**Tests.** `pytest`: **96 passed**, which is the 87 existing tests plus 9 new ones:
1. reward equivalence;
2. dataset equivalence and no held-out overlap;
3. TRL prompt tokenization = manual tokenization for all 32 prompts;
4. LoRA targets and 131,072 on the real architecture (meta device);
5. TRL dataloader batches = the manual stream, all 30 of them;
6. the reward rejects misaligned batches;
7. fp32 gradient equivalence;
8. score-shift statistics by hand;
9. label tokenization and the saved manual numbers.

`tests/test_grpo.py` sets a float64 default dtype at import. The new module pins float32 for its own tests and restores the previous default afterwards.

## B. Mathematical equivalence

Read in the installed TRL source (`trl/trainer/grpo_trainer.py`, `trl/trainer/utils.py`) and then measured.

| # | Item | Manual | TRL 1.18 (as configured) | Status |
|---|---|---|---|---|
| 1 | Advantage | (R−mean)/(σ+1e-4) | same form | equal form |
| 2 | Std convention | population (÷G) | **Bessel (÷(G−1))** (`nanstd`) | **differs**: A_TRL ≈ √(5/6)·A_manual = 0.9129·A_manual |
| 3 | Epsilon | 1e-4 | 1e-4 | equal |
| 4 | Loss reduction | Σ_t over mask / \|o\|, mean over the 12 sequences | identical for `loss_type="grpo"`, GA=1 | equal |
| 5 | EOS handling | mask through the first EOS (151645 or 151643) | same, with ids `[151645, 151643]` | equal |
| 6 | Truncation | kept whole | kept whole (`mask_truncated_completions=False`) | equal |
| 7 | Importance ratio | old = second identical no-grad pass, so ρ ≡ 1 | with `num_iterations=1` and GA aligned, old = `logp.detach()`, so ρ ≡ 1; logged clip ratios were 0 at every step | equal: clipping is inert in both |
| 8 | Gradient clipping | none | none (`max_grad_norm=0`) | equal |
| 9 | Optimizer | torch AdamW(1e-4, wd 0), constant LR | torch AdamW (HF Trainer), same hyperparameters, constant LR (logged 1e-4 at every step) | equal, **except zero-signal batches (below)** |
| 10 | Adapter precision | fp32 adapters, bf16 base, no autocast | same (PEFT upcast; `bf16=False`) | equal |
| — | Log-prob numerics | full bf16 logits, then fp32 log_softmax; `position_ids` from the mask | fused chunked LM head, no full logits; no `position_ids` (RoPE is relative, so mathematically equal) | numerical only |
| — | Reward text | strip the terminating EOS, then decode | TRL's `completions` decode with `skip_special_tokens=True`. **I used `completion_ids`** with the manual decode. A test shows that `POS<\|im_start\|>ITIVE` would otherwise score 1 instead of 0 | equal by construction |

**Fixed-rollout gradient comparison.** Setup: the first balanced batch (ids 630, 256), the same model and adapter init, and the same 12 completion token ids, rewards and masks. The rollouts reproduce manual attempt 1 exactly. TRL received the fixed completions through its `rollout_func` hook, so its own reward → advantage → mask → log-prob → loss → backward code ran. TRL's internal advantages equal the restated Bessel formula to 2.4e-7.

δ_g = ‖g_manual − g_TRL‖₂ / ‖g_manual‖₂:

| Setting | δ_g | cos | ‖g_TRL‖/‖g_manual‖ |
|---|---:|---:|---:|
| **fp32**, real model, manual with TRL's (Bessel) advantages | **2.0e-5** | 1.000000000 | 0.999998 |
| fp32, real model, raw manual (population std) | 8.71e-2 | 1.000000000 | 0.912881 (= 1 − δ, the Bessel factor) |
| fp32 noise floor: manual vs manual, chunks of 2 vs 3 rows | 1.2e-5 | 1.000000000 | 0.999999 |
| fp32, tiny Qwen3 unit test, Bessel-aligned | 7.2e-7 | 1 − 2.5e-13 | 1.0000001 |
| **bf16** (the training precision), Bessel-aligned | 0.103 | 0.9951 | 1.021 |
| bf16, raw manual | 0.112 | 0.9954 | 0.937 |
| bf16 noise floor: manual vs manual over 5 row-chunkings (1, 2, 3, 4, 6) | 0.050 – 0.125 | 0.9934 – 0.9989 | — |
| bf16, TRL vs each of those manual chunkings | 0.088 – 0.120 | 0.9939 – 0.9965 | — |

Per-token log-probs, TRL vs manual, bf16: max \|Δ\| = 4.8e-4.

**Conclusion for B.**
- With the advantages aligned, the two objectives are **exactly equivalent**: the fp32 discrepancy (2e-5) is at the fp32 noise floor (1e-5) on the real architecture.
- The only mathematical difference is the Bessel std. It is a uniform rescaling of each group's advantages by √(5/6), up to the 1e-4 epsilon, and AdamW is nearly invariant to that.
- In bf16, TRL's gradient is no further from the manual gradient than the manual gradient is from itself under a different batch shape.

A side finding: bf16 per-step gradients carry about 5–12% relative noise from batch shape alone.

## C. Training behavior

| | Manual balanced B=2 | TRL |
|---|---:|---:|
| Rollout batches generated | 31 | 30 |
| Optimizer steps | 30 (all with signal) | 30 (29 with signal + 1 zero-gradient) |
| Zero-signal batches (both groups constant) | 1 (attempt 14, skipped) | 1 (batch 14, **stepped**) |
| Batches by number of mixed groups {0, 1, 2} | {1, 10, 20} | {1, 7, 22} |
| Constant (zero-advantage) groups | 12 / 62 | 9 / 60 |
| Mean rollout reward, first 10 / last 10 steps | 0.592 / 0.608 | 0.567 / 0.567 |
| Grad-norm range | 0.20 – 1.00 | 0.21 – 1.16 (0 at step 14) |
| ‖θ − θ₀‖ at the end | 0.312 | 0.333 |
| Frozen tensors changed | 0 / 310 | 0 / 310 |
| Peak allocated / reserved (torch) | 1.64 / 2.44 GiB | 1.29 / 1.34 GiB |
| `nvidia-smi` peak during training (includes the CUDA context) | — | 1,494 MiB |

**Zero-signal handling.** This is the one optimizer difference that couldn't be configured away, and it was left as TRL implements it. The manual loop skips a batch with no mixed group and draws the next one. TRL still calls `optimizer.step()` with an exactly zero gradient, so AdamW applies a momentum-only update (‖Δθ‖ = 9.9e-3 at step 14) and advances its step counter. As a result, TRL's 30 steps covered batches 1–30, and the manual run's 30 steps covered attempts 1–13 and 15–31.

**Trajectories.**
- The same seed and the same prompt stream made the sampled data nearly identical: all 30 prompt pairs match, and the sampled responses are **identical in 25 of 30 batches**. They differ in batches 3, 7, 9, 25 and 28.
- On identical batches, the gradient-norm ratio TRL/manual starts at 0.91–0.94, close to the Bessel factor. It drifts to about 1.0–1.05 as the parameters diverge.
- The final adapters are close: ΔW = (α/r)BA has cosine **0.992** over all six modules (0.989–0.996 per module), and TRL's ‖ΔW‖ is 5% larger.
- Evaluation memory peaked at 2.75 GiB reserved, under the 4 GiB budget.

Notes:
- `GRPOTrainer.__init__` set `bos_token_id=None` on the model config to match the tokenizer. That has no effect, because the prompts never use a BOS token.
- The training reward is flat in both runs, so there is no evidence of learning from the reward curve.

## D. Evaluation

The same procedure as the manual runs: full validation n=1,066 (533/533), identical prompts, non-thinking mode, greedy decoding, 16 new tokens, batch 32, and the same strict parser through `baseline.predict`.

Consistency, checked in-process:
- the base predictions reproduce `baseline_validation.json` with **0 mismatches**;
- the in-process complete-label scores reproduce the saved base and manual scores **exactly** (max \|Δ\| = 0).

| Metric | Base | Manual GRPO (balanced B=2) | TRL GRPO |
|---|---:|---:|---:|
| Accuracy | 0.7486 | 0.7261 | **0.7139** |
| Macro F1 | 0.7493 | 0.7222 | **0.7069** |
| POSITIVE recall | 0.7148 | 0.8555 | **0.8762** |
| NEGATIVE recall | 0.7824 | 0.5966 | **0.5516** |
| Predicted POSITIVE | 0.466 | 0.630 | **0.662** |
| ROC AUC (complete-label s) | 0.8277 | 0.8366 | **0.8386** |
| Invalid rate | 0.0028 | 0.0019 | 0.0019 |

**Decision-score shift.** Δs = s_trained − s_base, with s = log P(POSITIVE<\|im_end\|>) − log P(NEGATIVE<\|im_end\|>).

| | TRL − base | Manual − base |
|---|---:|---:|
| Mean Δs | **+0.781** | +0.651 |
| SD Δs | 0.213 | 0.188 |
| Median Δs | +0.750 | +0.625 |
| Fraction Δs > 0 | **100%** (min +0.25) | 100% (min +0.125) |
| Pearson(s_base, s_trained) | 0.9968 | 0.9973 |
| Mean Δs, POSITIVE-gold | +0.904 | +0.755 |
| Mean Δs, NEGATIVE-gold | +0.659 | +0.546 |

The per-example correlation between the TRL and manual shifts is **0.926**. The medians and minima fall on multiples of 0.125, which reflects bf16 logit quantization.

**Prediction flips.**

| | base → TRL | base → manual | manual → TRL |
|---|---:|---:|---:|
| NEGATIVE → POSITIVE | **208** | 173 | 35 |
| POSITIVE → NEGATIVE | **0** | 0 | 0 |
| Incorrect → correct | 86 | 75 | 11 |
| Correct → incorrect | 123 | 99 | 24 |
| Invalid → valid | 1 (→ POSITIVE) | 1 | 0 |
| Valid → invalid | 0 | 0 | 0 |

**Paired bootstrap.** 10,000 resamples, seed 42, matched by id; the AUC resamples within each class.

| Comparison | Accuracy diff [95% CI] | AUC diff [95% CI] |
|---|---|---|
| TRL − base | −0.0347 [−0.0610, −0.0084] | +0.0108 [+0.0082, +0.0136] |
| TRL − manual | −0.0122 [−0.0235, −0.0009] | +0.0019 [+0.0004, +0.0036] |

These intervals cover evaluation-sample uncertainty only. Each implementation ran one training seed, so the TRL − manual differences do not establish that the implementations differ in expectation. Section B already shows that bf16 noise alone perturbs per-step gradients by about 10%.

## E. Research conclusion

1. **Does TRL show the same POSITIVE shift? Yes (Outcome A), and slightly larger.**
   - Δs > 0 on 100% of validation examples (mean +0.78 vs +0.65).
   - 208 NEGATIVE→POSITIVE flips and 0 POSITIVE→NEGATIVE flips.
   - Predicted-POSITIVE rate 0.662.
   - Base/trained score correlation 0.997.

   The shift is again a near-uniform offset with a small ranking gain: AUC +0.011, larger for POSITIVE-gold examples.

2. **Are the GRPO gradients equivalent under matched conditions? Yes, verified.**
   - fp32, real model: δ_g = 2.0e-5, the noise floor of 1.2e-5, once TRL's Bessel std is used.
   - Raw: δ_g = 0.0871 with cosine 1.000000000, which is exactly the √(5/6) rescaling.
   - bf16: TRL lies inside the manual implementation's own batch-shape noise.

3. **Remaining differences:**
   - (a) Bessel vs population std. A per-group scalar of about 0.913, largely absorbed by Adam.
   - (b) Zero-signal batches. TRL takes one momentum-only AdamW step, the manual run skips the batch. TRL therefore saw batches 1–30 and the manual run saw 1–13 and 15–31.
   - (c) Log-prob numerics: the fused fp32-accumulated head vs bf16 logits, and no `position_ids`. Max per-token \|Δ\| = 4.8e-4.
   - (d) Train vs eval module mode. Dropout is zero/disabled, so this has no effect.
   - (e) Sampled responses differ in 5 of 30 batches, as a consequence of (a)–(c).

   Which of these causes the larger TRL shift (+0.78 vs +0.65) is **not determined**. This is a hypothesis, not tested: the extra momentum step plus the trajectory divergence. The final adapters still agree to cosine 0.992.

4. **Is there evidence that the manual implementation is incorrect? No.**
   - An independent implementation, run under the same configuration, computes the same gradient: exactly in fp32, within noise in bf16.
   - Its training reproduces the drift in direction, uniformity and size.
   - Nothing here points to a bug specific to the manual code.

5. **Does this strengthen the "GRPO optimization dynamics" explanation? Yes, for "the objective under this data and schedule", as opposed to "the manual code".** It does not establish *which* property of the objective is responsible. Two things it does not show:
   - It does not show that GRPO is inherently POSITIVE-biased: the two runs share the data, prompt, base model, seed and nearly identical rollouts, so they are not independent samples of the training process.
   - It does not rule out an evaluation-independent artefact shared by both implementations, such as the bf16 base.

6. **The single most informative next experiment** tests the std-normalization mechanism (audit H7) by changing one variable.
   - Rerun this exact configuration with `scale_rewards="none"` (centered advantages without division by σ), keeping everything else fixed. This works in either implementation.
   - If the uniform POSITIVE shift shrinks markedly, the group-std weighting of the weaker class's prompts is implicated. If it persists, H7 is contradicted.
   - Two or three training seeds per arm would be needed before treating a difference as real.
   - This is a proposal only. It was not run, because the brief excluded changing the objective here.

**Not verified / limitations.**
- One training seed per implementation.
- The fixed-rollout comparison used a single batch.
- The bf16 noise floor comes from five batch shapes of that batch.
- TRL's `rollout_func` is marked experimental.
- `requirements.txt` does not yet pin `trl==1.18.0`.
