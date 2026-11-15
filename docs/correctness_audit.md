# Correctness audit: manual GRPO + LoRA on Qwen3-0.6B

Date: 2026-11-15. Branch `audit/grpo-lora-correctness`, based on `main` = `origin/main` = `06489d3` ("better structure"). The working tree was clean at the start.

**Environment.**
- Python 3.12.3; torch 2.14.1+cu126 (CUDA 12.6); transformers 4.57.6; peft 0.21.3; trl 1.18.0.
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB.

Everything below is measured unless it is labelled as reasoning. Commands and outputs are in §11.

## 1. Executive summary

**Verdict: CORRECT WITH LIMITATIONS.**

No critical or high-severity bug was found. Each of the following was checked independently of the implementation's own indexing:
- prompt / response / reward alignment;
- group advantages;
- causal token alignment;
- EOS and padding masks;
- the loss and its gradient;
- LoRA wiring;
- frozen weights;
- LR=0 behaviour;
- checkpoint round-trips;
- every baseline metric.

The audit fixed two defects, one MEDIUM and one LOW:
- **MEDIUM (test hygiene):** test results depended on the order in which test files ran.
- **LOW:** `CLIP_EPS` was logged but never passed to `grpo_step`.

It also records two artifact and comparability issues (§9) that are the owner's decision.

The limitations are properties of the chosen design, not bugs:
1. **Clipping is inert by construction.** The "old" log-probs come from a second, identical pass immediately before the single update. So ρ ≡ 1, and the update is REINFORCE with a group-standardized baseline and 1/|o| weighting. `init_diff` = 0.0 was measured at every step of every run, including this audit's smoke run.
2. **The "old policy" is not the sampler.** It is a bf16 teacher-forced recompute, not the bf16 sampling distribution. These differ by up to about 0.07 nats per token (earlier measurement, Stage 2).
3. **bf16 gradients vary with batch shape.** Relative noise is 5–12% from batch shape alone (`research/trl_reference/findings.md`). fp32 matches TRL to 2e-5.

Answers to the owner's questions:

| Question | Answer |
|---|---|
| Is the GRPO objective implemented correctly? | Yes. The clipped surrogate is mean over sequences of (mean over tokens), with A = (R−μ)/(σ_pop+1e-4). Clipping never acts under the one-update schedule (§3). |
| Are the right token probabilities being optimized? | Yes. Logits at column t−1 score token t; prompt, padding and post-EOS tokens are excluded; EOS is included (§4). |
| Are reward groups and advantages aligned? | Yes, for (B,G) ∈ {(1,4),(2,4),(2,8)} on value-level tests, and on every logged batch (§3.1, §6). |
| Do gradients flow through the intended LoRA parameters? | Yes, only through the 12 `lora_A`/`lora_B` tensors of the 6 asserted modules. 0 of 310 frozen tensors changed (§5). |
| Does training genuinely change the policy? | The parameters and the effective ΔW change, and the greedy validation predictions change. Whether the change is *useful* is a separate question: historical runs show no accuracy gain and a uniform shift towards POSITIVE (see the memory notes and the TRL findings). |
| Are the evaluation metrics trustworthy? | Yes. Every baseline number is reproduced exactly by an independent implementation (§7). |
| Is this a reliable reference for L2L? | Yes, under conditions (§12): fixed rollouts, fp32 comparison, and gradients compared alongside outputs. |

## 2. Architecture findings

Active pipeline (imports traced):

```
research/train_multistep.py  (the only training entry point; argparse CLI)
 ├─ qwen_grpo.sentiment   load_splits, build_prompts, parse_label/reward
 ├─ qwen_grpo.model       lora_model (bf16 base + fp32 PEFT LoRA), TARGETS, EXPECTED_MODULES, digest
 ├─ qwen_grpo.policy      generate_rollouts (no_grad sampling), compute_logprobs (teacher forcing)
 │    └─ qwen_grpo.grpo   response_mask, response_logprobs
 ├─ qwen_grpo.training    balanced_subset, batches, check_grouping, decode, group_rewards, grpo_step
 │    └─ qwen_grpo.grpo   group_advantages, grpo_loss
 ├─ qwen_grpo.evaluation  evaluate/predict (greedy), metrics, label_scores, AUC, bootstraps
 └─ research.baseline     only for the constants RESULTS, BENCHMARKS, predictions_path
research/baseline.py      base evaluation; plain AutoModelForCausalLM, no PEFT
research/trl_reference/   independent TRL reference; imports qwen_grpo, nothing imports it back
research/train_grpo.py, train_lora.py   historical Stage 1/2 scripts; not imported by the active pipeline
```

- **No hidden dependencies.** The core package does not import from `research/`, and the historical scripts are not on the active path.
- **One mild inversion:** `train_multistep.py` imports path constants from `research/baseline.py`. It is harmless and not worth changing during an audit.
- **No YAML workflow exists on `main`.** There is no `configs/`, no `qwen_grpo/config.py` and no `scripts/run.py`, so Phase 7's YAML checks are **not applicable**, and nothing was built for them. The CLI audit is in §6.

## 3. Mathematical findings

### 3.1 Grouping (B, G, N = BG)

- `generate_rollouts` relies on HF `generate(num_return_sequences=G)` expanding prompts with `repeat_interleave`. So row i belongs to prompt ⌊i/G⌋.
- The training loop checks this at every step with `check_grouping`, which compares each row's prompt tokens with prompt ⌊i/G⌋.
- `group_rewards` scores row i against `labels[i // G]` and reshapes the scores to [B, G].
- `group_advantages(...).flatten()` returns row-major [B·G], the same order as the rollout rows.

**New test:** `tests/test_audit.py::test_grouping_rewards_advantages_and_loss_weights`, parametrized over (B,G) ∈ {(1,4),(2,4),(2,8)}.
- **Setup:**
  - prompts have different lengths, so there is real left padding;
  - gold labels alternate between prompts;
  - group b has a different number of correct rows (1 vs 2, and 1 vs 5);
  - wrong rows alternate between the opposite label and a truncated junk response with |o| = 5.
- **Checks:**
  - each row carries its own prompt and its own scripted response;
  - rewards equal a hand table;
  - advantages equal (1−p)/(√(p(1−p))+ε) for correct rows and −p/(√(p(1−p))+ε) for wrong rows, written without `group_advantages`;
  - the gradient of the loss with respect to `logp` equals −A_i/(N·|o_i|) on valid tokens and 0 elsewhere.
- **Mutations caught:** scoring row i against `labels[i % B]` instead of `labels[i // G]` makes 2 audit tests fail.
- **Not caught by the audit tests:** expanding the prompt attention mask with `repeat` instead of `repeat_interleave` (`policy.py:25`). The audit's stand-in models ignore the attention mask. The existing `tests/test_policy.py::test_prompt_grouping` catches it (verified by mutation).

Micro-batching: there is none. One forward and one backward pass cover all N rows, so reordering across micro-batches cannot occur.

### 3.2 Advantages

`A_{b,g} = (R_{b,g} − μ_b) / (σ_b + 1e-4)`, where σ_b is the **population** std (÷G), computed per row of the [B, G] tensor.

- **Hand values:** the existing `test_advantages_hand_computed` test, and the B=2 opposite-label test with 12 hand values.
- **Per group, not joint:** existing tests, plus the new (B,G) test.
- **Constant groups:** σ = 0, so A = 0/1e-4 = **exactly 0**. The result is finite, never NaN.
- **Binary rewards:** with k correct out of G, p = k/G, σ = √(p(1−p)), A₊ = (1−p)/σ and A₋ = −p/σ. Σ_g A = 0 within every group.
- **Convention vs TRL:** TRL uses Bessel's std (÷(G−1)). That is a uniform per-group rescaling by √((G−1)/G), up to ε; for G=6 the factor is 0.913. This is a valid alternative choice, not a bug, and it was left unchanged.
- **Zero-variance groups when B>1:** a constant group contributes A=0 but stays in the mean over N. A B=2 batch with one mixed group therefore carries exactly **half** the gradient of that group alone at B=1. A new test pins this down: `test_partially_constant_batch_scales_signal_by_mixed_fraction`. TRL uses the same convention. AdamW largely normalizes the scale away, but logs should not read |g| as a measure of signal quality across batches with different numbers of mixed groups.

### 3.3 Ratios and the old policy

In `grpo_step`:
1. `old_logp` comes from `compute_logprobs` under `no_grad`, so it is detached.
2. `new_logp` comes from a second, identical pass with the graph kept.
3. `grpo_loss` detaches `old_logp` again.
4. `masked_max(new − old) > 1e-6` raises an error.

Consequences:
- There is **one optimizer update per freshly sampled batch**, and old = current parameters, so ρ = exp(0) = 1 exactly. `init_diff` was 0.0 at all logged steps: 120 historical steps, plus 2 + 2 in this audit's runs.
- At ρ=1, `min(ρA, clip(ρ)A) = A` and both branches have gradient A·∇logπ. **The clip mechanism is never exercised in training.** It is exercised only by the unit tests: the existing `test_clipping_cases` over all six sign/region cases, and the new end-to-end bigram test, which hits the low, in and high regions.
- **No claim that clipping helps can be made from these runs.**
- The effective objective is ∇J = (1/N) Σ_i (A_i/|o_i|) Σ_t ∇log π(o_{i,t}).
- The existing closed-form test `test_b2_opposite_labels_hand_rewards_and_independent_reference_gradient` already verifies this through `grpo_step`. It is a free-logit model with SGD at lr=1, so ΔZ = −∇L, and it is written without `grpo_loss` or `response_logprobs`.
- **The logged `loss` is ≈ 0 by construction.** At ρ=1 the per-sequence term is A_i whatever the length, and Σ_g A = 0 within every group. Logged values are about 1e-8. The loss column carries no information about training progress; the gradient does.
- The "old" policy is the bf16 teacher-forced pass, not the sampler's bf16 distribution. The difference is up to about 0.07 nats per token, and it depends on shape (Stage 2 measurement). With one update per batch this only matters for the importance-sampling interpretation, not for the computed gradient.

### 3.4 Loss

```
L = −(1/N) Σ_i (1/|o_i|) Σ_{t ∈ o_i} min(ρ_{i,t} A_i, clip(ρ_{i,t}, 1−ε, 1+ε) A_i),   ε = 0.2, β_KL = 0
```

- **Normalization:** per-sequence token mean, then mean over sequences. This is the historical convention, and it was left unchanged; it is not global token normalization.
- **Masked positions:** they are zeroed *before* the exponential, so NaN or inf at masked positions cannot leak. The existing test `test_masked_positions_ignored_even_if_nonfinite` covers this.
- **Empty responses:** an empty response raises an error. It cannot happen in practice, because the mask always includes the first token.
- **Verification:**
  - existing tests: hand values, finite differences (`test_loss_gradient_matches_finite_differences`), six clip cases, zero advantage, no gradient through `old_logp`;
  - **new:** `test_end_to_end_loss_by_hand_with_padding_eos_truncation_and_clipping` (§4).

## 4. Token log-probabilities and masks

- `response_logprobs` uses `logits[:, P−1 : −1]` to score `input_ids[:, P:]`. So the logit at column t−1 scores token t.
- Left padding keeps every response starting at column P.
- `position_ids` = cumsum(mask) − 1, the same as `generate()`.
- `log_softmax` runs in fp32: bf16 logits are upcast first.

**New end-to-end test (bigram model).** The logits at column t are W[token_t], so the reference log π(y_t) = log_softmax(W[y_{t−1}])[y_t] is known in closed form and does not depend on padding or position.
- **Inputs** cover:
  - three prompts of different lengths, left-padded;
  - responses with junk after an EOS;
  - a response ending on the pad id, which is also an EOS id, as in Qwen3;
  - a truncated response with no EOS;
  - an EOS-only response;
  - old and new policies that differ, so ratios fall in all three clip regions.
- **Result:** the loss from `compute_logprobs` → `grpo_loss` equals a pure-Python loop using `math.exp`, `min` and `max` to 1e-12.
- **Mutations caught:**
  - an off-by-one causal shift;
  - sample std instead of population std;
  - global token normalization;
  - a mask that drops EOS;
  - removal of the clip.

**Existing coverage, confirmed by reading:**
- left-padded teacher forcing equals `generate(output_logits=True)` and an unpadded single-row pass to 1e-5 (tiny fp32 Qwen3, `tests/test_policy.py`);
- `response_mask` cases: pad after EOS, pad as the first EOS, EOS only, truncated;
- the completion attention mask equals the loss mask.

**Masks:**
- prompt tokens are excluded by slicing;
- tokens after the first EOS are excluded by the position-based mask, not by `!= pad`, because the pad id is an EOS id;
- the denominator |o_i| counts tokens through EOS;
- a truncated response keeps all R tokens.

**Memory.**
- `compute_logprobs` materializes the full [N, T, V] bf16 logits. For N=12, T≈100 and V=151,936 that is about 365 MB (arithmetic, not a measurement).
- It also creates an fp32 [N, R, V] upcast and its `log_softmax`, both kept for backward.
- `grpo_step` runs the forward pass twice: an old pass under `no_grad` and a new pass with the graph. Given ρ ≡ 1, the old pass only serves the `init_diff` safety check.

There are two optimization candidates for Stages 4–5. **Neither is implemented, and neither is a correctness issue:**
1. Compute the LM head only on response columns.
2. Reuse `new_logp.detach()` as `old_logp` while the schedule has one update per batch.

## 5. LoRA findings

| Item | Value | How verified |
|---|---|---|
| Rank / alpha / scaling | r=8, α=16, α/r=2 | `model.py`; adapter_config.json |
| Dropout | 0.0 | `lora_config`; the model is never put in `train()` mode, which is consistent because dropout is 0 |
| Targets | `last2-attention`: `model.layers.{26,27}.self_attn.{q,v,o}_proj` (6 modules) | PEFT fullmatches the regex; module list asserted equal to `EXPECTED_MODULES` at runtime |
| Trainable parameters | 131,072 = Σ r·(in+out), fp32; base bf16 | asserted at runtime; checkpoint has 12 tensors and 131,072 fp32 parameters |
| Optimizer groups | AdamW over the `requires_grad` dict only (`lora_A`/`lora_B`), wd=0 | `train_multistep.py:137,155` |
| Frozen base | 0/310 tensors changed (SHA-256 digests) | historical runs and both audit runs |
| Gradients | finite; ∇A = 0 at step 1 (B=0 init makes ∂L/∂A = Bᵀ(·) = 0), nonzero from step 2 | smoke log: step 1 \|∇A\| = 0, \|∇B\| = 0.272; step 2 \|∇A\| = 0.0257, \|∇B\| = 0.327 |
| Effective change | ‖(α/r)BA‖ logged separately from ‖Δθ‖ | smoke: ‖Δθ‖ = 0.0256, 0.0258 per step; ‖(α/r)BA‖ = 0.0298, then 0.0445 |
| Initial model = base | B = 0, so (α/r)BA = 0 exactly; in-process base reproduction had 0 mismatches against the saved baseline (historical) | runtime assert; historical logs |

**LR=0 negative control.**
- **Unit test (new):** `test_lr_zero_step_leaves_parameters_and_policy_bitwise_unchanged` uses AdamW with lr=0 and wd=0.01. The gradient is nonzero, `update_norm` = 0, the parameters are bitwise unchanged, and log-probs after the step equal those before it.
- **Real model (audit run `_audit_lr0`, 2 steps):** gradient norm 0.272 / 0.328, `params_equal_init` True, subset predictions unchanged, 0/310 frozen tensors changed.
- **Historical (`grpo_lora_lr0_training.json`, recovered from git; see §9):** 10 steps, `control = {params_equal_init: True, subset_predictions_unchanged: True}`, `frozen_changed = []`.

## 6. Training findings

**Order of operations per batch:**
1. sample (`no_grad`);
2. reward;
3. skip if no group has mixed rewards;
4. advantage;
5. old log-probs (`no_grad`);
6. new log-probs (with graph);
7. check that old and new log-probs agree;
8. loss;
9. `zero_grad`;
10. backward;
11. finite-gradient check;
12. optimizer step;
13. `zero_grad`.

There is no gradient clipping (by design; logged as `grad_clipping: None`). `zero_grad(set_to_none=True)` runs before backward and after the step, so gradients cannot accumulate. Rollouts are deleted after every batch.

**Reward handling.**
- `decode` strips one terminating EOS and decodes. `parse_label` is an exact match after strip and upper-casing.
- The prompt template contains only the sentence; the label is never in the prompt.
- Labels reach only `group_rewards`, never generation.
- Audit runs: every logged batch was re-scored from its logged responses and labels with an independent one-liner, and all matched. Historical: 134/134 matched (previous audit).

**Zero-variance handling.**
- A batch is skipped only if **every** group is constant. It is consumed, not retried, and counted in `attempts`, `zero_signal` and `batches_by_mixed_groups`, but not in `step`.
- Partially constant batches step, with the dilution described in §3.2.
- TRL differs: it calls `optimizer.step()` with an exactly zero gradient, so AdamW's moment state still updates (`findings.md` §C).

**Step accounting** is logged separately for:
- rollout attempts (`groups_generated`);
- prompt groups (`attempts·B`);
- completions (`attempts·B·G`);
- batches by number of mixed groups;
- optimizer updates (`successful_steps`).

Historical: B=1 needed 36 attempts for 30 steps; B=2 balanced and unconstrained needed 31 attempts each, with mixed-group counts {0:1, 1:10, 2:20}.

**Determinism.** The `_audit` and `_audit_lr0` runs share seeds. They produced identical step-0 subset evaluations, and identical rollouts (equal `rollout_sha256`) and rewards at **both** steps. The lr>0 step-1 update was too small to change any sampled token under the same RNG stream. Their gradient norms are identical at step 1 (0.2721) and differ at step 2 (0.3281 vs 0.3280): on identical tokens, that difference is evidence that the lr>0 policy actually moved.

**Checkpointing.**
- `save_pretrained` writes the adapter only: 532 KB, 12 tensors, `adapter_config.json` with the target regex.
- The final checkpoint is reloaded from disk and asserted bitwise equal to the live adapter before full validation.
- Optimizer and RNG state are not saved, so a run cannot be resumed exactly; the docstring states this.
- There is no standalone "evaluate this manual checkpoint" CLI: manual runs evaluate in-process after reloading. The TRL `evaluate.py` loads its adapter in a separate process.

**CLI audit** (`research/train_multistep.py`, the substitute for Phase 7):

| Setting | Source | Reaches |
|---|---|---|
| `--lr` | CLI | AdamW; LR=0 control logic |
| `--steps`, `--max-attempts` | CLI | loop bounds |
| `--train-samples` | CLI | `balanced_subset(splits["train"], n, SEED)` |
| `--val-subset`, `--eval-every`, `--full-val` | CLI | evaluation |
| `--lora-targets` | CLI → `TARGETS` regex | `LoraConfig.target_modules`; module list asserted |
| `--batch-size`, `--batch-mode` | CLI, validated (cycle ⇔ B=1, balanced ⇒ B=2) | `batches()` |
| G=6, `MAX_NEW_TOKENS`=8, `SEED`=42 | module constants | `generate_rollouts`, seeds |
| `CLIP_EPS`=0.2 | module constant | **was not passed to `grpo_step`** (it used its own default of 0.2); fixed |
| temperature 1, top_p 1, top_k 0 | hard-coded in `policy.py` | `generate()`; the run log restates the same literals separately (a duplicated literal, LOW) |
| r, α | `model.py` constants | `LoraConfig`; imported into the log |
| model, dtype, non-thinking | `model.py`, `sentiment.build_prompts` (asserts the `<think>\n\n</think>\n\n` suffix) | model loading, chat template |

Because G is hard-coded, the requested **B=2, G=4 smoke test cannot be run through the entry point**. The smoke run used the real entry point with B=2, G=6. G=4 is covered by the value-level unit tests.

## 7. Evaluation findings

**Baseline recomputation.** An independent script that imports nothing from `qwen_grpo.evaluation` re-parsed every raw response, rebuilt the confusion matrix, computed precision/recall F1 and computed AUC by Mann–Whitney average ranks.

| Metric | Validation (mine / saved) | Test (mine / saved) |
|---|---|---|
| Accuracy | 0.748593 / 0.748593 | 0.732645 / 0.732645 |
| Macro-F1 | 0.749339 / 0.749339 (Δ 1e-16) | 0.733673 / 0.733673 |
| Recall NEG / POS | 0.782364 / 0.714822, identical | 0.767355 / 0.697936, identical |
| Predicted-POS rate | 0.466229, identical | 0.465291, identical |
| Invalid rate | 0.002814 (3/1066), identical | 0.003752 (4/1066), identical |
| ROC AUC | 0.827734 / 0.827734 | 0.816561 / 0.816561 |

- **Confusion matrices** (rows = predicted, columns = gold NEG/POS). Validation: NEG 417/149, POS 116/381, invalid 0/3. Test: NEG 409/157, POS 124/372, invalid 0/4. Both are identical to the saved matrices.
- **Invalid outputs** count as incorrect and as false negatives for their gold class. Every invalid output had gold label POSITIVE.
- **AUC scores** are finite. Ties: 1 on validation, 2 on test (bf16), counted as 0.5. The sign is correct: the mean score is +0.72 for POS-gold examples and −0.95 / −0.87 for NEG-gold examples.
- **AUC scoring path:** s(x) sums the teacher-forced log-probs of the full label tokenization plus `<|im_end|>`. It shares the verified `compute_logprobs`, and an existing test checks the factorized hand value.
- **Splits:** 1,500 / 1,066 / 1,066 examples, balanced, with 0 near-duplicate-key overlaps between any pair. Within each split all keys are unique.
- **No adapter in the baseline:** `research/baseline.py` loads a plain `AutoModelForCausalLM`. Inside training runs, base scores use `disable_adapter()`.
- **Non-thinking template:** `build_prompts` asserts that every prompt ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- **Tokenization:** `evaluation.generate` tokenizes with the default settings; training and `label_scores` pass `add_special_tokens=False`. The ids are identical for Qwen3 (new test).

**Comparability caveats** (MEDIUM, documented, not changed):
1. **Evaluation length changed.** `06489d3` changed the evaluation `MAX_NEW` from 16 to 4. The current baseline uses 4. Historical trained-run predictions used 16. On the validation set the base model's 1,066 responses are **identical** under 16 and 4, so the baseline itself is unaffected. A trained model that emits more than 3 tokens could parse differently. Rollouts use 8 tokens.
2. **Truncated responses can still count as correct.** `predict` ignores the `truncated` flag. A truncated response whose first tokens strip to a label (for example `"POSITIVE\n\n"` cut off at 4 tokens) counts as correct. The baseline has 0 truncated responses, so its numbers are unaffected. Treat this as a latent risk for trained models, and check the truncated rate whenever reporting a trained model.

## 8. TRL comparison

Re-read critically from `research/trl_reference/` (code, `config.json`, `findings.md`, results) and the passing tests.

| Item | Manual | TRL 1.18 (as configured) | Status |
|---|---|---|---|
| Advantage std | population | Bessel | **mathematical difference**: uniform ×√(5/6) per group |
| Loss reduction | per-sequence mean, then mean | same (`loss_type="grpo"`, GA=1) | equal |
| Clipping | inert (ρ≡1) | inert (`num_iterations=1`; logged clip ratio 0) | equal |
| KL | β=0 | β=0 | equal |
| Masks / EOS / truncation | through the first EOS; truncated rows kept | same (`mask_truncated_completions=False`) | equal |
| Gradient clipping | none | `max_grad_norm=0` | equal |
| Zero-signal batch | skipped, not stepped | stepped with zero gradient (AdamW moments move) | **optimizer-state difference** |
| Log-prob numerics | full bf16 logits, then fp32 `log_softmax` | fused chunked LM head | numerical only |

**Evidence.**
- **Unit test (tiny Qwen3, fp32, Bessel-aligned):** relative gradient difference < 1e-5 and |Δloss| < 1e-6 (passes). Without alignment the cosine is > 1 − 1e-9 and the norm ratio equals the Bessel factor, so the raw difference is that rescaling alone.
- **Real model with fixed rollouts** (`gradient_comparison.json`, `findings.md`):
  - fp32 aligned: δ = 2.0e-5 against an fp32 manual-vs-manual chunking floor of 1.2e-5;
  - bf16: δ = 0.103, cosine 0.995, inside the bf16 manual-vs-manual floor of 0.05–0.125.
- **Two tests skip on this branch** (§9): `test_dataset_matches_manual_subset` (training-subset ids equal the manual run's) and `test_complete_label_scoring_matches_saved_manual_results` (anchored label-score numbers). Run against the historical files recovered from git:
  - the subset test **passes**;
  - the label-score test cannot pass as written, because it reads `label_scores` from `validation.json`, a format no saved run has ever had (scores used to live in `training.json["final"]`);
  - re-running its assertions against the old location **holds**: base accuracy 0.7486 → trained 0.7261, score shift positive on 100% of examples, mean shift 0.650, Pearson r 0.997;
  - the historical in-run base scores equal the current baseline file to max |Δ| = 0.0.

**Conclusion.** With aligned advantages the two objectives are numerically equivalent in fp32. The remaining differences are documented conventions, not bugs.

## 9. Bugs found and fixed

| Severity | File | Problem | Evidence | Fix | Test |
|---|---|---|---|---|---|
| MEDIUM | `tests/test_grpo.py:8` | A module-level `torch.set_default_dtype(torch.float64)` leaked into every test module collected after it. `test_multistep.py` passed in the full suite but **failed when run alone**: fp32 advantages missed an `abs=1e-12` check by 5e-9. Test outcomes depended on order and partly masked fp32 behaviour. | `pytest tests/test_multistep.py` → 1 failed (`test_b2_opposite_labels...`, line 392) | Scoped to an autouse module fixture that restores the previous dtype. The affected test builds fp64 rewards explicitly, which its tolerance assumes. | Each test file now passes alone and in the full suite (§11) |
| LOW | `research/train_multistep.py:203` | `CLIP_EPS` was logged in the config but not passed to `grpo_step`, which used its own default. Both are 0.2, and ρ≡1, so behaviour is unchanged. The logged value simply wasn't wired to anything. | code reading | `grpo_step(..., clip_eps=CLIP_EPS)` | none possible: the effect is unobservable while ρ≡1; the smoke run exercises the line |

**Not fixed: these are the owner's decisions.**

| Severity | Item | Detail |
|---|---|---|
| MEDIUM (artifacts) | `06489d3` deleted the historical run logs | Deleted: `results/grpo_lora_*_training.json`, `*_step30_validation.json`, `grpo_lora_lr0_training.json` and the previous `grpo_lora_audit.md`. They were deleted, not moved. Effects: (a) 2 TRL tests skip; (b) `REFERENCES` in `train_multistep.py` quietly drops every saved run because of `p.exists()`; (c) on disk, the LR=0 evidence exists only as the adapter. All files are recoverable with `git show 06489d3^:<path>`. The audit read them from a scratch copy and did not restore them into the tree. Restoring them into the new `results/<run>/` layout also needs a format decision, because label scores moved from `training.json` to `validation.json`. |
| MEDIUM (comparability) | Evaluation `MAX_NEW` 16 → 4; truncated responses can count as correct | See §7. The baseline is unaffected; this is a risk for future comparisons. |
| LOW | Sampling literals duplicated | They appear in `policy.py` and again in the run log; G and `MAX_NEW_TOKENS` are not CLI flags. |

No CRITICAL or HIGH issues were found.

## 10. Remaining limitations

**Correctness-relevant, unresolved by design:**
- **Clipping is never active.** The algorithm in use is REINFORCE with a group-std baseline. Any PPO-style claim needs more than one update per batch.
- **old ≠ sampler in bf16.** The bf16 teacher-forced recompute differs from the sampling distribution by up to about 0.07 nats per token.
- **bf16 gradient noise from batch shape is 5–12%.** Small differences between configurations are not interpretable from one seed.

**Engineering limitations:**
- one training seed per configuration;
- no resume (no optimizer or RNG state is saved);
- no standalone checkpoint-evaluation CLI for manual runs;
- full [N, T, V] logits are materialized;
- the forward pass is run twice per step;
- G and the rollout length are not CLI-configurable.

**Gradient equivalence for chunked or micro-batched backward: not applicable.** The training path does one full forward and backward pass. Chunking exists only in evaluation scoring, which has no gradient, and in the research comparison script.

**Not re-run:** full baseline inference (the saved artifacts were sufficient), full training, and the TRL training run.

## 11. Test results

**Initial state** (main = `06489d3`):
```
$ .venv/bin/python -m pytest -q
84 passed, 2 skipped, 2 warnings in 21.80s
SKIPPED tests/test_trl_reference.py:103 / :196: saved manual run not found: results/grpo_lora_last2_b2_balanced
$ .venv/bin/python -m pytest -q tests/test_multistep.py
1 failed, 28 passed      # test_b2_opposite_labels_hand_rewards_and_independent_reference_gradient (order-dependent)
```

**Final state:**
```
$ for f in tests/test_*.py; do .venv/bin/python -m pytest -q $f; done
tests/test_audit.py:          7 passed
tests/test_grpo.py:           21 passed
tests/test_multistep.py:      29 passed
tests/test_policy.py:         7 passed
tests/test_sentiment.py:      20 passed
tests/test_trl_reference.py:  7 passed, 2 skipped
$ .venv/bin/python -m pytest -q
91 passed, 2 skipped, 2 warnings
```

**Mutation checks on `tests/test_audit.py`.** Each mutation was applied to `qwen_grpo/` and reverted with `git checkout`. Number of failing audit tests:

| Mutation | Failing tests |
|---|---|
| off-by-one causal slice | 1 |
| `labels[i % B]` instead of `labels[i // G]` | 2 |
| prompt-mask `repeat` instead of `repeat_interleave` (`policy.py:25`) | 0 (caught by existing `test_policy.py::test_prompt_grouping`) |
| sample std | 3 |
| global token normalization | 4 |
| mask drops EOS | 4 |
| clip removed | 1 |

**GPU smoke test** (the real entry point; G=6 because it is hard-coded; new tags, so nothing was overwritten):
```
$ .venv/bin/python -m research.train_multistep --lora-targets last2-attention --batch-size 2 --batch-mode balanced --steps 2 --tag _audit --no-full-val
trainable parameters: 131,072 of 596,180,992 (0.0220%)
  [val subset, step 0] acc 0.7812  F1 0.7810  rec POS 0.7500 NEG 0.8125  invalid 0.0000  pred POS 0.4688  (n=160)
step  1 (attempt   1) 630P,256N    mixed 2  R 0.67  loss -0.0000  |g| 2.721e-01 (A 0.00e+00, B 2.72e-01)  |dtheta| 2.56e-02  |(a/r)BA| 2.984e-02  alloc 1.13  peak res 1.87 GiB
step  2 (attempt   2) 1471P,514N   mixed 2  R 0.33  loss +0.0000  |g| 3.281e-01 (A 2.57e-02, B 3.27e-01)  |dtheta| 2.58e-02  |(a/r)BA| 4.448e-02  alloc 1.13  peak res 1.87 GiB
  [val subset, step 2] acc 0.7750  F1 0.7747  rec POS 0.7375 NEG 0.8125  invalid 0.0000  pred POS 0.4625  (n=160)
steps 2/2, batches 2, zero-signal batches 0, frozen changed 0/310, params equal init: False
peak allocated 1.56 GiB, peak reserved 1.87 GiB, device 4.00 GiB, budget 4.0 GiB: PASS
final_checkpoint: checkpoints/grpo_lora_last2_b2_balanced_audit_step2, reloaded_bitwise_equal: True

$ ... --lr 0 --tag _audit_lr0
step 1/2: |g| 2.721e-01 / 3.280e-01, |dtheta| 0.00e+00, |(a/r)BA| 0
LR=0 control: {'params_equal_init': True, 'subset_predictions_unchanged': True}; frozen changed 0/310
```

- **Logged rewards** were re-scored by hand from the logged responses and labels: all 4 batches matched. Step 1 rewards were `[[1,1,1,1,0,1],[0,0,1,1,1,0]]` for labels [POSITIVE, NEGATIVE].
- **Checkpoint:** 12 fp32 tensors, 131,072 parameters, 532 KB, adapter only.
- **Outputs** are left **untracked**:
  - `results/grpo_lora_last2_b2_balanced_audit{,_lr0}/`;
  - `checkpoints/grpo_lora_last2_b2_balanced_audit{,_lr0}_step2/`.

**Baseline recomputation script:** `/tmp/claude-1000/-home-aldiyar-qwen-grpo/c10730cd-d207-411e-9285-766a3fcb4c6f/scratchpad/baseline_check.py`. It lives in the session scratchpad, outside the repository, and may be wiped; its outputs are reproduced in §7.

## 12. Readiness for L2L

**Yes**, as the resident (non-streamed) reference, under these conditions:
1. **Compare on fixed rollouts.** Feed identical `input_ids`, masks and advantages to both paths through `compute_logprobs` → `grpo_loss` → backward. That bypasses sampling, whose bf16 path depends on shape.
2. **Compare in fp32 first.** Manual vs manual agrees to about 1e-5 in fp32, but bf16 batch-shape noise alone is 5–12% of the gradient. A bf16 L2L check needs a tolerance justified against that measured floor, not a fixed atol.
3. **Compare gradients, not just logits and loss.** The logged loss is ≈ 0 by construction (§3.3), so loss agreement is nearly vacuous. Compare the LoRA gradients per tensor (relative norm and cosine), and remember that ∇A = 0 at the first step.
4. **The reference is a single full forward and backward pass.** A streamed or checkpointed implementation must match Σ_k ∇L_k against this one-pass ∇L. No chunked reference exists in the training path yet.
