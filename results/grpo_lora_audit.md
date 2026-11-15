# GRPO + LoRA implementation audit (Qwen3-0.6B)

Date: 2026-11-15. Scope: `grpo.py`, `policy.py`, `sentiment.py`, `baseline.py`, `train_lora.py`, `train_multistep.py` and the saved 30-step artifacts.

The audit did not modify any training, evaluation, reward, prompt or data code, did not train, and overwrote no artifact. It added two tests. All diagnostics were bounded and used saved checkpoints.

## A. Executive verdict

**CORRECT WITH LIMITATIONS.**

No critical, major or minor correctness bug was found. Each item below was checked independently of the implementation's own indexing:
- prompt/response/reward alignment;
- group advantages;
- causal token alignment, including left padding;
- EOS and padding masks;
- loss reduction;
- autograd gradients against a closed-form reference;
- LoRA wiring;
- evaluation metrics;
- ROC AUC scoring.

The limitations are properties of the chosen schedule and of bf16:
- PPO clipping is never active, so each update is REINFORCE with a group-std baseline.
- The "old" policy is a teacher-forced bf16 pass, not the sampler's distribution.
- The bf16 decision boundary is sensitive to batch shape.

The POSITIVE drift is best described as a near-uniform shift of the first-token decision boundary. That shift is consistent with a property of the objective (section E, H6/H7). There is no evidence of a bug behind it.

## B. Findings

| Severity | File/function | Finding | Evidence | Impact |
|---|---|---|---|---|
| Design limitation | `train_multistep.py:134-140` `grpo_step`; `grpo.py:55` | Old and new log-probs come from two identical teacher-forced passes immediately before one update, so ρ ≡ 1. The clip and `torch.minimum` never act. The update is REINFORCE with group-std normalization and 1/\|o\| weighting. | `init_diff` (max \|new−old\| log-prob) is exactly 0.0 at all 120 logged steps across 4 runs, so 0 valid tokens lie outside [0.8, 1.2]. | Clipping cannot be credited with any stability. This is not a bug: it follows from one update per fresh rollout batch. |
| Numerical limitation | `policy.py:35-42` vs `generate()` | "Old" log-probs are teacher-forced bf16, not the sampling distribution. The bf16 mismatch between generate and teacher forcing was measured earlier at up to ~0.07 nats per token. | Measured during the Stage 2 one-step LoRA verification (generate scores vs teacher-forced, bf16). The fp32 tiny model agrees to 1e-5 (new test). | A small, uncorrected off-policy error per token. It is not class-specific and cannot explain a systematic drift. |
| Unverified hypothesis | `grpo.py:4-8` (std normalization) | With std normalization, a group's push on the binary label logit is 6σ_k, largest when p(correct) ≈ 0.5. If the model is worse on one class, that class's prompts push the shared bias harder. | At init, p(correct first token) on the 32 training prompts is 0.632 for POS and 0.707 for NEG. The expected shared push toward POS is +2.33 under GRPO weighting and +0.68 under plain REINFORCE p(1−p). | A bug-free candidate mechanism, not causally tested, that predicts a POS-ward shared shift at init. It predicts the same sign for every configuration (section E, H7). |
| Design limitation (minor) | `grpo.py:60` per-sequence 1/\|o\| | At the decision position the gradient is Σᵢ(Aᵢ/\|oᵢ\|)(e_{yᵢ} − π). The π coefficient C = ΣAᵢ/\|oᵢ\| is zero only when lengths are equal. | Canonical labels are both 3 tokens: `[17456,44350,151645]` and `[97127,23215,151645]`, which are exactly what greedy decoding emits. 182 of 186 sampled valid responses were canonical. Run-level mean C per step (approximate, from logs): −0.000 / −0.016 / +0.051 / +0.047 (o_proj B=1 / last2 B=1 / uncon / bal). The new test checks C by hand. | Valid POS and NEG responses receive equal weight. The residual comes only from invalid, truncated or non-canonical responses. It enters the POS−NEG logit push only as C·(π_POS − π_NEG), so its effect there is at most \|C\|. In the B=2 runs C ≈ 0.05, against a realized POS−NEG push of 0.11 to 0.28. Not negligible, but not dominant. |
| Numerical limitation | `label_scores` (`train_multistep.py:203-219`) vs `baseline.predict` | sign(s) disagrees with the greedy prediction on 28 (base) and 33 (balanced) of about 1,063 valid examples. | Every disagreement has \|s\| ≤ 0.37, against a median \|s\| of 1.0–1.1. Scoring uses batches of 8, greedy decoding uses 32 with a KV cache, all in bf16. | Affects only examples at the threshold. AUC uses s throughout and is unaffected. |
| Design limitation | `label_scores` | s compares only the canonical label strings. | Mean first-token mass P(POS)+P(NEG) is 0.94, but the minimum is 0.04. | For a few examples, s ranks two unlikely continuations. Treat AUC as a ranking of canonical-label preference. |

Severity classes not listed (critical, major and minor bug) had **no findings**. One test gap was found and closed (section C).

## C. Mathematical verification

Each item lists what was checked and whether the check is independent of the code.

**Advantages.** `grpo.py:4-8` computes (R − mean)/(population std + 1e-4) per row. Constant groups give exactly 0, not NaN.
- New test: 12 hand-computed rewards and advantages for B=2 with opposite labels (POS group: [1,1,0,1,0,0]; NEG group: [1,1,1,1,1,0]).
- Existing tests show that normalization is per group, not joint.

**Reward alignment.**
- Rows b·G…b·G+G−1 belong to prompt b: `generate` uses `repeat_interleave`, and every step asserts this with `check_grouping` (`train_multistep.py:92-95, 347`).
- All 134 logged batches across 4 runs were re-scored from their logged responses and labels, and 134 of 134 match.
- `parse_label` is an exact match after strip and upper-casing. It does no substring matching, so `"NEG"`, `"POSTIVE"` and `"NO NEGATIVE"` all score 0.

**Log-probabilities.**
- `grpo.py:31` scores column P−1+t for token y_t, verified against a hand loop and an "all mass on the next token" construction (existing tests).
- New test: on a tiny fp32 Qwen3 with a left-padded row, the teacher-forced log-probs equal `generate(output_logits=True)`'s own logits and an unpadded single-row pass to within 1e-5.
- Test gap closed: no earlier test compared left-padded teacher forcing with generate's own logits. Mutation checks on the new tests:
  - M1, `position_ids` replaced with `None`: the test still passes. This is legitimate: RoPE attention depends only on relative positions, so `policy.py:38` is not a correctness risk.
  - M2, off-by-one slice in `grpo.py:31`: caught.
  - M3, per-sequence normalization removed in `grpo.py:60`: caught.

**EOS and padding.**
- The mask covers tokens up to and including the first EOS; it is built from positions, not from `!= pad`, because pad = `<|endoftext|>` is itself an EOS id (`grpo.py:11-19`).
- A truncated response is kept whole.
- The completion attention mask equals this mask.
- Denominator = number of valid tokens (`grpo.py:57-60`).
- Covered by existing tests: pad-after-EOS, pad-as-first-EOS, EOS-only and truncated responses.
- In context, `prompt + label + <|im_end|>` tokenizes identically to the concatenation of the parts (checked on the real tokenizer).

**Loss and gradient.**
- The formula matches the stated objective: token ratio, `min` of the unclipped and clipped terms, per-sequence token mean, mean over all B·G sequences, sign −J, `old_logp.detach()`.
- New independent reference: a model whose logits are a free parameter Z per position. One `grpo_step` with SGD (lr=1) gives ΔZ equal to the hand-looped −∂L/∂Z = (1/N) Σᵢ (Aᵢ/|oᵢ|) Σ_t (e_{y_it} − π). That reference never calls `grpo_loss` or `response_logprobs`.
- At the decision position, the POS logit rises and junk falls (asserted). For POS, group 0's correct answers contribute +a0, group 1's wrong POSITIVE answer (A < 0) contributes −0.745, and the −Cπ term is small.
- Existing tests: gradcheck across the clip kinks; zero advantage gives zero gradient; a constant group contributes exactly nothing (B=2).

**LoRA.** Asserted at runtime in every run (`train_multistep.py:292-302`):
- exact module list (1 or 6 projections) and r·(in+out) parameters;
- only `lora_A`/`lora_B` are trainable, in fp32, on a bf16 base;
- B = 0 at init; dropout 0; α/r = 2;
- `zero_grad(set_to_none)` before and after each step;
- non-finite gradients block the step.

After each run, 0 of 310 frozen tensors had changed (SHA-256), and the LR=0 control left the adapter bitwise unchanged. There is no merging during training; generate runs with the adapter active, and `disable_adapter()` is used only for base scoring.

**Loop order.** For each batch:
1. sample;
2. generate with the current adapter (`no_grad`);
3. reward;
4. advantage;
5. old log-probs (`no_grad`);
6. new log-probs (with graph);
7. loss;
8. backward;
9. step.

No stale tensors survive, because rollouts are deleted every batch. Evaluation and checkpoint saving run after the step and read only parameters, and the final checkpoint reload is bitwise equal.

**Not verified.**
- The real-model LoRA gradient was checked against a first-order prediction only in the earlier `train_lora.py` work, not here.
- Training-seed variance is unmeasured.
- The audit's own model diagnostic peaked at **4.28 GiB reserved, above the 4 GiB local budget**. That was full-validation scoring of 4 adapters plus sampling, not training; the training runs peaked at ≤ 2.56 GiB.

## D. Evaluation verification

**Metrics.** In all 4 prediction files (n = 1,066), ids run 0…1065 in order. `prediction == parse_label(response)` and `correct == (prediction == label)` hold for every record. Invalid outputs count as incorrect, and as false negatives in F1. The confusion matrices reproduce the reported accuracy, recall and predicted-POS rate. The macro F1 formula 2TP/(gold+pred) was checked against a hand value (existing test).

**Prompts and decoding.** Every model is evaluated through `baseline.predict`: same prompt, greedy decoding, 16 new tokens, batch 32. The in-process base model reproduces `baseline_validation.json` with 0 mismatches in every run. Default versus `add_special_tokens=False` tokenization gives identical ids for Qwen (checked on 200 prompts).

**ROC AUC.**
- s(x) = Σ log P(POS, ITIVE, `<|im_end|>`) − Σ log P(NEG, ATIVE, `<|im_end|>`), scored on the evaluation prompt. The new token-alignment test also covers this path, because it reuses `compute_logprobs`.
- An existing test checks the factorized hand value.
- Sign convention: s > 0 agrees with a greedy POSITIVE prediction on 97% of examples, and every disagreement sits near the threshold.
- Re-scoring all 4 adapters in a separate process reproduced the saved s to within 1e-6.
- AUC uses continuous s, with POSITIVE as the positive class. Ties count as 0.5.

**Bootstrap.**
- Accuracy is paired by example id; alignment is asserted on (id, text, label).
- The AUC bootstrap resamples within each class using shared multiplicity vectors, so both models see the same resample. It is tested on a hand case.

**Checkpoints.** The final adapter is reloaded from disk and asserted bitwise equal before full validation. Reference adapters loaded for scoring are restored afterwards (`train_multistep.py:420-424`).

## E. POSITIVE-drift investigation

| Hypothesis | Evidence for | Evidence against | Conclusion |
|---|---|---|---|
| H1 reward misalignment | — | `check_grouping` passes at every step. 134 of 134 logged batches re-score identically. The new opposite-label test passes. | **Contradicted** |
| H2 tokenization asymmetry | — | Both labels are 2 tokens + EOS in context. Greedy decoding emits exactly the canonical ids. Mean P(second token) ≥ 0.998 and mean P(EOS) ≥ 0.9998 for both labels, before and after training. | **Contradicted** as a cause |
| H3 length normalization | C ≠ 0 whenever invalid, truncated or non-canonical responses appear (new test). In the B=2 runs the run-level mean C is about +0.05 per step. | Valid labels have equal \|o\| = 3. C varies in sign across runs (−0.016 in last2 B=1, which still drifted POS), and its effect on POS−NEG is bounded by \|C\|. | **Contradicted for valid responses.** A minor contribution from the invalid-response residual is not ruled out. |
| H4 first-token asymmetry | All of the change in s is in the first token: shift vs base is +0.36 / +0.54 / +0.65 (B=1 / uncon / bal); second-token and EOS terms are ≤ 1e-4. | — | **Supported** as the *location* of the change. Expected, since later tokens are saturated. It does not explain *why* the shift is POS-ward. |
| H5 prompt order | B=1 subset predicted-POS rate tracked recent prompt labels (0.71 at step 10 after 5 POS prompts; earlier run). | Balanced batches (one of each label every step) drifted the most. | **Contradicted** as the cause of the final drift; it explains B=1 trajectory swings. |
| H6 decision-boundary drift | The first-token shift is positive for **100%** of the 1,066 validation examples in all three last2 runs. For balanced: mean +0.65, sd 0.19, corr(base s, trained s) = 0.997. The shift is larger for POS-gold (+0.76) than NEG-gold (+0.55) examples, which is the small AUC gain. | — | **Supported**: a near-uniform bias shift plus a small ranking gain |
| H7 (new) std-normalized class weighting | At init, the first-token p(correct), renormalized over the two labels, is lower on POS training prompts (0.632 vs 0.707), so their groups have larger σ and push the shared bias harder. Expected push: +2.33 toward POS (GRPO) and +0.68 (REINFORCE). Valid in every configuration. | It predicts one sign for all configs, so it cannot explain *differences* between them. The feedback should be self-limiting (as POS accuracy rises, NEG groups push back), yet balanced overshoots to NEG recall 0.60. The realized first-token push from logs, Σ(Aᵢ/\|oᵢ\|) on POS minus NEG tokens per step (approximate: it re-tokenizes decoded text and infers truncation from length), is +0.106 / −0.026 / +0.280 / +0.110 (o_proj B=1 / last2 B=1 / uncon / bal). **last2 B=1 drifted POS despite a slightly negative push.** | **Plausible, partially supported; untested causally** |

Rollouts are not paired across runs, and every run uses a single training seed. The differences *between* configurations (0.555 vs 0.598 vs 0.630 predicted-POS) are not attributed to any mechanism and may be seed noise.

## F. Required fixes

**Mandatory correctness fixes: none.** No failure was demonstrated.

Added (tests only; no implementation change):
- `tests/test_policy.py:107`: left-padded teacher forcing matches generate's logits and an unpadded row.
- `tests/test_multistep.py:326`: B=2 opposite labels, hand rewards and advantages, closed-form gradient reference with a 1/|o| residual.

Result: 76 tests pass.

Optional research improvements, not bugs and not applied:
1. Report clipping as inactive, or use more than one update per rollout batch if PPO-style clipping is actually wanted.
2. Measure training-seed variance before attributing between-configuration differences.
3. If the label prior matters, consider a diagnostic that tracks the per-class p(correct) of the training prompts during training (it tests H7).

## G. Validity of previous experiments

1. **Mechanically valid 30-step runs?** Yes. The math, alignment, masking, LoRA wiring and frozen weights are verified, and the schedule is as designed. Clipping was inactive throughout.
2. **Rewards and advantages correct?** Yes: re-derived from the logs and verified by hand.
3. **Accuracy comparisons reliable?** Yes, as evaluation-sample statistics. The CIs exclude training-seed variance.
4. **AUC differences reliable?** Yes as computed: scoring is reproducible to 1e-6, the sign is correct and the bootstrap is correct. They are small (≤ 0.009) and single-seed, and they rank canonical-label preference only.
5. **Could any discovered issue explain the drift?** No implementation issue does. The drift is a uniform first-token boundary shift, and H7 is a bug-free candidate mechanism.

One previous statement needs correcting. The Stage 3C report listed "the 1/|o| length normalization" and "first-token tokenization asymmetry" as candidate causes. H2 and H3 are now contradicted for valid responses.

## H. Final recommendation

**Proceed to L2L streaming.**

The GRPO + LoRA pipeline is verified within the tested cases, so it can serve as the resident reference that Stage 3 and Stage 4 must match. The drift is a property of the objective and data under this schedule, not a bug. Repeating the experiments would not change the mechanics.

If the owner wants the drift explained before Stage 5's SHIP verdict, the single narrowly scoped diagnostic is to track per-class p(correct) on the training prompts at each checkpoint, which tests H7. That is a research choice, not a correctness prerequisite.

## Reproduction note

The numbers in sections B and E (push table, C, tokenization, per-token split of s, H7 expected push) come from three bounded scripts. They are **outside the repository**, in a session scratchpad that may be wiped:

- `/tmp/claude-1000/-home-aldiyar-qwen-grpo/1f194e09-f1ff-4fec-ae81-57877b2b7748/scratchpad/logs_diag.py`: reads the saved training logs; ±1 label push and response forms.
- `/tmp/claude-1000/-home-aldiyar-qwen-grpo/1f194e09-f1ff-4fec-ae81-57877b2b7748/scratchpad/scores_diag.py`: reads the saved label scores; uniform-shift decomposition.
- `/tmp/claude-1000/-home-aldiyar-qwen-grpo/1f194e09-f1ff-4fec-ae81-57877b2b7748/scratchpad/model_diag.py`: loads Qwen3-0.6B and the four step-30 adapters. Covers tokenization in context, first-token push from logs, per-token s, emitted ids and H7. This is the script that peaked at 4.28 GiB reserved.

Run them from the repo root with `.venv/bin/python`.
