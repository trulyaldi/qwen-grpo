# Verification: `checkpoints/default_step30`

## Experiment configuration

**No training log for this checkpoint** (no training log at results/default/training.json). The GRPO settings below are those of the YAML; only the LoRA settings are confirmed (against the checkpoint's adapter_config.json).

- Model Qwen/Qwen3-0.6B (bfloat16 base, fp32 LoRA), non-thinking; dataset Rotten Tomatoes (`data/sentiment.json`).
- LoRA r=8, α=16, dropout 0.0, targets `last2-attention`.
- GRPO: B=2 (balanced), G=6, lr 0.0001, 30 steps, clip ε 0.2, rollout max_new_tokens 8, seed 42.
- Held-out: `validation`, n=1066 {'NEGATIVE': 533, 'POSITIVE': 533}; greedy, max_new_tokens 4 (baseline settings).
- Paired bootstrap: 2000 resamples, 95% percentile CIs, seed 42.

## Model comparison

| Metric | Base | Adapter | Merged | LR=0 |
|---|---:|---:|---:|---:|
| Accuracy | 0.7486 | 0.7261 | 0.7326 | 0.7486 |
| Macro-F1 | 0.7493 | 0.7222 | 0.7293 | 0.7493 |
| Negative recall | 0.7824 | 0.5966 | 0.6098 | 0.7824 |
| Positive recall | 0.7148 | 0.8555 | 0.8555 | 0.7148 |
| Predicted-positive rate | 0.4662 | 0.6295 | 0.6229 | 0.4662 |
| Invalid rate | 0.0028 | 0.0019 | 0.0019 | 0.0028 |
| ROC AUC (label scores) | 0.8277 | 0.8366 | 0.8375 | 0.8277 |

- Base: reused results/baseline/validation.json; first 32 examples re-scored in-process: identical
- Adapter: checkpoints/default_step30 attached to Qwen/Qwen3-0.6B (bfloat16)
- Merged: PEFT merge_and_unload in torch.bfloat16 (W0 + (a/r)BA rounded to the base dtype), in place
- LR=0: LR=0 control checkpoints/default_lr0_step30 evaluated with the adapter attached

## Paired differences (new − reference)

`*` marks a CI that excludes zero. One training seed: the CIs cover evaluation sampling only.

| Comparison | Δ accuracy [CI] | Δ macro-F1 [CI] | Δ AUC [CI] | improved / degraded |
|---|---|---|---|---|
| adapter vs base | -0.0225 [-0.0469, +0.0019] | -0.0272 [-0.0508, -0.0035] * | +0.0089 [+0.0063, +0.0116] * | 75 / 99 |
| merged vs base | -0.0159 [-0.0394, +0.0075] | -0.0200 [-0.0438, +0.0034] | +0.0097 [+0.0072, +0.0123] * | 75 / 92 |
| merged vs adapter | +0.0066 [+0.0000, +0.0131] | +0.0071 [+0.0003, +0.0147] * | +0.0008 [-0.0009, +0.0026] | 10 / 3 |
| lr_zero vs base | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | 0 / 0 |
| adapter vs lr_zero | -0.0225 [-0.0469, +0.0019] | -0.0272 [-0.0508, -0.0035] * | +0.0089 [+0.0063, +0.0116] * | 75 / 99 |

## Verification evidence

### 1. Adapter parameters changed — PASS

- Initialization: reconstructed: torch.manual_seed(42) + lora_model(), as in train(); validated by the LR=0 control: True (the control validates the seeded procedure; that this checkpoint's run used the same seed is unconfirmed without its training log).
- ‖θ − θ₀‖₂ = 3.1166e-01; 12/12 LoRA tensors changed; checkpoint holds only LoRA tensors: True.
- Effective ‖(α/r)(BA − B₀A₀)‖_F per module: 26.self_attn.q_proj 1.266e-01 (2.44e-03 of ‖W₀‖), 26.self_attn.v_proj 8.373e-02 (2.77e-03 of ‖W₀‖), 26.self_attn.o_proj 8.092e-02 (2.03e-03 of ‖W₀‖), 27.self_attn.q_proj 1.464e-01 (3.29e-03 of ‖W₀‖), 27.self_attn.v_proj 8.199e-02 (2.48e-03 of ‖W₀‖), 27.self_attn.o_proj 7.629e-02 (1.81e-03 of ‖W₀‖).
- Training log: not available (no training log at results/default/training.json). Frozen-weight digests, gradient norms and reward curves of this run cannot be checked here.

### 2. Policy outputs changed

- Adapter vs base on 1066 held-out examples: mean |Δs| 0.6505, max |Δs| 1.3748, mean Δs +0.6505 (s = log P(POSITIVE⟨im_end⟩) − log P(NEGATIVE⟨im_end⟩)); greedy predictions differ on 16.32%; correctness changed on 174 (75 improved, 99 degraded).
- A nonzero change shows the adapter is active, not that it helps; a mean Δs far from 0 indicates a label-prior shift.

### 3. Merge equivalence — fp32 (merge correctness) PASS, bf16 (behavior preserved at deployment precision) FAIL

- fp32 on CPU, 8 examples, all positions of prompt + POSITIVE⟨im_end⟩: max |Z_adapter − Z_merged| 2.289e-05 = 6.22e-07 of max |Z| (tolerance 1e-04); the adapter itself moves logits by 7.560e-01 (power ratio 3.3e+04). Merged weights match W₀ + (α/r)BA to max 1.48e-08.
- bf16 on GPU (deployment precision): max |ΔZ| 2.542e-01 (6.92e-03 relative) vs adapter effect 8.750e-01; on the full split, greedy predictions differ on 1.22% of examples, max |Δs| 0.2501. Storing W₀ + (α/r)BA in bf16 distorts the update itself: ‖(W_merged − W₀) − (α/r)BA‖_F / ‖(α/r)BA‖_F = 0.40–0.62 across modules: ‖ΔW‖ is a small fraction of ‖W₀‖ (section 1), not far above bf16's relative precision (2⁻⁸). The attached adapter keeps the update in fp32. So the merge is algebraically correct (fp32 check), but a bf16 merge does not exactly preserve this adapter's behavior: deploy with the adapter attached or merge in fp32.

### 4. LR=0 negative control — PASS

- `checkpoints/default_lr0_step30`: adapter equals the seeded initialization bitwise: True; B = 0: True; greedy disagreement with base 0.00%, max |Δs| 0.0.
- The control shows that the training procedure itself (sampling, rewards, backward) changes nothing without optimizer steps. It cannot show that a trained run's improvement is real learning rather than noise or reward exploitation.

### 5. Reward integrity

- Training reward contract: strip() + upper() + exact match to POSITIVE/NEGATIVE. Probe strings rewarded for gold POSITIVE: `'POSITIVE'`, `'positive'`, `' POSITIVE '`, `'POSITIVE\n\n'` (of 12 probes).
- Independent check: response == canonical label. Disagreements (reward 1, not canonical):
  - Base: 0/1066; logged-correctness mismatches 0; top responses {'NEGATIVE': 566, 'POSITIVE': 497, 'Satisfying': 1, 'DECASIA': 1, 'Honest': 1}
  - LR=0: 0/1066; logged-correctness mismatches 0; top responses {'NEGATIVE': 566, 'POSITIVE': 497, 'Satisfying': 1, 'DECASIA': 1, 'Honest': 1}
  - Adapter: 0/1066; logged-correctness mismatches 0; top responses {'POSITIVE': 671, 'NEGATIVE': 393, 'DECASIA': 1, 'Honest': 1}
  - Merged: 0/1066; logged-correctness mismatches 0; top responses {'POSITIVE': 664, 'NEGATIVE': 400, 'DECASIA': 1, 'Honest': 1}
  - Training rollouts: not available (no training log).
- Zero disagreements show the model did not exploit the parser's leniency on these samples, not that exploitation is impossible.

## Interpretation

1. **Parameters changed**: section 1 (raw and effective LoRA deltas against the initial adapter).
2. **Policy changed**: section 2 (label-score shifts and prediction flips vs base).
3. **Better held-out performance**: only if the adapter-vs-base CIs exclude zero; otherwise not established.
4. **Merging preserves behavior**: section 3 (fp32 for correctness; bf16 for deployment-precision effects).
Rising training reward is none of these: it is measured on the policy's own samples from 32 training prompts, it can rise from exploiting the parser or a label prior, and the GRPO loss is ≈ 0 by construction (advantages sum to zero per group at ρ = 1).

## Plots

- `plots/model_comparison.png`
- not generated: training plots (reward, validation accuracy, mixed-reward fraction, gradient norm, loss): no training log at results/default/training.json

## Limitations

- One training seed: seed-to-seed variation of the trained adapter is not measured.
- Reward exploitation outside the probe strings and the observed responses is not covered.
- One held-out split from the training distribution: distribution shift and rare failures are not measured.
- bf16 results depend on batch shape and hardware; the fp32 merge check is the hardware-robust statement.
- Behavior outside the two-label task (fluency, other prompts) is not measured by classification metrics.
- Memory during verification: GPU peak reserved 2.60 GiB (torch allocator, GPU phase only; excludes the CUDA context); host peak RSS 4.65 GiB.
