# 07 — GRPO mathematics, equation by equation, next to the code

**What you will be able to explain after this file**
- The objective the code optimizes, written as an explicit finite sum over the 30 completions, and the Python line that computes each symbol.
- Why, with `num_iterations: 1`, the PPO clip and the importance ratio have **no effect at all** on the update, and what the gradient reduces to.
- What KL β = 0 removes, and what still limits how far one update moves the policy.
- How micro-batch scaling yields the exact full-batch gradient, and what AdamW does with that gradient (including the surprising first step).
- Why binary rewards, zero-signal groups and length normalization shape the learning signal, and how this connects to the label collapse measured on Qwen3-0.6B.

Symbols: prompts q_b (b = 1..B = 5), completions o_i (i = 1..N = 30, row i answers prompt b(i) = ⌊i/6⌋), tokens o_{i,t}, |o_i| = number of counted tokens (through the first EOS), θ = the 1,081,344 LoRA parameters, θ_old = their value when the batch was sampled.

---

## 1. The policy and its sequence probability

The policy is the autoregressive distribution of the frozen Qwen3-4B with the LoRA delta added:

$$
\pi_\theta(o \mid q) = \prod_{t=1}^{|o|} \pi_\theta(o_t \mid q, o_{<t}),\qquad
\log \pi_\theta(o \mid q) = \sum_t \log \pi_\theta(o_t \mid q, o_{<t}).
$$

Each factor is a softmax over the 151,936-row output head:

$$
\pi_\theta(v \mid q, o_{<t}) = \frac{\exp z_v}{\sum_{u} \exp z_u},\qquad z = E\,h_t,
$$

with h_t the final-norm hidden state at the column **before** token t and E the tied embedding matrix `[151936, 2560]`.

**Code.** `grpo.response_logprobs` (`qwen_grpo/grpo.py:22-38`) takes the logits of columns `P-1 … T-2`, upcasts bf16 → fp32, applies `log_softmax`, and gathers the realized token: `logp [n, R]` with `logp[i, t] = log π_θ(o_{i,t} | q, o_{i,<t})`. The L2L path computes the same thing via `output_head.streamed_logprobs` (`qwen_grpo/execution/output_head.py:279-305`, full head, `vocab_chunk_size: null`). Padding columns after EOS also get a number, but the mask removes them later.

Example for "POS ITIVE <|im_end|>": three log-probs, e.g. log π(POS | prompt), log π(ITIVE | prompt, POS), log π(<|im_end|> | prompt, POS, ITIVE). The last two are close to 0 for any sane model; essentially all the decision is in the first token. That is why a classification task generates almost no gradient through tokens 2 and 3.

## 2. Rewards and group-relative advantages

Reward: r_i = 1 if the decoded response, stripped and upper-cased, is exactly the gold label, else 0 (`qwen_grpo/sentiment.py:35-42`).

Advantage, within each group of G = 6 (`group_advantages`, `grpo.py:4-8`):

$$
A_i = \frac{r_i - \bar r_{b}}{\sigma_b + 10^{-4}},\qquad
\bar r_b = \tfrac16\textstyle\sum_{j\in b} r_j,\quad
\sigma_b = \sqrt{\tfrac16\textstyle\sum_{j\in b}(r_j-\bar r_b)^2}\ \ (\text{population std}).
$$

For binary rewards with k correct of 6, p = k/6, σ = √(p(1−p)):

$$
A_{\text{correct}} = \sqrt{\tfrac{1-p}{p}},\qquad A_{\text{wrong}} = -\sqrt{\tfrac{p}{1-p}}.
$$

| k | correct row | wrong row |
|---|---|---|
| 0 or 6 | — | 0 (exactly: 0 / 1e-4) |
| 1 | +2.236 | −0.447 |
| 2 | +1.414 | −0.707 |
| 3 | +1.000 | −1.000 |
| 4 | +0.707 | −1.414 |
| 5 | +0.447 | −2.236 |

(These are the values the repo function returns, computed in a scratch run **[code]**.) Properties used later:
- Σ_{i∈b} A_i = 0 in every group; Σ_{i∈b} A_i² = 6 for every mixed group (unit variance), 0 for uniform groups.
- The rare outcome gets the large advantage: one correct answer among six wrong-ish ones is pushed up hard (+2.24).
- **Advantages are per sequence**, shared by all its tokens (`adv.unsqueeze(1)`, `grpo.py:58`).

## 3. Importance ratio and the clipped surrogate

Per token:

$$
\rho_{i,t}(\theta) = \exp\big(\log\pi_\theta(o_{i,t}\mid\cdot) - \log\pi_{\theta_\text{old}}(o_{i,t}\mid\cdot)\big),
$$
$$
s_{i,t}(\theta) = \min\Big(\rho_{i,t}A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\,A_i\Big),\qquad \epsilon = 0.2.
$$

Sequence-normalized objective and loss (β = 0):

$$
J_i(\theta) = \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} s_{i,t}(\theta),\qquad
\mathcal L(\theta) = -\frac{1}{N}\sum_{i=1}^{N} J_i(\theta).
$$

**Code** (`grpo_loss`, `grpo.py:41-65`), line by line:

| math | code |
|---|---|
| log ρ, zero on padding | `log_ratio = torch.where(mask, logp - old_logp, 0)` (line 56) |
| ρ | `ratio = torch.exp(log_ratio)` (57) |
| s_{i,t} | `torch.minimum(ratio * adv, torch.clamp(ratio, 1-eps, 1+eps) * adv)` (59) |
| mask | `torch.where(mask, surrogate, 0)` (60) |
| \|o_i\| | `mask.sum(dim=1)`; an empty response raises (61-63) |
| J_i | `surrogate.sum(dim=1) / lengths` (64) |
| 𝓛 | `-per_sequence.mean()` (65) |

`old_logp` is detached (line 55): θ_old is a constant.

This is the GRPO objective of DeepSeekMath (Shao et al. 2024, arXiv 2402.03300) with β = 0, i.e. PPO's clipped surrogate (Schulman et al. 2017, arXiv 1707.06347) with a group-normalized advantage in place of a learned value baseline. Both papers are standard references; the repository's own literature record (`research/l2l/literature.md`) does not list them.

## 4. What `num_iterations = 1` does to all of this

### 4.1 The ratio is exactly 1

`prepare` computes `old_logp` under θ_old, in micro-batches of 6, without a graph (`training.py:93-94`). `grpo_update` (iteration 0) recomputes `logp` under **the same parameters**, in **the same micro-batches**, with the same kernels, and refuses to continue unless `max |logp − old_logp| ≤ 1e-6` (`training.py:119-123`; in the CPU simulation of this session `init_diff` was exactly 0.0, ratio range [1.0, 1.0] **[code]**). Hence ρ_{i,t} = 1 for every token.

### 4.2 The clip is inert, in value and in gradient

At ρ = 1: clip(1, 0.8, 1.2) = 1, so both arguments of `min` equal A_i. Value: s = A_i. Gradient: PyTorch's `minimum` backward sends half of the incoming gradient to each argument when they are equal, and `clamp` passes its gradient through unchanged when the input is strictly inside [0.8, 1.2]. Both halves are A_i·∂ρ/∂θ, so

$$
\nabla_\theta s_{i,t}\big|_{\theta=\theta_\text{old}} = A_i\,\nabla_\theta \rho_{i,t} = A_i\,\nabla_\theta\log\pi_\theta(o_{i,t}\mid\cdot)
$$

(using ∇ρ = ρ ∇log π = ∇log π at ρ = 1). Therefore

$$
\boxed{\nabla_\theta\mathcal L = -\frac1N\sum_{i=1}^{N}\frac{A_i}{|o_i|}\sum_{t=1}^{|o_i|}\nabla_\theta\log\pi_\theta(o_{i,t}\mid q_{b(i)}, o_{i,<t})}
$$

This is **length-normalized REINFORCE with a group-mean baseline, scaled by 1/σ_b**. A scratch check in this session fed random log-probs, a ragged mask and arbitrary advantages through the repo's `grpo_loss` and compared its autograd gradient with this formula: max difference 3.7e-9 (fp32 rounding) **[code]**.

### 4.3 Implementing a clip vs. exercising it

The code *implements* a clipped objective (and H03/H05 lab fixtures run 3–4 AdamW iterations with a learning rate at which clipping binds, so the clipping branch is *tested*, `qwen_grpo/execution/fixtures.py` docstring). The final run never *exercises* it: there is no second iteration in which ρ ≠ 1. Consequences:
- Changing `clip_eps` from 0.2 to 0.1 or 0.3 would leave the trajectory unchanged (bitwise, by the argument above: `clamp(1.0)` is 1.0 for any ε in (0,1)) **[code reasoning; not run]**.
- `clip_stats` is computed on the log-probs *before* the step (`training.py:139`), so the logged ratio is always 1 and the clip fraction always 0 at iteration 0. The console line hides them when `num_iterations == 1` (`trainer.py:759-760`). The comment "ratios at the updated parameters" only applies to iterations > 0.
- The "trust region" that PPO's clip provides is absent. What limits the step size is AdamW's per-element normalization and the learning rate (§7).

### 4.4 Is the update "on-policy"?

Mathematically yes: the tokens were sampled from π_θold and the update is evaluated at θ_old. Numerically, the sampler's probabilities (KV-cached decode or DFlash block verification in bf16) and the teacher-forced `old_logp` are different computations of the same quantity and can differ in the last bits; the ratio uses `old_logp`, so it cannot correct for that gap, and with one iteration it would not matter for the gradient anyway ([02-grpo-complete-step.md](02-grpo-complete-step.md) stage 9).

## 5. KL β = 0

GRPO normally adds −β·KL(π_θ ‖ π_ref) per token, with π_ref a frozen reference policy. Here β = 0 is hard-coded: `grpo_loss` has no KL term, no reference forward is run, and the trainer records `"kl_beta": 0.0` (`trainer.py:574`).

What that buys: no second policy evaluation (with L2L, a reference forward would cost five more 36-layer sweeps per step; because the reference is the base model, `model.disable_adapter()` would make it possible without extra weights).

What it costs: nothing in the loss pulls the policy back toward the base model. With an inert clip (§4.2), the only brakes are the learning rate, Adam's normalization, LoRA's low rank on two layers, and the 60-step budget. The diagnostic `approx_kl` = mean((ρ−1) − log ρ) (k3 estimator, `grpo.py:84-85`) is computed but is identically 0 at iteration 0, so the logs contain **no** measure of drift from the base policy. Drift is visible only through `change_from_init`, `effective_adapter_norm` (`trainer.py:742-743`), and the validation-subset predictions every 20 steps.

## 6. Sequence-length normalization

Each sequence's surrogate is divided by its own |o_i|. For this task:

| response | tokens counted | weight per token |
|---|---|---|
| `POS ITIVE <|im_end|>` / `NEG ATIVE <|im_end|>` | 3 | A/3 |
| `Positive <|im_end|>` | 2 | A/2 |
| truncated, no EOS | 4 | A/4 |

So within one group, the single-token-label variant gets 1.5× the per-token weight of the canonical spelling. Because the decision sits almost entirely in the first token (§1), this effectively scales the gradient on the first-token logit by 1/|o_i|. Known critique: per-sequence length normalization biases GRPO with respect to length (Liu et al. 2025, "Understanding R1-Zero-Like Training", arXiv 2503.20783; an external reference, not cited in the repo). With responses of 2–4 tokens, the effect here is a bounded ×1.5–2 reweighting, not a length-exploitation risk.

## 7. Micro-batch scaling and gradient accumulation

Chunk c holds n_c = 6 rows. `grpo_loss` on the chunk returns −(1/n_c)Σ_{i∈c}J_i. The code multiplies by n_c/N (`training.py:125-126`):

$$
\sum_{c=1}^{5}\frac{n_c}{N}\Big(-\frac{1}{n_c}\sum_{i\in c}J_i\Big) = -\frac1N\sum_{i=1}^{N}J_i = \mathcal L.
$$

Backward is linear, and autograd accumulates each chunk's gradient into the same `.grad` tensors, so after the fifth `backward()` the `.grad` holds ∇𝓛 for the whole batch. The only difference from one 30-row backward is floating-point summation order (fp32 accumulation of five partial gradients). Equivalence is tested against a resident run with the **same** chunking, bitwise, for micro-batch sizes {all, 1, 2, 4} (H05) **[test]**, and `tests/test_multistep.py` covers accumulation with one AdamW step per batch **[test]**.

**Normalization by N = 30, including uniform groups.** A uniform group contributes J_i = 0 and a zero gradient, but it still sits in the denominator N. With m mixed groups, the gradient is (m/5) × the average over mixed groups. Fewer mixed groups → smaller gradient. Whether that matters depends on Adam (§8): mostly it does not, because Adam is nearly scale-invariant.

## 8. AdamW, including its first step

With g_t = ∇𝓛 (fp32, per element), β₁ = 0.9, β₂ = 0.999, ε = 1e-8, weight decay 0, lr η = 2e-4 (`trainer.py:546`):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\quad
\theta_{t} = \theta_{t-1} - \eta\,\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon}.
$$

**Step 1.** m̂ = g, v̂ = g², so Δθ = −η·g/(|g| + ε) ≈ −η·sign(g) for any element with |g| ≫ 1e-8. Every LoRA element with a meaningful gradient moves by almost exactly 2e-4, whatever the gradient's size. The 1/σ advantage scaling, the 6/30 chunk scaling, and the m/5 dilution above all cancel out of the step size.

**Step 1 for A.** The forward is y = Wx + (α/r)·B(Ax) with B = 0 at initialization, so ∂𝓛/∂A = (α/r)·Bᵀ(∂𝓛/∂y)·xᵀ = 0 exactly. AdamW with g = 0 gives m = v = 0 and Δ = 0/(0 + ε) = 0: **A does not move on step 1; only B does** ([06-checkpointed-backward.md](06-checkpointed-backward.md) for the LoRA gradients).

**Later steps.** Per element, the step is η·m̂/√v̂: the sign-consistent part of recent gradients, normalized by their RMS. With one noisy 30-sample gradient per step, early steps stay close to ±η per element; the effective adapter change is the low-rank product (α/r)·B·A with α/r = 2.

**What the logs show.** `grad_norm`, `grad_norm_A/B`, `update_norm`, `change_from_init`, `effective_adapter_norm` per step (`trainer.py:735-746`).

## 9. Noise, binary rewards and zero signal

**Probability that a group carries signal.** If the policy answers a prompt correctly with probability p, then P(mixed group) = 1 − p⁶ − (1−p)⁶:

| p | P(mixed) | P(all 5 groups uniform), if every prompt had this p |
|---|---|---|
| 0.50 | 0.969 | ≈ 3e-8 |
| 0.75 | 0.822 | ≈ 1.8e-4 |
| 0.90 | 0.469 | ≈ 0.042 |
| 0.95 | 0.265 | ≈ 0.21 |

As the policy improves on a prompt, that prompt stops producing gradient. Measured on Qwen3-0.6B (lr 5e-4, B = 5, G = 6): 42 of 142 attempts were zero-signal and skipped **[measured]** (`runs/qwen/report.md`), consistent with many prompts already near p ≈ 0.9+.

**Variance.** Each update is estimated from at most 30 binary outcomes, of which uniform groups contribute nothing. The REINFORCE estimator of §4.2 has high variance per step; nothing in this configuration (single iteration, β = 0, no value baseline beyond the group mean) reduces it.

**Class-prediction bias.** The advantages within one group are label-symmetric, but the *parameters* are shared across all prompts: a LoRA direction that raises log π(POS) on every input is rewarded in POSITIVE-labelled groups and penalized in NEGATIVE-labelled ones. If the batches happen to contain more strongly-signalled POSITIVE groups for a while (no per-batch class balance is enforced, `training.py:35`), sign-like Adam steps can move the shared bias quickly. Measured on Qwen3-0.6B at lr 5e-4: the validation-subset predicted-POSITIVE rate went 0.47 (base) → 0.36 … 0.60 … 0.76 … 0.37 … 0.42 (step 90) → **0.96 (step 100)**, accuracy 0.80 → 0.54 in the last ten steps **[measured]** (`runs/qwen/report.md` §6). That instability is why the final config lowered lr to 2e-4. Whether 2e-4 avoids it is **untested at any scale**.

## 10. Summary: what each knob does mathematically here

| knob | effect on ∇𝓛 / on the update in this configuration |
|---|---|
| `clip_eps` | none (ρ ≡ 1) |
| KL β | none (absent) |
| `num_iterations` | 1: makes the two lines above true; > 1 would reuse the batch with ρ ≠ 1 and activate the clip |
| `micro_batch_size` | none mathematically (rounding order only) |
| `group_size` G | baseline quality and P(mixed group); larger G → more mixed groups, finer advantages |
| `batch_size` B | number of prompts per gradient; N = B·G normalizer |
| `learning_rate` | ≈ per-element step size (Adam) |
| `max_new_tokens` | which responses can complete; a 4-token cap allows `POS ITIVE <|im_end|>` with one token to spare |

## Open questions / unverified

- Whether lr 2e-4 avoids the label collapse seen at 5e-4 on 0.6B: needs the 4B final run (or a 0.6B run at 2e-4) with the predicted-POSITIVE rate tracked per checkpoint.
- The size of policy drift over 60 steps in KL terms: not logged; could be measured offline by scoring a fixed set of rollouts with `model.disable_adapter()` vs. the adapter (both are available through `backend.logprobs`).
- Whether the sampler-vs-scorer bf16 gap is negligible at 4B: compare the generator's step logits with `old_logp`.
