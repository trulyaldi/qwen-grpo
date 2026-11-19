# Results analysis — final Qwen3-4B GRPO + LoRA runs (audit, 2026-11-19)

Every number below either is read directly from an artifact (cited) or is derived with the formula shown.
Derived values were computed by `research/final_audit/analyze.py` (read-only; output in
`research/final_audit/analysis_output.json`). **INFERENCE** marks interpretation, not measurement.

This document describes the **150-update run `qwen3_4b_final_plain_150_v2`** (2026-11-19). It replaces the analysis of the
earlier 60-update run (`20261118T171404Z`), whose files were removed from the repo (archived in
`~/soup-assignment-results.tar.gz`). Numbers from that earlier run appear only in §8, labelled as such.

Artifacts:
- trained run: `runs/qwen3_4b_final_plain_150_v2/` (`training/training.json`, `training/training.log`, `training/memory/20261119T054525Z.json`)
- LR=0 control: `runs/qwen3_4b_final_plain_150_v2_lr0/` (same layout, memory file `20261119T064535Z.json`)
- test evaluation: `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json` ("t13")
- per-phase profile: `assignment_results/manual/profile_20261119T054235Z/results/t11_profile_manual.json` ("t11m")
- guarded stage records (external nvidia-smi sampling, 200 ms): `assignment_results/manual/<stage>_<timestamp>/guarded_run.json`
- dataset: `data/sentiment.json` (sha256 `c85d1cbc…7db72`, identical in run provenance and t13)

Both training runs and t13 were produced at commit `29852d1` with a clean tracked tree, torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, Qwen/Qwen3-4B revision `1cfa9a72…`, on an RTX 3060 12 GB (sm_86) on a remote Linux 7.0.0-38 host with 72 CPUs and 135,040,708,608 B RAM (`training.json: environment`, `t13: provenance`). This is a different host from the 2026-11-18 run (Linux 5.4, 504 GiB RAM).

The staged runner (`scripts/final150.py`, `assignment_results/final150_v2/state.json`) passed preflight, but its correctness stage **FAILED** on `pytest_cpu` (2 of 360 tests), so it blocked its own smoke and profile stages. Smoke, profile, training, control, merge and evaluation were then **run by hand, each under the guarded wrapper** (`scripts/guarded_run.py`, external nvidia-smi sampling, cap 8e9 B). See EXPERIMENT_INVENTORY.md.

---

## 1. Training (trained run)

| quantity | value | source / formula |
|---|---|---|
| successful optimizer steps | 150 | `training.json: successful_steps` |
| rollout attempts (batches) | 672 | `groups_generated` (the field name is misleading: it counts batches) |
| zero-signal batches (no update) | 522 | `len(zero_signal)` |
| prompt groups sampled | 3360 = 672 × 5 | `prompt_groups` |
| completions generated | 20160 = 3360 × 6 | `responses_generated` |
| training wall time | 3210.88 s | `train_seconds` (the guarded stage, including model load, took 3352 s) |
| updates per minute | 150 / (3210.88/60) = **2.80** | derived |
| seconds per successful update | 3210.88 / 150 = **21.41 s** | derived |
| batches by number of mixed groups | {0: 522, 1: 137, 2: 13, ≥3: 0} | `batches_by_mixed_groups` |
| groups: all-correct / all-wrong / mixed | 2905 / 292 / **163** of 3360 | derived from every attempt's `rewards` |
| unique training prompts visited | 1024 of 1024; 2336 repeat selections | `coverage`; derived |
| invalid rollout responses | 10 of 20160 (0.05%) | sum of `invalid_responses` |
| frozen base tensors changed | 0 of 398 | `frozen_changed: []` |
| final checkpoint reload | bitwise equal | `final_checkpoint.reloaded_bitwise_equal` |

### 1.1 GRPO optimisation diagnostics (150 steps)

| quantity | min | mean | max |
|---|---|---|---|
| logged loss | −3.18e-8 | −2.54e-9 | +1.59e-8 |
| grad norm (all LoRA) | 0.140 | 0.742 | 1.673 |
| per-step update norm ‖Δθ‖ | 0.0234 | 0.0499 | 0.143 |
| ratio (min / mean / max over tokens) | 1.0 | 1.0 | 1.0 |
| clip fraction, approx KL | 0 | 0 | 0 |
| ‖(α/r)BA‖ (effective adapter) | 0.162 (step 1) | 1.363 | 2.110; final 2.110 |
| ‖θ − θ₀‖ (change from init) | 0.140 | 1.358 | 1.996; final **1.9963** |

**Why the loss is ≈ 0 while gradients are not (verified from code):** with `num_iterations = 1` the ratio is exactly 1 at the update. `training.py:119-123` raises if |logp − old_logp| > 1e-6, and every step logged `init_diff = 0`. Inside the clip range the per-sequence surrogate equals Aᵢ, so L = −(1/N) Σᵢ Aᵢ (`grpo.py:55-65`). Group-normalized advantages sum to 0 within each group (`grpo.py:4-8`), so L = 0 up to fp32 rounding. The gradient is −(1/N) Σᵢ Aᵢ · (1/|oᵢ|) Σₜ ∇log π(oᵢₜ), which is nonzero whenever a mixed group exists. The logged scalar is the same detached quantity that is back-propagated (`training.py:125-131`).

**Consequence:** ratio 1.0, clip fraction 0 and approx-KL 0 are **structural, not evidence of a small policy change**. `clip_stats` is computed before `optimizer.step()`. The PPO clip never activates, so the objective is REINFORCE with a group-mean/std baseline.

**Gradient dilution (INFERENCE from code):** uniform groups stay in the N = 30 denominator (`grpo.py:65`). A step with one mixed group therefore carries 1/5 of a per-group gradient. AdamW's normalization largely cancels this scale.

### 1.2 LoRA change, by module (step 150; computed by this audit from the safetensors on CPU)

| module | ‖B₁₅₀‖ | ‖(α/r)BA‖_F | max abs |
|---|---|---|---|
| L34 q_proj | 0.933 | 1.374 | 7.14e-3 |
| L34 v_proj | 0.383 | 0.457 | 2.77e-3 |
| L34 o_proj | 0.490 | 0.705 | 4.05e-3 |
| L35 q_proj | 0.831 | 1.123 | 5.35e-3 |
| L35 v_proj | 0.359 | 0.449 | 3.33e-3 |
| L35 o_proj | 0.433 | 0.628 | 2.96e-3 |

- Combined: ‖(α/r)BA‖ = 2.110, matching `training.log`.
- step0 B = 0 exactly. Every A changed by 0.39–0.67 (Frobenius norm of A₁₅₀ − A₀).
- Trainable parameters: 1,081,344. That is 2 layers × (q 32·(2560+4096) + v 32·(2560+1024) + o 32·(4096+2560)) = 2 × 540,672, asserted at `trainer.py:580`.

### 1.3 Reward trajectory: no detectable rise

The per-step `mean_reward` printed in the log covers only batches that had a mixed group, so it is **selection-biased**: it fell from 0.857 over the first 10 steps to 0.733 over the last 10. The unbiased view uses the mean rollout reward of **every** attempt, in bins of 50 attempts:

| attempts | 1–50 | 51–100 | 101–150 | 151–200 | 201–250 | 251–300 | 301–350 | 351–400 | 401–450 | 451–500 | 501–550 | 551–600 | 601–650 | 651–672 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trained | 0.888 | 0.890 | 0.874 | 0.903 | 0.893 | 0.913 | 0.869 | 0.899 | 0.908 | 0.864 | 0.880 | 0.916 | 0.881 | 0.856 |
| LR=0 control | 0.890 | 0.881 | 0.853 | 0.891 | 0.908 (201–208) | — | — | — | — | — | — | — | — | — |

Over all responses the trained run averaged 0.8895 (20160 responses) and the control 0.8796 (6240). **The training-time reward did not measurably rise** (INFERENCE: the bins move up and down within 0.856–0.916 with no trend; treating the 250 prompt groups of a bin as the independent units, the sampling noise per bin is about ±0.02).

---

## 2. Zero-signal batches

- Fraction of zero-signal batches: 522 / 672 = **77.7%**.
- Fraction of groups that are mixed: p = 163 / 3360 = **4.85%**.
- If the 5 groups of a batch were independent, the expected zero-signal fraction is (1 − p)⁵ = (0.9515)⁵ = **0.780**. Observed: 0.777.
- Control: p = 33 / 1040 = 3.17%, so (1 − p)⁵ = 0.851. Observed: 0.856.
- Composition of the 522 zero-signal batches: 332 had all 5 groups all-correct, 156 had 4 all-correct and 1 all-wrong, 33 had 3 all-correct and 2 all-wrong, and 1 had 2 all-correct and 3 all-wrong.

**Interpretation (INFERENCE):**
- This is **expected behaviour, not a defect**.
  - Base accuracy on these prompts is about 0.89.
  - Temperature-1 samples from a confident 4B model are nearly deterministic per prompt.
  - With G = 6, a group is mixed only for the roughly 5% of prompts where the model is genuinely uncertain.
- Running 150 instead of 60 updates did not change this: the zero-signal fraction stayed at about 78%.
- Skipping is correct: a uniform group has advantage exactly 0 (`grpo.py:8`, ε = 1e-4), and the trainer skips before any log-prob pass (`training.py:90-91`, `trainer.py:775-786`).
- **Efficiency cost:** 150 updates were driven by only 163 informative groups, about 978 completions of 20160. About 55% of wall time went outside the timed successful steps, mostly to batches that produced no update (§6).
- This is an optimization opportunity:
  - harder or filtered prompts;
  - dynamic resampling until mixed (DAPO-style);
  - a larger G;
  - dropping uniform groups from the backward pass.
- It is not a correctness problem.
- The 292 all-wrong groups are prompts the model confidently mislabels. GRPO cannot learn from those without a mixed sample.

---

## 3. Validation subset during training (n = 128, the same 128 ids at each step; evaluated every 10 steps)

| step | accuracy | macro-F1 | recall POS | recall NEG | pred POS rate |
|---|---|---|---|---|---|
| 0 | 0.8594 | 0.8591 | 0.8125 | 0.9062 | 0.453 |
| 20 | 0.8750 | 0.8749 | 0.8438 | 0.9062 | 0.469 |
| 30 | 0.8906 | 0.8904 | 0.9375 | 0.8438 | 0.547 |
| 50 | 0.9062 | 0.9062 | 0.9219 | 0.8906 | 0.516 |
| 100 | 0.8828 | 0.8827 | 0.8594 | 0.9062 | 0.477 |
| 150 | 0.8984 | 0.8984 | 0.9062 | 0.8906 | 0.508 |

- From step 30 to step 150 the accuracy stayed between 0.8828 and 0.9062 (`training.json: evaluations`). The predicted-POSITIVE rate stayed between 0.453 and 0.547.
- LR=0 control: 0.8594 at steps 0, 10, 20 and 30, with identical predictions (`control_log.subset_predictions_unchanged: true`).
- Step 0 vs 150, paired: 6 examples went wrong → right and 1 went right → wrong. **Exact McNemar p = 0.125**.
- Step 140 vs 150: 3 predictions changed.
- The binomial SE at p ≈ 0.89, n = 128 is sqrt(0.89·0.11/128) = 0.028. The movements after step 30 are therefore **uninformative**: n = 128 cannot separate "diminishing returns" from noise.
- The subset was used only for monitoring. Checkpoint step 150 = `max_steps`, so no checkpoint was selected on validation.

---

## 4. Held-out test evaluation (t13): independently reproduced

The test split has n = 1066, 533 POSITIVE and 533 NEGATIVE. All four models used the same prompts (`scripts/final_eval.py:122`), the same template (`sentiment.build_prompts`, `enable_thinking=False`), greedy decoding, `max_new_tokens = 4`, and the `l2l_async` backend (`t13: decoding, backends`). The evaluation ran as its own guarded process (`assignment_results/manual/eval_20261119T070439Z/`, command in `guarded_run.json: argv`, logs `stdout.log` / `stderr.log`).

| model | checkpoint | accuracy | macro-F1 | AUC | invalid | pred POS | recall POS / NEG |
|---|---|---|---|---|---|---|---|
| base | adapter disabled | 0.87711 | 0.87874 | 0.95109 | 5 | 0.436 | 0.816 / 0.938 |
| trained | `runs/qwen3_4b_final_plain_150_v2/checkpoints/step150` (sha256 `f9fe5339…697e`) | 0.88649 | 0.88690 | 0.95228 | 1 | 0.489 | 0.876 / 0.897 |
| merged | merged from step150 by `scripts/merge.py` in its own guarded process (`assignment_results/manual/merge_20261119T070347Z/`) | 0.88649 | 0.88690 | 0.95229 | 1 | 0.489 | 0.876 / 0.897 |
| LR=0 control | `…_150_v2_lr0/checkpoints/step30` (sha256 `2fc09e61…d2b3`) | 0.87711 | 0.87874 | 0.95109 | 5 | 0.436 | 0.816 / 0.938 |

I recomputed accuracy, macro-F1 and AUC (Mann-Whitney with tie correction) from `predictions`, `label_scores` and the gold labels in `data/sentiment.json`. **All match the reported values to full precision** (`analysis_output.json: final evaluation`).

### 4.1 Statistical tests

| comparison | Δ accuracy | 95% CI (reported, paired bootstrap, 10k, seed 42) | 95% CI (re-implemented, 10k, seed 12345) | discordant b / c | exact McNemar p |
|---|---|---|---|---|---|
| trained − base | +0.00938 (+10 of 1066) | [−0.00375, +0.02251] | [−0.00469, +0.02345] | base-only-right 22 / trained-only-right 32 | **0.220** |
| merged − trained | 0 | [0, 0] | — | 0 / 0 | 1.0 |
| control − base | 0 | [0, 0] | — | 0 / 0 | 1.0 |

- AUC difference, trained − base: **+0.00120, class-stratified paired bootstrap CI [+0.00023, +0.00222]**. The interval excludes 0.
- AUC difference, merged − trained: +8.8e-6, CI [−2.8e-4, +3.1e-4].
- Wilson 95% CI for each single accuracy: base [0.856, 0.895], trained [0.866, 0.904].
- **Detectable effect (INFERENCE):** with 54 discordant pairs, a two-sided 5% test needs |b − c| ≳ 1.96·√54 ≈ 14.4, i.e. Δ ≳ 1.35 pp. The observed Δ is 0.94 pp (b − c = 10).

**Conclusion:**
- The accuracy and macro-F1 gains are **not statistically significant**. Both CIs include 0, and McNemar gives p = 0.22. The pre-registered H12 criterion (accuracy CI entirely above 0) is **not met**.
- The ranking gain **is** significant: the AUC CI lies entirely above 0. It is small (+0.0012).
- So learning is **small and only partly demonstrated**: ranking yes, accuracy no.

### 4.2 Did the model change? Yes, mostly as a class-bias shift, with a small ranking gain (INFERENCE from label scores)

The label score is s(x) = log P("POSITIVE"⟨im_end⟩ | x) − log P("NEGATIVE"⟨im_end⟩ | x), teacher-forced (`evaluation.py:143-160`).

| quantity | value |
|---|---|
| mean signed shift s_trained − s_base | **+6.01 nats** (median +5.75, SD 0.88) |
| same, POSITIVE-gold examples | +6.04 |
| same, NEGATIVE-gold examples | +5.99 |
| Spearman rank correlation of s_base vs s_trained | **0.9985** |
| median \|s_trained\| (median \|s_base\|) | 12.9 (15.8) |
| predictions changed (raw response string) | 56 of 1066 (5.3%) |

The adapter moved every example's score by about +6 nats toward POSITIVE while almost perfectly preserving the ranking. The predicted-POSITIVE rate rose from 0.436 to 0.489.

The base model was NEGATIVE-biased (0.436 vs a true rate of 0.5). Most of the learned change looks like **re-calibrating a decision threshold** (a class-prior shift). On top of that there is a small but statistically significant improvement in ranking (AUC +0.0012). Both are consistent with the small, non-significant accuracy gain.

The shift was about the same on both gold classes. It lowered recall NEG from 0.938 to 0.897 while raising recall POS from 0.816 to 0.876, so the two recalls are now closer together.

### 4.3 Merge fidelity

- Predictions agree on **1066/1066** (`merged_vs_trained_prediction_agreement = 1.0`). Accuracy and F1 are identical.
- Label scores are **not** identical:

  | statistic of \|s_merged − s_trained\| | value |
  |---|---|
  | median | 9.5e-7 |
  | p90 | 0.25 |
  | p99 | 0.25 |
  | max | 0.375 nats |
  | exactly equal | 400 of 1066 |

- One score changed sign: example 159 went from +0.125 to 0.0, yet its greedy prediction stayed POSITIVE.
- For scale: the adapter's own effect is 6.01 nats on average. The largest merge difference is about 6% of it and the mean difference (0.065) about 1%.
- **INFERENCE:** the noise is consistent with bf16 rounding of W₀ + (α/r)BA in the merged weights versus the fp32 LoRA branch on a bf16 base. At |s| ≈ 13 a bf16 ulp is 0.0625, so 0.375 nats is 6 ulps after 36 layers.
- `merge.all_within_half_ulp = true` (6 adapted tensors, max residual 8.8e-4) checks only that each merged bf16 tensor is a correct rounding of its own fp32 value (`export/merge.py:131-138`). It is **not** a model-level comparison.
- No logit tolerance was declared in advance for the 4B merge.
- The merge ran on the CPU in its own process with no CUDA context (`t13: merged_dir.provenance.device`; 9.4 s; max RSS 8,602,054,656 B). The merged weights were **not kept**: `assignment_results/final150_v2/merged/` is empty. The shard sha256 values are recorded in `t13: merged_dir.provenance.shards_sha256`, and the adapter sha256 in `t13: identity`.

### 4.4 Negative control (LR = 0)

- With lr = 0 and B initialized to 0, the adapter stays exactly at its initialization (`params_equal_init: true`). The 0-, 10-, 20- and 30-step control checkpoints are byte-identical to the trained run's step 0 (adapter sha256 `2fc09e61…` for all five files; checked by this audit).
- So the control model equals the base model **by construction**.
- Its test predictions and label scores are identical to base (max \|s_control − s_base\| = 0.0).
- What this establishes:
  - the evaluation pipeline is deterministic;
  - the adapter path adds nothing when B = 0.
- What it does **not** establish: that the trained model's change came from the reward signal. A shuffled-reward control would test that.
- The control ran 30 steps, not 150 (`configs/experiments/qwen3_4b_final_plain_150_v2_lr0.yaml`; the preflight checked `training.max_steps = 30`). With lr = 0 the step count cannot change the evaluated model. It does, however, mean the control is not "the same run".

### 4.5 Data separation

- Exact overlap: 0 for train∩test, train∩validation and validation∩test (computed here).
- Normalized-text overlap: 0 (computed by the earlier sub-audit on the same data file; its sha256 is unchanged).
- `make_splits` claims test first, then validation, then train, removing near-duplicates (`sentiment.py:70-86`).
- No within-split duplicates.
- t13's own check `test_texts_in_train = 0` covers exact matches against train only.
- Not checked: whether Qwen3-4B's pretraining saw Rotten Tomatoes. That would inflate base accuracy equally for all four models, not the training effect.

---

## 5. GPU memory

The assignment cap is "8 GB … as shown by nvidia-smi (CUDA context included)". Taken literally, 8 GB = **8,000,000,000 B** (7629.4 MiB). This run used exactly that number everywhere: the in-process guard (`memory.budget_bytes = 8000000000`, `budget_source: memory.budget_bytes`), the allocator cap and the external guarded wrapper (`cap_bytes = 8000000000`).

### 5.1 Training runs: two independent nvidia-smi monitors

In-process samples (`qwen_grpo/execution/profiler.py`, 0.5 s):

| quantity | trained | LR=0 control |
|---|---|---|
| samples | 5947 | 1654 |
| span | 3341.6 s | 923.8 s |
| median / max gap between samples | 0.561 s / 1.39 s | 0.557 s / 1.13 s |
| samples missing a numeric per-process reading | 0 | 0 |
| first sample before first log line | 46.1 s (covers model load) | 40.3 s |
| inter-step windows containing ≥ 1 sample | 149 / 149 | 29 / 29 |
| **peak per-process (nvidia-smi `used_memory`, includes the CUDA context)** | **2,864,709,632 B** (2732 MiB) | 2,864,709,632 B |
| peak device `memory.used` (all processes) | 2,898,395,136 B (2764 MiB) | 2,898,395,136 B |
| peak PyTorch reserved | 2,690,646,016 B | 2,690,646,016 B |
| peak PyTorch allocated | 1,806,041,088 B | 1,792,128,512 B |
| per-process − reserved at the peaks | 174,063,616 B (context plus non-allocator memory) | same |
| peak / 8e9 | **35.8%** | 35.8% |
| headroom to 8e9 (device peak) | 5,101,604,864 B | same |
| guard violations | 0 | 0 |
| guard limit actually used | **8,000,000,000 B** (`memory.budget_bytes`, `trainer.py:429-436`) | same |
| allocator cap | `set_per_process_memory_fraction(8e9 / 12,490,440,704 B = 0.6405)` (`trainer.py:433-435`) | same |
| phases labelled in telemetry | none (`phases: 0`) | none |

External guarded wrapper (`assignment_results/manual/train_20261119T054519Z/guarded_run.json`, `…/control_20261119T064530Z/guarded_run.json`, 200 ms):

| quantity | trained | LR=0 control |
|---|---|---|
| attributed numeric samples | 16,643 | 4,611 |
| max gap between samples | 0.332 s | 0.208 s |
| process peak / device peak | 2732 MiB / 2765 MiB | 2732 MiB / 2765 MiB |
| exit / violation | 0 / none | 0 / none |

- The two monitors agree on the per-process peak (2732 MiB).
- The peak was first reached at 06:00:53Z, just after the step-40 validation pass (around step 41–42). It then stayed on a plateau of 4298 in-process samples, because the caching allocator does not release reserved memory.
- **INFERENCE about missed transients:** both samplers see only instants (0.2 s and 0.56 s apart). But a short-lived PyTorch allocation that exceeds the reserved pool would grow `reserved`, and with no `empty_cache` that growth persists into later samples. The process could therefore exceed the sampled peak only through short-lived non-PyTorch allocations. Nothing indicates any, but this is not proven.
- **Guard enforcement not demonstrated on this host.** In the staged runner's `pytest_cpu`, the two tests that check the guard actually *stops* a run failed on the Soup host: `tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run` (DID NOT RAISE SystemExit) and `tests/test_profiler.py::test_the_guard_records_a_violation_and_interrupts_the_main_thread` (DID NOT RAISE KeyboardInterrupt) (`assignment_results/final150_v2/correctness/20261119T050522Z/pytest_cpu.xml`). Both pass locally at the same code (re-run by this audit on the laptop, CPU, 2026-11-19: 2 passed). The real runs never needed the guard to act: 0 violations and a peak of about 36% of the cap.

### 5.2 Other phases

| phase | nvidia-smi evidence | value |
|---|---|---|
| model load / init | trained-run telemetry, first 46.1 s | ≤ 1,342,177,280 B |
| rollout, old log-probs, forward, backward, optimizer (150 steps) | trained-run telemetry, unlabelled | ≤ 2,864,709,632 B |
| per-phase breakdown (2 steps) | t11m (in-process, 250 ms, labelled; 334 samples, none missing) | setup 1,300,234,240; rollout 1,960,837,120; old_logp 1,962,934,272; forward 1,962,934,272; backward 1,962,934,272 (**only 2 of 10 backward windows got a sample**); **optimizer: 0 samples** (phases of 0.002–0.06 s) |
| evaluation inside the profile (32 examples, adapter model) | t11m | 2,208,301,056 B (profile peak; 2106 MiB in the guarded record) |
| merge inside the profile | t11m: merge in a process that already holds the model on the GPU | process 1,962,934,272 B (that process's resident footprint, not the merge's); CPU RSS peak 17.5 GB |
| **merge of the evaluated step150 adapter** | `assignment_results/manual/merge_20261119T070347Z/guarded_run.json` | **no GPU process at all** (0 compute-app rows over 159 device samples); device 13 MiB; 32 s; max RSS 8,602,054,656 B |
| **evaluation of base / trained / merged / control (t13)** | `assignment_results/manual/eval_20261119T070439Z/guarded_run.json` (7295 numeric samples, 200 ms) | **process peak 1912 MiB, device 1945 MiB, 0 violations**. Allocator per-phase peaks (`t13: phases`): 1,567,870,464 B (trained, base, control), 1,563,545,088 B (merged) |
| smoke (4 steps) / kill-resume A, B before kill, B after resume | guarded records | 2362 MiB / 2364, 2220, 2126 MiB |

- The t13 top-level field `peak_gpu_allocated_bytes_eval = 1,195,164,160 B` is **not** a peak over the evaluation. It is read after the base pass, right after `run_model` reset the peak counter (`scripts/final_eval.py:149`, read at `:158-159`), so it is close to the memory allocated at that instant (INFERENCE from code). Use the per-phase values or, better, the nvidia-smi figures.

**Verdict:**
- Every phase stayed within 8 GB, shown by nvidia-smi per-process samples that include the CUDA context: training, rollouts, the control, the merge (no GPU use) and the **evaluation of the merged model** (1912 MiB).
- The merge code never touches the GPU (`export/merge.py:91-176`), and the guarded record confirms no CUDA context was created.
- The earlier gap "merged-model evaluation not measured" is **closed**.

---

## 6. Throughput and compute

### 6.1 Measured per successful step (150 timed steps; zero-signal batches have separate records in `zero_signal_timings`)

| component | total s | share of timed step | mean per step |
|---|---|---|---|
| rollout generation | 336.72 | **23.5%** | 2.24 s |
| old log-prob pass | 527.26 | **36.8%** | 3.52 s |
| update (forward + backward + AdamW) | 564.75 | **39.4%** | 3.76 s |
| total timed step | 1431.77 | 100% | **9.55 s** (median 9.49 s) |

- Rollout generation throughput on successful batches: 13,510 tokens / 336.72 s = **40.1 tokens/s**. That is 30 sequences decoded together, about 3 tokens each.
- Time outside timed steps: 3210.88 − 1431.77 = **1779.1 s (55%)**.
  - The 522 zero-signal attempts recorded 1174.4 s (`zero_signal_timings.attempt_s`; mean rollout 2.24 s).
  - The validation-subset passes at steps 10–150 took 594.9 s (`evaluation_times`). INFERENCE: the step-0 pass (42.4 s) appears to fall before the training clock starts, because adding it would exceed 1779 s.
  - The remaining ~10 s covers checkpoint writes and bookkeeping (INFERENCE).
- The LR=0 control's mean timed step was 9.64 s.

### 6.2 Weight transfer

- Bytes per full sweep of the 36 streamed layers: **7,267,018,752 B** (`t11m: streamed_bytes_per_pass`; 36 × 201,861,632 B).
- `weight_bytes_copied` per step is either 94,471,243,776 B (**13 sweeps**, 143 of 150 steps) or 101,738,262,528 B (**14 sweeps**, 7 steps). The difference is exactly one sweep (`training.json: timings`).
- Sweeps per step, from the code (`trainer.py`, `training.py`, `generator.py`):
  - **rollout:** 3 or 4 (prefill plus one decode sweep per further token). Every completion is about 3 tokens.
  - **old log-probs:** 5 (one sweep per micro-batch of 6; 30/6 = 5).
  - **training forward:** 5.
  - **backward:** recomputes layers 34–35 only. Those copies are not in this counter. The prediction for them is 2,018,616,320 B.
- Total bytes over the 150 successful steps: 1.4222e13 B.
- Pinned H2D bandwidth has two measurements that disagree:
  - **12,229,887,571 B/s**, median one-slab copy, measured on 2026-11-18 on the earlier host and used by `results/budget/residuals/qwen3-4b-final_150_v2.md`;
  - **7,253,086,144 B/s**, measured by this run's manual profile (`t11m: h2d_bandwidth_bytes_per_s`).
- Effective rate over step time: 1.4222e13 B / 1431.8 s = **9.93e9 B/s**. That is 81% of 12.23 GB/s, but 137% of 7.25 GB/s.
- **The 7.25 GB/s figure is inconsistent with the training run.** 14 sweeps at 7.25 GB/s would take 14.03 s, longer than the measured 9.55 s mean step. Within the same profile, the old-log-prob phase moved 72.67e9 B in 7.15 s (`t11m: transfers.old_logp`), i.e. about 10.2 GB/s. INFERENCE: the profile's one-slab bandwidth sample was noisy or contended. This audit does not claim which figure is right beyond that.
- Transfer-only lower bound at 12.23 GB/s: 7.72 s for a 13-sweep step and 8.32 s for a 14-sweep step. The measured mean step was 9.55 s, about 1.2× the bound.

**INFERENCE:**
- The GRPO step is still **dominated by weight transfer**: on average it moves weights at 81% of the earlier measured copy bandwidth.
- The steps are about 1 s slower than in the 60-update run (9.55 s vs 8.57 s). The residuals analysis of that run found the step at 1.01× the transfer bound (8.54 s vs 8.48 s). That close match does not reproduce here. The run was on a different host, which may explain part of the difference; this was not tested.
- **Overlap was never measured**: no sync-vs-async run and no CUDA event timeline exist, and H06 is untested. The earlier `comparisons/backend_performance.csv` (header-only) is no longer in the repo.
- No speed-up claim is supported.

**Why the step splits this way:**
- Rollouts are cheap because completions are 3–4 tokens: 3–4 sweeps.
- Old log-probs and the update each cost 5 sweeps because the micro-batch loop sits **outside** the layer loop, so each 6-row micro-batch re-streams all 36 layers.

**Optimization ideas (INFERENCE):**
- A layer-major micro-batch loop would cut those 10 sweeps to 2.
- Keeping 20 decoder layers resident is what the committed calculator recommends (`results/budget/predictions/qwen3-4b-final_8gb.md`, "Layer residency"): 6.56 GiB device predicted, below the cap. It would cut bytes per sweep from 7.27 GB to 3.23 GB.
- The configured run used 0 resident layers. The report must justify that choice.

### 6.3 Prediction vs measurement (`results/budget/predictions/qwen3-4b-final_8gb.md`, committed in f03eb9c before any GPU run)

| quantity | predicted | measured (150-update run) | gap |
|---|---|---|---|
| bytes per GRPO step | 103.757 GB (incl. 2.019 GB backward; 4 rollout sweeps) | 94.471 GB (143 steps) or 101.738 GB (7 steps); counter excludes backward | matches once backward and the shorter rollouts are excluded |
| transfer time per step | ≥ 8.65 s at an assumed 12.0 GB/s | step 9.55 s; transfer-only 7.72–8.32 s at 12.23 GB/s | consistent with a transfer-dominated step |
| rollout device peak | 2.425 GiB (2,604,126,643 B) | training-run process peak 2,864,709,632 B (2.668 GiB) | under-predicted by 10% |
| 150-update forecast (`results/budget/residuals/qwen3-4b-final_150_v2.md`, calibrated on the 60-update acceptance 0.226) | 665 attempts, 0.83 h | 672 attempts, 3211 s = 0.89 h | within 1% on attempts, 7% on time |

**Measurement defect, fixed:** the 2026-11-18 profile counted only non-prefetched loads, so its transfer table showed 0 bytes for most phases. Commit 5680e71 fixed this ("executor_bytes counts async prefetches"; `scripts/profile_run.py:54`, `trainer.py:501-507`). t11m now counts every phase. Its `transfer_share_of_wall` exceeds 1 for old_logp (1.40), forward (1.38) and evaluation (1.19). That is impossible if 7.25 GB/s were the real rate, which is another sign the profile's bandwidth figure is too low.

---

## 7. Summary of what the numbers support

| claim | supported? |
|---|---|
| 150 GRPO updates on Qwen3-4B with streamed weights, 0 frozen tensors changed | yes |
| peak nvidia-smi per-process memory during training 2.86e9 B (35.8% of 8e9) | yes, sampled at 0.56 s in-process and 0.2 s externally |
| merged-model evaluation within 8 GB | **yes**: 1912 MiB per process, 0 violations (guarded eval stage) |
| the guard would have stopped an over-budget run on this host | **not demonstrated**: the two enforcement unit tests failed on the Soup host (they pass locally) |
| adapter changed the model | yes: ‖Δθ‖ 2.00, mean label-score shift +6.0 nats, 56 test predictions changed |
| held-out accuracy improved significantly | **no**: +0.94 pp, CI [−0.38, +2.25], McNemar p = 0.22 |
| learned better discrimination | **yes, slightly**: AUC +0.0012, CI [+0.0002, +0.0022]; but rank correlation 0.9985, so most of the change is a bias shift (INFERENCE) |
| merged model matches the adapter | predictions yes (1066/1066); scores within 0.375 nats, no pre-declared tolerance |
| the step is transfer-dominated | suggested (INFERENCE): 81% of the earlier measured copy bandwidth; the new profile's bandwidth figure is inconsistent |
| async prefetch gives a speed-up | **not tested** |

---

## 8. Comparison with the earlier 60-update run (2026-11-18, `20261118T171404Z`; for comparison only)

| | 60 updates (earlier run) | 150 updates (this run) |
|---|---|---|
| test accuracy Δ vs base | +0.375 pp [−0.563, +1.313] | +0.938 pp [−0.375, +2.251] |
| discordant b / c, McNemar p | 12 / 16, p = 0.57 | 22 / 32, p = 0.22 |
| AUC Δ vs base | +0.0007 [−4e-8, +0.0015] | +0.0012 [+0.0002, +0.0022] |
| mean label-score shift | +3.10 nats | +6.01 nats |
| invalid test answers (of 1066) | 3 | 1 |
| ‖θ − θ₀‖ | 1.21 | 2.00 |
| zero-signal batches | 77% | 78% |
| mean timed step | 8.57 s | 9.55 s |
| guard limit | 8 GiB (8,589,934,592 B) | 8e9 B |
| merged-model eval memory | not measured | 1912 MiB (nvidia-smi) |

Sources for the 60-update column: the earlier version of this audit (its `analysis_output.json`), and the comparison table in `docs/results_20261119_150_v2.md`. The run's own files are in `~/soup-assignment-results.tar.gz`, not in the repo.
