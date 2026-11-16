# Resident reference (Ticket 00)

The resident GRPO + LoRA backend as of commit `805a7b1` is the reference that every L2L backend is compared against.
This ticket changes none of its code. It records what the backend does, which tests pin it down, and a frozen set of
fixtures with their resident outputs. The measured record is `research/l2l/baseline.json`, and the raw test log is
`research/l2l/baseline_tests.log` (JUnit: `baseline_tests.xml`).

## Behaviour

| Phase | Code | Behaviour |
|---|---|---|
| Loading | `model.lora_model` | `from_pretrained(name, dtype)` loads into CPU RAM, then `.to(device)` moves the whole model. `get_peft_model` adds LoRA A/B (fp32, `lora_dropout=0`, A by PEFT's Kaiming init under `torch.manual_seed(training.seed)`, B = 0). Base weights are bf16 and `requires_grad=False`. `embed_tokens.weight is lm_head.weight` (tied). |
| Forward | `policy.compute_logprobs` | HF `model(input_ids, attention_mask, position_ids=cumsum(mask)-1, use_cache=False, logits_to_keep=R+1)`, then `grpo.response_logprobs`: logit columns P−1..T−2, cast to fp32, full-vocabulary `log_softmax`, `gather`. `explicit_forward` reproduces this bitwise but is not wired into training. |
| Old log-probs | `training.prepare` | `compute_logprobs` under `no_grad`, once per rollout batch, in the same micro-batches as the updates (bf16 outputs depend on batch shape). |
| Backward | `training.grpo_update` | Per micro-batch: forward, `grpo_loss · n_c/N`, `.backward()` (frees that chunk's graph). Gradients accumulate in the LoRA `.grad`. Plain autograd: the graph keeps every layer's activations until its backward. Before the step: iteration 0 requires `max|logp − old_logp| ≤ 1e-6`; the loss and the gradients must be finite. |
| Optimizer | `training.grpo_update` | `AdamW(lora params, lr, weight_decay=0)`, no gradient clipping, `num_iterations` updates per rollout batch against a fixed `old_logp`. |
| Generation | `policy.generate_rollouts` | `model.generate(do_sample=True, temperature=1, top_p=1, top_k=0, num_return_sequences=G)` with HF's KV cache. Rows b·G..b·G+G−1 answer prompt b (`check_grouping`). Mask: tokens up to and including the first EOS (`grpo.response_mask`; Qwen's pad id is an EOS id). |
| Evaluation | `evaluation.evaluate`, `label_scores` | Greedy `model.generate`, chunks of 32, max 4 new tokens. Label scores are teacher-forced through `compute_logprobs`, chunks of 8. Base-model numbers come from `model.disable_adapter()`. |
| Checkpoint | `checkpointing.save_checkpoint` | `model.save_pretrained` (adapter only) + `training_state.pt` (optimizer, progress, log, RNG, pending rollout batch). Resume replays the batch stream and restores the RNG state. `load_training_state` uses `torch.load` without `map_location`. |
| Merge | `verification.merge_check`, `fp32_merge_check` | PEFT `merge_and_unload()` in place on the GPU (bf16), plus an fp32 CPU check with the whole model in fp32. The check compares weight residuals against `(α/r)·B·A` and logits against the adapter-attached model, with the adapter-vs-base distance as power. |

## Tests that pin the reference

| Property | Tests |
|---|---|
| Explicit vs HF forward, bitwise | `test_qwen3_forward`: `test_explicit_forward_equals_hf_logits`, `test_default_executor_reproduces_pre_executor_forward_and_hf`, `test_custom_executor_runs_every_layer_once_in_order`, `test_callback_sees_each_layer_output`, `test_selected_token_logprobs_equal_compute_logprobs`; padding: `test_padded_rows_match_unpadded_sequences` (fp32, 1e-5) |
| Teacher forcing / selective logits | `test_policy`: `test_left_padded_teacher_forcing_matches_generate_and_unpadded`, `test_selective_logits_match_the_full_logit_reference` |
| LoRA gradients | `test_qwen3_forward::test_lora_gradients_match_hf` (HF vs explicit vs executor, bitwise); `test_grpo::test_loss_gradient_matches_finite_differences`; `test_multistep::test_b2_opposite_labels_hand_rewards_and_independent_reference_gradient`; `test_trainer::test_one_grpo_step_decreases_the_grpo_loss_through_both_lora_factors` |
| GRPO updates | `test_multistep`: `test_first_lora_step_updates_only_B_and_leaves_base_bitwise_unchanged`, `test_one_iteration_equals_the_first_update_of_three`, `test_micro_batched_updates_equal_the_full_batch_updates`, `test_reused_batch_keeps_the_sampling_time_old_logp`, `test_frozen_parameters_not_updated`; `test_audit`: `test_end_to_end_loss_by_hand_with_padding_eos_truncation_and_clipping`, `test_lr_zero_step_leaves_parameters_and_policy_bitwise_unchanged` |
| Resume | `test_trainer`: `test_resumed_run_equals_uninterrupted_run_bitwise`, `test_resume_inside_a_reused_rollout_batch_equals_uninterrupted_run_bitwise` |
| Merge | `test_verification`: `test_merge_equivalence_fp32_passes_with_power`, `test_wrong_merge_fails_and_dropped_adapter_is_not_a_pass` |
| Frozen fixtures (new) | `test_fixtures` (below) |

Gap: no existing test runs a full optimizer update through `explicit_forward`. Its LoRA gradients are pinned bitwise,
but the update is not. Ticket 07 closes this.

## Frozen fixtures

`qwen_grpo/execution/fixtures.py` builds and `research/l2l/fixtures/{0.6b-like,4b-like}.pt` stores, for each of the two
tiny shapes of `test_qwen3_forward.py`:

- **Model**: random Qwen3 (seed 0, fp32, eager, V = 64, tied embeddings), LoRA r = 8, α = 16 on q/v/o of the last two
  layers, B ~ N(0, 0.5) so that the adapters change the output and both A and B receive gradient.
- **Rollout batch** (hand-built, not sampled): B = 2 prompts × G = 3, P = 6 with prompt lengths 6 and 3 (left padding),
  R = 4. The EOS ids are 1 and 2, and pad = 2 = an EOS id, as in Qwen3. Completion lengths are `[4, 4, 2, 1, 3, 2]`:
  a truncated row, EOS as the last token, EOS-only, mid-row EOS with a padded tail, and the second EOS id. Rewards
  `[[1,0,0],[0,1,1]]` (both groups mixed). The mask comes from `grpo.response_mask`; advantages and `old_logp` come from
  `training.prepare`.
- **Resident outputs**: HF logits `[6, 10, 64]` (asserted bitwise equal to `explicit_forward`), log-probs `[6, 4]`, the
  iteration-0 loss and all 12 LoRA gradients, then 3 `training.grpo_update` iterations (AdamW, lr 2e-2, ε = 0.2). Also
  saved: the LoRA tensors before and after, and the AdamW `state_dict`.

Measured contents (from `baseline.json`):

| shape | iteration 0 loss | LoRA grad norm min / max | clip fraction, iterations 0 / 1 / 2 | loss, iterations 1 / 2 |
|---|---|---|---|---|
| 0.6b-like | 4.0e-8 | 2.4e-3 / 0.40 | 0 / 0.062 / 0.25 | −0.034 / −0.084 |
| 4b-like | 4.0e-8 | 1.7e-2 / 0.42 | 0 / 0.125 / 0.25 | −0.051 / −0.068 |

The iteration-0 loss is −mean(A) up to rounding, about 0, because ratios are 1 and advantages sum to zero within each group.
The gradient is not zero. Clipping binds in iterations 1–2.

**Reproducibility.** `build`, `load` and `reference` run inside `fixtures.isolated()`: default dtype fp32
(`tests/test_grpo.py` switches it to float64 at import), 1 CPU thread, and a forked RNG seeded with 0. All three are
restored afterwards. Regenerating with `python -m qwen_grpo.execution.fixtures` gave byte-identical files (same SHA-256).
The tests compare with tolerance 0: inputs rebuilt from code, and resident outputs recomputed from the saved inputs. A
mismatch means code, libraries or platform changed. The fix is then a deliberate, recorded regeneration, not a wider
tolerance. The torch, transformers and peft versions are stored in each file's `meta`.

**Controlled fault and a limit of output comparisons.** Shifting one frozen layer-0 `q_proj` weight (0.0046) by 1e-6
changes logits, log-probs and gradients, and the test detects it. Shifting the same weight by one ulp (≈5e-10)
changes **no** element of `q_proj`'s output: its contribution is below the output's own rounding. So no comparison of
logits or gradients can see it. Measured: a shift of 1e-7 is already visible, and at 1e-4 the existing `INIT_TOL` guard in
`grpo_update` fires first. Consequence for Ticket 02: streamed weights must be verified on the weights themselves (digests),
not only through outputs.

**Not covered.** Rollout generation and sampling (the rollouts are hand-built; Ticket 06 needs its own reference),
bf16, sdpa, the real vocabulary size (Ticket 05), micro-batching, other LoRA placements (Ticket 04), and GPU execution.

## Baseline tests

`CUDA_VISIBLE_DEVICES="" python -m pytest -q -rA --continue-on-collection-errors --junitxml=research/l2l/baseline_tests.xml`
at `805a7b1` (tracked tree clean; untracked: `runs/default/` and `runs/grpo_clip_k3/`, which are the owner's run
outputs, and this ticket's outputs): **374 passed, 5 failed, 1 collection error**, 133 s. All 9 `test_fixtures` tests passed.
None of the failures involves code changed by Ticket 00:

| Test | Cause |
|---|---|
| `tests/test_trl_reference.py` (collection error) | imports `research/trl_reference`, deleted in `4424e0f` |
| `test_config::test_new_runs_reject_the_historical_schema_with_the_offending_keys` | `48b2028` rewrote `configs/default.yaml` in the current schema; the test still expects the historical one |
| `test_config::test_historical_configs_in_the_repository_still_load_as_records` | same commit; the test expects the old values (balanced, B = 2, 32 samples) |
| `test_config::test_current_configs_load_with_distinct_run_names` | failed before `48b2028` too (checked with the old `default.yaml`). Observed: `assert (1 >= 2)` on the config list `[configs/qwen.yaml]` |
| `test_historical::test_saved_log_and_checkpoint_configs_are_the_reference_yaml` | environment: `load_training_state`'s `torch.load` has no `map_location`, the saved state holds CUDA tensors, and CUDA is hidden. **Not run with a GPU.** |
| `test_historical::test_reference_run_resumes_under_the_current_code` | same cause. **Not run with a GPU.** |

`scripts/l2l_baseline.py` refuses to report PASS if any failure lacks a documented cause.

### Classification of the six baseline failures (after Ticket 01)

`python -m pytest -q tests/test_historical.py -rA` **with CUDA visible**: 19 passed, including both tests that failed
in the CPU baseline. Their cause is therefore confirmed: `torch.load` without `map_location` on CUDA-saved state when
CUDA is hidden. No production code was changed.

| # | Test | Class | Cause |
|---|---|---|---|
| 1 | `test_historical::test_saved_log_and_checkpoint_configs_are_the_reference_yaml` | environment | passes with CUDA visible; fails on a CPU-only run (row 17 of `architecture.md`) |
| 2 | `test_historical::test_reference_run_resumes_under_the_current_code` | environment | same |
| 3 | `test_config::test_new_runs_reject_the_historical_schema_with_the_offending_keys` | stale assumption, **real finding** | see below |
| 4 | `test_config::test_historical_configs_in_the_repository_still_load_as_records` | stale assumption | see below |
| 5 | `test_config::test_current_configs_load_with_distinct_run_names` | stale assumption (older than `48b2028`) | see below |
| 6 | `tests/test_trl_reference.py` (collection error) | obsolete | imports the deleted `research/trl_reference` |

None was caused by Ticket 00 or Ticket 01 code.

### Stale tests

No test was changed or deleted. Recommendations for the owner:

**3. `test_new_runs_reject_the_historical_schema_with_the_offending_keys`**
- *Purpose:* `config.load` must refuse historical-schema configs for new runs. The test checks this on
  `runs/qwen/config.yaml`, on `configs/default.yaml`, and on `configs/experiments/grpo_clip_k3.yaml`, which inherits from
  it.
- *Why it fails:* `48b2028` rewrote `default.yaml` in the current schema, so it and `grpo_clip_k3.yaml` now load as
  current configs.
- *Finding:* the four experiment configs that inherit from `default.yaml` without historical keys of their own
  (`grpo_32`, `grpo_clip_k3`, `grpo_low_lr`, `qwen3_4b`) now resolve to **different experiments**. They inherit rank 32,
  B = 5, 512 samples, lr 5e-4 instead of the historical rank 8, B = 2 balanced, 32 samples, lr 1e-4. `grpo_1000.yaml` is
  rejected because of its own `training.eval_every`. The test detected this. Whether the change was intended is the
  owner's decision.
- *Recommendation:* update, don't delete. Keep the `runs/qwen/config.yaml` case and the `batch_mode` /
  `legacy_batch_mode` cases unchanged. Replace the two repository files with a historical-schema YAML written in
  `tmp_path` (or with `runs/qwen/config.yaml` alone).
- *Still covering the behaviour:* `test_historical_settings_the_code_no_longer_implements_are_refused`,
  `test_historical_keys_move_to_their_current_names`, `test_historical_batch_modes_stay_readable_and_distinct_but_cannot_be_trained`,
  `test_the_saved_config_of_the_reference_run_upgrades_to_the_reference_yaml`.

**4. `test_historical_configs_in_the_repository_still_load_as_records`**
- *Purpose:* the historical configs kept in the repository still load (with `historical=True`) with their recorded
  values. Their run names stay stable. `grpo_clip_k3` differs from its parent only in `num_iterations`.
- *Why it fails:* it asserts the old `default.yaml` values (balanced, B = 2, rank 8).
- *Recommendation:* update once the owner decides what `configs/experiments/` should mean. Either restore the historical
  parent under another name (for example `configs/historical_default.yaml`) and point the experiments at it, or accept
  the rebasing and assert the new values. The `grpo_clip_k3`-differs-only-in-`num_iterations` assertion still holds and
  is worth keeping.
- *Still covering it:* the runs' own saved `config.yaml` files and `test_historical` (with CUDA) pin the recorded run
  configs.

**5. `test_current_configs_load_with_distinct_run_names`**
- *Purpose:* every current config in `configs/` loads, and no two share a `run.name`, so their outputs cannot collide.
- *Why it fails:* it excludes `default.yaml` (as historical) and requires at least 2 other configs; only `qwen.yaml`
  remains. It failed before `48b2028` too.
- *Recommendation:* update. Include `default.yaml` now that it is current. `default` and `qwen` are distinct, so the
  test would pass. Drop the `>= 2` count, which encodes how many files existed rather than a property.
- *Still covering it:* nothing else checks run-name uniqueness across files.

**6. `tests/test_trl_reference.py`**
- *Purpose:* the manual pipeline agrees with TRL's GRPO on rewards, data stream, prompts, LoRA targets and fp32
  gradients (equal after TRL's Bessel correction), plus the saved TRL evaluation.
- *Why it fails:* its subject, `research/trl_reference/`, was removed in `4424e0f`, and `trl` is no longer a dependency.
- *Recommendation:* remove it, together with the reference if that stays deleted, or restore both from
  `git show 4424e0f^:research/trl_reference/...` if an external reference is wanted again. It cannot be updated on its own.
- *Still covering the numerical part:* `test_multistep::test_b2_opposite_labels_hand_rewards_and_independent_reference_gradient`
  (an independent gradient), `test_grpo::test_loss_gradient_matches_finite_differences`, `test_audit::test_end_to_end_loss_by_hand_with_padding_eos_truncation_and_clipping`.
  The TRL-specific agreement (prompt rendering and data order identical to TRL's) is no longer covered.
