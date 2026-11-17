# Study report: H01-H12

Generated 2026-11-17T17:53:23+00:00 at commit 9ca0390 by `scripts/experiment.py report`. The recorded status is the one in `research/l2l/hypotheses.yaml` (changed only by hand, citing evidence); the entry columns are what the recorded outputs say now. A completed experiment is not a supported hypothesis.

## H01: Synchronous L2L produces the same logits as resident Qwen3 execution.

Recorded status: **SUPPORTED**; cited evidence: t03_forward_equivalence

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H01.lab_sync | t03_forward_equivalence | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 2690 comparisons within predefined thresholds | MET | none |
| H01.fault_wrong_layer *fault control* | t03_forward_equivalence_wrong_layer | PASS: fault injection: detected (status FAIL, execution PASS) | NOT_EVALUATED | none |
| H01.lab_async_cuda | t08_forward_equivalence_async | NOT TESTED: no result research/l2l/experiments/t08_forward_equivalence_async | NOT_EVALUATED | none |
| H01.fault_unsafe_reuse *fault control* | t08_forward_equivalence_async_unsafe_reuse | NOT TESTED: no result research/l2l/experiments/t08_forward_equivalence_async_unsafe_reuse | NOT_EVALUATED | none |

## H02: Checkpointed L2L backward produces the same LoRA gradients as resident autograd.

Recorded status: **SUPPORTED**; cited evidence: t04_gradient_equivalence_cpu, t04_gradient_equivalence_cuda

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H02.lab_sync_cpu | t04_gradient_equivalence_cpu | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 5296 comparisons within predefined thresholds | MET | none |
| H02.lab_sync_cuda | t04_gradient_equivalence_cuda | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 5296 comparisons within predefined thresholds | MET | none |
| H02.fault_detach_boundary *fault control* | t04_gradient_equivalence_detach_boundary | PASS: fault injection: detected (status FAIL, execution PASS) | NOT_EVALUATED | none |
| H02.lab_async_cuda | t08_gradient_equivalence_async | NOT TESTED: no result research/l2l/experiments/t08_gradient_equivalence_async | NOT_EVALUATED | none |
| H02.fault_unsafe_reuse *fault control* | t08_gradient_equivalence_async_unsafe_reuse | NOT TESTED: no result research/l2l/experiments/t08_gradient_equivalence_async_unsafe_reuse | NOT_EVALUATED | none |

## H03: Memory-efficient output-head computation preserves GRPO log-probabilities and gradients.

Recorded status: **SUPPORTED**; cited evidence: t05_logprob_equivalence_cpu, t05_logprob_equivalence_cpu_v2, t05_logprob_equivalence_raw_logit_v2

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H03.lab_v2_cpu | t05_logprob_equivalence_cpu_v2 | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 20304 comparisons within predefined thresholds | MET | none |
| H03.fault_raw_logit *fault control* | t05_logprob_equivalence_raw_logit_v2 | PASS: fault injection: detected (status FAIL, execution PASS) | NOT_EVALUATED | none |

## H04: Streamed generation follows the same target-policy distribution as resident generation.

Recorded status: **SUPPORTED**; cited evidence: t06_generation_equivalence_cpu, t06_generation_equivalence_cuda

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H04.lab_sync_cpu | t06_generation_equivalence_cpu | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 950 comparisons within predefined thresholds | MET | none |
| H04.lab_sync_cuda | t06_generation_equivalence_cuda | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 1292 comparisons within predefined thresholds | MET | none |
| H04.fault_stale_position *fault control* | t06_generation_equivalence_stale_position | PASS: fault injection: detected (status FAIL, execution PASS) | NOT_EVALUATED | none |
| H04.fault_no_padding_mask *fault control* | t06_generation_equivalence_no_padding_mask | PASS: fault injection: detected (status FAIL, execution PASS) | NOT_EVALUATED | none |
| H04.lab_async_cuda | t08_generation_equivalence_async | NOT TESTED: no result research/l2l/experiments/t08_generation_equivalence_async | NOT_EVALUATED | none |

## H05: A complete streamed GRPO + LoRA optimizer update is numerically equivalent to a resident update.

Recorded status: **SUPPORTED**; cited evidence: t07_update_equivalence_cpu, t07_update_equivalence_cpu_v2, t07_update_equivalence_cuda_v2, real_step_20261117T063325Z, real_step_20261117T063548Z, compare_qwen_resident_short_vs_qwen_l2l_short, compare_qwen_resident_vs_qwen_l2l

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H05.lab_sync_cpu | t07_update_equivalence_cpu_v2 | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 6528 comparisons within predefined thresholds | MET | none |
| H05.lab_sync_cuda | t07_update_equivalence_cuda_v2 | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 6528 comparisons within predefined thresholds | MET | none |
| H05.fault_advantage *fault control* | t07_update_equivalence_advantage | PASS: fault injection: detected (status FAIL, execution PASS) | NOT_EVALUATED | none |
| H05.lab_async_cuda | t08_update_equivalence_async | NOT TESTED: no result research/l2l/experiments/t08_update_equivalence_async | NOT_EVALUATED | none |
| H05.real_0.6b_bf16 *supplementary* | h05_real_0.6b_bf16 | NOT TESTED: not run here: ['run:qwen3_0.6b_resident', 'run:qwen3_0.6b_l2l_sync', 'run:qwen3_0.6b_l2l_async'] | NOT_EVALUATED | none |
| H05.real_0.6b_fp32_diagnostic *diagnostic* *supplementary* | h05_real_0.6b_fp32_diagnostic | NOT TESTED: not run here: ['run:kaggle_06b_fp32_resident', 'run:kaggle_06b_fp32_l2l_sync', 'run:kaggle_06b_fp32_l2l_async'] | NOT_EVALUATED | none |

## H06: Double-buffered asynchronous prefetching reduces execution time without changing numerical results.

Recorded status: **untested**; cited evidence: none

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H06.prefetch_timing | t08_prefetch_timing | NOT TESTED: no result research/l2l/experiments/t08_prefetch_timing | NOT_EVALUATED | none |

## H07: Bounded layer residency reduces GPU weight-memory dependence on total model depth.

Recorded status: **SUPPORTED**; cited evidence: t02_store_depth

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H07.store_depth_copy_only | t02_store_depth | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: streamed peak 3148800 B at every depth; resident peak [3239936, 6389760, 12689408, 25288704, 50487296] | MET | none |
| H07.depth_residency_execution | t11_depth_residency | NOT TESTED: no result research/l2l/experiments/t11_depth_residency | NOT_EVALUATED | none |

## H08: An interrupted and resumed run preserves the same committed training sequence as an uninterrupted run.

Recorded status: **SUPPORTED**; cited evidence: t09_resume_tiny_cpu

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H08.resume_tiny_cpu | t09_resume_tiny_cpu | PASS: research/l2l/experiments/t09_resume_tiny_cpu.json: pass True | MET | 20261117T175211Z_H08.resume_tiny_cpu: PASS |
| H08.resume_qwen3_0.6b | t09_resume_qwen_stream | NOT TESTED: no result research/l2l/experiments/t09_resume_qwen_stream.json | NOT_EVALUATED | none |

## H09: Tensor-by-tensor merging preserves the adapter-attached model's behaviour.

Recorded status: **untested**; cited evidence: t10_merge_equivalence_tiny

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H09.merge_tiny_cpu | t10_merge_equivalence_tiny | PASS: status PASS, execution PASS, hypothesis_result SUPPORTED: all 30 comparisons within predefined thresholds | MET | none |
| H09.merge_qwen3_0.6b | t10_merge_equivalence_qwen3_0.6b | NOT TESTED: no result research/l2l/experiments/t10_merge_equivalence_qwen3_0.6b | NOT_EVALUATED | none |

## H10: DFlash reduces target-model execution overhead while preserving the current LoRA policy's output distribution.

Recorded status: **untested**; cited evidence: none

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H10.dflash_bench | t12_dflash_bench | NOT TESTED: R11 NOT TESTED: missing t12_dflash_bench; R12 NOT TESTED:  | NOT_EVALUATED | none |

## H11: The complete Qwen3-4B system respects the 8 GB GPU-memory budget.

Recorded status: **untested**; cited evidence: none

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H11.smoke *supplementary* | h11_qwen3_4b_smoke | NOT TESTED: not run here: ['run:qwen3_4b_smoke'] | NOT_EVALUATED | none |
| H11.profile | t11_profile_qwen3_4b_async | NOT TESTED: R13 NOT TESTED: missing t11_profile_qwen3_4b_async | NOT_EVALUATED | none |

## H12: The trained policy has meaningfully learned the intended task rather than merely changing parameters or exploiting the reward.

Recorded status: **untested**; cited evidence: none

| entry | evidence name | outputs say | criterion | last execution |
|---|---|---|---|---|
| H12.train_final *supplementary* | h12_train_final | NOT TESTED: not run here: ['run:qwen3_4b_final', 'run:qwen3_4b_final_lr0'] | NOT_EVALUATED | none |
| H12.final_eval | t13_final_eval_qwen3_4b_final | NOT TESTED: R14 NOT TESTED: missing t13_final_eval_qwen3_4b_final; R15 NOT TESTED: missing control evaluation | NOT_EVALUATED | none |

