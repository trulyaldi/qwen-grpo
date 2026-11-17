# Final report: GRPO + LoRA with L2L streaming (Soup assignment)

**Verdict: DON'T SHIP**

Every status below is derived by `scripts/final_report.py` from the files in `research/l2l/experiments/`; nothing is entered by hand. Code, raw results and nvidia-smi samples are kept separately in the repository.

## Requirements

| id | requirement | essential | status | evidence |
|---|---|---|---|---|
| R1 | L2L forward equals resident (H01) | yes | **PASS** | t03_forward_equivalence |
| R2 | checkpointed backward equals resident (H02) | yes | **PASS** | t04_gradient_equivalence_cuda |
| R3 | memory-efficient output head (H03) | no | **PASS** | t05_logprob_equivalence_cpu_v2 |
| R4 | streamed generation equals resident (H04) | yes | **PASS** | t06_generation_equivalence_cuda |
| R5 | complete GRPO update equals resident (H05) | yes | **PASS** | t07_update_equivalence_cuda_v2 |
| R6 | async prefetch keeps results exact | yes | **NOT TESTED** | missing ['t08_gradient_equivalence_async', 't08_update_equivalence_async'] |
| R7 | async prefetch is faster (H06) | no | **NOT TESTED** | missing t08_prefetch_timing |
| R8 | GPU weight residency bounded in depth (H07) | yes | **NOT TESTED** | missing ['t11_depth_residency'] |
| R9 | deterministic crash-consistent resume (H08) | yes | **PASS** | t09_resume_tiny_cpu (tiny model only) |
| R10 | streaming merge preserves behaviour (H09) | yes | **NOT TESTED** | missing ['t10_merge_equivalence_qwen3_0.6b'] |
| R11 | DFlash on-policy correctness (H10) | yes | **NOT TESTED** | missing t12_dflash_bench |
| R12 | DFlash speedup (H10) | no | **NOT TESTED** |  |
| R13 | Qwen3-4B within the 8 GB GPU budget (H11) | yes | **NOT TESTED** | missing t11_profile_qwen3_4b_async |
| R14 | held-out task improvement with Qwen3-4B, 1-2k prompts (H12) | yes | **NOT TESTED** | missing t13_final_eval_qwen3_4b_final |
| R15 | negative control behaves as no learning | yes | **NOT TESTED** | missing control evaluation |

## Silent-failure audit

| stage | silent failure | detection | fault / evidence | occurred | detected by tooling |
|---|---|---|---|---|---|
| Data | train/test overlap, unbalanced labels | split by source split + text check in final_eval; balanced_subset | final_eval data_separation count | not observed | yes (tested) |
| Tokenizer/template | thinking-mode template, wrong padding side | build_prompts asserts the non-thinking block; left padding checked by check_grouping | test_policy / gen_lab chat prompts | no | yes (tested) |
| Reward | reward hacking (formatting, label without content) | strict canonical-label reward; invalid-rate and prediction-distribution tracking; LR=0 control | t13 held-out metrics | see held-out results | partly |
| GRPO | wrong advantage / stale old log-probs / ratio != 1 at it0 | INIT_TOL check; update_equivalence | t07 corrupted-advantage fault (caught) | no | yes (tested) |
| L2L forward/backward | wrong layer weights, stale buffer, detached boundary | bitwise equivalence cases; recomputation checksum | t03 wrong_layer, t04 detach_boundary, t08 unsafe_reuse | no | yes (tested) |
| Output head | unnormalized logit, missing remainder chunk | logprob_equivalence | t05 raw_logit fault | a shape-dependent 1e-7 drift on one CPU (fixed, bc5a05d) | yes (tested) |
| Generation | stale position, attended padding, EOS handling | generation_equivalence | t06 faults | no | yes (tested) |
| Disk streaming/resume | duplicate/missing samples, double-applied update, torn checkpoint | crash injection; COMMIT.json validation | t09 six crash points + corrupted checkpoint | no | yes (tested) |
| DFlash | drafter replacing the policy, wrong acceptance | greedy equality; chi-square vs exact law | tests/test_dflash (random/oracle drafters) | no | yes (tiny models only) |
| Merge | dropped scale, wrong module, lost delta in bf16 | half-ulp residual check; power ratio; L2L eval | merge_equivalence power requirement | no | yes (tested) |
| Inference/memory | budget exceeded outside the allocator | nvidia-smi sampling + guard | profile runs | unknown under WSL2 | only with per-process nvidia-smi |

The full registry (every stage, Soup's observed coverage, fault-injection tests): `docs/silent_failures.md`.

## Evidence freshness

PASS evidence and the engine files (`qwen_grpo/execution`, `qwen_grpo/training.py`, `qwen_grpo/grpo.py`, `qwen_grpo/policy.py`, `qwen_grpo/model.py`, `qwen_grpo/streaming`, `qwen_grpo/export`) modified or deleted since it was produced (new files are not listed). A non-empty list means the result predates the current code: it is reported, not re-labelled, and should be re-run before it is relied on.

| requirement | evidence | commit | engine files changed since |
|---|---|---|---|
| R1 | t03_forward_equivalence | c4b391be66 | `qwen_grpo/execution/fixtures.py`, `qwen_grpo/execution/layer_stream.py`, `qwen_grpo/execution/qwen3_forward.py`, `qwen_grpo/execution/weight_store.py`, `qwen_grpo/policy.py`, `qwen_grpo/training.py` |
| R2 | t04_gradient_equivalence_cuda | 5cc0c0a2a0 | `qwen_grpo/execution/fixtures.py`, `qwen_grpo/execution/layer_stream.py`, `qwen_grpo/execution/output_head.py`, `qwen_grpo/execution/weight_store.py`, `qwen_grpo/policy.py` |
| R3 | t05_logprob_equivalence_cpu_v2 | 6657edfe6b | none |
| R4 | t06_generation_equivalence_cuda | 853091f28c | `qwen_grpo/execution/fixtures.py`, `qwen_grpo/execution/generator.py`, `qwen_grpo/execution/layer_stream.py`, `qwen_grpo/execution/output_head.py`, `qwen_grpo/execution/weight_store.py` |
| R5 | t07_update_equivalence_cuda_v2 | a8aab7fd1c | `qwen_grpo/execution/backend.py`, `qwen_grpo/execution/fixtures.py`, `qwen_grpo/execution/layer_stream.py`, `qwen_grpo/execution/output_head.py`, `qwen_grpo/execution/weight_store.py` |
| R9 | t09_resume_tiny_cpu | 726e2ab03c | `qwen_grpo/execution/backend.py`, `qwen_grpo/execution/fixtures.py` |

## What must change

- R6 async prefetch keeps results exact: NOT TESTED (missing ['t08_gradient_equivalence_async', 't08_update_equivalence_async'])
- R8 GPU weight residency bounded in depth (H07): NOT TESTED (missing ['t11_depth_residency'])
- R10 streaming merge preserves behaviour (H09): NOT TESTED (missing ['t10_merge_equivalence_qwen3_0.6b'])
- R11 DFlash on-policy correctness (H10): NOT TESTED (missing t12_dflash_bench)
- R13 Qwen3-4B within the 8 GB GPU budget (H11): NOT TESTED (missing t11_profile_qwen3_4b_async)
- R14 held-out task improvement with Qwen3-4B, 1-2k prompts (H12): NOT TESTED (missing t13_final_eval_qwen3_4b_final)
- R15 negative control behaves as no learning: NOT TESTED (missing control evaluation)
