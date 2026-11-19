# Report-ready facts (audit, 2026-11-19)

Every fact below cites an artifact or source line. Use the wording as given, or narrower. INFERENCE marks interpretation. These facts describe the **150-update run** (`qwen3_4b_final_plain_150_v2`, 2026-11-19). The earlier 60-update run (2026-11-18) appears only where labelled as such. Abbreviations:
- RA = `runs/qwen3_4b_final_plain_150_v2`
- RC = `runs/qwen3_4b_final_plain_150_v2_lr0`
- T13 = `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json`
- T11M = `assignment_results/manual/profile_20261119T054235Z/results/t11_profile_manual.json`
- AR = `assignment_results/final150_v2` (staged runner)
- AM = `assignment_results/manual` (guarded manual stages)

## Architecture

- **Streamed layers.** All 36 Qwen3-4B decoder layers are streamed from pinned host memory; there are 0 resident decoder layers (RA `config.yaml`).
  - Each layer is one page-locked slab of 201,861,632 B, registered with `cudaHostRegister` and with no pageable fallback (`qwen_grpo/execution/weight_store.py:124-150`).
  - One full sweep moves 7,267,018,752 B (T11M `streamed_bytes_per_pass`).
- **GPU buffers.** Two fixed GPU layer buffers (`weight_buffers: 2`, `weight_store.py:248-250`). The prefetch depth is 1.
  - A side CUDA copy stream with ready/free events orders copies and compute (`qwen_grpo/execution/prefetch.py:70,108-121,189-190,201-205`).
  - Buffer reuse is refused at runtime by `_claim` (`prefetch.py:89-106`).
- **What stays on the GPU.** The tied embedding/LM head (151936×2560 bf16 = 777,912,320 B), the final norm and the LoRA parameters stay resident (`layer_stream.py:165-169`).
- **Activation checkpointing.** Only each layer's input is saved. Backward re-streams the layer, recomputes it, checks an exact checksum against the forward, and back-propagates (`qwen_grpo/execution/backward.py:407-424`).
- **Generation and evaluation.** Rollouts, greedy evaluation and label scoring use the same streamed engine (`StreamedGenerator`, `generator.py:87-90`). The KV cache is on the GPU.
  - Recorded backends: `training.json: rollout_backend`, `T13: backends` (all `l2l_async`).
- **Merge.** Runs on the CPU, tensor by tensor, from lazily opened safetensors, in a separate process with no CUDA context. W = bf16(W₀ + (α/r)·B·A), written in 1 GiB shards (`qwen_grpo/export/merge.py:91-176`; `scripts/merge.py`).
- **Disk streaming.** Prompts are streamed from `streaming/prompts/prompts.jsonl` with byte offsets. Each of the 672 rollout batches is written to `streaming/rollouts/group_*.pt` with `index.jsonl` (RA).

## Methods

- **Model and prompt.** Qwen/Qwen3-4B, revision `1cfa9a72…`, bf16 base and fp32 LoRA.
  - Non-thinking chat template (`enable_thinking=False`). A runtime assert checks for the empty `<think></think>` suffix (`qwen_grpo/sentiment.py:23-32`).
- **Task.** Binary sentiment on Rotten Tomatoes.
  - Reward = 1 if `response.strip().upper()` equals the gold label, else 0 (`sentiment.py:35-42`).
  - Splits: test 1066 (533/533), validation 1066, train pool 1500, all with 0 text overlap after normalization (`sentiment.py:70-86`; recomputed in this audit).
  - 1024 balanced training prompts were used (RA `training.json: config.train_ids`).
- **GRPO.**
  - B = 5 prompts × G = 6 samples, temperature 1, no top-k/top-p, `max_new_tokens = 4`.
  - Advantages A = (r − mean)/(std_pop + 1e-4) per group (`qwen_grpo/grpo.py:4-8`).
  - Token-level ratio with PPO clip ε = 0.2. Per-sequence length normalization, then the mean over the 30 completions (`grpo.py:55-65`).
  - No KL term (`kl_beta = 0`). One update per rollout batch (`num_iterations = 1`).
  - Micro-batches of 6, scaled by n_c/N (`training.py:125-126`).
  - Batches where every group has a uniform reward are skipped without an update (`training.py:90-91`, `trainer.py:775-786`).
  - 150 optimizer steps (`max_steps`), at most 1200 attempts.
- **LoRA.**
  - r = 32, α = 64, dropout 0, on q/v/o_proj of layers 34–35.
  - 1,081,344 trainable parameters (0.0269%), asserted at `trainer.py:580`.
  - A uses PEFT's default init and B = 0.
  - AdamW, lr 2e-4, weight decay 0, no gradient clipping, no schedule (`training.json: config`).
- **Integrity checks during training.**
  - A sha256 digest of all 398 frozen tensors before and after: 0 changed (RA `training.json: frozen_changed`).
  - The final adapter reloads bitwise-equal (`final_checkpoint.reloaded_bitwise_equal`).
  - Old log-probs equal the update's log-probs at iteration 0, asserted at 1e-6 (`training.py:119-123`).
- **Evaluation.**
  - Greedy, `max_new_tokens = 4`, on the 1066-example test split.
  - Four models: base (adapter disabled), trained (step 150), merged (CPU merge of step 150, written by `scripts/merge.py` beforehand), LR=0 control (step 30).
  - Paired bootstrap of the accuracy difference: 10,000 resamples, seed 42.
  - Class-stratified paired bootstrap for AUC on s(x) = log P(POSITIVE⟨im_end⟩) − log P(NEGATIVE⟨im_end⟩) (`qwen_grpo/evaluation.py:86-92,143-189`; `scripts/final_eval.py`).

## Experimental setup

- **Hardware and software.** Rented RTX 3060 12 GB (sm_86), Linux 7.0.0-38 host with 72 CPUs and 135,040,708,608 B RAM; torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, Python 3.12.15. Commit `29852d1`, clean tracked tree (RA `training.json: environment`; T13 `provenance`).
- **Staged runner** (`scripts/final150.py`, AR `state.json`, 2026-11-19):
  - preflight PASS (05:05 UTC): every config key checked, including the 8,000,000,000 B budget with the guard on and DFlash off;
  - correctness stage **FAIL** (05:05–05:17): every lab item passed, but `pytest_cpu` had 2 failures of 360 tests (the two guard-enforcement tests; they pass locally);
  - smoke and profile **BLOCKED** by that gate;
  - Soup checks: first attempt UNAVAILABLE, second attempt executed (Soup v0.78.0).
- **Guarded manual stages.** Smoke (05:37–05:41 UTC), profile (05:42–05:44), training (05:45–06:41), LR=0 control (06:45–07:01), merge (07:03–07:04) and the test evaluation (07:04–07:29) were then started by hand, each under `scripts/guarded_run.py` with 200 ms nvidia-smi sampling and cap 8e9 B (AM `*/guarded_run.json`; the command of each is recorded in `argv`). The config hashes match the preflight (T13 `config_sha256` = preflight `config_150_v2` sha256).
- **No orchestrated verdict.** There is no generated requirements matrix or final verdict for this run. Requirement statuses in this audit are the auditor's judgement from the artifacts.
- **Memory measurement.** Two monitors: in-process nvidia-smi sampling every 0.5 s (median gap 0.56 s) and the external guarded wrapper every 0.2 s. Both record per-process `used_memory`, CUDA context included. Guard and allocator cap set to exactly 8,000,000,000 B (`memory.budget_bytes`; `trainer.py:429-436`).

## Results

### Training (RA)

- 150 updates from 672 rollout batches. 522 batches (77.7%) were skipped as zero-signal. 3360 prompt groups, 20160 completions, 3210.9 s, 2.80 updates/min (RA `training.json`).
- Only 163 of 3360 groups (4.85%) had mixed rewards. (1 − 0.0485)⁵ = 0.78 predicts the observed zero-signal rate (RESULTS_ANALYSIS §2). 137 updates had one mixed group, 13 had two.
- The adapter moved:
  - ‖θ − θ₀‖ = 1.9963;
  - ‖(α/r)BA‖ = 2.110;
  - per-step update norms 0.023–0.143;
  - grad norms 0.14–1.67 (RA `training.log`, T13 `lora_change`).
- The logged GRPO loss stayed within ±3.2e-8. With one update per batch this is expected: L = −mean(A) = 0, while the gradient is nonzero (RESULTS_ANALYSIS §1.1).
- Rollout reward over all attempts was flat: 0.856–0.916 per 50-attempt bin. Trained 0.890 vs control 0.880 overall.
- Validation subset (n = 128): 0.859 at step 0, between 0.883 and 0.906 from step 30 on, 0.898 at step 150. The control stayed at 0.859. Step 0 vs 150: 6 fixed, 1 broken (McNemar p = 0.125).

### Held-out test (T13, n = 1066)

| model | accuracy | macro-F1 | AUC | invalid (of 1066) | predicted POSITIVE | recall POS / NEG |
|---|---|---|---|---|---|---|
| base | 0.8771 | 0.8787 | 0.9511 | 5 | 0.436 | 0.816 / 0.938 |
| trained (step 150) | 0.8865 | 0.8869 | 0.9523 | 1 | 0.489 | 0.876 / 0.897 |
| merged | 0.8865 | 0.8869 | 0.9523 | 1 | 0.489 | 0.876 / 0.897 |
| LR=0 control | 0.8771 | 0.8787 | 0.9511 | 5 | 0.436 | 0.816 / 0.938 |

These values were independently recomputed from the predictions and gold labels.

- Trained − base accuracy: **+0.938 pp**, 95% paired-bootstrap CI **[−0.375, +2.251] pp**. Discordant pairs: 22 vs 32, exact McNemar p = 0.22. The CI includes 0, so the accuracy gain is **not significant**.
- AUC difference: **+0.0012, CI [+0.0002, +0.0022]**. The CI excludes 0: a small but significant ranking improvement.
- Invalid outputs fell from 5 to 1 of 1066. The predicted-POSITIVE rate rose from 0.436 to 0.489 (true rate 0.5). Recall moved from NEGATIVE toward POSITIVE.
- **INFERENCE:** the adapter shifted the label score s(x) by +6.01 nats on average (SD 0.88) toward POSITIVE while keeping the ranking nearly identical (Spearman 0.9985 with base). The learned change is mostly a class-bias shift, plus a small real gain in ranking.
- **Merged vs adapter:** predictions identical on 1066/1066. |Δs| has median 9.5e-7 and max 0.375 nats, against a training-induced shift of 6.01 nats. Each of the 6 merged tensors is a correctly rounded bf16 value of its fp32 sum (max residual 8.8e-4; T13 `merge`).
- **LR=0 control:** the adapter is bit-identical to its initialization (B = 0), so predictions and scores are identical to base (T13 `control_log`).

### Compared with the 60-update run (2026-11-18; for comparison only)

| | 60 updates (earlier run) | 150 updates (this run) |
|---|---|---|
| test accuracy Δ vs base | +0.375 pp [−0.563, +1.313] | +0.938 pp [−0.375, +2.251] |
| McNemar (base-only / trained-only right) | 12 / 16, p = 0.57 | 22 / 32, p = 0.22 |
| AUC Δ vs base | +0.0007 [−4e-8, +0.0015] | +0.0012 [+0.0002, +0.0022] |
| mean label-score shift | +3.10 nats | +6.01 nats |
| invalid test answers | 3 | 1 |
| zero-signal batches | 77% | 78% |

Sources: `docs/results_20261119_150_v2.md` ("Compared with the 60-update run") and the earlier version of this audit; the 60-update run's files are archived in `~/soup-assignment-results.tar.gz`.

### Memory (cap 8,000,000,000 B = 7629 MiB per process; every stage exited 0 with 0 violations)

- **Training:** peak nvidia-smi per-process 2,864,709,632 B (2732 MiB), device 2,898,395,136 B, PyTorch reserved 2,690,646,016 B, allocated 1,806,041,088 B.
  - 5947 in-process samples, 0 missing, 0 violations (RA `training/memory/20261119T054525Z.json`). The external wrapper (16,643 samples) agrees: 2732 MiB (AM `train_20261119T054519Z/guarded_run.json`).
  - That is 35.8% of 8 × 10⁹ B. The LR=0 control has an identical peak (RC).
- **Other stages (guarded, nvidia-smi process peak):** smoke 2362 MiB; profile 2106 MiB; merge: no GPU process at all (device 13 MiB, CPU only); **evaluation of base, trained, merged and control: 1912 MiB**; kill/resume A 2364 MiB, B after the resume 2126 MiB.
- **Per-phase profile, two steps** (T11M, 250 ms sampling): setup 1.30e9 B, rollout 1.96e9 B, old log-probs 1.96e9 B, forward 1.96e9 B, evaluation (32 examples) 2.21e9 B per process. The optimizer phases were shorter than the sampling interval and got no sample; only 2 of 10 backward windows got one.
- Host memory: training VmHWM 14.39 GiB; CPU RSS peak 17.5 GB during the profile's merge.
- Predicted rollout device peak was 2.425 GiB (committed before any GPU run, `results/budget/predictions/qwen3-4b-final_8gb.md`). Measured: 2.668 GiB.

### Throughput (RA `training.json: timings`; successful steps only)

- Mean step 9.55 s (median 9.49 s): rollout 23.5%, old log-probs 36.8%, update 39.4%.
- Rollout generation 40.1 tokens/s (30 sequences of about 3 tokens).
- Each step copies 13 or 14 full weight sweeps (94.5 GB in 143 steps, 101.7 GB in 7):
  - rollout: 3–4 sweeps;
  - old log-probs: 5 micro-batches × 1 sweep;
  - update forward: 5 sweeps.
- Pinned H2D bandwidth: two measurements disagree. 12.23 GB/s (2026-11-18, earlier host; used by `results/budget/residuals/qwen3-4b-final_150_v2.md`) and 7.25 GB/s (this run's profile, T11M). At 7.25 GB/s, 14 sweeps would take 14.03 s, longer than the measured 9.55 s step, so that figure is inconsistent with the training run. Report both; do not claim which is right.
- **INFERENCE:** the step is dominated by weight transfer. It moves weights at 9.93 GB/s on average, 81% of the 12.23 GB/s figure. The step is about 1 s slower than in the 60-update run (9.55 s vs 8.57 s), whose step the residuals analysis found at 1.01× the transfer bound.
- 55% of training wall time (1779 s) fell outside successful steps: rollouts of skipped batches (1174 s), validation passes and checkpoints.
- **Forecast vs result:** calibrated on the 60-update run's acceptance (0.226), the forecast was 665 attempts and 0.83 h for 150 updates. Actual: 672 attempts, 3211 s = 0.89 h (`results/budget/residuals/qwen3-4b-final_150_v2.md`).

### Correctness evidence (tiny random Qwen3 models; RTX 3060; async L2L vs resident; AR correctness stage, `evidence/20261119T05*Z_H*/`)

- Forward: **2690/2690 bitwise**, including the same-host fixture tie-backs. The 2026-11-18 failure (2 cross-host CPU fixture comparisons) is gone. Those comparisons are now non-gating diagnostics, and on this host the resident forward shows the same drift as L2L (`research/l2l/experiments/t08_forward_equivalence_async_r20261119T051124Z/`).
- Gradients: 5296/5296 bitwise equal.
- Full GRPO update incl. AdamW state: 6528/6528 bitwise.
- Generation: 1292/1292 within bounds.
- Unsafe buffer reuse, injected: 66 (forward) and 80 (gradient) configurations, all refused at runtime with `BufferReuseError`; none went unnoticed.
- Merge (tiny, CPU): 30/30 within bounds.
- Qwen3-0.6B: resident vs synchronous L2L, 100 GRPO steps, every training row, adapter and optimizer state bitwise equal (`research/l2l/experiments/compare_qwen_resident_vs_qwen_l2l.json`).

### Deterministic resume after a real kill (`research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json`)

- Qwen3-4B, 6 steps, checkpoint every 2. Run B was SIGKILLed from outside after 3 step lines, with checkpoints 0 and 2 committed. A new process resumed it from the latest checkpoint and finished 6/6.
- Equal to the uninterrupted run A: committed training rows, rollout records on disk, the sample multiset (no gaps, no duplicates), adapter and AdamW state at every checkpoint, final torch RNG. `pass: true`. All three processes stayed under the cap.
- On the tiny CPU model, the external-kill test and the six injected crash points also pass.

### Soup v0.78.0 checks (`research/soup/0.78.0_20261119T073222Z/summary.json`)

- Reward stress, real reward: robust, 0 of 4600 attacks accepted; reference accepted 100%.
- Reward stress, deliberately permissive verifier (positive control): gameable, answer spray accepted 800/800, i.e. **detected**.
- `soup ship` on the 150-update model: **DON'T SHIP** (`failed_rule: missing_baseline`). The task win 0.8771 → 0.8865 was recognised, but no general benchmark or `--baseline` was given, so forgetting could not be checked; Soup did not use the paired CI. On the LR=0 control: DON'T SHIP (no task win).
- `soup data doctor` (clean + 6 faults) and `soup adapters check` (4 adapters) **did not run their checks**: exit 1 for missing `soup-cli[train]` dependencies (transformers, numpy). `soup doctor` ran in a different interpreter from training.

### Pipeline sanity on a model with headroom (different setting; disclose the caveats)

- Qwen3-0.6B, 100 steps, lr 5e-4, resident backend, RTX 3050 Ti laptop, dirty tree.
- Test accuracy 0.733 → 0.796 (step 90), Δ +6.4 pp, CI [+4.2, +8.5] (`runs/qwen/evaluation/step90/test.json`).

## Hypotheses (see HYPOTHESES_VERDICT.md)

| hypothesis | status |
|---|---|
| H01, H02, H04, H05 (forward, backward, generation, update equivalence) | SUPPORTED on tiny models (H01 async now passes 2690/2690) |
| H08 (resume) | SUPPORTED; also shown on Qwen3-4B with a real SIGKILL and a new-process resume |
| H09 (merge) | INCONCLUSIVE under its criterion (real Qwen3-0.6B variant not run) |
| H11 (8 GB budget) | SUPPORTED by its rule; every phase also measured in guarded stages |
| **H12 (meaningful learning)** | **NOT SUPPORTED** (accuracy CI includes 0), although the AUC gain is significant |
| H06 (async speed-up), H10 (DFlash) | NOT TESTED |

## Limitations (state these)

- DFlash rollouts were implemented and unit-tested on CPU but **never run on Qwen3-4B** (disabled in this run).
- Copy/compute overlap and any async speed-up were **not measured**.
- The two unit tests that check the memory guard **stops** an over-budget run failed on the Soup host (they pass locally). The guard was armed at 8e9 B and never triggered, but its enforcement was not demonstrated on that host.
- The staged runner's correctness gate failed on those tests, so smoke, profile, training, control, merge and evaluation were run by hand (each under the guarded wrapper). There is no single orchestrated verdict.
- The two H2D bandwidth measurements disagree (12.23 vs 7.25 GB/s), and the 7.25 GB/s figure is inconsistent with the measured step times.
- The merged weights were not kept (`assignment_results/final150_v2/merged/` is empty); only their shard sha256 values are recorded (T13 `merged_dir`).
- Kill/resume on the real model was shown for one kill point in one 6-step run.
- The LR=0 control is identical to base by construction. It ran 30 steps, not 150.
- Async equivalence was tested on tiny random models with `verify=True`. Real-model async equivalence (stage C, Qwen3-0.6B) was not rerun.
- The validation subset of 128 is too small to read trends.
- The base model may have seen Rotten Tomatoes in pretraining.
- Soup's data doctor and adapter checks did not run (missing dependencies); that is an installation issue, not evidence about what they detect.

## Conclusions supported by the evidence

1. A Qwen3-4B GRPO + LoRA run of 150 updates executed end to end with every decoder layer streamed from pinned host memory. nvidia-smi showed at most 2.86 × 10⁹ B per process during training, and at most 1912 MiB during the evaluation of the base, trained, merged and control models. The merge used no GPU.
2. The streaming engine is numerically exact against resident execution in every tested configuration (tiny models, async, same host), and a real-model run resumes deterministically after a SIGKILL.
3. On Qwen3-4B, 150 updates changed the policy and gave a **small, partly demonstrated** improvement: ranking improved significantly (AUC +0.0012, CI [+0.0002, +0.0022]), invalid answers fell from 5 to 1, and predictions became more balanced. The accuracy gain (+0.94 pp) is **not** statistically significant. Most of the change is a POSITIVE bias shift (INFERENCE).
4. The likely reasons the gain is small (INFERENCE) are:
   - the base is already 87.7% accurate;
   - only 4.85% of sampled groups carried a learning signal, so 78% of batches gave no update;
   - only 163 informative groups drove the 150 updates.
5. The training step is dominated by weight transfer (INFERENCE). Generation is about a quarter of a step because completions are 3 tokens. Old log-probs and the update cost 5 sweeps each because micro-batches re-stream the model.
6. **Verdict supported: DON'T SHIP.** The accuracy improvement over the base model is not significant, and Soup's ship check also says DON'T SHIP (missing forgetting baseline).

## CLAIMS I MUST NOT MAKE

| tempting claim | why not |
|---|---|
| "GRPO significantly improved accuracy" / "the model learned the task" | accuracy CI [−0.38, +2.25] pp includes 0; McNemar p = 0.22; H12 NOT SUPPORTED |
| "Accuracy improved by 0.9 points" (without the CI) | a point estimate inside noise |
| "Training did not improve the model at all" | AUC improved significantly (+0.0012, CI [+0.0002, +0.0022]); invalid answers fell 5 → 1 |
| "The model got much better at sentiment discrimination" | the AUC gain is small; rank correlation with base is 0.9985; most of the change is a bias shift |
| "Reward rose during training" | the unbiased rollout reward is flat; the per-step reward is selection-biased |
| "Validation accuracy plateaued, showing diminishing returns" | n = 128, SE 0.028; the curve moves within noise after step 30 |
| "The memory guard is proven to stop over-budget runs" | its two enforcement tests failed on the Soup host; it never had to act in the real runs |
| "Peak memory 1.73 GiB" or "2.51 GiB" as the budget figure | those are allocator numbers that exclude the CUDA context; use the nvidia-smi 2,864,709,632 B |
| "The eval peaked at 1.20 GB" (`peak_gpu_allocated_bytes_eval`) | that field is read after a peak-counter reset, not over the whole evaluation; use the nvidia-smi 1912 MiB |
| "Asynchronous prefetching overlaps transfer and compute" or "speeds up training" | no overlap trace and no sync baseline; H06 NOT TESTED |
| "PCIe bandwidth is 7.25 GB/s" (or 12.23 GB/s) as a settled fact | the two measurements disagree, and 7.25 GB/s contradicts the step times |
| "L2L is numerically equivalent on the real 4B model" | async equivalence only on tiny models with verify on; real-model equivalence only for sync on 0.6B |
| "DFlash accelerated rollouts" or "DFlash gives no speed-up" (as a measurement) | DFlash never ran on the real model |
| "Deterministic resume holds for any interruption" | one SIGKILL point in one 6-step run, plus tiny-model crash points |
| "The merged model is numerically identical to the adapter" | predictions identical; scores differ by up to 0.375 nats |
| "The negative control proves the improvement is caused by the reward" | LR=0 equals base by construction; a shuffled-reward control was not run |
| "PPO clipping / KL stayed small, showing stable updates" | with one update per batch, ratio = 1, clip = 0 and KL = 0 by construction |
| "The orchestrated pipeline passed" / "the final run passed the gate" | the staged runner's correctness gate failed; the GPU stages were run by hand |
| "Fully reproducible" | commands, logs and hashes are recorded, but the merged weights were not kept and there is no single orchestrated run |
| "Soup's data doctor / adapter checks passed" or "found nothing" | they did not run (missing dependencies) |
| "Ready to ship" | DON'T SHIP is the supported verdict |
