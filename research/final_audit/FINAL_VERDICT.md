# Final audit verdict: Soup AI Engineer (Hard Track v5)

- **Audit date:** 2026-11-19 (updated for the 150-update run `qwen3_4b_final_plain_150_v2`; this replaces the audit of the 60-update run `20261118T171404Z`).
- **Assignment source:** `Soup-Task-01-AI-Engineer-v5.pdf` (outside the repo, sha256 `79e2f126…0f35`).
- **Artifacts audited:**
  - `assignment_results/final150_v2/` (staged runner `scripts/final150.py`: `state.json`, `preflight/`, `correctness/`, `soup/`, `export_manifest.sha256`)
  - `assignment_results/manual/{smoke,profile,train,control,merge,eval}_20261119T*/` (stages run by hand under `scripts/guarded_run.py`, with nvidia-smi telemetry)
  - `runs/qwen3_4b_final_plain_150_v2{,_lr0,_smoke}/`, `runs/qwen3_4b_resume_kill_kill{A,B}/`
  - `research/l2l/experiments/t08_*_r20261119T05*Z`, `t09_*`, `t10_merge_equivalence_tiny_r20261119T051649Z`, `t13_final_eval_qwen3_4b_final_plain_150_v2.json`; profile `assignment_results/manual/profile_20261119T054235Z/results/t11_profile_manual.json`
  - `evidence/20261119T05*Z_H*`
  - `research/soup/0.78.0_20261119T073222Z/`
  - `results/budget/residuals/qwen3-4b-final_150_v2.{md,json}`
  - the source at commit `29852d1` (clean tree)
- **Supporting files:** REQUIREMENTS_MATRIX, EXPERIMENT_INVENTORY, RESULTS_ANALYSIS, HYPOTHESES_VERDICT, RISKS_AND_GAPS, RECOMMENDED_NEXT_ACTIONS, REPORT_READY_FACTS (this directory). Numbers can be regenerated with `python3 -I research/final_audit/analyze.py`.
- **No generated verdict exists for this run.** The staged runner's correctness gate FAILED (only `pytest_cpu`), so it BLOCKED smoke and profile; those stages, training, the control, the merge and the evaluation were then run by hand under the guarded wrapper. Every status below is this auditor's judgement from the artifacts, not the output of an orchestrator.

## Verdicts

| dimension | verdict | why |
|---|---|---|
| **Assignment compliance** | **NOT COMPLETE: reportable with disclosure** | Core materials are SATISFIED: Qwen3-4B non-thinking, no SFT, GRPO + LoRA, verifiable reward, 1024 prompts, own L2L, disk streaming, real-model kill/resume, CPU tensor-wise merge, a 4-model held-out comparison with CIs, and nvidia-smi evidence for training, rollouts, merge and merged-model evaluation. Still unmet or unevidenced: DFlash (2c, never run on the real model); measured copy/compute overlap; Soup's `data doctor` and `adapters check` (they failed to start for missing `soup-cli[train]` dependencies). The PDF says "Not every part has to work … tell us what happened", so these gaps are acceptable **only if stated as gaps**. |
| **Research validity** | **READY WITH QUALIFICATIONS** | Hypotheses were pre-registered and the main result is judged by its own rule: H12 is NOT SUPPORTED (accuracy CI includes 0). Data separation is clean (0 test texts in train). Statistics recompute exactly. The significant AUC gain is a secondary result: AUC was listed as a reported quantity before the run (`docs/FINAL_150_STEP_EXPERIMENT.md`, commit `5680e71`) but is not the H12 criterion. The 150-update run was a second attempt after the 60-update null, which must be disclosed (forking paths). H06 and H10 were never tested. H01 is now closed (2690/2690 on the RTX 3060 host). |
| **Engineering correctness** | **READY WITH QUALIFICATIONS** | No P0/P1 defect in GRPO/LoRA math, masks, advantage normalization, micro-batch scaling, gradient flow or reward (code audit plus CPU tests). On the RTX 3060, async L2L matches resident execution: forward 2690/2690, gradients 5296/5296, updates 6528/6528, generation 1292/1292; injected unsafe buffer reuse refused 66/66 and 80/80. Real-model kill/resume passes. Qualifications: async equivalence only on tiny random models with `verify=True`; two guard-enforcement unit tests failed on the Soup host (they pass locally). The profile's H2D transfer counter is fixed in `5680e71`. |
| **GPU memory** | **READY (all four named phases measured)** | Every stage ran under the external guarded wrapper (nvidia-smi every 200 ms, cap 8 × 10⁹ B = 7629 MiB), all exit 0, 0 violations. Training per-process peak 2732 MiB (in-process sampler: 2,864,709,632 B, 35.8% of 8e9); control 2732 MiB; merge: no GPU process (CPU merge, device 13 MiB); evaluation of base/trained/merged/control 1912 MiB. The in-process guard budget is now exactly 8e9 B. Qualifications: 200 ms sampling gives a lower bound on the true peak; the in-process guard's stop-on-violation was not demonstrated on the Soup host (two failed unit tests). |
| **Model learning** | **SMALL, PARTLY DEMONSTRATED (ranking yes, accuracy no)** | Test Δ accuracy +0.938 pp, CI [−0.375, +2.251]; McNemar p = 0.22 (22 vs 32 discordant). AUC +0.0012, CI [+0.0002, +0.0022], which excludes 0. Invalid answers 5 → 1; recall NEG/POS 0.938/0.816 → 0.897/0.876; predicted-POSITIVE 0.436 → 0.489. Label scores moved +6.01 nats on average with Spearman 0.9985 against base: still mostly a shift of the label prior, with a small real ranking improvement (INFERENCE). Rollout reward stayed flat (per-50-attempt bins 0.856–0.916). |
| **Reproducibility** | **READY WITH QUALIFICATIONS** | Recorded: commit `29852d1` (clean), config sha256 `568afd24…`, dataset sha256 `c85d1cbc…`, model revision `1cfa9a72…`, package versions, adapter sha256s (t13 `identity`), merged-shard sha256s (t13 `merged_dir.provenance`), bitwise checkpoint reload, the exact command, stdout and stderr of every stage (`assignment_results/manual/*/guarded_run.json`), and `export_manifest.sha256`. Missing: the merged checkpoint itself (`assignment_results/final150_v2/merged/` is empty); a single orchestrated run and verdict. |
| **Overall readiness to write the final report** | **READY WITH QUALIFICATIONS** | Everything needed for an honest **DON'T SHIP** report exists. No further GPU run is required. The report (`reports/Soup_Final_Research_Report.{html,pdf}`, 3 pages) was rewritten for the 150-update run on 2026-11-19; its evidence index is `reports/REPORT_EVIDENCE_MAP.md` and `reports/verify_report_numbers.py` passes 93/93. |

## Supported ship decision

**DON'T SHIP.** The trained artifact is not demonstrably more accurate than the base model on 1066 held-out examples (accuracy CI includes 0). It ranks slightly better (AUC CI excludes 0) and gives fewer invalid answers, but most of the change is a shift of the label prior. Soup's own `soup ship` also says DON'T SHIP (`failed_rule: missing_baseline`: no general benchmark suite or `--baseline` was supplied, so forgetting could not be checked).

What must change first:
1. A task, or prompt selection, where the 4B base has headroom. Only 4.85% of groups were mixed (163 of 3360) and 77.7% of rollout batches carried no signal.
2. A pre-registered stopping rule and sample size that can detect the targeted accuracy effect (the paired design detects about 1.4 pp at this discordance; `approx_mde_accuracy_given_discordance` 0.0135 in `analysis_output.json`).
3. A forgetting baseline (a general benchmark run on base and trained), so that `soup ship` can evaluate its regression rule.

## Answers A–K

**A. Did we implement GRPO + LoRA correctly? YES.**
- Code audit and CPU tests found no defect in advantages, the clipped token-level surrogate, per-sequence normalization, micro-batch scaling, zero-signal skipping, old log-probs, or LoRA gradient flow through the checkpointed L2L backward.
- Evidence: `qwen_grpo/grpo.py:4-65`, `qwen_grpo/training.py:90-146`, `qwen_grpo/execution/backward.py:60-102,407-450`, `tests/test_grpo.py`, `tests/test_multistep.py`, H02/H05 bitwise on the RTX 3060 (`evidence/20261119T051240Z_H02.lab_async_cuda`, `evidence/20261119T051445Z_H05.lab_async_cuda`).
- Caveat, not a blocker: with `num_iterations = 1` the clip never activates (ratio = 1, clip 0 at all 150 steps), so the objective is REINFORCE with a group baseline. The ratio/clip/KL diagnostics are structurally uninformative.

**B. Did we demonstrate genuine layer-by-layer CPU-to-GPU streaming? YES.**
- Each successful step copied 13–14 full sweeps of 7,267,018,752 B from pinned slabs (94,471,243,776 or 101,738,262,528 B per step).
- The PyTorch allocator peaked at 1,806,041,088 B, and nvidia-smi at 2,864,709,632 B per process, against 8.04e9 B of bf16 weights. The model was never resident.
- Evidence: `runs/qwen3_4b_final_plain_150_v2/training/training.json: timings`; t11 profile `streamed_bytes_per_pass`, `pinned_bytes` (`assignment_results/manual/profile_20261119T054235Z/results/t11_profile_manual.json`); `weight_store.py:124-150`.
- Disclosure: the tied embedding/LM head (0.78 GB) stays resident.

**C. Did we demonstrate the required memory compliance? YES, for all four named phases (sampled).**
- Training and rollouts: 2732 MiB per process under the external sampler (16,643 samples, max gap 0.332 s), and 2,864,709,632 B under the in-process sampler (5947 samples), 0 violations (`assignment_results/manual/train_20261119T054519Z/guarded_run.json`; `runs/qwen3_4b_final_plain_150_v2/training/memory/20261119T054525Z.json`). Control likewise (2732 MiB).
- Merge of the step-150 adapter: no CUDA context (0 compute-app rows over 159 device samples; device 13 MiB), 32 s, CPU max RSS 8,602,054,656 B (`assignment_results/manual/merge_20261119T070347Z/guarded_run.json`).
- **Evaluation of the merged model:** the whole eval process (base, trained, merged, control) peaked at 1912 MiB over 7295 samples. In the merged pass's window (07:17:24–07:22:54Z from t13 `phases`) the peak was also 1912 MiB (auditor's computation from `assignment_results/manual/eval_20261119T070439Z/telemetry/nvidia_smi_apps.csv`).
- The in-process guard budget is exactly 8e9 B (`qwen_grpo/trainer.py:421-436`).
- Remaining qualifications: 200 ms sampling is a lower bound on the true peak; the in-process guard's stop-on-violation tests failed on the Soup host (RISKS_AND_GAPS P1-2).

**D. Did we demonstrate correct asynchronous execution? YES, within scope.**
- Async L2L matched resident execution on the RTX 3060 at commit `29852d1`:
  - forward 2690/2690 (H01; the 2026-11-18 fixture-only failure is gone, fixed in `5680e71`);
  - gradients 5296/5296;
  - updates incl. AdamW 6528/6528;
  - generation 1292/1292 within bounds.
- Evidence: `evidence/20261119T05*Z_H0*`, `research/l2l/experiments/t08_*_r20261119T05*Z/`.
- Scope limits: tiny random models with `verify=True`. H01's gate is now the same-host comparison (redefined in `5680e71`, before this run); the cross-host fixture drift (max 1.34e-7 on the 0.6b-like fixture) is still recorded as a non-gating diagnostic.

**E. Did we demonstrate an asynchronous performance benefit? NO.**
- No sync-vs-async measurement and no overlap timeline exist. H06 is untested.
- Mean timed step 9.55 s. The new profile measured H2D at 7.25 GB/s, which would make 14 sweeps take 14.03 s, longer than the measured step; that bandwidth figure is therefore inconsistent with the training step times. The residuals file used the 12.23 GB/s measured on 2026-11-18 and found the 60-update run's steps transfer-bound (8.54 s vs an 8.48 s bound). Both figures are reported; neither is a sync-vs-async comparison.

**F. Did the model improve, and is it statistically convincing? PARTLY.**
- Accuracy: +0.938 pp (54 discordant pairs: 32 vs 22), 95% CI [−0.375, +2.251]; McNemar p = 0.22. **Not significant**; the pre-registered H12 rule is not met.
- AUC: +0.0012, CI [+0.0002, +0.0022]. **Significant**, but small and not the pre-registered criterion.
- Invalid answers 5 → 1; recall rebalanced; no label collapse.
- Evidence: t13 `paired`, `paired_auc`; RESULTS_ANALYSIS §4.

**G. Did merging preserve the trained model's behaviour? YES.**
- Predictions identical on 1066/1066. Accuracy, F1 and invalid rate identical. AUC difference +8.8e-6, CI [−2.8e-4, +3.1e-4].
- Label-score differences: median 9.5e-7, max 0.375 nats; every adapted tensor within half an ulp (max residual 8.8e-4), consistent with bf16 merge rounding (INFERENCE).
- No pre-declared logit tolerance for the 4B merge (t13 `merge`, `merged_vs_trained_prediction_agreement` 1.0).

**H. Were the original research hypotheses adequately tested? MOSTLY.**
- Correctness H01/H02/H04/H05 were well tested on the RTX 3060 (tiny models) plus 0.6B sync from earlier runs.
- H08 now has a real Qwen3-4B kill/resume PASS (`t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json`).
- H06 and H10 were never tested. H09 only on tiny CPU models (plus the 4B merge prediction agreement in t13).
- H12 was tested and is NOT SUPPORTED by its accuracy rule.
- Evidence: HYPOTHESES_VERDICT.md, `research/l2l/hypotheses.yaml`.

**I. Have we satisfied every mandatory Soup assignment requirement? NO.**
- DFlash (2c) was never run on the real model.
- Overlap was not measured.
- Soup checks only partly ran: reward stress (real reward robust 0/4600; permissive positive control caught 800/800) and `soup ship` ran; `data doctor` (7 inputs) and `adapters check` (4 checkpoints) failed to start; `doctor` ran in a different interpreter from training.
- Closed since the last audit: merged-model evaluation memory, real kill/resume, guard at 8e9 B, H01.
- The assignment explicitly allows disclosed non-completion (REQUIREMENTS_MATRIX.md).

**J. Is further GPU experimentation necessary? No.**
- The memory and resume gaps are closed. No more training is needed, and none is recommended.
- **Optional:**
  - B1: a ~10-minute sync-vs-async profile (overlap, H06);
  - C1: a ~30–60-minute DFlash bench.
- CPU-only, cheap: install `soup-cli[train]` and re-run `data doctor` and `adapters check` (B2); re-run the two failed guard tests on the Soup host to find out why they fail there (B3).

**K. Are we ready to write the final research report? YES, with qualifications.**
- Write it as DON'T SHIP with a small, partly demonstrated learning effect (ranking yes, accuracy no).
- Use REPORT_READY_FACTS.md, and avoid every item in its "CLAIMS I MUST NOT MAKE".
- Done 2026-11-19: report rewritten in the assignment's five sections; evidence index regenerated.

## Three highest-priority next actions (by importance)

1. **Write the verdict as DON'T SHIP and the learning result precisely:** accuracy +0.94 pp not significant (H12 NOT SUPPORTED), AUC +0.0012 significant, invalid answers down, mostly a label-prior shift (+6.01 nats, Spearman 0.9985). Disclose that the 150-update run followed a 60-update null and that the staged runner's correctness gate failed on `pytest_cpu`, so the remaining stages were run by hand under the guarded wrapper. Writing only.
2. **Disclose the unmet engineering items with the evidence you do have:**
   - DFlash: implemented and CPU-tested, never run; give an Amdahl bound (rollouts are 23.5% of a timed step) and optionally run the bench.
   - Overlap and speed-up: unmeasured; report both bandwidth figures (12.23 vs 7.25 GB/s) without choosing.
   - Soup: what ran and what failed to start, and why.
   - Guard: two enforcement tests failed on the Soup host; the runs never came near the cap (peak ≈36%).
3. **Close the cheap CPU gaps:** install `soup-cli[train]` and re-run Soup's data and adapter checks; re-run the two guard tests on the Soup host. Both are minutes on CPU.
