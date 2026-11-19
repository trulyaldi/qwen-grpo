# Requirements matrix (audit, 2026-11-19)

**Authoritative source:** `/mnt/c/Users/aldik/Downloads/Soup-Task-01-AI-Engineer-v5.pdf` (sha256 `79e2f126…0f35`). It is not in the repository. Quoted wording is from that PDF.

**Optional vs mandatory:** the assignment does not label items. Everything under TASK / HARD CONSTRAINT / DELIVERABLES is treated as mandatory. Two passages qualify how failures should be handled:
- "Not every part has to work: if something breaks or your credit runs out, tell us what happened, what you tried and what you learned."
- "Don't edit or hide failed results."

So an unmet item can still be **reportable** if it is honestly disclosed. It is not thereby satisfied.

**Statuses:** SATISFIED / PARTIALLY SATISFIED / NOT SATISFIED / NOT TESTED / INCONCLUSIVE / NOT APPLICABLE.

**Whose statuses these are:** this run (150 updates, commit `29852d1`) was driven by the staged runner `scripts/final150.py` and then by hand. The runner's correctness gate FAILED on `pytest_cpu` (2 of 360 tests), so it never produced a requirements matrix or verdict. Every status below is the auditor's judgement from the artifacts.

**Run identifiers:**
- RA = `runs/qwen3_4b_final_plain_150_v2`
- RC = `runs/qwen3_4b_final_plain_150_v2_lr0`
- T13 = `research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json`
- FS = `assignment_results/final150_v2` (staged runner: `state.json`, `preflight/`, `correctness/`, `soup/`)
- MAN = `assignment_results/manual` (guarded manual stages: `smoke_20261119T053734Z`, `profile_20261119T054235Z`, `train_20261119T054519Z`, `control_20261119T064530Z`, `merge_20261119T070347Z`, `eval_20261119T070439Z`)
- T11 = `MAN/profile_20261119T054235Z/results/t11_profile_manual.json`

## Materials and setup

| # | requirement (wording) | implementation | evidence | measured | status | remaining uncertainty |
|---|---|---|---|---|---|---|
| R1 | "Hardware: rented Ampere-or-newer GPU … everything capped at 8 GB GPU memory · Host RAM ≥16 GB" | — | FS `state.json` preflight (`cuda_bf16`, `host_ram`), `t13.provenance` | RTX 3060 12 GB sm_86; host MemTotal 135,040,708,608 B | SATISFIED | — |
| R2 | "Target: Qwen/Qwen3-4B in non-thinking mode" | `qwen_grpo/sentiment.py:23-32` (`enable_thinking=False`, runtime assert on the empty `<think></think>` suffix) | RA `config.yaml`; T13 `identity.model_revisions` 1cfa9a72… | assert held for every prompt in both runs and T13 | SATISFIED | the 4B template was not re-rendered offline (INFERENCE from the assert) |
| R3 | "No SFT stage" | training starts from the base checkpoint; step0 B = 0 | RA `checkpoints/step0` | — | SATISFIED | — |
| R4 | "Train it directly with GRPO + LoRA" | `qwen_grpo/grpo.py`, `training.py`, `trainer.py`; PEFT LoRA `model.py:71-73` | RA | 150 updates; ‖Δθ‖ 1.9963 | SATISFIED | see R5 for the reward |
| R5 | "automatic, verifiable reward … 1–2k prompts, in a language you can evaluate" | exact-match label reward `sentiment.py:35-42`; 1024 balanced train prompts | RA `training.json: config.train_ids`; T13 `data_separation` | 1024 prompts, 3360 selections | SATISFIED | — |
| R6 | "Pick a reward that can move within a few dozen steps and justify it" | — | RESULTS_ANALYSIS §1.3, §2 | rollout reward flat (per-50-attempt bins 0.856–0.916); 77.7% zero-signal batches; base test accuracy 0.877 | **PARTIALLY SATISFIED** | the chosen task left little headroom on 4B; the report must justify the choice honestly (it worked on 0.6B: +6.4 pp) |
| R7 | "Stack: Soup, or another one if you explain why" | own stack | README | — | PARTIALLY SATISFIED | the explanation must be in the report |

## Hard constraint: the 8 GB cap

| # | requirement | implementation | evidence | measured | status | uncertainty |
|---|---|---|---|---|---|---|
| C1 | "Training … must … stay within 8 GB … as shown by nvidia-smi (CUDA context included)" | external supervisor `scripts/guarded_run.py` (raw nvidia-smi every 200 ms, cap 8e9 B); in-process sampler `qwen_grpo/execution/profiler.py` (0.5 s); guard `trainer.py:421-436` | MAN `train_…/guarded_run.json` + `telemetry/*.csv`; RA `training/memory/20261119T054525Z.json`; RC `…/20261119T064535Z.json`; MAN `control_…` | external: per-process peak 2732 MiB over 16,643 samples, max gap 0.332 s; in-process: 2,864,709,632 B (35.8% of 8e9), 5947 samples, 0 missing; 0 violations; control the same peak | **SATISFIED (sampled)** | transients shorter than 200 ms are not seen (a sampled peak is a lower bound); no phase labels in the training memory file |
| C2 | "rollout generation … within 8 GB" | rollouts happen inside the training process | same files; T11 rollout phases | covered by C1; profile rollout phases 1,937,768,448 and 1,960,837,120 B per process | SATISFIED (sampled) | — |
| C3 | "the merge … within 8 GB" | `export/merge.py:121-153`: CPU-only, tensor by tensor, lazy `safe_open`; run as its own process (`scripts/merge.py`) | MAN `merge_20261119T070347Z/guarded_run.json`; T13 `merged_dir.provenance` | step-150 adapter; 32 s; no CUDA context (0 compute-app rows over 159 samples; device 13 MiB); CPU max RSS 8,602,054,656 B | **SATISFIED** | — |
| C4 | "the evaluation of the merged model must … stay within 8 GB" | `scripts/final_eval.py --merged-dir` (merged model through `load_merged_l2l`, `final_eval.py:70-80,162-177`) | MAN `eval_20261119T070439Z/guarded_run.json` + `telemetry/nvidia_smi_apps.csv`; T13 `phases` | whole eval process (4 models) per-process peak 1912 MiB over 7295 samples, 0 violations; merged-pass window 07:17:24–07:22:54Z also 1912 MiB (auditor's computation from the CSV) | **SATISFIED (sampled)** | the planned per-model `phase_memory.json` slice was not written; the merged-window figure is the auditor's |
| C5 | "If you rent a card with more memory, enforce the cap yourself and verify it: an allocator limit is not a cap until nvidia-smi confirms it" | in-process guard budget = 8,000,000,000 B (`memory.budget_bytes`; `trainer.py:421-436`, allocator fraction 8e9 / card); external supervisor with the same cap around every stage | FS preflight (`memory.budget_bytes` 8000000000, `memory.guard` true); MAN `*/guarded_run.json` (`cap_bytes` 8000000000) | all stages exit 0, violation None; usage ≤ 36% of the cap | **SATISFIED with disclosure** | two guard-enforcement unit tests failed on the Soup host (`tests/test_memory_guard.py::test_a_sample_above_the_budget_stops_the_run`, `tests/test_profiler.py::test_the_guard_records_a_violation_and_interrupts_the_main_thread`); they pass locally. Stop-on-violation by the in-process guard is therefore not demonstrated on that host. The external supervisor's tests (`tests/test_guarded_run.py`, 10) passed there |

## Task 1: memory and time budget

| # | requirement | implementation | evidence | measured | status | uncertainty |
|---|---|---|---|---|---|---|
| B1 | "Estimate GPU and CPU-RAM use for the GRPO step and for the merge: weights, per-layer buffers, activation checkpoints, logits (~151k vocab), rollout KV cache, reference log-probs, LoRA and optimizer states, the drafter, and the peak during the merge" (before running) | `qwen_grpo/budget/calculator.py`, `scripts/budget.py` | `results/budget/predictions/qwen3-4b-final_8gb.md` (created 2026-11-18T13:58:44Z, before any Qwen3-4B measurement); `qwen3-4b-final-dflash_8gb.md` sizes the drafter | per-phase table incl. merge (CPU 2.141 GiB) | SATISFIED | — |
| B2 | "Decide how many layers stay resident … and justify it" | config `resident_layers: 0` | prediction file "recommended: 20" | 0 chosen | **PARTIALLY SATISFIED** | the calculator recommends 20 (6.56 GiB, 56% fewer bytes per sweep); the report must justify 0 (e.g. "maximal streaming demonstrates the mechanism", "simulate a smaller card") |
| B3 | "estimate step time from PCIe bandwidth for both training and generation" | calculator PCIe table | same file | ≥ 8.65 s per step at an assumed 12 GB/s | SATISFIED | — |
| B4 | "compare with the actual peak (allocated, reserved, nvidia-smi) and measured time, and explain the gaps" | `scripts/budget_residuals.py` | `results/budget/residuals/qwen3-4b-final_150_v2.md` | per-phase residuals with notes (measured on the 60-update run and its profile: whole-run peak 2.67 GiB vs analytical 2.43 GiB; step 8.54 s vs a transfer bound of 8.48 s at 12.23 GB/s); 150-update forecast 665 attempts / 0.83 h vs actual 672 / 0.89 h; the 150-update run's peak (2,864,709,632 B) equals the 60-update run's | **PARTIALLY SATISFIED** | the residuals are computed on the earlier run; the 150-update run's steps are ~1 s slower (mean 9.55 s), and the new profile's bandwidth (7.25 GB/s) is inconsistent with its step times (14 sweeps would take 14.03 s). Neither is explained |

## Task 2a: L2L

| # | requirement | implementation | evidence | measured | status | uncertainty |
|---|---|---|---|---|---|---|
| L1 | "Implement them yourself, do not just import a library" | `qwen_grpo/execution/{weight_store,layer_stream,prefetch,backward,generator,output_head}.py` | code | — | SATISFIED | uses HF/PEFT layer modules unchanged, which is allowed |
| L2 | "Weights live in pinned CPU memory" | `weight_store.py:124-150` (`cudaHostRegister` slabs, no pageable fallback) | T11 `pinned_bytes = 7,267,018,752` | — | SATISFIED | — |
| L3 | "only the current (plus prefetched) layer is on the GPU" | 2 fixed buffers `weight_store.py:248-250`; `layer_stream.py:206-230`; bytes copied per step = 13–14 × 7.267e9 | RA `training.json: timings` (94,471,243,776 or 101,738,262,528 B per step); allocated peak 1,806,041,088 B, well below the 8.04e9 B of weights | — | **SATISFIED with disclosure** | the tied embedding/LM head (777,912,320 B), final norm and LoRA stay resident (`layer_stream.py:165-169`) |
| L4 | "activations are checkpointed at layer boundaries" | `backward.py:407-424` (saves the layer input; recomputes with re-streamed weights; checksum) | H02 | — | SATISFIED | the final config computes gradients only through layers 34–35 |
| L5 | "copy and compute overlap on a separate CUDA stream" | `prefetch.py:70,108-124,188-204` (copy stream, ready/free events) | none measuring overlap | mean step 9.55 s; transfer-bound by INFERENCE (residuals file, 60-update run: 8.54 s vs 8.48 s bound) | **PARTIALLY SATISFIED** (implemented; overlap **not demonstrated**) | H06 not tested; no event timeline; the profile's 7.25 GB/s bandwidth figure contradicts the step times |
| L6 | "The same applies to rollout generation and evaluation" | `StreamedGenerator` (`generator.py:87-90`) for rollouts, greedy eval and scoring | `training.json: rollout_backend`; T13 `backends` all `l2l_async` | — | SATISFIED | — |
| L7 | "Prove equivalence: gradients and logits match a reference non-L2L run on a small config within a tolerance you justify" | `scripts/l2l_lab.py`, `qwen_grpo/{forward,backward,gen,update}_lab.py`; `research/l2l/tolerances.yaml` (justified bounds) | FS correctness; `evidence/20261119T05*Z_H0*`; `research/l2l/experiments/t08_*_r20261119T05*Z/`; `compare_qwen_resident_vs_qwen_l2l.json` | on the RTX 3060: logits and every layer 2690/2690, gradients 5296/5296, updates 6528/6528 (all bitwise), generation 1292/1292 within bounds | **SATISFIED** | tiny random models, async with `verify=True`. H01 now gates on the same-host comparison (rule changed in `5680e71`, before the run); the cross-host fixture drift (max 1.34e-7) is kept as a non-gating diagnostic |
| L8 | "Report how a GRPO step splits between generation and training, and why" | `training.json: timings` | RESULTS_ANALYSIS §6.1 | rollout 23.5%, old log-probs 36.8%, update 39.4% of timed steps | SATISFIED (data) | the explanation must be written: 3–4 vs 5 + 5 weight sweeps |

## Task 2b: streaming

| # | requirement | implementation | evidence | measured | status | uncertainty |
|---|---|---|---|---|---|---|
| S1 | "Prompts and rollout data are streamed from disk" | `qwen_grpo/streaming/` (`PromptStream`, rollout records) | RA `streaming/prompts/`, `streaming/rollouts/group_000001…672.pt` + `index.jsonl` | 672 rollout records | SATISFIED | — |
| S2 | "with deterministic resume: kill the job mid-epoch, resume, and show the same samples with no duplicates and no gaps" | `scripts/resume_kill_test.py` (external SIGKILL of the process group, resume in a new process); `trainer.py:72-112` crash points | `research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json` (+ `_raw/`); `runs/qwen3_4b_resume_kill_kill{A,B}/`; tiny CPU: `t09_resume_kill_tiny_cpu_r20261119T051046Z.json`, `t09_resume_tiny_cpu_r20261119T051532Z.json` | Qwen3-4B: B SIGKILLed after 3 step lines with checkpoints 0 and 2 committed, resumed, finished 6/6; equal to A on committed rows, rollout records, sample multiset (no gap, no duplicate), adapter + AdamW at every checkpoint, final torch RNG; `pass: true`; memory PASS for A, B1, B2 | **SATISFIED** | a short run (6 steps); per-token log-probs and gradients are not compared (not recorded by the trainer) |
| S3 | "The merge runs tensor-by-tensor or shard-by-shard, so the full model is never resident on the GPU" | `export/merge.py:121-153` (CPU, 1 GiB shards) | H09 tiny (`evidence/20261119T051649Z_H09.merge_tiny_cpu`); MAN `merge_…` | CPU only; no CUDA context | SATISFIED | — |
| S4 | "Show the peak" (of the merge) | guarded merge stage; T11 merge phase | MAN `merge_…/guarded_run.json`; T11 | real step-150 merge: GPU none, CPU max RSS 8,602,054,656 B; profile merge phase CPU RSS 17,511,149,568 B (inside a process that also holds the streamed model) | SATISFIED | — |

## Task 2c: DFlash rollouts

| # | requirement | implementation | evidence | measured | status | uncertainty |
|---|---|---|---|---|---|---|
| D1 | "Use the drafter z-lab/Qwen3-4B-DFlash-b16 (~1 GB, kept resident) … integrate the speculative loop into your own L2L engine" | `qwen_grpo/execution/dflash.py` (`DFlashDrafter`, `dflash_generate`) | `tests/test_dflash.py` (CPU, tiny/oracle drafters) | — | **NOT SATISFIED (implemented, never run on the real model)** | intentionally excluded (`a1d59ea`; `dflash.enabled: false` checked by FS preflight) |
| D2 | "Report acceptance length and GRPO step time with and without it" | `scripts/dflash_bench.py` | none | — | NOT TESTED | — |
| D3 | "Show that rollouts stay on-policy: verifier runs the current LoRA weights, greedy output token-identical to plain decoding, and … a distribution check" | dflash tests | CPU tiny only | — | NOT TESTED on the real model | — |
| D4 | "Measure how acceptance changes as the policy drifts" | bench `--adapters` | none | — | NOT TESTED | — |
| D5 | "If DFlash gives no speedup in your setup, show why" | — | INFERENCE available: rollouts are 23.5% of a timed step and ≤ 4 sweeps, so even perfect drafting saves at most ~2–3 of 14 sweeps (≤ ~20%); the batch row-min acceptance (`docs/l2l/12_dflash.md:60-64`) reduces it further | — | **PARTIALLY addressable from existing data** (an Amdahl bound, not a measurement) | — |

## Task 3: prove the model trained

| # | requirement | implementation | evidence | measured | status | uncertainty |
|---|---|---|---|---|---|---|
| V0 | "Why doesn't a rising reward prove learning … (GRPO loss sits near 0 by construction …)" | — | RESULTS_ANALYSIS §1.1 (derivation) | logged loss \|L\| ≤ 3.2e-8 | SATISFIED (explainable) | must be written |
| V1 | "Write a check that tells a truly trained policy from a run that finished without meaningfully changing it, or one that only learned to game the reward" | `scripts/final_eval.py`, `qwen_grpo/evaluation.py`, `qwen_grpo/verification.py` | T13 | — | SATISFIED (check exists and ran) | — |
| V2 | "Compare the base, the adapter-attached policy, the merged model … and a negative control (… lr=0)" | `final_eval.py` (four models, same prompts, decoding and backend) | T13 | four models on 1066 identical prompts | SATISFIED | control is equal to base by construction and ran 30 updates, not "the same run" |
| V3 | "merge must match the adapter up to numerical noise" | — | T13 | predictions 1066/1066; scores \|Δ\| median 9.5e-7, max 0.375 nats; AUC diff +8.8e-6, CI [−2.8e-4, +3.1e-4]; every adapted tensor within half an ulp (max residual 8.8e-4) | SATISFIED | no pre-declared logit tolerance; `all_within_half_ulp` is a per-tensor rounding check only |
| V4 | "Use a held-out set and confidence intervals" | paired bootstrap `evaluation.py:86-92,173-189` | T13 | trained − base accuracy +0.938 pp [−0.375, +2.251]; AUC +0.0012 [+0.0002, +0.0022] | SATISFIED (method) | accuracy not significant (H12 NOT SUPPORTED); AUC gain significant but small and not the H12 criterion |
| V5 | "Run it, include the output" | — | T13 JSON; MAN `eval_20261119T070439Z/{stdout,stderr}.log`, `guarded_run.json: argv` | — | SATISFIED | — |
| V6 | "state what the check would NOT detect" | `tests/silent_failures/`; `docs/SOUP_CHECK_COVERAGE.md` (last column) | — | — | PARTIALLY SATISFIED | must be written for the final report (e.g. bias shift vs discrimination, base-model contamination, cross-host reproducibility) |

## Task 4: silent failures

| # | requirement | implementation | evidence | status | uncertainty |
|---|---|---|---|---|---|
| F1 | "Find … what … can break without a single error. For each: what could go wrong, how you checked … what evidence supports your conclusion" | `tests/silent_failures/`, runtime asserts (template, grouping, ratio = 1, frozen digest, backward checksum, buffer ownership), fault injection | FS correctness: fault injections refused 66/66 (forward) and 80/80 (gradient) with `BufferReuseError`; CPU suite 341 passed, 2 failed, 17 skipped (`FS/correctness/20261119T050522Z/pytest_cpu/stdout.log`) | **PARTIALLY SATISFIED** | the two failures are the guard-enforcement tests (C5); the write-up per failure mode is still to be done |
| F2 | "whether Soup's built-in checks (data doctor, pre-flight, soup ship) catch it" | `qwen_grpo/integrations/soup.py`, `scripts/soup_checks.py` (restored in `5680e71`) | `research/soup/0.78.0_20261119T073222Z/summary.json` and logs; first attempt `research/soup/unavailable_20261119T073027Z/`; older CPU evidence `research/soup/0.77.2_cpu/` | **PARTIALLY SATISFIED** | ran: reward stress (real reward 0/4600 attacks accepted; permissive positive control: answer spray 800/800, detected); `soup ship` on the trained model and the control (both DON'T SHIP, `missing_baseline`). Did not run: `data doctor` (7/7 exit 1, needs `soup-cli[train]`), `adapters check` (4/4 exit 1, numpy missing). `doctor` ran in a different interpreter from training |

## Task 5 and deliverables

| # | requirement | status | note |
|---|---|---|---|
| V | "One call, SHIP or DON'T SHIP … If DON'T SHIP, state exactly what must change first" | SATISFIED | **DON'T SHIP** with four concrete prerequisites, report §5 (`reports/Soup_Final_Research_Report.pdf`) |
| X1 | "Report (≤3 pages)" | SATISFIED | `reports/Soup_Final_Research_Report.pdf`: 3 A4 pages (pdfinfo), five assignment sections; numbers checked by `reports/verify_report_numbers.py` (93/93); evidence index `reports/REPORT_EVIDENCE_MAP.md` |
| X2 | "raw timestamped logs" | SATISFIED | RA/RC `training/training.log` (UTC timestamps); MAN `*/stdout.log`, `stderr.log`; FS `correctness/*/` logs |
| X3 | "raw nvidia-smi output sampled during the whole run" | SATISFIED | MAN `*/telemetry/nvidia_smi_{apps,gpu}.csv` for smoke, profile, train, control, merge and eval (200 ms); raw `smi_*_raw` strings in every RA/RC memory sample |
| X4 | "the verification script from part 3" | SATISFIED | `scripts/final_eval.py` (+ `qwen_grpo/evaluation.py`) |
| X5 | "the L2L equivalence test" | SATISFIED | `scripts/l2l_lab.py`, `qwen_grpo/*_lab.py`, `tests/test_layer_stream.py`, `test_backward.py`, `test_prefetch.py`, `tests/test_h01_same_host.py` |
| X6 | "the finished notebook or code" | SATISFIED | repository at `29852d1` |
| X7 | "one paragraph on what surprised you and what still concerns you" | SATISFIED | report p. 3, "What surprised me, what still concerns me" |
| X8 | AI tools note: "where you used them, which suggestions you accepted, and what you verified, changed, or redid yourself" | SATISFIED | report p. 3, "Provenance and AI tools" |

## Roll-up

| area | status |
|---|---|
| Materials (model, GRPO + LoRA, reward, data) | SATISFIED; R6 partial |
| 8 GB cap | training, rollout, merge and merged-model evaluation SATISFIED (sampled, 200 ms); guard at exactly 8e9 B; in-process stop-on-violation tests failed on the Soup host |
| Budget | mostly SATISFIED; residency justification outstanding; the 150-update step-time and bandwidth gaps unexplained |
| L2L | implemented and equivalent on the RTX 3060 (tiny async, 2690/5296/6528/1292) and on 0.6B sync; **overlap not demonstrated** |
| Streaming | from disk SATISFIED; real Qwen3-4B kill/resume SATISFIED |
| **DFlash** | **NOT SATISFIED** (never run on the real model) |
| Proving training | method SATISFIED; result: **accuracy gain not significant; small significant AUC gain** |
| Silent failures / Soup | partial; Soup reward stress and ship ran, data doctor and adapter checks did not start |
| Verdict and report | drafted, not audited here |

**Every mandatory requirement satisfied? No.** DFlash (2c) and measured overlap (L5) are unmet, and the Soup checks (F2) ran only in part. The memory evidence (C3, C4, C5) and real resume (S2) that were open in the 60-update audit are now closed. The assignment explicitly allows reporting unfinished parts, so the open items must be stated as such, not claimed.
