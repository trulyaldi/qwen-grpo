# Soup's built-in checks: what they are, what they catch, what they miss

The assignment allows "Soup or your own stack". This project trains with its own GRPO + L2L engine. Soup is used only
as an external CLI that audits this project's artifacts. It never trains, generates, merges or loads Qwen3-4B for this
project: its trainer and `soup ship`'s live mode would load the model outside the 8 GB L2L engine.

## Which Soup, and how it was established

- **Package:** PyPI `soup-cli`, repository https://github.com/MakazhanAlpamys/Soup, entry point `soup = soup_cli.cli:run`.
- **0.77.2:** installed and **executed** on 2026-11-17 (CPU, this workstation). The raw evidence was archived at the
  cleanup and is now restored byte-identical (27 files, sha256 checked) in `research/soup/0.77.2_cpu/`
  (`summary.json` + one log and JSON report per call).
- **0.78.0** (current on 2026-11-19): the wheel was downloaded and its source read **statically**. Every command and
  option `scripts/soup_checks.py` uses exists in it (`soup_cli/cli.py`, `commands/{data_doctor,adapters,reward,ship,
  advise,plan,train}.py`). It was **not executed**: installing it on this host was not permitted in this session. So
  0.78.0 behaviour is unverified; the findings below are 0.77.2's.
- **Setup (own venv; never touches the training venv):**
  `python3 -m venv .soup-venv && .soup-venv/bin/pip install soup-cli==0.78.0 && export SOUP_BIN=$PWD/.soup-venv/bin/soup`
- **Run:** `python scripts/soup_checks.py` (or `scripts/final150.py soup`). It runs with `--no-telemetry`,
  `SOUP_NO_AUDIT_LOG=1` and HF offline. Inputs are written into the new evidence directory `research/soup/<ver>_<UTC>/`;
  `data/` is never modified and nothing is uploaded.

## The three checks the assignment names, mapped to real commands

| assignment term | real command (0.77.2 executed / 0.78.0 present) | what it actually does | applicable here? |
|---|---|---|---|
| **data doctor** | `soup data doctor <jsonl> --model <tok> --max-length N --sample K -o r.json` | Renders SFT chat rows through the tokenizer's template. Checks: `chat_template`, `template_render`, `generation_markers`, `eos_in_labels`, `bos_duplication`, `system_role`, `unknown_roles`, `truncation_risk`. Overall OK / MINOR / … | Partly. It sees our prompt + gold label as an SFT row, not our GRPO prompt (which adds the empty non-thinking block at generation time) and not our completion masks |
| **pre-flight** | `soup advise run <data> --goal …` (fine-tune vs RAG decision), `soup plan` (training plan from a `soup.yaml`: cost, ETA, VRAM), the hardware-fit pre-flight inside `soup train` | Plans **Soup's own trainer** (model, quantization, sharding, data length) | **No.** There is no `soup.yaml`, because the trainer is not Soup's. Recorded as NOT_RUN with the reason; the `--help` output is kept as evidence of what it checks. It cannot predict the L2L engine's memory: streamed weights, two buffers, a checkpointed backward |
| **soup ship** | `soup ship --evidence ev.json -o verdict.json` (offline: scores supplied; exit 0 SHIP / 2 DON'T SHIP) | Leg 1: task win (tuned > base). Leg 2: no general benchmark regressed by more than 5%. Live mode loads base and tuned models with Soup's inference (not used) | Yes in offline mode, with our real held-out accuracies |
| (also) `soup adapters check <dir> --json` | Reads LoRA tensors: zero B, Frobenius norm, non-finite values, shape mismatches, orphaned keys; verdict alive / inactive | Yes, on our PEFT checkpoints |
| (also) `soup reward stress <verifier.py> --references refs.jsonl` | Attacks a reward function (sentinel, wrapped junk, answer spray) and measures gameability; exit 0 robust / 2 gameable | Yes, on our exact reward function |
| (also) `soup doctor` | Soup's own interpreter: GPU, dependency versions | Describes Soup's venv, not the training venv |

## Observed behaviour (0.77.2, executed; `research/soup/0.77.2_cpu/summary.json`)

| check | input | Soup's output | detected the planted problem? |
|---|---|---|---|
| data doctor | 200 real prompts + labels (Qwen3 tokenizer) | MINOR: template has no `{% generation %}` markers; every other check OK; p95 81 tokens | n/a (clean) |
| data doctor | **fault:** 50 rows with no assistant turn | MINOR, **identical to clean**; `eos_in_labels`: "no assistant turns to check" → OK | **No.** Rows with nothing to train on pass silently |
| reward stress | the GRPO reward (`qwen_grpo.sentiment.reward`) | robust, gameability 0% (sentinel, wrapped junk, answer spray all rejected) | n/a (correct reward) |
| reward stress | **fault:** substring-match reward | GAMEABLE: answer spray 800/800 accepted; exit 2 | **Yes** |
| adapters check | step-0 adapter (B = 0) | `inactive: all lora_B layers are zero (6/6)`; exit 2 | **Yes.** An untrained adapter is flagged |
| adapters check | trained adapter (0.6B, step 90) | alive, total Frobenius 3.03 | n/a. "Alive" says the tensors changed, not that the policy learned |
| ship | reference run's numbers, no general suite | DON'T SHIP: "No general-suite benchmarks were measured" | refusal for missing leg-2 evidence |
| ship | same + a synthetic unchanged benchmark | SHIP | n/a |
| ship | **fault:** a gain of one example in 1066 (+0.09 pp) with the unchanged benchmark | **SHIP** | **No.** The offline verdict has no confidence interval: a noise-level difference "wins" |

`scripts/soup_checks.py` repeats these on the Qwen3-4B artifacts and adds:

- data-doctor fixtures: empty assistant turn, unknown role, rows longer than `--max-length` (32), a non-canonical
  label spelling (`positive.`), and the label leaked into the prompt;
- adapter checks on the 4B step-0, LR = 0 (step 30) and trained (step 60; step 150 when it exists) checkpoints;
- `soup ship` on the **real** 60-update held-out numbers. Base 0.8771 vs trained 0.8809: +0.38 pp, paired 95% CI
  [−0.56, +1.31] pp, McNemar p = 0.57. Our verdict on these numbers is "not significant".

What 0.77.2 implies for these new checks (a prediction, to be confirmed by the run):

- `soup ship` will say SHIP on the 4B numbers if a leg-2 suite is supplied, and DON'T SHIP without one, for a reason
  unrelated to significance;
- `adapters check` will flag the LR = 0 control as inactive.

## Silent-failure coverage matrix

Columns:

- **local check:** a test or runtime check in this repository that fails on the fault;
- **Soup check:** the Soup command that could see it;
- **Soup executed?:** E = executed (0.77.2), P = prepared in `soup_checks.py` (0.78.0, not yet executed), — = none
  applicable.

| stage | what can silently go wrong | local check (evidence) | Soup check | Soup executed? | Soup detects? | remains undetected |
|---|---|---|---|---|---|---|
| data | train/test overlap, duplicates, unbalanced subset | `tests/silent_failures/test_sf_pipeline.py::test_split_leakage_and_duplicates_are_reported`; `final_eval.py` `data_separation` (0 test texts in train) | none (data doctor checks formatting, not leakage) | P (`fault_label_leak`) | expected no | near-duplicates by paraphrase; pre-training exposure of the base model to Rotten Tomatoes |
| data | invalid or non-canonical labels (reward 0 everywhere) | `test_invalid_labels_are_reported` | data doctor (no label semantics) | P (`fault_noncanonical_label`) | expected no | — |
| data | rows with no or empty target | n/a for GRPO (targets are sampled); `test_sf_pipeline` checks labels | data doctor `eos_in_labels` | E (no-assistant), P (empty) | **no** (0.77.2) | — |
| tokenization | prompts truncated | `test_prompts_are_never_truncated`; budget uses the longest prompt (100 tokens) | data doctor `truncation_risk` | E (clean: OK), P (`fault_truncation`) | yes (by design) | — |
| chat template | thinking mode enabled; wrong template | `test_a_thinking_template_is_refused`; `sentiment.build_prompts` asserts non-thinking | data doctor `chat_template`, `template_render`, `system_role`, `unknown_roles` | E, P (`fault_unknown_role`) | template present / renderable only | Soup renders the SFT template, not our GRPO generation prompt with the empty `<think>` block |
| reward | gameable or permissive reward | `test_a_permissive_reward_is_caught` | `reward stress` | E | **yes** (GAMEABLE) | reward hacking through distribution shift that the attack set does not try |
| GRPO | misgrouped rollouts; zero-advantage batches trained on; old/new log-prob mismatch; wrong micro-batch scaling | `test_misgrouped_rollouts_are_refused`; `grpo_update` INIT_TOL assert; `tests/test_multistep.py` (micro-batch = full batch); `zero_signal` accounting | none | — | — | a correct implementation of a GRPO step that does not suit the task (77% zero-signal batches) |
| LoRA | adapter never trained or inactive; non-finite or missing tensors | `test_an_inactive_or_unchanged_adapter_is_caught`; `test_missing_extra_and_nonfinite_adapter_tensors_are_caught`; trainer `params_equal_init`, `frozen_changed` | `adapters check` | E (0.6B), P (4B incl. LR = 0) | **yes** (inactive); non-finite by design | "alive but not useful": changed tensors that do not improve the policy |
| L2L | wrong layer streamed; buffer overwritten while read; detached boundary; stale positions | lab fault cases `H01.fault_wrong_layer`, `H0{1,2}.fault_unsafe_reuse`, `H02.fault_detach_boundary`, `H04.fault_*`; `prefetch.BufferReuseError`; same-host bitwise equivalence (`tests/test_h01_same_host.py`) | none | — | — | a race outside the ownership rules that the `verify=False` performance path hides (H06 checks first-trial bitwise equality in that mode) |
| disk streaming | duplicated or skipped samples after a crash | `test_duplicate_or_skipped_stream_records_are_caught`; `scripts/resume_experiment.py` (in-process); `scripts/resume_kill_test.py` (external SIGKILL: PASS, `t09_resume_kill_tiny_cpu_r20261119T040414Z.json`) | none | — | — | kill/resume on Qwen3-4B (prepared, GPU) |
| checkpointing | partial checkpoint used; optimizer or RNG state not restored | crash-consistent publication (`streaming/resume.py`); `compare_runs.py` (LoRA + AdamW bitwise); kill test (torch RNG, Python RNG restore) | `adapters check` (adapter tensors only) | P | adapter readable only | — |
| merge | scale dropped; tensors not merged; merged ≠ adapter | `tests/silent_failures/test_sf_engine.py::test_a_dropped_merge_scale_is_detected`; merge half-ulp self-check; `final_eval.py` merged-vs-trained agreement and score diffs | none run (`soup merge` / `adapters merge` are Soup's own, not used) | — | — | bf16 rounding-scale differences: judged by agreement and score diffs, without a pre-declared tolerance |
| inference | evaluation silently uses another backend, batch shape or decoding; an improvement within noise | `test_ignored_execution_settings_are_refused`; `final_eval.py` (same prompts, backend, decoding for all four models; paired bootstrap CI; LR = 0 control) | `ship` (offline) | E (0.6B synthetic), P (4B real numbers) | **no** for noise-level gains (0.77.2: SHIP on +1/1066) | a policy change that is a uniform label-bias shift (AUC +0.0007 in the 60-update run) |

## What this means for "would `soup ship` have caught it?"

The failure this project cares most about is a run that finishes and looks better but is not:

- the 60-update run gained +0.38 pp on held-out data, with a CI that includes 0;
- the change was mostly a shift of the label threshold.

Soup's offline `ship` compares two point estimates. In 0.77.2 it shipped a one-in-1066 gain. 0.78.0 has a
`--noise-floor` option whose semantics were not verified here. The project's own check, paired bootstrap CIs plus the
LR = 0 control plus the label-score analysis, is what separates "trained" from "moved by noise".

`soup adapters check` does catch the cheapest silent failure: a run that finished with an unchanged adapter.
