# Cleanup report (2026-11-17)

Branch `cleanup/assignment-research`. Nothing in the GRPO, LoRA, L2L, prefetching, streaming, DFlash or merge
algorithms was changed. No git history was rewritten.

## Starting point and restore points

| what | where |
|---|---|
| `refactor/config-cli-preflight` at `20e7315` (16 local commits after `3bc4cbe`: profiles, backend-aware validate/test, memory guard) | tag `restore/pre-cleanup-refactor` |
| `feat/l2l-streaming-research` at `10e61ba` (Kaggle workflow; pushed, checked out in the `qwen-grpo-kaggle` worktree, untouched) | tag `restore/pre-cleanup-feat` |
| the uncommitted work found in the tree (deletions of the CLI, preflight, silent-failure registry, Soup integration, four docs; an older README; a rewritten `tests/test_memory_guard.py`) | branch `backup/wip-refactor-2026-11-17` (exact snapshot); the test rewrite also as `stash@{0}` |

The cleanup branch starts from `20e7315`, merges `10e61ba` (one `.gitignore` conflict, both sides kept), and then
decides each uncommitted deletion on its merits (below). Baseline full CPU suite at the merge (`1f32767`): 754 passed,
1 failed (`test_sf_registry.py::[I3]`: the registry cited a test name that a later commit had renamed), 117 skipped.

No Kaggle job was running (all four submissions pinned `3bc4cbe`: cpu ×2 PASS_WITH_SKIPS, gpu-tests FAIL, gpu-short
PASS_WITH_SKIPS). Their evidence was archived, not modified.

## External archive (verified)

`/home/trulyaldi/qwen-grpo-archive/2026-11-17_pre-cleanup/` (outside the repository):

| archive | content |
|---|---|
| `runs_qwen.tar.gz` | `runs/qwen` exactly as tracked at `20e7315`: every checkpoint, every `training_state.pt` |
| `research_l2l_experiments.tar.gz` | every lab result as tracked at `20e7315` (uncompressed originals) |
| `research_soup.tar.gz` | the recorded soup_cli 0.77.2 outputs |
| `kaggle_outputs.tar.gz` | the four Kaggle submissions' downloaded evidence and the ledger |

`MANIFEST.sha256` (171 tracked files, from git objects) and `MANIFEST_kaggle.sha256` (114 files) were checked with
`sha256sum -c` after extracting every archive into a temporary directory: all OK. `PROVENANCE.md` there says how to
restore; git history (`git checkout 20e7315 -- runs/qwen`, or the pushed `10e61ba`) is the second copy.

## Evidence kept in the tree

- **Every lab result stays**, negative and fault-injection runs included (t05 v1 NOT_SUPPORTED, t07 cuda execution
  FAIL, every `--perturb` control). The 14 results over 1 MiB are stored as `result.json.gz`: lossless, written only
  after the decompressed bytes hashed to the original; `research/l2l/experiments` went from 98.1 MB to 9.8 MB.
  `research/l2l/evidence_index.json` records each output's sha256 (all 57 equal the archived originals), statuses,
  provenance and the hypotheses citing it. All 29 lab directories re-derive to their recorded statuses from the raw data.
- **`runs/qwen`** (75 MB → 7.6 MB) keeps the record of the 0.6B learning result (config, manifest, report, summary,
  plots, training log, validation and test evaluations) and the step-0 and step-90 adapters: step 90 is the input of
  the H09 real-model merge case (the H09 criterion names it) and both are used by the adapter fault-injection tests.
  Checkpoints 10–80 and 100 and every optimizer state are archived. `tests/test_historical.py` validates them from the
  archive (`QWEN_GRPO_ARCHIVE`), else those two tests skip with the reason: checked both ways (2 skipped / 11 passed).

## Removed

| file(s) | why | replacement / where it went |
|---|---|---|
| `qwen_grpo/cli.py`, `qwen_grpo/__main__.py`, pyproject `qwen-grpo` entry, `tests/test_cli.py` | a second entry point duplicating `scripts/run.py` | `scripts/run.py`; its `research` command → `scripts/experiment.py` |
| `qwen_grpo/diagnostics/preflight.py` | its GPU mode ran `profile_run.py` (the H11 experiment itself); budget estimates stay in `scripts/budget.py` | `experiment.py` probe: kernel launch, native bf16, VRAM, fixture package versions, native Linux, per-process nvidia-smi |
| `qwen_grpo/diagnostics/silent_failures.py`, `tests/silent_failures/test_sf_registry.py`, `docs/silent_failures.md` | a claims table tied to the CLI, preflight and Soup; it was already stale (the baseline's only failure) | the fault-injection tests stay (`test_sf_pipeline.py`, `test_sf_engine.py`); the audit table in `final_report.py` |
| `qwen_grpo/integrations/soup.py`, `research/soup/` | optional external-tool diagnostics, not part of the training or evidence pipeline | archived (`research_soup.tar.gz`) |
| `research/l2l/runbook.yaml`, `scripts/laptop_checks.sh` | superseded | `configs/studies/assignment.yaml` + `scripts/experiment.py` (every runbook command is an entry) |
| `scripts/l2l_baseline.py` | one-off generator of the frozen Ticket 00 snapshot (its outputs `research/l2l/baseline*` stay) | git history |
| `research/train_grpo.py`, `research/train_lora.py` | single-step diagnostics from before the trainer; imported by nothing | `scripts/l2l_real_step.py`, the trainer |
| `configs/default.yaml` | byte-for-byte the reference config under another name | `configs/qwen.yaml`; its Ticket 00 hash is still checked (`test_config.py`) |
| `configs/experiments/grpo_{32,1000,clip_k3,low_lr}.yaml`, `qwen3_4b.yaml` | old experiments with no runs or evidence; `qwen3_4b.yaml` superseded by `qwen3_4b_resident.yaml` | git history; the historical-schema test now uses an explicit fixture |
| `configs/gpu/qwen3_4b_l2l_async.yaml` | identical to `configs/experiments/qwen3_4b_l2l.yaml`, the file the H11 criterion names | `qwen3_4b_l2l.yaml` |
| `docs/status.md` | duplicated the README status, the hypotheses and the final report | README "Known limitations"; this report |

## Moved or renamed (resolved configs unchanged)

`configs/gpu/qwen3_0.6b_{resident,l2l_sync,l2l_async}.yaml`, `qwen3_4b_{l2l_sync,resident}.yaml` → `configs/experiments/`;
`qwen_{resident,l2l}{,_short}.yaml` → `configs/historical/` (the H05 evidence runs); `qwen_local.yaml` →
`qwen3_0.6b_streaming.yaml`, `qwen_stream_short.yaml` → `qwen3_0.6b_resume.yaml` (only `run.name` differs). Every config
that remains, the eight lab configs included, resolves to the same sha256 as before, so the config hashes recorded in
the lab evidence still match.

## Added

- `configs/experiments/kaggle_06b_fp32_*.yaml`: float32 diagnostics of the 0.6B backend comparison for the T4.
- `configs/studies/assignment.yaml`, `scripts/experiment.py` (`plan`, `run`, `report`, `index`), `qwen_grpo/evidence.py`,
  `tests/test_experiment.py`.
- `research/l2l/evidence_index.json`.

## Changed behaviour (each with a test)

- `scripts/run.py verify` follows the config's backend: resident → `verification.py`; L2L → `scripts/final_eval.py`
  through the run's backend, written to `runs/<run>/verification/step<N>/final_eval_test.json`; `--control <run>`.
  It used to refuse L2L runs.
- `scripts/final_eval.py`: `--step`, `--out`, and it refuses to overwrite an existing result (it used to overwrite the
  H12 evidence file silently).
- `scripts/compare_runs.py`: compares every recorded per-step field (29, was 12), rollout records of streaming runs,
  and checks that the configs differ only in run/execution/memory; states the predefined rule it applies
  (`tolerances.yaml` `update_equivalence`, all thresholds 0); reports runtime and memory in a separate section that
  never enters the verdict. On the recorded H05 pairs (`qwen_resident{,_short}` vs `qwen_l2l{,_short}`) it still finds
  them identical.
- `qwen_grpo/execution/fixtures.py`: a saved fixture without a recorded CPU no longer counts as matching this host.
  The saved Ticket 00 fixtures predate the host record, so on Kaggle (same packages, another CPU) four tests compared
  a CPU-bitwise reference across CPUs and failed (`gpu-tests`, commit `3bc4cbe`). The suite now builds fixtures
  in-session there, and the saved-artifact tests skip with the reason.
- `scripts/final_report.py`: `requirements()` is reused by the dispatcher; reads compressed results. All 15 requirement
  statuses are unchanged.
- `kaggle/run.py`: `gpu-short` trains the three committed fp32 configs and compares sync and async with resident; the
  config check covers the new layout; every mode records the study plan.
- `research/l2l/hypotheses.yaml`: one comment pointer (H08: `laptop_checks.sh` → the study entry). No status,
  criterion or evidence list changed. `tolerances.yaml` and `protocol.md` are unchanged.

## Retained although ticket-specific

The lab cases (`qwen_grpo/*_lab.py`, `lab.py`), `scripts/l2l_lab.py`, `l2l_real_step.py`, `resume_experiment.py`,
`profile_run.py`, `dflash_bench.py`, `final_eval.py`, `merge.py`, `budget.py` and the budget calculator: each is the
only implementation of a hypothesis test or of a capability. The per-ticket documents in `docs/l2l/` and
`docs/correctness_audit.md` are history and are not edited.
