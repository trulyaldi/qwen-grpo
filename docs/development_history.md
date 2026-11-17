# Development history

How the system was built, for readers who need the reasoning behind a component. Nothing here is needed to operate
the project: start from `README.md`, `docs/configuration.md` and `docs/deployment.md`.

## Before L2L: the resident GRPO + LoRA implementation

- **The resident implementation** (`qwen_grpo/grpo.py`, `training.py`, `trainer.py`) was audited line by line against
  the GRPO objective in `docs/correctness_audit.md`, including a comparison with TRL's GRPOTrainer. The TRL reference
  was removed in `4424e0f`; recover it with `git show 4424e0f^:research/trl_reference/findings.md`.
- **Line references.** That audit cites paths and line numbers from before the `qwen_grpo/` + `research/`
  reorganisation. Read it against commit `a038e17`.
- **The older audit.** `results/grpo_lora_audit.md`, with the POSITIVE-drift analysis, was removed in `06489d3`.
  Recover it with `git show 06489d3^:results/grpo_lora_audit.md`.
- **The reference run.** `runs/qwen` (Qwen3-0.6B, 100 steps, resident) is the reference result. Tests and docs read
  it, so it stays tracked.

## Tickets 00–13 (L2L, streaming, DFlash)

Each ticket's design notes, measurements and decisions:

| ticket | document | outcome |
|---|---|---|
| 00 | [`l2l/00_budget.md`](l2l/00_budget.md), [`l2l/resident_reference.md`](l2l/resident_reference.md), [`l2l/architecture.md`](l2l/architecture.md) | budget calculator and predictions (`results/budget/predictions/`), resident baseline and fixtures, audit of resident assumptions |
| 01 | [`l2l/01_forward.md`](l2l/01_forward.md) | explicit Qwen3 forward (owns the layer loop), bitwise equal to HF |
| 02 | [`l2l/02_weight_store.md`](l2l/02_weight_store.md) | pinned per-layer slabs (`cudaHostRegister`), GPU buffer pool |
| 03 | [`l2l/03_sync_forward.md`](l2l/03_sync_forward.md) | synchronous streamed forward (H01) |
| 04 | [`l2l/04_checkpointed_backward.md`](l2l/04_checkpointed_backward.md) | layer-boundary checkpoints, recompute in backward (H02) |
| 05 | [`l2l/05_output_head.md`](l2l/05_output_head.md) | chunked output head: H03 NOT_SUPPORTED by its rule; training keeps the exact full head |
| 06 | [`l2l/06_generator.md`](l2l/06_generator.md) | streamed prefill/decode with an explicit KV cache, HF sampling semantics (H04) |
| 07 | [`l2l/07_update.md`](l2l/07_update.md) | backends wired into the trainer; full update equivalence (H05); 0.6B runs resident vs L2L identical |
| 08 | [`l2l/08_prefetch.md`](l2l/08_prefetch.md) | asynchronous prefetch on a copy stream with event-based buffer ownership (H06; GPU evidence pending) |
| 09 | [`l2l/09_streaming.md`](l2l/09_streaming.md) | disk-backed prompts and rollouts, crash-consistent checkpoints, deterministic resume (H08) |
| 10 | [`l2l/10_merge.md`](l2l/10_merge.md) | tensor-by-tensor merge into sharded checkpoints, merged evaluation through L2L (H09) |
| 11 | [`l2l/11_profiling.md`](l2l/11_profiling.md) | profiler with nvidia-smi sampling and a memory guard; depth residency (H07); 4B budget (H11) |
| 12 | [`l2l/12_dflash.md`](l2l/12_dflash.md) | DFlash drafter re-implemented from the official code; exact verification by the LoRA policy (H10) |
| 13 | [`l2l/13_final.md`](l2l/13_final.md) | held-out evaluation, LR=0 control, evidence-derived final report |

Raw results are in `research/l2l/experiments/`; per-ticket command and test logs are named `ticketNN_*.log`. The
hypotheses, protocol and tolerances are in `research/l2l/` (`hypotheses.yaml`, `protocol.md`, `tolerances.yaml`); the
commands that test each hypothesis are in `configs/studies/assignment.yaml` (run by `scripts/experiment.py`;
the earlier `research/l2l/runbook.yaml` was folded into it).

**Notes in the ticket documents that are now out of date:**
- "training refuses `l2l_sync` / `l2l_async`" (03, 07);
- "DFlash is not implemented" (00).

**The refactor after Ticket 13** added:
- configuration profiles and the `memory` section;
- the `qwen-grpo` CLI with preflight, Soup diagnostics and the silent-failure registry (all removed again in the
  cleanup: `docs/cleanup_report.md`; git history keeps them, e.g. commit 20e7315).

See `git log` from `c123f7e`.

## Runs from before `runs/`

The earliest runs used `checkpoints/<run>_step<N>/`, `results/<run>/`, `results/benchmarks/<run>.{json,md}`,
`results/eval/` and `results/verification/`. Of these, only the shared base-model baseline is still in the
repository:
- `results/baseline/{validation,test}.json`, written by `python -m research.baseline --split validation` and read by
  `verify`;
- `results/benchmarks/baseline.{json,md}`.

The code still accepts the old layout (`scripts/run.py <cmd> --config <yaml> --checkpoint checkpoints/<run>_step<N>`),
and such runs cannot be resumed into `runs/`.

`research/train_grpo.py` and `research/train_lora.py` were the Stage 1–2 single-step scripts, imported by nothing; they
were removed in the cleanup (git history: commit 20e7315). `research/baseline.py` still produces `results/baseline/`.
