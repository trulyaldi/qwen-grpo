#!/usr/bin/env bash
# LOCAL laptop validation of the engineering the Qwen3-4B assignment reuses, on Qwen3-0.6B and a 4 GB GPU
# (RTX 3050 Ti): GRPO + LoRA, async L2L from pinned CPU weights with copy/compute overlap, activation checkpointing,
# disk-backed streaming, deterministic resume, memory/time profiling, resident-vs-L2L equivalence, silent failures.
#
#   bash scripts/run_assignment_local.sh              new run: assignment_results_local/<UTC run-id>/
#   bash scripts/run_assignment_local.sh --dry-run    what would run here; downloads, trains and writes nothing
#   bash scripts/run_assignment_local.sh --resume     continue the latest local run (same commit, diff and configs)
#   bash scripts/run_assignment_local.sh --status     stage table of the latest local run
#   bash scripts/run_assignment_local.sh --from-stage F   re-run stage F onward in the latest local run
#
# Stages: A preflight, B correctness, C Qwen3-0.6B equivalence, D profiling, F training (gated on B and C).
# DFlash is NOT tested (its drafter is for Qwen3-4B only) and nothing here verifies the Qwen3-4B assignment.
# Exit: 0 = local validation PASS; 1 = report written, not PASS; 2 = usage or resume refused; 130 = interrupted.
# Never commits, pushes or uploads. Same machinery as scripts/run_assignment.sh (this only selects the local runner).
cd "$(dirname "$0")/.." || exit 2
export ASSIGNMENT_RUNNER=scripts/assignment_local.py
export ASSIGNMENT_STAGE_LIST="A B C D F"
export ASSIGNMENT_ENTRYPOINT=scripts/run_assignment_local.sh
exec bash scripts/run_assignment.sh "$@"
