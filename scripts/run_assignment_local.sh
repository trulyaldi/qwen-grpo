#!/usr/bin/env bash
# LOCAL laptop run on Qwen3-0.6B and a 4 GB GPU (RTX 3050 Ti). Default: the real GRPO + LoRA training through async
# L2L (pinned CPU weights, 2 buffers, prefetch 1, activation checkpointing) with disk streaming and the memory guard.
#
#   bash scripts/run_assignment_local.sh                 new run in experiment mode (the default): A preflight,
#                                                        F 2-step smoke then 10-step training, report
#   bash scripts/run_assignment_local.sh --experiment    the same, named explicitly
#   bash scripts/run_assignment_local.sh --full-tests    new run with the full correctness protocol (stages A B C D F)
#   bash scripts/run_assignment_local.sh --fresh         new run that first moves the existing local training runs
#                                                        aside (runs/<name>.archived-<UTC>, never deleted) to retrain
#   bash scripts/run_assignment_local.sh --dry-run       what would run here; downloads, trains and writes nothing
#   bash scripts/run_assignment_local.sh --resume        continue the latest local run in its own mode (same code/configs)
#   bash scripts/run_assignment_local.sh --experiment --resume   continue the latest experiment-mode run only
#   bash scripts/run_assignment_local.sh --status        stage table and the active experiment of the latest local run
#   bash scripts/run_assignment_local.sh --from-stage F  re-run stage F onward in the latest local run
#
# Experiment mode does not re-run the correctness suite: earlier --full-tests evidence is recorded only when its code
# identity equals the current one, else NOT_TESTED (STALE). DFlash is never enabled; nothing here verifies Qwen3-4B.
# Existing completed runs with the same code, config and GPU are reused (the report says REUSED PRIOR EVIDENCE - NO NEW
# TRAINING EXECUTED); with other code they are INCONCLUSIVE until moved aside with --fresh.
# Exit: 0 = PASS; 1 = report written, not PASS; 2 = usage or resume refused; 130 = interrupted. Never commits/pushes.
# The mode (experiment = "fast" in state.json / full) is recorded in each run; --status and --resume read it there.
cd "$(dirname "$0")/.." || exit 2
PY="${PYTHON:-python}"
FULL=0 EXP=0 ARGS=() CONTINUES=0
for a in "$@"; do
  case "$a" in
    --full-tests) FULL=1 ;;
    --experiment) EXP=1 ;;
    --resume|--from-stage) CONTINUES=1; ARGS+=("$a") ;;
    *) ARGS+=("$a") ;;
  esac
done
[ "$FULL$EXP" = 11 ] && { echo "--experiment and --full-tests are exclusive" >&2; exit 2; }
if [ "$CONTINUES" = 1 ]; then  # a continued run keeps the mode it was started in; a mode flag selects runs of that mode
  WANT=$([ "$FULL" = 1 ] && echo full || { [ "$EXP" = 1 ] && echo fast; } || true)
  MODE="$("$PY" scripts/assignment_local.py mode ${WANT:+--want "$WANT"})" || exit 2
else
  MODE=$([ "$FULL" = 1 ] && echo full || echo fast)
fi
export ASSIGNMENT_RESUME_FLAG=$([ "$MODE" = full ] && echo --full-tests || echo --experiment)
export ASSIGNMENT_LOCAL_MODE="$MODE"
export ASSIGNMENT_RUNNER=scripts/assignment_local.py
export ASSIGNMENT_STAGE_LIST=$([ "$MODE" = full ] && echo "A B C D F" || echo "A F")
export ASSIGNMENT_ENTRYPOINT=scripts/run_assignment_local.sh
exec bash scripts/run_assignment.sh "${ARGS[@]}"
