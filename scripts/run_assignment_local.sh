#!/usr/bin/env bash
# LOCAL laptop run on Qwen3-0.6B and a 4 GB GPU (RTX 3050 Ti). Default: the real GRPO + LoRA training through async
# L2L (pinned CPU weights, 2 buffers, prefetch 1, activation checkpointing) with disk streaming and the memory guard.
#
#   bash scripts/run_assignment_local.sh               new run: A preflight, F 2-step smoke then 10-step training, report
#   bash scripts/run_assignment_local.sh --full-tests  new run with the full correctness protocol (stages A B C D F)
#   bash scripts/run_assignment_local.sh --dry-run     what would run here; downloads, trains and writes nothing
#   bash scripts/run_assignment_local.sh --resume      continue the latest local run in its own mode (same code/configs)
#   bash scripts/run_assignment_local.sh --status      stage table and the active experiment of the latest local run
#   bash scripts/run_assignment_local.sh --from-stage F   re-run stage F onward in the latest local run
#
# The default does not re-run the correctness suite: earlier --full-tests evidence is recorded only when its code
# identity equals the current one, else NOT_TESTED (STALE). DFlash is never enabled; nothing here verifies Qwen3-4B.
# Exit: 0 = PASS; 1 = report written, not PASS; 2 = usage or resume refused; 130 = interrupted. Never commits/pushes.
# The mode (fast default / full) is recorded in each run's state.json; --status and --resume read it from there.
# Default stages: A preflight, F training. --full-tests: A B C D F (training gated on B and C there).
cd "$(dirname "$0")/.." || exit 2
PY="${PYTHON:-python}"
FULL=0 ARGS=() CONTINUES=0
for a in "$@"; do
  case "$a" in
    --full-tests) FULL=1 ;;
    --resume|--from-stage) CONTINUES=1; ARGS+=("$a") ;;
    *) ARGS+=("$a") ;;
  esac
done
if [ "$CONTINUES" = 1 ]; then  # a continued run keeps the mode it was started in
  MODE="$("$PY" scripts/assignment_local.py mode)" || exit 2
  if [ "$FULL" = 1 ] && [ "$MODE" != full ]; then
    echo "--full-tests: the latest local run was started in $MODE mode; resume it without --full-tests, or start a new run" >&2
    exit 2
  fi
else
  MODE=$([ "$FULL" = 1 ] && echo full || echo fast)
fi
export ASSIGNMENT_LOCAL_MODE="$MODE"
export ASSIGNMENT_RUNNER=scripts/assignment_local.py
export ASSIGNMENT_STAGE_LIST=$([ "$MODE" = full ] && echo "A B C D F" || echo "A F")
export ASSIGNMENT_ENTRYPOINT=scripts/run_assignment_local.sh
exec bash scripts/run_assignment.sh "${ARGS[@]}"
