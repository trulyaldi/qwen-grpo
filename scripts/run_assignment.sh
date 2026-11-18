#!/usr/bin/env bash
# The whole Soup Hard Track v5 protocol in one command, on a rented native-Linux Ampere-or-newer GPU host (run it
# under tmux). Bash only sequences; Python runs and measures (scripts/assignment.py); configuration lives in YAML
# (configs/studies/assignment.yaml and the experiment configs it names).
#
#   bash scripts/run_assignment.sh                    new full qualification run: assignment_results/<UTC run-id>/
#   bash scripts/run_assignment.sh --experiment       new experiment run: essential checks, then the 4B experiments
#   bash scripts/run_assignment.sh --dry-run          what would run here; downloads, trains and writes nothing
#   bash scripts/run_assignment.sh --resume           continue the latest run of the mode (same commit, diff, configs)
#   bash scripts/run_assignment.sh --from-stage F     re-run stage F and later stages of the latest run (new attempts)
#   bash scripts/run_assignment.sh --status           stage table of the latest run
#   --experiment and --full-tests (= the default full mode) combine with each of the above; --fresh (new runs only)
#   moves existing fixed-name training runs / Qwen3-4B results aside (<path>.archived-<UTC>, never deleted).
#
# Full: A preflight, B correctness, C Qwen3-0.6B reference, D Qwen3-4B memory, E DFlash, F training, G evaluation,
# H audit. Experiment: A D B F G H without DFlash (no C, no E; a reduced B; plain-L2L final run and LR=0 control,
# qwen3_4b_final_plain*.yaml; scripts/assignment.py). Then the report is always written, also after a failure or
# Ctrl-C. An experiment run never gets SHIP and cannot satisfy the qualification.
# Exit: 0 = verdict SHIP; 1 = report written, not SHIP (always for --experiment); 2 = usage or resume refused;
# 130 = interrupted. Never provisions or destroys machines, never commits, pushes or uploads.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1
# scripts/run_assignment_local.sh reuses this script for the local laptop validation by setting these two
RUNNER="${ASSIGNMENT_RUNNER:-scripts/assignment.py}"
STAGE_LIST="${ASSIGNMENT_STAGE_LIST:-A B C D E F G H}"
ME="${ASSIGNMENT_ENTRYPOINT:-scripts/run_assignment.sh}"

usage() { sed -n '2,22p' "$ME" | sed 's/^# \{0,1\}//'; }

MODE=new STAGE="" FRESH=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE=dry ;;
    --resume) MODE=resume ;;
    --from-stage) shift; STAGE="${1:-}"; MODE=from
                  case " A B C D E F G H " in *" $STAGE "*) [ -n "$STAGE" ] ;; *) false ;; esac ||
                    { echo "--from-stage needs one of $STAGE_LIST" >&2; exit 2; } ;;
    --status) MODE=status ;;
    --experiment|--full-tests)  # the production runner's modes (the local wrapper consumes its own flags)
      [ "$RUNNER" = scripts/assignment.py ] || { echo "unknown argument: $1" >&2; exit 2; }
      WANT=$([ "$1" = --experiment ] && echo experiment || echo full)
      [ -z "${ASSIGNMENT_MODE:-}" ] || [ "$ASSIGNMENT_MODE" = "$WANT" ] ||
        { echo "--experiment and --full-tests are exclusive" >&2; exit 2; }
      export ASSIGNMENT_MODE="$WANT" ;;
    --fresh) FRESH=(--fresh) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[ "${ASSIGNMENT_MODE:-}" = experiment ] && ASSIGNMENT_RESUME_FLAG="${ASSIGNMENT_RESUME_FLAG:---experiment}"
if [ "${#FRESH[@]}" -gt 0 ] && [ "$MODE" != new ] && [ "$MODE" != dry ]; then
  echo "--fresh starts a new run; it cannot be combined with --resume, --from-stage or --status" >&2; exit 2
fi

case "$MODE" in
  dry) exec "$PY" "$RUNNER" dry-run ;;
  status) exec "$PY" "$RUNNER" status ;;
  new) RUN_DIR="$("$PY" "$RUNNER" init "${FRESH[@]}")" || exit 2 ;;
  resume) RUN_DIR="$("$PY" "$RUNNER" init --resume)" || exit 2 ;;
  from) RUN_DIR="$("$PY" "$RUNNER" init --from-stage "$STAGE")" || exit 2 ;;
esac
[ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] || { echo "no run directory" >&2; exit 2; }
# the stage order is the run's own (recorded mode), unless the local wrapper fixed it
[ -n "${ASSIGNMENT_STAGE_LIST:-}" ] || STAGE_LIST="$("$PY" "$RUNNER" order --run-dir "$RUN_DIR")" || exit 2

# one orchestrator per run directory
exec 9>"$RUN_DIR/.lock"
if ! flock -n 9; then echo "another $(basename "$ME") holds $RUN_DIR" >&2; exit 2; fi

mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/orchestrator.log"
# every line of this script's output and of the stage runners', UTC-stamped, to the terminal and the log
# The logger ignores Ctrl-C / SIGTERM / SIGHUP (it shares the terminal's process group): it must outlive the stages it
# records, and a dead pipe would break the report. tee keeps writing the log if the terminal goes away.
exec > >(trap '' INT TERM HUP
         while IFS= read -r line; do printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)" "$line"; done |
           tee -i --output-error=warn -a "$LOG") 2>&1

INTERRUPTED=0 CHILD=""
on_signal() {
  INTERRUPTED=1
  echo "received $1: stopping the current stage gracefully (its workload gets SIGINT, then SIGTERM/SIGKILL)"
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
}
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGHUP' HUP

echo "assignment run $RUN_DIR ($MODE) at commit $(git rev-parse --short HEAD 2>/dev/null), stages $STAGE_LIST"
for S in $STAGE_LIST; do
  [ "$INTERRUPTED" = 1 ] && { echo "stage $S not started: interrupted"; continue; }
  "$PY" "$RUNNER" stage "$S" --run-dir "$RUN_DIR" &
  CHILD=$!
  while :; do
    wait "$CHILD"; RC=$?
    kill -0 "$CHILD" 2>/dev/null || break   # wait returned because of a trapped signal: keep waiting for the child
  done
  CHILD=""
  if [ "$RC" -eq 130 ]; then INTERRUPTED=1; fi
  if [ "$RC" -ne 0 ] && [ "$RC" -ne 130 ]; then echo "stage $S runner exited $RC (recorded; later stages check their own gates)"; fi
done

echo "writing the report (always)"
"$PY" "$RUNNER" report --run-dir "$RUN_DIR"; RRC=$?
echo "results: $RUN_DIR"
[ "$INTERRUPTED" = 1 ] && { echo "interrupted: resume with  bash $ME${ASSIGNMENT_RESUME_FLAG:+ $ASSIGNMENT_RESUME_FLAG} --resume"; exit 130; }
[ "$RRC" -eq 0 ] && exit 0
[ "$RRC" -eq 1 ] && exit 1
echo "report generation failed (exit $RRC)"; exit 1
