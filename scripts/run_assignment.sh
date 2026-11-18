#!/usr/bin/env bash
# The whole Soup Hard Track v5 protocol in one command, on a rented native-Linux Ampere-or-newer GPU host (run it
# under tmux). Bash only sequences; Python runs and measures (scripts/assignment.py); configuration lives in YAML
# (configs/studies/assignment.yaml and the experiment configs it names).
#
#   bash scripts/run_assignment.sh                    new run: assignment_results/<UTC run-id>/
#   bash scripts/run_assignment.sh --dry-run          what would run here; downloads, trains and writes nothing
#   bash scripts/run_assignment.sh --resume           continue the latest run (same commit, diff and configs only)
#   bash scripts/run_assignment.sh --from-stage F     re-run stage F and later stages of the latest run (new attempts)
#   bash scripts/run_assignment.sh --status           stage table of the latest run
#
# Stages: A preflight, B correctness, C Qwen3-0.6B reference, D Qwen3-4B memory, E DFlash, F training,
# G evaluation, H audit. Then the report is always written, also after a failure or Ctrl-C.
# Exit: 0 = verdict SHIP; 1 = report written, verdict DON'T SHIP; 2 = usage or resume refused; 130 = interrupted.
# Never provisions or destroys machines, never commits, pushes or uploads.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; }

MODE=new STAGE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE=dry ;;
    --resume) MODE=resume ;;
    --from-stage) shift; STAGE="${1:-}"; MODE=from
                  case "$STAGE" in A|B|C|D|E|F|G|H) ;; *) echo "--from-stage needs one of A B C D E F G H" >&2; exit 2 ;; esac ;;
    --status) MODE=status ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$MODE" in
  dry) exec "$PY" scripts/assignment.py dry-run ;;
  status) exec "$PY" scripts/assignment.py status ;;
  new) RUN_DIR="$("$PY" scripts/assignment.py init)" || exit 2 ;;
  resume) RUN_DIR="$("$PY" scripts/assignment.py init --resume)" || exit 2 ;;
  from) RUN_DIR="$("$PY" scripts/assignment.py init --from-stage "$STAGE")" || exit 2 ;;
esac
[ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] || { echo "no run directory" >&2; exit 2; }

# one orchestrator per run directory
exec 9>"$RUN_DIR/.lock"
if ! flock -n 9; then echo "another run_assignment.sh holds $RUN_DIR" >&2; exit 2; fi

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

echo "assignment run $RUN_DIR ($MODE) at commit $(git rev-parse --short HEAD 2>/dev/null)"
for S in A B C D E F G H; do
  [ "$INTERRUPTED" = 1 ] && { echo "stage $S not started: interrupted"; continue; }
  "$PY" scripts/assignment.py stage "$S" --run-dir "$RUN_DIR" &
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
"$PY" scripts/assignment.py report --run-dir "$RUN_DIR"; RRC=$?
echo "results: $RUN_DIR"
[ "$INTERRUPTED" = 1 ] && { echo "interrupted: resume with  bash scripts/run_assignment.sh --resume"; exit 130; }
[ "$RRC" -eq 0 ] && exit 0
[ "$RRC" -eq 1 ] && exit 1
echo "report generation failed (exit $RRC)"; exit 1
