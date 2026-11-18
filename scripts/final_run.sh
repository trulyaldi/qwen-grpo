#!/usr/bin/env bash
# DEPRECATED: the final run is now one stage of the complete protocol in scripts/run_assignment.sh (stage F, gated on
# the correctness and 8 GB qualification stages, followed by the held-out evaluation and the report). This wrapper
# only forwards; it no longer commits anything.
#   bash scripts/final_run.sh [--resume | --from-stage F | --dry-run | --status]
echo "scripts/final_run.sh is deprecated: running scripts/run_assignment.sh $*" >&2
exec bash "$(dirname "$0")/run_assignment.sh" "$@"
