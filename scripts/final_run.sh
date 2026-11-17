#!/usr/bin/env bash
# Ticket 13 on the GPU host: the final Qwen3-4B GRPO run (hours; resumable), the LR=0 negative control, the held-out
# test evaluation of base / trained / merged / control through L2L, and the reports regenerated from all evidence.
# Every step goes through the study manifest (configs/studies/assignment.yaml): an entry whose prerequisite run is
# missing is BLOCKED, completed runs are skipped, and nothing is overwritten. The Qwen3-4B profile's memory guard stops a
# run at the first nvidia-smi sample above 8 GiB. Commits only evidence (runs/ is never committed). Does not push.
#   bash scripts/final_run.sh
set -u
cd "$(dirname "$0")/.."
resume() {  # resume <run name>: continue an interrupted or failed run from its latest valid checkpoint
  local m="runs/$1/manifest.json"
  [ -f "$m" ] || return 0
  case "$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['status'])" "$m")" in
    completed|complete) ;;
    *) python scripts/run.py train --config "runs/$1/config.yaml" --resume latest || exit 1 ;;
  esac
}
python scripts/experiment.py run --entry H11.smoke || exit 1
resume qwen3_4b_final
resume qwen3_4b_final_lr0
python scripts/experiment.py run --entry H12.train_final || exit 1
python scripts/experiment.py run --entry H12.final_eval || exit 1
python scripts/experiment.py report
git add evidence research/l2l/experiments research/l2l/final_results.json research/l2l/final_report.md 2>/dev/null
git commit -q -m "Final Qwen3-4B run: evaluation and reports at $(git rev-parse --short HEAD)" && git log --oneline -1
echo "now: update research/l2l/hypotheses.yaml by hand (cite the new evidence), then git push"
