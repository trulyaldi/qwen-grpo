#!/usr/bin/env bash
# Ticket 13 on the GPU host, after scripts/laptop_checks.sh: the final Qwen3-4B GRPO run (hours; resumable), the LR=0
# negative control, the held-out test evaluation of base / trained / merged / control through L2L, and the final
# report regenerated from all evidence. Commits only evidence files (runs/ is not committed). Does not push.
#   bash scripts/final_run.sh
set -u
cd "$(dirname "$0")/.."
status() { python - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path("runs") / sys.argv[1] / "training" / "training.json"
print(json.loads(p.read_text()).get("status", "?") if p.exists() else "absent")
PY
}
train() {  # train <config> <run name>: new run, or resume an interrupted one from its latest valid checkpoint
  case "$(status "$2")" in
    absent) python scripts/run.py train --config "$1" ;;
    complete|completed) echo "$2: complete" ;;
    *) python scripts/run.py train --config "runs/$2/config.yaml" --resume latest ;;
  esac
}
train configs/experiments/qwen3_4b_final.yaml qwen3_4b_final || exit 1
train configs/experiments/qwen3_4b_final_lr0.yaml qwen3_4b_final_lr0 || exit 1
E=research/l2l/experiments/t13_final_eval_qwen3_4b_final.json
[ -e "$E" ] || python scripts/final_eval.py --run qwen3_4b_final --control qwen3_4b_final_lr0 --device cuda || exit 1
python scripts/final_report.py
git add "$E" research/l2l/final_results.json research/l2l/final_report.md 2>/dev/null
git commit -q -m "Ticket 13: final Qwen3-4B evaluation and final report at $(git rev-parse --short HEAD)" && git log --oneline -1
echo "now: git push"
