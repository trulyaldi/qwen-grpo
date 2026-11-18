#!/usr/bin/env python3
"""Stands in for pytest, the CUDA check, budget.py, compare_runs.py and run.py in tests. Behaviour by argument:
--junitxml=X writes a passing junit report (FAKE_PYTEST_FAIL=1: failing); --out X writes a passing CUDA check;
--out-dir X writes a budget prediction; anything else exits FAKE_TOOL_EXIT (default 0). The process stays alive
FAKE_TOOL_SLEEP seconds (default 0.6) so the telemetry can sample it."""
import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
time.sleep(float(os.environ.get("FAKE_TOOL_SLEEP", "0.6")))
for a in args:
    if a.startswith("--junitxml="):
        fail = os.environ.get("FAKE_PYTEST_FAIL") == "1"
        Path(a.split("=", 1)[1]).write_text(f'<testsuites><testsuite tests="3" failures="{int(fail)}" errors="0" '
                                            'skipped="0"/></testsuites>')
        sys.exit(1 if fail else 0)
if "--out" in args:
    Path(args[args.index("--out") + 1]).write_text(json.dumps({"ok": True, "detail": "fake CUDA check"}))
    sys.exit(0)
if "--out-dir" in args:
    d = Path(args[args.index("--out-dir") + 1])
    d.mkdir(parents=True, exist_ok=True)
    cfg = args[args.index("--config") + 1] if "--config" in args else ""
    name = "qwen3-4b-final_8gb.json" if "final" in cfg else Path(cfg).stem.replace("qwen3-4b", "qwen3-4b_8gb") + ".json"
    (d / name).write_text(os.environ.get("FAKE_PREDICTION", "{}"))
    sys.exit(0)
sys.exit(int(os.environ.get("FAKE_TOOL_EXIT", "0")))
