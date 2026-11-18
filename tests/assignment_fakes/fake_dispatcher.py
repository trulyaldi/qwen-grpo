#!/usr/bin/env python3
"""Stands in for `python scripts/experiment.py` in tests: same command line and the same outputs the stage runner reads
(evidence/<UTC>_<entry>/results.json, or a DONE line). FAKE_SCENARIO is a JSON file {entry id: {"status": ...,
"sleep": s, "lines": [[text, seconds after], ...] written live to the
evidence step log as the real dispatcher's child does, "files": {path under ASSIGNMENT_EXPERIMENTS_ROOT or ASSIGNMENT_RUNS_ROOT (prefix runs/): JSON}}}. Each
invocation is appended to FAKE_LOG."""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(args) + "\n")
if args and args[0] == "plan":
    print("plan: fake study planner")
    sys.exit(0)
if args[0] != "run":
    sys.exit(2)
entries = args[args.index("--entry") + 1:]
entries = [e for e in entries if not e.startswith("--")]
if "--dry-run" in args:
    for e in entries:
        print(f"{e} ({e}): would run:")
    sys.exit(0)
scenario = json.loads(Path(os.environ["FAKE_SCENARIO"]).read_text())
e = entries[0]
spec = scenario.get(e, scenario.get("*", {"status": "PASS"}))
d = Path(os.environ["ASSIGNMENT_EVIDENCE_ROOT"]) / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{e}"
if spec.get("lines"):
    (d / "raw").mkdir(parents=True)
    print(f"{e}: running ({e}) -> evidence/{d.name}", flush=True)
    print("   $ python scripts/run.py train --config fake.yaml", flush=True)
    with open(d / "raw" / "step01.log", "w") as log:
        for text, wait in spec["lines"]:
            log.write(text + "\n")
            log.flush()
            time.sleep(wait)
time.sleep(spec.get("sleep", 0.6))
for rel, data in spec.get("files", {}).items():
    root = Path(os.environ["ASSIGNMENT_RUNS_ROOT"]) if rel.startswith("runs/") else Path(os.environ["ASSIGNMENT_EXPERIMENTS_ROOT"])
    p = root / (rel[5:] if rel.startswith("runs/") else rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data))
import yaml  # noqa: E402  the evidence name the real dispatcher records is the study entry's `name`
study = yaml.safe_load(Path(os.environ.get("ASSIGNMENT_STUDY", Path(__file__).parents[2] / "configs" / "studies" /
                                           "assignment.yaml")).read_text())
if inc := study.get("include"):  # as scripts/experiment.py: entries reused from another study
    other = yaml.safe_load((Path(os.environ.get("ASSIGNMENT_STUDY", "")).parent / inc["study"]).read_text())
    study["entries"] = [x for x in other["entries"] if x["id"] in inc["entries"]] + list(study.get("entries") or [])
name = spec.get("evidence_name", next((x["name"] for x in study["entries"] if x["id"] == e), e))
d.mkdir(parents=True, exist_ok=True)
(d / "results.json").write_text(json.dumps({"entry": e, "evidence_name": name, "status": spec.get("status", "PASS"),
                                            "criterion": spec.get("criterion", "NOT_EVALUATED"),
                                            "detail": spec.get("detail", "fake"), "outputs": []}))
print(f"{e}: {spec.get('status', 'PASS')}")
sys.exit(0 if spec.get("status", "PASS") == "PASS" else 2)
