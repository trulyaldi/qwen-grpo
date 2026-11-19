"""The 150-update Qwen3-4B experiment, stage by stage, every GPU workload under raw nvidia-smi telemetry and the exact
8,000,000,000-byte cap (docs/FINAL_150_STEP_EXPERIMENT.md).

    python scripts/final150.py dry-run                 checks and the plan; runs, downloads and writes nothing
    python scripts/final150.py preflight               environment, hardware, configs, budget files
    python scripts/final150.py correctness             CPU tests, external kill/resume (tiny), async-L2L lab cases on this GPU
    python scripts/final150.py smoke                   4-update GPU smoke run: proves the raw telemetry files are produced
    python scripts/final150.py profile                 H11-style phase profile + H06 sync/async prefetch timing (Qwen3-4B)
    python scripts/final150.py train                   the 150-update run              (gate: preflight, correctness, smoke, profile)
    python scripts/final150.py resume                  continue an interrupted 150-update run from its latest valid checkpoint
    python scripts/final150.py control                 the LR = 0 control (30 updates) (same gate)
    python scripts/final150.py merge                   tensor-by-tensor CPU merge of the final adapter (monitored: no GPU use)
    python scripts/final150.py eval                    held-out test: base / trained / merged / control (monitored)
    python scripts/final150.py soup                    Soup's checks (scripts/soup_checks.py; UNAVAILABLE without Soup)
    python scripts/final150.py report                  requirement-compliance summary from the recorded stages
    python scripts/final150.py export                  tar.gz of all evidence + sha256 manifest (outside the repo)
    python scripts/final150.py all                     preflight .. report in order, stopping at the first failed gate
    python scripts/final150.py status
Options: --retry (re-run a stage whose last attempt did not PASS; a PASSed stage is never re-run), --waive ITEM
--reason TEXT (correctness only: record an explicit, reported waiver of one failing item; never silent).

Results: assignment_results/final150_v2/ (state.json, <stage>/<UTC>/ per attempt, never reused). Each monitored
workload: <attempt>/<item>/{stdout.log, stderr.log, guarded_run.json, telemetry/{nvidia_smi_apps.csv,
nvidia_smi_gpu.csv, nvidia_smi_*.stderr, pids.csv, telemetry.json}} (qwen_grpo.telemetry.run_guarded). A stage is
PASS only if every item's tool verdict is PASS and every capped item's memory status is PASS; missing or incomplete
telemetry is INCONCLUSIVE, never PASS. DFlash is out of scope: recorded as NOT_RUN. Nothing here deletes a run,
overwrites a result, uploads or publishes anything.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo import telemetry  # noqa: E402

CAP = 8_000_000_000
RESULTS = Path(os.environ.get("FINAL150_RESULTS", ROOT / "assignment_results" / "final150_v2"))
EVIDENCE = Path(os.environ.get("ASSIGNMENT_EVIDENCE_ROOT", ROOT / "evidence"))
PY = sys.executable
CFG = "configs/experiments/qwen3_4b_final_plain_150_v2.yaml"
CFG_LR0 = "configs/experiments/qwen3_4b_final_plain_150_v2_lr0.yaml"
CFG_SMOKE = "configs/experiments/qwen3_4b_final_plain_150_v2_smoke.yaml"
RUN, RUN_LR0, RUN_SMOKE = "qwen3_4b_final_plain_150_v2", "qwen3_4b_final_plain_150_v2_lr0", "qwen3_4b_final_plain_150_v2_smoke"
PREDICTION = "results/budget/predictions/qwen3-4b-final_8gb.json"
EVAL_OUT = "research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json"
LAB_ENTRIES = ["H01.lab_async_cuda", "H01.fault_unsafe_reuse", "H02.lab_async_cuda", "H02.fault_unsafe_reuse",
               "H04.lab_async_cuda", "H05.lab_async_cuda", "H08.resume_tiny_cpu", "H09.merge_tiny_cpu"]
PYTEST = ["tests/test_grpo.py", "tests/test_sentiment.py", "tests/test_training_invariants.py", "tests/test_multistep.py",
          "tests/test_streaming.py", "tests/test_merge.py", "tests/test_profiler.py", "tests/test_memory_guard.py",
          "tests/test_telemetry.py", "tests/test_config.py", "tests/test_trainer.py", "tests/test_prefetch.py",
          "tests/test_h01_same_host.py", "tests/test_h2d_accounting.py", "tests/test_final150.py",
          "tests/test_guarded_run.py", "tests/silent_failures"]
STAGES = ["preflight", "correctness", "smoke", "profile", "train", "control", "merge", "eval", "soup"]
GATES = {"train": ["preflight", "correctness", "smoke", "profile"], "control": ["preflight", "correctness", "smoke", "profile"],
         "resume": ["preflight", "correctness", "smoke", "profile"], "merge": ["train"], "eval": ["train", "control", "merge"],
         "smoke": ["preflight", "correctness"], "profile": ["preflight", "correctness", "smoke"]}
EXPECTED = {"run.name": RUN, "training.max_steps": 150, "evaluation.every": 10, "memory.budget_bytes": CAP,
            "memory.guard": True, "dflash.enabled": False, "execution.backend": "l2l_async", "model.name": "Qwen/Qwen3-4B",
            "training.batch_size": 5, "training.group_size": 6, "training.max_new_tokens": 4, "lora.rank": 32,
            "lora.alpha": 64, "training.learning_rate": 2e-4, "training.clip_eps": 0.2, "streaming.enabled": True}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(p):
    s = str(p)
    return s[len(str(ROOT)) + 1:] if s.startswith(str(ROOT) + "/") else s


def load_state():
    p = RESULTS / "state.json"
    return json.loads(p.read_text()) if p.exists() else {"created_utc": utc(), "stages": {}, "waivers": {}}


def save_state(state):
    RESULTS.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1, default=str) + "\n")
    tmp.replace(RESULTS / "state.json")


def last(state, stage):
    att = state["stages"].get(stage) or []
    return att[-1] if att else None


def passed(state, stage):
    a = last(state, stage)
    return a is not None and a["status"] == "PASS"


def gate_problems(state, stage):
    return [f"{g}: {(last(state, g) or {}).get('status', 'not run')}" for g in GATES.get(stage, []) if not passed(state, g)]


def config_checks(path, expected):
    """(ok, rows) for one experiment config: the resolved values the protocol requires."""
    from qwen_grpo import config
    cfg = config.load(ROOT / path)
    flat = {}

    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}.")
            else:
                flat[f"{prefix}{k}"] = v
    walk(config.as_dict(cfg))
    rows = [{"key": k, "expected": v, "actual": flat.get(k), "ok": flat.get(k) == v} for k, v in expected.items()]
    rows.append({"key": "budget_bytes(cfg)", "expected": CAP, "actual": config.budget_bytes(cfg),
                 "ok": config.budget_bytes(cfg) == CAP})
    rows.append({"key": "training.max_attempts >= 4 * max_steps (acceptance >= 25% not assumed)",
                 "expected": True, "actual": cfg.training.max_attempts, "ok": cfg.training.max_attempts >= 4 * cfg.training.max_steps})
    return all(r["ok"] for r in rows), rows, config.sha256(cfg)


# --- items ---------------------------------------------------------------------------------------------------------

def guarded(attempt_dir, item, argv, cap=CAP, expect_gpu=True):
    rec = telemetry.run_guarded(attempt_dir / item, [str(a) for a in argv], cap_bytes=cap, cwd=ROOT,
                                expect_gpu=expect_gpu, attribution_timeout_s=900, rel=rel)
    rec["files_present"] = telemetry.required_files_present(attempt_dir / item / "telemetry")
    (attempt_dir / item / "guarded_run.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"   {item}: exit {rec.get('exit')}, memory {rec.get('memory_status')} "
          f"({(rec.get('telemetry') or {}).get('memory_reason') or rec.get('error')})", flush=True)
    return rec


def plain(attempt_dir, item, argv):
    d = attempt_dir / item
    d.mkdir(parents=True, exist_ok=False)
    t0, started = time.time(), utc()
    with open(d / "stdout.log", "wb") as so, open(d / "stderr.log", "wb") as se:
        p = subprocess.run([str(a) for a in argv], cwd=ROOT, stdout=so, stderr=se,
                           env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": os.environ.get(
                               "FINAL150_CPU_TESTS_CUDA", "")})
    rec = {"argv": [rel(a) for a in argv], "started_utc": started, "ended_utc": utc(), "seconds": round(time.time() - t0, 1),
           "exit": p.returncode}
    (d / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"   {item}: exit {p.returncode}", flush=True)
    return rec


def item_status(rec, tool_ok):
    """PASS only if the tool said so AND (for a capped item) memory is PASS."""
    mem = rec.get("memory_status")
    if mem == "FAIL":
        return "FAIL", "GPU memory cap exceeded (nvidia-smi)"
    if not tool_ok:
        return "FAIL", f"tool verdict not PASS (exit {rec.get('exit')})"
    if mem not in (None, "PASS", "NOT_JUDGED"):
        return "INCONCLUSIVE", f"memory telemetry {mem}: {(rec.get('telemetry') or {}).get('memory_reason')}"
    return "PASS", "tool PASS" + (", memory PASS" if mem == "PASS" else "")


def dispatched_status(entry, since):
    """The study dispatcher's verdict for `entry`: the newest evidence/<UTC>_<entry>/results.json written after since."""
    cands = sorted(p for p in EVIDENCE.glob(f"*_{entry}") if p.stat().st_mtime >= since - 1)
    if not cands or not (cands[-1] / "results.json").exists():
        return None, None
    r = json.loads((cands[-1] / "results.json").read_text())
    return r.get("status"), rel(cands[-1])


# --- stages --------------------------------------------------------------------------------------------------------

def stage_preflight(d, state, a):
    items = []
    ok, rows, sha = config_checks(CFG, EXPECTED)
    exp_lr0 = {**EXPECTED, "run.name": RUN_LR0, "training.learning_rate": 0.0, "training.max_steps": 30}
    ok0, rows0, sha0 = config_checks(CFG_LR0, exp_lr0)
    items.append({"id": "config_150_v2", "status": "PASS" if ok else "FAIL", "rows": rows, "sha256": sha})
    items.append({"id": "config_lr0", "status": "PASS" if ok0 else "FAIL", "rows": rows0, "sha256": sha0})
    from qwen_grpo import runs
    for name in (RUN, RUN_LR0):
        exists = runs.run_dir(name).exists()
        items.append({"id": f"run_dir_{name}", "status": "PASS" if not exists or a.stage == "resume" else "FAIL",
                      "detail": f"runs/{name} {'exists: use resume, never overwrite' if exists else 'absent: new run'}"})
    smi = telemetry.smi_binary()
    items.append({"id": "nvidia_smi", "status": "PASS" if smi else "FAIL", "detail": smi or "not found (fail closed)"})
    if smi:
        q = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv"],
                           capture_output=True, text=True)
        (d / "nvidia_smi_query_gpu.csv").write_text(q.stdout + q.stderr)
    probe = subprocess.run([PY, "-c", "import json,torch;ok=torch.cuda.is_available();"
                            "print(json.dumps({'cuda':ok,'bf16':ok and torch.cuda.is_bf16_supported(),"
                            "'name':torch.cuda.get_device_name(0) if ok else None,'torch':torch.__version__}))"],
                           capture_output=True, text=True)
    cap = json.loads(probe.stdout or "{}") if probe.returncode == 0 else {}
    items.append({"id": "cuda_bf16", "status": "PASS" if cap.get("cuda") and cap.get("bf16") else "FAIL", "detail": cap})
    mem = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) * 1024
    free = shutil.disk_usage(ROOT).free
    items.append({"id": "host_ram", "status": "PASS" if mem >= 32e9 else "FAIL", "detail": f"MemTotal {mem} B (>= 32e9)"})
    items.append({"id": "disk", "status": "PASS" if free >= 40e9 else "FAIL", "detail": f"free {free} B (>= 40e9)"})
    for f in (PREDICTION, "results/budget/residuals/qwen3-4b-final_150_v2.json"):
        items.append({"id": f"budget_file:{f}", "status": "PASS" if (ROOT / f).exists() else "FAIL"})
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (d / "provenance.json").write_text(json.dumps({"git_commit": git("rev-parse", "HEAD"),
                                                   "git_status": git("status", "--porcelain", "--untracked-files=no"),
                                                   "python": sys.version, "probe": cap}, indent=1) + "\n")
    return items


def stage_correctness(d, state, a):
    items = []
    rec = plain(d, "pytest_cpu", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={d}/pytest_cpu.xml",
                                  *PYTEST])
    items.append({"id": "pytest_cpu", "status": "PASS" if rec["exit"] == 0 else "FAIL", "record": rec})
    rec = plain(d, "resume_kill_tiny_cpu", [PY, "scripts/resume_kill_test.py", "--tiny", "--out",
                                            f"t09_resume_kill_tiny_cpu_r{stamp()}"])
    items.append({"id": "resume_kill_tiny_cpu", "status": "PASS" if rec["exit"] == 0 else "FAIL", "record": rec})
    for entry in LAB_ENTRIES:
        t0 = time.time()
        rec = guarded(d, entry, [PY, "scripts/experiment.py", "run", "--entry", entry, "--again"],
                      expect_gpu=not entry.endswith("_cpu"))
        status, where = dispatched_status(entry, t0)
        s, why = item_status(rec, status == "PASS")
        if entry in state.get("waivers", {}) and s != "PASS":
            s, why = "WAIVED", f"{why}; explicit waiver: {state['waivers'][entry]}"
        items.append({"id": entry, "status": s, "detail": why, "tool_status": status, "evidence": where,
                      "memory_status": rec.get("memory_status")})
    return items


def stage_smoke(d, state, a):
    from qwen_grpo import runs
    if runs.run_dir(RUN_SMOKE).exists():
        return [{"id": "smoke_train", "status": "FAIL", "detail": f"runs/{RUN_SMOKE} exists (from an earlier smoke "
                 "attempt): move it aside by hand to re-run; nothing is deleted here"}]
    rec = guarded(d, "smoke_train", [PY, "scripts/run.py", "train", "--config", CFG_SMOKE])
    m = runs.manifest(RUN_SMOKE) or {}
    s, why = item_status(rec, rec.get("exit") == 0 and m.get("status") == "completed")
    files = rec.get("files_present", {})
    t = rec.get("telemetry") or {}
    raw_ok = all(files.get(f) for f in ("nvidia_smi_apps.csv", "nvidia_smi_gpu.csv", "pids.csv", "telemetry.json"))
    checks = {"raw_files_present": raw_ok, "attributed_numeric_samples": t.get("attributed_numeric_samples"),
              "device_samples": (t.get("sampling") or {}).get("device_samples"),
              "max_gap_s": (t.get("sampling") or {}).get("max_gap_s"),
              "checkpoints": sorted(p.name for p in (runs.run_dir(RUN_SMOKE) / "checkpoints").glob("step*"))
              if runs.run_dir(RUN_SMOKE).exists() else []}
    if s == "PASS" and not (raw_ok and (checks["attributed_numeric_samples"] or 0) > 0):
        s, why = "INCONCLUSIVE", "raw telemetry missing or unattributed"
    if s == "PASS" and checks["checkpoints"] != ["step0", "step2", "step4"]:
        s, why = "FAIL", f"checkpoints {checks['checkpoints']} != step0/2/4"
    return [{"id": "smoke_train", "status": s, "detail": why, "checks": checks, "memory_status": rec.get("memory_status")}]


def stage_profile(d, state, a):
    items = []
    rec = guarded(d, "phase_profile", [PY, "scripts/profile_run.py", "--config", CFG, "--prediction", PREDICTION,
                                       "--steps", "2", "--out-dir", rel(d / "phase_profile_result"),
                                       "--out", "t11_profile_qwen3_4b_150_v2"])
    res = d / "phase_profile_result" / "t11_profile_qwen3_4b_150_v2.json"
    verdict = json.loads(res.read_text()).get("verdict") if res.exists() else None
    s, why = item_status(rec, rec.get("exit") == 0 and verdict in ("PASS", "INCOMPLETE_EVIDENCE"))
    items.append({"id": "phase_profile", "status": s, "detail": f"{why}; profile verdict {verdict}"})
    t0 = time.time()
    rec = guarded(d, "H06.prefetch_timing_4b", [PY, "scripts/experiment.py", "run", "--entry", "H06.prefetch_timing_4b",
                                                "--again"])
    status, where = dispatched_status("H06.prefetch_timing_4b", t0)
    # H06 is a measurement: its hypothesis may be NOT_SUPPORTED (no speed-up) and that is a result, not a failure.
    # The stage needs it to have EXECUTED validly (bitwise sync == async on the first trial) within the cap.
    s, why = item_status(rec, status in ("PASS", "FAIL") and rec.get("exit") is not None)
    items.append({"id": "H06.prefetch_timing_4b", "status": s, "detail": why, "tool_status": status, "evidence": where,
                  "note": "tool_status FAIL can mean 'no speed-up' (hypothesis outcome) or an execution error: read "
                          "the lab result's execution_status before citing it"})
    return items


def stage_train(d, state, a, cfg=CFG, name=RUN, item="train", resume=False):
    from qwen_grpo import runs
    if runs.run_dir(name).exists() and not resume:
        return [{"id": item, "status": "FAIL", "detail": f"runs/{name} exists: `final150.py resume` continues it; "
                 "nothing is overwritten"}]
    argv = [PY, "scripts/run.py", "train", "--config", cfg] if not resume else \
        [PY, "scripts/run.py", "train", "--config", f"runs/{name}/config.yaml", "--resume", "latest"]
    rec = guarded(d, item, argv)
    m = runs.manifest(name) or {}
    log = runs.training_dir(name) / "training.json"
    tr = json.loads(log.read_text()) if log.exists() else {}
    s, why = item_status(rec, rec.get("exit") == 0 and m.get("status") == "completed")
    facts = {"successful_steps": tr.get("successful_steps"), "attempts": tr.get("groups_generated"),
             "zero_signal": tr.get("zero_signal_groups"), "evaluated_steps": [e["step"] for e in tr.get("evaluations", [])],
             "manifest_status": m.get("status")}
    want = 30 if name == RUN_LR0 else 150
    if s == "PASS" and facts["successful_steps"] != want:
        s, why = "FAIL", f"{facts['successful_steps']} of {want} updates (attempt budget exhausted?)"
    if s == "PASS" and facts["evaluated_steps"] != list(range(0, want + 1, 10)):
        s, why = "FAIL", f"subset evaluations at {facts['evaluated_steps']}, expected every 10 from 0 to {want}"
    return [{"id": item, "status": s, "detail": why, "facts": facts, "memory_status": rec.get("memory_status")}]


def merged_dir():
    return RESULTS / "merged" / f"{RUN}_step150"


def stage_merge(d, state, a):
    from qwen_grpo import runs
    ck = runs.checkpoint_path(RUN, 150)
    if merged_dir().exists():
        return [{"id": "merge", "status": "FAIL", "detail": f"{rel(merged_dir())} exists: never overwritten"}]
    merged_dir().parent.mkdir(parents=True, exist_ok=True)
    rec = guarded(d, "merge", [PY, "scripts/merge.py", "--adapter", rel(ck), "--out", rel(merged_dir())], expect_gpu=False)
    s, why = item_status(rec, rec.get("exit") == 0 and (merged_dir() / "merge_provenance.json").exists())
    return [{"id": "merge", "status": s, "detail": why, "memory_status": rec.get("memory_status"),
             "child_max_rss_bytes": rec.get("children_max_rss_bytes")}]


def stage_eval(d, state, a):
    if (ROOT / EVAL_OUT).exists():
        return [{"id": "final_eval", "status": "FAIL", "detail": f"{EVAL_OUT} exists: never overwritten"}]
    rec = guarded(d, "final_eval", [PY, "scripts/final_eval.py", "--run", RUN, "--control", RUN_LR0, "--merged-dir",
                                    rel(merged_dir()), "--out", EVAL_OUT])
    s, why = item_status(rec, rec.get("exit") == 0 and (ROOT / EVAL_OUT).exists())
    out = {"id": "final_eval", "status": s, "detail": why, "memory_status": rec.get("memory_status")}
    if (ROOT / EVAL_OUT).exists():
        r = json.loads((ROOT / EVAL_OUT).read_text())
        out["phase_memory"] = telemetry.phase_peaks(d / "final_eval" / "telemetry", r.get("phases", []))
        out["phases_without_samples"] = [p["name"] for p in out["phase_memory"] if p["attributed_samples"] == 0]
        (d / "phase_memory.json").write_text(json.dumps(out["phase_memory"], indent=1) + "\n")
        if s == "PASS" and out["phases_without_samples"]:
            s = out["status"] = "INCONCLUSIVE"
            out["detail"] = f"no nvidia-smi sample inside {out['phases_without_samples']}"
        out["paired"] = r.get("paired")
    return [out]


def stage_soup(d, state, a):
    rec = plain(d, "soup_checks", [PY, "scripts/soup_checks.py"])
    st = {0: "PASS", 2: "UNAVAILABLE"}.get(rec["exit"], "FAIL")
    return [{"id": "soup_checks", "status": st, "detail": "PASS = the checks executed (their findings are evidence, "
             "not a verdict); UNAVAILABLE = Soup not installed (docs/SOUP_CHECK_COVERAGE.md)"}]


RUNNERS = {"preflight": stage_preflight, "correctness": stage_correctness, "smoke": stage_smoke,
           "profile": stage_profile, "train": stage_train,
           "resume": lambda d, s, a: stage_train(d, s, a, resume=True),
           "control": lambda d, s, a: stage_train(d, s, a, cfg=CFG_LR0, name=RUN_LR0, item="control"),
           "merge": stage_merge, "eval": stage_eval, "soup": stage_soup}


def run_stage(name, a):
    state = load_state()
    if name != "resume" and passed(state, name) and not a.retry:
        print(f"{name}: already PASS ({last(state, name)['dir']}); not re-run")
        return 0
    if name != "resume" and last(state, name) and not passed(state, name) and not a.retry and name in (
            "train", "control", "merge", "eval"):
        print(f"{name}: last attempt {last(state, name)['status']}; re-run only with --retry (expensive stage)")
        return 1
    if probs := gate_problems(state, name):
        print(f"{name}: BLOCKED by its gate: {probs}")
        state["stages"].setdefault(name, []).append({"status": "BLOCKED", "detail": probs, "at": utc(), "dir": None})
        save_state(state)
        return 1
    d = RESULTS / name / stamp()
    d.mkdir(parents=True, exist_ok=False)
    print(f"== {name} -> {rel(d)}", flush=True)
    started = utc()
    items = RUNNERS[name](d, state, a)
    statuses = [i["status"] for i in items]
    status = "PASS" if all(x in ("PASS", "WAIVED") for x in statuses) else \
        "FAIL" if "FAIL" in statuses else "INCONCLUSIVE"
    if name == "soup":
        status = items[0]["status"] if items[0]["status"] in ("PASS", "UNAVAILABLE") else "FAIL"
    att = {"dir": rel(d), "started_utc": started, "ended_utc": utc(), "status": status, "items": items}
    (d / "stage.json").write_text(json.dumps(att, indent=1, default=str) + "\n")
    state = load_state()
    state["stages"].setdefault("train" if name == "resume" else name, []).append(att)
    save_state(state)
    for i in items:
        print(f"   [{i['status']}] {i['id']}: {i.get('detail', '')}")
    print(f"== {name}: {status}")
    return 0 if status in ("PASS", "UNAVAILABLE") else 1


def cmd_dry_run(a):
    print(__doc__.split("\n\n")[0] + "\n")
    ok_all = True
    for path, exp in ((CFG, EXPECTED), (CFG_LR0, {**EXPECTED, "run.name": RUN_LR0, "training.learning_rate": 0.0,
                                                  "training.max_steps": 30})):
        ok, rows, sha = config_checks(path, exp)
        ok_all &= ok
        print(f"{path}  sha256 {sha[:16]}  {'OK' if ok else 'MISMATCH'}")
        for r in rows:
            if not r["ok"]:
                print(f"   MISMATCH {r['key']}: expected {r['expected']!r}, got {r['actual']!r}")
    from qwen_grpo import config, runs
    c = config.load(ROOT / CFG)
    print(f"\nschedule: {c.training.max_steps} successful updates (max {c.training.max_attempts} rollout batches), subset "
          f"evaluation + checkpoint at steps {list(range(0, c.training.max_steps + 1, c.evaluation.every))}")
    print(f"GPU cap: {CAP} bytes (= {CAP / 2**30:.4f} GiB), in-process guard budget {config.budget_bytes(c)} bytes, "
          f"external nvidia-smi supervisor scripts/guarded_run.py / qwen_grpo.telemetry.run_guarded")
    for n in (RUN, RUN_LR0, RUN_SMOKE):
        print(f"runs/{n}: {'EXISTS (train refuses; use resume)' if runs.run_dir(n).exists() else 'absent (new)'}")
    print(f"nvidia-smi: {telemetry.smi_binary() or 'NOT FOUND (GPU stages would refuse to start: fail closed)'}")
    state = load_state()
    print(f"\nresults: {rel(RESULTS)}  ({'existing state' if (RESULTS / 'state.json').exists() else 'no state yet'})")
    for s in STAGES:
        lt = last(state, s)
        print(f"  {s:12s} last {lt['status'] if lt else '-':12s} gate {GATES.get(s, [])} "
              f"{'(gate open)' if not gate_problems(state, s) else '(blocked: ' + ', '.join(gate_problems(state, s)) + ')'}")
    print("  dflash       NOT_RUN (out of scope for this experiment; docs/REMAINING_LIMITATIONS.md)")
    print("\ncommands each stage runs (every GPU one wrapped by run_guarded with the cap):")
    plan = {"correctness": [f"pytest {' '.join(PYTEST)}", "scripts/resume_kill_test.py --tiny"] +
            [f"scripts/experiment.py run --entry {e} --again" for e in LAB_ENTRIES],
            "smoke": [f"scripts/run.py train --config {CFG_SMOKE}"],
            "profile": [f"scripts/profile_run.py --config {CFG} --prediction {PREDICTION} --steps 2",
                        "scripts/experiment.py run --entry H06.prefetch_timing_4b --again"],
            "train": [f"scripts/run.py train --config {CFG}"],
            "resume": [f"scripts/run.py train --config runs/{RUN}/config.yaml --resume latest"],
            "control": [f"scripts/run.py train --config {CFG_LR0}"],
            "merge": [f"scripts/merge.py --adapter runs/{RUN}/checkpoints/step150 --out {rel(merged_dir())}"],
            "eval": [f"scripts/final_eval.py --run {RUN} --control {RUN_LR0} --merged-dir {rel(merged_dir())} --out {EVAL_OUT}"],
            "soup": ["scripts/soup_checks.py"]}
    for s, cmds in plan.items():
        for c_ in cmds:
            print(f"  {s:12s} {c_}")
    print(f"\ndry run: {'OK' if ok_all else 'CONFIG MISMATCH'} (nothing was run or written)")
    return 0 if ok_all else 1


def cmd_report(a):
    state = load_state()
    rows = []
    for s in STAGES:
        lt = last(state, s)
        rows.append((s, lt["status"] if lt else "NOT_RUN", lt["dir"] if lt and lt.get("dir") else "-"))
    req = [("150 successful updates, validation every 10", "train"), ("LR = 0 negative control", "control"),
           ("8e9-byte cap, nvidia-smi, training", "train"), ("8e9-byte cap, merge (no GPU use)", "merge"),
           ("8e9-byte cap, final + merged-model evaluation", "eval"), ("L2L correctness on this GPU", "correctness"),
           ("raw telemetry proven by smoke run", "smoke"), ("async prefetch overlap measured (H06 4B)", "profile"),
           ("Soup checks", "soup")]
    md = ["# 150-update experiment: requirement compliance", "", f"Generated {utc()} from `{rel(RESULTS)}/state.json`.",
          "", "| stage | last status | attempt |", "|---|---|---|"] + [f"| {s} | {st} | {dd} |" for s, st, dd in rows]
    md += ["| dflash | NOT_RUN | out of scope (docs/REMAINING_LIMITATIONS.md) |", "", "| requirement | status | stage |",
           "|---|---|---|"]
    md += [f"| {r} | {(last(state, s) or {}).get('status', 'NOT_RUN')} | {s} |" for r, s in req]
    md += ["| DFlash rollouts (task 2c) | NOT_RUN | — |", "", "Waivers: " + (json.dumps(state.get("waivers")) or "none")]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "compliance.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


def cmd_export(a):
    from qwen_grpo import runs
    out = Path(a.to or Path.home()) / f"final150_v2_evidence_{stamp()}.tar.gz"
    paths = [RESULTS, *(runs.run_dir(n) for n in (RUN, RUN_LR0, RUN_SMOKE))]
    paths += [p for p in (ROOT / "research" / "l2l" / "experiments").glob("*150_v2*")]
    paths += [p for p in (ROOT / "research" / "l2l" / "experiments").glob("t09_resume_kill_*")]
    paths += [p for p in (ROOT / "research" / "soup").iterdir()] if (ROOT / "research" / "soup").exists() else []
    paths += [p for p in (ROOT / "results" / "budget" / "residuals").glob("*")]
    paths += [p for p in EVIDENCE.iterdir() if p.is_dir()] if EVIDENCE.exists() else []
    paths = [p for p in paths if p.exists() and not (p == merged_dir() or merged_dir() in p.parents)]
    import hashlib
    manifest = []
    for p in paths:
        for f in ([p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())):
            if merged_dir() in f.parents:
                continue  # the merged weights (~8 GB) are reproducible from the adapter; their sha256 are in merge_provenance.json
            manifest.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel(f)}")
    (RESULTS / "export_manifest.sha256").write_text("\n".join(manifest) + "\n")
    with tarfile.open(out, "w:gz") as tar:
        for p in paths + [RESULTS / "export_manifest.sha256"]:
            tar.add(p, arcname=rel(p), filter=lambda ti: None if str(merged_dir().relative_to(ROOT)) in ti.name else ti)
    print(f"wrote {out} ({out.stat().st_size} B, {len(manifest)} files; manifest {rel(RESULTS / 'export_manifest.sha256')})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["dry-run", "status", "report", "export", "all", "resume", *STAGES])
    ap.add_argument("--retry", action="store_true")
    ap.add_argument("--waive")
    ap.add_argument("--reason")
    ap.add_argument("--to", help="export: directory for the tar.gz (default: home)")
    a = ap.parse_args(argv)
    if a.waive:
        if not a.reason or a.waive not in LAB_ENTRIES:
            raise SystemExit("--waive ITEM needs --reason and ITEM must be a correctness lab entry")
        state = load_state()
        state.setdefault("waivers", {})[a.waive] = f"{a.reason} (recorded {utc()})"
        save_state(state)
        print(f"waiver recorded for {a.waive}; it is reported as WAIVED, never PASS")
        if a.stage in ("status",):
            return 0
    if a.stage == "dry-run":
        return cmd_dry_run(a)
    if a.stage in ("status", "report"):
        return cmd_report(a)
    if a.stage == "export":
        return cmd_export(a)
    if a.stage == "all":
        for s in STAGES:
            rc = run_stage(s, a)
            if rc and s != "soup":
                print(f"stopped after {s}")
                cmd_report(a)
                return rc
        return cmd_report(a)
    a.stage_name = a.stage
    return run_stage(a.stage, a)


if __name__ == "__main__":
    sys.exit(main())
