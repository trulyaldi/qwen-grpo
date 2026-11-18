"""Stage runner behind scripts/run_assignment.sh: one assignment run = assignment_results/<run-id>/.

    python scripts/assignment.py init  [--resume | --from-stage S]   # prints the run directory
    python scripts/assignment.py stage S --run-dir D                  # S in A..H
    python scripts/assignment.py report --run-dir D                   # scripts/assignment_report.py
    python scripts/assignment.py status [--run-dir D]
    python scripts/assignment.py dry-run

This file sequences existing tools and records what they did. It never computes an experimental result itself:
  experiments  every study entry (configs/studies/assignment.yaml) runs as `python scripts/experiment.py run --entry`,
               whose own hardware/prerequisite checks, raw logs and verdicts (evidence/<UTC>_<entry>/results.json) are
               read back unchanged
  tests        pytest suites, judged by their exit code and junit XML
  telemetry    every item runs in its own process group under qwen_grpo.telemetry (raw nvidia-smi every 200 ms, the
               study's gpu_memory_cap enforced by a watchdog on the Qwen3-4B stages)

Stages and gates (a gate that is not met records the stage's items as BLOCKED, with the reason, and runs nothing):
  A preflight      provenance, environment, capabilities, resolved configs, study plan, budget predictions (read-only)
  B correctness    unit tests, tiny-model equivalence and fault-injection lab cases on this host (fresh evidence)
  C reference      Qwen3-0.6B bf16: resident vs L2L sync/async, prefetch timing, resume, real-model merge      gate: GPU
  D memory         Qwen3-4B guarded smoke run, then the phase profiler                                        gate: GPU, host
  E dflash         DFlash benchmark, and the smoke run with DFlash rollouts (end-to-end GRPO steps)          gate: D smoke
  F training       final run and LR=0 control, both with DFlash rollouts (streaming runs: resumed from their latest
                   crash-consistent checkpoint when interrupted; an interrupted non-streaming run, e.g. stage C's, is
                   BLOCKED, never resumed)
                   gate: B essential correctness, C real-model equivalence, D smoke and profile with PASS memory, and
                   (the final config enables DFlash) E's benchmark (greedy DFlash == plain on Qwen3-4B) and DFlash smoke
                   run with PASS memory
  G evaluation     held-out test evaluation (base / trained / merged / control), DFlash acceptance on the primary
                   run's checkpoints step0/20/40/60                                                            gate: F
Primary run: qwen3_4b_final.yaml, 60 updates (checkpoints every 20); LR=0 control 30 updates; the DFlash smoke 10.
Never run here: opt_in study entries (H12.train_extended = the 210-update continuation, H10.dflash_drift_extended);
check_no_opt_in refuses a stage list naming one.
  H audit          Soup checks if Soup is installed (else NOT_TESTED); the silent-failure matrix is built by the report

Two modes, recorded per run in state.json (`mode`; runs made before it existed are "full"):
  full        (default; --full-tests is an alias) everything above. The only mode that can produce SHIP.
  experiment  --experiment: the stages run in the order A D B F G H, without DFlash: C (Qwen3-0.6B reference) and
              E (DFlash) are not run and their requirements are NOT_TESTED. F trains qwen3_4b_final_plain.yaml and its
              LR=0 control (qwen3_4b_final*.yaml with dflash disabled: plain L2L rollouts, own run names, every other
              setting inherited) and G evaluates them (H12.train_final_plain, H12.final_eval_plain; no DFlash drift).
              B is reduced to the async-L2L lab cases the final run depends on (forward, gradient, generation, update;
              this GPU), the two unsafe-reuse fault controls, and tiny resume and merge - no pytest suites, no sync /
              CPU lab matrix. F's gate: those B items PASS except H02.fault_unsafe_reuse (a known FAIL, recorded as a
              waived limitation, never as PASS), D's smoke and profile with PASS memory, and the primary config's
              DFlash disabled. Stage A predicts the plain configuration only (no drafter metadata is fetched). The
              report never gives SHIP for an experiment run.
--resume continues the latest run of the requested mode only; --fresh (new runs only) moves the fixed-name training
runs and Qwen3-4B results of the mode's stages aside to <path>.archived-<UTC> (never deleted) so they are re-run.
Every item's stdout/stderr is streamed live with a heartbeat every ASSIGNMENT_HEARTBEAT_S seconds (default 30).

Statuses: PASS, FAIL, BLOCKED, INCONCLUSIVE, NOT_TESTED. An item's status is what its tool judged (never its exit code
alone); a capped item whose telemetry shows a sample above the cap is FAIL whatever the tool said. Resume re-runs only
stages that did not finish. Evidence that predates this run is re-run under a new name (stages B, C) or, where that is
impossible (training runs, fixed-name Qwen3-4B results), checked for provenance and otherwise marked INCONCLUSIVE.

Test seams (environment): ASSIGNMENT_RESULTS_ROOT, ASSIGNMENT_EVIDENCE_ROOT, ASSIGNMENT_DISPATCHER (a command that
replaces `python scripts/experiment.py`), ASSIGNMENT_PROBE_JSON (capabilities instead of probing),
ASSIGNMENT_NVIDIA_SMI, ASSIGNMENT_STUDY, ASSIGNMENT_GRACE_S, ASSIGNMENT_STAGE_ITEMS (JSON: stage -> item ids),
ASSIGNMENT_RUNS_ROOT, ASSIGNMENT_TOOL_OVERRIDES (JSON: tool -> command), ASSIGNMENT_EXPERIMENTS_ROOT (report).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo import telemetry  # noqa: E402

RESULTS = Path(os.environ.get("ASSIGNMENT_RESULTS_ROOT", ROOT / "assignment_results"))
EVIDENCE = Path(os.environ.get("ASSIGNMENT_EVIDENCE_ROOT", ROOT / "evidence"))
STUDY = Path(os.environ.get("ASSIGNMENT_STUDY", ROOT / "configs" / "studies" / "assignment.yaml"))
RUNS_DIR = Path(os.environ.get("ASSIGNMENT_RUNS_ROOT", ROOT / "runs"))
EXP_DIR = Path(os.environ.get("ASSIGNMENT_EXPERIMENTS_ROOT", ROOT / "research" / "l2l" / "experiments"))
GRACE_S = float(os.environ.get("ASSIGNMENT_GRACE_S", "60"))
STAGES = "ABCDEFGH"
STAGE_NAMES = {"A": "preflight", "B": "correctness", "C": "reference_0.6b", "D": "memory_4b", "E": "dflash",
               "F": "training", "G": "evaluation", "H": "audit"}
RAW_DIRS = {"A": "preflight", "B": "correctness", "C": "l2l", "D": "memory", "E": "dflash", "F": "training",
            "G": "evaluation", "H": "soup_checks"}
STATUSES = ("PASS", "FAIL", "BLOCKED", "INCONCLUSIVE", "NOT_TESTED")
SEVERITY = {"FAIL": 5, "INCONCLUSIVE": 4, "BLOCKED": 3, "NOT_TESTED": 2, "PASS": 1}
DISPATCH_MAP = {"PASS": "PASS", "FAIL": "FAIL", "BLOCKED": "BLOCKED", "NOT TESTED": "NOT_TESTED", "DRY_RUN": "NOT_TESTED"}

# Which study entries each stage runs, in order. B and C produce fresh evidence on this host (--again when an older
# result exists); D-G judge fixed-name Qwen3-4B results and training runs, which cannot be renamed.
# Dropped from the default protocol by design (the study entries stay for manual runs; the report shows them as
# NOT_TESTED with this reason, scripts/assignment_report.py DROPPED): H03 (chunked output head; execution.vocab_chunk_size
# is refused for training, so it is off the final path) and H07 (bounded residency; copy-only evidence exists, the
# execution-level case is not on the final path).
STAGE_ENTRIES = {
    "B": ["H01.lab_sync", "H01.fault_wrong_layer", "H01.lab_async_cuda", "H01.fault_unsafe_reuse",
          "H02.lab_sync_cpu", "H02.lab_sync_cuda", "H02.fault_detach_boundary", "H02.lab_async_cuda",
          "H02.fault_unsafe_reuse", "H04.lab_sync_cpu", "H04.lab_sync_cuda", "H04.fault_stale_position",
          "H04.fault_no_padding_mask", "H04.lab_async_cuda", "H05.lab_sync_cpu", "H05.lab_sync_cuda",
          "H05.fault_advantage", "H05.lab_async_cuda", "H08.resume_tiny_cpu", "H09.merge_tiny_cpu"],
    "C": ["H05.real_0.6b_bf16", "H06.prefetch_timing", "H08.resume_qwen3_0.6b", "H09.merge_qwen3_0.6b"],
    "D": ["H11.smoke", "H11.profile"],
    "E": ["H10.dflash_bench", "H10.smoke_dflash"],
    "F": ["H12.train_final"],
    "G": ["H12.final_eval", "H10.dflash_drift"],
}
FRESH = {"B", "C"}
CAPPED = {"D", "E", "F", "G"}  # the 8 GB cap applies to every Qwen3-4B workload
# Correctness the final run depends on: L2L (sync and async, on this GPU) equals resident for forward, backward,
# generation and the whole update, the unsafe-reuse faults are caught, resume and merge are exact.
ESSENTIAL_B = ["unit_tests_cpu", "H01.lab_sync", "H01.lab_async_cuda", "H01.fault_unsafe_reuse", "H02.lab_sync_cuda",
               "H02.lab_async_cuda", "H02.fault_unsafe_reuse", "H04.lab_sync_cuda", "H04.lab_async_cuda",
               "H05.lab_sync_cuda", "H05.lab_async_cuda", "H08.resume_tiny_cpu", "H09.merge_tiny_cpu"]
PYTEST_CPU = ["tests/test_grpo.py", "tests/test_sentiment.py", "tests/test_policy.py", "tests/test_training_invariants.py",
              "tests/test_multistep.py", "tests/test_streaming.py", "tests/test_merge.py", "tests/test_profiler.py",
              "tests/test_memory_guard.py", "tests/test_telemetry.py", "tests/test_dflash_training.py"]
PYTEST_SILENT = ["tests/silent_failures"]
PYTEST_CUDA = ["tests/test_prefetch.py", "tests/test_backward.py", "tests/test_generator.py", "tests/test_layer_stream.py",
               "tests/test_backend.py", "tests/test_output_head.py", "tests/test_weight_store.py", "tests/test_dflash.py"]
# Budgets recomputed in stage A. qwen3-4b_final_dflash: the final run as configured (DFlash rollouts);
# qwen3-4b_final: the same shape with plain rollouts (the plain-L2L baseline and the H11.profile run)
BUDGET_CONFIGS = ("configs/budget/qwen3-4b_final_dflash.yaml", "configs/budget/qwen3-4b_final.yaml",
                  "configs/budget/qwen3-4b.yaml", "configs/budget/qwen3-0.6b.yaml")
FINAL_RUNS = {"qwen3_4b_final": "configs/experiments/qwen3_4b_final.yaml",
              "qwen3_4b_final_lr0": "configs/experiments/qwen3_4b_final_lr0.yaml"}
PRIMARY, CONTROL = list(FINAL_RUNS)               # the trained run and its LR=0 control (install_experiment rebinds)
TRAIN_ID, EVAL_ID = "H12.train_final", "H12.final_eval"

# --experiment: prioritise the Qwen3-4B experiments; the full qualification protocol above is unchanged
MODE, MODE_KEY = "full", "mode"  # the active mode, and its key in state.json (scripts/assignment_local.py: local_mode)
MODES = ("full", "experiment")
# Experiment mode runs no DFlash at all: no stage E, the plain-L2L final configs (dflash disabled, nothing else
# changed, own run names so their checkpoints never mix with a DFlash-trained qwen3_4b_final), no DFlash tests or drift.
EXPERIMENT_STAGES = "ADBFGH"
EXPERIMENT_FINAL_RUNS = {"qwen3_4b_final_plain": "configs/experiments/qwen3_4b_final_plain.yaml",
                         "qwen3_4b_final_plain_lr0": "configs/experiments/qwen3_4b_final_plain_lr0.yaml"}
EXPERIMENT_TRAIN_ID, EXPERIMENT_EVAL_ID = "H12.train_final_plain", "H12.final_eval_plain"
EXPERIMENT_STAGE_ENTRIES = {
    "B": ["H01.lab_async_cuda", "H01.fault_unsafe_reuse", "H02.lab_async_cuda", "H02.fault_unsafe_reuse",
          "H04.lab_async_cuda", "H05.lab_async_cuda", "H08.resume_tiny_cpu", "H09.merge_tiny_cpu"],
    "D": STAGE_ENTRIES["D"], "F": [EXPERIMENT_TRAIN_ID], "G": [EXPERIMENT_EVAL_ID],
}
# H02.fault_unsafe_reuse stays FAIL (its lab_fault judge counts a refusal by the backward checksum as "not detected";
# 32 of 80 injected configurations ran to wrong results caught only by the offline resident comparison). It is the
# only item experiment mode does not require for F: recorded in state["waivers"] and the report, never as PASS.
# Corrected 2026-11-18 (research/l2l/corrections/h02_unsafe_reuse_20261118): prefetch.BufferReuseError now refuses
# the unsafe copy at runtime and the lab records it as a detection. The waiver and the recorded FAILs are unchanged;
# a new run records whatever status it gets.
WAIVED = {"H02.fault_unsafe_reuse": "historical fault-injection FAIL (runtime protection against async buffer "
                                    "corruption is incomplete); experiment mode trains without it, the report lists it "
                                    "as a known limitation, and no qualification claim depends on it"}
EXPERIMENT_ESSENTIAL_B = [i for i in EXPERIMENT_STAGE_ENTRIES["B"] if i not in WAIVED]
# the plain final configuration's budget only, recomputed with --no-dflash-estimate (no drafter metadata fetched)
EXPERIMENT_BUDGET_CONFIGS = ("configs/budget/qwen3-4b_final.yaml",)

STOP = {"signal": None}  # set by SIGINT/SIGTERM: finish the current item gracefully, then stop


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str) + "\n")
    tmp.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def worst(statuses):
    statuses = [s for s in statuses if s]
    return max(statuses, key=lambda s: SEVERITY[s]) if statuses else "NOT_TESTED"


# --- study, configs, provenance ----------------------------------------------------------------------------------

def load_study():
    study = yaml.safe_load(STUDY.read_text())
    if inc := study.get("include"):  # entries reused verbatim from another study (scripts/experiment.py does the same)
        other = yaml.safe_load((STUDY.parent / inc["study"]).read_text())
        by_id = {e["id"]: e for e in other["entries"]}
        study["entries"] = [by_id[i] for i in inc["entries"] if i in by_id] + list(study.get("entries") or [])
    entries = {e["id"]: e for e in study["entries"]}
    check_no_opt_in(entries)
    return study, entries


def check_no_opt_in(entries, stage_entries=None):
    """The default protocol never runs an opt_in entry (e.g. the 210-update extension H12.train_extended): a stage list
    naming one, or an override (ASSIGNMENT_STAGE_ITEMS) doing so, is refused before anything runs."""
    lists = dict(stage_entries or STAGE_ENTRIES)
    if os.environ.get("ASSIGNMENT_STAGE_ITEMS"):
        lists.update(json.loads(os.environ["ASSIGNMENT_STAGE_ITEMS"]))
    bad = sorted({i for ids in lists.values() for i in ids if (entries.get(i) or {}).get("opt_in")
                  and (entries.get(i) or {}).get("protocol") != MODE})
    if bad:
        raise SystemExit(f"opt-in entries {bad} are never part of the default assignment protocol; run them with "
                         f"`python scripts/experiment.py run --entry <id>`")


def cap_bytes(study):
    return float(study["gpu_memory_cap"]["bytes"])


def tool(name, default):
    """argv prefix of an external tool; ASSIGNMENT_TOOL_OVERRIDES (JSON {name: "command ..."}) replaces it in tests."""
    over = json.loads(os.environ.get("ASSIGNMENT_TOOL_OVERRIDES") or "{}")
    return over[name].split() if name in over else default


def dispatcher():
    cmd = os.environ.get("ASSIGNMENT_DISPATCHER")
    return cmd.split() if cmd else [sys.executable, str(ROOT / "scripts" / "experiment.py"), "--study", str(STUDY)]


def mode_entries(entries):
    """The study entries this mode can run (experiment mode: its own stage lists only, so stage A resolves no DFlash
    config and records no drafter); full mode: every entry, as before."""
    if MODE == "full":
        return entries
    ids = {i for s in STAGES for i in stage_ids(s)}
    return {k: v for k, v in entries.items() if k in ids}


def config_paths(entries):
    paths = set(FINAL_RUNS.values())
    for e in entries.values():
        paths |= set(e.get("configs", []))
        for s in e["steps"]:
            argv = s["argv"] if isinstance(s, dict) else s
            paths |= {argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--config" and argv[i + 1].endswith(".yaml")
                      and not argv[i + 1].startswith("runs/")}
    return sorted(p for p in paths if (ROOT / p).exists())


def resolved_configs(paths):
    """{path: {sha256, resolved}} with qwen_grpo.config (the trainer's own loader and hash)."""
    from qwen_grpo import config
    out = {}
    for p in paths:
        try:
            cfg = config.load(ROOT / p)
            out[p] = {"sha256": config.sha256(cfg), "resolved": config.as_dict(cfg)}
        except Exception as e:  # noqa: BLE001 - recorded; budget configs use another schema
            out[p] = {"error": f"{type(e).__name__}: {e}", "file_sha256": sha256_file(ROOT / p)}
    return out


EXPERIMENT_IDENTITY = ("git_commit", "experiment_code_sha256", "study_sha256")


def code_identity():
    """Two identities. The experiment identity (commit; experiment_code_sha256 = the content of every tracked and
    untracked code file, qwen_grpo.evidence.code_sha256; study and resolved config hashes) must match exactly for a
    resume: otherwise earlier attempts' evidence would describe other code. report_code_sha256 identifies the report
    generator alone: when only it changes, a resume re-runs no experiment and the report is regenerated."""
    from qwen_grpo.evidence import code_sha256, report_code_sha256
    _, entries = load_study()
    configs = resolved_configs(config_paths(entries))
    tracked = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": (git("rev-parse", "HEAD") or "").strip() or None,
            "experiment_code_sha256": code_sha256(ROOT), "report_code_sha256": report_code_sha256(ROOT),
            "tracked_dirty": None if tracked is None else bool(tracked.strip()), "study_sha256": sha256_file(STUDY),
            "config_sha256": {p: c.get("sha256") or c.get("file_sha256") for p, c in configs.items()}}


def probe():
    if os.environ.get("ASSIGNMENT_PROBE_JSON"):
        return json.loads(Path(os.environ["ASSIGNMENT_PROBE_JSON"]).read_text())
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment", ROOT / "scripts" / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.probe()


# --- state -------------------------------------------------------------------------------------------------------

def state_path(run_dir):
    return Path(run_dir) / "state.json"


def load_state(run_dir):
    return read_json(state_path(run_dir))


def save_state(run_dir, state):
    state["updated"] = utc()
    write_json(state_path(run_dir), state)


def last_attempt(state, stage):
    att = state["stages"].get(stage, {}).get("attempts", [])
    return att[-1] if att else None


def item(state, stage, item_id):
    a = last_attempt(state, stage)
    return next((i for i in (a or {}).get("items", []) if i["id"] == item_id), None) if a else None


def facts(state):
    a = last_attempt(state, "A")
    return (a or {}).get("facts", {})


# --- running one item --------------------------------------------------------------------------------------------

def _forward(signum, frame):
    STOP["signal"] = signal.Signals(signum).name


def install_signals():
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _forward)


HEARTBEAT_S = float(os.environ.get("ASSIGNMENT_HEARTBEAT_S", "30"))
STEP_LINE = re.compile(r"^step\s+(\d+)\s+\(attempt")
EVIDENCE_LINE = re.compile(r"-> evidence/(\S+)")


def descendants(pid):
    """[(pid, command line)] of every live descendant of `pid` (Linux /proc), parents first."""
    kids = {}
    for p in Path("/proc").glob("[0-9]*"):
        try:
            ppid = int((p / "stat").read_text().rsplit(")", 1)[1].split()[1])
            kids.setdefault(ppid, []).append((int(p.name), (p / "cmdline").read_bytes().replace(b"\0", b" ")
                                              .decode(errors="replace").strip()))
        except (OSError, IndexError, ValueError):
            continue
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(0), []):
            out.append(c)
            todo.append(c[0])
    return out


class Live(threading.Thread):
    """Streams an item's output files to the terminal as they grow (stdout.log, stderr.log, and the step logs of the
    dispatcher's evidence directory), with a heartbeat; keeps <run>/active.json current. Reads files only: the raw
    logs are written by run_item_quiet and never modified."""

    def __init__(self, out_dir, argv):
        super().__init__(daemon=True)
        self.out_dir, self.argv, self.t0, self.done = Path(out_dir), argv, time.time(), threading.Event()
        self.run_dir = next((p for p in self.out_dir.parents if (p / "state.json").exists()), None)
        self.files, self.step, self.pids, self.last_out = {}, None, [], time.time()
        self.started = utc()

    def tail(self, path, label):
        pos, buf = self.files.get(path, (0, b""))
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                data = f.read()
        except OSError:
            return
        lines = (buf + data).replace(b"\r", b"\n").split(b"\n")
        self.files[path] = (pos + len(data), lines.pop())
        for raw in lines:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            self.last_out = time.time()
            if m := STEP_LINE.match(line):
                self.step = int(m[1])
            if m := EVIDENCE_LINE.search(line):
                for root in (EVIDENCE, ROOT / "evidence"):
                    if (root / m[1]).exists():
                        self.files.setdefault(root / m[1] / "raw", None)
            print(f"   {label}| {line}", flush=True)

    def poll(self):
        for name in ("stdout.log", "stderr.log"):
            self.tail(self.out_dir / name, "" if name == "stdout.log" else "stderr")
        for raw_dir in [k for k, v in self.files.items() if v is None]:  # an evidence raw/ directory: its step logs
            for log in sorted(raw_dir.glob("step*.log")):
                self.tail(log, log.stem)
            self.files[raw_dir] = None

    def active(self, final=False):
        if self.run_dir is None:
            return
        rec = {"item": self.out_dir.name, "argv": self.argv, "raw_dir": rel(self.out_dir), "started_utc": self.started,
               "elapsed_s": round(time.time() - self.t0, 1), "orchestrator_pid": os.getpid(),
               "pids": [{"pid": p, "cmd": c[:300]} for p, c in self.pids], "last_step": self.step,
               "updated_utc": utc(), "running": not final}
        write_json(self.run_dir / "active.json", rec)

    def heartbeat(self):
        self.pids = descendants(os.getpid())
        main = next((f"{p} ({c[:120]})" for p, c in reversed(self.pids) if "run.py" in c or "pytest" in c), None)
        e = int(time.time() - self.t0)
        print(f"   [heartbeat {utc()}] {self.out_dir.name}: elapsed {e // 60}m{e % 60:02d}s, pids "
              f"{[p for p, _ in self.pids if 'nvidia-smi' not in _]}" + (f", workload pid {main}" if main else "")
              + f", trainer step {self.step if self.step is not None else '-'}, last output "
              f"{int(time.time() - self.last_out)}s ago", flush=True)
        self.active()

    def run(self):
        print(f"-> {self.out_dir.name}: started {self.started}\n   $ {' '.join(map(str, self.argv))}", flush=True)
        time.sleep(0.5)
        self.pids = descendants(os.getpid())
        print(f"   pid {[p for p, c in self.pids if 'nvidia-smi' not in c][:4]}", flush=True)
        self.active()
        beat = time.time()
        while not self.done.wait(0.5):
            self.poll()
            if time.time() - beat >= HEARTBEAT_S:
                beat = time.time()
                self.heartbeat()

    def stop(self):
        self.done.set()
        self.join(timeout=5)
        self.poll()
        self.active(final=True)
        print(f"<- {self.out_dir.name}: ended after {time.time() - self.t0:.0f} s", flush=True)


def run_item(out_dir, argv, env=None, cap=None, telemetry_on=True):
    """run_item_quiet with the item's output streamed live (Live)."""
    live = Live(out_dir, argv)
    live.start()
    try:
        return run_item_quiet(out_dir, argv, env=env, cap=cap, telemetry_on=telemetry_on)
    finally:
        live.stop()


def run_item_quiet(out_dir, argv, env=None, cap=None, telemetry_on=True):
    """Run `argv` from the repository root in its own process group, under telemetry. Returns the record. On a
    watchdog violation or a stop signal: SIGINT to the group, SIGTERM after GRACE_S, SIGKILL after 1.5 GRACE_S."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)  # never reused: an attempt gets a new directory
    rec = {"argv": [rel(a) if str(a).startswith(str(ROOT)) else str(a) for a in argv], "started_utc": utc(),
           "stdout": rel(out_dir / "stdout.log"), "stderr": rel(out_dir / "stderr.log"), "raw_dir": rel(out_dir)}
    run_env = {**os.environ, "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
               **(env or {})}
    violation = {}
    tel = telemetry.Telemetry(out_dir / "telemetry", cap_bytes=cap, on_violation=violation.update) if telemetry_on \
        else None
    t0 = time.time()
    with open(out_dir / "stdout.log", "wb") as so, open(out_dir / "stderr.log", "wb") as se:
        if tel:
            tel.start()
        try:
            child = subprocess.Popen(argv, cwd=ROOT, env=run_env, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        except OSError as e:
            if tel:
                tel.stop()
            rec.update(exit=None, error=f"{type(e).__name__}: {e}", ended_utc=utc(), seconds=0.0)
            return rec
        if tel:
            tel.attach(child.pid)
        escalation, sent = None, []
        while child.poll() is None:
            if (violation or STOP["signal"]) and escalation is None:
                escalation = time.time()
                sent.append(("SIGINT", utc()))
                _killpg(child.pid, signal.SIGINT)
            elif escalation is not None:
                waited = time.time() - escalation
                if waited > GRACE_S and len(sent) == 1:
                    sent.append(("SIGTERM", utc()))
                    _killpg(child.pid, signal.SIGTERM)
                elif waited > 1.5 * GRACE_S and len(sent) == 2:
                    sent.append(("SIGKILL", utc()))
                    _killpg(child.pid, signal.SIGKILL)
            time.sleep(0.2)
        # the group may still hold processes (a child that outlived its parent): end them before telemetry stops
        _killpg(child.pid, signal.SIGTERM)
        if tel:
            tel.stop()
    rec.update(exit=child.returncode, ended_utc=utc(), seconds=round(time.time() - t0, 1),
               signals_sent=sent, interrupted=bool(STOP["signal"]), stop_signal=STOP["signal"],
               violation=violation or None)
    if tel:
        rec["telemetry"] = telemetry.summarize(out_dir / "telemetry", cap_bytes=cap)
        if cap is not None:
            rec["memory_status"] = rec["telemetry"]["memory_status"]
    return rec


def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


# --- dispatcher entries ------------------------------------------------------------------------------------------

DONE_LINE = re.compile(r"^(?P<id>\S+): DONE \(.*?\): (?P<outcome>PASS|FAIL|BLOCKED|NOT TESTED), criterion (?P<crit>\S+) - "
                       r"(?P<detail>.*)$")


def evidence_dirs(entry_id):
    return {p for p in EVIDENCE.glob(f"*_{entry_id}") if (p / "results.json").exists()} if EVIDENCE.exists() else set()


def entry_outputs(entry):
    """What the entry produces, as `run:<name>` or a repository path (the dispatcher's `outputs`)."""
    outs = []
    if (entry.get("judge") in ("lab", "lab_fault")):
        outs.append(f"research/l2l/experiments/{entry['name']}")
    for s in entry["steps"]:
        if isinstance(s, dict) and s.get("creates"):
            outs.append(s["creates"].replace("{name}", entry["name"]))
    return outs


def repo_path(o):
    """A study output path; research/l2l/experiments/ honours ASSIGNMENT_EXPERIMENTS_ROOT."""
    prefix = "research/l2l/experiments/"
    return EXP_DIR / o[len(prefix):] if o.startswith(prefix) else ROOT / o


def output_exists(o):
    if o.startswith("run:"):
        m = read_json(RUNS_DIR / o[4:] / "manifest.json")
        return bool(m) and m.get("status") in ("completed", "complete")
    return repo_path(o).exists()


def produced_here(state, entry):
    outs = entry_outputs(entry)
    return bool(outs) and all(o in state.get("produced", []) or (o.startswith("run:") and o[4:] in state.get("started_runs", []))
                              for o in outs)


def run_configs(entry):
    """{run name: config path} of the training runs an entry creates."""
    out = {}
    for s in entry["steps"]:
        if isinstance(s, dict) and str(s.get("creates", "")).startswith("run:"):
            out[s["creates"][4:]] = s["argv"][s["argv"].index("--config") + 1]
    return out


def run_ok(name):
    """(completed with frozen weights unchanged and a final checkpoint, reason), as scripts/experiment.py run_state."""
    m = read_json(RUNS_DIR / name / "manifest.json")
    if m is None:
        return False, f"runs/{name} does not exist"
    if m.get("status") not in ("completed", "complete"):
        return False, f"runs/{name} is {m.get('status')}"
    log = read_json(RUNS_DIR / name / "training" / "training.json") or {}
    if log.get("frozen_changed") != []:
        return False, f"runs/{name}: frozen weights not confirmed unchanged ({log.get('frozen_changed')!r})"
    if not log.get("final_checkpoint"):
        return False, f"runs/{name}: no final checkpoint"
    return True, f"runs/{name} completed, frozen weights unchanged"


def resume_runs(state, entry, cap, attempt_dir):
    """Resume the entry's interrupted training runs from their latest valid checkpoint (the trainer validates it and
    the streamed sampler state) - only with the exact config hash and commit they started with. Returns item records."""
    from qwen_grpo import config
    items, ident = [], state["identity"]
    for name, cfg_path in run_configs(entry).items():
        m = read_json(RUNS_DIR / name / "manifest.json")
        if m is None:
            continue
        rid = f"resume:{name}"
        current = config.sha256(config.load(ROOT / cfg_path))
        if m.get("config_sha256") != current:
            items.append({"id": rid, "kind": "resume", "status": "FAIL",
                          "detail": f"stale configuration: runs/{name} was trained with config sha "
                                    f"{str(m.get('config_sha256'))[:12]}, {cfg_path} resolves to {current[:12]}; refusing "
                                    "to resume or reuse it (move it aside to train again)"})
            continue
        if m.get("status") in ("completed", "complete"):
            continue
        if name in state.get("memory_violations", []):
            items.append({"id": rid, "kind": "resume", "status": "FAIL",
                          "detail": f"runs/{name} exceeded the GPU memory cap earlier in this run: not resumed"})
            continue
        if not config.load(ROOT / cfg_path).streaming.enabled:
            # trainer.save publishes crash-consistent checkpoints (COMMIT.json, sampler state) only for streaming runs;
            # a non-streaming checkpoint can be torn, and resuming such runs is not what H08 tested
            items.append({"id": rid, "kind": "resume", "status": "BLOCKED",
                          "detail": f"runs/{name} is {m.get('status')} and {cfg_path} has no disk streaming: its "
                                    "checkpoints are not crash-consistent (no COMMIT.json), so it is not resumed. Move "
                                    f"it aside (python scripts/run.py remove --run {name} --confirm) and run the entry "
                                    "again from the start."})
            continue
        if name not in state.get("started_runs", []):
            items.append({"id": rid, "kind": "resume", "status": "BLOCKED",
                          "detail": f"runs/{name} is {m.get('status')} but was started outside this assignment run: its "
                                    "manifest records only a commit and a dirty flag, so the code it ran cannot be "
                                    "verified. Resume it from the assignment run that started it, or move it aside."})
            continue
        if m.get("git_commit") != ident["git_commit"]:
            items.append({"id": rid, "kind": "resume", "status": "FAIL",
                          "detail": f"runs/{name} was started at commit {str(m.get('git_commit'))[:10]}, the code is now "
                                    f"{str(ident['git_commit'])[:10]}: refusing to resume across code versions"})
            continue
        rec = run_item(Path(attempt_dir) / f"resume_{name}",
                       [*tool("run_py", [sys.executable, str(ROOT / "scripts" / "run.py")]), "train", "--config",
                        f"runs/{name}/config.yaml", "--resume", "latest"], cap=cap)
        rec.update(id=rid, kind="resume", run=name,
                   origin="started by this assignment run" if name in state.get("started_runs", [])
                   else "started outside this assignment run (same commit and config hash)")
        if rec.get("memory_status") == "FAIL":
            state.setdefault("memory_violations", []).append(name)
            rec["status"], rec["detail"] = "FAIL", f"GPU memory cap exceeded: {rec['telemetry']['memory_reason']}"
        elif rec.get("interrupted"):
            rec["status"], rec["detail"] = "INCONCLUSIVE", f"interrupted by {rec['stop_signal']}"
        else:
            ok, why = run_ok(name)
            rec["status"] = "PASS" if rec["exit"] == 0 and ok else "FAIL"
            rec["detail"] = f"resumed from the latest valid checkpoint; exit {rec['exit']}; {why}"
        state.setdefault("resumes", []).append({"run": name, "utc": rec.get("started_utc"), "status": rec["status"]})
        items.append(rec)
    return items


def gpu_name_of(o):
    """The GPU an existing output was produced on, from its own provenance (None if it does not say)."""
    if o.startswith("run:"):
        log = read_json(RUNS_DIR / o[4:] / "training" / "training.json") or {}
        return ((log.get("environment") or {}).get("gpu") or {}).get("name"), log.get("config_sha256")
    p = repo_path(o)
    data = None
    if p.is_dir():
        from qwen_grpo.evidence import read_result
        data = read_result(p)
    elif p.exists():
        data = read_json(p)
    stack = [data]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            g = x.get("gpu")
            if isinstance(g, str):
                return g, None
            if isinstance(g, dict) and isinstance(g.get("name"), str):
                return g["name"], None
            stack += [v for v in x.values() if isinstance(v, dict)]
    return None, None


HELDOUT_COMPARED = ("run", "control", "checkpoint_adapter_sha256", "control_adapter_sha256", "config_sha256",
                    "dataset_sha256", "split", "n_test", "code_sha256", "model_revisions", "overrides")


def latest_checkpoint_dir(name):
    """The checkpoint final_eval evaluates (qwen_grpo.runs.latest_checkpoint semantics, under RUNS_DIR)."""
    d = RUNS_DIR / name / "checkpoints"
    steps = sorted((int(p.name[4:]), p) for p in d.glob("step*") if p.name[4:].isdigit()) if d.exists() else []
    return steps[-1][1] if steps else None


def heldout_identity_expected():
    """What H12.final_eval's result must be evidence of now: the current adapters of the final and control runs, the
    final config, the data file and test split, this experiment code, the cached base-model revision, no overrides.
    A value that cannot be established is None (and then nothing is reused)."""
    from qwen_grpo import config
    from qwen_grpo.evidence import code_sha256, file_sha256
    from qwen_grpo.sentiment import SPLITS_PATH, load_splits
    from qwen_grpo import trainer
    cfg = config.load(ROOT / FINAL_RUNS[PRIMARY])
    sha = {}
    for name in FINAL_RUNS:
        ck = latest_checkpoint_dir(name)
        f = ck / "adapter_model.safetensors" if ck else None
        sha[name] = file_sha256(f) if f and f.exists() else None
    return {"run": PRIMARY, "control": CONTROL,
            "checkpoint_adapter_sha256": sha[PRIMARY], "control_adapter_sha256": sha[CONTROL],
            "config_sha256": config.sha256(cfg), "dataset_sha256": file_sha256(SPLITS_PATH), "split": "test",
            "n_test": len(load_splits()["test"]), "code_sha256": code_sha256(ROOT),
            "model_revisions": {m: trainer.model_revision(m) for m in trainer.models(cfg)},
            "overrides": {"storage": None, "n": None, "step": None}}


def _short(v):
    return v[:12] if isinstance(v, str) else v


def heldout_problems(data):
    got = (data or {}).get("identity")
    if not isinstance(got, dict):
        return ["records no evaluation identity (written before identities existed, or by other code)"]
    exp, out = heldout_identity_expected(), []
    for k in HELDOUT_COMPARED:
        if exp.get(k) is None or (k == "model_revisions" and None in exp[k].values()):
            out.append(f"{k}: cannot be established now ({exp.get(k)!r})")
        elif got.get(k) != exp[k]:
            out.append(f"{k}: evidence {_short(got.get(k))!r} != current {_short(exp[k])!r}")
    return out


def _find(data, key):
    stack = [data]
    while stack:
        x = stack.pop(0)
        if isinstance(x, dict):
            if key in x and not isinstance(x[key], dict):
                return x[key]
            stack += [v for v in x.values() if isinstance(v, dict)]
    return None


def foreign_check(entry, cap_facts):
    """For outputs that exist but were not produced by this assignment run: (ok, detail). Reuse needs the evidence's
    own identity to match the present: the GPU model, the experiment code by content (qwen_grpo.evidence.code_sha256),
    a training run's config hash, and for the held-out evaluation every field of heldout_identity_expected()."""
    from qwen_grpo import config
    from qwen_grpo.evidence import code_sha256, read_result
    problems, notes = [], []
    gpu, code = cap_facts.get("gpu"), code_sha256(ROOT)
    for o in entry_outputs(entry):
        if not output_exists(o):
            continue
        if o.startswith("run:"):
            data = read_json(RUNS_DIR / o[4:] / "training" / "training.json") or {}
        else:
            p = repo_path(o)
            data = read_result(p) if p.is_dir() else read_json(p)
        name = gpu_name_of(o)[0]
        if name is None:
            problems.append(f"{o}: records no GPU")
        elif gpu is None or name != gpu.split(" (")[0]:
            problems.append(f"{o}: produced on {name}, this host has {gpu}")
        if entry["id"] == EVAL_ID:
            problems += [f"{o}: {p_}" for p_ in heldout_problems(data)]
        else:
            ev_code = _find(data, "code_sha256")
            if ev_code is None or code is None or ev_code != code:
                problems.append(f"{o}: experiment code {_short(ev_code)!r} != current {_short(code)!r}")
        if o.startswith("run:") and MODE == "full" and (by := started_in_other_modes(o[4:])):
            problems.append(f"{o}: trained by {', '.join(by)}; a full qualification run never reuses training from "
                            "another mode")
        if o.startswith("run:"):
            cfg_path = run_configs(entry).get(o[4:])
            if cfg_path and config.sha256(config.load(ROOT / cfg_path)) != data.get("config_sha256"):
                problems.append(f"{o}: trained with config sha {_short(data.get('config_sha256'))}, {cfg_path} now differs")
        if not problems:
            notes.append(f"{o}: same GPU model ({name}), same experiment code and configuration")
    return (not problems), "; ".join(problems or notes)


def mode_of_state(s):
    return (s or {}).get(MODE_KEY) or "full"


def started_in_other_modes(name):
    """'<run-id> (<mode>)' of every assignment run under RESULTS, in another mode than this one, that started runs/<name>."""
    out = []
    for d in sorted(RESULTS.glob("*")) if RESULTS.exists() else []:
        s = read_json(d / "state.json")
        if s and mode_of_state(s) != MODE and name in s.get("started_runs", []):
            out.append(f"assignment run {d.name} ({mode_of_state(s)} mode)")
    return out


def run_entry(run_dir, state, stage, entry_id, entries, cap, cap_facts, attempt_dir):
    """The entry through the dispatcher, plus any resume items its training runs need. Returns a list of records."""
    entry = entries.get(entry_id)
    if entry is None:
        return [{"id": entry_id, "kind": "entry", "status": "NOT_TESTED", "detail": f"{entry_id} is not in {rel(STUDY)}"}]
    cap = cap if stage in CAPPED else None
    pre = resume_runs(state, entry, cap, attempt_dir)
    save_state(run_dir, state)
    bad = [i for i in pre if i["status"] != "PASS"]
    if bad:
        return pre + [blocked(entry_id, f"not started: {bad[0]['id']} is {bad[0]['status']}: {bad[0]['detail']}")]
    outs = entry_outputs(entry)
    existing = bool(outs) and all(output_exists(o) for o in outs)
    again, reused_note = False, None
    has_runs = any(o.startswith("run:") for o in outs)
    if existing and not produced_here(state, entry):
        if stage in FRESH and not has_runs:
            again = True  # older evidence is never trusted on a new host: re-run under a new name
        else:
            ok, why = foreign_check(entry, cap_facts)
            if not ok:
                return pre + [{"id": entry_id, "kind": "entry", "status": "INCONCLUSIVE", "reused": True,
                               "detail": f"stale evidence from outside this run, not re-run (fixed names cannot be "
                                         f"renamed): {why}. Move it aside to re-run."}]
            reused_note = f"reused after provenance check: {why}"
    for name in run_configs(entry):  # runs this assignment run starts are its own from the first attempt on
        if not (RUNS_DIR / name).exists() and name not in state.setdefault("started_runs", []):
            state["started_runs"].append(name)
    save_state(run_dir, state)
    before = evidence_dirs(entry_id)
    argv = [*dispatcher(), "run", "--entry", entry_id] + (["--again"] if again else [])
    rec = run_item(attempt_dir / entry_id.replace("/", "_"), argv, cap=cap)
    rec.update(id=entry_id, kind="entry", again=again, reused=bool(reused_note),
               prior_evidence=bool(reused_note))  # outputs made outside this run, reused after the provenance check
    new = sorted(evidence_dirs(entry_id) - before)
    stdout = (ROOT / rec["stdout"]).read_text(errors="replace") if (ROOT / rec["stdout"]).exists() else ""
    if new:
        res = read_json(new[-1] / "results.json")
        rec.update(evidence_dir=rel(new[-1]), evidence_name=res.get("evidence_name"),
                   dispatcher_status=res.get("status"), criterion=res.get("criterion"), detail=res.get("detail"),
                   outputs=res.get("outputs"))
        status = DISPATCH_MAP.get(res.get("status"), "INCONCLUSIVE")
        if status == "PASS" and rec["exit"] not in (0,):
            status, rec["detail"] = "INCONCLUSIVE", f"results.json says PASS but the dispatcher exited {rec['exit']}"
    else:
        m = next((DONE_LINE.match(l) for l in stdout.splitlines() if DONE_LINE.match(l)), None)
        if m:
            status = DISPATCH_MAP[m["outcome"]]
            rec.update(dispatcher_status="DONE", criterion=m["crit"], detail=m["detail"], reused=True,
                       reuse=reused_note or "output produced earlier in this assignment run")
            if has_runs and all(o.startswith("run:") for o in outs):
                # the dispatcher does not judge existing runs (exit_codes needs a fresh execution): judge their state
                states = [run_ok(o[4:]) for o in outs]
                status = "PASS" if all(ok for ok, _ in states) else "FAIL"
                rec["detail"] = "; ".join(why for _, why in states)
        else:
            status = "FAIL"
            rec["detail"] = f"the dispatcher wrote no result (exit {rec['exit']}); see {rec['stderr']}"
    if rec.get("interrupted") and status != "PASS":
        status, rec["detail"] = "INCONCLUSIVE", f"interrupted by {rec['stop_signal']}; " + str(rec.get("detail"))
    if rec.get("memory_status") == "FAIL":
        status = "FAIL"
        rec["detail"] = f"GPU memory cap exceeded ({rec['telemetry']['memory_reason']}); workload stopped; " + \
            str(rec.get("detail"))
    if rec.get("memory_status") == "FAIL":
        state.setdefault("memory_violations", []).extend(run_configs(entry))
    rec["status"] = status
    if not again and new and not rec.get("interrupted"):
        for o in outs:  # what this run produced (a FAIL is evidence too)
            if output_exists(o) and o not in state.setdefault("produced", []):
                state["produced"].append(o)
    return pre + [rec]


def run_pytest(attempt_dir, item_id, paths, cuda):
    xml = attempt_dir / item_id / "junit.xml"
    env = {} if cuda else {"CUDA_VISIBLE_DEVICES": ""}
    argv = [*tool("pytest", [sys.executable, "-m", "pytest"]), "-q", "-p", "no:cacheprovider", *paths,
            f"--junitxml={xml}"]
    rec = run_item(attempt_dir / item_id, argv, env=env, telemetry_on=cuda)
    rec.update(id=item_id, kind="pytest", cuda=cuda)
    counts = junit_counts(xml)
    rec["junit"] = counts
    if rec.get("interrupted"):
        rec["status"], rec["detail"] = "INCONCLUSIVE", f"interrupted by {rec['stop_signal']}"
    elif counts is None:
        rec["status"], rec["detail"] = "FAIL", f"no junit report (exit {rec['exit']})"
    elif rec["exit"] != 0 or counts["failures"] or counts["errors"]:
        rec["status"], rec["detail"] = "FAIL", f"exit {rec['exit']}: {counts}"
    elif counts["tests"] == counts["skipped"]:
        rec["status"], rec["detail"] = "INCONCLUSIVE", f"every test skipped: {counts}"
    else:
        rec["status"], rec["detail"] = "PASS", f"{counts['tests'] - counts['skipped']} passed, {counts['skipped']} skipped"
    return rec


def junit_counts(xml):
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(xml).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}


def blocked(item_id, why, kind="entry"):
    return {"id": item_id, "kind": kind, "status": "BLOCKED", "detail": why}


# --- stages ------------------------------------------------------------------------------------------------------

def stage_A(run_dir, state, attempt_dir, study, entries):
    items, f = [], {}
    env_dir = Path(run_dir) / "environment"
    env_dir.mkdir(exist_ok=True)
    cap = probe()
    f.update({k: cap.get(k) for k in ("cuda", "bf16", "gpu", "capability", "vram_gib", "per_process_smi", "linux_native",
                                       "fixture_packages", "packages", "platform", "cuda_error", "host_ram_gib")})
    write_json(env_dir / "capabilities.json", cap)
    # host resources
    mem = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    disk = {}
    for name, p in (("repository", ROOT), ("hf_home", hf if hf.exists() else hf.parent)):
        try:
            disk[name] = {"path": str(p), "free_bytes": shutil.disk_usage(p).free}
        except OSError as e:
            disk[name] = {"path": str(p), "error": str(e)}
    host = study.get("host_requirements", {})
    ram_need, disk_need = float(host.get("ram_bytes_4b", 32e9)), float(host.get("disk_bytes_4b", 40e9))
    f["host"] = {"meminfo": mem, "disk": disk, "ram_required_bytes": ram_need, "disk_required_bytes": disk_need}
    f["ram_ok"] = mem.get("MemTotal", 0) >= ram_need * 0.97  # "32 GB" hosts report a little less as MemTotal
    f["disk_ok"] = all(d.get("free_bytes", 0) >= disk_need for d in disk.values())
    write_json(env_dir / "host.json", f["host"])
    # nvidia-smi identity (driver, CUDA, compute capability)
    smi = telemetry.smi_binary()
    if smi:
        q = subprocess.run([smi, "--query-gpu=name,driver_version,compute_cap,memory.total,pci.bus_id",
                            "--format=csv"], capture_output=True, text=True)
        v = subprocess.run([smi, "--version"], capture_output=True, text=True)
        (env_dir / "nvidia_smi_identity.txt").write_text(f"$ nvidia-smi --version\n{v.stdout}{v.stderr}\n$ nvidia-smi "
                                                         f"--query-gpu=...\n{q.stdout}{q.stderr}")
        f["nvidia_smi"] = {"version": v.stdout.strip(), "identity": q.stdout.strip()}
    else:
        f["nvidia_smi"] = None
    # CUDA streams/events + pinned copies, run in a child under telemetry: this is also the check that nvidia-smi
    # reports this pipeline's processes with a numeric per-process figure
    rec = run_item(attempt_dir / "cuda_check", [*tool("cuda_check", [sys.executable, str(ROOT / "scripts" / "assignment_cuda_check.py")]),
                                                "--out", str(attempt_dir / "cuda_check" / "result.json")],
                   cap=cap_bytes(study))
    out = read_json(Path(attempt_dir) / "cuda_check" / "result.json") or {}
    rec.update(id="cuda_streams_events", kind="check", result=out)
    tel = rec.get("telemetry") or {}
    f["telemetry_per_process_numeric"] = bool(tel.get("attributed_numeric_samples"))
    if not f.get("cuda"):
        rec["status"], rec["detail"] = "BLOCKED", f"no usable CUDA GPU ({cap.get('cuda_error') or cap.get('gpu')})"
    elif out.get("ok"):
        rec["status"], rec["detail"] = "PASS", out.get("detail")
    else:
        rec["status"], rec["detail"] = "FAIL", out.get("detail") or f"check exited {rec['exit']}"
    f["cuda_streams_ok"] = rec["status"] == "PASS"
    items.append(rec)
    # resolved configs, seeds, models, dataset
    rc_dir = Path(run_dir) / "resolved_configs"
    rc_dir.mkdir(exist_ok=True)
    configs = resolved_configs(config_paths(mode_entries(entries)))
    for p, c in configs.items():
        (rc_dir / (p.replace("/", "__"))).write_text(
            f"# {p} sha256 {c.get('sha256') or c.get('file_sha256')}\n"
            + (yaml.safe_dump(c["resolved"], sort_keys=False) if "resolved" in c else f"# {c['error']}\n"))
    write_json(rc_dir / "hashes.json", {p: {k: v for k, v in c.items() if k != "resolved"} for p, c in configs.items()})
    from qwen_grpo.trainer import model_revision
    models = sorted({c["resolved"]["model"]["name"] for c in configs.values() if "resolved" in c}
                    | {c["resolved"]["dflash"]["draft_model"] for c in configs.values()
                       if "resolved" in c and c["resolved"].get("dflash", {}).get("draft_model")})
    f["models"] = {m: {"cached_revision": model_revision(m)} for m in models}
    f["dataset"] = {"file": "data/sentiment.json", "sha256": sha256_file(ROOT / "data" / "sentiment.json")}
    f["seeds"] = {p: {"training.seed": c["resolved"]["training"]["seed"], "data.subset_seed": c["resolved"]["data"]["subset_seed"]}
                  for p, c in configs.items() if "resolved" in c}
    write_json(env_dir / "models_data_seeds.json", {k: f[k] for k in ("models", "dataset", "seeds")})
    items.append(assignment_config_check(configs))
    f["assignment_config"] = items[-1]["status"]
    # the study planner (read-only)
    rec = run_item(attempt_dir / "study_plan", [*dispatcher(), "plan", "--target", "local"], telemetry_on=False)
    rec.update(id="study_plan", kind="command", status="PASS" if rec["exit"] == 0 else "FAIL",
               detail=f"exit {rec['exit']}")
    items.append(rec)
    items.append(predictions(run_dir, attempt_dir))
    # blockers
    blockers = []
    if not f.get("cuda"):
        blockers.append(("C-G", f"no usable CUDA GPU: {cap.get('cuda_error') or 'torch.cuda.is_available() is False'}"))
    elif not f.get("bf16"):
        blockers.append(("C-G", f"{cap.get('gpu')} has no native bf16 (compute capability {cap.get('capability')})"))
    if f.get("cuda") and not f["cuda_streams_ok"]:
        blockers.append(("C-G", "CUDA streams/events check failed"))
    if not f["ram_ok"]:
        blockers.append(("D-G", f"host RAM {mem.get('MemTotal', 0) / 1e9:.1f} GB < {ram_need / 1e9:.0f} GB"))
    if not f["disk_ok"]:
        blockers.append(("D-G", f"free disk {disk} < {disk_need / 1e9:.0f} GB"))
    if not f.get("linux_native"):
        blockers.append(("D-G", "not native Linux (WSL2): no per-process nvidia-smi memory"))
    if f.get("cuda") and not f.get("per_process_smi"):
        blockers.append(("F", "nvidia-smi does not list this process with a numeric memory figure: the 8 GB cap "
                              "cannot be verified, so D's profile is INCONCLUSIVE and the final run is not started. "
                              "Use a host whose container shows its PIDs to nvidia-smi."))
    if f.get("cuda") and not f.get("fixture_packages"):
        blockers.append(("B", f"package versions differ from the lab fixtures ({cap.get('fixture_packages_saved')}): "
                              "the lab equivalence cases are BLOCKED, so F is not started"))
    f["blockers"] = [{"stages": s, "reason": r} for s, r in blockers]
    write_json(env_dir / "blockers.json", f["blockers"])
    return items, f


def assignment_config_check(configs):
    """The assignment's fixed choices, read from the resolved final config (not from documentation)."""
    c = configs.get(FINAL_RUNS[PRIMARY], {}).get("resolved")
    if c is None:
        return {"id": "assignment_config", "kind": "check", "status": "FAIL", "detail": "final config does not resolve"}
    checks = {
        "model is Qwen/Qwen3-4B": c["model"]["name"] == "Qwen/Qwen3-4B",
        "bf16 base weights (no quantization)": c["model"]["dtype"] == "bfloat16",
        "1-2k training prompts": 1000 <= c["data"]["train_samples"] <= 2000,
        "GRPO groups (G >= 2)": c["training"]["group_size"] >= 2,
        "LoRA adapter (rank > 0)": c["lora"]["rank"] > 0,
        "L2L backend": c["execution"]["backend"] in ("l2l_sync", "l2l_async"),
        "disk streaming enabled": bool(c.get("streaming", {}).get("enabled")),
        "memory guard on": bool(c.get("memory", {}).get("guard")),
    }
    if MODE == "experiment":  # plain L2L rollouts by design; everything else is the full mode's final configuration
        checks["plain L2L rollouts (dflash disabled, no drafter)"] = (c.get("dflash") or {}).get("enabled") is False \
            and not (c.get("dflash") or {}).get("draft_model")
        checks.update(experiment_consistency(c))
    else:
        checks["DFlash rollouts (z-lab/Qwen3-4B-DFlash-b16)"] = (c.get("dflash") or {}).get("enabled") is True \
            and (c.get("dflash") or {}).get("draft_model") == "z-lab/Qwen3-4B-DFlash-b16"
    lr0 = configs.get(FINAL_RUNS[CONTROL], {}).get("resolved") or {}
    checks["LR=0 control: learning_rate 0"] = (lr0.get("training") or {}).get("learning_rate") == 0
    checks["LR=0 control: same rollout generator as the final run (dflash section)"] = lr0.get("dflash") == c.get("dflash")
    checks["LR=0 control: same execution, batch shape and data as the final run"] = bool(lr0) and all(
        lr0.get(k) == c.get(k) for k in ("model", "lora", "data", "execution", "streaming", "memory")) and all(
        (lr0.get("training") or {}).get(k) == c["training"].get(k)
        for k in ("batch_size", "group_size", "max_new_tokens", "micro_batch_size", "num_iterations", "seed"))
    notes = ["non-thinking template: sentiment.build_prompts appends the empty <think></think> block (asserted in code, "
             "tests/test_sentiment.py)", "no SFT stage: the pipeline has none (GRPO from the base model)",
             "reward: exact canonical label (sentiment.reward), verifiable, binary",
             f"DFlash in the final run (config): {bool(c.get('dflash', {}).get('enabled'))}; " + (
                 "experiment mode: plain L2L rollouts (StreamedGenerator), DFlash intentionally excluded; that the "
                 "rollouts were plain is judged from the training logs (requirement T3)" if MODE == "experiment" else
                 "that rollouts actually ran through DFlash is judged from the training logs (requirement D0), not "
                 "from this check"),
             f"LR=0 control: {lr0.get('training', {}).get('max_steps')} steps vs {c['training']['max_steps']} trained steps "
             "(shorter schedule: matched in data, seed and batch shape, not in length)"]
    bad = [k for k, v in checks.items() if not v]
    return {"id": "assignment_config", "kind": "check", "status": "FAIL" if bad else "PASS",
            "detail": ("violated: " + ", ".join(bad)) if bad else "all fixed assignment choices hold",
            "checks": checks, "notes": notes}


def experiment_consistency(c):
    """Experiment mode: the plain primary config is the full mode's final config except run name and dflash, and the
    plain-L2L smoke (H11.smoke, qwen3_4b_smoke.yaml) runs the same training tensor shapes."""
    from qwen_grpo import config
    full = config.as_dict(config.load(ROOT / "configs" / "experiments" / "qwen3_4b_final.yaml"))
    smoke = config.as_dict(config.load(ROOT / "configs" / "experiments" / "qwen3_4b_smoke.yaml"))
    shape = ("batch_size", "group_size", "micro_batch_size", "max_new_tokens", "num_iterations")
    return {
        "plain primary = qwen3_4b_final.yaml except run name and dflash": all(
            c.get(k) == v for k, v in full.items() if k not in ("run", "dflash")),
        "plain-L2L smoke (qwen3_4b_smoke.yaml) has the final run's model, LoRA, execution, memory guard and batch "
        "shape (B, G, micro-batch, new tokens, iterations), plain rollouts": all(
            smoke[k] == c.get(k) for k in ("model", "lora", "execution", "streaming", "memory", "dflash")) and all(
            smoke["training"][k] == c["training"][k] for k in shape),
    }


def predictions(run_dir, attempt_dir):
    """Theoretical budgets before any GPU stage: the committed predictions (copied with their hashes) and a fresh
    computation (scripts/budget.py). The directory is made read-only afterwards."""
    pred = Path(run_dir) / "predictions"
    if pred.exists():  # resume: predictions are written once
        return {"id": "budget_predictions", "kind": "command", "status": "PASS", "detail": "kept from the first attempt",
                "files": sorted(rel(p) for p in pred.rglob("*") if p.is_file())}
    (pred / "committed").mkdir(parents=True)
    src = ROOT / "results" / "budget" / "predictions"
    hashes = {}
    for p in sorted(src.glob("*")) if src.exists() else []:
        shutil.copy2(p, pred / "committed" / p.name)
        hashes[p.name] = sha256_file(p)
    log = git("log", "-1", "--format=%H %cI", "--", "results/budget/predictions")
    statuses, runs = [], []
    # the final configuration's budget first; the historical Step-0 budgets are recomputed only as a record
    for cfg in BUDGET_CONFIGS:
        out = pred / "calculator"
        rec = run_item(Path(attempt_dir) / f"budget_{Path(cfg).stem}",
                       [*tool("budget", [sys.executable, str(ROOT / "scripts" / "budget.py")]), "--config", cfg,
                        "--out-dir", str(out)] + (["--no-dflash-estimate"] if MODE == "experiment" else []),
                       telemetry_on=False)
        runs.append({"config": cfg, "exit": rec["exit"], "stdout": rec["stdout"], "stderr": rec["stderr"]})
        statuses.append(rec["exit"] == 0)
    write_json(pred / "provenance.json", {"committed": {"sha256": hashes, "last_commit": (log or "").strip()},
                                          "recomputed": runs, "created_utc": utc(),
                                          "kind": "theoretical predictions, written before any GPU stage of this run"})
    for p in pred.rglob("*"):
        p.chmod(0o444 if p.is_file() else 0o555)
    pred.chmod(0o555)
    ok = all(statuses)
    if ok:
        status, detail = "PASS", "recomputed (scripts/budget.py) and the committed predictions recorded"
    elif hashes:
        status, detail = "PASS", ("recomputation failed (see its log; it needs the Hub's safetensors headers); the "
                                  "committed predictions, made before any measurement, were recorded")
    else:
        status, detail = "FAIL", "no prediction could be recorded"
    return {"id": "budget_predictions", "kind": "command", "status": status, "detail": detail, "recomputed": runs,
            "committed": hashes}


def gate(stage, state):
    """[] if `stage` may run, else the reasons it may not."""
    f, why = facts(state), []
    gpu = bool(f.get("cuda") and f.get("bf16") and f.get("cuda_streams_ok"))
    if stage in "CDEFG" and not gpu:
        why.append("stage A: no usable native-bf16 CUDA GPU (or the stream/event check failed)")
    if stage in "DEFG" and not (f.get("ram_ok") and f.get("disk_ok")):
        why.append("stage A: host RAM or disk below the Qwen3-4B requirement")
    if stage == "E":
        s = item(state, "D", "H11.smoke")
        if not s or s["status"] != "PASS" or s.get("memory_status") == "FAIL":
            why.append(f"stage D: H11.smoke is {s and s['status']} (memory {s and s.get('memory_status')})")
    if stage == "F":
        r = item(state, "A", "assignment_config")  # the resolved final configs hold the assignment's fixed choices
        if not r or r["status"] != "PASS":
            why.append(f"stage A: assignment_config is {r and r['status']}"
                       + (f" ({r.get('detail')})" if r and r.get("detail") else ""))
        for i in ESSENTIAL_B:
            r = item(state, "B", i)
            if not r or r["status"] != "PASS":
                why.append(f"stage B: {i} is {r and r['status']}")
        r = item(state, "C", "H05.real_0.6b_bf16")
        if "C" in STAGES and (not r or r["status"] != "PASS"):  # experiment mode has no stage C
            why.append(f"stage C: H05.real_0.6b_bf16 is {r and r['status']}")
        for i in ("H11.smoke", "H11.profile"):
            r = item(state, "D", i)
            if not r or r["status"] != "PASS" or r.get("memory_status") != "PASS":
                why.append(f"stage D: {i} is {r and r['status']}, nvidia-smi memory {r and r.get('memory_status')} "
                           "(the 8 GB qualification must PASS on per-process evidence)")
        if "E" not in STAGES and final_uses_dflash():  # experiment mode has no stage E: plain rollouts only
            why.append(f"{FINAL_RUNS[PRIMARY]} enables DFlash, but this mode runs no DFlash checks (stage E); its "
                       "final run must use plain L2L rollouts")
        elif final_uses_dflash():  # the final run's rollouts go through DFlash: its real-model checks must pass first
            r = item(state, "E", "H10.dflash_bench")
            if not r or r["status"] != "PASS":
                why.append(f"stage E: H10.dflash_bench is {r and r['status']} (greedy DFlash must equal plain decoding "
                           "on Qwen3-4B before the final run samples through DFlash)")
            r = item(state, "E", "H10.smoke_dflash")
            if not r or r["status"] != "PASS" or r.get("memory_status") != "PASS":
                why.append(f"stage E: H10.smoke_dflash is {r and r['status']}, nvidia-smi memory "
                           f"{r and r.get('memory_status')} (the final run's DFlash path must qualify under the cap)")
    if stage == "G":
        r = item(state, "F", TRAIN_ID)
        if not r or r["status"] != "PASS":
            why.append(f"stage F: {TRAIN_ID} is {r and r['status']}")
    return why


def final_uses_dflash():
    """dflash.enabled of the resolved final config (True if it cannot be resolved: the stricter gate)."""
    from qwen_grpo import config
    try:
        return bool(config.load(ROOT / FINAL_RUNS[PRIMARY]).dflash.enabled)
    except (Exception, SystemExit):  # noqa: BLE001
        return True


def stage_ids(stage):
    if os.environ.get("ASSIGNMENT_STAGE_ITEMS"):
        return json.loads(os.environ["ASSIGNMENT_STAGE_ITEMS"]).get(stage, STAGE_ENTRIES.get(stage, []))
    return STAGE_ENTRIES.get(stage, [])


def at_or_after(stage, ref):
    """`stage` runs at or after `ref` in this mode's order (experiment mode runs D before B)."""
    return stage in STAGES and ref in STAGES and STAGES.index(stage) >= STAGES.index(ref)


def carried(state, stage, item_id):
    """An item finished (not interrupted) in an earlier attempt of this stage, unless the stage is being forced."""
    if state.get("force_from") and at_or_after(stage, state["force_from"]):
        return None
    for a in reversed(state["stages"].get(stage, {}).get("attempts", [])[:-1]):
        for i in a.get("items", []):
            if i["id"] == item_id and not i.get("interrupted") and i["status"] != "BLOCKED" \
                    and not str(i.get("detail", "")).startswith("not started"):
                return {**i, "carried_over_from_attempt": a["n"]}
    return None


def stage_entries(stage, run_dir, state, attempt_dir, study, entries):
    f, items = facts(state), []
    for eid in stage_ids(stage):
        if STOP["signal"]:
            items.append({"id": eid, "kind": "entry", "status": "NOT_TESTED", "detail": f"not started: {STOP['signal']}"})
            continue
        prev = carried(state, stage, eid)
        if prev:
            items.append(prev)
            continue
        items += run_entry(run_dir, state, stage, eid, entries, cap_bytes(study), f, attempt_dir)
        a = last_attempt(state, stage)
        a["items"] = [i for i in a["items"] if i["kind"] != "entry" and not i["id"].startswith("resume:")] + items
        save_state(run_dir, state)
    return items


def stage_B(run_dir, state, attempt_dir, study, entries, suites=None):
    items = []
    for item_id, paths, cuda in suites if suites is not None else (("unit_tests_cpu", PYTEST_CPU, False),
                                           ("silent_failure_tests", PYTEST_SILENT, False),
                                           ("unit_tests_cuda", PYTEST_CUDA, True)):
        prev = carried(state, "B", item_id)
        if prev:
            items.append(prev)
        elif STOP["signal"]:
            items.append({"id": item_id, "kind": "pytest", "status": "NOT_TESTED", "detail": f"not started: {STOP['signal']}"})
        elif cuda and not facts(state).get("cuda"):
            items.append(blocked(item_id, "no usable CUDA GPU", "pytest"))
        else:
            items.append(run_pytest(attempt_dir, item_id, paths, cuda))
        last_attempt(state, "B")["items"] = list(items)
        save_state(run_dir, state)
    return items + stage_entries("B", run_dir, state, attempt_dir, study, entries)


def stage_B_experiment(run_dir, state, attempt_dir, study, entries):
    """The reduced lab list only (no pytest suites, no DFlash tests); the waived item's outcome is recorded as it is."""
    items = stage_B(run_dir, state, attempt_dir, study, entries, suites=())
    state["waivers"] = [{"item": i["id"], "stage": "B", "status": i["status"], "detail": i.get("detail"),
                         "evidence_dir": i.get("evidence_dir"), "waiver": WAIVED[i["id"]]}
                        for i in items if i["id"] in WAIVED]
    save_state(run_dir, state)
    return items


def stage_C(run_dir, state, attempt_dir, study, entries):
    items = stage_entries("C", run_dir, state, attempt_dir, study, entries)
    real = next((i for i in items if i["id"] == "H05.real_0.6b_bf16"), None)
    if real and real.get("dispatcher_status") == "DONE" and real["status"] == "PASS":
        # the runs exist, so the dispatcher did not compare them again: compare them now (compare_runs.py exits 0 iff
        # comparable and identical) and let that decide the item
        for b in ("qwen3_0.6b_l2l_sync_short", "qwen3_0.6b_l2l_async_short"):
            out = Path(attempt_dir) / f"compare_resident_vs_{b}"
            rec = run_item(out, [*tool("compare_runs", [sys.executable, str(ROOT / "scripts" / "compare_runs.py")]),
                                 "qwen3_0.6b_resident_short", b,
                                 "--out", str(out / "compare.json")], telemetry_on=False)
            rec.update(id=f"compare_resident_vs_{b}", kind="command",
                       status="PASS" if rec["exit"] == 0 else "FAIL", detail=f"compare_runs.py exit {rec['exit']}")
            items.append(rec)
            if rec["status"] != "PASS":
                real["status"], real["detail"] = "FAIL", f"{rec['id']}: runs differ ({rec['detail']})"
    return items


def stage_G(run_dir, state, attempt_dir, study, entries):
    ev = entries.get(EVAL_ID)
    out = ev and entry_outputs(ev)
    if out and not all(output_exists(o) for o in out):
        state.setdefault("test_set_evaluations", []).append({"utc": utc(), "attempt_dir": rel(attempt_dir)})
    return stage_entries("G", run_dir, state, attempt_dir, study, entries)


def stage_H(run_dir, state, attempt_dir, study, entries):
    """Soup's own checks, when Soup is installed. Soup never trains or evaluates here, so nothing runs outside the
    8 GB engine; checks that would need that are NOT_TESTED with the reason."""
    soup = os.environ.get("SOUP_BIN") or shutil.which("soup")
    items = []
    if not soup or not Path(soup).exists():
        why = "Soup is not installed on this host (no `soup` on PATH, SOUP_BIN unset)"
        return [{"id": f"soup_{n}", "kind": "soup", "status": "NOT_TESTED", "detail": why}
                for n in ("version", "doctor", "data_doctor", "adapters_check_trained", "adapters_check_step0", "ship",
                          "preflight")]
    env = {"SOUP_NO_AUDIT_LOG": "1"}

    def call(item_id, args, ok_codes=(0,)):
        rec = run_item(Path(attempt_dir) / item_id, [soup, "--no-telemetry", *args], env=env, telemetry_on=False)
        rec.update(id=item_id, kind="soup",
                   status="PASS" if rec["exit"] in ok_codes else "FAIL",
                   detail=f"soup exit {rec['exit']} (PASS = the check ran and wrote its report; its findings are in "
                          "the log, interpreted in reports/silent_failures.md)")
        return rec
    items.append(call("soup_version", ["--version"]))
    items.append(call("soup_doctor", ["doctor"], ok_codes=(0, 1)))
    from qwen_grpo.sentiment import load_splits, prompt_text
    rows = load_splits()["train"][:200]
    jl = Path(attempt_dir) / "data_doctor_input.jsonl"
    jl.parent.mkdir(parents=True, exist_ok=True)
    jl.write_text("".join(json.dumps({"messages": [{"role": "user", "content": prompt_text(r["text"])},
                                                   {"role": "assistant", "content": r["label"]}]}) + "\n" for r in rows))
    items.append(call("soup_data_doctor", ["data", "doctor", rel(jl), "--model", "Qwen/Qwen3-4B", "--max-length", "512",
                                           "-o", rel(Path(attempt_dir) / "data_doctor_report.json")], ok_codes=(0, 1, 2)))
    from qwen_grpo import runs
    for tag, step in (("trained", None), ("step0", 0)):
        ckpt = (runs.latest_checkpoint(PRIMARY) if step is None else runs.checkpoint_path(PRIMARY, 0)) \
            if (RUNS_DIR / PRIMARY).exists() else None
        if ckpt and Path(ckpt).exists():
            items.append(call(f"soup_adapters_check_{tag}", ["adapters", "check", rel(ckpt), "--json"], ok_codes=(0, 1, 2)))
        else:
            items.append({"id": f"soup_adapters_check_{tag}", "kind": "soup", "status": "NOT_TESTED",
                          "detail": "no final-run checkpoint (stage F did not produce one)"})
    items.append({"id": "soup_ship", "kind": "soup", "status": "NOT_TESTED",
                  "detail": "not run: `soup ship` loads the base and tuned models with Soup's own inference, outside the "
                            "L2L engine (Qwen3-4B resident does not fit the 8 GB cap), and its --evidence mode requires "
                            "scores bound to a soup.yaml config_sha that this project does not have"})
    items.append({"id": "soup_preflight", "kind": "soup", "status": "NOT_TESTED",
                  "detail": "not run: Soup's pre-flight (`soup advise` / `soup plan`) plans Soup's own trainer from a "
                            "soup.yaml; training here uses this repository's engine, so there is no soup.yaml to check"})
    return items


RUNNERS = {"A": stage_A, "B": stage_B, "C": stage_C, "D": stage_entries, "E": stage_entries, "F": stage_entries,
           "G": stage_G,
           "H": stage_H}


def cmd_stage(args):
    run_dir = Path(args.run_dir)
    state = load_state(run_dir)
    stage = args.stage
    install_signals()
    st = state["stages"].setdefault(stage, {"name": STAGE_NAMES[stage], "attempts": []})
    last = st["attempts"][-1] if st["attempts"] else None
    forced = state.get("force_from") and at_or_after(stage, state["force_from"]) \
        and not state.get("forced_done", {}).get(stage)
    if last and last.get("final") and not forced:
        print(f"stage {stage} ({STAGE_NAMES[stage]}): {last['status']} in attempt {last['n']} - not re-run")
        return 0
    study, entries = load_study()
    n = len(st["attempts"]) + 1
    attempt_dir = run_dir / "raw" / RAW_DIRS[stage] / f"stage{stage}_attempt{n}"
    attempt = {"n": n, "started_utc": utc(), "raw_dir": rel(attempt_dir), "items": [], "final": False}
    st["attempts"].append(attempt)
    save_state(run_dir, state)
    print(f"== stage {stage} ({STAGE_NAMES[stage]}), attempt {n}")
    why = gate(stage, state) if stage != "A" else []
    if why:
        ids = stage_ids(stage)
        attempt["items"] = [blocked(i, "; ".join(why)) for i in ids] or [blocked(stage, "; ".join(why), "stage")]
        attempt["blocked_by"] = why
    else:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if stage == "A":
            attempt["items"], attempt["facts"] = stage_A(run_dir, state, attempt_dir, study, entries)
        else:
            runner = RUNNERS[stage]
            attempt["items"] = runner(stage, run_dir, state, attempt_dir, study, entries) if runner is stage_entries \
                else runner(run_dir, state, attempt_dir, study, entries)
    attempt["ended_utc"] = utc()
    attempt["status"] = worst(i["status"] for i in attempt["items"])
    attempt["interrupted"] = bool(STOP["signal"]) or any(i.get("interrupted") for i in attempt["items"])
    attempt["final"] = not attempt["interrupted"]
    if attempt["interrupted"]:
        attempt["status"] = "INCONCLUSIVE"
    if forced:
        state.setdefault("forced_done", {})[stage] = utc()
    st["status"] = attempt["status"]
    save_state(run_dir, state)
    for i in attempt["items"]:
        mem = f" [nvidia-smi {i['memory_status']}]" if i.get("memory_status") else ""
        print(f"   {i['status']:12s} {i['id']}{mem}: {str(i.get('detail'))[:200]}")
    print(f"== stage {stage}: {attempt['status']}" + (f" (interrupted by {STOP['signal']})" if attempt["interrupted"] else ""))
    return 130 if attempt["interrupted"] else 0


# --- init, status, dry-run ---------------------------------------------------------------------------------------

def install_experiment():
    """Point this module at the experiment protocol (module globals are read at call time)."""
    global MODE, STAGES, STAGE_ENTRIES, ESSENTIAL_B, FRESH, BUDGET_CONFIGS, FINAL_RUNS, PRIMARY, CONTROL, TRAIN_ID, \
        EVAL_ID
    MODE, STAGES, STAGE_ENTRIES = "experiment", EXPERIMENT_STAGES, EXPERIMENT_STAGE_ENTRIES
    ESSENTIAL_B, FRESH, BUDGET_CONFIGS = EXPERIMENT_ESSENTIAL_B, {"B"}, EXPERIMENT_BUDGET_CONFIGS
    FINAL_RUNS, TRAIN_ID, EVAL_ID = EXPERIMENT_FINAL_RUNS, EXPERIMENT_TRAIN_ID, EXPERIMENT_EVAL_ID
    PRIMARY, CONTROL = list(FINAL_RUNS)
    RUNNERS["B"] = stage_B_experiment


def latest_run(mode=None):
    """The latest run directory (of `mode` only, if given)."""
    runs = sorted(p for p in RESULTS.glob("*") if (p / "state.json").exists()) if RESULTS.exists() else []
    if mode is not None:
        runs = [p for p in runs if mode_of_state(load_state(p)) == mode]
    return runs[-1] if runs else None


def fresh_targets():
    """Outputs with fixed names that this mode's non-FRESH stages would reuse: training runs and Qwen3-4B results."""
    _, entries = load_study()
    out = []
    for s in STAGES:
        if s in FRESH:
            continue
        for eid in STAGE_ENTRIES.get(s, []):
            for o in entry_outputs(entries[eid]) if eid in entries else []:
                p = RUNS_DIR / o[4:] if o.startswith("run:") else repo_path(o)
                if p.exists() and p not in out:
                    out.append(p)
    return out


def archive_fresh(stamp_):
    """--fresh: move each existing fixed-name output aside to <path>.archived-<stamp> (nothing is deleted)."""
    moved = []
    for p in fresh_targets():
        dst = p.with_name(f"{p.name}.archived-{stamp_}")
        p.rename(dst)
        moved.append({"from": rel(p), "to": rel(dst)})
        print(f"--fresh: moved {rel(p)} -> {rel(dst)}", file=sys.stderr)
    return moved


def cmd_init(args):
    ident = code_identity()
    if args.resume or args.from_stage:
        if args.fresh:
            print("--fresh starts a new run; it cannot be combined with --resume or --from-stage", file=sys.stderr)
            return 2
        run_dir = Path(args.run_dir) if args.run_dir else latest_run(MODE)
        if run_dir is None:
            other = latest_run()
            print(f"no earlier assignment run in {MODE} mode to resume under {RESULTS}"
                  + (f" (the latest run, {other.name}, was started in {mode_of_state(load_state(other))} mode)"
                     if other else ""), file=sys.stderr)
            return 2
        if mode_of_state(load_state(run_dir)) != MODE:
            print(f"refusing: {run_dir} was started in {mode_of_state(load_state(run_dir))} mode, not {MODE}",
                  file=sys.stderr)
            return 2
        state = load_state(run_dir)
        diffs = [k for k in EXPERIMENT_IDENTITY if state["identity"].get(k) != ident.get(k) or ident.get(k) is None]
        diffs += [f"config {p}" for p in set(state["identity"]["config_sha256"]) | set(ident["config_sha256"])
                  if state["identity"]["config_sha256"].get(p) != ident["config_sha256"].get(p)]
        if diffs:
            print(f"refusing to resume {run_dir}: the code or configuration changed since it started ({', '.join(diffs)}). "
                  "Start a new run (without --resume); the old one stays as it is.", file=sys.stderr)
            return 2
        old_report = state["identity"].get("report_code_sha256")
        if old_report != ident.get("report_code_sha256"):
            state.setdefault("history", []).append({"utc": utc(), "event": f"report code changed ({str(old_report)[:12]} -> "
                                                    f"{str(ident.get('report_code_sha256'))[:12]}): the report is "
                                                    "regenerated, no experiment is re-run"})
            state["identity"]["report_code_sha256"] = ident.get("report_code_sha256")
        if args.from_stage:
            if args.from_stage not in STAGES:
                print(f"--from-stage must be one of {', '.join(STAGES)}", file=sys.stderr)
                return 2
            state["force_from"], state["forced_done"] = args.from_stage, {}
            state.setdefault("history", []).append({"utc": utc(), "event": f"from-stage {args.from_stage}"})
        else:
            state.setdefault("history", []).append({"utc": utc(), "event": "resume"})
        save_state(run_dir, state)
        print(run_dir)
        return 0
    run_id = stamp()
    run_dir = RESULTS / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    archived = archive_fresh(run_id) if args.fresh else []
    untracked = (git("status", "--porcelain", "--untracked-files=all") or "").splitlines()
    manifest = {"run_id": run_id, "created_utc": utc(),
                "command": "bash scripts/run_assignment.sh" + (" --experiment" if MODE == "experiment" else "")
                + (" --fresh" if args.fresh else ""), MODE_KEY: MODE, "stage_order": STAGES,
                "identity": ident, "untracked_files": [l[3:] for l in untracked if l.startswith("??")],
                "study": rel(STUDY), "gpu_memory_cap": load_study()[0]["gpu_memory_cap"],
                "python": sys.version.split()[0], "archived_by_fresh": archived}
    write_json(run_dir / "manifest.json", manifest)
    save_state(run_dir, {"run_id": run_id, "created_utc": manifest["created_utc"], MODE_KEY: MODE,
                         "identity": ident, "stages": {}, "produced": [], "archived_by_fresh": archived,
                         "history": [{"utc": utc(), "event": "new" + (" --fresh" if args.fresh else "")}]})
    print(run_dir)
    return 0


def cmd_status(args):
    run_dir = Path(args.run_dir) if args.run_dir else latest_run(os.environ.get("ASSIGNMENT_MODE") or None)
    if run_dir is None:
        print("no assignment run yet")
        return 1
    state = load_state(run_dir)
    print(f"run {run_dir} (commit {str(state['identity']['git_commit'])[:10]}, updated {state.get('updated')}, "
          f"mode {mode_of_state(state)})")
    for s in STAGES:
        st = state["stages"].get(s)
        if not st:
            print(f"  {s} {STAGE_NAMES[s]:15s} not started")
            continue
        a = st["attempts"][-1]
        counts = {}
        for i in a["items"]:
            counts[i["status"]] = counts.get(i["status"], 0) + 1
        print(f"  {s} {STAGE_NAMES[s]:15s} {a.get('status', 'RUNNING'):12s} attempt {a['n']} "
              f"{' '.join(f'{k}={v}' for k, v in sorted(counts.items()))}"
              + (f"  blocked by: {'; '.join(a['blocked_by'])}" if a.get("blocked_by") else ""))
    v = read_json(run_dir / "requirements.json")
    if v:
        print(f"verdict: {v['verdict']}")
        from qwen_grpo.evidence import report_code_sha256
        if (v.get("report_identity") or {}).get("report_code_sha256") != report_code_sha256(ROOT):
            print("the report is stale: scripts/assignment_report.py changed since it was written. Regenerate it "
                  f"(no experiment re-runs): python scripts/assignment.py report --run-dir {run_dir}")
    return 0


def cmd_dry_run(args):
    """What a run would do here. Downloads nothing, trains nothing, writes nothing."""
    study, entries = load_study()
    cap = probe()
    print(f"dry run at commit {(git('rev-parse', '--short', 'HEAD') or '?').strip()} - nothing is written or run")
    print(f"GPU {cap.get('gpu')}, bf16 {cap.get('bf16')}, per-process nvidia-smi {cap.get('per_process_smi')}, native "
          f"Linux {cap.get('linux_native')}, fixture packages {cap.get('fixture_packages')}, host RAM {cap.get('host_ram_gib')} GiB")
    c = study["gpu_memory_cap"]
    print(f"GPU memory cap: {c['bytes']:.0f} bytes = {c['bytes'] / 2**30:.3f} GiB ({c['wording']})")
    print(f"mode: {MODE}; stage order: {' '.join(STAGES)}")
    if MODE == "experiment":
        print_experiment_plan()
    print("A preflight: provenance, capabilities, CUDA streams/events under nvidia-smi, resolved configs, study plan, "
          f"budget predictions {', '.join(BUDGET_CONFIGS)} -> predictions/ (read-only)")
    if MODE == "experiment":
        print("B essential correctness: the async-L2L lab cases below; no pytest suites (no DFlash tests)")
    else:
        print(f"B correctness: pytest {' '.join(PYTEST_CPU + PYTEST_SILENT)} (CPU); {' '.join(PYTEST_CUDA)} (CUDA)")
    for s in [x for x in STAGES if x in "BCDEFG"]:
        if s == "F":
            for name in FINAL_RUNS:
                m = read_json(RUNS_DIR / name / "manifest.json")
                print(f"F resume check runs/{name}: " + ("absent" if m is None else f"status {m.get('status')}"))
        ids = STAGE_ENTRIES.get(s, [])
        if ids:
            print(f"{s} {STAGE_NAMES[s]}:" + (" (lab evidence shown as DONE predates this run: a real run re-runs it "
                                              "here under a new name with --again, and never reuses it)"
                                              if s in FRESH else ""), flush=True)
            subprocess.run([*dispatcher(), "run", "--dry-run", "--entry", *ids], cwd=ROOT)
    if "H" in STAGES:
        print("H audit: Soup " + ("found: " + (os.environ.get("SOUP_BIN") or shutil.which("soup"))
                                  if (os.environ.get("SOUP_BIN") or shutil.which("soup")) else "not installed: NOT_TESTED"))
    t = fresh_targets()
    print("--fresh would move aside: " + (", ".join(rel(p) for p in t) if t else "nothing (no fixed-name output exists)"))
    return 0


def print_experiment_plan():
    """The resolved facts an experiment run depends on, read from the configs (not from documentation)."""
    from qwen_grpo import config
    for name, path in FINAL_RUNS.items():
        c = config.load(ROOT / path)
        print(f"{name}: {c.model.name} {c.model.dtype}, {c.training.max_steps} updates, B={c.training.batch_size} "
              f"G={c.training.group_size}, {c.training.max_new_tokens} new tokens, lr {c.training.learning_rate}, clip "
              f"{c.training.clip_eps}, micro-batch {c.training.micro_batch_size}, {c.training.num_iterations} "
              f"iteration(s), eval every {c.evaluation.every} on {c.data.val_subset}, pool {c.data.train_samples}, "
              f"LoRA r{c.lora.rank}/a{c.lora.alpha} {c.lora.targets}, {c.execution.backend}, "
              f"disk streaming {c.streaming.enabled}, DFlash {c.dflash.enabled} ({c.dflash.draft_model}), memory guard "
              f"{c.memory.guard}")
    print(f"B: {', '.join(STAGE_ENTRIES['B'])}; waived for the F gate (recorded, stays FAIL if it fails): "
          f"{', '.join(WAIVED)}")
    print(f"F gate (experiment): B {', '.join(ESSENTIAL_B)} PASS; D H11.smoke and H11.profile PASS with per-process "
          f"nvidia-smi memory PASS; {FINAL_RUNS[PRIMARY]} has DFlash disabled (plain L2L rollouts)")
    print("not run in this mode: stage C (Qwen3-0.6B reference), stage E and every DFlash item (benchmark, DFlash "
          "smoke, drift, DFlash tests; DFlash requirements are NOT_TESTED, intentionally excluded), the CPU / "
          "silent-failure / CUDA pytest suites, the sync / CPU lab cases and their fault controls (--full-tests runs "
          "them)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--resume", action="store_true")
    i.add_argument("--from-stage")
    i.add_argument("--run-dir")
    i.add_argument("--fresh", action="store_true")
    s = sub.add_parser("stage")
    s.add_argument("stage", choices=list("ABCDEFGH"))
    s.add_argument("--run-dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--run-dir", required=True)
    t = sub.add_parser("status")
    t.add_argument("--run-dir")
    sub.add_parser("dry-run")
    o = sub.add_parser("order", help="the stage order of a run directory (for scripts/run_assignment.sh)")
    o.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    if MODE_KEY == "mode":  # this module's own entry point (scripts/assignment_local.py installs its own modes)
        if a.cmd in ("stage", "report", "order") or (a.cmd == "status" and a.run_dir):
            mode = mode_of_state(load_state(a.run_dir))
        else:
            mode = os.environ.get("ASSIGNMENT_MODE") or "full"
        if mode not in MODES:
            raise SystemExit(f"ASSIGNMENT_MODE must be one of {MODES}, not {mode!r}")
        if mode == "experiment":
            install_experiment()
    if a.cmd == "order":
        print(" ".join(STAGES))
        return 0
    if a.cmd == "stage" and a.stage not in STAGES:
        print(f"stage {a.stage} is not part of the {MODE} protocol ({STAGES})", file=sys.stderr)
        return 2
    if a.cmd == "report":
        import importlib.util
        spec = importlib.util.spec_from_file_location("assignment_report", ROOT / "scripts" / "assignment_report.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(Path(a.run_dir))
    return {"init": cmd_init, "stage": cmd_stage, "status": cmd_status, "dry-run": cmd_dry_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
