"""Hypothesis experiments H01-H12 from one study manifest (configs/studies/assignment.yaml).

    python scripts/experiment.py plan   [--target local|kaggle-cpu|kaggle-t4|ampere|all]
    python scripts/experiment.py run    --hypothesis H05 [--entry H05.real_0.6b_bf16] [--again] [--dry-run]
    python scripts/experiment.py report
    python scripts/experiment.py index  [--compress-over-mib N]

A small dispatcher, not a workflow engine. Every manifest entry names existing commands (scripts/l2l_lab.py,
scripts/run.py, scripts/compare_runs.py, scripts/resume_experiment.py, scripts/profile_run.py, scripts/dflash_bench.py,
scripts/final_eval.py); this file only checks that an entry can run here, runs its commands as subprocesses, keeps
their raw output and reads their verdicts. It never re-implements a test or a criterion, never overrides a tolerance,
and never edits research/l2l/hypotheses.yaml: a hypothesis' recorded status changes only by hand, citing evidence.

plan     which entries can run on this machine (probed) or on a named target, and why the others cannot
run      for each selected entry: hardware and prerequisite checks (unmet -> BLOCKED, recorded, never PASS); entries
         whose evidence exists are DONE and not re-run unless --again (then under a new name: nothing is overwritten);
         otherwise the steps run and evidence/<UTC>_<entry>/ gets manifest.json (commit, dirty tree, hardware,
         packages, configs and their sha256, seeds, precision, commands with UTC times and exit codes),
         resolved_config.yaml, raw/ (every step's output, nvidia-smi samples on GPU entries, the run logs and memory
         samples of training runs), results.json and summary.md
report   per hypothesis: the recorded status and evidence (authoritative), what every entry's outputs say now, and the
         dispatcher executions; regenerates research/l2l/final_report.md (scripts/final_report.py);
         writes evidence/study_report.md
index    research/l2l/evidence_index.json: every recorded experiment output with its sha256, statuses, provenance and
         the hypotheses citing it; --compress-over-mib stores large lab results as lossless result.json.gz

Outcomes: PASS / FAIL / BLOCKED / NOT TESTED (entry), and the hypothesis criterion where an existing tool evaluates it:
MET / NOT_MET / INCONCLUSIVE / NOT_EVALUATED. A completed experiment is not a supported hypothesis.
"""
import argparse
import gzip
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo.evidence import EXPERIMENTS, read_bytes, read_result, result_path, sha256_bytes  # noqa: E402

STUDY = ROOT / "configs" / "studies" / "assignment.yaml"
EVIDENCE = ROOT / "evidence"
INDEX = ROOT / "research" / "l2l" / "evidence_index.json"
CAPABILITIES = ("cuda", "bf16", "fixture_packages", "linux_native", "per_process_smi")
JUDGES = ("lab", "lab_fault", "exit_codes", "pass_key", "requirements")
CRITERION = {"SUPPORTED": "MET", "NOT_SUPPORTED": "NOT_MET", "INCONCLUSIVE": "INCONCLUSIVE"}
# Static capability profiles for `plan --target`; `local` is probed. Kaggle CPU kernels install torch from the CPU
# index (2.14.1+cpu), which the saved lab fixtures refuse; the T4 has no native bfloat16.
TARGETS = {
    "kaggle-cpu": {"cuda": False, "bf16": False, "fixture_packages": False, "linux_native": True, "vram_gib": 0.0,
                   "per_process_smi": False, "gpu": None},
    "kaggle-t4": {"cuda": True, "bf16": False, "fixture_packages": True, "linux_native": True, "vram_gib": 15.0,
                  "per_process_smi": False, "gpu": "Tesla T4 (7.5)"},
    "ampere": {"cuda": True, "bf16": True, "fixture_packages": True, "linux_native": True, "vram_gib": 24.0,
               "per_process_smi": True, "gpu": "Ampere or newer, >= 24 GB, ASSUMING per-process nvidia-smi (containers "
                                               "often hide it: run `plan --target local` on the host before paying)"},
}

PROBE = r'''
import importlib.metadata, json, os, platform, sys, warnings
import torch
cap = {"cuda": False, "bf16": False, "vram_gib": 0.0, "gpu": None, "capability": None, "cuda_error": None,
       "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}}
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    cap.update(gpu=p.name, capability=f"{p.major}.{p.minor}", vram_gib=round(p.total_memory / 2**30, 2))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (torch.ones(8, device="cuda") * 2).sum().item()  # a build without this GPU's kernels fails here
        cap.update(cuda=True, bf16=torch.cuda.is_bf16_supported(including_emulation=False))
    except Exception as e:
        cap["cuda_error"] = f"{type(e).__name__}: {str(e).splitlines()[0]}"[:160]
    # assignment-grade memory evidence: nvidia-smi must list THIS process (it holds a CUDA context now) with a number;
    # WSL2 shows [N/A] and containers often hide host PIDs (the memory guard then falls back to device-wide figures)
    import shutil, subprocess
    if cap["cuda"] and shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        mine = [l.split(",") for l in r.stdout.splitlines() if l.split(",")[0].strip() == str(os.getpid())]
        cap["per_process_smi"] = bool(mine) and mine[0][1].strip().isdigit()
        cap["nvidia_smi_apps"] = r.stdout.strip()[:500]
try:
    meta = torch.load("research/l2l/fixtures/0.6b-like.pt", weights_only=True)["meta"]["packages"]
    cap["fixture_packages_saved"] = meta
    cap["fixture_packages"] = meta == cap["packages"]
except Exception as e:
    cap["fixture_packages"], cap["fixture_packages_saved"] = False, f"{type(e).__name__}: {e}"
print(json.dumps(cap))
'''


# --- manifest ---------------------------------------------------------------------------------------------------

def load_study(path=STUDY):
    """The study manifest, checked: unique ids, known capabilities and judges, hypotheses that exist, and no key that
    could redefine a criterion or a threshold."""
    study = yaml.safe_load(Path(path).read_text())
    hyps = yaml.safe_load((ROOT / study["hypotheses"]).read_text())
    seen, errors = set(), []
    for e in study["entries"]:
        hid, eid = e["id"].split(".")[0], e["id"]
        if eid in seen:
            errors.append(f"duplicate id {eid}")
        seen.add(eid)
        if hid not in hyps:
            errors.append(f"{eid}: {hid} is not in {study['hypotheses']}")
        if bad := set(e.get("requires", [])) - set(CAPABILITIES):
            errors.append(f"{eid}: unknown capabilities {sorted(bad)}")
        if judge_kind(e) not in JUDGES:
            errors.append(f"{eid}: unknown judge {e.get('judge')!r}")
        if forbidden := {"criterion", "statement", "status", "tolerances", "threshold", "atol", "rtol"} & set(e):
            errors.append(f"{eid}: {sorted(forbidden)} belong to hypotheses.yaml / tolerances.yaml, not the study")
        if not e.get("steps"):
            errors.append(f"{eid}: no steps")
    if errors:
        raise SystemExit("invalid study manifest:\n  " + "\n  ".join(errors))
    return study, hyps


def judge_kind(entry):
    j = entry.get("judge")
    return j if isinstance(j, str) else next(iter(j), None) if isinstance(j, dict) else None


def steps(entry):
    return [s if isinstance(s, dict) else {"argv": s} for s in entry["steps"]]


def fill(value, name, evidence=None):
    return value.replace("{name}", name).replace("{evidence}", str(evidence) if evidence else "{evidence}")


# --- capabilities and prerequisites -------------------------------------------------------------------------------

def probe():
    """This machine's capabilities, probed in a child process so this process never holds a CUDA context (it would
    count in device-wide memory readings of the experiments it starts)."""
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=600)
    cap = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"probe_error": r.stderr[-2000:]}
    rel = platform.release().lower()
    cap["linux_native"] = sys.platform == "linux" and "microsoft" not in rel and "wsl" not in rel
    cap["platform"] = platform.platform()
    try:
        cap["host_ram_gib"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        cap["host_ram_gib"] = None
    return cap


def unmet(entry, cap):
    """Why `entry` cannot run with capabilities `cap` ([] = it can)."""
    why = []
    for c in entry.get("requires", []):
        if not cap.get(c):
            why.append({"cuda": f"no usable CUDA GPU{': ' + cap['cuda_error'] if cap.get('cuda_error') else ''}",
                        "bf16": f"no native bfloat16 ({cap.get('gpu') or 'no GPU'}); the entry's precision is "
                                f"{entry.get('precision')}",
                        "fixture_packages": "package versions differ from the saved lab fixtures "
                                            f"({cap.get('fixture_packages_saved', 'torch 2.14.1+cu126')})",
                        "linux_native": "not native Linux (WSL2 has no per-process nvidia-smi memory)",
                        "per_process_smi": "nvidia-smi does not report this process's GPU memory (WSL2, or a container "
                                           "hiding host PIDs): memory evidence would be diagnostic only"}[c])
    if (need := entry.get("min_vram_gib")) and cap.get("cuda") and cap.get("vram_gib", 0) < need:
        why.append(f"GPU has {cap.get('vram_gib')} GiB, the entry needs >= {need} GiB")
    return why


def run_state(name):
    """(completed and valid, reason) for a training run under runs/."""
    m, t = ROOT / "runs" / name / "manifest.json", ROOT / "runs" / name / "training" / "training.json"
    if not m.exists():
        return False, f"run {name} does not exist"
    status = json.loads(m.read_text()).get("status")
    if status not in ("completed", "complete"):
        return False, f"run {name} is {status} (resume it: python scripts/run.py train --config runs/{name}/config.yaml --resume latest)"
    log = json.loads(t.read_text()) if t.exists() else {}
    if log.get("frozen_changed") is None or log["frozen_changed"]:  # the changed frozen tensors; [] = none
        return False, f"run {name}: frozen weights not confirmed unchanged ({log.get('frozen_changed')!r})"
    if not log.get("final_checkpoint"):
        return False, f"run {name}: no final checkpoint"
    return True, f"run {name} completed"


def unmet_prerequisites(entry):
    pre, why = entry.get("prerequisites") or {}, []
    for name in pre.get("runs", []):
        ok, reason = run_state(name)
        if not ok:
            why.append(reason)
    for name in pre.get("lab", []):
        r = read_result(EXPERIMENTS / name)
        if r is None:
            why.append(f"lab result {name} missing")
        elif r.get("fault_injection"):
            if not (r.get("status") == "FAIL" and r.get("execution_status") == "PASS"):
                why.append(f"fault run {name}: fault not detected ({r.get('status')})")
        elif r.get("status") != "PASS":
            why.append(f"lab result {name} is {r.get('status')}")
    return why


def outputs(entry, name):
    """Paths whose existence means the entry has run: its lab directory, and every step's `creates`."""
    out = [EXPERIMENTS / name] if judge_kind(entry) in ("lab", "lab_fault") else []
    return out + [fill(s["creates"], name) for s in steps(entry) if s.get("creates")]


def exists(output):
    if isinstance(output, str) and output.startswith("run:"):
        return run_state(output[4:])[0]
    return (ROOT / output).exists()


def done(entry):
    outs = outputs(entry, entry["name"])
    return bool(outs) and all(exists(o) for o in outs)


# --- judging ------------------------------------------------------------------------------------------------------

def final_report():
    spec = importlib.util.spec_from_file_location("final_report", ROOT / "scripts" / "final_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def judge(entry, name, exits=None):
    """(outcome, criterion, detail) from what the entry's tools recorded."""
    kind, spec = judge_kind(entry), entry["judge"]
    if kind in ("lab", "lab_fault"):
        r = read_result(EXPERIMENTS / name)
        if r is None:
            return "NOT TESTED", "NOT_EVALUATED", f"no result research/l2l/experiments/{name}"
        status, execution = r.get("status"), r.get("execution_status", r.get("status"))
        if kind == "lab_fault":
            ok = status == "FAIL" and execution == "PASS"
            return ("PASS" if ok else "FAIL"), "NOT_EVALUATED", \
                f"fault injection: {'detected' if ok else 'NOT detected'} (status {status}, execution {execution})"
        outcome = {"PASS": "PASS", "BLOCKED": "BLOCKED"}.get(status, "FAIL")
        crit = CRITERION.get(r.get("hypothesis_result"), "NOT_EVALUATED")
        return outcome, crit, f"status {status}, execution {execution}, hypothesis_result {r.get('hypothesis_result')}" \
            + (f": {r.get('hypothesis_reason')}" if r.get("hypothesis_reason") else "")
    if kind == "pass_key":
        p = ROOT / fill(spec["pass_key"], name)
        if not p.exists():
            return "NOT TESTED", "NOT_EVALUATED", f"no result {rel(p)}"
        ok = json.loads(p.read_text()).get("pass") is True
        return ("PASS" if ok else "FAIL"), ("MET" if ok else "NOT_MET"), f"{rel(p)}: pass {ok}"
    if kind == "exit_codes":
        if exits is None:
            missing = [o for o in outputs(entry, name) if not exists(o)]
            return ("NOT TESTED", "NOT_EVALUATED", f"not run here: {missing}") if missing else \
                ("NOT TESTED", "NOT_EVALUATED", "runs exist; comparisons are re-run by `run`")
        bad = [f"step {i + 1} exit {rc}" for i, rc in enumerate(exits) if rc not in (0, "skipped")]
        return ("FAIL" if bad else "PASS"), "NOT_EVALUATED", "; ".join(bad) or \
            "every step exited 0" + (" (supplementary evidence; the criterion is judged on its lab case)"
                                     if entry.get("supplementary") else "")
    reqs = {r["id"]: r for r in final_report().requirements()[0]}
    got = {i: reqs[i] for i in spec["requirements"]}
    statuses = [r["status"] for r in got.values()]
    detail = "; ".join(f"{i} {r['status']}: {r['evidence']}" for i, r in got.items())
    outcome = "PASS" if all(s == "PASS" for s in statuses) else "FAIL" if "FAIL" in statuses else \
        "BLOCKED" if "BLOCKED" in statuses else "NOT TESTED"
    crit = {"PASS": "MET", "FAIL": "NOT_MET", "BLOCKED": spec.get("blocked_means", "NOT_EVALUATED")}.get(outcome,
                                                                                                          "NOT_EVALUATED")
    if crit == "MET" and spec.get("unchecked_clauses"):
        crit, detail = "NOT_EVALUATED", detail + "; criterion clauses no tool derives yet (check by hand): " + \
            "; ".join(spec["unchecked_clauses"])
    return outcome, crit, detail


# --- running ------------------------------------------------------------------------------------------------------

def rel(path):
    """`path` relative to the repository root when it lies inside it (as recorded in evidence), else absolute."""
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def config_paths(entry):
    paths = list(entry.get("configs", []))
    for s in steps(entry):
        argv = s["argv"]
        paths += [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--config"]
    return list(dict.fromkeys(paths))


def resolved_configs(entry):
    from qwen_grpo import config
    out = {}
    for p in config_paths(entry):
        cfg = config.load(ROOT / p)
        out[p] = {"sha256": config.sha256(cfg), "resolved": config.as_dict(cfg),
                  "seeds": {"training.seed": cfg.training.seed, "data.subset_seed": cfg.data.subset_seed},
                  "precision": cfg.model.dtype, "backend": cfg.execution.backend}
    return out


class SmiSampler:
    """nvidia-smi every second for the whole entry, raw CSV (device and per-process figures)."""

    def __init__(self, raw, enabled):
        self.raw, self.procs, self.enabled = raw, [], enabled and shutil.which("nvidia-smi")

    def __enter__(self):
        if self.enabled:
            for name, query in (("nvidia_smi_gpu.csv", ["--query-gpu=timestamp,index,name,memory.used,memory.total,"
                                                         "utilization.gpu"]),
                                ("nvidia_smi_apps.csv", ["--query-compute-apps=timestamp,pid,process_name,used_memory"])):
                f = open(self.raw / name, "w")
                self.procs.append((subprocess.Popen(["nvidia-smi", *query, "--format=csv", "-l", "1"], stdout=f,
                                                    stderr=subprocess.STDOUT), f))
        return self

    def __exit__(self, *exc):
        for p, f in self.procs:
            p.terminate()
            p.wait(timeout=30)
            f.close()
        return False


def copy_run(name, dest):
    """The small files of a training run (logs, memory samples, manifest, config), never its checkpoints."""
    src = ROOT / "runs" / name
    for rel in ("manifest.json", "config.yaml", "summary.json", "report.md", "training/training.json",
                "training/training.log", "training/validation.json"):
        if (src / rel).exists():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dest / rel)
    if (src / "training" / "memory").is_dir():
        shutil.copytree(src / "training" / "memory", dest / "training" / "memory", dirs_exist_ok=True)


def record(entry, name, cap, status, detail, commands=(), exits=None, started=None, evidence=None):
    """Write evidence/<id>/ for one execution (BLOCKED ones included) and return its results."""
    evidence = evidence or new_evidence_dir(entry)
    outcome, crit, why = (status, "NOT_EVALUATED", detail) if status in ("BLOCKED", "DRY_RUN") else \
        judge(entry, name, exits)
    outs = []
    for o in outputs(entry, name):
        if isinstance(o, str) and o.startswith("run:"):
            outs.append({"output": o, "exists": exists(o), "state": run_state(o[4:])[1]})
            continue
        p = ROOT / o
        f = result_path(p) if p.is_dir() else (p if p.exists() else None)
        outs.append({"output": rel(o) if isinstance(o, Path) else o, "exists": p.exists(),
                     "file": rel(f) if f else None,
                     "sha256_uncompressed": sha256_bytes(read_bytes(f)) if f else None})
    res = {"experiment_id": evidence.name, "entry": entry["id"], "hypothesis": entry["id"].split(".")[0],
           "evidence_name": name, "status": outcome, "criterion": crit, "detail": why,
           "supplementary": bool(entry.get("supplementary")), "diagnostic": bool(entry.get("diagnostic")),
           "outputs": outs}
    try:
        configs = resolved_configs(entry)
    except Exception as e:  # noqa: BLE001 - recorded
        configs = {"error": f"{type(e).__name__}: {e}"}
    man = {"experiment_id": evidence.name, "entry": entry, "evidence_name": name,
           "study": {"path": rel(STUDY), "sha256": sha256_bytes(STUDY.read_bytes())},
           "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                   "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                   "dirty_files": (git("status", "--porcelain", "--untracked-files=no") or "").splitlines()},
           "utc_start": started or utc(), "utc_end": utc(), "python": sys.version.split()[0], "hardware": cap,
           "model": entry.get("model"), "precision": entry.get("precision"), "diagnostic": bool(entry.get("diagnostic")),
           "configs": {p: {k: v for k, v in c.items() if k != "resolved"} for p, c in configs.items()}
           if "error" not in configs else configs,
           "commands": list(commands)}
    (evidence / "manifest.json").write_text(json.dumps(man, indent=1, default=str) + "\n")
    if "error" not in configs:
        (evidence / "resolved_config.yaml").write_text("".join(
            f"--- # {p} sha256 {c['sha256']}\n" + yaml.safe_dump(c["resolved"], sort_keys=False)
            for p, c in configs.items()) or "# this entry names no YAML config\n")
    (evidence / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    lines = [f"# {entry['id']} ({name})", "", f"- status: **{outcome}**", f"- hypothesis criterion: **{crit}**",
             f"- detail: {why}", f"- commit {man['git']['commit']} (tracked tree dirty: {man['git']['tracked_dirty']})",
             f"- hardware: {cap.get('gpu') or 'CPU only'}, bf16 {cap.get('bf16')}, {cap.get('platform')}",
             f"- precision {entry.get('precision')}" + (" (DIAGNOSTIC: not evidence for the hypothesis' own precision)"
                                                       if entry.get("diagnostic") else ""),
             "", "| step | exit | seconds | log |", "|---|---|---|---|"]
    lines += [f"| `{' '.join(c['argv'])}` | {c['exit']} | {c.get('seconds', '')} | {c.get('log', '')} |" for c in commands]
    lines += ["", "Recorded statuses in research/l2l/hypotheses.yaml are not changed by this run."]
    (evidence / "summary.md").write_text("\n".join(lines) + "\n")
    return res


def new_evidence_dir(entry):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = EVIDENCE / f"{stamp}_{entry['id']}"
    (d / "raw").mkdir(parents=True, exist_ok=False)  # never reused, never overwritten
    return d


def execute(entry, cap, again=False, dry_run=False):
    name = entry["name"]
    if done(entry) and not again:
        outcome, crit, why = judge(entry, name)
        print(f"{entry['id']}: DONE (evidence exists, never overwritten; --again re-runs it under a new name): "
              f"{outcome}, criterion {crit} - {why}")
        return {"entry": entry["id"], "status": "DONE", "outcome": outcome, "criterion": crit}
    if again:
        if any(s.get("creates", "").startswith("run:") for s in steps(entry)):
            raise SystemExit(f"{entry['id']}: --again cannot rename training runs (they are named by their configs); "
                             "remove or rename runs/<name> first (python scripts/run.py remove --run <name>)")
        name = f"{name}_r{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    blocked = unmet(entry, cap) + unmet_prerequisites(entry)
    if dry_run:
        print(f"{entry['id']} ({name}): " + ("BLOCKED: " + "; ".join(blocked) if blocked else "would run:"))
        for s in steps(entry):
            print("   $ python " + " ".join(fill(a, name) for a in s["argv"]))
        return {"entry": entry["id"], "status": "DRY_RUN"}
    if blocked:
        res = record(entry, name, cap, "BLOCKED", "; ".join(blocked))
        print(f"{entry['id']}: BLOCKED - {res['detail']}  (evidence/{res['experiment_id']})")
        return res
    evidence, started, commands, exits = new_evidence_dir(entry), utc(), [], []
    env = {**os.environ, "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false"}
    if "cuda" not in entry.get("requires", []):
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU entries never see a GPU
    print(f"{entry['id']}: running ({name}) -> evidence/{evidence.name}")
    with SmiSampler(evidence / "raw", "cuda" in entry.get("requires", [])):
        for i, s in enumerate(steps(entry), 1):
            argv = [fill(a, name, evidence) for a in s["argv"]]
            if s.get("creates") and exists(fill(s["creates"], name)):
                commands.append({"argv": argv, "exit": "skipped", "reason": f"{fill(s['creates'], name)} exists"})
                exits.append("skipped")
                continue
            if exits and exits[-1] not in (0, "skipped"):
                commands.append({"argv": argv, "exit": "not run", "reason": "an earlier step failed"})
                exits.append("not run")
                continue
            log = evidence / "raw" / f"step{i:02d}.log"
            t0, start = time.time(), utc()
            print(f"   $ python {' '.join(argv)}")
            with open(log, "w") as f:
                rc = subprocess.run([sys.executable, *argv], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            commands.append({"argv": argv, "utc_start": start, "utc_end": utc(), "seconds": round(time.time() - t0, 1),
                             "exit": rc, "log": f"raw/{log.name}"})
            exits.append(rc)
            print(f"     exit {rc} after {time.time() - t0:.0f} s")
    for s in steps(entry):
        if s.get("creates", "").startswith("run:") and (ROOT / "runs" / s["creates"][4:]).exists():
            copy_run(s["creates"][4:], evidence / "raw" / "runs" / s["creates"][4:])
    res = record(entry, name, cap, None, None, commands, exits, started, evidence)
    print(f"{entry['id']}: {res['status']}, criterion {res['criterion']} - {res['detail']}  (evidence/{evidence.name})")
    return res


# --- commands -----------------------------------------------------------------------------------------------------

def cmd_plan(args):
    study, _ = load_study(args.study)
    targets = {"local": probe()} if args.target in ("local", "all") else {}
    targets.update({t: TARGETS[t] for t in TARGETS if args.target in (t, "all")})
    for t, cap in targets.items():
        print(f"\n== {t}: GPU {cap.get('gpu') or 'none'}" + (f" (unusable: {cap['cuda_error']})" if cap.get("cuda_error") else "")
              + f", bf16 {cap.get('bf16')}, fixture packages {cap.get('fixture_packages')}, native Linux "
              f"{cap.get('linux_native')}, per-process nvidia-smi {cap.get('per_process_smi')}")
        for e in study["entries"]:
            hw = unmet(e, cap)
            pre = unmet_prerequisites(e)
            state = "DONE" if done(e) else "RUN" if not hw and not pre else "BLOCKED"
            tags = "".join(f" [{k}]" for k in ("diagnostic", "supplementary") if e.get(k))
            why = "; ".join(hw + pre)
            print(f"  {state:7s} {e['id']:34s} {e.get('model', ''):10s} {','.join(e.get('precision', [])):17s}{tags}"
                  + (f"  <- {why}" if why and state != "DONE" else ""))
    return 0


def cmd_run(args):
    study, _ = load_study(args.study)
    entries = [e for e in study["entries"] if (args.entry and e["id"] in args.entry)
               or (not args.entry and args.hypothesis and e["id"].split(".")[0] in args.hypothesis)]
    if not entries:
        raise SystemExit("no matching entry: give --hypothesis H.. and/or --entry <id> (see `plan`)")
    cap = probe()
    results = [execute(e, cap, again=args.again, dry_run=args.dry_run) for e in entries]
    bad = [r for r in results if r.get("status") in ("FAIL", "BLOCKED", "NOT TESTED")]
    return 2 if bad else 0


def cmd_report(args):
    study, hyps = load_study(args.study)
    fr = final_report()
    fr.main()  # research/l2l/final_report.md and final_results.json, derived from the evidence files
    executions = {}
    for p in sorted(EVIDENCE.glob("*/results.json")):
        r = json.loads(p.read_text())
        executions.setdefault(r["entry"], []).append(r)
    md = ["# Study report: H01-H12", "",
          f"Generated {utc()} at commit {git('rev-parse', '--short', 'HEAD')} by `scripts/experiment.py report`. The "
          "recorded status is the one in `research/l2l/hypotheses.yaml` (changed only by hand, citing evidence); the "
          "entry columns are what the recorded outputs say now. A completed experiment is not a supported hypothesis.",
          ""]
    for hid, h in hyps.items():
        md += [f"## {hid}: {h['statement']}", "",
               f"Recorded status: **{h['status']}**; cited evidence: {', '.join(h.get('evidence') or []) or 'none'}", "",
               "| entry | evidence name | outputs say | criterion | last execution |", "|---|---|---|---|---|"]
        for e in (e for e in study["entries"] if e["id"].split(".")[0] == hid):
            outcome, crit, why = judge(e, e["name"])
            last = executions.get(e["id"], [])
            tags = "".join(f" *{k}*" for k in ("diagnostic", "supplementary") if e.get(k)) + \
                (" *fault control*" if judge_kind(e) == "lab_fault" else "")
            uncited = outcome == "PASS" and judge_kind(e) == "lab" and e["name"] not in (h.get("evidence") or [])
            md.append(f"| {e['id']}{tags} | {e['name']}{' (not cited)' if uncited else ''} | {outcome}: {why} | {crit} | "
                      + (f"{last[-1]['experiment_id']}: {last[-1]['status']}" if last else "none") + " |")
        md.append("")
    out = EVIDENCE / "study_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print(f"written {rel(out)} and research/l2l/final_report.md")
    return 0


def cmd_index(args):
    """research/l2l/evidence_index.json; optionally compress large lab results losslessly (verified before the
    original is removed: the decompressed bytes must hash to the original's sha256)."""
    hyps = yaml.safe_load((ROOT / "research" / "l2l" / "hypotheses.yaml").read_text())
    cited = {}
    for hid, h in hyps.items():
        for n in h.get("evidence") or []:
            cited.setdefault(n, []).append(hid)
    old = {e["name"]: e for e in json.loads(INDEX.read_text())["experiments"]} if INDEX.exists() else {}
    entries = []
    for p in sorted(EXPERIMENTS.iterdir()):
        name = p.name.removesuffix(".json")
        f = result_path(p) if p.is_dir() else p
        if f is None:
            continue
        data = read_bytes(f)
        sha = sha256_bytes(data)
        if old.get(name, {}).get("sha256") not in (None, sha):
            raise SystemExit(f"{f}: content differs from the indexed sha256 {old[name]['sha256']}: evidence changed")
        if args.compress_over_mib and f.name == "result.json" and len(data) > args.compress_over_mib * 2**20:
            gz = f.with_name("result.json.gz")
            gz.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
            if sha256_bytes(gzip.decompress(gz.read_bytes())) != sha:
                gz.unlink()
                raise SystemExit(f"{gz}: round trip failed; original kept")
            f.unlink()
            f = gz
        r = json.loads(data) if f.suffix in (".json", ".gz") else None
        e = {"name": name, "file": rel(f), "sha256": sha, "bytes": len(data),
             "stored_bytes": f.stat().st_size, "cited_by": cited.get(name, [])}
        if isinstance(r, dict):
            hyp = r.get("hypothesis")
            prov = r.get("provenance") or {}
            e.update({k: r.get(k) for k in ("case", "status", "execution_status", "hypothesis_result", "fault_injection",
                                             "identical", "pass", "verdict") if k in r},
                     hypothesis=hyp.get("id") if isinstance(hyp, dict) else hyp,
                     commit=prov.get("git_commit"), tracked_dirty=prov.get("git_tracked_dirty", prov.get("git_dirty")),
                     gpu=prov.get("gpu"), platform=prov.get("platform"))
        entries.append(e)
    missing = sorted(set(cited) - {e["name"] for e in entries})
    INDEX.write_text(json.dumps({
        "about": "Every recorded output under research/l2l/experiments/: sha256 of its uncompressed bytes, statuses as "
                 "recorded, provenance, and the hypotheses whose `evidence` cites it. Regenerate with "
                 "`python scripts/experiment.py index`; it refuses if a recorded file's content changed. Raw copies: "
                 "git history and the external archive named in docs/cleanup_report.md.",
        "cited_but_missing": missing, "experiments": entries}, indent=1) + "\n")
    print(f"indexed {len(entries)} outputs -> {rel(INDEX)}" + (f"; cited but missing: {missing}" if missing else ""))
    return 1 if missing else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", type=Path, default=STUDY)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--target", default="all", choices=["local", *TARGETS, "all"])
    r = sub.add_parser("run")
    r.add_argument("--hypothesis", nargs="*", default=[])
    r.add_argument("--entry", nargs="*", default=[])
    r.add_argument("--again", action="store_true", help="re-run entries whose evidence exists, under a new name")
    r.add_argument("--dry-run", action="store_true")
    sub.add_parser("report")
    i = sub.add_parser("index")
    i.add_argument("--compress-over-mib", type=float, default=0.0)
    args = ap.parse_args(argv)
    return {"plan": cmd_plan, "run": cmd_run, "report": cmd_report, "index": cmd_index}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
