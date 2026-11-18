"""Stage runner behind scripts/run_assignment_local.sh: the LOCAL laptop validation on Qwen3-0.6B (4 GB GPU).

    python scripts/assignment_local.py init  [--resume]         # prints the run directory
    python scripts/assignment_local.py stage S --run-dir D      # S in A B C D F
    python scripts/assignment_local.py report --run-dir D       # scripts/assignment_local_report.py
    python scripts/assignment_local.py status [--run-dir D]
    python scripts/assignment_local.py dry-run

The machinery is scripts/assignment.py, unchanged: provenance and code identity, run state, resume rules, raw
nvidia-smi telemetry per item in its own process group, the dispatcher (scripts/experiment.py), stage gates, BLOCKED /
NOT_TESTED records and checksums. This file only swaps in the local protocol:
  study      configs/studies/assignment_local.yaml (tiny-model correctness entries included verbatim from the 4B study)
  results    assignment_results_local/<UTC run-id>/
  stages     A preflight      provenance, capabilities, CUDA streams/events under nvidia-smi, resolved local configs,
                              the local budget prediction (read-only)
             B correctness    CPU / silent-failure / CUDA unit tests (DFlash tests excluded: not in scope) and the
                              tiny-model equivalence and fault-injection lab cases, fresh on this host
             C equivalence    Qwen3-0.6B resident vs async L2L: one real GRPO step (forward log-probs, gradients,
                              rollouts, updates) and the 10-step training runs compared bitwise   gate: GPU, B essential
             D profiling      every phase's memory and time, transfer bytes, against the local budget      gate: GPU
             F training       2-step smoke, 10-step async L2L + disk-streaming run (compared bitwise with the in-memory
                              run), deterministic resume under six injected crashes
                              gate: B essential, C both PASS, D not FAIL (memory guard), smoke before the rest
  No DFlash stage (the only supported drafter is for Qwen3-4B) and no held-out evaluation or Soup stage: the local
  report marks them NOT_TESTED and never claims the Qwen3-4B assignment.

Two modes, recorded per run in state.json (local_mode; runs made before it existed are "full"):
  fast (default)  A preflight (+ model/dataset availability, earlier correctness evidence recorded only if its code
                  identity equals this run's, else STALE) and F = 2-step smoke, then the 10-step run
                  H12.local_train_fast_0.6b (same run, no comparison with stage C). Gate: stage A facts only.
  full            --full-tests: stages A B C D F as above.
Every item's stdout/stderr (and the dispatcher's step logs) is streamed to the terminal while it runs, with a
heartbeat every ASSIGNMENT_HEARTBEAT_S seconds (default 30): elapsed time, PIDs and the trainer's last step.
The active item is kept in <run>/active.json for --status.
"""
import fcntl
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import assignment as A  # noqa: E402

LOCAL_STUDY = ROOT / "configs" / "studies" / "assignment_local.yaml"
LOCAL_STAGES = "ABCDF"
LOCAL_STAGE_NAMES = {"A": "preflight", "B": "correctness", "C": "equivalence_0.6b", "D": "profiling_0.6b",
                     "F": "training_0.6b"}
LOCAL_RAW_DIRS = {"A": "preflight", "B": "correctness", "C": "equivalence", "D": "profiling", "F": "training"}
LOCAL_STAGE_ENTRIES = {
    "B": list(A.STAGE_ENTRIES["B"]),
    "C": ["H05.local_real_step_0.6b", "H05.local_runs_0.6b", "H06.prefetch_timing"],
    "D": ["H11.local_profile_0.6b"],
    "F": ["H12.local_smoke_0.6b", "H12.local_train_0.6b", "H08.local_resume_0.6b"],
}
LOCAL_RUNS = {"local_qwen3_0.6b_resident": "configs/experiments/qwen3_0.6b_local_resident.yaml",
              "local_qwen3_0.6b_l2l_async": "configs/experiments/qwen3_0.6b_local_l2l_async.yaml",
              "local_qwen3_0.6b_smoke": "configs/experiments/qwen3_0.6b_local_smoke.yaml",
              "local_qwen3_0.6b_streaming": "configs/experiments/qwen3_0.6b_local_streaming.yaml"}
# entry -> bitwise comparisons (reference run, candidate run, extra config sections allowed to differ). They are part of
# the entries' own steps; when the runs already existed the dispatcher reports DONE without comparing, so they are
# repeated here and decide the entry (as scripts/assignment.py stage C does for the 4B study)
COMPARISONS = {"H05.local_runs_0.6b": [("local_qwen3_0.6b_resident", "local_qwen3_0.6b_l2l_async", [])],
               "H12.local_train_0.6b": [("local_qwen3_0.6b_l2l_async", "local_qwen3_0.6b_streaming", ["streaming"])]}
LOCAL_PYTEST_CPU = [p for p in A.PYTEST_CPU if "dflash" not in p]
LOCAL_PYTEST_CUDA = [p for p in A.PYTEST_CUDA if "dflash" not in p]
MODES = ("fast", "full")
FAST_STAGES = "AF"
FAST_STAGE_ENTRIES = {"F": ["H12.local_smoke_0.6b", "H12.local_train_fast_0.6b"]}
FAST_RUN_NAMES = ("local_qwen3_0.6b_smoke", "local_qwen3_0.6b_streaming")


def local_config_check(configs):
    """The local protocol's fixed choices, read from the resolved local configs (not from documentation)."""
    res = {name: configs.get(path, {}).get("resolved") for name, path in LOCAL_RUNS.items()}
    if any(c is None for c in res.values()):
        return {"id": "local_config", "kind": "check", "status": "FAIL",
                "detail": f"local configs do not resolve: {[n for n, c in res.items() if c is None]}"}
    checks = {}
    for name, c in res.items():
        t, lo, x = c["training"], c["lora"], c["execution"]
        checks[f"{name}: Qwen/Qwen3-0.6B bf16"] = c["model"]["name"] == "Qwen/Qwen3-0.6B" and c["model"]["dtype"] == "bfloat16"
        checks[f"{name}: B=5 G=6 4 new tokens lr 5e-4 clip 0.2"] = (t["batch_size"], t["group_size"], t["max_new_tokens"],
                                                                     t["learning_rate"], t["clip_eps"]) == (5, 6, 4, 5e-4, 0.2)
        checks[f"{name}: LoRA r32/a64 q/v/o last 2 layers"] = (lo["rank"], lo["alpha"]) == (32, 64) and \
            lo["targets"] in ("last2-attention", {"modules": ["q_proj", "v_proj", "o_proj"], "num_layers": 2})
        checks[f"{name}: DFlash disabled"] = not (c.get("dflash") or {}).get("enabled")
        checks[f"{name}: memory guard on"] = bool((c.get("memory") or {}).get("guard"))
        checks[f"{name}: evaluation every 5"] = c["evaluation"]["every"] == 5
    checks["smoke: 2 steps"] = res["local_qwen3_0.6b_smoke"]["training"]["max_steps"] == 2
    checks["training experiment: 10 steps"] = res["local_qwen3_0.6b_streaming"]["training"]["max_steps"] == 10
    for name in ("local_qwen3_0.6b_l2l_async", "local_qwen3_0.6b_streaming", "local_qwen3_0.6b_smoke"):
        x = res[name]["execution"]
        checks[f"{name}: async L2L, pinned store, 2 buffers, prefetch 1, checkpointing"] = (
            x["backend"], x["weight_storage"], x["weight_buffers"], x["prefetch_depth"], x["activation_checkpointing"]) \
            == ("l2l_async", "pinned", 2, 1, True)
    checks["resident reference is resident"] = res["local_qwen3_0.6b_resident"]["execution"]["backend"] == "resident"
    checks["disk streaming only in the streaming / smoke runs"] = bool(
        res["local_qwen3_0.6b_streaming"]["streaming"]["enabled"] and res["local_qwen3_0.6b_smoke"]["streaming"]["enabled"]
        and not res["local_qwen3_0.6b_l2l_async"]["streaming"]["enabled"]
        and not res["local_qwen3_0.6b_resident"]["streaming"]["enabled"])
    bad = [k for k, v in checks.items() if not v]
    return {"id": "local_config", "kind": "check", "status": "FAIL" if bad else "PASS",
            "detail": ("violated: " + ", ".join(bad)) if bad else "all fixed local choices hold",
            "checks": checks, "notes": ["DFlash is out of scope locally: z-lab/Qwen3-4B-DFlash-b16 drafts for Qwen3-4B "
                                        "only, and no substitute drafter is used"]}


def prior_correctness(state, gpu):
    """The latest earlier local run that ran the correctness stages (B, C) with exactly this run's code identity
    (experiment_code_sha256: every code, config and test file) and GPU: its per-item statuses, never re-judged.
    Otherwise STALE (identity differs) or NOT_TESTED (none)."""
    ident = state["identity"]
    found = []
    for d in sorted(A.RESULTS.glob("*"), reverse=True) if A.RESULTS.exists() else []:
        s = A.read_json(d / "state.json")
        if not s or d.name == state["run_id"] or not any(s["stages"].get(x) for x in "BC"):
            continue
        same = s["identity"].get("experiment_code_sha256") == ident.get("experiment_code_sha256") and \
            (A.facts(s).get("gpu") == gpu)
        stages = {}
        for x in "BC":
            a = A.last_attempt(s, x)
            if a:
                stages[x] = {"attempt": a["n"], "status": a.get("status"), "started_utc": a.get("started_utc"),
                             "ended_utc": a.get("ended_utc"), "items": {i["id"]: i["status"] for i in a["items"]}}
        found.append({"run_id": d.name, "git_commit": s["identity"].get("git_commit"),
                      "experiment_code_sha256": s["identity"].get("experiment_code_sha256"), "gpu": A.facts(s).get("gpu"),
                      "same_identity": same, "stages": stages})
    use = next((r for r in found if r["same_identity"]), None)
    if use:
        status, detail = "PRIOR", f"recorded from run {use['run_id']} (same code identity and GPU), not re-run"
    elif found:
        use, status = found[0], "STALE"
        detail = (f"latest correctness evidence is run {use['run_id']} (commit {str(use['git_commit'])[:10]}, code "
                  f"{str(use['experiment_code_sha256'])[:12]} vs now {str(ident.get('experiment_code_sha256'))[:12]}, "
                  f"GPU {use['gpu']}): code identity or GPU differs, so it is not used")
    else:
        status, detail = "NOT_TESTED", "no earlier local run with correctness stages under " + A.rel(A.RESULTS)
    return {"status": status, "detail": detail, "evidence": use, "candidates": len(found),
            "note": "evidence produced outside scripts/run_assignment_local.sh (e.g. pytest by hand) has no recorded "
                    "code identity and is not used"}


def fast_extras(run_dir, state, items, f):
    """Fast mode, stage A: model/dataset availability (gates F), earlier correctness evidence (recorded, never a gate),
    and the training runs that already exist (reused or refused by the unchanged rules, never overwritten)."""
    models = f.get("models") or {}
    data = A.ROOT / "data" / "sentiment.json"
    checks = {f"{m} cached locally (revision {v.get('cached_revision')})": bool(v.get("cached_revision"))
              for m, v in models.items()}
    checks["Qwen/Qwen3-0.6B among the resolved models"] = "Qwen/Qwen3-0.6B" in models
    checks[f"dataset {A.rel(data)} present"] = data.is_file()
    bad = [k for k, v in checks.items() if not v]
    items.append({"id": "local_assets", "kind": "check", "status": "FAIL" if bad else "PASS", "checks": checks,
                  "detail": ("missing: " + "; ".join(bad)) if bad else "model snapshot and dataset available"})
    f["local_assets_ok"] = not bad
    if not f.get("ram_ok"):  # recorded, not a gate in this mode (the full mode's gate keeps it)
        h = f.get("host") or {}
        f["host_ram_warning"] = (f"host RAM {(h.get('meminfo') or {}).get('MemTotal', 0) / 1e9:.1f} GB < "
                                 f"{h.get('ram_required_bytes', 0) / 1e9:.0f} GB (study host_requirements): training is "
                                 "not blocked in this mode; watch CPU RSS (timings[].cpu_rss_bytes)")
        print(f"   WARNING: {f['host_ram_warning']}")
    pc = prior_correctness(state, f.get("gpu"))
    f["prior_correctness"] = {k: pc[k] for k in ("status", "detail")}
    A.write_json(Path(run_dir) / "environment" / "prior_correctness.json", pc)
    print(f"   earlier correctness evidence: {pc['status']}: {pc['detail']}")
    existing = []
    for name in FAST_RUN_NAMES:
        m = A.read_json(A.RUNS_DIR / name / "manifest.json")
        if m is not None:
            existing.append(f"runs/{name} ({m.get('status')})")
    f["existing_runs"] = existing
    if existing:
        print(f"   NOTE: {', '.join(existing)} already exist and are never overwritten: a completed run with the same "
              "code, config and GPU is reused (not retrained; the report says REUSED PRIOR EVIDENCE), otherwise it is "
              "reported INCONCLUSIVE. To train again without deleting anything: bash scripts/run_assignment_local.sh "
              "--experiment --fresh (moves them to runs/<name>.archived-<UTC>)")


_stage_A = A.stage_A


def local_stage_A(run_dir, state, attempt_dir, study, entries):
    items, f = _stage_A(run_dir, state, attempt_dir, study, entries)
    if state.get("local_mode") == "fast":
        fast_extras(run_dir, state, items, f)
    blockers = []
    for b in f.get("blockers", []):  # rephrase the 4B-oriented blockers for the local stages (A..F without E)
        if b["stages"] == "F" and "per-process" in b["reason"]:
            blockers.append({"stages": "none (recorded)", "reason": "nvidia-smi does not list this pipeline's processes "
                             "with a numeric memory figure (typical under WSL2): measured GPU memory is the allocator's "
                             "and the device-wide reading; per-process figures are reported INCONCLUSIVE"})
        else:
            blockers.append({"stages": b["stages"].replace("D-G", "C-F").replace("C-G", "C-F"), "reason": b["reason"]})
    f["blockers"] = blockers
    A.write_json(Path(run_dir) / "environment" / "blockers.json", blockers)
    return items, f


def local_gate(stage, state):
    f, why = A.facts(state), []
    gpu = bool(f.get("cuda") and f.get("bf16") and f.get("cuda_streams_ok"))
    if stage in "CDF" and not gpu:
        why.append("stage A: no usable native-bf16 CUDA GPU (or the stream/event check failed)")
    if stage in "CDF" and not (f.get("ram_ok") and f.get("disk_ok")):
        why.append("stage A: host RAM or disk below the local requirement")
    if stage in "CF":
        for i in A.ESSENTIAL_B:
            r = A.item(state, "B", i)
            if not r or r["status"] != "PASS":
                why.append(f"stage B: {i} is {r and r['status']}")
    if stage == "F":
        for i in ("H05.local_real_step_0.6b", "H05.local_runs_0.6b"):
            r = A.item(state, "C", i)
            if not r or r["status"] != "PASS":
                why.append(f"stage C: {i} is {r and r['status']} (resident and async L2L must agree before training)")
        r = A.item(state, "D", "H11.local_profile_0.6b")
        if r and (r["status"] == "FAIL" or r.get("memory_status") == "FAIL"):
            why.append(f"stage D: H11.local_profile_0.6b is {r['status']} (memory {r.get('memory_status')})")
    return why


def fast_gate(stage, state):
    """Fast mode: training is gated on stage A's facts only (the correctness stages are not part of this mode)."""
    f, why = A.facts(state), []
    if stage == "F":
        if not (f.get("cuda") and f.get("bf16") and f.get("cuda_streams_ok")):
            why.append("stage A: no usable native-bf16 CUDA GPU (or the stream/event check failed)")
        if not f.get("disk_ok"):
            why.append("stage A: free disk below the local requirement")
        if f.get("assignment_config") != "PASS":
            why.append(f"stage A: local_config is {f.get('assignment_config')} (B, G, LoRA, lr, clip, DFlash off, guard "
                       "on, async L2L settings must hold as configured)")
        if not f.get("local_assets_ok"):
            why.append("stage A: local_assets is not PASS (model snapshot or dataset missing)")
    return why


def compare(attempt_dir, a, b, allow):
    out = Path(attempt_dir) / f"compare_{a}_vs_{b}"
    argv = [*A.tool("compare_runs", [sys.executable, str(ROOT / "scripts" / "compare_runs.py")]), a, b,
            *[x for s in allow for x in ("--allow-differ", s)], "--out", str(out / "compare.json")]
    rec = A.run_item(out, argv, telemetry_on=False)
    rec.update(id=f"compare_{a}_vs_{b}", kind="command", compare_file=A.rel(out / "compare.json"),
               status="PASS" if rec["exit"] == 0 else "FAIL",
               detail=f"compare_runs.py exit {rec['exit']} (0 iff comparable and bitwise identical)")
    return rec


def with_comparisons(stage):
    def runner(run_dir, state, attempt_dir, study, entries):
        items = A.stage_entries(stage, run_dir, state, attempt_dir, study, entries)
        for it in list(items):
            if it["id"] not in COMPARISONS or it["status"] != "PASS":
                continue
            for a, b, allow in COMPARISONS[it["id"]]:
                rec = compare(attempt_dir, a, b, allow)
                items.append(rec)
                if rec["status"] != "PASS":
                    it["status"], it["detail"] = "FAIL", f"{rec['id']}: runs differ or are not comparable ({rec['detail']})"
        return items
    return runner


def stage_F(run_dir, state, attempt_dir, study, entries):
    """The smoke first: if it does not pass, the 10-step run and the resume experiment are not started."""
    ids, items = A.stage_ids("F"), []
    for eid in ids:
        if eid != ids[0] and not any(i["id"] == ids[0] and i["status"] == "PASS" for i in items):
            items.append(A.blocked(eid, f"{ids[0]} did not pass: the 10-step run and the resume test are not started"))
            continue
        full = A.STAGE_ENTRIES
        A.STAGE_ENTRIES = {**full, "F": [eid]}  # one entry at a time through the unchanged stage runner
        try:
            items += with_comparisons("F")(run_dir, state, attempt_dir, study, entries)
        finally:
            A.STAGE_ENTRIES = full
    return items


def install(mode="full"):
    """Point scripts/assignment.py at the local protocol (module globals are read at call time)."""
    if "ASSIGNMENT_STUDY" not in os.environ:
        A.STUDY = LOCAL_STUDY
    if "ASSIGNMENT_RESULTS_ROOT" not in os.environ:
        A.RESULTS = ROOT / "assignment_results_local"
    A.STAGES, A.STAGE_NAMES, A.RAW_DIRS = LOCAL_STAGES, LOCAL_STAGE_NAMES, LOCAL_RAW_DIRS
    A.STAGE_ENTRIES = LOCAL_STAGE_ENTRIES
    A.FRESH, A.CAPPED = {"B", "C", "D"}, {"C", "D", "F"}
    A.PYTEST_CPU, A.PYTEST_CUDA = LOCAL_PYTEST_CPU, LOCAL_PYTEST_CUDA
    A.FINAL_RUNS = LOCAL_RUNS
    A.BUDGET_CONFIGS = ("configs/budget/qwen3-0.6b_local.yaml",)
    A.assignment_config_check = local_config_check
    A.gate = local_gate
    A.RUNNERS = {"A": local_stage_A, "B": A.stage_B, "C": with_comparisons("C"), "D": A.stage_entries, "F": stage_F}
    A.stage_A = local_stage_A  # cmd_stage calls stage_A by name, not through RUNNERS
    A.MODE_KEY, A.MODE = "local_mode", mode
    if mode == "fast":
        A.STAGES, A.STAGE_ENTRIES, A.FRESH, A.CAPPED = FAST_STAGES, FAST_STAGE_ENTRIES, set(), {"F"}
        A.gate = fast_gate
        A.RUNNERS = {"A": local_stage_A, "F": stage_F}


def mode_of(run_dir):
    """The mode a run was started in (runs made before the fast mode existed ran the full protocol)."""
    s = A.read_json(Path(run_dir) / "state.json") if run_dir else None
    return (s or {}).get("local_mode", "full") if s else None


def requested_mode():
    m = os.environ.get("ASSIGNMENT_LOCAL_MODE", "fast")
    if m not in MODES:
        raise SystemExit(f"ASSIGNMENT_LOCAL_MODE must be one of {MODES}, not {m!r}")
    return m


def cmd_status(run_dir):
    """The stage table, then the active (or last) item from <run>/active.json and whether an orchestrator holds the
    run's lock."""
    rc = A.main(["status"] + (["--run-dir", str(run_dir)] if run_dir else []))
    run_dir = Path(run_dir) if run_dir else A.latest_run()
    if run_dir is None:
        return rc
    print(f"mode: {mode_of(run_dir)}")
    alive = None
    if (run_dir / ".lock").exists():
        with open(run_dir / ".lock") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
                alive = False
            except OSError:
                alive = True
    print(f"orchestrator: {'running' if alive else 'not running'}")
    act = A.read_json(run_dir / "active.json")
    if act:
        state = "active" if act.get("running") and alive else "last" if not act.get("running") else \
            "STOPPED (no orchestrator holds the run; an interrupted item)"
        print(f"{state} item: {act['item']} started {act['started_utc']}, elapsed {act['elapsed_s']:.0f} s at "
              f"{act['updated_utc']}, trainer step {act.get('last_step')}\n  $ {' '.join(map(str, act['argv']))}")
        for p in act.get("pids", []):
            print(f"  pid {p['pid']}: {p['cmd'][:160]}")
        print(f"  logs: {act['raw_dir']}/stdout.log, stderr.log")
    return rc


def fast_dry_run():
    study, entries = A.load_study()
    cap = A.probe()
    print(f"dry run at commit {(A.git('rev-parse', '--short', 'HEAD') or '?').strip()} - nothing is written or run")
    print(f"GPU {cap.get('gpu')}, bf16 {cap.get('bf16')}, VRAM {cap.get('vram_gib')} GiB, per-process nvidia-smi "
          f"{cap.get('per_process_smi')}, native Linux {cap.get('linux_native')}")
    c = study["gpu_memory_cap"]
    print(f"GPU memory watchdog: {c['bytes'] / 2**30:.3f} GiB ({c['wording']}); memory.guard on in the configs")
    print("A preflight: provenance, capabilities, CUDA streams/events, resolved local configs (B5 G6, LoRA r32/a64, "
          "DFlash off, guard on), model and dataset availability, earlier correctness evidence (recorded, not re-run)")
    print("correctness suite (stages B C D, the resume experiment): NOT run in this mode; use --full-tests")
    t = A.fresh_targets()
    print("--fresh would move aside: " + (", ".join(A.rel(p) for p in t) if t else "nothing (no local run exists)"))
    for name in FAST_RUN_NAMES:
        m = A.read_json(A.RUNS_DIR / name / "manifest.json")
        print(f"runs/{name}: " + ("absent (will be trained)" if m is None else
                                  f"exists, status {m.get('status')} (never overwritten: reused or refused)"))
    print("F training: 2-step smoke, then (only if it passes) the 10-step run:", flush=True)
    import subprocess
    subprocess.run([*A.dispatcher(), "run", "--dry-run", "--entry", *FAST_STAGE_ENTRIES["F"]], cwd=ROOT)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    install()
    run_dir = Path(argv[argv.index("--run-dir") + 1]) if "--run-dir" in argv else None
    cmd = argv[:1]
    if cmd == ["mode"]:  # the mode a continued run is in: the latest run's, or --want's if a run of that mode exists
        want = argv[argv.index("--want") + 1] if "--want" in argv else None
        if want is None:
            print(mode_of(A.latest_run()) or requested_mode())
            return 0
        install(want)
        if A.latest_run(want) is None:
            last = A.latest_run()
            print(f"no earlier local run in {want} mode to continue" + (
                f" (the latest local run, {last.name}, was started in {mode_of(last)} mode)" if last else "")
                + "; start a new run instead", file=sys.stderr)
            return 2
        print(want)
        return 0
    if cmd in (["stage"], ["report"]):
        mode = mode_of(run_dir)
    elif cmd == ["status"]:
        mode = mode_of(run_dir or A.latest_run()) or "fast"
    elif cmd == ["init"] and ("--resume" in argv or "--from-stage" in argv):
        # the wrapper resolved the mode (scripts/run_assignment_local.sh `mode`); A.cmd_init continues the latest run
        # of that mode and refuses a --run-dir of another one
        mode = os.environ.get("ASSIGNMENT_LOCAL_MODE") or mode_of(run_dir or A.latest_run()) or "fast"
    else:
        mode = requested_mode()
    install(mode)
    if cmd == ["report"]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("assignment_local_report",
                                                      ROOT / "scripts" / "assignment_local_report.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(run_dir)
    if cmd == ["status"]:
        return cmd_status(run_dir)
    if cmd == ["dry-run"]:
        if mode == "fast":
            print("LOCAL run, default mode = --experiment (Qwen3-0.6B, 4 GB GPU): A preflight, F real GRPO + LoRA + async L2L training "
                  "(2-step smoke, then 10 steps). DFlash, the Qwen3-4B 8 GB cap, held-out test evaluation and Soup are "
                  "out of scope (NOT_TESTED). Nothing below is run.")
            return fast_dry_run()
        print("LOCAL laptop validation (Qwen3-0.6B, 4 GB GPU): stages A B C D F; DFlash, the Qwen3-4B 8 GB cap, "
              "held-out evaluation and Soup are out of scope (NOT_TESTED). Nothing below is run.")
    if cmd == ["init"] and "--resume" not in argv and "--from-stage" not in argv:
        import contextlib
        import io
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = A.main(argv)
        if rc == 0:
            d = Path(out.getvalue().strip().splitlines()[-1])
            state = A.load_state(d)
            state["local_mode"] = mode
            A.save_state(d, state)
            man = A.read_json(d / "manifest.json")
            man.update(local_mode=mode, command="bash scripts/run_assignment_local.sh"
                       + (" --full-tests" if mode == "full" else " --experiment") + (" --fresh" if "--fresh" in argv
                                                                                     else ""))
            A.write_json(d / "manifest.json", man)
        sys.stdout.write(out.getvalue())
        return rc
    return A.main(argv)


if __name__ == "__main__":
    sys.exit(main())
