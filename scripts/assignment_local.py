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
"""
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


def local_stage_A(run_dir, state, attempt_dir, study, entries):
    items, f = A.stage_A(run_dir, state, attempt_dir, study, entries)
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


def install():
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


def main(argv=None):
    install()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["report"]:
        import importlib.util
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        spec = importlib.util.spec_from_file_location("assignment_local_report",
                                                      ROOT / "scripts" / "assignment_local_report.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main(run_dir)
    if argv[:1] == ["dry-run"]:
        print("LOCAL laptop validation (Qwen3-0.6B, 4 GB GPU): stages A B C D F; DFlash, the Qwen3-4B 8 GB cap, "
              "held-out evaluation and Soup are out of scope (NOT_TESTED). Nothing below is run.")
    return A.main(argv)


if __name__ == "__main__":
    sys.exit(main())
