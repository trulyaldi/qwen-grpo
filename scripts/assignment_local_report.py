"""Report of one LOCAL laptop validation run (assignment_results_local/<run-id>/), called by
`python scripts/assignment_local.py report --run-dir D` (always, also after a failure or an interrupt).

Same rules as scripts/assignment_report.py, whose helpers it reuses: only this run's state.json, raw telemetry and the
outputs of items that produced a judged result in this run are read; a missing measurement stays MISSING (null), never
0; no requirement is PASS without the measurement it names; a script existing or a config enabling a feature is never
evidence. Statuses: PASS / FAIL / BLOCKED / INCONCLUSIVE / NOT_TESTED. Verdict: LOCAL VALIDATION PASS iff every
essential local requirement is PASS, FAIL if one is FAIL, else INCOMPLETE.
Out of scope, always NOT_TESTED and never claimed: DFlash (the only supported drafter is for Qwen3-4B; no substitute
drafter is used), the Qwen3-4B 8 GB cap, held-out test evaluation / merged model, Soup checks.
Writes requirements.json, measurements/*.{csv,json}, comparisons/*.csv, reports/local_report.md, checksums.sha256.
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("assignment_report", ROOT / "scripts" / "assignment_report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

MIB, GIB = 2**20, 2**30
TINY_EQUIVALENCE = {  # quantity checked -> lab entries (sync on CPU / CUDA, async on CUDA)
    "forward logits": ["H01.lab_sync", "H01.lab_async_cuda"],
    "gradients": ["H02.lab_sync_cpu", "H02.lab_sync_cuda", "H02.lab_async_cuda"],
    "log-probs / output head": ["H03.lab_v2_cpu"],
    "greedy and sampled generation": ["H04.lab_sync_cpu", "H04.lab_sync_cuda", "H04.lab_async_cuda"],
    "GRPO update (adapter after the step)": ["H05.lab_sync_cpu", "H05.lab_sync_cuda", "H05.lab_async_cuda"],
}
FAULTS = {"H01.fault_wrong_layer": "a wrong layer's weights streamed", "H01.fault_unsafe_reuse": "a GPU buffer reused "
          "before its consumer finished (async)", "H02.fault_detach_boundary": "a detached checkpoint boundary (lost "
          "gradient)", "H02.fault_unsafe_reuse": "unsafe buffer reuse in backward", "H03.fault_raw_logit": "raw logits "
          "instead of log-softmax", "H04.fault_stale_position": "stale position ids in decoding",
          "H04.fault_no_padding_mask": "missing padding mask", "H05.fault_advantage": "a corrupted advantage"}
RUNS = {"local_qwen3_0.6b_resident": ("C", "H05.local_runs_0.6b", "resident reference (10 steps)"),
        "local_qwen3_0.6b_l2l_async": ("C", "H05.local_runs_0.6b", "async L2L, in-memory (10 steps)"),
        "local_qwen3_0.6b_smoke": ("F", "H12.local_smoke_0.6b", "integration smoke (2 steps, streaming)"),
        "local_qwen3_0.6b_streaming": ("F", "H12.local_train_0.6b", "local training experiment (10 steps, streaming)")}


def comparison(run, stage, entry_id, a, b):
    """compare_runs.py result for runs a vs b produced in this run: the orchestrator's re-comparison, else the entry's
    own step output in its evidence directory."""
    rec = run.item(stage, f"compare_{a}_vs_{b}")
    if rec and rec.get("compare_file"):
        d = R.read_json(ROOT / rec["compare_file"]) or R.read_json(Path(rec["compare_file"]))
        if d:
            return d, rec["compare_file"]
    it = run.item(stage, entry_id)
    if it and it.get("status") in R.USABLE and it.get("evidence_dir"):
        ed = Path(it["evidence_dir"]) if Path(it["evidence_dir"]).is_absolute() else ROOT / it["evidence_dir"]
        for p in sorted(ed.glob("compare_*.json")) if ed.exists() else []:
            d = R.read_json(p)
            if d and d.get("a") == a and d.get("b") == b:
                return d, str(p)
    return None, None


def compare_detail(c):
    if not c:
        return "no comparison result"
    ctl = c.get("controls") or {}
    ck = c.get("checkpoints") or {}
    return (f"identical {c.get('identical')}: training rows {(c.get('training_rows') or {}).get('equal')} "
            f"({(c.get('training_rows') or {}).get('steps')} steps), zero-signal {c.get('zero_signal_equal')}, greedy "
            f"evaluations {(c.get('evaluations') or {}).get('equal')}, checkpoints adapter/AdamW all equal "
            f"{all(all((v.get(k) or {}).get('equal', True) for k in ('adapter', 'optimizer')) for v in ck.values()) if ck else None}"
            f" ({len(ck)} checkpoints), frozen unchanged {all(v.get('frozen_changed') == [] for v in ctl.values()) if ctl else None}")


def requirements(run, m):
    Q = []
    st = lambda s, i: R.status_of(run.item(s, i))

    def add(rid, clause, essential, status, detail, evidence=(), measured=None):
        assert status in R.STATUS_ORDER, status
        Q.append({"id": rid, "clause": clause, "essential": essential, "status": status, "detail": detail,
                  "evidence": [str(e) for e in evidence if e], "measured": measured})

    it = run.item("A", "local_config")
    add("L1", "local configuration: Qwen3-0.6B bf16, B5 G6, 4 tokens, lr 5e-4, clip 0.2, LoRA r32/a64 q/v/o last 2, "
        "2-step smoke, 10-step run, evaluation every 5, guard on, DFlash off", True, R.status_of(it),
        (it or {}).get("detail", "stage A not run"), ["resolved_configs/"])
    for rid, iid, clause in (("L2", "unit_tests_cpu", "CPU unit tests (GRPO, policy, streaming, merge, profiler, guard, "
                                                       "telemetry)"),
                             ("L3", "silent_failure_tests", "silent-failure tests (tests/silent_failures)"),
                             ("L4", "unit_tests_cuda", "CUDA unit tests (prefetch, backward, generator, streaming, "
                                                        "backend, output head, weight store)")):
        i = run.item("B", iid)
        add(rid, clause, True, R.status_of(i), (i or {}).get("detail", "not run"), [(i or {}).get("raw_dir")])

    # tiny-model equivalence, with tolerances and maximum deviations
    rows, statuses = [], []
    for what, ids in TINY_EQUIVALENCE.items():
        for iid in ids:
            i = run.item("B", iid)
            lab = R.lab_summary(i.get("evidence_name")) if i and i["status"] in R.USABLE else None
            # a PASS without the lab result that shows its deviations is not a measured PASS
            statuses.append("INCONCLUSIVE" if R.status_of(i) == "PASS" and not (lab or {}).get("quantities")
                            else R.status_of(i))
            for q, v in ((lab or {}).get("quantities") or {}).items():
                rows.append([what, iid, q, v["n"], v["max_abs"], v["bound_max"], v["fails"], ";".join(v["dtypes"]),
                             lab.get("gpu") or lab.get("device")])
    m["tiny_equivalence_rows"] = rows
    worst_dev = {}
    for r in rows:
        k = r[0]
        worst_dev[k] = max(worst_dev.get(k, (0.0, 0.0)), (r[4], r[5]))
    add("L5", "resident vs L2L (sync and async) on tiny models: forward logits, gradients, log-probs, generation, GRPO "
        "update", True, R.worst(statuses),
        "; ".join(f"{k}: max |dev| {R.fmt(v[0], '', 3)} (bound {R.fmt(v[1], '', 3)})" for k, v in worst_dev.items())
        or "no lab result", ["comparisons/equivalence.csv"])
    fs = {f: st("B", f) for f in FAULTS}
    add("L6", "silent failures are caught: every injected fault is detected", True, R.worst(fs.values()),
        "; ".join(f"{FAULTS[k]}: {'detected' if v == 'PASS' else v}" for k, v in fs.items()))

    # Qwen3-0.6B resident vs async L2L
    i = run.item("C", "H05.local_real_step_0.6b")
    real = m.get("real_step")
    if real:
        c = real.get("comparison") or {}
        its = c.get("iterations") or []
        ok = real.get("all_equal") is True
        add("L7", "Qwen3-0.6B, one real GRPO step: resident == async L2L (rollouts, forward log-probs, every LoRA "
            "gradient, loss, update; 2 updates)", True, "PASS" if ok and R.status_of(i) == "PASS" else "FAIL",
            f"rollouts equal {real.get('rollouts_equal')}; old log-probs max |dev| "
            f"{R.fmt((c.get('old_logp') or {}).get('max_abs'), '', 3)}; " + "; ".join(
                f"update {k}: loss equal {x['loss']['equal']}, grads equal {x['all_grads_equal']} (worst rel "
                f"{R.fmt(x['worst_grad_rel'], '', 3)}), updates equal {x['all_updates_equal']}" for k, x in enumerate(its))
            + " (tolerance: bitwise, research/l2l/tolerances.yaml update_equivalence)", [m.get("real_step_file")])
    else:
        add("L7", "Qwen3-0.6B, one real GRPO step: resident == async L2L", True, R.missing(i),
            R.why(i, "H05.local_real_step_0.6b"))
    c, f = comparison(run, "C", "H05.local_runs_0.6b", "local_qwen3_0.6b_resident", "local_qwen3_0.6b_l2l_async")
    i = run.item("C", "H05.local_runs_0.6b")
    m["compare_resident_l2l"] = c
    add("L8", "Qwen3-0.6B, 10-step GRPO runs: resident == async L2L bitwise (rollouts, rewards, losses, greedy "
        "evaluations, adapters and AdamW at every checkpoint)", True,
        ("PASS" if c.get("identical") else "FAIL") if c else R.missing(i), compare_detail(c) if c else
        R.why(i, "H05.local_runs_0.6b"), [f])
    logs = m["logs"]
    frozen = {n: lg.get("frozen_changed") for n, lg in logs.items()}
    if frozen:
        bad = {n: v for n, v in frozen.items() if v != []}
        add("L9", "no frozen base weight changed in any local run", True, "FAIL" if bad else "PASS",
            "; ".join(f"{n}: {('unchanged' if v == [] else v)}" for n, v in frozen.items()))
    else:
        add("L9", "no frozen base weight changed in any local run", True, R.worst([R.missing(run.item(s, e))
                                                                                 for s, e, _ in RUNS.values()]),
            "no local training log")
    pt = m.get("prefetch")
    i = run.item("C", "H06.prefetch_timing")
    add("L10", "CUDA copy/compute overlap measured (async vs sync step time, H06)", False,
        R.status_of(i) if pt else R.missing(i),
        (f"{pt['status']}: {json.dumps(pt.get('conclusion'), default=str)[:240]}" if pt else R.why(i, "H06.prefetch_timing")))

    # profiling
    prof, i = m.get("profile"), run.item("D", "H11.local_profile_0.6b")
    if prof:
        phases = (prof.get("profile") or {}).get("phases", [])
        measured = [p for p in phases if p.get("peak_allocated") is not None]
        viol = (prof.get("profile") or {}).get("violations")
        add("L11", "GPU memory measured per phase (allocated, reserved, device) and compared with the prediction", True,
            "FAIL" if viol else ("PASS" if measured else "INCONCLUSIVE"),
            "; ".join(f"{p['phase']}: alloc {R.fmt(R.gib(p.get('peak_allocated')), ' GiB', 2)}, reserved "
                      f"{R.fmt(R.gib(p.get('peak_reserved')), ' GiB', 2)}, device {R.fmt(R.gib(p.get('device_used_peak')), ' GiB', 2)}"
                      for p in measured) + (f"; guard violations {viol}" if viol else ""),
            [m.get("profile_file"), "measurements/memory.csv"])
        ev = (prof.get("profile") or {}).get("evidence")
        add("L12", "NVIDIA-reported per-process GPU memory", False, "PASS" if ev == "assignment" else "INCONCLUSIVE",
            f"profiler evidence grade {ev!r}; " + ("per-process nvidia-smi readings on every sample" if ev == "assignment"
                                                   else "nvidia-smi gave no numeric per-process figure (WSL2 lists no "
                                                        "process memory; containers may hide PIDs): device-wide readings "
                                                        "and allocator figures only"), [m.get("profile_file")])
        tr = prof.get("transfers") or {}
        secs = {p["phase"]: p.get("seconds") for p in phases}
        add("L13", "pinned CPU footprint, L2L transfer bytes, H2D bandwidth, forward / backward / generation / training "
            "times", False,
            "PASS" if prof.get("pinned_bytes") and tr and secs else "INCONCLUSIVE",
            f"pinned {R.fmt(R.gib(prof.get('pinned_bytes')), ' GiB', 3)}, H2D "
            f"{R.fmt((prof.get('h2d_bandwidth_bytes_per_s') or 0) / 1e9 or None, ' GB/s', 2)}; transfers "
            + ", ".join(f"{k} {R.fmt((v.get('bytes') or 0) / 1e9, ' GB', 2)}" for k, v in tr.items())
            + "; seconds " + ", ".join(f"{k} {R.fmt(v, '', 2)}" for k, v in secs.items()), [m.get("profile_file")])
    else:
        for rid, cl in (("L11", "GPU memory measured per phase and compared with the prediction"),
                        ("L12", "NVIDIA-reported per-process GPU memory"),
                        ("L13", "pinned footprint, transfer bytes, phase times")):
            add(rid, cl, rid == "L11", R.missing(i), R.why(i, "H11.local_profile_0.6b"))

    # training
    i = run.item("F", "H12.local_smoke_0.6b")
    sm = logs.get("local_qwen3_0.6b_smoke")
    if sm and R.status_of(i) == "PASS":
        s14 = "PASS" if sm.get("successful_steps") == 2 else "FAIL"
    else:
        s14 = R.status_of(i) if sm else R.missing(i)
    add("L14", "2-step integration smoke completes (async L2L + pinned store + streaming + guard)", True, s14,
        f"updates {sm.get('successful_steps')}" if sm else R.why(i, "H12.local_smoke_0.6b"))
    lg, i = logs.get("local_qwen3_0.6b_streaming"), run.item("F", "H12.local_train_0.6b")
    if lg:
        rows = lg.get("training", [])
        ok = lg.get("successful_steps") == 10 and rows and (rows[-1].get("change_from_init") or 0) > 0 and \
            lg.get("params_equal_init") is False and lg.get("frozen_changed") == []
        cov = R.coverage_of(lg) or {}
        add("L15", "10-step GRPO + LoRA run: metrics recorded, adapter changed, frozen weights unchanged", True,
            "PASS" if ok else "FAIL",
            f"updates {lg.get('successful_steps')}, attempts {lg.get('groups_generated')} ({lg.get('zero_signal_groups')} "
            f"zero-signal); reward first/last {R.fmt(rows[0].get('mean_reward') if rows else None, '', 2)}/"
            f"{R.fmt(rows[-1].get('mean_reward') if rows else None, '', 2)}; median |g| "
            f"{R.fmt(R.median([r.get('grad_norm') for r in rows]), '', 3)}; LoRA change from init "
            f"{R.fmt(rows[-1].get('change_from_init') if rows else None, '', 4)}; |(a/r)BA| "
            f"{R.fmt(rows[-1].get('effective_adapter_norm') if rows else None, '', 4)}; prompts visited "
            f"{cov.get('unique_prompts_visited')}/{cov.get('train_pool_prompts')} (training rewards only; no held-out "
            f"evaluation locally)", [str(R.RUNS / 'local_qwen3_0.6b_streaming' / 'training' / 'training.json')])
    else:
        add("L15", "10-step GRPO + LoRA run", True, R.missing(i), R.why(i, "H12.local_train_0.6b"))
    c, f = comparison(run, "F", "H12.local_train_0.6b", "local_qwen3_0.6b_l2l_async", "local_qwen3_0.6b_streaming")
    m["compare_streaming"] = c
    add("L16", "disk-backed prompt / rollout streaming == in-memory async L2L, bitwise", True,
        ("PASS" if c.get("identical") else "FAIL") if c else R.missing(i), compare_detail(c) if c else
        R.why(i, "H12.local_train_0.6b"), [f])
    res, i = m.get("resume"), run.item("F", "H08.local_resume_0.6b")
    tiny = st("B", "H08.resume_tiny_cpu")
    if res:
        add("L17", "deterministic resume: six injected crashes, resumed runs equal the uninterrupted one (no lost or "
            "duplicated sample)", True, "PASS" if res.get("pass") is True and R.status_of(i) == "PASS" else "FAIL",
            f"Qwen3-0.6B: pass {res.get('pass')}, crash points {list((res.get('B') or {}).keys())}; epoch problems "
            f"{res.get('A_epoch_problems')}; tiny CPU (stage B): {tiny}", [m.get("resume_file")])
    else:
        add("L17", "deterministic resume under injected crashes", True, R.missing(i),
            R.why(i, "H08.local_resume_0.6b") + f"; tiny CPU (stage B): {tiny}")
    perf = m.get("perf_rows") or []
    add("L18", "resident vs async L2L timing (training runs and the real step)", False,
        "PASS" if perf else "NOT_TESTED", "; ".join(f"{r[0]}: {R.fmt(r[1], ' s', 2)}" for r in perf) or "no timing",
        ["comparisons/backend_performance.csv"])

    # out of scope, by design
    for rid, cl, why in (
            ("X1", "DFlash speculative rollouts", "out of scope locally: z-lab/Qwen3-4B-DFlash-b16 is trained for "
             "Qwen3-4B only and no substitute drafter is used; validated only in the Qwen3-4B assignment run"),
            ("X2", "Qwen3-4B under the 8 GB cap", "not this model; the local cap is the 4 GiB card"),
            ("X3", "held-out test evaluation, merged model, LR=0 control", "not part of the local protocol"),
            ("X4", "Soup checks", "not part of the local protocol")):
        add(rid, cl, False, "NOT_TESTED", why)
    return Q


def main(run_dir):
    run = R.Run(run_dir)
    if run.state.get("local_mode") == "fast":
        return fast_main(run)
    d = run.dir
    for sub in ("measurements", "comparisons", "reports"):
        (d / sub).mkdir(exist_ok=True)
    m = {"logs": {}}
    for name, (stage, iid, _) in RUNS.items():
        lg = run.run_log(name, stage, iid)
        if lg:
            m["logs"][name] = lg
    real = run.output("C", "H05.local_real_step_0.6b")
    if real:
        m["real_step"], m["real_step_file"] = real, str(R.EXP / f"{run.item('C', 'H05.local_real_step_0.6b')['evidence_name']}.json")
    prof = run.output("D", "H11.local_profile_0.6b")
    if prof:
        m["profile"], m["profile_file"] = prof, str(R.EXP / f"{run.item('D', 'H11.local_profile_0.6b')['evidence_name']}.json")
    res = run.output("F", "H08.local_resume_0.6b")
    if res:
        m["resume"], m["resume_file"] = res, str(R.EXP / f"{run.item('F', 'H08.local_resume_0.6b')['evidence_name']}.json")
    pi = run.item("C", "H06.prefetch_timing")
    m["prefetch"] = R.lab_summary(pi.get("evidence_name")) if pi and pi["status"] in R.USABLE else None
    perf = []
    for name in ("local_qwen3_0.6b_resident", "local_qwen3_0.6b_l2l_async", "local_qwen3_0.6b_streaming"):
        lg = m["logs"].get(name)
        if lg:
            perf.append([f"{name} median step", R.median([t.get("step_s") for t in lg.get("timings", [])]),
                         R.median([t.get("rollout_s") for t in lg.get("timings", [])]),
                         R.median([t.get("update_s") for t in lg.get("timings", [])]),
                         max((r.get("peak_reserved_gib") or 0 for r in lg.get("training", [])), default=None), "measured"])
    if real and real.get("resident_generation"):
        perf.append(["real step L2L generation", (real.get("l2l_generation") or {}).get("seconds"), None, None,
                     None, "measured, one step"])
        perf.append(["real step resident generation", real["resident_generation"].get("seconds"), None, None, None,
                     "measured, one step"])
    m["perf_rows"] = perf
    reqs = requirements(run, m)
    ess = [r for r in reqs if r["essential"]]
    verdict = "PASS" if all(r["status"] == "PASS" for r in ess) else \
        "FAIL" if any(r["status"] == "FAIL" for r in ess) else "INCOMPLETE"
    from qwen_grpo.evidence import code_sha256, report_code_sha256
    (d / "requirements.json").write_text(json.dumps({
        "verdict": f"LOCAL VALIDATION {verdict}", "scope": "Qwen3-0.6B engineering validation on a 4 GB GPU; does NOT "
        "verify DFlash or the Qwen3-4B assignment", "rule": "PASS iff every essential local requirement is PASS",
        "requirements": reqs, "report_identity": {"report_code_sha256": report_code_sha256(),
                                                  "experiment_code_sha256_now": code_sha256()},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1, default=str) + "\n")

    # measurements
    mem = []
    for stage, it in run.all_items():
        t = it.get("telemetry")
        if t:
            mem.append([stage, it["id"], "whole item", "nvidia-smi per-process", "measured" if t["process_peak_mib"]
                        is not None else "unavailable", None if t["process_peak_mib"] is None else
                        int(t["process_peak_mib"] * MIB), t["raw_files"]["apps"], t.get("memory_reason")])
            mem.append([stage, it["id"], "whole item", "nvidia-smi device memory.used", "measured" if
                        t["device_peak_used_mib"] is not None else "unavailable", None if t["device_peak_used_mib"]
                        is None else int(t["device_peak_used_mib"] * MIB), t["raw_files"]["gpu"], "all processes"])
    for p in ((prof or {}).get("profile") or {}).get("phases", []):
        for q, k in (("PyTorch peak allocated", "peak_allocated"), ("PyTorch peak reserved", "peak_reserved"),
                     ("cudaMemGetInfo device used", "device_used_peak"), ("nvidia-smi per-process", "process_used_peak"),
                     ("CPU RSS", "cpu_rss_peak")):
            mem.append(["D", "H11.local_profile_0.6b", p["phase"], q, "measured" if p.get(k) is not None else
                        "unavailable", p.get(k), m.get("profile_file"), f"{p.get('samples')} samples"])
    if prof and prof.get("pinned_bytes") is not None:
        mem.append(["D", "H11.local_profile_0.6b", "setup", "pinned host memory (weight store)", "measured",
                    prof["pinned_bytes"], m.get("profile_file"), ""])
    for r in ((prof or {}).get("prediction") or {}).get("rows", []):
        for q, k in (("predicted allocated", "predicted_allocated"), ("predicted reserved", "predicted_reserved"),
                     ("predicted device", "predicted_device"), ("predicted CPU", "predicted_cpu")):
            mem.append(["D", "budget", r["phase"], q, "theoretical", r.get(k), prof["prediction"]["file"],
                        f"budget phase {r.get('predicted_phase')}"])
    for name, lg in m["logs"].items():
        pk = max((r.get("peak_reserved_gib") or 0 for r in lg.get("training", [])), default=0)
        mem.append([RUNS[name][0], name, "training steps", "PyTorch peak reserved", "measured" if pk else "unavailable",
                    int(pk * GIB) if pk else None, str(R.RUNS / name / "training" / "training.json"), "excludes CUDA context"])
    R.write_csv(d / "measurements" / "memory.csv", ["stage", "item", "phase", "quantity", "kind", "bytes", "source",
                                                    "note"], mem)
    timing = [[name, t.get("step"), t.get("rollout_s"), t.get("prepare_s"), t.get("update_s"), t.get("step_s"),
               t.get("weight_bytes_copied"), t.get("backend")] for name, lg in m["logs"].items()
              for t in lg.get("timings", [])]
    timing += [[f"profile:{p['phase']}", None, None, None, None, p.get("seconds"),
                ((prof.get("transfers") or {}).get(p["phase"]) or {}).get("bytes"), "l2l_async"]
               for p in ((prof or {}).get("profile") or {}).get("phases", [])]
    R.write_csv(d / "measurements" / "timing.csv", ["run_or_phase", "step", "rollout_s", "prepare_s", "update_s",
                                                    "step_s", "weight_bytes_copied", "backend"], timing)
    R.write_csv(d / "comparisons" / "equivalence.csv", ["checked", "item", "quantity", "comparisons", "max_abs_error",
                                                        "max_bound", "fails", "dtypes", "hardware"],
                m["tiny_equivalence_rows"])
    R.write_csv(d / "comparisons" / "backend_performance.csv", ["what", "median_step_or_seconds", "median_rollout_s",
                                                                "median_update_s", "peak_reserved_gib", "kind"], perf)
    (d / "measurements" / "local.json").write_text(json.dumps(
        {k: v for k, v in m.items() if k not in ("logs",)}, indent=1, default=str)[:2_000_000] + "\n")

    # report
    blockers = run.facts().get("blockers", [])
    md = ["# Local laptop validation (Qwen3-0.6B)", "",
          f"**Verdict: LOCAL VALIDATION {verdict}** (PASS iff every essential local requirement is PASS).", "",
          "Scope: the engineering components the Qwen3-4B assignment reuses, exercised on Qwen3-0.6B and a 4 GB GPU. "
          "**This run does not verify DFlash or the Qwen3-4B assignment** (8 GB cap, held-out learning); those are "
          "NOT_TESTED here by design.", "",
          f"Run `{d.name}`, commit `{str((run.state.get('identity') or {}).get('git_commit'))[:10]}`, GPU "
          f"{run.facts().get('gpu')}, per-process nvidia-smi {run.facts().get('per_process_smi')}, native Linux "
          f"{run.facts().get('linux_native')}.", "", "## Requirements", "",
          R.table(["id", "requirement", "essential", "status", "detail"],
                  [[r["id"], r["clause"], "yes" if r["essential"] else "no", f"**{r['status']}**", r["detail"][:400]]
                   for r in reqs]), "", "## Stages", ""]
    for s in "ABCDF":
        a = run.attempt(s)
        md.append(f"- {s}: " + ("not started" if not a else f"{a.get('status')} (attempt {a['n']})"
                                + (f", blocked by: {'; '.join(a['blocked_by'])}" if a.get("blocked_by") else "")))
    md += ["", "## Numerical tolerances and maximum deviations", "",
           "Every resident-vs-L2L threshold in research/l2l/tolerances.yaml for these cases is bitwise (0) unless the "
           "lab case states its own bound (bf16 forward/gradient grids); the bound column is the one each comparison "
           "was judged against.", "",
           R.table(["checked", "item", "quantity", "n", "max |dev|", "bound", "fails"],
                   [[r[0], r[1], r[2], r[3], R.fmt(r[4], '', 3), R.fmt(r[5], '', 3), r[6]]
                    for r in m["tiny_equivalence_rows"]]) if m["tiny_equivalence_rows"] else "MISSING: no lab result.", "",
           "## Memory: predicted vs measured", ""]
    rows = ((prof or {}).get("prediction") or {}).get("rows", [])
    md.append(R.table(["phase", "predicted alloc", "measured alloc", "predicted reserved", "measured reserved",
                       "predicted device", "measured device", "measured per-process"],
                      [[r["phase"], R.fmt(R.gib(r.get("predicted_allocated")), " GiB", 2),
                        R.fmt(R.gib(r.get("observed_peak_allocated")), " GiB", 2),
                        R.fmt(R.gib(r.get("predicted_reserved")), " GiB", 2),
                        R.fmt(R.gib(r.get("observed_peak_reserved")), " GiB", 2),
                        R.fmt(R.gib(r.get("predicted_device")), " GiB", 2),
                        R.fmt(R.gib(r.get("observed_device_peak")), " GiB", 2),
                        R.fmt(R.gib(r.get("observed_process_peak")), " GiB", 2)] for r in rows])
              if rows else "MISSING: no profile in this run.")
    md += ["", "## Silent-failure findings", ""]
    md += [f"- {r['id']} **{r['status']}**: {r['clause']} - {r['detail'][:300]}" for r in reqs
           if r["status"] in ("FAIL", "INCONCLUSIVE") or r["id"] == "L6"]
    md += [f"- preflight: {b['stages']}: {b['reason']}" for b in blockers]
    md += ["", "## Monitoring limits", "",
           "- nvidia-smi is sampled every 200 ms (telemetry) / 0.5 s (memory guard): measured peaks are lower bounds.",
           "- Under WSL2, nvidia-smi lists no per-process memory: the memory guard falls back to the device-wide "
           "cudaMemGetInfo figure (other processes included, an over-estimate) and per-process requirements are "
           "INCONCLUSIVE, never PASS.",
           "- The local memory guard's budget is the 4 GiB card itself: it records rather than limits.",
           "- One run per configuration: timings are indicative, not a benchmark (H06 is the repeated timing case).",
           "", "Raw evidence: `raw/`, `environment/`, `measurements/`, `comparisons/`; checksums in `checksums.sha256`."]
    (d / "reports" / "local_report.md").write_text("\n".join(md) + "\n")
    lines = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(d)}")
    (d / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(f"LOCAL VALIDATION {verdict} ({sum(r['status'] == 'PASS' for r in reqs)}/{len(reqs)} PASS); report "
          f"{d / 'reports' / 'local_report.md'}")
    for r in reqs:
        print(f"  {r['status']:12s} {r['id']:4s} {r['clause'][:100]}")
    return 0 if verdict == "PASS" else 1


# --- default (fast) mode: the real training run, no correctness stages ----------------------------------------------

FAST_RUNS = {"local_qwen3_0.6b_smoke": ("F", "H12.local_smoke_0.6b", 2, "2-step integration smoke"),
             "local_qwen3_0.6b_streaming": ("F", "H12.local_train_fast_0.6b", 10, "10-step training experiment")}


def sign_test_p(b, c):
    """Exact two-sided McNemar (sign) test on the discordant pairs b (wrong -> right) and c (right -> wrong)."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def checkpoint_identity(name):
    d = R.RUNS / name / "checkpoints"
    h = R.adapter_hash(d)
    if not h:
        return None
    ck = d / h["checkpoint"]
    try:
        from qwen_grpo.streaming.resume import validate
        ok, why = validate(ck, resumable=False)
    except Exception as e:  # noqa: BLE001 - recorded, not fatal
        ok, why = None, f"{type(e).__name__}: {e}"
    return {**h, "path": str(ck), "commit_json": (ck / "COMMIT.json").exists(), "valid": ok, "why": why}


def guard_report(name):
    files = sorted((R.RUNS / name / "training" / "memory").glob("*.json"))
    reps = [r for r in (R.read_json(f) for f in files) if r]
    if not reps:
        return None
    return {"files": [str(f) for f in files], "violations": [v for r in reps for v in (r.get("violations") or [])],
            "evidence": [r.get("evidence") for r in reps], "wsl2": any(r.get("wsl2") for r in reps),
            "peak": {k: max(((r.get("peak") or {}).get(k) or 0 for r in reps), default=None)
                     for k in ("device_used", "process_used", "reserved", "allocated")},
            "budget_bytes": reps[-1].get("budget_bytes")}


def fast_requirements(run, m):
    Q = []

    def add(rid, clause, essential, status, detail, evidence=()):
        assert status in R.STATUS_ORDER, status
        Q.append({"id": rid, "clause": clause, "essential": essential, "status": status, "detail": detail,
                  "evidence": [str(e) for e in evidence if e]})

    f = run.facts()
    it = run.item("A", "local_config")
    add("T1", "configuration as specified: Qwen3-0.6B bf16, B=5 G=6, 4 tokens, lr 5e-4, clip 0.2, LoRA r32/a64 q/v/o "
        "last 2 layers, async L2L (pinned, 2 buffers, prefetch 1, checkpointing), disk streaming, guard on, eval every 5, "
        "DFlash off", True, R.status_of(it), (it or {}).get("detail", "stage A not run"), ["resolved_configs/"])
    cu, a = run.item("A", "cuda_streams_events"), run.item("A", "local_assets")
    pre = R.worst([R.status_of(cu), R.status_of(a)]) if f.get("bf16") else ("FAIL" if f.get("cuda") else "BLOCKED")
    add("T2", "preflight: CUDA, native bf16, CUDA streams/events, model snapshot and dataset available", True, pre,
        f"GPU {f.get('gpu')}, bf16 {f.get('bf16')}; streams: {(cu or {}).get('detail')}; assets: {(a or {}).get('detail')}"
        + (f"; WARNING {f['host_ram_warning']}" if f.get("host_ram_warning") else ""))
    for rid, name in (("T3", "local_qwen3_0.6b_smoke"), ("T4", "local_qwen3_0.6b_streaming")):
        stage, iid, steps, what = FAST_RUNS[name]
        i, lg = run.item(stage, iid), m["logs"].get(name)
        if lg:
            rows = lg.get("training", [])
            ok = lg.get("successful_steps") == steps and bool(rows) and (rows[-1].get("change_from_init") or 0) > 0 \
                and lg.get("params_equal_init") is False and lg.get("frozen_changed") == []
            cov = R.coverage_of(lg) or {}
            add(rid, f"{what}: real rollouts, rewards, GRPO gradients and {steps} AdamW updates; LoRA changed", True,
                "PASS" if ok and R.status_of(i) == "PASS" else "FAIL",
                f"{'reused existing run (not trained in this launch)' if (i or {}).get('reused') else 'trained in this launch'}; "
                f"updates {lg.get('successful_steps')}/{steps}, rollout batches {lg.get('groups_generated')} "
                f"({lg.get('zero_signal_groups')} zero-advantage batches skipped); reward per step "
                f"{[R.fmt(r.get('mean_reward'), '', 3) for r in rows]}; LoRA |theta - theta0| "
                f"{R.fmt(rows[-1].get('change_from_init') if rows else None, '', 4)}, |(a/r)BA| "
                f"{R.fmt(rows[-1].get('effective_adapter_norm') if rows else None, '', 4)}; unique prompts "
                f"{cov.get('unique_prompts_visited')}/{cov.get('train_pool_prompts')}",
                [str(R.RUNS / name / "training" / "training.json"), (i or {}).get("raw_dir")])
        else:
            add(rid, f"{what}", True, R.missing(i), R.why(i, iid))
    fz = {n: lg.get("frozen_changed") for n, lg in m["logs"].items()}
    add("T5", "frozen base weights unchanged (digest of every frozen tensor before and after)", True,
        ("FAIL" if any(v != [] for v in fz.values()) else "PASS") if fz else "NOT_TESTED",
        "; ".join(f"{n}: {'unchanged' if v == [] else v}" for n, v in fz.items()) or "no training log")
    # memory
    i = run.item("F", "H12.local_train_fast_0.6b")
    lg = m["logs"].get("local_qwen3_0.6b_streaming")
    t, g = (i or {}).get("telemetry") or {}, m.get("guard")
    mem = (lg or {}).get("memory") or {}
    viol = bool((g or {}).get("violations")) or (i or {}).get("memory_status") == "FAIL"
    dev = t.get("device_peak_used_mib")
    if not lg:
        st = R.missing(i)
    elif viol:
        st = "FAIL"
    else:  # per-process figure if nvidia-smi gives one (not under WSL2), else the whole device's
        pp = t.get("process_peak_mib")
        st = "PASS" if (pp is not None and pp * MIB <= run.cap) or (dev is not None and dev * MIB <= run.cap) \
            else "INCONCLUSIVE"
    add("T6", "GPU memory within the 4 GiB card: no guard / watchdog violation; device-wide nvidia-smi peak", True, st,
        f"PyTorch peak allocated {R.fmt(mem.get('peak_allocated_gib'), ' GiB', 3)}, peak reserved "
        f"{R.fmt(mem.get('peak_reserved_gib'), ' GiB', 3)} (allocator, excludes the CUDA context: not device evidence); "
        f"nvidia-smi device memory.used peak {R.fmt(dev, ' MiB', 0)} of {R.fmt(t.get('device_total_mib'), ' MiB', 0)} "
        f"(whole GPU, every process; on a 4 GiB card it cannot exceed the card, so the evidence is completion without "
        f"CUDA OOM or guard violation); per-process peak {R.fmt(t.get('process_peak_mib'), ' MiB', 0)}; guard violations {(g or {}).get('violations')}; guard peak device "
        f"{R.fmt(R.gib((g or {}).get('peak', {}).get('device_used')), ' GiB', 3)}",
        [(g or {}).get("files", [None])[-1], (t.get("raw_files") or {}).get("gpu")])
    pp = t.get("process_peak_mib")
    add("T7", "NVIDIA-reported per-process GPU memory", False, "PASS" if pp is not None else "INCONCLUSIVE",
        f"per-process peak {pp:.0f} MiB" if pp is not None else
        f"nvidia-smi gave no numeric per-process figure ({t.get('memory_reason') or 'no telemetry'}); typical under "
        "WSL2: only device-wide and allocator figures exist")
    tm = (lg or {}).get("timings") or []
    tot = {k: sum(x.get(k) or 0 for x in tm) for k in ("rollout_s", "prepare_s", "update_s", "step_s")}
    add("T8", "timing: rollout generation vs old-log-prob preparation vs update (forward + backward + AdamW)", False,
        "PASS" if tm else R.missing(i),
        f"over {len(tm)} steps: rollout {tot['rollout_s']:.1f} s, prepare {tot['prepare_s']:.1f} s, update "
        f"{tot['update_s']:.1f} s, steps {tot['step_s']:.1f} s; train_seconds {R.fmt((lg or {}).get('train_seconds'), ' s', 1)}"
        f"; orchestrator item {R.fmt((i or {}).get('seconds'), ' s', 1)} (incl. model load and evaluation)"
        if tm else "no timing")
    # evaluation: step 0 is the base policy (LoRA B = 0 is asserted by the trainer)
    evs = {e["step"]: e for e in (lg or {}).get("evaluations", [])}
    if evs and 0 in evs and max(evs) > 0:
        b, fin = evs[0], evs[max(evs)]
        mb, mf = b["metrics"], fin["metrics"]
        pb = {p["id"]: p["correct"] for p in b["predictions"]}
        pf = {p["id"]: p["correct"] for p in fin["predictions"]}
        up = sum(1 for k in pf if pf[k] and not pb.get(k))
        down = sum(1 for k in pf if not pf[k] and pb.get(k))
        pval = sign_test_p(up, down)
        m["eval"] = {"base": mb, "final": mf, "final_step": max(evs), "wrong_to_right": up, "right_to_wrong": down,
                     "sign_test_p": pval}
        add("T9", "validation subset: base policy (step 0) vs final adapter: accuracy, macro F1, class recall, invalid "
            "rate, prediction distribution", False, "PASS",
            f"n={mf['n']}; step 0 -> {max(evs)}: acc {mb['accuracy']:.3f} -> {mf['accuracy']:.3f}, F1 "
            f"{mb['macro_f1']:.3f} -> {mf['macro_f1']:.3f}, recall POS {mb['recall_POSITIVE']:.3f} -> "
            f"{mf['recall_POSITIVE']:.3f}, NEG {mb['recall_NEGATIVE']:.3f} -> {mf['recall_NEGATIVE']:.3f}, invalid "
            f"{mb['invalid_rate']:.3f} -> {mf['invalid_rate']:.3f}, predicted POS {mb['predicted_positive_rate']:.3f} -> "
            f"{mf['predicted_positive_rate']:.3f}")
        better = mf["accuracy"] > mb["accuracy"]
        add("T10", "improvement over the base policy (paired, exact sign test on the validation subset)", False,
            ("PASS" if better else "FAIL") if pval < 0.05 else "INCONCLUSIVE",
            f"{up} wrong->right, {down} right->wrong of n={mf['n']}, exact two-sided p = {pval:.3g}"
            + ("" if pval < 0.05 else ": no improvement is claimed")
            + "; validation subset only, single seed, not the held-out test split")
    else:
        add("T9", "validation subset: base vs final adapter", False, R.missing(i), "no step-0 and final evaluation")
        add("T10", "improvement over the base policy", False, "NOT_TESTED", "no evaluation")
    ck = m.get("checkpoint")
    fc = (lg or {}).get("final_checkpoint") or {}
    add("T11", "checkpoint identity: final adapter checkpoint committed (COMMIT.json), reloaded bitwise", False,
        ("PASS" if ck.get("valid") and fc.get("reloaded_bitwise_equal") else "FAIL") if ck and lg else R.missing(i),
        f"{fc.get('path')} (step {fc.get('step')}), adapter sha256 {str(ck.get('sha256'))[:16]}, COMMIT.json "
        f"{ck.get('commit_json')}, valid {ck.get('valid')} {ck.get('why') or ''}, reloaded bitwise "
        f"{fc.get('reloaded_bitwise_equal')}" if ck else "no checkpoint")
    res = {n: lg.get("resumed") for n, lg in m["logs"].items() if lg.get("resumed")}
    rr = [r for r in run.all_items() if r[1]["id"].startswith("resume:")]
    add("T12", "resume: restored from the latest valid committed checkpoint", False,
        R.worst([r[1]["status"] for r in rr]) if rr else ("PASS" if res else "NOT_TESTED"),
        "; ".join(f"{n}: resumed from {x['from']} at optimizer step {x['step']} (attempt {x['attempts']})"
                  for n, v in res.items() for x in v)
        or ("; ".join(f"{r[1]['id']}: {r[1]['status']} {r[1].get('detail')}" for r in rr))
        or "no interruption in this run (the injected-crash resume experiment runs under --full-tests)")
    pc = f.get("prior_correctness") or {}
    add("T13", "correctness suite (CPU / CUDA unit tests, silent failures, L2L equivalence): not re-run in this mode",
        False, "PASS" if pc.get("status") == "PRIOR" and m.get("prior_ok") else "NOT_TESTED",
        f"{pc.get('status', 'NOT_TESTED')}: {pc.get('detail', 'not recorded')}" + (f"; {m['prior_summary']}"
                                                                                   if m.get("prior_summary") else ""),
        ["environment/prior_correctness.json"])
    for rid, cl, why in (
            ("X1", "DFlash speculative rollouts", "never enabled locally: z-lab/Qwen3-4B-DFlash-b16 is for Qwen3-4B only; "
             "this run does not validate the official Qwen3-4B DFlash drafter"),
            ("X2", "Qwen3-4B under the 8 GB cap", "not this model; the local limit is the 4 GiB card"),
            ("X3", "held-out test evaluation, merged model, LR=0 control", "not part of the local protocol"),
            ("X4", "Soup checks", "not part of the local protocol")):
        add(rid, cl, False, "NOT_TESTED", why)
    return Q


def unsafe_reuse_finding(pc, results_root):
    """H02.fault_unsafe_reuse from the earlier run that recorded it (scripts/assignment_report.py does the analysis)."""
    ev = (pc or {}).get("evidence") or {}
    st = R.read_json(Path(results_root) / ev.get("run_id", "") / "state.json") if ev else None
    it = next((i for a in reversed((st or {}).get("stages", {}).get("B", {}).get("attempts", []))
               for i in a.get("items", []) if i["id"] == "H02.fault_unsafe_reuse"), None)
    return R.unsafe_reuse_finding(it, ev.get("run_id"))


unsafe_reuse_md = R.unsafe_reuse_md


def fast_main(run):
    d = run.dir
    for sub in ("measurements", "reports"):
        (d / sub).mkdir(exist_ok=True)
    m = {"logs": {}}
    for name, (stage, iid, _, _) in FAST_RUNS.items():
        lg = run.run_log(name, stage, iid)
        if lg:
            m["logs"][name] = lg
    m["guard"] = guard_report("local_qwen3_0.6b_streaming") if "local_qwen3_0.6b_streaming" in m["logs"] else None
    m["checkpoint"] = checkpoint_identity("local_qwen3_0.6b_streaming") if "local_qwen3_0.6b_streaming" in m["logs"] \
        else None
    pc = R.read_json(d / "environment" / "prior_correctness.json") or {}
    if pc.get("evidence"):
        its = {k: v for s in pc["evidence"]["stages"].values() for k, v in s["items"].items()}
        counts = {}
        for v in its.values():
            counts[v] = counts.get(v, 0) + 1
        m["prior_ok"] = bool(its) and all(v == "PASS" for v in its.values())
        m["prior_summary"] = (f"run {pc['evidence']['run_id']}: {counts}; not PASS: "
                              + (", ".join(f"{k} {v}" for k, v in its.items() if v != "PASS") or "none"))
    m["unsafe_reuse"] = unsafe_reuse_finding(pc, d.parent)
    reqs = fast_requirements(run, m)
    ess = [r for r in reqs if r["essential"]]
    verdict = "PASS" if all(r["status"] == "PASS" for r in ess) else \
        "FAIL" if any(r["status"] == "FAIL" for r in ess) else "INCOMPLETE"
    cons = fast_conclusions(reqs)
    banner = R.REUSE_BANNER if (run.item("F", "H12.local_train_fast_0.6b") or {}).get("prior_evidence") else None
    from qwen_grpo.evidence import code_sha256, report_code_sha256
    (d / "requirements.json").write_text(json.dumps({
        "verdict": f"LOCAL TRAINING {verdict}", "mode": "fast", "mode_name": "experiment", "conclusions": cons,
        "reuse_banner": banner, "prior_evidence": R.prior_evidence(run),
        "scope": "Qwen3-0.6B GRPO + LoRA + async L2L + disk streaming on a 4 GB GPU; the correctness suite was not "
                 "re-run; does NOT verify DFlash or the Qwen3-4B assignment",
        "rule": "PASS iff every essential requirement (T1-T6) is PASS", "requirements": reqs,
        "report_identity": {"report_code_sha256": report_code_sha256(), "experiment_code_sha256_now": code_sha256()},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}, indent=1, default=str) + "\n")
    R.write_csv(d / "measurements" / "training.csv", ["run", "step", "attempt", "mean_reward", "loss", "grad_norm",
                                                      "update_norm", "change_from_init", "effective_adapter_norm",
                                                      "peak_allocated_gib", "peak_reserved_gib"],
                [[n, r.get("step"), r.get("attempt"), r.get("mean_reward"), r.get("loss"), r.get("grad_norm"),
                  r.get("update_norm"), r.get("change_from_init"), r.get("effective_adapter_norm"),
                  r.get("peak_allocated_gib"), r.get("peak_reserved_gib")]
                 for n, lg in m["logs"].items() for r in lg.get("training", [])])
    R.write_csv(d / "measurements" / "timing.csv", ["run", "step", "rollout_s", "prepare_s", "update_s", "step_s",
                                                    "weight_bytes_copied", "backend"],
                [[n, t.get("step"), t.get("rollout_s"), t.get("prepare_s"), t.get("update_s"), t.get("step_s"),
                  t.get("weight_bytes_copied"), t.get("backend")] for n, lg in m["logs"].items()
                 for t in lg.get("timings", [])])
    R.write_csv(d / "measurements" / "evaluation.csv", ["run", "step", "n", "accuracy", "macro_f1", "recall_POSITIVE",
                                                        "recall_NEGATIVE", "invalid_rate", "predicted_positive_rate",
                                                        "predicted_negative_rate"],
                [[n, e["step"], *(e["metrics"].get(k) for k in ("n", "accuracy", "macro_f1", "recall_POSITIVE",
                                                                  "recall_NEGATIVE", "invalid_rate",
                                                                  "predicted_positive_rate", "predicted_negative_rate"))]
                 for n, lg in m["logs"].items() for e in lg.get("evaluations", [])])
    mem = []
    for stage, it in run.all_items():
        t = it.get("telemetry")
        if t:
            mem.append([stage, it["id"], "nvidia-smi per-process peak", t.get("process_peak_mib"), t.get("memory_reason")])
            mem.append([stage, it["id"], "nvidia-smi device memory.used peak (all processes)", t.get("device_peak_used_mib"),
                        ""])
    for n, lg in m["logs"].items():
        mm = lg.get("memory") or {}
        for k in ("peak_allocated_gib", "peak_reserved_gib"):
            mem.append(["F", n, f"PyTorch {k}", None if mm.get(k) is None else mm[k] * 1024, "allocator; no CUDA context"])
    R.write_csv(d / "measurements" / "memory.csv", ["stage", "item", "quantity", "MiB", "note"], mem)
    (d / "measurements" / "local.json").write_text(json.dumps({k: v for k, v in m.items() if k != "logs"}, indent=1,
                                                              default=str) + "\n")
    f = run.facts()
    md = ["# Local training run (Qwen3-0.6B, experiment mode)", "",
          f"**Verdict: LOCAL TRAINING {verdict}** (PASS iff every essential requirement T1-T6 is PASS). This is "
          "experimental evidence on Qwen3-0.6B, not a qualification of the Qwen3-4B assignment.", "",
          *R.conclusions_md(cons, banner),
          "Scope: one real GRPO + LoRA training run through asynchronous L2L (pinned CPU weights, 2 GPU buffers, "
          "prefetch depth 1, activation checkpointing) with disk-backed streaming, on a 4 GB GPU. The correctness suite "
          "was **not** re-run (see T13). **This local experiment does not validate the official Qwen3-4B DFlash "
          "drafter** or anything about Qwen3-4B under the 8 GB cap.", "",
          f"Run `{d.name}`, commit `{str((run.state.get('identity') or {}).get('git_commit'))[:10]}`, code "
          f"`{str((run.state.get('identity') or {}).get('experiment_code_sha256'))[:12]}`, GPU {f.get('gpu')}, "
          f"per-process nvidia-smi {f.get('per_process_smi')}, native Linux {f.get('linux_native')}.", "",
          "## Requirements", "",
          R.table(["id", "requirement", "essential", "status", "detail"],
                  [[r["id"], r["clause"], "yes" if r["essential"] else "no", f"**{r['status']}**", r["detail"][:600]]
                   for r in reqs]), "", "## Stages", ""]
    for s in "AF":
        a = run.attempt(s)
        md.append(f"- {s}: " + ("not started" if not a else f"{a.get('status')} (attempt {a['n']})"
                                + (f", blocked by: {'; '.join(a['blocked_by'])}" if a.get("blocked_by") else "")))
    md += ["", "## Known correctness finding: H02.fault_unsafe_reuse", ""] + unsafe_reuse_md(m["unsafe_reuse"])
    md += ["", "## Limitations", "",
           "- 10 updates on 64 training prompts, one seed: a smoke-scale experiment, not evidence of learning.",
           "- Evaluation is the 32-example validation subset (greedy); no held-out test split, no confidence interval "
           "beyond the exact sign test in T10.",
           "- nvidia-smi is sampled (lower-bound peaks); under WSL2 there is no per-process figure, and the device-wide "
           "figure includes every process on the GPU.",
           "- The 4 GiB guard budget is the card itself: it records rather than limits.",
           "", "Raw evidence: `raw/`, `environment/`, `measurements/`, the runs under `runs/`; checksums in "
               "`checksums.sha256`."]
    (d / "reports" / "local_report.md").write_text("\n".join(md) + "\n")
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(d)}" for p in sorted(d.rglob("*"))
             if p.is_file() and p.name not in ("checksums.sha256", ".lock")]
    (d / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(f"LOCAL TRAINING {verdict} ({sum(r['status'] == 'PASS' for r in reqs)}/{len(reqs)} PASS); report "
          f"{d / 'reports' / 'local_report.md'}")
    if banner:
        print(banner)
    for r in reqs:
        print(f"  {r['status']:12s} {r['id']:4s} {r['clause'][:100]}")
    print("conclusions:")
    for c in cons:
        print(f"  {c['status']:12s} {c['conclusion']}")
    return 0 if verdict == "PASS" else 1


def fast_conclusions(reqs):
    """Six separate conclusions of a local experiment run, from its requirements (never inferred from each other)."""
    Q = {r["id"]: r for r in reqs}

    def w(ids, conclusion):
        return {"conclusion": conclusion, "status": R.worst(Q[i]["status"] for i in ids),
                "detail": "; ".join(f"{i} {Q[i]['status']}: {Q[i]['detail'][:160]}" for i in ids)}
    return [w(["T3", "T4"], "training completed (2-step smoke and 10-step run, every update)"),
            w(["T5", "T13"], "training correctness established (frozen weights unchanged; L2L equivalence suite: "
                             "not re-run in this mode, earlier evidence only if same code and GPU)"),
            w(["T10"], "performance improvement established (validation subset, exact sign test)"),
            w(["T6"], "memory compliance established (the local 4 GiB card; not the assignment's 8 GB cap)"),
            w(["X1"], "DFlash validity established"),
            {"conclusion": "full assignment qualification established", "status": "NOT_TESTED",
             "detail": "a local Qwen3-0.6B run never qualifies the Qwen3-4B assignment"}]


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
