"""Predicted vs measured memory and time for the Qwen3-4B plain-L2L configuration, and a forecast for the 150-update run.

    python scripts/budget_residuals.py [--out results/budget/residuals/qwen3-4b-final_150_v2]

Three kinds of number, never mixed (each value in the output names its kind and source file):
  ANALYTICAL   results/budget/predictions/qwen3-4b-final_8gb.json, written by scripts/budget.py on 2026-11-18 BEFORE
               any Qwen3-4B measurement (provenance.created, kind "theoretical prediction"). Unchanged here.
  MEASURED     the 60-update run and its profile on the RTX 3060 (2026-11-18): runs/qwen3_4b_final_plain/training/
               training.json (per-step timings, weight bytes), its memory guard samples (nvidia-smi per process,
               allocator), research/l2l/experiments/t11_profile_qwen3_4b_async.json (per-phase peaks, measured pinned
               H2D bandwidth). Residual = measured - analytical.
  CALIBRATED   the 150-update forecast. It uses the MEASURED numbers above (acceptance rate, step times, bytes), so it is
               NOT an independent pre-run prediction; its analytical counterpart (transfer-bound time from the
               analytical bytes at the assumed 12 GB/s and at the measured bandwidth) is given beside it.
DFlash: excluded from this experiment; the drafter rows of the analytical budget are reported as "not evaluated".
Writes <out>.json and <out>.md; refuses to overwrite.
"""
import argparse
import glob
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results/budget/predictions/qwen3-4b-final_8gb.json"
RUN = ROOT / "runs/qwen3_4b_final_plain/training"
PROFILE = ROOT / "research/l2l/experiments/t11_profile_qwen3_4b_async.json"
CAP = 8_000_000_000
GIB = 2**30


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "results/budget/residuals/qwen3-4b-final_150_v2")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--max-attempts", type=int, default=1200)
    ap.add_argument("--eval-every", type=int, default=10)
    a = ap.parse_args(argv)
    if Path(f"{a.out}.json").exists():
        raise SystemExit(f"{a.out}.json exists: never overwritten")
    pred = json.loads(PRED.read_text())
    tr = json.loads((RUN / "training.json").read_text())
    mem = json.loads(Path(sorted(glob.glob(str(RUN / "memory" / "*.json")))[0]).read_text())
    prof = json.loads(PROFILE.read_text())
    phases = {p["phase"]: p for p in pred["phases"]}
    transfers = {t["name"]: t for t in pred["transfers"]}
    bw_assumed = pred["config"]["hardware"]["pcie_gb_per_s"] * 1e9
    bw_meas = prof["h2d_bandwidth_bytes_per_s"]
    pp = {}
    for p in prof["profile"]["phases"]:  # several forward/backward windows: keep each phase's maxima, sum of seconds
        q = pp.setdefault(p["phase"], {"seconds": 0.0, "windows": 0})
        q["seconds"] += p["seconds"]
        q["windows"] += 1
        for k in ("peak_allocated", "peak_reserved", "process_used_peak", "device_used_peak", "cpu_rss_peak"):
            if p.get(k) is not None:
                q[k] = max(q.get(k) or 0, p[k])
    T = tr["timings"]
    med = lambda k: statistics.median(x[k] for x in T if x.get(k) is not None)
    steps, attempts, zero = tr["successful_steps"], tr["groups_generated"], len(tr["zero_signal"])
    zero_s = (tr["train_seconds"] - sum(x["step_s"] for x in T)) / zero  # includes 4 subset evals + checkpoint writes

    # --- memory: analytical vs measured -------------------------------------------------------------------------
    memory_rows = []
    pairs = [("initialization", "setup"), ("rollout", "rollout"), ("training", "forward"), ("training", "backward"),
             ("training", "optimizer"), ("training", "old_logp"), ("evaluation", "evaluation"), ("merge", "merge")]
    for ph, mp in pairs:
        g, m = phases[ph]["gpu_peak"], pp.get(mp, {})
        memory_rows.append({
            "analytical_phase": ph, "measured_phase": mp,
            "analytical_allocated": g["gpu_allocated"], "measured_allocated": m.get("peak_allocated"),
            "analytical_reserved": g["gpu_reserved"], "measured_reserved": m.get("peak_reserved"),
            "analytical_device_incl_context": g["gpu_device"], "measured_smi_process": m.get("process_used_peak"),
            "residual_smi_minus_device": None if m.get("process_used_peak") is None
            else m["process_used_peak"] - g["gpu_device"],
            "note": ("merge: analytical 0 (CPU merge); the profile merged inside a process that already held the "
                     "model on the GPU, so its GPU figures are that process's resident footprint, not the merge's"
                     if ph == "merge" else
                     "no nvidia-smi sample: the window is shorter than the sampling interval" if
                     m.get("process_used_peak") is None else "")})
    whole = {"source": "runs/qwen3_4b_final_plain/training/memory (60 updates, 1922 samples, evidence "
                       f"{mem.get('evidence')})", "smi_process_peak": mem["peak"]["process_used"],
             "allocator_reserved_peak": mem["peak"]["reserved"], "allocator_allocated_peak": mem["peak"]["allocated"],
             "cpu_rss_peak": mem["peak"]["VmRSS"], "analytical_max_device": max(p["gpu_peak"]["gpu_device"]
                                                                                for k, p in phases.items()
                                                                                if not k.startswith("dflash")),
             "cap_bytes": CAP, "headroom_vs_cap": CAP - mem["peak"]["process_used"]}
    whole["residual_smi_minus_analytical_max"] = whole["smi_process_peak"] - whole["analytical_max_device"]

    # --- time: analytical transfer bound vs measured --------------------------------------------------------------
    rollout_b = transfers["rollout prefill"]["bytes"] + transfers["rollout decode"]["bytes"]
    old_b = transfers["old log-probs forward"]["bytes"]
    upd_b = transfers["training forward (x1 iterations)"]["bytes"] + transfers["backward recomputation (x1, layers >= 34)"]["bytes"]
    time_rows = []
    for name, b, meas in (("rollout (4 sweeps)", rollout_b, med("rollout_s")), ("old log-probs (5 micro-batches)", old_b,
                                                                              med("prepare_s")),
                          ("update: forward + backward + AdamW", upd_b, med("update_s")),
                          ("GRPO step (successful update)", rollout_b + old_b + upd_b, med("step_s"))):
        time_rows.append({"phase": name, "analytical_bytes": b, "analytical_transfer_s_at_12GBps": b / bw_assumed,
                          "transfer_s_at_measured_bw": b / bw_meas, "measured_median_s": meas,
                          "residual_s_vs_analytical": meas - b / bw_assumed,
                          "measured_over_transfer_bound": meas / (b / bw_meas)})
    measured_bytes_step = statistics.median(x["weight_bytes_copied"] for x in T)

    # --- 150-update forecast (CALIBRATED) ------------------------------------------------------------------------
    p_acc = steps / attempts
    lo, hi = wilson(steps, attempts)
    evals = a.steps // a.eval_every + 1
    eval_b = transfers["evaluation (128 examples)"]["bytes"]
    score_b = math.ceil(128 / 8) * pred["residency"]["sweep"][0]["sweep_bytes"]  # label scores: 16 batches of 8
    eval_s = (eval_b + score_b) / bw_meas
    zero_rollout_s = med("rollout_s")  # a zero-signal attempt generates and is rejected before old log-probs

    def forecast(p):
        att = math.ceil(a.steps / p)
        return {"acceptance": p, "attempts": att, "within_max_attempts": att <= a.max_attempts,
                "seconds": a.steps * med("step_s") + (att - a.steps) * zero_rollout_s + evals * eval_s,
                "h2d_bytes": a.steps * measured_bytes_step + (att - a.steps) * rollout_b + evals * (eval_b + score_b)}
    fc = {"kind": "CALIBRATED (uses the 60-update run's measurements; not an independent pre-run prediction)",
          "acceptance_60_update": {"successful": steps, "attempts": attempts, "rate": p_acc, "wilson95": [lo, hi]},
          "point": forecast(p_acc), "pessimistic_wilson_low": forecast(lo), "at_max_attempts_rate": forecast(a.steps / a.max_attempts),
          "evaluations": evals, "eval_s_each_transfer_bound": eval_s,
          "eval_note": "evaluation time is transfer-bound analytical (greedy 128 prompts + 16 label-score batches at the "
                       "measured bandwidth); the 60-update run did not time its subset evaluations",
          "memory": {"expected_smi_process_peak": whole["smi_process_peak"],
                     "label_scores_logits_upper_bound_bytes": 8 * 102 * 151936 * 4,
                     "note": "per-step shapes are unchanged from the 60-update run (same B, G, micro-batch, tokens), so "
                             "the per-step peak is expected to repeat; the subset label scores add at most one fp32 "
                             "[8, T<=102, 151936] logits tensor during evaluations (upper bound above); the cap is "
                             "enforced, not assumed"}}
    analytical150 = {"kind": "ANALYTICAL (transfer-bound, analytical bytes; acceptance unknown before any run: the "
                             "attempt count is the configured bound)",
                     "per_update_s_at_12GBps": (rollout_b + old_b + upd_b) / bw_assumed,
                     "per_zero_attempt_s_at_12GBps": rollout_b / bw_assumed,
                     "worst_case_seconds_at_max_attempts": a.steps * (rollout_b + old_b + upd_b) / bw_assumed
                     + (a.max_attempts - a.steps) * rollout_b / bw_assumed + evals * (eval_b + score_b) / bw_assumed}
    out = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "sources": {"analytical": str(PRED.relative_to(ROOT)), "analytical_created": pred["provenance"]["created"],
                       "measured_run": str(RUN.relative_to(ROOT)), "measured_profile": str(PROFILE.relative_to(ROOT))},
           "bandwidth": {"assumed_bytes_per_s": bw_assumed, "measured_pinned_h2d_bytes_per_s": bw_meas},
           "memory_phases": memory_rows, "memory_whole_run": whole, "time_phases": time_rows,
           "bytes_per_update_measured": measured_bytes_step, "bytes_per_update_analytical": rollout_b + old_b + upd_b,
           "zero_signal_attempt_s_upper_bound": zero_s, "forecast_150": fc, "analytical_150": analytical150,
           "dflash": "excluded from this experiment (plain L2L rollouts); the analytical DFlash rows are not evaluated",
           "cpu_ram": {"analytical_init_peak": phases["initialization"]["cpu_peak_bytes"],
                       "measured_rss_peak_run": mem["peak"]["VmRSS"], "measured_rss_peak_profile_merge":
                       pp.get("merge", {}).get("cpu_rss_peak")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(json.dumps(out, indent=1) + "\n")
    g = lambda b: "—" if b is None else f"{b / GIB:.2f}"
    md = [f"# Qwen3-4B plain L2L: predicted vs measured, and the 150-update forecast\n",
          f"Generated by `scripts/budget_residuals.py` at {out['created_utc']}. Analytical = `{out['sources']['analytical']}` "
          f"(created {out['sources']['analytical_created']}, before any Qwen3-4B measurement). Measured = the 60-update "
          "run and the H11 profile (RTX 3060, 2026-11-18). Cap: 8,000,000,000 B = 7.45 GiB.\n",
          "## Memory (GiB)\n", "| analytical phase | measured window | alloc pred | alloc meas | reserved pred | reserved meas "
          "| device pred (incl. context) | nvidia-smi process meas | residual | note |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in memory_rows:
        md.append(f"| {r['analytical_phase']} | {r['measured_phase']} | {g(r['analytical_allocated'])} | {g(r['measured_allocated'])} "
                  f"| {g(r['analytical_reserved'])} | {g(r['measured_reserved'])} | {g(r['analytical_device_incl_context'])} "
                  f"| {g(r['measured_smi_process'])} | {g(r['residual_smi_minus_device'])} | {r['note']} |")
    md += ["", f"Whole 60-update run: nvidia-smi per-process peak {g(whole['smi_process_peak'])} GiB "
               f"({whole['smi_process_peak']:.0f} B) vs analytical maximum {g(whole['analytical_max_device'])} GiB "
               f"(residual {g(whole['residual_smi_minus_analytical_max'])} GiB); allocator reserved {g(whole['allocator_reserved_peak'])} GiB; "
               f"headroom to the cap {g(whole['headroom_vs_cap'])} GiB. Host RSS peak {g(whole['cpu_rss_peak'])} GiB "
               f"(analytical init {g(phases['initialization']['cpu_peak_bytes'])} GiB).\n",
           "## Time per GRPO step\n", f"Pinned H2D bandwidth: assumed {bw_assumed / 1e9:.1f} GB/s, measured "
           f"{bw_meas / 1e9:.2f} GB/s.\n", "| phase | analytical bytes (GB) | transfer bound @12 GB/s (s) | @measured (s) | "
           "measured median (s) | measured / transfer bound |", "|---|---|---|---|---|---|"]
    for r in time_rows:
        md.append(f"| {r['phase']} | {r['analytical_bytes'] / 1e9:.1f} | {r['analytical_transfer_s_at_12GBps']:.2f} | "
                  f"{r['transfer_s_at_measured_bw']:.2f} | {r['measured_median_s']:.2f} | {r['measured_over_transfer_bound']:.2f} |")
    md += ["", f"Measured weight bytes per successful update: {measured_bytes_step / 1e9:.1f} GB (analytical "
               f"{(rollout_b + old_b + upd_b) / 1e9:.1f} GB). A ratio near 1 means the step is transfer-bound: the GPU "
               "waits for PCIe, so copy/compute overlap can at best hide the (small) compute time.\n",
           "## 150-update forecast (CALIBRATED: uses the measurements above)\n",
           f"Acceptance in the 60-update run: {steps}/{attempts} = {p_acc:.3f} (Wilson 95% [{lo:.3f}, {hi:.3f}]).\n",
           "| scenario | acceptance | attempts | fits max_attempts | time (h) | H2D (TB) |", "|---|---|---|---|---|---|"]
    for k in ("point", "pessimistic_wilson_low", "at_max_attempts_rate"):
        f = fc[k]
        md.append(f"| {k} | {f['acceptance']:.3f} | {f['attempts']} | {f['within_max_attempts']} | {f['seconds'] / 3600:.2f} | "
                  f"{f['h2d_bytes'] / 1e12:.1f} |")
    md += ["", f"Analytical counterpart (no measurement): {analytical150['per_update_s_at_12GBps']:.1f} s per update, "
               f"{analytical150['per_zero_attempt_s_at_12GBps']:.1f} s per zero-signal attempt, worst case at max_attempts "
               f"{analytical150['worst_case_seconds_at_max_attempts'] / 3600:.2f} h.\n",
           f"Memory: {fc['memory']['note']}.\n", "DFlash: " + out["dflash"] + ".\n"]
    Path(f"{a.out}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
