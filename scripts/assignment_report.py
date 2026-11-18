"""Aggregate one assignment run (assignment_results/<run-id>/) into measurements, comparisons, requirement statuses and
reports. Called by `python scripts/assignment.py report --run-dir D` (always, also after a failure or an interrupt).

Reads only this run's state.json (what each item did and how it was judged), the raw telemetry of its items, and the
outputs of items that produced a result in this run (status PASS or FAIL). Evidence from outside the run is never
used to fill a gap. Every number is labelled measured, theoretical or unavailable. A missing measurement stays null
(MISSING in the reports), never 0. No requirement is PASS without the measurement it names. The verdict is SHIP only if
every essential requirement is PASS.
"""
import csv
import hashlib
import json
import os
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo.evidence import EXPERIMENTS, read_result  # noqa: E402

EXP = Path(os.environ.get("ASSIGNMENT_EXPERIMENTS_ROOT", EXPERIMENTS))
RUNS = Path(os.environ.get("ASSIGNMENT_RUNS_ROOT", ROOT / "runs"))
MIB, GIB = 2**20, 2**30
USABLE = ("PASS", "FAIL")  # an item whose tool produced a judged result in this run
SEED = 0
STATUS_ORDER = {"FAIL": 5, "INCONCLUSIVE": 4, "BLOCKED": 3, "NOT_TESTED": 2, "PASS": 1}


# --- small helpers -----------------------------------------------------------------------------------------------

def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError, TypeError):
        return None


def worst(statuses):
    s = [x for x in statuses if x]
    return max(s, key=lambda x: STATUS_ORDER[x]) if s else "NOT_TESTED"


def fmt(v, unit="", nd=3):
    if v is None:
        return "MISSING"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{nd}f}{unit}" if isinstance(v, float) else f"{v}{unit}"
    return str(v)


def gib(b):
    return None if b is None else b / GIB


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |" for r in rows]
    return "\n".join(out)


def median(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def bootstrap_ci(values, stat=statistics.median, n=5000, seed=SEED):
    v = [x for x in values if x is not None]
    if len(v) < 2:
        return None
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(v) for _ in v]) for _ in range(n))
    return [boots[int(0.025 * n)], boots[int(0.975 * n) - 1]]


def ratio_ci(a, b, n=5000, seed=SEED):
    """Median(b) / median(a) with a bootstrap 95% CI (independent resampling of both arms)."""
    a, b = [x for x in a if x is not None], [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return (statistics.median(b) / statistics.median(a) if a and b else None), None
    rng = random.Random(seed)
    boots = sorted(statistics.median([rng.choice(b) for _ in b]) / statistics.median([rng.choice(a) for _ in a])
                   for _ in range(n))
    return statistics.median(b) / statistics.median(a), [boots[int(0.025 * n)], boots[int(0.975 * n) - 1]]


def permutation_test(x, y, n=10000, seed=SEED):
    """Two-sample test of equal categorical distributions: chi-square statistic of the 2 x K table, p-value by
    permutation of the pooled labels (no asymptotic assumption)."""
    if not x or not y:
        return None

    def chi2(a, b):
        cats = set(a) | set(b)
        na, nb, tot = len(a), len(b), len(a) + len(b)
        s = 0.0
        for c in cats:
            ca, cb = a.count(c), b.count(c)
            for obs, n_ in ((ca, na), (cb, nb)):
                exp = n_ * (ca + cb) / tot
                s += (obs - exp) ** 2 / exp if exp else 0.0
        return s
    obs = chi2(x, y)
    pool, rng, hits = list(x) + list(y), random.Random(seed), 0
    for _ in range(n):
        rng.shuffle(pool)
        hits += chi2(pool[:len(x)], pool[len(x):]) >= obs - 1e-12
    return {"chi2": obs, "p_value": (hits + 1) / (n + 1), "permutations": n, "n_plain": len(x), "n_dflash": len(y),
            "categories": len(set(x) | set(y))}


# --- the run ------------------------------------------------------------------------------------------------------

class Run:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self.state = read_json(self.dir / "state.json") or {"stages": {}}
        self.manifest = read_json(self.dir / "manifest.json") or {}
        self.cap = float((self.manifest.get("gpu_memory_cap") or {}).get("bytes") or 8e9)
        self.cap_gib_ref = float((self.manifest.get("gpu_memory_cap") or {}).get("binary_reference_bytes") or 8 * GIB)

    def attempt(self, stage):
        a = self.state["stages"].get(stage, {}).get("attempts", [])
        return a[-1] if a else None

    def items(self, stage):
        a = self.attempt(stage)
        return a.get("items", []) if a else []

    def item(self, stage, item_id):
        return next((i for i in self.items(stage) if i["id"] == item_id), None)

    def facts(self):
        return (self.attempt("A") or {}).get("facts", {})

    def all_items(self):
        for s in "ABCDEFGH":
            for i in self.items(s):
                yield s, i

    def output(self, stage, item_id, suffix=".json"):
        """The JSON output of an item that produced a judged result in this run (else None)."""
        it = self.item(stage, item_id)
        if not it or it["status"] not in USABLE:
            return None
        name = it.get("evidence_name") or ENTRY_NAMES.get(item_id)
        if not name:
            return None
        r = read_json(EXP / f"{name}{suffix}")
        return r if r is not None else read_result(EXP / name)

    def run_log(self, name, stage, item_id):
        it = self.item(stage, item_id)
        if not it or it["status"] not in USABLE:
            return None
        return read_json(RUNS / name / "training" / "training.json")


ENTRY_NAMES = {"H11.profile": "t11_profile_qwen3_4b_async", "H10.dflash_bench": "t12_dflash_bench",
               "H10.dflash_drift": "t12_dflash_drift_final", "H12.final_eval": "t13_final_eval_qwen3_4b_final",
               "H08.resume_tiny_cpu": "t09_resume_tiny_cpu", "H08.resume_qwen3_0.6b": "t09_resume_qwen_stream"}


def status_of(it):
    return it["status"] if it else "NOT_TESTED"


def missing(it):
    """Status of a requirement whose measurement file is absent: the item's own non-PASS status, or INCONCLUSIVE when
    the item says PASS (a PASS is never inherited without the measurement it names)."""
    s = status_of(it)
    return "INCONCLUSIVE" if s == "PASS" else s


def why(it, item_id):
    if not it:
        return f"{item_id}: not run in this assignment run"
    return f"{item_id}: {it['status']} - {str(it.get('detail'))[:240]}"


# --- measurements -------------------------------------------------------------------------------------------------

def memory_rows(run):
    """memory.csv: every memory figure with its kind (measured / theoretical / unavailable) and its source."""
    rows = []
    for stage, it in run.all_items():
        t = it.get("telemetry")
        if not t:
            continue
        src = t["raw_files"]
        rows.append([stage, it["id"], "whole item", "nvidia-smi per-process (process group, CUDA context incl.)",
                     "measured" if t["process_peak_mib"] is not None else "unavailable",
                     None if t["process_peak_mib"] is None else int(t["process_peak_mib"] * MIB), src["apps"],
                     f"{t['attributed_numeric_samples']} samples; {t['memory_reason']}"])
        rows.append([stage, it["id"], "whole item", "nvidia-smi device-wide memory.used",
                     "measured" if t["device_peak_used_mib"] is not None else "unavailable",
                     None if t["device_peak_used_mib"] is None else int(t["device_peak_used_mib"] * MIB), src["gpu"],
                     "all processes on the GPU"])
    prof = run.output("D", "H11.profile")
    if prof:
        f = EXP / "t11_profile_qwen3_4b_async.json"
        for p in prof.get("profile", {}).get("phases", []):
            for q, k in (("PyTorch peak allocated", "peak_allocated"), ("PyTorch peak reserved", "peak_reserved"),
                         ("cudaMemGetInfo device used (peak of samples)", "device_used_peak"),
                         ("nvidia-smi per-process (profiler, own PID)", "process_used_peak"),
                         ("CPU RSS peak", "cpu_rss_peak")):
                v = p.get(k)
                rows.append(["D", "H11.profile", p["phase"], q, "measured" if v is not None else "unavailable", v, str(f),
                             f"{p.get('samples')} samples in the phase"])
        rows.append(["D", "H11.profile", "setup", "pinned host memory (weight store)",
                     "measured" if prof.get("pinned_bytes") is not None else "unavailable", prof.get("pinned_bytes"), str(f),
                     "sum of pinned slab bytes"])
        for p in (prof.get("prediction") or {}).get("rows", []):
            for q, k in (("predicted allocated", "predicted_allocated"), ("predicted reserved", "predicted_reserved"),
                         ("predicted device", "predicted_device"), ("predicted CPU", "predicted_cpu")):
                rows.append(["D", "H11.profile", p["phase"], q, "theoretical" if p.get(k) is not None else "unavailable",
                             p.get(k), prof["prediction"]["file"], f"budget phase {p['predicted_phase']}"])
    for name, stage, iid in (("qwen3_4b_smoke", "D", "H11.smoke"), ("qwen3_4b_smoke_dflash", "E", "H10.smoke_dflash"),
                             ("qwen3_4b_final", "F", "H12.train_final"), ("qwen3_4b_final_lr0", "F", "H12.train_final")):
        log = run.run_log(name, stage, iid)
        if not log:
            continue
        pk = max((r.get("peak_reserved_gib") or 0 for r in log.get("training", [])), default=None)
        pa = max((r.get("peak_allocated_gib") or 0 for r in log.get("training", [])), default=None)
        rss = max((t.get("cpu_rss_bytes") or 0 for t in log.get("timings", [])), default=None)
        f = str(RUNS / name / "training" / "training.json")
        rows.append([stage, name, "training steps", "PyTorch peak reserved (max over steps)", "measured" if pk else "unavailable",
                     None if not pk else int(pk * GIB), f, "allocator only: excludes the CUDA context"])
        rows.append([stage, name, "training steps", "PyTorch peak allocated (max over steps)", "measured" if pa else "unavailable",
                     None if not pa else int(pa * GIB), f, "allocator only"])
        rows.append([stage, name, "training steps", "CPU RSS (max over steps)", "measured" if rss else "unavailable", rss or None,
                     f, ""])
    pred = predictions(run)
    if pred:
        for ph in pred["data"].get("phases", []):
            g = ph.get("gpu_peak", {})
            for q in ("gpu_allocated", "gpu_reserved", "gpu_device", "cpu"):
                rows.append(["A", "budget", ph["phase"], f"predicted {q}", "theoretical", g.get(q), pred["file"],
                             "; ".join(ph.get("notes", []))[:200]])
            for m in ph.get("moments", []):
                for comp in m.get("gpu", []):
                    rows.append(["A", "budget", ph["phase"], f"component: {comp['name']}", "theoretical", comp.get("bytes"),
                                 pred["file"], f"{comp.get('kind')}; measured counterpart: unavailable (not instrumented "
                                               "per component)"])
    return rows


FINAL_PREDICTION = "qwen3-4b-final_8gb.json"  # configs/budget/qwen3-4b_final.yaml: the configuration that runs


def predictions(run):
    """The prediction of the final configuration only. The historical Step-0 prediction (qwen3-4b_8gb.json) describes
    another configuration (micro-batch 6, GPU merge) and is never used for a comparison."""
    for sub in ("calculator", "committed"):
        d = read_json(run.dir / "predictions" / sub / FINAL_PREDICTION)
        if d:
            return {"file": str(run.dir / "predictions" / sub / FINAL_PREDICTION), "data": d, "source": sub}
    return None


def timing_rows(run):
    rows = []
    for name, stage, iid in (("qwen3_4b_smoke", "D", "H11.smoke"), ("qwen3_4b_smoke_dflash", "E", "H10.smoke_dflash"),
                             ("qwen3_4b_final", "F", "H12.train_final"), ("qwen3_4b_final_lr0", "F", "H12.train_final")):
        log = run.run_log(name, stage, iid)
        for t in (log or {}).get("timings", []):
            rows.append([name, t.get("step"), t.get("rollout_s"), t.get("prepare_s"), t.get("update_s"), t.get("step_s"),
                         t.get("rollout_tokens"), t.get("weight_bytes_copied"), t.get("backend"), "measured"])
    prof = run.output("D", "H11.profile")
    for p in (prof or {}).get("profile", {}).get("phases", []):
        rows.append(["profile:" + p["phase"], None, None, None, None, p["seconds"], None, None, "l2l_async", "measured"])
    return rows


def throughput_rows(run):
    rows = []
    prof = run.output("D", "H11.profile")
    if prof:
        bw = prof.get("h2d_bandwidth_bytes_per_s")
        rows.append(["measured pinned H2D bandwidth", None, bw, None, None, None, "measured" if bw else "unavailable"])
        for ph, t in (prof.get("transfers") or {}).items():
            rows.append([ph, t.get("bytes"), bw, t.get("predicted_transfer_s"), t.get("phase_wall_s"),
                         t.get("transfer_share_of_wall"), "measured bytes and wall time; transfer time = bytes / measured bandwidth"])
    pred = predictions(run)
    if pred:
        for t in pred["data"].get("transfers", []):
            rows.append([f"predicted: {t['name']}", t.get("bytes"), None, t.get("transfer_s"), None, None,
                         "theoretical (budget calculator, assumed bandwidth)"])
    return rows


def lab_summary(name):
    r = read_result(EXP / name) if name else None
    if not r:
        return None
    comps = r.get("comparisons") or []
    by_q = {}
    for c in comps:
        q = by_q.setdefault(c.get("quantity"), {"n": 0, "max_abs": 0.0, "bound_max": 0.0, "fails": 0, "dtypes": set(),
                                                "shape_example": c.get("shape")})
        q["n"] += 1
        q["max_abs"] = max(q["max_abs"], c.get("max_abs") or 0.0)
        q["bound_max"] = max(q["bound_max"], c.get("bound") or 0.0)
        q["fails"] += not c.get("pass")
        q["dtypes"].add(c.get("dtype"))
    prov = r.get("provenance") or {}
    return {"name": name, "case": r.get("case"), "status": r.get("status"), "execution_status": r.get("execution_status"),
            "hypothesis_result": r.get("hypothesis_result"), "fault_injection": r.get("fault_injection"),
            "device": r.get("device"), "gpu": prov.get("gpu"), "packages": prov.get("packages"),
            "commit": prov.get("git_commit"), "comparisons": len(comps),
            "quantities": {k: {**v, "dtypes": sorted(map(str, v["dtypes"]))} for k, v in by_q.items()},
            "conclusion": r.get("conclusion")}


def correctness(run):
    out = []
    for stage in "BC":
        for it in run.items(stage):
            rec = {"stage": stage, "item": it["id"], "kind": it.get("kind"), "status": it["status"],
                   "detail": it.get("detail"), "evidence_dir": it.get("evidence_dir")}
            if it.get("kind") == "pytest":
                rec["junit"] = it.get("junit")
            if it.get("kind") == "entry" and it["status"] in USABLE:
                rec["lab"] = lab_summary(it.get("evidence_name"))
            out.append(rec)
    return out


def dflash(run):
    d = {"benchmarks": {}, "end_to_end": None, "distribution_checks": []}
    for iid, stage in (("H10.dflash_bench", "E"), ("H10.dflash_drift", "G")):
        b = run.output(stage, iid)
        if not b:
            continue
        pols = []
        for p in b.get("policies", []):
            row = {"policy": p["policy"], "runs": []}
            for r in p.get("runs", []):
                acc = r.get("sample_accepted_per_pass_trials") or [r.get("sample_accepted_per_pass")]
                acc_flat = [x for v in acc for x in (v if isinstance(v, list) else [v]) if isinstance(x, (int, float))]
                ratio_g, ci_g = ratio_ci(r.get("greedy_plain_s_trials") or [r.get("greedy_plain_s")],
                                         r.get("greedy_dflash_s_trials") or [r.get("greedy_dflash_s")])
                ratio_s, ci_s = ratio_ci(r.get("sample_plain_s_trials") or [r.get("sample_plain_s")],
                                         r.get("sample_dflash_s_trials") or [r.get("sample_dflash_s")])
                plain = [x for v in r.get("sample_plain_responses", []) for x in v]
                dfl = [x for v in r.get("sample_dflash_responses", []) for x in v]
                test = permutation_test(plain, dfl)
                if test:
                    d["distribution_checks"].append({"benchmark": iid, "policy": p["policy"], "max_new": r["max_new"], **test})
                row["runs"].append({
                    "max_new": r["max_new"], "greedy_equal": r.get("greedy_equal"),
                    "greedy_passes_per_token": r.get("greedy_passes_per_token"),
                    "sample_passes_per_token": r.get("sample_passes_per_token"),
                    "accepted_per_pass_mean": statistics.mean(acc_flat) if acc_flat else None,
                    "accepted_per_pass_ci95": bootstrap_ci(acc_flat, statistics.mean),
                    "accepted_per_pass_values": len(acc_flat),
                    "greedy_time_ratio_dflash_over_plain": ratio_g, "greedy_time_ratio_ci95": ci_g,
                    "sample_time_ratio_dflash_over_plain": ratio_s, "sample_time_ratio_ci95": ci_s,
                    "greedy_plain_s": r.get("greedy_plain_s"), "greedy_dflash_s": r.get("greedy_dflash_s"),
                    "sample_plain_s": r.get("sample_plain_s"), "sample_dflash_s": r.get("sample_dflash_s"),
                    "tokens_per_s_plain": (b["B"] * b["G"] * r["max_new"] / r["sample_plain_s"]) if r.get("sample_plain_s") else None,
                    "tokens_per_s_note": "upper bound: B*G*max_new tokens per call (rows that stop at EOS generate fewer)",
                    "peak_allocated_dflash": r.get("sample_dflash_peak"), "peak_allocated_plain": r.get("sample_plain_peak"),
                    "trials": len(r.get("greedy_plain_s_trials") or [1])})
            gr, gci = ratio_ci(p.get("grpo_step_plain_s_trials") or [p.get("grpo_step_plain_s")],
                               p.get("grpo_step_dflash_s_trials") or [p.get("grpo_step_dflash_s")])
            row.update(grpo_step_plain_s=p.get("grpo_step_plain_s"), grpo_step_dflash_s=p.get("grpo_step_dflash_s"),
                       grpo_step_ratio=gr, grpo_step_ratio_ci95=gci)
            pols.append(row)
        d["benchmarks"][iid] = {"file": str(EXP / f"{ENTRY_NAMES[iid]}.json"), "drafter": b.get("drafter"),
                                "drafter_bytes": b.get("drafter_bytes"), "all_greedy_equal": b.get("all_greedy_equal"),
                                "budget_verdict_8GiB": b.get("budget_verdict"), "repeats": b.get("repeats", 1),
                                "policies": pols}
    plain, dfl = run.run_log("qwen3_4b_smoke", "D", "H11.smoke"), run.run_log("qwen3_4b_smoke_dflash", "E", "H10.smoke_dflash")
    if plain and dfl:
        a = [t["step_s"] for t in plain.get("timings", [])[1:]]  # step 1 includes warm-up
        b_ = [t["step_s"] for t in dfl.get("timings", [])[1:]]
        ra, rci = ratio_ci(a, b_)
        d["end_to_end"] = {"plain_step_s": a, "dflash_step_s": b_, "ratio_dflash_over_plain": ra, "ratio_ci95": rci,
                           "rollout_s_plain": [t.get("rollout_s") for t in plain.get("timings", [])[1:]],
                           "rollout_s_dflash": [t.get("rollout_s") for t in dfl.get("timings", [])[1:]],
                           "note": "4-step guarded smoke runs, same data/seed/batch; step 1 excluded as warm-up; "
                                   "few steps: wide intervals"}
    return d


def learning(run):
    ev = run.output("G", "H12.final_eval")
    log = run.run_log("qwen3_4b_final", "F", "H12.train_final")
    ctl = run.run_log("qwen3_4b_final_lr0", "F", "H12.train_final")
    out = {"evaluation": None, "training": None, "control_training": None}
    if ev:
        sc = ev.get("label_scores", {})
        diffs = [abs(a - b) for a, b in zip(sc.get("trained", []), sc.get("merged", []))] if sc.get("merged") else []
        out["evaluation"] = {
            "file": str(EXP / "t13_final_eval_qwen3_4b_final.json"), "n_test": ev.get("n_test"), "split": ev.get("split"),
            "checkpoint": ev.get("checkpoint"), "models": ev.get("models"), "paired": ev.get("paired"),
            "paired_auc": ev.get("paired_auc"), "data_separation": ev.get("data_separation"), "merge": ev.get("merge"),
            "merged_vs_trained_prediction_agreement": ev.get("merged_vs_trained_prediction_agreement"),
            "merged_vs_trained_label_score_abs_diff": {"max": max(diffs) if diffs else None,
                                                       "mean": statistics.mean(diffs) if diffs else None, "n": len(diffs)},
            "lora_change": ev.get("lora_change"), "control_log": ev.get("control_log"),
            "test_set_evaluation_attempts": len(run.state.get("test_set_evaluations", []))}
    for key, lg, name in (("training", log, "qwen3_4b_final"), ("control_training", ctl, "qwen3_4b_final_lr0")):
        if not lg:
            continue
        rows = lg.get("training", [])
        rewards = [r.get("mean_reward") for r in rows]
        attempts = sorted([r["attempt"] for r in rows] + [z["attempt"] for z in lg.get("zero_signal", [])])
        out[key] = {
            "run": name, "successful_updates": lg.get("successful_steps"), "rollout_batches": lg.get("groups_generated"),
            "zero_signal_batches": lg.get("zero_signal_groups"), "batches_by_mixed_groups": lg.get("batches_by_mixed_groups"),
            "reward_first10_mean": statistics.mean(rewards[:10]) if rewards[:10] else None,
            "reward_last10_mean": statistics.mean(rewards[-10:]) if rewards[-10:] else None,
            "grad_norm_median": median([r.get("grad_norm") for r in rows]),
            "update_norm_median": median([r.get("update_norm") for r in rows]),
            "clip_frac_max": max((r.get("clip_frac") or 0 for r in rows), default=None),
            "change_from_init_final": rows[-1].get("change_from_init") if rows else None,
            "frozen_changed": lg.get("frozen_changed"), "params_equal_init": lg.get("params_equal_init"),
            "control": lg.get("control"), "resumed": lg.get("resumed"), "final_checkpoint": lg.get("final_checkpoint"),
            "attempts_contiguous_unique": attempts == list(range(attempts[0], attempts[0] + len(attempts))) if attempts else None,
            "train_seconds": lg.get("train_seconds"), "config_sha256": lg.get("config_sha256"),
            "adapter_sha256": adapter_hash(RUNS / name / "checkpoints"),
        }
    return out


def adapter_hash(ckpt_dir):
    if not Path(ckpt_dir).exists():
        return None
    steps = sorted((p for p in Path(ckpt_dir).glob("step*") if p.name[4:].isdigit()), key=lambda p: int(p.name[4:]))
    if not steps:
        return None
    f = steps[-1] / "adapter_model.safetensors"
    return {"checkpoint": steps[-1].name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} if f.exists() else None


# --- requirements -------------------------------------------------------------------------------------------------

def requirements(run, m):
    R = []
    f = run.facts()
    cap_mib = run.cap / MIB

    def add(rid, clause, essential, status, detail, evidence=(), measured=None):
        assert status in STATUS_ORDER, status
        R.append({"id": rid, "clause": clause, "essential": essential, "status": status, "detail": detail,
                  "evidence": list(evidence), "measured": measured})

    def items_status(pairs, need_usable=True):
        its = [(s, i, run.item(s, i)) for s, i in pairs]
        return worst(status_of(it) for _, _, it in its), "; ".join(why(it, i) for _, i, it in its), \
            [it.get("raw_dir") or it.get("evidence_dir") for _, _, it in its if it]

    # T1
    it = run.item("A", "assignment_config")
    add("T1", "Qwen3-4B non-thinking, no SFT, GRPO + LoRA, verifiable reward, 1-2k prompts", True, status_of(it),
        (it or {}).get("detail", "stage A not run") + ("; " + "; ".join(it.get("notes", [])) if it else ""),
        ["resolved_configs/"])

    # memory
    def mem(rid, clause, pairs, phases=()):
        its = [(s, i, run.item(s, i)) for s, i in pairs]
        ran = [(s, i, it) for s, i, it in its if it and it["status"] not in ("BLOCKED", "NOT_TESTED")]
        prof = run.output("D", "H11.profile")
        phase_vals = []
        for p in (prof or {}).get("profile", {}).get("phases", []):
            if p["phase"] in phases:
                phase_vals.append((p["phase"], p.get("process_used_peak")))
        if not ran:
            st = worst(status_of(it) for _, _, it in its) if any(its_[2] for its_ in its) else "NOT_TESTED"
            add(rid, clause, True, "BLOCKED" if st == "BLOCKED" else "NOT_TESTED",
                "no item of this phase ran: " + "; ".join(why(it, i) for _, i, it in its))
            return
        sts = [it.get("memory_status") or "INCONCLUSIVE" for _, _, it in ran]
        st = worst(sts)
        if any(v is not None and v > run.cap for _, v in phase_vals):
            st = "FAIL"
        peaks = {i: (it.get("telemetry") or {}).get("process_peak_mib") for _, i, it in ran}
        detail = (f"cap {run.cap:.0f} B = {cap_mib:.0f} MiB (8 GB, decimal; 8 GiB = 8192 MiB shown for reference). "
                  + "; ".join(f"{i}: per-process peak {fmt(v, ' MiB', 0)} [{it.get('memory_status')}]"
                              for (_, i, it), v in zip(ran, peaks.values()))
                  + ("; profiler phases (own PID, 250 ms): " + ", ".join(f"{p} {fmt(None if v is None else v / MIB, ' MiB', 0)}"
                                                                          for p, v in phase_vals) if phase_vals else "")
                  + ". Sampling every 200 ms may miss shorter spikes.")
        not_run = [i for s, i, it in its if not it or it["status"] in ("BLOCKED", "NOT_TESTED")]
        if not_run and st == "PASS":
            st, detail = "INCONCLUSIVE", detail + f" Not run: {not_run}."
        add(rid, clause, True, st, detail, [(it.get("telemetry") or {}).get("raw_files", {}).get("apps") for _, _, it in ran],
            {"process_peak_mib": peaks, "profile_phase_process_peak_bytes": dict(phase_vals)})

    train_pairs = [("D", "H11.smoke"), ("F", "H12.train_final")] + \
        [("F", i["id"]) for i in run.items("F") if i["id"].startswith("resume:")]
    mem("M1", "training within 8 GB (nvidia-smi, CUDA context included)", train_pairs,
        ("old_logp", "forward", "backward", "optimizer"))
    mem("M2", "rollout generation within 8 GB (plain and DFlash)",
        [("D", "H11.profile"), ("E", "H10.dflash_bench"), ("E", "H10.smoke_dflash")], ("rollout",))
    mem("M3", "streamed merge within 8 GB", [("D", "H11.profile"), ("G", "H12.final_eval")], ("merge",))
    mem("M4", "evaluation of the merged model within 8 GB", [("G", "H12.final_eval")], ("evaluation",))

    # budget
    pit = run.item("A", "budget_predictions")
    prov = read_json(run.dir / "predictions" / "provenance.json") or {}
    first_gpu = min((run.attempt(s) or {}).get("started_utc", "9999") for s in "CDEFG")
    before = bool(prov.get("created_utc")) and prov["created_utc"] <= first_gpu
    add("P1", "memory and time budget before running anything", True,
        "PASS" if pit and pit["status"] == "PASS" and before else status_of(pit) if pit else "NOT_TESTED",
        f"{why(pit, 'budget_predictions')}; written {prov.get('created_utc')} (first GPU stage {first_gpu}); read-only "
        "directory predictions/", ["predictions/"])
    prof = run.output("D", "H11.profile")
    rows = (prof or {}).get("prediction", {}).get("rows", []) if prof else []
    if not prof:
        add("P2", "predicted vs measured peak (allocated, reserved, nvidia-smi), gaps explained", False,
            missing(run.item("D", "H11.profile")), why(run.item("D", "H11.profile"), "H11.profile") + " (no profile result)")
    else:
        have_proc = any(r.get("observed_process_peak") is not None for r in rows)
        add("P2", "predicted vs measured peak (allocated, reserved, nvidia-smi), gaps explained", False,
            "PASS" if rows and have_proc else "INCONCLUSIVE",
            f"{len(rows)} phases compared (comparisons/memory and reports/memory_analysis.md)"
            + ("" if have_proc else "; no per-process nvidia-smi reading in the profile"), ["measurements/memory.csv"])
    tr = (prof or {}).get("transfers") or {}
    add("P3", "step time from PCIe bandwidth vs measured", False,
        "PASS" if tr and prof.get("h2d_bandwidth_bytes_per_s") else (missing(run.item("D", "H11.profile")) if not prof
                                                                      else "INCONCLUSIVE"),
        f"measured bandwidth {fmt(gib(prof.get('h2d_bandwidth_bytes_per_s')) if prof else None, ' GiB/s')}; "
        f"{len(tr)} phases with bytes copied and wall time" if prof else why(run.item("D", "H11.profile"), "H11.profile"),
        ["measurements/throughput.csv"])
    pr = predictions(run)
    add("P4", "resident vs streamed layer split decided and justified", False,
        "PASS" if pr else "NOT_TESTED",
        f"configured: every decoder layer streamed (resident_layers 0); calculator recommendation: "
        f"{(pr or {}).get('data', {}).get('residency', {}).get('recommendation')}" if pr else "no prediction recorded",
        ["predictions/"])

    # L2L
    def lab(rid, clause, essential, pairs):
        st, d, ev = items_status(pairs)
        add(rid, clause, essential, st, d, ev)
    lab("L2", "L2L logits equal the resident reference (sync and async, this GPU, bf16 + fp32)", True,
        [("B", "H01.lab_sync"), ("B", "H01.lab_async_cuda"), ("B", "H01.fault_wrong_layer"), ("B", "H01.fault_unsafe_reuse")])
    lab("L3", "L2L LoRA gradients equal the resident reference", True,
        [("B", "H02.lab_sync_cuda"), ("B", "H02.lab_async_cuda"), ("B", "H02.fault_detach_boundary"),
         ("B", "H02.fault_unsafe_reuse")])
    lab("L4", "L2L rollout generation and evaluation equal resident", True,
        [("B", "H04.lab_sync_cuda"), ("B", "H04.lab_async_cuda"), ("B", "H04.fault_stale_position")])
    lab("L5", "complete GRPO optimizer step equal (tiny models; real Qwen3-0.6B runs bitwise)", True,
        [("B", "H05.lab_sync_cuda"), ("B", "H05.lab_async_cuda"), ("B", "H05.fault_advantage"), ("C", "H05.real_0.6b_bf16")])
    tl = [r for r in m["timing"] if r[0] in ("qwen3_4b_smoke", "qwen3_4b_final") and r[2] is not None]
    if tl:
        ro, pp, up = median([r[2] for r in tl]), median([r[3] for r in tl]), median([r[4] for r in tl])
        tot = sum(x for x in (ro, pp, up) if x)
        add("L6", "GRPO step split between generation and training, explained", False, "PASS",
            f"medians over {len(tl)} steps: rollout {fmt(ro, ' s', 1)} ({ro / tot:.0%}), old log-probs {fmt(pp, ' s', 1)}, "
            f"update {fmt(up, ' s', 1)}; each decode step re-streams all 36 layers, so rollout cost ~ (new tokens) x "
            "weight transfer", ["measurements/timing.csv"], {"rollout_s": ro, "prepare_s": pp, "update_s": up})
    else:
        add("L6", "GRPO step split between generation and training, explained", False, "NOT_TESTED",
            "no Qwen3-4B training timings in this run")
    lab("L7", "async prefetch overlap and bounded GPU weight residency (performance)", False,
        [("C", "H06.prefetch_timing"), ("B", "H07.depth_residency_execution")])

    # streaming
    st, d, ev = items_status([("B", "H08.resume_tiny_cpu"), ("C", "H08.resume_qwen3_0.6b")])
    lt = m["learning"].get("training") or {}
    if lt.get("resumed"):
        ok = lt.get("attempts_contiguous_unique")
        d += f"; final run resumed {len(lt['resumed'])} time(s): attempts contiguous and unique {ok}"
        if ok is False:
            st = "FAIL"
    add("S1", "streaming from disk with deterministic resume (kill, resume, no duplicates, no gaps)", True, st, d, ev)
    st, d, ev = items_status([("B", "H09.merge_tiny_cpu"), ("C", "H09.merge_qwen3_0.6b")])
    e = m["learning"].get("evaluation")
    if e:
        hu = (e.get("merge") or {}).get("all_within_half_ulp")
        d += f"; Qwen3-4B merge: every merged tensor within half an ulp: {hu}"
        if hu is False:
            st = "FAIL"
    add("S2", "merge tensor by tensor, never the full model on the GPU; peak shown", True, st, d, ev)

    # DFlash
    final_cfg = read_json(run.dir / "resolved_configs" / "hashes.json")
    final_dflash = None
    p = run.dir / "resolved_configs" / "configs__experiments__qwen3_4b_final.yaml"
    if p.exists():
        import yaml
        doc = yaml.safe_load("\n".join(l for l in p.read_text().splitlines() if not l.startswith("#"))) or {}
        final_dflash = bool((doc.get("dflash") or {}).get("enabled"))
    add("D0", "the final GRPO run uses DFlash-accelerated rollouts", False,
        "NOT_TESTED" if final_dflash is None else ("PASS" if final_dflash else "FAIL"),
        "final config not resolved" if final_dflash is None else
        ("dflash.enabled in the final config" if final_dflash else
         "the final run's config has dflash.enabled false: DFlash is measured end to end only in the 4-step "
         "qwen3_4b_smoke_dflash run and in dflash_bench.py, not in the shipped training run"),
        ["resolved_configs/configs__experiments__qwen3_4b_final.yaml"] if final_cfg else [])
    dfl = m["dflash"]
    bench = dfl["benchmarks"].get("H10.dflash_bench")
    bit = run.item("E", "H10.dflash_bench")
    add("D1", "drafter z-lab/Qwen3-4B-DFlash-b16 resident, conditioned on target hidden features inside the L2L loop",
        False, "PASS" if bench and bench.get("drafter") == "z-lab/Qwen3-4B-DFlash-b16" and bench.get("drafter_bytes") else
        ("FAIL" if bench else missing(bit)),
        (f"drafter {bench.get('drafter')} loaded on the GPU ({fmt(gib(bench.get('drafter_bytes')), ' GiB')}); the hidden-"
         "feature conditioning is the code path of qwen_grpo/execution/dflash.py (five target layers' features captured "
         "in the streamed layer loop), tested exactly on tiny models") if bench else why(bit, "H10.dflash_bench"),
        [bench["file"]] if bench else [])
    benches = [b for b in dfl["benchmarks"].values()]
    if benches:
        eq = all(b.get("all_greedy_equal") is True for b in benches)
        add("D2", "verifier = current LoRA policy; greedy output token-identical to plain decoding", False,
            "PASS" if eq else "FAIL", "; ".join(f"{k}: all greedy equal {b.get('all_greedy_equal')}"
                                                 for k, b in dfl["benchmarks"].items()), [b["file"] for b in benches])
    else:
        add("D2", "verifier = current LoRA policy; greedy output token-identical to plain decoding", False,
            missing(bit), why(bit, "H10.dflash_bench") + " (no benchmark result)")
    tiny = run.item("B", "unit_tests_cuda")
    checks = dfl["distribution_checks"]
    rej = [c for c in checks if c["p_value"] < 1e-3]
    if status_of(tiny) == "FAIL" or rej:
        st = "FAIL"
    elif status_of(tiny) == "PASS" and checks:
        st = "PASS"
    else:
        st = "INCONCLUSIVE" if status_of(tiny) == "PASS" or checks else worst([missing(tiny), missing(bit)])
    add("D3", "sampling follows the target law (distribution check)", False, st,
        f"exact-law test on tiny models (tests/test_dflash.py on CUDA): {status_of(tiny)}; Qwen3-4B two-sample permutation "
        f"tests of sampled responses: {len(checks)} run, {len(rej)} reject at p < 1e-3 (min p "
        f"{fmt(min((c['p_value'] for c in checks), default=None), '', 4)}); a non-rejection with "
        f"{checks[0]['n_plain'] if checks else 0} samples per arm has limited power", ["measurements/dflash.json"])
    if bench:
        accs = [(p_["policy"], r["max_new"], r["accepted_per_pass_mean"], r["accepted_per_pass_ci95"])
                for p_ in bench["policies"] for r in p_["runs"]]
        add("D4", "acceptance length reported", False, "PASS" if any(a[2] is not None for a in accs) else "INCONCLUSIVE",
            "; ".join(f"{pol} max_new {mn}: {fmt(a, '', 2)} accepted/pass CI {ci}" for pol, mn, a, ci in accs),
            ["measurements/dflash.json"])
        steps = [(p_["policy"], p_["grpo_step_ratio"], p_["grpo_step_ratio_ci95"]) for p_ in bench["policies"]]
        e2e = dfl.get("end_to_end")
        add("D5", "GRPO step time with and without DFlash", False,
            "PASS" if any(s[1] is not None for s in steps) and e2e else "INCONCLUSIVE",
            "benchmark (one GRPO step per repeat): " + "; ".join(f"{pol}: dflash/plain {fmt(r_, '', 3)} CI {ci}"
                                                                 for pol, r_, ci in steps)
            + (f"; end to end (smoke runs): dflash/plain step time {fmt(e2e['ratio_dflash_over_plain'], '', 3)} "
               f"CI {e2e['ratio_ci95']}" if e2e else "; no end-to-end DFlash GRPO run"),
            ["comparisons/dflash_vs_plain.csv"])
        cis = [ci for _, _, ci in steps if ci] + ([e2e["ratio_ci95"]] if e2e and e2e.get("ratio_ci95") else [])
        sp = "PASS" if cis and all(ci[1] < 1 for ci in cis) else "FAIL" if cis and all(ci[0] > 1 for ci in cis) else \
            "INCONCLUSIVE"
        add("D5b", "DFlash speeds up the GRPO step (if not: why)", False, sp,
            "speedup claimed only where the ratio's 95% CI lies below 1; target passes per token "
            + ", ".join(fmt(r["sample_passes_per_token"], "", 2) for p_ in bench["policies"] for r in p_["runs"]),
            ["comparisons/dflash_vs_plain.csv"])
    else:
        for rid, cl in (("D4", "acceptance length reported"), ("D5", "GRPO step time with and without DFlash"),
                        ("D5b", "DFlash speeds up the GRPO step (if not: why)")):
            add(rid, cl, False, missing(bit), why(bit, "H10.dflash_bench") + " (no benchmark result)")
    drift = dfl["benchmarks"].get("H10.dflash_drift")
    dit = run.item("G", "H10.dflash_drift")
    add("D6", "acceptance as the policy drifts from the drafter's base", False,
        "PASS" if drift else ("INCONCLUSIVE" if bench else missing(dit)),
        "real checkpoints of the final run: " + "; ".join(
            f"{p_['policy']}: " + ", ".join(fmt(r["accepted_per_pass_mean"], "", 2) for r in p_["runs"]) for p_ in drift["policies"])
        if drift else ("synthetic drift only (LoRA B scales in the benchmark); " + why(dit, "H10.dflash_drift")
                       if bench else why(dit, "H10.dflash_drift")), ["measurements/dflash.json"])

    # learning
    git_ = run.item("G", "H12.final_eval")
    if e and e.get("paired"):
        tb = e["paired"].get("trained_minus_base", {})
        sep = (e.get("data_separation") or {}).get("test_texts_in_train")
        st = "FAIL" if sep else ("PASS" if tb.get("ci_low", -1) > 0 else "FAIL")
        add("V1", "held-out improvement over the base model with a 95% CI", True, st,
            f"test n={e['n_test']}: trained - base accuracy {fmt(tb.get('diff'), '', 4)} 95% CI [{fmt(tb.get('ci_low'), '', 4)}, "
            f"{fmt(tb.get('ci_high'), '', 4)}]; test texts in training split: {sep}; test-set evaluations started in this "
            f"run: {e['test_set_evaluation_attempts']}", [e["file"]], tb)
        mt = e["paired"].get("merged_minus_trained", {})
        agree, hu = e.get("merged_vs_trained_prediction_agreement"), (e.get("merge") or {}).get("all_within_half_ulp")
        dd = e["merged_vs_trained_label_score_abs_diff"]
        ok = hu is True and agree is not None and agree >= 0.99 and mt.get("ci_low", 1) <= 0 <= mt.get("ci_high", -1)
        add("V2", "merged model matches the adapter-attached policy up to numerical noise", True, "PASS" if ok else "FAIL",
            f"tensors within half an ulp {hu}; prediction agreement {fmt(agree, '', 4)} (criterion >= 0.99); merged - trained "
            f"accuracy {fmt(mt.get('diff'), '', 4)} CI [{fmt(mt.get('ci_low'), '', 4)}, {fmt(mt.get('ci_high'), '', 4)}] (must "
            f"contain 0); label-score |diff| max {fmt(dd['max'], '', 4)} mean {fmt(dd['mean'], '', 5)} (reported)", [e["file"]])
        cb, cl = e["paired"].get("control_minus_base"), e.get("control_log") or {}
        ctrl = (e.get("models") or {}).get("control") or {}
        lt_c = m["learning"].get("control_training") or {}
        if cb is None:
            add("V3", "negative control (lr = 0) behaves as no learning", True, "NOT_TESTED", "no control evaluated")
        else:
            ok = cl.get("params_equal_init") is True and ctrl.get("changed_vs_base") == 0 and cb["ci_low"] <= 0 <= cb["ci_high"]
            add("V3", "negative control (lr = 0) behaves as no learning", True, "PASS" if ok else "FAIL",
                f"adapter equal to its initialization {cl.get('params_equal_init')}; test predictions changed vs base "
                f"{fmt(ctrl.get('changed_vs_base'), '', 4)}; control - base {fmt(cb['diff'], '', 4)} CI [{fmt(cb['ci_low'], '', 4)}, "
                f"{fmt(cb['ci_high'], '', 4)}]. Schedule: control {lt_c.get('successful_updates')} updates vs trained "
                f"{lt.get('successful_updates')} (same data order, seed and batch shape; shorter, so it controls for the "
                "pipeline, not for the length of training)", [e["file"]])
        tr_m, base_m = (e.get("models") or {}).get("trained", {}), (e.get("models") or {}).get("base", {})
        pos = tr_m.get("predicted_positive_rate")
        neg = tr_m.get("predicted_negative_rate")
        inv_ok = tr_m.get("invalid_rate") is not None and base_m.get("invalid_rate") is not None and \
            tr_m["invalid_rate"] <= base_m["invalid_rate"]
        coll_ok = pos is not None and neg is not None and max(pos, neg) <= 0.95
        add("V4", "learning, not reward gaming: invalid rate does not rise, no collapse to one label", True,
            "PASS" if inv_ok and coll_ok else "FAIL",
            f"invalid rate base {fmt(base_m.get('invalid_rate'), '', 4)} -> trained {fmt(tr_m.get('invalid_rate'), '', 4)}; "
            f"predicted POSITIVE {fmt(pos, '', 3)}, NEGATIVE {fmt(neg, '', 3)} (collapse = one label > 0.95, our "
            f"operationalisation); recall POS {fmt(tr_m.get('recall_POSITIVE'), '', 3)} NEG {fmt(tr_m.get('recall_NEGATIVE'), '', 3)}",
            [e["file"]])
    else:
        for rid, cl in (("V1", "held-out improvement over the base model with a 95% CI"),
                        ("V2", "merged model matches the adapter-attached policy up to numerical noise"),
                        ("V3", "negative control (lr = 0) behaves as no learning"),
                        ("V4", "learning, not reward gaming: invalid rate does not rise, no collapse to one label")):
            add(rid, cl, True, missing(git_), why(git_, "H12.final_eval") + " (no evaluation result)")

    # silent failures, Soup, telemetry, logs
    faults = [(s, i["id"]) for s in "B" for i in run.items(s) if ".fault_" in i["id"]]
    sf = run.item("B", "silent_failure_tests")
    st = worst([status_of(sf)] + [status_of(run.item(s, i)) for s, i in faults])
    add("F1", "silent failures: what, how checked, evidence (fault injection detected)", False, st,
        f"silent-failure tests {status_of(sf)}; {sum(status_of(run.item(s, i)) == 'PASS' for s, i in faults)}/{len(faults)} "
        "injected faults detected (reports/silent_failures.md)", ["reports/silent_failures.md"])
    soup = [i for i in run.items("H") if i["id"].startswith("soup_")]
    ran = [i for i in soup if i["status"] in USABLE]
    add("F2", "whether Soup's data doctor, pre-flight and soup ship catch them", False,
        "NOT_TESTED" if not ran else worst(i["status"] for i in ran if i["status"] in USABLE),
        "; ".join(f"{i['id']}: {i['status']}" for i in soup) or "stage H not run", ["raw/soup_checks/"])
    gpu_items = [(s, i) for s in "CDEFG" for i in run.items(s) if i.get("telemetry")]
    no_rows = [i["id"] for _, i in gpu_items if not i["telemetry"].get("gpu_rows")]
    add("X2", "raw nvidia-smi sampled during the whole run", False,
        "NOT_TESTED" if not gpu_items else ("PASS" if not no_rows else "INCONCLUSIVE"),
        f"{len(gpu_items)} GPU items with per-item raw CSVs at 200 ms" + (f"; no device rows for {no_rows}" if no_rows else ""),
        ["raw/"])
    add("X3", "raw timestamped logs", False, "PASS" if (run.dir / "logs" / "orchestrator.log").exists() else "INCONCLUSIVE",
        "logs/orchestrator.log (UTC per line) and per-item stdout/stderr under raw/", ["logs/", "raw/"])
    return R


# --- reports ------------------------------------------------------------------------------------------------------

AUDIT_EXTRA = {  # stage -> (fault-control items of this run, Soup check aimed at the stage, its stage-H item ids)
    "Data": ([], "soup data doctor (chat template / EOS / truncation)", ["soup_data_doctor"]),
    "Tokenizer/template": (["H04.fault_no_padding_mask"], "soup data doctor", ["soup_data_doctor"]),
    "Reward": ([], "soup ship (task win vs base); no reward check in this run", ["soup_ship"]),
    "GRPO": (["H05.fault_advantage"], "soup pre-flight", ["soup_preflight"]),
    "L2L forward/backward": (["H01.fault_wrong_layer", "H02.fault_detach_boundary", "H01.fault_unsafe_reuse",
                              "H02.fault_unsafe_reuse"], "none (Soup has no L2L engine)", []),
    "Output head": (["H03.fault_raw_logit"], "none", []),
    "Generation": (["H04.fault_stale_position", "H04.fault_no_padding_mask"], "none", []),
    "Disk streaming/resume": ([], "none", []),
    "DFlash": ([], "none", []),
    "Merge": ([], "soup adapters check (adapter tensors only); soup ship",
              ["soup_adapters_check_trained", "soup_adapters_check_step0", "soup_ship"]),
    "Inference/memory": ([], "soup ship / pre-flight", ["soup_ship", "soup_preflight"]),
}


def silent_failures_md(run, reqs):
    import importlib.util
    spec = importlib.util.spec_from_file_location("final_report", ROOT / "scripts" / "final_report.py")
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    soup = {i["id"]: i for i in run.items("H")}
    rows = []
    for stage, failure, detection, fault, occurred, tooling in fr.AUDIT:
        faults, soup_check, soup_ids = AUDIT_EXTRA.get(stage, ([], "none", []))
        measured = "; ".join(f"{f}: {status_of(run.item('B', f))}" for f in faults) or "no fault control in this run"
        soup_res = "; ".join(f"{k}: {status_of(soup.get(k))}" for k in soup_ids) or "n/a"
        rows.append([stage, failure, detection, fault, measured, soup_check, soup_res,
                     "see limitations" if "no fault control" in measured else "detected" if "FAIL" not in measured else "NOT detected"])
    sf = run.item("B", "silent_failure_tests")
    md = ["# Silent-failure audit", "",
          "Chain: data → tokenizer/template → reward → GRPO → L2L → streaming → DFlash → merge → inference. "
          "The 'injected fault' column is the fault-injection run of *this* assignment run (PASS = the fault was detected). "
          "The Soup column reports only what Soup did in this run; a check that was not run is NOT_TESTED and makes no "
          "claim.", "",
          f"Silent-failure test suite (tests/silent_failures, fault injection on tiny models): **{status_of(sf)}** "
          f"({(sf or {}).get('detail')})", "",
          table(["stage", "silent failure", "detection", "controlled fault / evidence", "injected fault in this run",
                 "Soup check aimed at it", "Soup result here", "outcome"], rows), "",
          "## What these checks do not detect", "",
          "- A reward that is correct on our labels but a poor proxy for the task (the reward *is* the label match here).",
          "- Bugs that affect the resident reference and L2L identically (equivalence tests compare the two).",
          "- Numerical differences on another GPU or software stack (bitwise equality is per machine).",
          "- Memory spikes shorter than the 200 ms nvidia-smi interval.",
          "- Forgetting of capabilities outside sentiment classification (no general benchmark is evaluated).",
          "- Test-set leakage through near-duplicates that differ in text (separation is checked by exact text)."]
    return "\n".join(md) + "\n"


def memory_md(run, m):
    md = ["# Memory analysis", "",
          f"Cap: **{run.cap:.0f} bytes** (\"8 GB of GPU memory as shown by nvidia-smi, CUDA context included\"), i.e. "
          f"{run.cap / MIB:.1f} MiB = {run.cap / GIB:.3f} GiB. The binary reading, 8 GiB = {run.cap_gib_ref / MIB:.0f} MiB, is "
          "the threshold of the repository's in-process guard and of H11's pre-registered criterion; it is shown for "
          "reference only. nvidia-smi was sampled every 200 ms (raw CSV per item); spikes between samples are not seen.",
          "", "## Measured per item (nvidia-smi)", ""]
    rows = []
    for stage, it in run.all_items():
        t = it.get("telemetry")
        if t and (t.get("gpu_rows") or t.get("apps_rows")):
            rows.append([stage, it["id"], fmt(t["process_peak_mib"], "", 0), fmt(t["device_peak_used_mib"], "", 0),
                         t["attributed_numeric_samples"], it.get("memory_status", "n/a (not capped)"),
                         t["raw_files"]["apps"].replace(str(run.dir) + "/", "")])
    md.append(table(["stage", "item", "process peak MiB", "device peak MiB", "per-process samples", "status vs cap",
                     "raw"], rows) if rows else "MISSING: no GPU item produced nvidia-smi samples in this run.")
    prof = run.output("D", "H11.profile")
    md += ["", "## Predicted vs measured, per phase (H11 profile)", ""]
    if prof and (prof.get("prediction") or {}).get("rows"):
        md.append(table(["phase", "pred. allocated GiB", "meas. peak allocated", "pred. reserved", "meas. peak reserved",
                         "pred. device", "meas. device (cudaMemGetInfo)", "meas. nvidia-smi process", "pred. CPU",
                         "meas. CPU RSS"],
                        [[r["phase"], fmt(gib(r.get("predicted_allocated")), "", 2), fmt(gib(r.get("observed_peak_allocated")), "", 2),
                          fmt(gib(r.get("predicted_reserved")), "", 2), fmt(gib(r.get("observed_peak_reserved")), "", 2),
                          fmt(gib(r.get("predicted_device")), "", 2), fmt(gib(r.get("observed_device_peak")), "", 2),
                          fmt(gib(r.get("observed_process_peak")), "", 2), fmt(gib(r.get("predicted_cpu")), "", 2),
                          fmt(gib(r.get("observed_cpu_rss_peak")), "", 2)] for r in prof["prediction"]["rows"]]))
        md += ["", "Structural reasons for gaps (true by construction, not fitted): allocated < reserved (allocator "
               "cache); the nvidia-smi process figure adds the CUDA context, cuBLAS workspaces and non-PyTorch "
               "allocations to reserved; cudaMemGetInfo counts every process on the device. Remaining gaps: "
               "**researcher to explain** from the per-phase numbers above."]
    else:
        md.append("MISSING: the H11 profile did not produce a result in this run "
                  f"({why(run.item('D', 'H11.profile'), 'H11.profile')}).")
    pred = predictions(run)
    md += ["", "## Theoretical breakdown (budget calculator, before any run)", ""]
    if pred:
        rows = []
        for ph in pred["data"].get("phases", []):
            for mo in ph.get("moments", [])[:1]:
                for c in mo.get("gpu", []):
                    rows.append([ph["phase"], c["name"], fmt(None if c.get("bytes") is None else c["bytes"] / MIB, "", 0),
                                 c.get("kind")])
        md += [f"Source: `{Path(pred['file']).relative_to(run.dir) if str(pred['file']).startswith(str(run.dir)) else pred['file']}` "
               f"({pred['source']}). Per-component GPU memory is theoretical only: no measured counterpart exists per "
               "component (the instruments measure totals).", "", table(["phase", "component", "MiB", "kind"], rows)]
    else:
        md.append("MISSING: no prediction recorded.")
    tr = [r for r in m["throughput"]]
    md += ["", "## PCIe: predicted vs measured transfer time", "",
           table(["phase", "bytes", "bandwidth B/s", "transfer s (bytes / bandwidth)", "wall s", "transfer share", "kind"],
                 [[r[0], fmt(r[1]), fmt(r[2], "", 0), fmt(r[3], "", 2), fmt(r[4], "", 2), fmt(r[5], "", 2), r[6]] for r in tr])
           if tr else "MISSING"]
    return "\n".join(md) + "\n"


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if x is None else x for x in r])


def summary_md(run, reqs, verdict, must, m):
    f = run.facts()
    e = m["learning"].get("evaluation")
    ess = [r for r in reqs if r["essential"]]
    md = [f"# GRPO + LoRA on Qwen3-4B under an 8 GB cap: {verdict}", "",
          f"*Draft generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from assignment run "
          f"`{run.state.get('run_id')}` at commit `{str(run.state.get('identity', {}).get('git_commit'))[:10]}`"
          f"{' (dirty tree)' if run.state.get('identity', {}).get('tracked_dirty') else ''}, GPU {f.get('gpu') or 'none'}. "
          "Every number below comes from this run's files; MISSING means it was not measured. Sections marked "
          "**[researcher]** must be written by hand.*", "",
          "## Verdict", "",
          f"**{verdict}.** Rule: SHIP only if every essential requirement is PASS ({sum(r['status'] == 'PASS' for r in ess)}/"
          f"{len(ess)} are)." + ("" if not must else " What must change first:"), ""]
    md += [f"- {x}" for x in must[:8]]
    md += ["", "## Requirements", "", table(["id", "requirement", "status"],
                                             [[r["id"], r["clause"], f"**{r['status']}**"] for r in reqs]),
           "", "## Memory (nvidia-smi per process, CUDA context included; cap 8e9 B = 7629 MiB)", ""]
    md.append(table(["requirement", "status", "evidence"], [[r["id"], r["status"], r["detail"][:300]]
                                                            for r in reqs if r["id"] in ("M1", "M2", "M3", "M4")]))
    md += ["", "Predicted vs measured per phase and PCIe time: `reports/memory_analysis.md`.", "",
           "## L2L, streaming, merge", ""]
    md += [f"- {r['id']} {r['clause']}: **{r['status']}** - {r['detail'][:260]}"
           for r in reqs if r["id"] in ("L2", "L3", "L4", "L5", "L6", "S1", "S2")]
    md += ["", "## DFlash", ""]
    md += [f"- {r['id']} {r['clause']}: **{r['status']}** - {r['detail'][:260]}" for r in reqs if r["id"].startswith("D")]
    md += ["", "## Did the model learn? (held-out test split, used once)", ""]
    if e and e.get("models"):
        md.append(table(["model", "accuracy", "macro F1", "invalid", "pred. POSITIVE", "recall POS", "recall NEG",
                         "changed vs base"],
                        [[k, fmt(v.get("accuracy"), "", 4), fmt(v.get("macro_f1"), "", 4), fmt(v.get("invalid_rate"), "", 4),
                          fmt(v.get("predicted_positive_rate"), "", 3), fmt(v.get("recall_POSITIVE"), "", 3),
                          fmt(v.get("recall_NEGATIVE"), "", 3), fmt(v.get("changed_vs_base"), "", 3)]
                         for k, v in e["models"].items()]))
        md += ["", "Paired bootstrap 95% CIs: " + "; ".join(f"{k} {fmt(v.get('diff'), '', 4)} [{fmt(v.get('ci_low'), '', 4)}, "
                                                          f"{fmt(v.get('ci_high'), '', 4)}]" for k, v in e["paired"].items())]
    else:
        md.append("MISSING: no held-out evaluation in this run.")
    md += [f"- {r['id']}: **{r['status']}** - {r['detail'][:260]}" for r in reqs if r["id"].startswith("V")]
    md += ["", "A rising reward, a finished run or a nonzero adapter change is not counted as learning: only the paired "
           "held-out difference against the base, with the LR=0 control as the no-learning reference and the "
           "invalid-rate / label-collapse checks against reward gaming. What these checks cannot see: "
           "`reports/silent_failures.md` and `reports/limitations.md`.", "",
           "## Silent failures and Soup", ""]
    md += [f"- {r['id']}: **{r['status']}** - {r['detail'][:260]}" for r in reqs if r["id"] in ("F1", "F2")]
    md += ["", "## What surprised me, what still concerns me **[researcher]**", "",
           "_To be written by the researcher. Candidates from this run's numbers (check before using): "
           + "; ".join(f"{r['id']} is {r['status']}" for r in reqs if r["status"] != "PASS")[:600] + "._", "",
           "## AI-assisted work **[researcher]**", "",
           "_State where AI tools were used, which suggestions were accepted, and what was verified, changed or redone. "
           "Known from the repository: this orchestration layer (scripts/run_assignment.sh, scripts/assignment*.py, "
           "qwen_grpo/telemetry.py) was drafted with Claude Code; list what you checked yourself._", "",
           "## Reproducibility", "",
           f"`bash scripts/run_assignment.sh` at commit `{run.state.get('identity', {}).get('git_commit')}`; configs and their "
           "sha256 in `resolved_configs/`; environment in `environment/`; raw logs and nvidia-smi CSVs in `raw/`; file "
           "hashes in `checksums.sha256`.", "",
           "## References", "",
           "- Shao et al., DeepSeekMath (GRPO), arXiv:2402.03300. - Hu et al., LoRA, arXiv:2106.09685. - Qwen3 technical "
           "report, arXiv:2505.09388. - DFlash (z-lab), arXiv:2602.06036 (as cited in docs/l2l/12_dflash.md)."]
    return "\n".join(md) + "\n"


def limitations_md(run, reqs, m):
    md = ["# Limitations and open items", "", "## Requirements that are not PASS", ""]
    md += [f"- **{r['id']} {r['status']}** ({'essential' if r['essential'] else 'supporting'}): {r['clause']}. {r['detail']}"
           for r in reqs if r["status"] != "PASS"] or ["- none"]
    md += ["", "## Blockers recorded by preflight", ""]
    md += [f"- stages {b['stages']}: {b['reason']}" for b in run.facts().get("blockers", [])] or ["- none"]
    md += ["", "## Skipped or blocked stages", ""]
    for s in "ABCDEFGH":
        a = run.attempt(s)
        if a and a.get("blocked_by"):
            md.append(f"- stage {s}: blocked by " + "; ".join(a["blocked_by"]))
        elif not a:
            md.append(f"- stage {s}: not started")
    md += ["", "## Standing limitations of the method", "",
           "- nvidia-smi is sampled every 200 ms: a measured peak is a lower bound; shorter spikes are invisible.",
           "- The cap is judged on the process group's per-process readings. Where nvidia-smi does not list our PIDs (containers), "
           "memory requirements are INCONCLUSIVE, never PASS.",
           "- The LR=0 control runs 30 updates against 210 for the trained run: it is matched in data order, seed, batch "
           "shape and code path, not in length.",
           "- The final run does not use DFlash; DFlash is measured in a separate benchmark and a 4-step smoke run.",
           "- The 4B sampling-distribution check is a two-sample test of a few hundred samples; only the tiny-model test is "
           "exact.",
           "- Bitwise L2L equivalence holds per machine and software stack; it is re-established on this GPU in stage B.",
           "- The base model is already strong on this task; no forgetting benchmark is evaluated.",
           "- Per-component GPU memory (buffers, activations, logits, KV cache, LoRA/optimizer, drafter) is theoretical; "
           "only totals are measured."]
    return "\n".join(md) + "\n"


def experimental_md(run, m):
    md = ["# Experimental results (full)", "", "## Items of this run", ""]
    rows = []
    for s, it in run.all_items():
        rows.append([s, it["id"], it["status"], it.get("memory_status", ""), fmt(it.get("seconds")),
                     "reused" if it.get("reused") else ("again" if it.get("again") else ""),
                     str(it.get("detail"))[:160]])
    md.append(table(["stage", "item", "status", "nvidia-smi", "seconds", "evidence", "detail"], rows))
    md += ["", "## Numerical equivalence (lab cases of this run)", ""]
    rows = []
    for c in m["correctness"]:
        lab = c.get("lab")
        if not lab:
            continue
        for q, v in lab["quantities"].items():
            rows.append([c["item"], lab["name"], q, v["n"], f"{v['max_abs']:.3g}", f"{v['bound_max']:.3g}", v["fails"],
                         ",".join(v["dtypes"]), v["shape_example"], lab.get("gpu") or lab.get("device"), lab["status"]])
    md.append(table(["item", "result", "quantity", "comparisons", "max |err|", "max bound", "fails", "dtypes",
                     "shape (example)", "hardware", "status"], rows) if rows else "MISSING")
    md += ["", "## DFlash", "", "```json", json.dumps(m["dflash"], indent=1, default=str)[:20000], "```",
           "", "## Learning", "", "```json", json.dumps(m["learning"], indent=1, default=str)[:20000], "```"]
    return "\n".join(md) + "\n"


def main(run_dir):
    run = Run(run_dir)
    d = run.dir
    for sub in ("measurements", "comparisons", "reports"):
        (d / sub).mkdir(exist_ok=True)
    m = {}
    mem = memory_rows(run)
    write_csv(d / "measurements" / "memory.csv", ["stage", "item", "phase", "quantity", "kind", "bytes", "source", "note"], mem)
    m["timing"] = timing_rows(run)
    write_csv(d / "measurements" / "timing.csv", ["run_or_phase", "step", "rollout_s", "prepare_s", "update_s", "step_s",
                                                  "rollout_tokens", "weight_bytes_copied", "backend", "kind"], m["timing"])
    m["throughput"] = throughput_rows(run)
    write_csv(d / "measurements" / "throughput.csv", ["phase", "bytes", "bandwidth_bytes_per_s", "transfer_s", "wall_s",
                                                      "transfer_share", "kind"], m["throughput"])
    m["correctness"] = correctness(run)
    m["dflash"] = dflash(run)
    m["learning"] = learning(run)
    peaks = [{"stage": s, "item": i["id"], **{k: i["telemetry"].get(k) for k in
             ("process_peak_mib", "device_peak_used_mib", "attributed_numeric_samples")},
              "memory_status": i.get("memory_status") or "not capped (the cap is judged on Qwen3-4B stages D-G)",
              "raw": i["telemetry"]["raw_files"]} for s, i in run.all_items() if i.get("telemetry")]
    for name, obj in (("memory_summary", {"cap_bytes": run.cap, "cap_mib": run.cap / MIB,
                                          "binary_reference_bytes": run.cap_gib_ref, "note": "measured values only; "
                                          "null = unavailable", "items": peaks}),
                      ("correctness", m["correctness"]), ("dflash", m["dflash"]), ("learning", m["learning"])):
        (d / "measurements" / f"{name}.json").write_text(json.dumps(obj, indent=1, default=str) + "\n")
    reqs = requirements(run, m)
    ess_bad = [r for r in reqs if r["essential"] and r["status"] != "PASS"]
    verdict = "SHIP" if not ess_bad else "DON'T SHIP"
    must = [f"{r['id']} {r['clause']}: {r['status']} - {r['detail'][:200]}" for r in ess_bad]
    from qwen_grpo.evidence import code_sha256, report_code_sha256
    run_code = (run.state.get("identity") or {}).get("experiment_code_sha256")
    report_identity = {"report_code_sha256": report_code_sha256(), "experiment_code_sha256_of_run": run_code,
                       "experiment_code_sha256_now": code_sha256(),
                       "note": "the evidence was produced by the run's experiment code; this report only summarises it"}
    (d / "requirements.json").write_text(json.dumps({"verdict": verdict, "rule": "SHIP iff every essential requirement "
                                                     "is PASS", "must_change": must, "requirements": reqs,
                                                     "report_identity": report_identity,
                                                     "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                                    indent=1, default=str) + "\n")
    # comparisons
    write_csv(d / "comparisons" / "l2l_equivalence.csv", ["item", "result", "quantity", "comparisons", "max_abs_error",
                                                          "max_bound", "fails", "dtypes", "hardware", "status"],
              [[c["item"], c["lab"]["name"], q, v["n"], v["max_abs"], v["bound_max"], v["fails"], ";".join(v["dtypes"]),
                c["lab"].get("gpu") or c["lab"].get("device"), c["lab"]["status"]]
               for c in m["correctness"] if c.get("lab") for q, v in c["lab"]["quantities"].items()])
    perf = []
    for name in ("qwen3_0.6b_resident", "qwen3_0.6b_l2l_sync", "qwen3_0.6b_l2l_async"):
        lg = run.run_log(name, "C", "H05.real_0.6b_bf16")
        if lg:
            perf.append([name, lg.get("train_seconds"), median([t.get("step_s") for t in lg.get("timings", [])]),
                         max((r.get("peak_reserved_gib") or 0 for r in lg.get("training", [])), default=None), "measured, one run"])
    pt = lab_summary((run.item("C", "H06.prefetch_timing") or {}).get("evidence_name")) \
        if status_of(run.item("C", "H06.prefetch_timing")) in USABLE else None
    if pt and pt.get("conclusion"):
        perf.append(["H06 async/sync step-time ratio", pt["conclusion"].get("ratio"), None, None,
                     f"CI {pt['conclusion'].get('ci95')}"])
    write_csv(d / "comparisons" / "backend_performance.csv", ["backend_or_metric", "train_seconds_or_ratio",
                                                              "median_step_s", "peak_reserved_gib", "note"], perf)
    dv = []
    for k, b in m["dflash"]["benchmarks"].items():
        for p in b["policies"]:
            for r in p["runs"]:
                dv.append([k, json.dumps(p["policy"]), r["max_new"], r["greedy_equal"], r["greedy_passes_per_token"],
                           r["sample_passes_per_token"], r["accepted_per_pass_mean"], r["accepted_per_pass_ci95"],
                           r["sample_plain_s"], r["sample_dflash_s"], r["sample_time_ratio_dflash_over_plain"],
                           r["sample_time_ratio_ci95"], p["grpo_step_plain_s"], p["grpo_step_dflash_s"], p["grpo_step_ratio"],
                           p["grpo_step_ratio_ci95"], r["trials"]])
    e2e = m["dflash"].get("end_to_end")
    if e2e:
        dv.append(["smoke runs (end to end)", "base", 4, None, None, None, None, None, median(e2e["plain_step_s"]),
                   median(e2e["dflash_step_s"]), e2e["ratio_dflash_over_plain"], e2e["ratio_ci95"], None, None, None, None,
                   len(e2e["plain_step_s"])])
    write_csv(d / "comparisons" / "dflash_vs_plain.csv",
              ["benchmark", "policy", "max_new", "greedy_equal", "greedy_passes_per_token", "sample_passes_per_token",
               "accepted_per_pass_mean", "accepted_ci95", "sample_plain_s", "sample_dflash_s", "sample_ratio",
               "sample_ratio_ci95", "grpo_step_plain_s", "grpo_step_dflash_s", "grpo_step_ratio", "grpo_step_ratio_ci95",
               "trials"], dv)
    ev = m["learning"].get("evaluation")
    write_csv(d / "comparisons" / "model_artifacts.csv",
              ["model", "accuracy", "macro_f1", "invalid_rate", "predicted_positive_rate", "recall_POSITIVE",
               "recall_NEGATIVE", "auc", "changed_vs_base", "n"],
              [[k, v.get("accuracy"), v.get("macro_f1"), v.get("invalid_rate"), v.get("predicted_positive_rate"),
                v.get("recall_POSITIVE"), v.get("recall_NEGATIVE"), v.get("auc"), v.get("changed_vs_base"), v.get("n")]
               for k, v in ((ev or {}).get("models") or {}).items()])
    # reports
    R = d / "reports"
    (R / "requirements_matrix.md").write_text(
        "# Requirements matrix\n\n" + f"Verdict: **{verdict}** (SHIP iff every essential requirement is PASS)\n\n"
        + table(["id", "assignment clause", "essential", "status", "detail", "evidence"],
                [[r["id"], r["clause"], "yes" if r["essential"] else "no", f"**{r['status']}**", r["detail"],
                  ", ".join(str(x) for x in r["evidence"] if x)] for r in reqs]) + "\n")
    (R / "silent_failures.md").write_text(silent_failures_md(run, reqs))
    (R / "memory_analysis.md").write_text(memory_md(run, m))
    (R / "experimental_results.md").write_text(experimental_md(run, m))
    (R / "limitations.md").write_text(limitations_md(run, reqs, m))
    summ = summary_md(run, reqs, verdict, must, m)
    words = len(summ.split())
    if words > 1500:
        summ += f"\n> Length check: {words} words; a three-page report holds about 1500. Cut before submitting.\n"
    (R / "summary.md").write_text(summ)
    (R / "report_tables.md").write_text("# Report-ready tables\n\n" + table(
        ["id", "requirement", "status"], [[r["id"], r["clause"], r["status"]] for r in reqs]) + "\n\n"
        + "\n".join(memory_md(run, m).splitlines()[4:]) + "\n")
    (R / "final_verdict.md").write_text(
        f"# Final verdict: {verdict}\n\nRule (fixed before the run): SHIP only if every essential requirement is PASS. "
        "The verdict is computed by scripts/assignment_report.py from requirements.json; it is never written by hand.\n\n"
        + table(["essential requirement", "status", "detail"],
                [[f"{r['id']} {r['clause']}", r["status"], r["detail"][:300]] for r in reqs if r["essential"]])
        + ("\n\n## What must change first\n\n" + "\n".join(f"- {x}" for x in must) if must else "") + "\n")
    # checksums of everything in the run directory (raw included)
    lines = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "checksums.sha256" and not p.name.endswith(".tmp"):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            lines.append(f"{h.hexdigest()}  {p.relative_to(d)}")
    (d / "checksums.sha256").write_text("\n".join(lines) + "\n")
    print(f"verdict: {verdict} ({sum(r['status'] == 'PASS' for r in reqs)}/{len(reqs)} requirements PASS); "
          f"reports in {d / 'reports'}")
    for r in reqs:
        print(f"  {r['status']:12s} {r['id']:4s} {r['clause']}")
    return 0 if verdict == "SHIP" else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
