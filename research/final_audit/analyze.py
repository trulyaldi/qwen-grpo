"""Read-only audit analysis of the final Qwen3-4B GRPO+LoRA artifacts.

Reads (never writes) the run, telemetry and evaluation artifacts and prints the
derived numbers used in research/final_audit/*.md, with formulas. Output is also
written to research/final_audit/analysis_output.json.

Usage (CPU only, stdlib + numpy-free):
    python3 -I research/final_audit/analyze.py
"""
import json
import math
import random
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/qwen3_4b_final_plain_150_v2"
CTL = ROOT / "runs/qwen3_4b_final_plain_150_v2_lr0"
EVAL = ROOT / "research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json"
DATA = ROOT / "data/sentiment.json"
OUT = {}


def load(p):
    return json.loads(Path(p).read_text())


def section(name):
    print("\n" + "=" * 8, name, "=" * 8)


# ---------------------------------------------------------------- training
def training(run_dir, tag):
    section(f"training {tag}")
    t = load(run_dir / "training/training.json")
    steps, zs, tim = t["training"], t["zero_signal"], t["timings"]
    o = {}
    o["successful_steps"] = len(steps)
    o["zero_signal_batches"] = len(zs)
    o["attempts"] = t["groups_generated"]
    o["train_seconds"] = t["train_seconds"]
    # group-level reward structure across every attempt
    allrec = sorted(steps + zs, key=lambda r: r["attempt"])
    g_all1 = g_all0 = g_mixed = 0
    resp_rewards = []
    invalid = 0
    for r in allrec:
        invalid += r.get("invalid_responses", 0)
        for grp in r["rewards"]:
            resp_rewards += grp
            if all(x == 1.0 for x in grp):
                g_all1 += 1
            elif all(x == 0.0 for x in grp):
                g_all0 += 1
            else:
                g_mixed += 1
    ngroups = g_all1 + g_all0 + g_mixed
    o["groups_total"] = ngroups
    o["groups_all_correct"] = g_all1
    o["groups_all_wrong"] = g_all0
    o["groups_mixed"] = g_mixed
    o["frac_groups_mixed"] = g_mixed / ngroups
    o["responses_total"] = len(resp_rewards)
    o["response_mean_reward_all_attempts"] = sum(resp_rewards) / len(resp_rewards)
    o["invalid_responses_total"] = invalid
    o["invalid_rate_rollouts"] = invalid / len(resp_rewards)
    # zero-signal batch composition
    zs_comp = Counter()
    for r in zs:
        a1 = sum(all(x == 1.0 for x in g) for g in r["rewards"])
        zs_comp[f"{a1}_all_correct_of_5"] += 1
    o["zero_signal_batch_composition"] = dict(zs_comp)
    # analytical zero-signal expectation from per-group mixed fraction p
    p = g_mixed / ngroups
    o["expected_zero_signal_frac_if_independent"] = (1 - p) ** 5
    o["observed_zero_signal_frac"] = len(zs) / t["groups_generated"]
    # rollout accuracy over attempts (all batches, unbiased by selection)
    bins = {}
    for r in allrec:
        b = (r["attempt"] - 1) // 50
        bins.setdefault(b, []).append(r["mean_reward"])
    o["rollout_mean_reward_by_50_attempts"] = {
        f"{50*b+1}-{50*b+50}": round(sum(v) / len(v), 4) for b, v in sorted(bins.items())
    }
    # prompt coverage
    sel = [pid for r in allrec for pid in r["prompt_ids"]]
    o["prompt_selections"] = len(sel)
    o["unique_prompts"] = len(set(sel))
    o["repeat_selections"] = len(sel) - len(set(sel))
    # step stats
    if steps:
        keys = ["loss", "grad_norm", "update_norm", "ratio_mean", "ratio_min", "ratio_max",
                "clip_frac", "approx_kl", "effective_adapter_norm", "change_from_init", "mean_reward"]
        for k in keys:
            v = [s[k] for s in steps]
            o[f"{k}_min"], o[f"{k}_max"] = min(v), max(v)
            o[f"{k}_mean"] = sum(v) / len(v)
        o["abs_loss_max"] = max(abs(s["loss"]) for s in steps)
        o["mixed_groups_per_update"] = dict(Counter(s["mixed_groups"] for s in steps))
        o["final_change_from_init"] = steps[-1]["change_from_init"]
        o["final_effective_adapter_norm"] = steps[-1]["effective_adapter_norm"]
        o["final_lora_B_norm"] = steps[-1]["lora_B_norm"]
        o["lora_A_norm_first_last"] = [steps[0]["lora_A_norm"], steps[-1]["lora_A_norm"]]
        o["mean_reward_successful_first10_last10"] = [
            sum(s["mean_reward"] for s in steps[:10]) / 10,
            sum(s["mean_reward"] for s in steps[-10:]) / 10]
    # timing (successful steps only; zero-signal batches have no timing record)
    if tim:
        S = lambda k: sum(x[k] for x in tim)
        o["timed_records"] = len(tim)
        o["sum_rollout_s"] = S("rollout_s")
        o["sum_prepare_s"] = S("prepare_s")
        o["sum_update_s"] = S("update_s")
        o["sum_step_s"] = S("step_s")
        o["sum_rollout_tokens"] = S("rollout_tokens")
        o["rollout_tokens_per_s_successful"] = o["sum_rollout_tokens"] / o["sum_rollout_s"]
        o["share_rollout"] = o["sum_rollout_s"] / o["sum_step_s"]
        o["share_prepare"] = o["sum_prepare_s"] / o["sum_step_s"]
        o["share_update"] = o["sum_update_s"] / o["sum_step_s"]
        wb = sorted(set(x["weight_bytes_copied"] for x in tim))
        o["weight_bytes_copied_per_step_distinct"] = wb[:5]
        o["sum_weight_bytes_copied_successful"] = S("weight_bytes_copied")
        o["effective_bytes_per_s_successful_steps"] = o["sum_weight_bytes_copied_successful"] / o["sum_step_s"]
        o["seconds_not_in_timed_steps"] = t["train_seconds"] - o["sum_step_s"]
        o["updates_per_min"] = len(steps) / (t["train_seconds"] / 60)
        o["seconds_per_update"] = t["train_seconds"] / len(steps)
        o["mean_step_s"] = o["sum_step_s"] / len(tim)
    # validation subset evaluations
    ev = t["evaluations"]
    o["val_subset"] = [(e["step"], e["metrics"]["accuracy"], e["metrics"]["macro_f1"]) for e in ev]
    if len(ev) >= 2:
        p0 = {x["id"]: x["correct"] for x in ev[0]["predictions"]}
        pN = {x["id"]: x["correct"] for x in ev[-1]["predictions"]}
        b = sum(1 for i in p0 if p0[i] and not pN[i])
        c = sum(1 for i in p0 if not p0[i] and pN[i])
        o["val_subset_step0_vs_last_discordant"] = {"base_right_trained_wrong": b, "base_wrong_trained_right": c,
                                                   "mcnemar_exact_p": mcnemar(b, c)}
        if len(ev) >= 3:
            pa = {x["id"]: x["prediction"] for x in ev[-2]["predictions"]}
            pb = {x["id"]: x["prediction"] for x in ev[-1]["predictions"]}
            o[f"val_subset_step{ev[-2]['step']}_vs_{ev[-1]['step']}_pred_changes"] = sum(pa[i] != pb[i] for i in pa)
    o["frozen_changed"] = t["frozen_changed"]
    o["params_equal_init"] = t["params_equal_init"]
    o["final_checkpoint"] = t.get("final_checkpoint")
    for k, v in o.items():
        print(f"{k}: {v}")
    return o, t


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------- memory
def memory(run_dir, tag, t):
    section(f"memory {tag}")
    files = sorted((run_dir / "training/memory").glob("*.json"))
    m = load(files[0])
    s = m["samples"]
    ts = [x["t"] for x in s]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    o = {"file": str(files[0].relative_to(ROOT)), "budget_bytes": m["budget_bytes"], "interval_s": m["interval_s"],
         "n_samples": len(s), "span_s": ts[-1] - ts[0], "first_t": ts[0], "last_t": ts[-1],
         "gap_median_s": st.median(gaps), "gap_max_s": max(gaps), "gaps_over_2s": sum(g > 2 for g in gaps),
         "violations": len(m["violations"]), "phases_labelled": len(m["phases"]),
         "peak_record": m["peak"]}
    for k in ["process_used", "device_used", "reserved", "allocated"]:
        vals = [x.get(k) for x in s]
        good = [v for v in vals if isinstance(v, (int, float))]
        o[f"{k}_missing"] = len(vals) - len(good)
        o[f"{k}_max"] = max(good)
    o["process_used_max_vs_8e9"] = o["process_used_max"] / 8e9
    o["process_used_max_vs_8GiB"] = o["process_used_max"] / 2**33
    o["headroom_to_8e9_bytes"] = 8e9 - o["device_used_max"]
    o["context_overhead_process_minus_reserved_at_peak"] = o["process_used_max"] - o["reserved_max"]
    smi_bad = sum(1 for x in s if not x.get("smi_apps_raw") or not x.get("smi_gpu_raw"))
    o["samples_missing_smi_raw"] = smi_bad
    # coverage against the timestamped training log
    log = (run_dir / "training/training.log").read_text().splitlines()
    from datetime import datetime, timezone
    def ts_of(line):
        return datetime.strptime(line[:26], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
    stamps = [ts_of(l) for l in log if l[:4] == "2026"]
    o["log_first"], o["log_last"] = stamps[0], stamps[-1]
    o["telemetry_starts_before_log"] = ts[0] <= stamps[0]
    o["telemetry_ends_after_log"] = ts[-1] >= stamps[-1]
    o["telemetry_start_to_first_log_s"] = stamps[0] - ts[0]
    o["last_log_to_telemetry_end_s"] = ts[-1] - stamps[-1]
    # samples per inter-step window
    step_lines = [ts_of(l) for l in log if " step " in l[:40]]
    covered = 0
    for a, b in zip(step_lines, step_lines[1:]):
        if any(a <= x <= b for x in ts):
            covered += 1
    o["inter_step_windows"] = len(step_lines) - 1
    o["inter_step_windows_with_sample"] = covered
    for k, v in o.items():
        print(f"{k}: {v}")
    return o


# ---------------------------------------------------------------- evaluation
def evaluation():
    section("final evaluation (test)")
    e = load(EVAL)
    data = load(DATA)
    gold = [x["label"] for x in data["test"]]
    ids = e["label_scores"]["ids"]
    assert ids == list(range(len(gold))), "ids are not 0..n-1"
    o = {"n": len(gold), "gold_counts": dict(Counter(gold))}
    preds = e["predictions"]
    for m in ["base", "trained", "merged", "control"]:
        p = preds[m]
        acc = sum(a == g for a, g in zip(p, gold)) / len(gold)
        f1s = []
        for lab in ["POSITIVE", "NEGATIVE"]:
            tp = sum(a == lab and g == lab for a, g in zip(p, gold))
            fp = sum(a == lab and g != lab for a, g in zip(p, gold))
            fn = sum(a != lab and g == lab for a, g in zip(p, gold))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        invalid = sum(a not in ("POSITIVE", "NEGATIVE") for a in p)
        sc = e["label_scores"][m]
        o[m] = {"accuracy_recomputed": acc, "accuracy_reported": e["models"][m]["accuracy"],
                "macro_f1_recomputed": sum(f1s) / 2, "macro_f1_reported": e["models"][m]["macro_f1"],
                "invalid": invalid, "auc_recomputed": auc(sc, gold), "auc_reported": e["models"][m]["auc"],
                "wilson95": wilson(acc, len(gold))}
    def disc(a, b):
        pa, pb = preds[a], preds[b]
        b_ = sum(x == g and y != g for x, y, g in zip(pa, pb, gold))  # a right, b wrong
        c_ = sum(x != g and y == g for x, y, g in zip(pa, pb, gold))
        changed = sum(x != y for x, y in zip(pa, pb))
        return {"a_right_b_wrong": b_, "a_wrong_b_right": c_, "pred_changed": changed,
                "mcnemar_exact_p": mcnemar(b_, c_)}
    o["base_vs_trained"] = disc("base", "trained")
    o["trained_vs_merged"] = disc("trained", "merged")
    o["base_vs_control"] = disc("base", "control")
    # paired bootstrap of accuracy difference (independent re-implementation)
    rng = random.Random(12345)
    cb = [int(x == g) for x, g in zip(preds["base"], gold)]
    ct = [int(x == g) for x, g in zip(preds["trained"], gold)]
    n = len(gold)
    diffs = []
    for _ in range(10000):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(ct[i] - cb[i] for i in idx) / n)
    diffs.sort()
    o["paired_bootstrap_trained_minus_base_recomputed"] = {
        "diff": (sum(ct) - sum(cb)) / n, "ci95": [diffs[249], diffs[9749]]}
    o["paired_reported"] = e["paired"]
    o["paired_auc_reported"] = e["paired_auc"]
    # merged vs trained score differences
    st_, sm = e["label_scores"]["trained"], e["label_scores"]["merged"]
    d = [abs(a - b) for a, b in zip(st_, sm)]
    o["merged_vs_trained_score_absdiff"] = {"max": max(d), "mean": sum(d) / len(d),
                                           "n_exact_equal": sum(x == 0 for x in d),
                                           "n_sign_flip": sum((a > 0) != (b > 0) for a, b in zip(st_, sm))}
    sb, sc_ = e["label_scores"]["base"], e["label_scores"]["control"]
    o["control_vs_base_score_absdiff_max"] = max(abs(a - b) for a, b in zip(sb, sc_))
    o["trained_vs_base_score_absdiff"] = {"max": max(abs(a - b) for a, b in zip(st_, sb)),
                                         "mean": sum(abs(a - b) for a, b in zip(st_, sb)) / n}
    # minimum detectable effect, rough: with discordant count D, 95% two-sided needs |b-c| ~ 1.96*sqrt(D)
    D = o["base_vs_trained"]["a_right_b_wrong"] + o["base_vs_trained"]["a_wrong_b_right"]
    o["approx_mde_accuracy_given_discordance"] = 1.96 * math.sqrt(D) / n
    o["meta"] = {k: e[k] for k in ["checkpoint", "decoding", "execution", "backends", "merge",
                                  "lora_change", "data_separation", "peak_gpu_allocated_bytes_eval"]}
    # data separation re-check
    tr = {x["text"].strip().lower() for x in data["train"]}
    va = {x["text"].strip().lower() for x in data["validation"]}
    te = {x["text"].strip().lower() for x in data["test"]}
    o["overlap_train_test"] = len(tr & te)
    o["overlap_train_val"] = len(tr & va)
    o["overlap_val_test"] = len(va & te)
    o["dup_within_test"] = len(data["test"]) - len(te)
    for k, v in o.items():
        print(f"{k}: {v}")
    return o


def auc(scores, gold):
    pos = [s for s, g in zip(scores, gold) if g == "POSITIVE"]
    neg = [s for s, g in zip(scores, gold) if g == "NEGATIVE"]
    allv = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    # rank-based (Mann-Whitney) with average ranks for ties
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2
        for k in range(i, j):
            ranks.setdefault(k, r)
        i = j
    rp = sum(ranks[k] for k, (s, y) in enumerate(allv) if y == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def wilson(p, n, z=1.96):
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


if __name__ == "__main__":
    OUT["trained"], _t = training(RUN, "trained")
    OUT["control"], _c = training(CTL, "lr0 control")
    OUT["memory_trained"] = memory(RUN, "trained", _t)
    OUT["memory_control"] = memory(CTL, "lr0 control", _c)
    OUT["evaluation"] = evaluation()
    (Path(__file__).parent / "analysis_output.json").write_text(json.dumps(OUT, indent=1, default=str))
