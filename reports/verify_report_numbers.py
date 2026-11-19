"""Recomputes the numbers quoted in reports/Soup_Final_Research_Report.pdf from the saved artifacts and asserts them.

Read-only, CPU only, standard library only. Run from the repository root:
    python3 -I reports/verify_report_numbers.py
"""
import bisect
import csv
import json
import math
import random
from pathlib import Path

R = Path.cwd()
RUN = R / "runs/qwen3_4b_final_plain_150_v2/training"
CTL = R / "runs/qwen3_4b_final_plain_150_v2_lr0/training"
T13 = R / "research/l2l/experiments/t13_final_eval_qwen3_4b_final_plain_150_v2.json"
T11 = R / "assignment_results/manual/profile_20261119T054235Z/results/t11_profile_manual.json"
AR = R / "assignment_results/final150_v2"
MANUAL = R / "assignment_results/manual"
LABELS = ("NEGATIVE", "POSITIVE")
checks = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    checks.append(ok)
    print(f"{'OK ' if ok else 'BAD'} {name}: {got!r} (report: {want!r})")


# --- training run ---------------------------------------------------------------------------------------------------
t = json.loads((RUN / "training.json").read_text())
check("successful updates", t["successful_steps"], 150)
check("rollout batches", t["groups_generated"], 672)
check("zero-signal batches", len(t["zero_signal"]), 522)
check("prompt groups", t["prompt_groups"], 3360)
check("completions", t["responses_generated"], 20160)
check("train seconds", round(t["train_seconds"], 1), 3210.9)
check("updates per minute", round(150 / (t["train_seconds"] / 60), 2), 2.80)
check("frozen tensors changed", len(t["frozen_changed"]), 0)
check("final adapter reloads bitwise", t["final_checkpoint"]["reloaded_bitwise_equal"], True)
cfg = t["config"]
check("B, G, micro-batch", (cfg["B"], cfg["G"], cfg["micro_batch_size"]), (5, 6, 6))
check("lr, clip, iterations, kl", (cfg["lr"], cfg["clip_eps"], cfg["num_iterations"], cfg["kl_beta"]), (2e-4, 0.2, 1, 0.0))
tr = t["training"]
check("max |loss|", round(max(abs(x["loss"]) for x in tr), 9), 3.2e-8, 1e-9)
check("max init_diff (old vs current log-prob)", max(x["init_diff"] for x in tr), 0.0)
check("ratio always 1, clip 0", (min(x["ratio_min"] for x in tr), max(x["ratio_max"] for x in tr),
                                 max(x["clip_frac"] for x in tr)), (1.0, 1.0, 0.0))
check("||theta - theta0|| final", round(tr[-1]["change_from_init"], 2), 2.00)
check("logged reward, first 10 steps", round(sum(x["mean_reward"] for x in tr[:10]) / 10, 2), 0.86)
check("logged reward, last 10 steps", round(sum(x["mean_reward"] for x in tr[-10:]) / 10, 2), 0.73)
recs = sorted(tr + t["zero_signal"], key=lambda x: x["attempt"])
flat = lambda r: [y for g in r["rewards"] for y in g]
bins = []
for i in range(0, len(recs), 50):
    rr = [y for r in recs[i:i + 50] for y in flat(r)]
    bins.append(sum(rr) / len(rr))
check("all-rollout reward bins within [0.855, 0.917]", all(0.855 <= b <= 0.917 for b in bins), True)
groups = [g for r in recs for g in r["rewards"]]
mixed = sum(0 < sum(g) < len(g) for g in groups)
check("groups mixed / all-correct / all-wrong",
      (mixed, sum(sum(g) == len(g) for g in groups), sum(sum(g) == 0 for g in groups)), (163, 2905, 292))
check("(1 - p_mixed)^5", round((1 - mixed / len(groups)) ** 5, 2), 0.78)
tm = t["timings"]
S = sum(x["step_s"] for x in tm)
check("mean step s", round(S / len(tm), 2), 9.55)
check("rollout / old-logp / update share %", tuple(round(100 * sum(x[k] for x in tm) / S, 1)
                                                for k in ("rollout_s", "prepare_s", "update_s")), (23.5, 36.8, 39.4))
sweep = json.loads(T11.read_text())["streamed_bytes_per_pass"]
check("bytes per sweep", sweep, 7_267_018_752)
check("sweeps per step", sorted({x["weight_bytes_copied"] // sweep for x in tm}), [13, 14])
bw = json.loads(T11.read_text())["h2d_bandwidth_bytes_per_s"]
check("H2D GB/s", round(bw / 1e9, 2), 7.25)
check("14-sweep transfer time s", round(14 * sweep / bw, 2), 14.03)
check("time outside timed steps %", round(100 * (t["train_seconds"] - S) / t["train_seconds"]), 55)

# --- memory ---------------------------------------------------------------------------------------------------------
for d, label in ((RUN, "trained"), (CTL, "control")):
    m = json.loads(next((d / "memory").glob("*.json")).read_text())
    check(f"{label}: peak per-process B", int(m["peak"]["process_used"]), 2_864_709_632)
    check(f"{label}: violations", len(m["violations"]), 0)
    check(f"{label}: guard budget B (8e9)", m["budget_bytes"], 8_000_000_000)
m = json.loads(next((RUN / "memory").glob("*.json")).read_text())
check("peak device B", m["peak"]["device_used"], 2_898_395_136)
check("peak reserved B", m["peak"]["reserved"], 2_690_646_016)
check("peak allocated B", m["peak"]["allocated"], 1_806_041_088)
check("samples", len(m["samples"]), 5947)
check("samples without per-process value", sum(not isinstance(s["process_used"], (int, float)) for s in m["samples"]), 0)
gaps = sorted(b["t"] - a["t"] for a, b in zip(m["samples"], m["samples"][1:]))
check("median sample gap s", round(gaps[len(gaps) // 2], 2), 0.56)
check("peak / 8e9 %", round(100 * m["peak"]["process_used"] / 8e9, 1), 35.8)
check("process - reserved GB", round((m["peak"]["process_used"] - m["peak"]["reserved"]) / 1e9, 2), 0.17)
check("host VmHWM GiB", round(max(s["VmHWM"] for s in m["samples"]) / 2**30, 2), 14.39)
merge = [p for p in json.loads(T11.read_text())["profile"]["phases"] if p["phase"] == "merge"][0]
check("merge RSS peak GB", round(merge["cpu_rss_peak"] / 1e9, 1), 17.5)
phases = json.loads(T11.read_text())["profile"]["phases"]
before = [p["cpu_anon_peak"] for p in phases[:phases.index(merge)] if p["cpu_anon_peak"] is not None][-1]
check("merge anonymous growth GB (vs last sampled phase before it)", round((merge["cpu_anon_peak"] - before) / 1e9, 2), 0.32)

# --- held-out evaluation --------------------------------------------------------------------------------------------
e = json.loads(T13.read_text())
gold = [LABELS[r["label"]] if isinstance(r["label"], int) else r["label"]
        for r in json.loads((R / "data/sentiment.json").read_text())["test"]]
check("n test / POSITIVE", (len(gold), gold.count("POSITIVE")), (1066, 533))
P, Sc = e["predictions"], e["label_scores"]
norm = lambda p: p.strip().upper() if p.strip().upper() in LABELS else None
acc = {k: sum(norm(p) == g for p, g in zip(P[k], gold)) / len(gold) for k in P}


def f1(k):
    out = []
    for c in LABELS:
        tp = sum(norm(p) == c and g == c for p, g in zip(P[k], gold))
        fp = sum(norm(p) == c and g != c for p, g in zip(P[k], gold))
        fn = sum(norm(p) != c and g == c for p, g in zip(P[k], gold))
        out.append(2 * tp / (2 * tp + fp + fn))
    return sum(out) / 2


def auc(s):
    neg = sorted(x for x, g in zip(s, gold) if g == "NEGATIVE")
    pos = [x for x, g in zip(s, gold) if g == "POSITIVE"]
    w = sum(bisect.bisect_left(neg, x) + 0.5 * (bisect.bisect_right(neg, x) - bisect.bisect_left(neg, x)) for x in pos)
    return w / len(pos) / len(neg)


want = {"base": (0.8771, 0.8787, 0.9511, 5, 0.436), "trained": (0.8865, 0.8869, 0.9523, 1, 0.489),
        "merged": (0.8865, 0.8869, 0.9523, 1, 0.489), "control": (0.8771, 0.8787, 0.9511, 5, 0.436)}
for k, w in want.items():
    got = (round(acc[k], 4), round(f1(k), 4), round(auc(Sc[k]), 4), sum(norm(p) is None for p in P[k]),
           round(sum(norm(p) == "POSITIVE" for p in P[k]) / len(gold), 3))
    check(f"{k}: acc, F1, AUC, invalid, POS rate", got, w)
cb = [norm(p) == g for p, g in zip(P["base"], gold)]
ct = [norm(p) == g for p, g in zip(P["trained"], gold)]
b, c = sum(x and not y for x, y in zip(cb, ct)), sum(y and not x for x, y in zip(cb, ct))
check("discordant base-only / trained-only", (b, c), (22, 32))
p = min(1.0, 2 * sum(math.comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c))
check("exact McNemar p", round(p, 2), 0.22)
pr = e["paired"]["trained_minus_base"]
check("saved paired bootstrap diff, CI (pp)", (round(100 * pr["diff"], 3), round(100 * pr["ci_low"], 3),
                                               round(100 * pr["ci_high"], 3)), (0.938, -0.375, 2.251))
rng, d = random.Random(0), [y - x for x, y in zip(cb, ct)]
bs = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(2000))
check("independent bootstrap CI contains 0 (2000 resamples)", bs[50] < 0 < bs[1949], True)
pa = e["paired_auc"]["trained_minus_base"]
check("saved AUC diff", round(pa["auc_diff"], 4), 0.0012)
check("AUC CI (excludes 0)", (round(pa["ci"][0], 4), round(pa["ci"][1], 4)), (0.0002, 0.0022))
sh = [y - x for x, y in zip(Sc["base"], Sc["trained"])]
mu = sum(sh) / len(sh)
check("mean score shift nats", round(mu, 2), 6.01)
check("SD score shift", round((sum((x - mu) ** 2 for x in sh) / len(sh)) ** 0.5, 2), 0.88)


def ranks(v):
    o = sorted(range(len(v)), key=v.__getitem__)
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


ra, rb, mr = ranks(Sc["base"]), ranks(Sc["trained"]), (len(gold) - 1) / 2
check("Spearman base vs trained", round(sum((x - mr) * (y - mr) for x, y in zip(ra, rb))
                                        / sum((x - mr) ** 2 for x in ra), 4), 0.9985)
check("changed predictions", sum(x != y for x, y in zip(P["base"], P["trained"])), 56)
dm = sorted(abs(x - y) for x, y in zip(Sc["merged"], Sc["trained"]))
check("merged = trained predictions", sum(x == y for x, y in zip(P["merged"], P["trained"])), 1066)
check("merge |ds| median", dm[len(dm) // 2] < 2e-6, True)
check("merge |ds| max", round(dm[-1], 2), 0.38)
check("control identical to base", (P["control"] == P["base"], max(abs(x - y) for x, y in zip(Sc["control"], Sc["base"]))),
      (True, 0.0))
check("recall POS/NEG trained", (round(e["models"]["trained"]["recall_POSITIVE"], 3),
                                 round(e["models"]["trained"]["recall_NEGATIVE"], 3)), (0.876, 0.897))
check("eval allocator peak per model B (t13 phases; the top-level field is reset per pass)",
      {p["name"]: p["peak_allocated_bytes"] for p in e["phases"]},
      {"trained": 1_567_870_464, "base": 1_567_870_464, "merged": 1_563_545_088, "control": 1_567_870_464})

# --- equivalence (async L2L vs resident on the RTX 3060; evidence/<ts>_<item>/results.json) ------------------------
import re
ev = {}
for f in sorted(R.glob("evidence/20261119T05*Z_H0*.lab_async_cuda/results.json")):
    r = json.loads(f.read_text())
    ev[r["entry"].split(".")[0]] = (int(re.search(r"all (\d+) comparisons", r["detail"]).group(1)), r["status"])
check("H01 comparisons, status", ev["H01"], (2690, "PASS"))
check("H02 comparisons, status", ev["H02"], (5296, "PASS"))
check("H04 comparisons, status", ev["H04"], (1292, "PASS"))
check("H05 comparisons, status", ev["H05"], (6528, "PASS"))
c06 = json.loads((R / "research/l2l/experiments/compare_qwen_resident_vs_qwen_l2l.json").read_text())
check("0.6B resident vs sync L2L rows equal", (c06["training_rows"]["equal"], c06["training_rows"]["steps"]), (True, [100, 100]))
state = json.loads((AR / "state.json").read_text())
check("staged run: stage status (last attempt)", {k: v[-1]["status"] for k, v in state["stages"].items()},
      {"preflight": "PASS", "correctness": "FAIL", "smoke": "BLOCKED", "profile": "BLOCKED", "soup": "PASS"})
corr = {i["id"]: i["status"] for i in state["stages"]["correctness"][-1]["items"]}
check("correctness items failing", sorted(k for k, v in corr.items() if v != "PASS"), ["pytest_cpu"])
kill = json.loads((R / "research/l2l/experiments/t09_resume_kill_qwen3_4b_resume_kill_r20261119T073842Z.json").read_text())
check("real-model SIGKILL resume", (kill["pass"], kill["kill"]["signal"], kill["kill"]["checkpoints_at_kill"]), (True, "SIGKILL", [0, 2]))
want_stage = {"smoke": 2362.0, "profile": 2106.0, "train": 2732.0, "control": 2732.0, "merge": None, "eval": 1912.0}
for g in sorted(MANUAL.glob("*/guarded_run.json")):
    r = json.loads(g.read_text())
    stage = g.parent.name.split("_")[0]
    check(f"guarded stage {stage}: exit, violation, nvidia-smi process peak MiB",
          (r["exit"], r["violation"], r["telemetry"]["process_peak_mib"]), (0, None, want_stage[stage]))

# --- claims added in the five-section revision -----------------------------------------------------------------------
import hashlib
import xml.etree.ElementTree as ET
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ck = lambda r, k: R / f"runs/{r}/checkpoints/step{k}/adapter_model.safetensors"
check("control adapter step30 == its step0 == trained step0 (sha256)",
      sha(ck("qwen3_4b_final_plain_150_v2_lr0", 30)) == sha(ck("qwen3_4b_final_plain_150_v2_lr0", 0))
      == sha(ck("qwen3_4b_final_plain_150_v2", 0)), True)
check("trained adapter step150 differs from init", sha(ck("qwen3_4b_final_plain_150_v2", 150)) != sha(ck("qwen3_4b_final_plain_150_v2", 0)), True)
dd = [abs(x - y) for x, y in zip(Sc["merged"], Sc["trained"])]
check("merge: scores within 4e-5 / exactly equal", (sum(x < 4e-5 for x in dd), sum(x == 0 for x in dd)), (633, 400))
big = [x for x in dd if x >= 4e-5]
check("merge: other diffs are multiples of 1/16 in [0.0625, 0.375]",
      (len(big) == 433 and all(abs(x * 16 - round(x * 16)) < 1e-2 for x in big), round(min(big), 4), round(max(big), 4)), (True, 0.0625, 0.375))
check("max |trained score|", round(max(abs(x) for x in Sc["trained"])), 26)
mg = json.loads(next(MANUAL.glob("merge_*/guarded_run.json")).read_text())
check("merge stage max RSS B", mg["children_max_rss_bytes"], 8_602_054_656)
check("discordant MDE pp", round(100 * 1.96 * math.sqrt(b + c) / len(gold), 2), 1.35)
check("mean rollout / old-logp / update s", tuple(round(sum(x[k] for x in tm) / len(tm), 2) for k in ("rollout_s", "prepare_s", "update_s")), (2.24, 3.52, 3.76))
soup = json.loads((R / "research/soup/0.78.0_20261119T073222Z/summary.json").read_text())
sc = {c["name"]: c for c in soup["checks"]}
rs = sc["reward_stress"]["report"]["attacks"]
check("soup reward stress: attacks, accepted", (sum(a["n"] for a in rs), sum(a["accepted"] for a in rs)), (4600, 0))
check("soup permissive verifier verdict", sc["reward_stress_permissive"]["verdict"], "gameable")
check("soup ship verdicts", (sc["ship_t13_150_update"]["verdict"], sc["ship_t13_150_update_lr0_control"]["verdict"]), ("DON'T SHIP", "DON'T SHIP"))
check("soup data doctor / adapters check runs that exited 0",
      sum(c.get("exit") == 0 for n, c in sc.items() if n.startswith(("data_doctor", "adapters_check"))), 0)
check("soup pre-flight not run", sorted(n for n, c in sc.items() if c.get("status") == "NOT_RUN"), ["advise", "plan", "train hardware-fit"])
ts = ET.parse(next(AR.glob("correctness/*/pytest_cpu.xml"))).getroot().find("testsuite")
check("pytest on host: tests, failures", (int(ts.get("tests")), int(ts.get("failures"))), (360, 2))

print(f"\n{sum(checks)}/{len(checks)} checks passed")
raise SystemExit(0 if all(checks) else 1)
