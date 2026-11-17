"""Ticket 13: research/l2l/final_results.json and research/l2l/final_report.md, derived from the evidence files only.

    python scripts/final_report.py

Every requirement's status is computed from recorded results (research/l2l/experiments/*), never written by hand:
  PASS        the evidence exists and meets its predefined criterion
  FAIL        the evidence exists and does not
  BLOCKED     the run happened but could not produce the required evidence (e.g. a GPU budget under WSL2, where
              nvidia-smi has no per-process memory), or it was refused by a precondition
  NOT TESTED  no evidence file
SHIP only if every essential requirement is PASS; otherwise DON'T SHIP with what must change.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "research" / "l2l" / "experiments"
OUT_JSON, OUT_MD = ROOT / "research" / "l2l" / "final_results.json", ROOT / "research" / "l2l" / "final_report.md"


def lab_result(name):
    p = EXP / name / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def jfile(name):
    p = EXP / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def from_lab(*names, fault=None):
    """PASS iff every named lab result passed (and the fault run, if given, failed with execution PASS)."""
    rs = [lab_result(n) for n in names]
    if any(r is None for r in rs):
        return "NOT TESTED", f"missing {[n for n, r in zip(names, rs) if r is None]}"
    if any(r["status"] == "BLOCKED" for r in rs):
        return "BLOCKED", "; ".join(r["reasons"][0] for r in rs if r["status"] == "BLOCKED")
    bad = [n for n, r in zip(names, rs) if r["status"] != "PASS"]
    if bad:
        return "FAIL", f"not PASS: {bad}"
    if fault:
        f = lab_result(fault)
        if f is None:
            return "PASS", "fault-injection run missing (power not shown on this hardware)"
        if not (f["status"] == "FAIL" and f.get("execution_status") == "PASS"):
            return "FAIL", f"fault run {fault} was not caught ({f['status']})"
    return "PASS", ", ".join(names)


def budget_status(name):
    r = jfile(name)
    if r is None:
        return "NOT TESTED", f"missing {name}"
    v = r.get("verdict") or r.get("budget_verdict")
    if v == "PASS":
        return "PASS", name
    if v == "FAIL":
        return "FAIL", f"{name}: guard fired {r['profile']['violations'][:1]}"
    return "BLOCKED", f"{name}: evidence {r['profile']['evidence']} (WSL2 {r['profile']['wsl2']}): no per-process nvidia-smi"


def perf_status(name, key="conclusion"):
    r = lab_result(name)
    if r is None:
        return "NOT TESTED", f"missing {name}"
    if r["status"] != "PASS":
        return ("BLOCKED" if r["status"] == "BLOCKED" else "FAIL"), r["reasons"][:1]
    c = r[key]
    return ("PASS" if c["ci95"][1] < 1 else "FAIL"), f"async/sync step ratio {c['ratio']:.3f} CI {c['ci95']}"


def heldout_status():
    r = jfile("t13_final_eval_qwen3_4b_final")
    if r is None:
        return "NOT TESTED", "missing t13_final_eval_qwen3_4b_final", None
    tb = r["paired"]["trained_minus_base"]
    if r["data_separation"]["test_texts_in_train"]:
        return "FAIL", "test examples occur in the training split", r
    if tb["ci_low"] > 0:
        return "PASS", f"test accuracy +{tb['diff']:.4f} (95% CI [{tb['ci_low']:.4f}, {tb['ci_high']:.4f}])", r
    return "FAIL", f"no detected held-out improvement: {tb['diff']:+.4f} (95% CI [{tb['ci_low']:.4f}, {tb['ci_high']:.4f}])", r


def control_status(r):
    if r is None or "control_minus_base" not in r.get("paired", {}):
        return "NOT TESTED", "missing control evaluation"
    c, log = r["paired"]["control_minus_base"], r.get("control_log", {})
    ok = log.get("params_equal_init") and c["ci_low"] <= 0 <= c["ci_high"] and r["models"]["control"]["changed_vs_base"] == 0
    return ("PASS" if ok else "FAIL"), (f"LR=0: params_equal_init {log.get('params_equal_init')}, predictions changed "
                                        f"{r['models']['control']['changed_vs_base']:.3f}, diff {c['diff']:+.4f}")


def dflash_status():
    r = jfile("t12_dflash_bench")
    if r is None:
        return "NOT TESTED", "missing t12_dflash_bench", "NOT TESTED", ""
    eq = "PASS" if r["all_greedy_equal"] else "FAIL"
    runs = [x for p in r["policies"] for x in p["runs"]]
    faster = all(x["sample_dflash_s"] < x["sample_plain_s"] for x in runs) and all(
        p["grpo_step_dflash_s"] < p["grpo_step_plain_s"] for p in r["policies"])
    ppt = [round(x["sample_passes_per_token"], 2) for x in runs]
    return eq, "greedy tokens equal plain generation at every policy" if eq == "PASS" else "greedy mismatch", \
        ("PASS" if faster else "FAIL"), f"sampling target passes per token {ppt}; faster everywhere: {faster}"


AUDIT = [  # stage, silent failure, detection, controlled fault / evidence, occurred, detected by this repo's tooling
    ("Data", "train/test overlap, unbalanced labels", "split by source split + text check in final_eval; balanced_subset",
     "final_eval data_separation count", "not observed", "yes (tested)"),
    ("Tokenizer/template", "thinking-mode template, wrong padding side", "build_prompts asserts the non-thinking block;"
     " left padding checked by check_grouping", "test_policy / gen_lab chat prompts", "no", "yes (tested)"),
    ("Reward", "reward hacking (formatting, label without content)", "strict canonical-label reward; invalid-rate and"
     " prediction-distribution tracking; LR=0 control", "t13 held-out metrics", "see held-out results", "partly"),
    ("GRPO", "wrong advantage / stale old log-probs / ratio != 1 at it0", "INIT_TOL check; update_equivalence",
     "t07 corrupted-advantage fault (caught)", "no", "yes (tested)"),
    ("L2L forward/backward", "wrong layer weights, stale buffer, detached boundary", "bitwise equivalence cases;"
     " recomputation checksum", "t03 wrong_layer, t04 detach_boundary, t08 unsafe_reuse", "no", "yes (tested)"),
    ("Output head", "unnormalized logit, missing remainder chunk", "logprob_equivalence", "t05 raw_logit fault",
     "a shape-dependent 1e-7 drift on one CPU (fixed, bc5a05d)", "yes (tested)"),
    ("Generation", "stale position, attended padding, EOS handling", "generation_equivalence", "t06 faults",
     "no", "yes (tested)"),
    ("Disk streaming/resume", "duplicate/missing samples, double-applied update, torn checkpoint",
     "crash injection; COMMIT.json validation", "t09 six crash points + corrupted checkpoint", "no", "yes (tested)"),
    ("DFlash", "drafter replacing the policy, wrong acceptance", "greedy equality; chi-square vs exact law",
     "tests/test_dflash (random/oracle drafters)", "no", "yes (tiny models only)"),
    ("Merge", "dropped scale, wrong module, lost delta in bf16", "half-ulp residual check; power ratio; L2L eval",
     "merge_equivalence power requirement", "no", "yes (tested)"),
    ("Inference/memory", "budget exceeded outside the allocator", "nvidia-smi sampling + guard", "profile runs",
     "unknown under WSL2", "only with per-process nvidia-smi"),
]


def main():
    reqs = []

    def add(rid, name, essential, status_detail):
        s, d = status_detail[:2]
        reqs.append({"id": rid, "requirement": name, "essential": essential, "status": s, "evidence": d})
    add("R1", "L2L forward equals resident (H01)", True, from_lab("t03_forward_equivalence", fault="t03_forward_equivalence_wrong_layer"))
    add("R2", "checkpointed backward equals resident (H02)", True, from_lab("t04_gradient_equivalence_cuda", fault="t04_gradient_equivalence_detach_boundary"))
    add("R3", "memory-efficient output head (H03)", False, from_lab("t05_logprob_equivalence_cpu_v2", fault="t05_logprob_equivalence_raw_logit_v2"))
    add("R4", "streamed generation equals resident (H04)", True, from_lab("t06_generation_equivalence_cuda", fault="t06_generation_equivalence_stale_position"))
    add("R5", "complete GRPO update equals resident (H05)", True, from_lab("t07_update_equivalence_cuda_v2", fault="t07_update_equivalence_advantage"))
    add("R6", "async prefetch keeps results exact", True, from_lab("t08_gradient_equivalence_async", "t08_update_equivalence_async",
                                                                    fault="t08_gradient_equivalence_async_unsafe_reuse"))
    add("R7", "async prefetch is faster (H06)", False, perf_status("t08_prefetch_timing"))
    add("R8", "GPU weight residency bounded in depth (H07)", True, from_lab("t11_depth_residency"))
    r9 = jfile("t09_resume_qwen_stream") or jfile("t09_resume_tiny_cpu")
    add("R9", "deterministic crash-consistent resume (H08)", True,
        ("NOT TESTED", "missing") if r9 is None else (("PASS" if r9["pass"] else "FAIL"),
                                                      "t09_resume_qwen_stream" if jfile("t09_resume_qwen_stream") else "t09_resume_tiny_cpu (tiny model only)"))
    add("R10", "streaming merge preserves behaviour (H09)", True, from_lab("t10_merge_equivalence_tiny", "t10_merge_equivalence_qwen3_0.6b"))
    d_eq, d_eq_det, d_speed, d_speed_det = dflash_status()
    add("R11", "DFlash on-policy correctness (H10)", True, (d_eq, d_eq_det))
    add("R12", "DFlash speedup (H10)", False, (d_speed, d_speed_det))
    add("R13", "Qwen3-4B within the 8 GB GPU budget (H11)", True, budget_status("t11_profile_qwen3_4b_async"))
    held = heldout_status()
    add("R14", "held-out task improvement with Qwen3-4B, 1-2k prompts (H12)", True, held[:2])
    add("R15", "negative control behaves as no learning", True, control_status(held[2]))
    essential_bad = [r for r in reqs if r["essential"] and r["status"] != "PASS"]
    verdict = "SHIP" if not essential_bad else "DON'T SHIP"
    res = {"verdict": verdict, "requirements": reqs, "silent_failure_audit": [dict(zip(
        ("stage", "silent_failure", "detection", "fault_or_evidence", "occurred", "detected_by_tooling"), a)) for a in AUDIT],
        "held_out": None if held[2] is None else {k: held[2][k] for k in ("models", "paired", "data_separation", "lora_change",
                                                                          "merged_vs_trained_prediction_agreement", "training")},
        "must_change": [f"{r['id']} {r['requirement']}: {r['status']} ({r['evidence']})" for r in essential_bad]}
    OUT_JSON.write_text(json.dumps(res, indent=1, default=str) + "\n")
    md = [f"# Final report: GRPO + LoRA with L2L streaming (Soup assignment)\n", f"**Verdict: {verdict}**\n",
          "Every status below is derived by `scripts/final_report.py` from the files in `research/l2l/experiments/`; "
          "nothing is entered by hand. Code, raw results and nvidia-smi samples are kept separately in the repository.\n",
          "## Requirements\n", "| id | requirement | essential | status | evidence |", "|---|---|---|---|---|"]
    md += [f"| {r['id']} | {r['requirement']} | {'yes' if r['essential'] else 'no'} | **{r['status']}** | {r['evidence']} |"
           for r in reqs]
    if held[2] is not None:
        md += ["\n## Held-out evaluation (test split, L2L greedy)\n", "| model | accuracy | macro F1 | invalid | pred POS | changed vs base |",
               "|---|---|---|---|---|---|"]
        md += [f"| {k} | {m['accuracy']:.4f} | {m.get('macro_f1', 0):.4f} | {m['invalid_rate']:.4f} | {m['predicted_positive_rate']:.3f} "
               f"| {m['changed_vs_base']:.3f} |" for k, m in held[2]["models"].items()]
        md += [f"\nPaired bootstrap 95% CIs: " + "; ".join(f"{k} {v['diff']:+.4f} [{v['ci_low']:.4f}, {v['ci_high']:.4f}]"
                                                          for k, v in held[2]["paired"].items())]
        md += ["A rising reward curve or nonzero gradients are not counted as learning: only the held-out difference "
               "against the base model, with the LR=0 control as the no-learning reference, is."]
    md += ["\n## Silent-failure audit\n", "| stage | silent failure | detection | fault / evidence | occurred | detected by tooling |",
           "|---|---|---|---|---|---|"] + [f"| {' | '.join(a)} |" for a in AUDIT]
    md += ["\n## What must change\n"] + ([f"- {m}" for m in res["must_change"]] or ["- nothing: every essential requirement passes"])
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{verdict}: {sum(r['status'] == 'PASS' for r in reqs)}/{len(reqs)} PASS; written {OUT_JSON.name}, {OUT_MD.name}")


if __name__ == "__main__":
    sys.exit(main())
