"""Compare two finished training runs under runs/ (e.g. resident vs L2L of the same config) and report every difference.

    python scripts/compare_runs.py qwen3_0.6b_resident qwen3_0.6b_l2l_sync
    python scripts/compare_runs.py kaggle_06b_fp32_resident kaggle_06b_fp32_l2l_async --out evidence/<id>/compare.json

Run A is the reference. Two separate parts:

NUMERICAL EQUIVALENCE (the verdict). The rule is research/l2l/tolerances.yaml update_equivalence for B's backend:
every threshold there is 0 (l2l_sync, and l2l_async by same_as), so every quantity must be bitwise equal; the rule
read from the file is recorded with the result. Compared exactly:
  configs            the resolved configs may differ only in run.*, execution.* and memory.* (else not comparable)
  training rows      per optimizer step, every recorded field except timings and memory: prompt ids, rollout hash
                     (sha256 of the generated token ids) and decoded responses, rewards, labels, invalid responses,
                     loss, clipping statistics (clip_frac, ratio min/mean/max, outside_frac, approx_kl), gradient norms
                     (all, A, B), update norm, adapter norms, change from init, init_diff (iteration-0 old vs current)
  zero-signal        the rollout hashes of skipped batches
  rollout records    streaming runs only: the sha256 of every group's input ids, response ids, rewards, advantages
                     and old log-probs (qwen_grpo/streaming/rollouts.py)
  evaluations        validation-subset predictions at every evaluation step
  checkpoints        every common checkpoint: LoRA tensors and AdamW state (bitwise, else max abs difference)
  final validation   predictions and label scores, when both runs wrote training/validation.json
  controls           frozen weights unchanged in both runs
Not recorded by the trainer, so not compared (coverage lists them): per-token current-policy log-probs and per-step
gradient tensors; for non-streaming runs also advantages and old log-probs (functions of the compared rewards and of
the compared parameters, but not compared directly).

PERFORMANCE (never part of the verdict): train time, time per step, per-step phase timings, allocator peaks and
memory-guard samples of each run. One run each: indicative, not a benchmark (H06 is the timing experiment).

Writes research/l2l/experiments/compare_<A>_vs_<B>.json (or --out); exit status 0 iff comparable and identical.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402
from peft import load_peft_weights  # noqa: E402

from qwen_grpo import checkpointing, config, runs  # noqa: E402

RESOURCE_KEYS = {"step_seconds", "cuda_allocated_gib", "peak_allocated_gib", "peak_reserved_gib", "timings"}
COMPARABLE_PREFIXES = ("run.", "execution.", "memory.")  # where the configs of a backend comparison may differ
COVERAGE = {
    "generated rollout tokens": "rollout_sha256 (sha256 of the response token ids) and decoded responses, every step",
    "rollout hashes": "every step and every zero-signal batch",
    "rewards": "every step",
    "advantages": "streaming runs: rollout records (exact); otherwise not recorded (a function of the compared rewards)",
    "old-policy log-probs": "streaming runs: rollout records (exact); otherwise not recorded",
    "current-policy log-probs": "not recorded per token; ratio min/mean/max, outside_frac and approx_kl (functions of "
                                "current and old log-probs) and init_diff are compared every step",
    "GRPO loss": "every step",
    "clipping statistics": "clip_frac, ratio_min/mean/max, outside_frac, approx_kl, every step",
    "LoRA gradients": "norms every step (grad_norm, grad_norm_A, grad_norm_B); tensors are not saved",
    "LoRA parameter updates": "update_norm and change_from_init every step; LoRA tensors bitwise at every checkpoint",
    "AdamW optimizer state": "bitwise at every common checkpoint",
    "frozen-weight integrity": "frozen_changed of both runs (digest of every frozen tensor before and after)",
    "validation predictions": "subset predictions at every evaluation; final validation predictions and label scores",
    "memory and runtime": "PERFORMANCE section only (never part of the equivalence verdict)",
}


def load_log(name):
    path = runs.training_dir(name) / "training.json"
    if not path.exists():
        raise SystemExit(f"no training log for run {name!r}: {path}")
    return json.loads(path.read_text())


def tolerance_rule(backend):
    """The update_equivalence thresholds for `backend` from tolerances.yaml, and whether they mean bitwise equality."""
    tol = yaml.safe_load((ROOT / "research" / "l2l" / "tolerances.yaml").read_text())["update_equivalence"]
    if backend not in tol:  # resident vs resident: the same code twice
        return {"source": "none (same backend on both sides)", "exact": True}
    entry = tol[backend]
    key = entry.get("same_as", backend)
    tables = tol[key]
    zero = all(v.get("atol") == 0 and v.get("rtol") == 0 for dt in tables.values() if isinstance(dt, dict)
               for v in dt.get("quantities", {}).values())
    return {"source": f"research/l2l/tolerances.yaml update_equivalence.{backend}" + (f" (same_as {key})" if key != backend else ""),
            "exact": zero}


def tensor_diff(a, b):
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return {"equal": False, "why": f"{a.dtype} {list(a.shape)} vs {b.dtype} {list(b.shape)}"}
        eq = bool(torch.equal(a, b))
        return {"equal": eq, "max_abs": 0.0 if eq else (a.double() - b.double()).abs().max().item()}
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return {"equal": False, "why": f"keys differ: {sorted(set(a) ^ set(b))[:5]}"}
        parts = {k: tensor_diff(a[k], b[k]) for k in a}
        bad = {k: v for k, v in parts.items() if not v["equal"]}
        return {"equal": not bad, "differing": dict(list(bad.items())[:10]), "n_differing": len(bad)}
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return {"equal": False, "why": f"lengths {len(a)} vs {len(b)}"}
        parts = [tensor_diff(x, y) for x, y in zip(a, b)]
        return {"equal": all(p["equal"] for p in parts)}
    return {"equal": a == b, **({} if a == b else {"a": a, "b": b})}


def numeric_gap(rows_a, rows_b, key):
    """Largest |a - b| of a numeric row field over the common steps (diagnostic only; the rule is exact equality)."""
    gaps = [abs(x[key] - y[key]) for x, y in zip(rows_a, rows_b)
            if isinstance(x.get(key), (int, float)) and isinstance(y.get(key), (int, float))]
    return max(gaps) if gaps else None


def compare_rows(rows_a, rows_b):
    keys = sorted({k for r in rows_a + rows_b for k in r} - RESOURCE_KEYS)
    first = next((i for i, (x, y) in enumerate(zip(rows_a, rows_b)) if any(x.get(k) != y.get(k) for k in keys)), None)
    out = {"steps": [len(rows_a), len(rows_b)], "fields": keys, "equal": len(rows_a) == len(rows_b) and first is None,
           "first_differing_step": None if first is None else rows_a[first].get("step")}
    if first is not None:
        x, y = rows_a[first], rows_b[first]
        out["at_first_difference"] = {k: [x.get(k), y.get(k)] for k in keys if x.get(k) != y.get(k)}
        out["max_abs_gap"] = {k: numeric_gap(rows_a, rows_b, k) for k in keys if numeric_gap(rows_a, rows_b, k)}
    return out


def compare_records(name_a, name_b, cfg_a, cfg_b):
    """Streaming runs: the per-group tensor sha256 of the committed rollout records (index.jsonl)."""
    def index(name, cfg):
        if not cfg.get("streaming", {}).get("enabled"):
            return None
        d = Path(cfg["streaming"]["directory"])
        p = (d if d.is_absolute() else runs.run_dir(name) / d) / "rollouts" / "index.jsonl"
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else None
    ia, ib = index(name_a, cfg_a), index(name_b, cfg_b)
    if ia is None or ib is None:
        return {"compared": False, "why": "both runs need streaming.enabled with rollout records"}
    ka = [(r["group_id"], r["sha256"], r["policy_version"], r["occurrences"]) for r in ia]
    kb = [(r["group_id"], r["sha256"], r["policy_version"], r["occurrences"]) for r in ib]
    first = next((i for i, (x, y) in enumerate(zip(ka, kb)) if x != y), None)
    return {"compared": True, "groups": [len(ka), len(kb)], "equal": ka == kb,
            "first_differing_group": None if first is None else ka[first][0]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference run name (e.g. qwen3_0.6b_resident)")
    ap.add_argument("b", help="candidate run name (e.g. qwen3_0.6b_l2l_sync)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-differ", action="append", default=[], metavar="SECTION",
                    help="also let this config section differ (e.g. streaming: in-memory vs disk-streaming runs, which "
                         "must still agree bitwise); recorded in the result")
    args = ap.parse_args(argv)
    A, B = load_log(args.a), load_log(args.b)
    cfg_a, cfg_b = config.normalized(A["yaml_config"]), config.normalized(B["yaml_config"])
    backends = [cfg_a["execution"]["backend"], cfg_b["execution"]["backend"]]
    diffs = config.differences(cfg_a, cfg_b)
    allowed = COMPARABLE_PREFIXES + tuple(f"{x.rstrip('.')}." for x in args.allow_differ)
    blocking = [d for d in diffs if not d.startswith(allowed)]
    rep = {"a": args.a, "b": args.b, "backends": backends,
           "configs": {"differences": diffs, "comparable": not blocking, "not_allowed": blocking,
                       "rule": f"only {', '.join(allowed)} may differ",
                       "allowed_beyond_default": list(args.allow_differ)},
           "tolerance": tolerance_rule(backends[1]), "coverage": COVERAGE}

    rows_a, rows_b = A.get("training", []), B.get("training", [])
    num = {"training_rows": compare_rows(rows_a, rows_b),
           "zero_signal_equal": [z["rollout_sha256"] for z in A.get("zero_signal", [])]
           == [z["rollout_sha256"] for z in B.get("zero_signal", [])],
           "rollout_records": compare_records(args.a, args.b, cfg_a, cfg_b)}
    ev_a, ev_b = A.get("evaluations", []), B.get("evaluations", [])
    num["evaluations"] = {"count": [len(ev_a), len(ev_b)],
                          "equal": [e["predictions"] for e in ev_a] == [e["predictions"] for e in ev_b],
                          "accuracy_a": [[e["step"], e["metrics"]["accuracy"]] for e in ev_a],
                          "accuracy_b": [[e["step"], e["metrics"]["accuracy"]] for e in ev_b]}
    ck_a, ck_b = dict(runs.checkpoints(args.a)), dict(runs.checkpoints(args.b))
    num["checkpoints"] = {}
    for step in sorted(set(ck_a) & set(ck_b)):
        row = {"adapter": tensor_diff(load_peft_weights(str(ck_a[step]), device="cpu"),
                                      load_peft_weights(str(ck_b[step]), device="cpu"))}
        sa, sb = (Path(ck_a[step]) / checkpointing.STATE), (Path(ck_b[step]) / checkpointing.STATE)
        if sa.exists() and sb.exists():
            load = lambda f: torch.load(f, map_location="cpu", weights_only=True)["optimizer"]["state"]  # GPU runs, any host
            row["optimizer"] = tensor_diff(load(sa), load(sb))
        num["checkpoints"][step] = row
    num["checkpoints_only_in_one"] = sorted(set(ck_a) ^ set(ck_b))
    va, vb = runs.training_dir(args.a) / "validation.json", runs.training_dir(args.b) / "validation.json"
    if va.exists() and vb.exists():
        Va, Vb = json.loads(va.read_text()), json.loads(vb.read_text())
        num["final_validation"] = {"predictions_equal": Va.get("predictions") == Vb.get("predictions"),
                                   "label_scores_equal": Va.get("label_scores") == Vb.get("label_scores")}
    num["controls"] = {r: {"frozen_changed": L.get("frozen_changed"), "params_equal_init": L.get("params_equal_init"),
                           "status": L.get("status")} for r, L in ((args.a, A), (args.b, B))}
    checks = {"training_rows": num["training_rows"]["equal"], "zero_signal": num["zero_signal_equal"],
              "rollout_records": num["rollout_records"].get("equal", True), "evaluations": num["evaluations"]["equal"],
              "same_checkpoints": not num["checkpoints_only_in_one"],
              "checkpoints": all(v["adapter"]["equal"] and v.get("optimizer", {"equal": True})["equal"]
                                 for v in num["checkpoints"].values()),
              "final_validation": all(num.get("final_validation", {"x": True}).values()),
              # frozen_changed: the frozen tensors whose digest changed ([] or False = none); None = not recorded
              "frozen_unchanged": all(c["frozen_changed"] is not None and not c["frozen_changed"]
                                      for c in num["controls"].values()),
              "both_completed": all(c["status"] in ("completed", "complete") for c in num["controls"].values())}
    num["checks"] = checks
    num["identical"] = all(checks.values())
    rep["numerical_equivalence"] = num
    rep["identical"] = rep["configs"]["comparable"] and num["identical"]

    rep["performance"] = {
        "note": "one run each on its own hardware: indicative, not a benchmark; never part of the verdict",
        **{r: {"train_seconds": L.get("train_seconds"), "steps": L.get("successful_steps"),
               "seconds_per_step": (L.get("train_seconds") or 0) / max(L.get("successful_steps") or 1, 1),
               "phase_seconds_total": {k: round(sum((row.get("timings") or {}).get(k) or 0 for row in L.get("training", [])), 3)
                                       for k in sorted({k for row in L.get("training", []) for k, v in (row.get("timings") or {}).items()
                                                        if isinstance(v, (int, float))})},
               "peak_reserved_gib_max": max((row.get("peak_reserved_gib") or 0 for row in L.get("training", [])), default=None),
               "memory": L.get("memory")} for r, L in ((args.a, A), (args.b, B))}}
    rep["provenance"] = {r: {k: (L.get("environment") or {}).get(k) for k in ("git_commit", "git_dirty", "packages", "gpu",
                                                                            "device", "host")}
                         for r, L in ((args.a, A), (args.b, B))}
    pa, pb = rep["provenance"][args.a], rep["provenance"][args.b]
    rep["same_environment"] = (pa.get("packages"), (pa.get("gpu") or {}).get("name")) == \
        (pb.get("packages"), (pb.get("gpu") or {}).get("name"))

    out = args.out or ROOT / "research" / "l2l" / "experiments" / f"compare_{args.a}_vs_{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    tr = num["training_rows"]
    print(f"{args.a} ({backends[0]}) vs {args.b} ({backends[1]}); rule: {rep['tolerance']['source']} "
          f"({'bitwise' if rep['tolerance']['exact'] else 'NOT all-zero thresholds: exact comparison still applied'})")
    print(f"configs comparable {rep['configs']['comparable']}" + (f" (differ outside run/execution/memory: {blocking})"
                                                                  if blocking else f" (differences: {diffs})"))
    if not rep["same_environment"]:
        print("WARNING: the two runs ran with different packages or GPUs; bitwise equality is only expected on the same "
              "machine and software")
    print("NUMERICAL EQUIVALENCE")
    print(f"  training rows: {tr['steps']} steps, {len(tr['fields'])} fields, identical {tr['equal']}"
          + ("" if tr["equal"] else f"; first difference at step {tr['first_differing_step']}: {tr.get('at_first_difference')}"))
    print(f"  zero-signal identical {num['zero_signal_equal']}; rollout records {num['rollout_records']}")
    print(f"  subset evaluations identical {num['evaluations']['equal']} (accuracy {num['evaluations']['accuracy_a']})")
    for step, v in num["checkpoints"].items():
        print(f"  checkpoint step{step}: adapter identical {v['adapter']['equal']}, optimizer identical "
              f"{v.get('optimizer', {}).get('equal')}")
    if "final_validation" in num:
        print(f"  final validation: {num['final_validation']}")
    print(f"  controls: {num['controls']}")
    print("PERFORMANCE (indicative, not part of the verdict)")
    for r in (args.a, args.b):
        v = rep["performance"][r]
        print(f"  {r}: {v['train_seconds'] or 0:.0f} s, {v['seconds_per_step']:.1f} s/step, phases {v['phase_seconds_total']}, "
              f"peak reserved {v['peak_reserved_gib_max']} GiB (allocator, excludes the CUDA context)")
    for r, v in rep["provenance"].items():
        print(f"  {r}: commit {(v['git_commit'] or '?')[:10]} (dirty {v['git_dirty']}), device {v['device']}, "
              f"gpu {(v['gpu'] or {}).get('name')}")
    print(f"IDENTICAL: {rep['identical']}   written: {out}")
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
