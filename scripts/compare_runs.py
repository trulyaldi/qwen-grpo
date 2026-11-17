"""Compare two finished training runs under runs/ (e.g. resident vs L2L of the same config) and report every difference.

    python scripts/compare_runs.py qwen_resident qwen_l2l
    python scripts/compare_runs.py qwen_resident_short qwen_l2l_short

Compared (run A is the reference):
  training rows      per optimizer step: rollout hash, rewards, loss, clipping statistics, gradient / update norms,
                     init_diff, change from init (exact equality)
  zero-signal        the rollout hashes of skipped batches
  evaluations        validation-subset predictions at every evaluation step
  checkpoints        every common checkpoint: adapter tensors and AdamW state (bitwise, else max abs difference)
  final validation   training/validation.json: predictions and label scores, when both runs wrote one
  controls           frozen weights unchanged, parameters changed from init
Also recorded: each run's provenance (commit, dirty flag, packages, GPU) and its train time, time per step
and GPU memory (one run each: indicative, not a benchmark).
Writes research/l2l/experiments/compare_<A>_vs_<B>.json; exit status 0 iff everything compared is identical.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import load_peft_weights  # noqa: E402

from qwen_grpo import checkpointing, runs  # noqa: E402

KEYS = ("attempt", "rollout_sha256", "rewards", "loss", "grad_norm", "update_norm", "change_from_init", "clip_frac",
        "ratio_min", "ratio_max", "approx_kl", "init_diff")


def load_log(name):
    path = runs.training_dir(name) / "training.json"
    if not path.exists():
        raise SystemExit(f"no training log for run {name!r}: {path}")
    return json.loads(path.read_text())


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference run name (e.g. qwen_resident)")
    ap.add_argument("b", help="candidate run name (e.g. qwen_l2l)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    A, B = load_log(args.a), load_log(args.b)
    rep = {"a": args.a, "b": args.b, "backends": [A.get("yaml_config", {}).get("execution", {}).get("backend"),
                                                 B.get("yaml_config", {}).get("execution", {}).get("backend")]}

    rows_a, rows_b = A.get("training", []), B.get("training", [])
    first_diff = next((i for i, (x, y) in enumerate(zip(rows_a, rows_b))
                       if [x.get(k) for k in KEYS] != [y.get(k) for k in KEYS]), None)
    rep["training_rows"] = {"steps": [len(rows_a), len(rows_b)], "equal": len(rows_a) == len(rows_b) and first_diff is None,
                            "first_differing_step": None if first_diff is None else rows_a[first_diff].get("step")}
    if first_diff is not None:
        x, y = rows_a[first_diff], rows_b[first_diff]
        rep["training_rows"]["at_first_difference"] = {k: [x.get(k), y.get(k)] for k in KEYS if x.get(k) != y.get(k)}
    rep["zero_signal_equal"] = [z["rollout_sha256"] for z in A.get("zero_signal", [])] == \
        [z["rollout_sha256"] for z in B.get("zero_signal", [])]
    ev_a, ev_b = A.get("evaluations", []), B.get("evaluations", [])
    rep["evaluations"] = {"count": [len(ev_a), len(ev_b)], "equal": [e["predictions"] for e in ev_a] == [e["predictions"] for e in ev_b],
                          "accuracy": [[e["step"], e["metrics"]["accuracy"]] for e in ev_a],
                          "accuracy_b": [[e["step"], e["metrics"]["accuracy"]] for e in ev_b]}

    ck_a, ck_b = dict(runs.checkpoints(args.a)), dict(runs.checkpoints(args.b))
    rep["checkpoints"] = {}
    for step in sorted(set(ck_a) & set(ck_b)):
        row = {"adapter": tensor_diff(load_peft_weights(str(ck_a[step]), device="cpu"),
                                      load_peft_weights(str(ck_b[step]), device="cpu"))}
        sa, sb = (Path(ck_a[step]) / checkpointing.STATE), (Path(ck_b[step]) / checkpointing.STATE)
        if sa.exists() and sb.exists():
            row["optimizer"] = tensor_diff(checkpointing.load_training_state(ck_a[step])["optimizer"]["state"],
                                           checkpointing.load_training_state(ck_b[step])["optimizer"]["state"])
        rep["checkpoints"][step] = row
    rep["checkpoints_only_in_one"] = sorted(set(ck_a) ^ set(ck_b))

    va, vb = runs.training_dir(args.a) / "validation.json", runs.training_dir(args.b) / "validation.json"
    if va.exists() and vb.exists():
        Va, Vb = json.loads(va.read_text()), json.loads(vb.read_text())
        rep["final_validation"] = {"predictions_equal": Va.get("predictions") == Vb.get("predictions"),
                                   "label_scores_equal": Va.get("label_scores") == Vb.get("label_scores")}
    rep["final"] = {"a": A.get("final", {}).get("trained"), "b": B.get("final", {}).get("trained")}
    rep["controls"] = {r: {"frozen_changed": L.get("frozen_changed"), "params_equal_init": L.get("params_equal_init"),
                           "status": L.get("status")} for r, L in ((args.a, A), (args.b, B))}
    rep["provenance"] = {r: {k: (L.get("environment") or {}).get(k) for k in ("git_commit", "git_dirty", "packages", "gpu",
                                                                            "device")}
                         for r, L in ((args.a, A), (args.b, B))}
    rep["resources"] = {r: {"train_seconds": L.get("train_seconds"), "steps": L.get("successful_steps"),
                            "seconds_per_step": (L.get("train_seconds") or 0) / max(L.get("successful_steps") or 1, 1),
                            "memory": L.get("memory")} for r, L in ((args.a, A), (args.b, B))}

    checks = [rep["training_rows"]["equal"], rep["zero_signal_equal"], rep["evaluations"]["equal"],
              not rep["checkpoints_only_in_one"], all(v["adapter"]["equal"] and v.get("optimizer", {"equal": True})["equal"]
                                                      for v in rep["checkpoints"].values()),
              all(rep.get("final_validation", {"x": True}).values()),
              all(not c["frozen_changed"] for c in rep["controls"].values())]
    rep["identical"] = all(checks)

    out = args.out or ROOT / "research" / "l2l" / "experiments" / f"compare_{args.a}_vs_{args.b}.json"
    out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    tr = rep["training_rows"]
    print(f"{args.a} ({rep['backends'][0]}) vs {args.b} ({rep['backends'][1]})")
    print(f"training rows: {tr['steps']} steps, identical {tr['equal']}"
          + ("" if tr["equal"] else f", first difference at step {tr['first_differing_step']}: {tr.get('at_first_difference')}"))
    print(f"zero-signal batches identical {rep['zero_signal_equal']}; subset evaluations identical {rep['evaluations']['equal']} "
          f"(accuracy {rep['evaluations']['accuracy']})")
    for step, v in rep["checkpoints"].items():
        print(f"checkpoint step{step}: adapter identical {v['adapter']['equal']}, optimizer identical "
              f"{v.get('optimizer', {}).get('equal')}")
    if "final_validation" in rep:
        print(f"final validation: predictions identical {rep['final_validation']['predictions_equal']}, label scores "
              f"identical {rep['final_validation']['label_scores_equal']}; trained accuracy {rep['final']}")
    print(f"controls: {rep['controls']}")
    for r, v in rep["resources"].items():
        mem = v["memory"] or {}
        print(f"{r}: train {v['train_seconds']:.0f} s, {v['seconds_per_step']:.1f} s/step, memory {mem}")
    for r, v in rep["provenance"].items():
        print(f"{r}: commit {(v['git_commit'] or '?')[:10]} (dirty {v['git_dirty']}), device {v['device']}, "
              f"gpu {(v['gpu'] or {}).get('name')}")
    print(f"IDENTICAL: {rep['identical']}   written: {out}")
    return 0 if rep["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
