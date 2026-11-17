"""H08 experiment (Ticket 09): an uninterrupted streaming run A against runs B that crash at injected points and resume
from their latest valid checkpoint; everything committed must be identical.

    python scripts/resume_experiment.py --config configs/experiments/qwen_stream_short.yaml --device cuda   # laptop
    python scripts/resume_experiment.py --tiny --device cpu                                                # any host

Runs (fresh names derived from the config's run.name: <name>_A, <name>_B_<point>; existing ones are overwritten):
  A   uninterrupted
  B   for every crash point: QWEN_GRPO_CRASH=<point@n> (trainer.crash_point) -> the process "dies" with InjectedCrash,
      then trainer.train(resume="latest") continues from the highest-step valid checkpoint
Compared, A vs each B: the committed training rows (step, attempt, prompt ids, rollout hash, rewards, loss, gradient /
update norms, iteration, clipping), zero-signal attempts, subset evaluations, attempt and optimizer-step counters; the
committed on-disk rollout records (group ids, policy versions, (epoch, position, sample id) occurrences, generated
tokens, rewards); every common checkpoint's adapter and AdamW state (scripts/compare_runs.py); and, for B alone: no
duplicate or missing sample occurrence in any epoch, no interrupted publication left, every checkpoint valid.
--tiny: the trainer tests' tiny random Qwen3 (real tokenizer and data) with content-hash rewards, on CPU.
Writes research/l2l/experiments/<out>.json (default t09_resume_<name>.json) and exits 0 iff every B matches A.
"""
import argparse
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qwen_grpo import config, runs, trainer  # noqa: E402
from qwen_grpo.streaming import resume as R  # noqa: E402

# tiny setup (checkpoints every 2 steps, 3 GRPO iterations per group): every crash falls after the step-2 checkpoint
DEFAULT_CRASHES = ["prompt_read@3", "rollouts_generated@3", "update@4", "files_written@3", "committed_before_rename@3",
                   "after_checkpoint@2"]
# configs/experiments/qwen_stream_short.yaml (checkpoints every 3 steps, 2 iterations per group): after step 3
QWEN_CRASHES = ["prompt_read@4", "rollouts_generated@4", "update@5", "files_written@3", "committed_before_rename@3",
                "after_checkpoint@2"]
KEYS = ("step", "attempt", "prompt_ids", "rollout_sha256", "rewards", "loss", "grad_norm", "update_norm", "iteration",
        "clip_frac", "init_diff")


def committed(out):
    return {"rows": [[r.get(k) for k in KEYS] for r in out["training"]],
            "zero_signal": [[z["attempt"], z["rollout_sha256"]] for z in out["zero_signal"]],
            "evaluations": [e["predictions"] for e in out["evaluations"]],
            "attempts": out["groups_generated"], "steps": out["successful_steps"]}


def records(cfg, name):
    d = trainer.streaming_dir(renamed(cfg, name, False)) / "rollouts"
    out = {}
    for line in (d / "index.jsonl").read_text().splitlines():
        e = json.loads(line)
        rec = torch.load(d / e["file"], weights_only=True)
        out[e["group_id"]] = rec
    return out


def epoch_problems(recs, n):
    occ = [tuple(o) for g in sorted(recs) for o in recs[g]["occurrences"]]
    problems = []
    if len({(e, s) for e, _, s in occ}) != len(occ):
        problems.append("a sample occurs twice within one epoch")
    by = {}
    for e, pos, s in occ:
        by.setdefault(e, []).append(pos)
    for e, pos in by.items():
        if sorted(pos) != list(range(len(pos))):
            problems.append(f"epoch {e}: positions not contiguous")
        if e < max(by) and len(pos) != n:
            problems.append(f"epoch {e}: {len(pos)} of {n} samples")
    return problems


def renamed(cfg, name, overwrite):
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, name=name, overwrite=overwrite))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--crash", nargs="*", help="crash points point@n (default: one set per setup, all after the first "
                    "resumable checkpoint)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.tiny:
        cfg = tiny_setup()
    else:
        cfg = config.load(args.config)
    args.crash = args.crash or (DEFAULT_CRASHES if args.tiny else QWEN_CRASHES)
    if not cfg.streaming.enabled:
        raise SystemExit("the config must enable streaming")
    base = cfg.run.name
    os.environ.pop("QWEN_GRPO_CRASH", None)
    a_name = f"{base}_A"
    A = trainer.train(renamed(cfg, a_name, True), device=args.device)
    rep = {"config": str(args.config) if args.config else "tiny", "device": args.device, "A": a_name,
           "A_attempts": A["groups_generated"], "A_steps": A["successful_steps"],
           "A_zero_signal": len(A["zero_signal"]), "B": {}}
    rec_a = records(cfg, a_name)
    n = len(A["config"]["train_ids"])
    rep["A_epoch_problems"] = epoch_problems(rec_a, n)
    from scripts import compare_runs
    for spec in args.crash:
        b_name = f"{base}_B_{spec.split('@')[0]}"
        cfg_b = renamed(cfg, b_name, True)
        row = {"crash": spec}
        os.environ["QWEN_GRPO_CRASH"] = spec
        trainer._CRASH_SEEN.clear()
        try:
            trainer.train(cfg_b, device=args.device)
            row["crashed"] = False
        except trainer.InjectedCrash as e:
            row["crashed"], row["crash_message"] = True, str(e)
        finally:
            os.environ.pop("QWEN_GRPO_CRASH", None)
        latest, rejected = R.latest_valid(b_name)
        row["resumed_from"] = None if latest is None else runs.rel(latest)
        row["rejected_checkpoints"] = [(runs.rel(p), why) for p, why in rejected]
        B = trainer.train(renamed(cfg, b_name, False), resume="latest", device=args.device)
        ca, cb = committed(A), committed(B)
        row["committed_equal"] = ca == cb
        row["first_difference"] = next((k for k in ca if ca[k] != cb[k]), None)
        rec_b = records(cfg, b_name)
        row["records_equal"] = rec_a.keys() == rec_b.keys() and all(
            rec_a[g]["policy_version"] == rec_b[g]["policy_version"]
            and [tuple(o) for o in rec_a[g]["occurrences"]] == [tuple(o) for o in rec_b[g]["occurrences"]]
            and torch.equal(rec_a[g]["response_ids"], rec_b[g]["response_ids"])
            and torch.equal(rec_a[g]["rewards"], rec_b[g]["rewards"]) for g in rec_a)
        row["epoch_problems"] = epoch_problems(rec_b, n)
        ckdir = runs.run_dir(b_name) / "checkpoints"
        row["leftover_tmp"] = [p.name for p in ckdir.glob(".tmp-*")]
        row["quarantined"] = [p.name for p in ckdir.glob(".invalid-*")]
        row["invalid_checkpoints"] = [runs.rel(p) for s, p in runs.checkpoints(b_name)
                                      if not R.validate(p, resumable=s > 0)[0]]
        out = Path(os.devnull)
        row["checkpoints_identical"] = compare_runs.main([a_name, b_name, "--out", str(out)]) == 0
        row["pass"] = (row["crashed"] and row["committed_equal"] and row["records_equal"] and not row["epoch_problems"]
                       and not row["leftover_tmp"] and not row["invalid_checkpoints"] and row["checkpoints_identical"])
        rep["B"][spec] = row
        print(f"{spec}: crashed {row['crashed']}, resumed from {row['resumed_from']}, committed equal "
              f"{row['committed_equal']}, records equal {row['records_equal']}, checkpoints identical "
              f"{row['checkpoints_identical']}, epoch problems {row['epoch_problems']} -> {'PASS' if row['pass'] else 'FAIL'}")
    rep["provenance"] = {"git_commit": trainer.git_state()}
    rep["pass"] = not rep["A_epoch_problems"] and all(r["pass"] for r in rep["B"].values())
    path = ROOT / "research" / "l2l" / "experiments" / f"{args.out or f't09_resume_{base}'}.json"
    path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(f"H08 {'SUPPORTED' if rep['pass'] else 'NOT SUPPORTED'} on this experiment; written: {path}")
    return 0 if rep["pass"] else 1


def tiny_setup():
    """The trainer tests' tiny model and content-hash rewards (with zero-signal batches), outputs in a scratch runs root."""
    import tempfile
    import yaml
    sys.path.insert(0, str(ROOT))
    from tests.test_config import RAW
    from tests.test_streaming import content_rewards
    from tests.test_trainer import SETTINGS, tiny_lora_model
    trainer.lora_model, trainer.group_rewards = tiny_lora_model, content_rewards
    runs.ROOT = Path(tempfile.mkdtemp(prefix="t09_resume_"))
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = "tiny_stream"
    raw["model"]["dtype"] = "float32"
    raw["data"].update(train_samples=4, val_subset=2)
    raw["training"].update({**SETTINGS, "max_steps": 6, "num_iterations": 3})
    raw["evaluation"].update(every=2, full_validation=False)
    raw["streaming"] = {"enabled": True, "directory": "streaming"}
    path = runs.ROOT / "tiny.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


if __name__ == "__main__":
    sys.exit(main())
