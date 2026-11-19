"""H08 with a real external kill: an uninterrupted training process A against a process B that is SIGKILLed from the
outside mid-epoch (after its first resumable checkpoint, with uncommitted progress on disk) and restarted as a NEW
process from its latest valid checkpoint. Complements scripts/resume_experiment.py (in-process InjectedCrash at
seven code points), which stays unchanged.

    python scripts/resume_kill_test.py --tiny                                   # CPU smoke, any host, ~1-2 min
    python scripts/resume_kill_test.py --config configs/experiments/qwen3_4b_resume_kill.yaml --device cuda \
        --cap-bytes 8000000000                                                  # GPU-realistic (never run implicitly)

Processes (each its own session; GPU mode: each under qwen_grpo.telemetry.run_guarded with the cap):
  A    `train` of run <name>_killA, uninterrupted
  B1   `train` of run <name>_killB; the supervisor waits for B1 to pass its first resumable checkpoint and to have
       uncommitted progress, then sends SIGKILL to B1's process group (no Python exception, no cleanup handlers run).
       --tiny: B1 runs with QWEN_GRPO_STALL=<point>@<n>:<s> (trainer.stall_point prints a marker and sleeps there, so
       the kill lands at a known position mid-epoch; it raises nothing). Otherwise: killed when its stdout shows a
       "step N" line with N > the first checkpoint's step (or --stall gives the same hook).
  B2   a new process: `train --resume latest` (trainer: highest-step valid checkpoint; invalid / partial ones rejected)
Compared, A vs B (what an uninterrupted run would have produced):
  committed training rows (step, attempt, prompt ids, rollout hash, rewards, loss, grad/update norms, iteration,
  clipping), zero-signal attempts, subset-evaluation predictions, attempt and optimizer-step counters
  on-disk rollout records: group ids, policy versions, (epoch, position, sample id) occurrences in order, generated
  tokens, rewards
  sample accounting over the whole run: no sample twice within an epoch, contiguous positions, every completed epoch
  complete; and the multiset of (epoch, sample) occurrences equal to A's (no gap, no duplicate)
  every common checkpoint: LoRA tensors and AdamW state (scripts/compare_runs.py, bitwise); the final checkpoint's saved
  torch (CPU, CUDA) RNG state equal to A's; B's Python RNG state equal to the one saved in the checkpoint it resumed
  from (Python's global RNG is unseeded and unused by training, so it differs between processes and is not compared
  across A and B)
  B's checkpoint directory: no leftover temporary or invalid checkpoint
Not compared (listed in the result): wall-clock timings, memory samples, the exact instant of the kill, and anything
the trainer does not record (per-token log-probs, gradient tensors).
Writes research/l2l/experiments/<--out, default t09_resume_kill_<name>_r<UTC>>.json; exit 0 iff every check passes.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TINY_STALL = "rollouts_generated@3:900"  # tiny setup: after the step-2 checkpoint, a rollout group on disk uncommitted
STEP_LINE = re.compile(r"^step\s+(\d+)\s+\(attempt")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


# --- child: one training process ------------------------------------------------------------------------------------

def child(a):
    from qwen_grpo import config, runs, trainer
    if a.tiny:
        from tests.test_streaming import content_rewards
        from tests.test_trainer import tiny_lora_model
        trainer.lora_model, trainer.group_rewards = tiny_lora_model, content_rewards
        runs.ROOT = Path(a.root)
    cfg = config.load(a.config, historical=a.resume is not None)
    trainer.train(cfg, resume=a.resume, device=a.device)
    return 0


# --- supervisor ------------------------------------------------------------------------------------------------------

def run_name_config(base_cfg, name, where, tiny):
    """A YAML that extends the given config with only run.name changed (overwrite false: an existing run is refused)."""
    p = Path(where) / f"{name}.yaml"
    p.write_text(f"extends: {Path(base_cfg).resolve()}\nrun:\n  name: {name}\n  overwrite: {'true' if tiny else 'false'}\n")
    return p


def launch(argv, out_dir, cap, gpu, env, on_started=None):
    """Run argv; GPU mode under run_guarded (telemetry + cap), CPU mode as a plain session leader with logs."""
    from qwen_grpo import telemetry
    if gpu:
        return telemetry.run_guarded(out_dir, argv, cap_bytes=cap, cwd=ROOT, env=env, on_started=on_started,
                                     attribution_timeout_s=900)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rec = {"argv": argv, "started_utc": utc()}
    with open(out_dir / "stdout.log", "wb") as so, open(out_dir / "stderr.log", "wb") as se:
        p = subprocess.Popen(argv, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}, stdout=so,
                             stderr=se, stdin=subprocess.DEVNULL, start_new_session=True)
        if on_started:
            on_started(p)
        p.wait()
    rec.update(exit=p.returncode, ended_utc=utc())
    return rec


def killer(proc_box, out_dir, first_ckpt_step, ckpt_dir, marker, stop):
    """SIGKILL the process group once the kill condition holds; record what was seen at that moment."""
    seen = {}
    log = Path(out_dir) / "stdout.log"
    while not stop.is_set():
        p = proc_box.get("p")
        text = log.read_text(errors="replace") if log.exists() else ""
        steps = [int(m[1]) for m in (STEP_LINE.match(line) for line in text.splitlines()) if m]
        ckpts = sorted(int(d.name[4:]) for d in Path(ckpt_dir).glob("step*") if d.name[4:].isdigit())
        ready = first_ckpt_step in ckpts and ((marker and marker in text) or
                                               (not marker and steps and max(steps) > first_ckpt_step))
        if p is not None and ready and p.poll() is None:
            seen.update(killed_utc=utc(), last_step_line=max(steps) if steps else None, checkpoints_at_kill=ckpts,
                        signal="SIGKILL", pgid=p.pid)
            os.killpg(p.pid, signal.SIGKILL)
            return seen
        if p is not None and p.poll() is not None:
            seen["exited_before_kill"] = p.returncode
            return seen
        time.sleep(0.2)
    return seen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("child")
    c.add_argument("--config", type=Path, required=True)
    c.add_argument("--resume")
    c.add_argument("--device", default="cpu")
    c.add_argument("--tiny", action="store_true")
    c.add_argument("--root")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cap-bytes", type=int, default=8_000_000_000)
    ap.add_argument("--stall", help="QWEN_GRPO_STALL for B1 (point@n:seconds); --tiny default " + TINY_STALL)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "child":
        return child(a)
    if not a.tiny and not a.config:
        raise SystemExit("give --tiny or --config")
    from qwen_grpo import config, runs
    from scripts import resume_experiment as RE
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    work = Path(tempfile.mkdtemp(prefix="t09_kill_"))
    if a.tiny:
        cfg = RE.tiny_setup()  # writes tiny.yaml under a scratch runs root (runs.ROOT)
        base_yaml, root, device, gpu = runs.ROOT / "tiny.yaml", runs.ROOT, "cpu", False
        stall = a.stall or TINY_STALL
    else:
        cfg = config.load(a.config)
        base_yaml, root, device, gpu, stall = a.config, None, a.device, a.device.startswith("cuda"), a.stall
    if not cfg.streaming.enabled:
        raise SystemExit("the config must enable streaming (disk-backed prompts, crash-consistent checkpoints)")
    base = cfg.run.name
    names = {"A": f"{base}_killA", "B": f"{base}_killB"}
    if not a.tiny:
        for n in names.values():
            if runs.run_dir(n).exists():
                raise SystemExit(f"{runs.rel(runs.run_dir(n))} exists: this test never reuses or deletes a run")
    yamls = {k: run_name_config(base_yaml, n, work, a.tiny) for k, n in names.items()}
    me = [sys.executable, str(Path(__file__).resolve()), "child", "--device", device] + \
         (["--tiny", "--root", str(root)] if a.tiny else [])
    out = ROOT / "research" / "l2l" / "experiments" / f"{a.out or f't09_resume_kill_{base}_r{stamp}'}"
    raw = Path(f"{out}_raw")
    rep = {"mode": "tiny_cpu" if a.tiny else f"real_{device}", "config": str(a.config or "tiny"), "runs": names,
           "started_utc": utc(), "stall": stall, "cap_bytes": a.cap_bytes if gpu else None, "raw": runs.rel(raw)}
    every = cfg.evaluation.every
    print(f"A: uninterrupted {names['A']}", flush=True)
    rep["A"] = launch(me + ["--config", str(yamls["A"])], raw / "A", a.cap_bytes, gpu, {})
    print(f"B1: {names['B']} until killed after checkpoint step{every}", flush=True)
    box, stop, seen = {}, threading.Event(), {}
    ckdir = runs.run_dir(names["B"]) / "checkpoints"  # runs.ROOT: the scratch root in --tiny mode
    marker = "QWEN_GRPO_STALL reached" if stall else None
    t = threading.Thread(target=lambda: seen.update(killer(box, raw / "B1", every, ckdir, marker, stop)), daemon=True)
    t.start()
    rep["B1"] = launch(me + ["--config", str(yamls["B"])], raw / "B1", a.cap_bytes, gpu,
                       {"QWEN_GRPO_STALL": stall} if stall else {}, on_started=lambda p: box.update(p=p))
    stop.set()
    t.join(timeout=10)
    rep["kill"] = seen
    print(f"B1 ended: {seen}", flush=True)
    print("B2: new process, --resume latest", flush=True)
    rep["B2"] = launch(me + ["--config", str(runs.run_dir(names["B"]) / "config.yaml"), "--resume", "latest"],
                       raw / "B2", a.cap_bytes, gpu, {})
    rep["checks"], rep["compared"], rep["not_compared"] = compare(cfg, names, RE)
    rep["pass"] = (bool(seen.get("killed_utc")) and rep["B1"].get("exit") == -signal.SIGKILL
                   and rep["A"].get("exit") == 0 and rep["B2"].get("exit") == 0 and all(rep["checks"].values()))
    if gpu:
        rep["memory_status"] = {k: rep[k].get("memory_status") for k in ("A", "B1", "B2")}
        rep["pass"] = rep["pass"] and all(v == "PASS" for v in rep["memory_status"].values())
    rep["ended_utc"] = utc()
    path = Path(f"{out}.json")
    if path.exists():
        raise SystemExit(f"{path} exists: results are never overwritten")
    path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep["checks"], indent=1))
    print(f"H08 external kill/resume: {'PASS' if rep['pass'] else 'FAIL'}; written {runs.rel(path)}")
    return 0 if rep["pass"] else 1


def compare(cfg, names, RE):
    import torch
    from qwen_grpo import runs
    from qwen_grpo.checkpointing import STATE
    from scripts import compare_runs
    A = json.loads((runs.training_dir(names["A"]) / "training.json").read_text())
    B = json.loads((runs.training_dir(names["B"]) / "training.json").read_text())
    ca, cb = RE.committed(A), RE.committed(B)
    rec_a, rec_b = RE.records(cfg, names["A"]), RE.records(cfg, names["B"])
    n = len(A["config"]["train_ids"])
    occ = lambda recs: sorted((e, s) for g in recs for e, _, s in (tuple(o) for o in recs[g]["occurrences"]))
    fa, fb = runs.latest_checkpoint(names["A"]), runs.latest_checkpoint(names["B"])
    sa, sb = (torch.load(Path(f) / STATE, weights_only=False) for f in (fa, fb))
    resumed_from = torch.load(runs.ROOT / B["resumed"][-1]["from"] / STATE, weights_only=False) if B.get("resumed") \
        else {"rng": {"python": None}}

    def same(x, y):
        if isinstance(x, torch.Tensor):
            return isinstance(y, torch.Tensor) and torch.equal(x, y)
        if isinstance(x, (list, tuple)):
            return isinstance(y, (list, tuple)) and len(x) == len(y) and all(same(p, q) for p, q in zip(x, y))
        if isinstance(x, dict):
            return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
        return x == y

    ckdir = runs.run_dir(names["B"]) / "checkpoints"
    checks = {
        "both_completed": A.get("status") == B.get("status") == "complete",
        "resumed_session_recorded": bool(B.get("resumed")),
        "committed_rows_equal": ca == cb,
        "rollout_records_equal": rec_a.keys() == rec_b.keys() and all(
            rec_a[g]["policy_version"] == rec_b[g]["policy_version"]
            and [tuple(o) for o in rec_a[g]["occurrences"]] == [tuple(o) for o in rec_b[g]["occurrences"]]
            and torch.equal(rec_a[g]["response_ids"], rec_b[g]["response_ids"])
            and torch.equal(rec_a[g]["rewards"], rec_b[g]["rewards"]) for g in rec_a),
        "no_epoch_problems_A": not RE.epoch_problems(rec_a, n),
        "no_epoch_problems_B": not RE.epoch_problems(rec_b, n),
        "sample_multiset_equal_no_gap_no_duplicate": occ(rec_a) == occ(rec_b),
        "checkpoints_identical_adapter_and_adamw": compare_runs.main([names["A"], names["B"], "--out", os.devnull]) == 0,
        # torch's global RNG drives rollout sampling: it must match the uninterrupted run exactly
        "final_torch_rng_equal": same(sa["rng"]["torch"], sb["rng"]["torch"]) and same(sa["rng"]["cuda"], sb["rng"]["cuda"]),
        # Python's global `random` is never seeded or drawn from by training (samplers use their own random.Random(seed),
        # whose state is part of the compared sampler/rollout records), so A and B start from different OS-entropy
        # states and cannot be compared. What can be checked: B2 restored the state saved in the checkpoint it resumed
        # from, and nothing consumed it afterwards (B's final state equals that saved state).
        "python_rng_restored_from_resume_checkpoint": same(sb["rng"]["python"], resumed_from["rng"]["python"]),
        "final_counters_equal": (sa["step"], sa["attempts"]) == (sb["step"], sb["attempts"]),
        "no_leftover_tmp_or_invalid": not list(ckdir.glob(".tmp-*")) and not list(ckdir.glob(".invalid-*")),
    }
    compared = ["training rows (step, attempt, prompt ids, rollout sha256, rewards, loss, grad/update norms, iteration, "
                "clip_frac, init_diff)", "zero-signal attempts (attempt, rollout sha256)", "subset-evaluation predictions",
                "attempt and optimizer-step counters", "rollout records on disk (group ids, policy versions, ordered "
                "occurrences, response token ids, rewards)", "(epoch, sample) occurrence multiset",
                "every common checkpoint's LoRA tensors and AdamW state (compare_runs)", "final checkpoint torch (and CUDA) "
                "RNG state vs A", "B's Python RNG state vs the state saved in its resume checkpoint"]
    not_compared = ["Python's global RNG across A and B (unseeded and unused by training: different per process)",
                    "wall-clock timings and memory samples (differ by construction)", "the exact instant of the kill "
                    "(recorded, not compared)", "per-token log-probs and gradient tensors (not recorded by the trainer)"]
    return checks, compared, not_compared


if __name__ == "__main__":
    sys.exit(main())
