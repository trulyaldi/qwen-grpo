"""qwen-grpo: the project's command line (docs/commands in README.md). A thin dispatcher: every command delegates to the
existing implementation (scripts/*.py, qwen_grpo.*), so behaviour and outputs are those documented there.

    qwen-grpo preflight --config <yaml> [--gpu]        offline checks (no GPU, no downloads); --gpu: on the GPU host
    qwen-grpo train     --config <yaml>                a new run in runs/<run.name>/ (gated by the offline preflight)
    qwen-grpo resume    --run <name> [--checkpoint P]  continue a run (latest valid checkpoint by default)
    qwen-grpo validate  --run <name> [--step N]        held-out validation split (resident configs)
    qwen-grpo test      --run <name> [--step N]        test split (resident configs; final report only)
    qwen-grpo verify    --run <name> [--control <run>] base / adapter / merged / LR=0 evidence
    qwen-grpo merge     --run <name> [--step N]        streamed tensor-by-tensor merge into runs/<name>/merged/step<N>/
    qwen-grpo report    --run <name>                   rebuild summary.json, plots and report.md (no model)
    qwen-grpo remove    --run <name> [--confirm]       dry run unless --confirm
    qwen-grpo research  list | run <H..> [--execute]   hypotheses, their evidence, and the commands that test them
    qwen-grpo ship                                     the evidence-derived SHIP / DON'T SHIP verdict
    qwen-grpo soup      <check> ...                    optional Soup diagnostics (qwen_grpo/integrations/soup.py)
    qwen-grpo faults    [--list]                       the silent-failure registry and its fault-injection evidence

Exit codes: 0 success / PASS / SHIP; 1 error; 2 a check failed (preflight FAIL or BLOCKED for a gated run, DON'T
SHIP, a failed experiment); 3 usage error. Every invocation is appended to runs/_cli/history.jsonl (command, arguments,
start and end UTC, exit code). Nothing here starts a GPU job unless the command itself is one (train, resume, verify of
an L2L run, merge --evaluate, research run --execute of a cuda entry, preflight --gpu).
"""
import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "research" / "l2l" / "experiments"
RUNBOOK = ROOT / "research" / "l2l" / "runbook.yaml"
HYPOTHESES = ROOT / "research" / "l2l" / "hypotheses.yaml"
OK, ERROR, CHECK_FAILED, USAGE = 0, 1, 2, 3


def script(name):
    """scripts/<name>.py as a module (the scripts are the implementations; they are not a package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"qwen_grpo_scripts.{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def history(record):
    """Append one invocation record to runs/_cli/history.jsonl ('_' cannot start a run name, so it never collides)."""
    path = ROOT / "runs" / "_cli" / "history.jsonl"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# --- run resolution -------------------------------------------------------------------------------------------

def run_config(name):
    from qwen_grpo import config, runs
    path = runs.run_dir(name) / "config.yaml"
    if not path.exists():
        raise SystemExit(f"no run {name!r}: {runs.rel(path)} does not exist (train it first: qwen-grpo train --config ...)")
    return path, config.load(path, historical=True)


# --- commands -------------------------------------------------------------------------------------------------

def cmd_preflight(a):
    from qwen_grpo.diagnostics import preflight
    report = preflight.run(a.config, gpu=a.gpu, allow_network=a.allow_metadata_download,
                           pcie_gbps=a.pcie_gbps, host_ram_gib=a.host_ram_gib, model_steps=not a.hardware_only)
    path = preflight.save(report)
    print(preflight.render(report))
    print(f"\nwrote {preflight.show(path)}")
    status = report["overall"] if a.gpu else report.get("overall_offline", report["overall"])
    return OK if status == "PASS" else CHECK_FAILED  # offline: the offline checks; --gpu: everything incl. gpu.measured


def cmd_train(a):
    from qwen_grpo import config
    from qwen_grpo.diagnostics import preflight
    cfg = config.load(a.config)
    gate = preflight.gate(a.config, cfg, skip=a.skip_preflight)
    print(gate["message"])
    if not gate["ok"]:
        return CHECK_FAILED
    from qwen_grpo.trainer import train
    train(cfg)
    return OK


def cmd_resume(a):
    from qwen_grpo import runs
    from qwen_grpo.trainer import train
    _, cfg = run_config(a.run)
    if a.checkpoint is not None:
        resume = Path(a.checkpoint)
    elif cfg.streaming.enabled:
        resume = "latest"  # the highest-step crash-consistent checkpoint (trainer validates it)
    else:
        resume = runs.latest_checkpoint(a.run)
        if resume is None:
            raise SystemExit(f"run {a.run!r} has no checkpoint to resume from")
    print(f"resuming run {a.run!r} from {resume if resume == 'latest' else runs.rel(resume)}")
    train(cfg, resume=resume)
    return OK


def cmd_run_py(a):
    """validate / test / verify (resident) / report / remove: scripts/run.py with --run."""
    argv = [a.command, "--run", a.run]
    if getattr(a, "step", None) is not None:
        argv += ["--step", str(a.step)]
    if getattr(a, "confirm", False):
        argv.append("--confirm")
    script("run").main(argv)
    return OK


def cmd_verify(a):
    _, cfg = run_config(a.run)
    if cfg.execution.backend == "resident":
        if a.control:
            raise SystemExit("verify of a resident run takes its LR=0 control from verification.control_checkpoint / "
                             "train_control in the run's config, not --control")
        return cmd_run_py(a)
    # L2L runs: the held-out evaluation through the L2L engine (base, trained, streamed-merged, LR=0 control)
    argv = ["--run", a.run, "--device", a.device]
    if a.control:
        argv += ["--control", a.control]
    if a.n:
        argv += ["--n", str(a.n)]
    script("final_eval").main(argv)
    return OK


def cmd_merge(a):
    from qwen_grpo import runs
    ckpt = runs.resolve(a.run, a.step)[1]
    out = Path(a.out) if a.out else runs.run_dir(a.run) / "merged" / ckpt.name
    if out.exists():
        raise SystemExit(f"{runs.rel(out)} exists; refusing to overwrite a merged checkpoint (choose --out)")
    argv = ["--adapter", str(ckpt), "--out", str(out)]
    if a.evaluate:
        argv += ["--evaluate", "--device", a.device, "--n", str(a.n)]
    script("merge").main(argv)
    print(f"merged checkpoint: {runs.rel(out)}/ (merge_report.json inside)")
    return OK


def cmd_ship(a):
    script("final_report").main()  # every requirement's status from research/l2l/experiments/, never by hand
    results = json.loads((ROOT / "research" / "l2l" / "final_results.json").read_text())
    verdict = results.get("verdict")
    for line in results.get("must_change", []):
        print(f"  must change: {line}")
    print(f"\nverdict (research/l2l/final_report.md): {verdict}")
    if a.soup:  # beside ours, never instead: Soup's offline ship has no CI, no control, no L2L / budget evidence
        from qwen_grpo.integrations import soup
        print(soup.render(soup.ship_from_final_results(results, a.soup_bin)) + "  (informational; does not change the verdict)")
    return OK if verdict == "SHIP" else CHECK_FAILED


def runbook():
    import yaml
    book = yaml.safe_load(RUNBOOK.read_text())
    hyp = yaml.safe_load(HYPOTHESES.read_text())
    return book, hyp


def evidence_state(entry):
    """exists / missing, and the saved status if the output is a lab result or a JSON with a top-level status."""
    path = EXPERIMENTS / entry["output"]
    if not path.exists():
        return {"exists": False, "status": None}
    result = path / "result.json" if path.is_dir() else path
    status = None
    if result.exists():
        try:
            status = json.loads(result.read_text()).get("status")
        except (ValueError, AttributeError):
            status = None
    return {"exists": True, "status": status}


def cmd_research(a):
    book, hyp = runbook()
    if a.action == "list":
        for h in sorted(hyp):
            if not h.startswith("H"):
                continue
            entries = book.get(h, [])
            print(f"{h} [{hyp[h].get('kind')}] status {hyp[h].get('status')}: {hyp[h].get('statement', '').strip()}")
            if hyp[h].get("evidence"):
                print(f"    recorded evidence: {', '.join(hyp[h]['evidence'])}")
            for e in entries:
                st = evidence_state(e)
                state = f"{'present' if st['exists'] else 'MISSING'}" + (f" ({st['status']})" if st["status"] else "")
                print(f"    {e['name']:45} {e['device']:4} {e['model']:10} {'fault ' if e.get('fault') else ''}{state}")
        return OK
    h = a.hypothesis.upper()
    if h not in book:
        raise SystemExit(f"unknown hypothesis {h!r}: one of {sorted(book)}")
    code = OK
    for e in book[h]:
        cmd = [sys.executable, *e["argv"]]
        env_prefix = 'CUDA_VISIBLE_DEVICES="" ' if e["device"] == "cpu" else ""
        line = env_prefix + shlex.join(["python", *e["argv"]])
        if evidence_state(e)["exists"]:
            print(f"skip {e['name']}: research/l2l/experiments/{e['output']} exists (never overwritten)")
            continue
        if not a.execute:
            print(f"[dry run] {e['name']} ({e['device']}, model {e['model']}):\n    {line}")
            continue
        if e["device"] == "cuda":
            import torch
            if not torch.cuda.is_available():
                print(f"BLOCKED {e['name']}: needs a CUDA GPU")
                code = CHECK_FAILED
                continue
        missing = [r for r in e.get("requires_runs", []) if not (ROOT / "runs" / r / "training" / "training.json").exists()]
        if missing:
            print(f"BLOCKED {e['name']}: needs the training runs {missing}")
            code = CHECK_FAILED
            continue
        env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": ""} if e["device"] == "cpu" else {})}
        print(f"$ {line}", flush=True)
        if subprocess.run(cmd, cwd=ROOT, env=env).returncode != 0:
            code = CHECK_FAILED
    if not a.execute:
        print("\nnothing executed: add --execute to run the commands above (cuda entries need a GPU; Qwen3 entries download "
              "weights)")
    return code


def cmd_soup(a):
    from qwen_grpo.integrations import soup
    return soup.cli(a)


def cmd_faults(a):
    from qwen_grpo.diagnostics import silent_failures
    return silent_failures.cli(a)


# --- parser ---------------------------------------------------------------------------------------------------

def parser():
    ap = argparse.ArgumentParser(prog="qwen-grpo", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="configuration, resource and hardware checks before GPU work")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--gpu", action="store_true", help="also run the GPU checks (needs the target GPU; loads the model)")
    p.add_argument("--hardware-only", action="store_true", help="with --gpu: hardware checks only, no model steps")
    p.add_argument("--allow-metadata-download", action="store_true",
                   help="fetch config.json / safetensors headers (kilobytes, never weights) when not cached")
    p.add_argument("--pcie-gbps", type=float, help="effective pinned H2D GB/s for the time estimate (else assumed 12)")
    p.add_argument("--host-ram-gib", type=float, help="host RAM of the target machine (else this host's MemTotal)")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("train", help="start a new run")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--skip-preflight", action="store_true",
                   help="start without the GPU-preflight record a memory-guarded config requires (recorded in the log)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("resume", help="continue an interrupted run")
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", help="a checkpoint of the run (default: latest valid)")
    p.set_defaults(func=cmd_resume)

    for name, helptext in (("validate", "evaluate a checkpoint on the validation split (resident runs)"),
                           ("test", "evaluate a checkpoint on the test split (resident runs; final report only)")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--run", required=True)
        p.add_argument("--step", type=int)
        p.set_defaults(func=cmd_run_py)

    p = sub.add_parser("verify", help="base / trained / merged / LR=0 comparison with CIs")
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, help="resident runs: checkpoint step (default latest)")
    p.add_argument("--control", help="L2L runs: the LR=0 control run")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, help="L2L runs: first n test examples (default all)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("merge", help="streamed LoRA merge of a run's checkpoint")
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--out")
    p.add_argument("--evaluate", action="store_true", help="greedy validation predictions of the merged model (L2L)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=64)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("report", help="rebuild a run's report from its files")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_run_py)

    p = sub.add_parser("remove", help="delete a run's outputs (dry run unless --confirm)")
    p.add_argument("--run", required=True)
    p.add_argument("--confirm", action="store_true")
    p.set_defaults(func=cmd_run_py)

    p = sub.add_parser("research", help="hypotheses, evidence and the commands that test them")
    rs = p.add_subparsers(dest="action", required=True)
    rs.add_parser("list")
    r = rs.add_parser("run")
    r.add_argument("hypothesis", help="e.g. H06")
    r.add_argument("--execute", action="store_true", help="run the commands (default: print them)")
    p.set_defaults(func=cmd_research)

    p = sub.add_parser("ship", help="the evidence-derived SHIP / DON'T SHIP verdict")
    p.add_argument("--soup", action="store_true", help="also run Soup's offline ship check on the held-out scores")
    p.add_argument("--soup-bin", help="the Soup executable (default: $SOUP_BIN, then PATH)")
    p.set_defaults(func=cmd_ship)

    p = sub.add_parser("soup", help="optional Soup diagnostics")
    from qwen_grpo.integrations import soup as soup_mod
    soup_mod.add_arguments(p)
    p.set_defaults(func=cmd_soup)

    p = sub.add_parser("faults", help="silent-failure registry")
    from qwen_grpo.diagnostics import silent_failures as sf_mod
    sf_mod.add_arguments(p)
    p.set_defaults(func=cmd_faults)
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:  # argparse: --help (0) or a usage error (2 -> USAGE)
        return OK if e.code == 0 else USAGE
    record = {"command": args.command, "argv": argv, "start": utc()}
    t0 = time.time()
    try:
        code = args.func(args)
    except SystemExit as e:  # refusals from the implementations carry their reason as the message
        if e.code not in (None, 0) and not isinstance(e.code, int):
            print(f"error: {e.code}", file=sys.stderr)
        code = OK if e.code in (None, 0) else e.code if isinstance(e.code, int) else ERROR
    except KeyboardInterrupt:
        code = ERROR
        print("interrupted", file=sys.stderr)
    history({**record, "end": utc(), "seconds": round(time.time() - t0, 3), "exit": code})
    return code


if __name__ == "__main__":
    sys.exit(main())
