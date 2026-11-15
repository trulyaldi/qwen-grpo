"""Where experiment artifacts live: the only module that builds output paths.

Every run trained from now on is self-contained under runs/<name>/:

    config.yaml  manifest.json  summary.json  report.md  plots/
    training/{training.json, training.log, validation.json}
    checkpoints/step<N>/                     adapter (+ training_state.pt: resumable); step0 = initial adapter
    evaluation/step<N>/{validation,test}.json
    verification/step<N>/{report.md, plots/, evidence/}

runs/.<name>.lock holds the PID of the process training <name>. Runs from before this layout keep theirs and are only
read: checkpoints/<name>_step<N>/, results/<name>/, results/eval/, results/verification/, results/benchmarks/. The
saved baseline results/baseline/ is shared by all runs and belongs to none. Paths are derived from ROOT at call time
(tests point ROOT at a temporary directory).
"""
import datetime
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("running", "completed", "interrupted", "failed")


def rel(path):
    """Repository-relative path string if `path` is inside the repository, else the absolute path."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT.resolve())) if path.is_relative_to(ROOT.resolve()) else str(path)


# --- layout ------------------------------------------------------------------------------------------------------

def runs_dir():
    return ROOT / "runs"


def run_dir(name):
    return runs_dir() / name


def lock_path(name, directory=None):
    """<directory>/.<name>.lock (default runs/; legacy runs used results/)."""
    return Path(directory if directory is not None else runs_dir()) / f".{name}.lock"


def checkpoint_path(name, step):
    return run_dir(name) / "checkpoints" / f"step{step}"


def training_dir(name):
    return run_dir(name) / "training"


def evaluation_path(name, step, split):
    return run_dir(name) / "evaluation" / f"step{step}" / f"{split}.json"


def verification_dir(name, step):
    return run_dir(name) / "verification" / f"step{step}"


def baseline_path(split):
    """Saved base-model predictions (research/baseline.py), shared by every run."""
    return ROOT / "results" / "baseline" / f"{split}.json"


def legacy_results(name):
    return ROOT / "results" / name


def legacy_checkpoint(name, step):
    return ROOT / "checkpoints" / f"{name}_step{step}"


# --- checkpoints -------------------------------------------------------------------------------------------------

def parse_checkpoint(checkpoint):
    """(run name, step, layout): layout "run" for runs/<name>/checkpoints/step<N>, "legacy" for any <name>_step<N>
    directory (checkpoints/ before runs/ existed), else (None, None, None)."""
    p = Path(os.path.abspath(checkpoint))
    if (m := re.fullmatch(r"step(\d+)", p.name)) and p.parent.name == "checkpoints" \
            and p.parent.parent.parent == Path(os.path.abspath(runs_dir())):
        return p.parent.parent.name, int(m.group(1)), "run"
    if m := re.fullmatch(r"(.+)_step(\d+)", p.name):
        return m.group(1), int(m.group(2)), "legacy"
    return None, None, None


def sibling(checkpoint, step):
    """The checkpoint of the same run at another step, in the same layout."""
    name, _, layout = parse_checkpoint(checkpoint)
    return checkpoint_path(name, step) if layout == "run" else Path(checkpoint).parent / f"{name}_step{step}"


def training_log_for(checkpoint):
    name, _, layout = parse_checkpoint(checkpoint)
    if name is None:
        return None
    return training_dir(name) / "training.json" if layout == "run" else legacy_results(name) / "training.json"


def evaluation_output(checkpoint, split):
    name, step, layout = parse_checkpoint(checkpoint)
    if layout == "run":
        return evaluation_path(name, step, split)
    return ROOT / "results" / "eval" / Path(checkpoint).resolve().name / f"{split}.json"


def verification_output(checkpoint):
    """(report directory, evidence directory) of verify for this checkpoint."""
    name, step, layout = parse_checkpoint(checkpoint)
    if layout == "run":
        return verification_dir(name, step), verification_dir(name, step) / "evidence"
    out = ROOT / "results" / "verification" / Path(checkpoint).resolve().name
    return out, out


def checkpoints(name):
    """[(step, path)] of the run's adapters, by step."""
    d = run_dir(name) / "checkpoints"
    found = [(int(m.group(1)), p) for p in (d.iterdir() if d.is_dir() else [])
             if (m := re.fullmatch(r"step(\d+)", p.name)) and (p / "adapter_config.json").is_file()]
    return sorted(found)


def latest_checkpoint(name):
    """Highest-step adapter of the run (step0, the initial adapter, only if it is the only one), or None."""
    found = checkpoints(name)
    return found[-1][1] if found else None


def resolve(name, step=None):
    """(config path, checkpoint) of runs/<name> at `step` (default: the latest checkpoint)."""
    if not run_dir(name).is_dir():
        hint = (f"; {rel(legacy_results(name))} is a run from before runs/ existed: use --config and --checkpoint"
                if legacy_results(name).exists() else "")
        raise SystemExit(f"no run {name!r}: {rel(run_dir(name))} does not exist{hint}")
    ckpt = latest_checkpoint(name) if step is None else checkpoint_path(name, step)
    if ckpt is None or not (ckpt / "adapter_config.json").is_file():
        have = [s for s, _ in checkpoints(name)]
        raise SystemExit(f"run {name!r} has no checkpoint at step {step} (available: {have})")
    return run_dir(name) / "config.yaml", ckpt


def legacy_outputs(name):
    """Pre-runs/ artifacts named <name> (names matched in full): a new run of that name would make it ambiguous."""
    bench, ckpts = ROOT / "results" / "benchmarks", ROOT / "checkpoints"
    found = [legacy_results(name), bench / f"{name}.json", bench / f"{name}.md"]
    found += [p for p in (ckpts.iterdir() if ckpts.is_dir() else []) if re.fullmatch(rf"{re.escape(name)}_step\d+", p.name)]
    return sorted(p for p in found if p.exists())


# --- metadata ----------------------------------------------------------------------------------------------------

def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    return path


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def create(name, config_sha256, environment):
    """manifest.json of a new run: identity (unique id from name, creation time and config hash), provenance, status."""
    created = now()
    stamp = created.replace("-", "").replace(":", "").replace("+0000", "Z")
    return write_json(run_dir(name) / "manifest.json", {
        "run": name, "id": f"{name}-{stamp}-{config_sha256[:8]}", "created": created, "updated": created,
        "config_sha256": config_sha256, "git_commit": environment.get("git_commit"), "git_dirty": environment.get("git_dirty"),
        "status": "running", "status_history": [{"status": "running", "at": created}], "artifacts": []})


def manifest(name):
    return read_json(run_dir(name) / "manifest.json")


def set_status(name, status, **extra):
    assert status in STATUSES, status
    m = manifest(name)
    t = now()
    m.update(status=status, updated=t, **extra)
    m["status_history"].append({"status": status, "at": t, **extra})
    write_json(run_dir(name) / "manifest.json", m)


def status(name, lock_owner):
    """The recorded status, except that a run recorded as running with no live training process was interrupted
    (killed without a chance to record it). `lock_owner(path)` returns the live PID holding a lock, or None."""
    m = manifest(name)
    if m is None:
        return None
    if m["status"] == "running" and lock_owner(lock_path(name)) is None:
        return "interrupted"
    return m["status"]


def artifacts(name):
    """Every file under runs/<name>/, relative to it, sorted (what the manifest lists)."""
    d = run_dir(name)
    return sorted(str(Path(top, f).relative_to(d)) for top, _, files in os.walk(d) for f in files)


def record_artifacts(name):
    m = manifest(name)
    m["artifacts"] = [a for a in artifacts(name) if a != "manifest.json"]
    m["updated"] = now()
    write_json(run_dir(name) / "manifest.json", m)
