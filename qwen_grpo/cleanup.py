"""`scripts/run.py remove`: delete the generated artifacts of one training run, refusing whenever ownership is not
certain. Dry run unless confirmed.

remove --run <N> (remove_run): a run under runs/ owns exactly runs/<N>/ and a stale runs/.<N>.lock.

remove --config --checkpoint (remove), for runs from before runs/ existed: a run <N> owns exactly (names matched in
full, never by prefix):

    checkpoints/<N>_step<k>/                 adapters + training state           (trainer)
    results/<N>/                             training.json, validation.json, plots/ (trainer)
    results/benchmarks/<N>.json, <N>.md      final metrics                       (trainer)
    results/eval/<N>_step<k>/                validate / test outputs             (scripts/run.py)
    results/verification/<N>_step<k>/        verify outputs                      (scripts/run.py)
    results/.<N>.lock                        only if stale (its process is gone)

Nothing else is ever deleted: no parent directory, configs, data, baselines, or another run's files.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from qwen_grpo import runs
from qwen_grpo.checkpointing import check_adapter_matches, lock_owner, lock_path
from qwen_grpo.config import RUN_NAME, as_dict
from qwen_grpo.verification import comparable

SCANNED = ("*.json", "*.md", "*.yaml", "*.py")  # files searched for references to the run's artifacts


def output_dirs(root):
    root = Path(root)
    return {"checkpoints": root / "checkpoints", "results": root / "results", "benchmarks": root / "results" / "benchmarks",
            "eval": root / "results" / "eval", "verification": root / "results" / "verification"}


def children(directory, pattern):
    """Entries of `directory` whose whole name matches `pattern` (no glob, no recursion, symlinks listed as such)."""
    return sorted(p for p in directory.iterdir() if re.fullmatch(pattern, p.name)) if directory.is_dir() else []


def artifacts(root, name):
    d, n = output_dirs(root), re.escape(name)
    found = [*children(d["checkpoints"], rf"{n}_step\d+"), *children(d["results"], n),
             *children(d["benchmarks"], rf"{n}\.(json|md)"), *children(d["eval"], rf"{n}_step\d+"),
             *children(d["verification"], rf"{n}_step\d+")]
    lock = lock_path(name, d["results"])
    if lock.exists() and lock_owner(lock) is None:
        found.append(lock)
    return found


def size(path):
    """Bytes on disk without following symlinks."""
    if not path.is_dir() or path.is_symlink():
        return path.lstat().st_size
    return sum(os.lstat(os.path.join(top, f)).st_size for top, _, files in os.walk(path, followlinks=False) for f in files)


def symlinks(path):
    """The path itself if it is a symlink, else every symlink inside it (directories included)."""
    if path.is_symlink() or not path.is_dir():
        return [path] if path.is_symlink() else []
    return [Path(top) / e for top, dirs, files in os.walk(path, followlinks=False) for e in dirs + files
            if (Path(top) / e).is_symlink()]


def git_tracked(root, paths):
    """Paths that contain files committed to git (deleting them frees no space: git keeps the objects)."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", *map(str, paths)], cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []  # not a git repository
    files = [Path(root) / f for f in out.stdout.decode().split("\0") if f]
    return [p for p in paths if any(f == p or p in f.parents for f in files)]


def references(root, paths):
    """Files outside `paths` that mention one of them by repository-relative path (another run's verification, a
    config's negative_control.checkpoint, documentation, research code)."""
    root = Path(root)
    rels = [p.relative_to(root).as_posix() for p in paths]
    pattern = re.compile("|".join(rf"(?<![\w.-]){re.escape(r)}(?![\w.-])" for r in rels))
    roots = [root / "runs", root / "results", root / "configs", root / "docs", root / "research", root / "README.md"]
    hits = {}
    for base in roots:
        files = [base] if base.is_file() else [f for g in SCANNED for f in base.rglob(g)] if base.is_dir() else []
        for f in files:
            if f.is_symlink() or any(f == p or p in f.parents for p in paths):
                continue
            if found := sorted(set(pattern.findall(f.read_text(errors="ignore")))):
                hits[f.relative_to(root).as_posix()] = found
    return hits


def ownership_problems(cfg, root, checkpoint, found):
    """Reasons the artifacts cannot be attributed to this config's run with certainty (empty list: they can)."""
    name, d = cfg.run.name, output_dirs(root)
    problems = []
    ckpt = Path(os.path.abspath(checkpoint))
    if ckpt.parent != Path(os.path.abspath(d["checkpoints"])) or not re.fullmatch(rf"{re.escape(name)}_step\d+", ckpt.name):
        return [f"{checkpoint} is not a checkpoint of run {name!r} (expected {d['checkpoints']}/{name}_step<N>)"]
    log_path = d["results"] / name / "training.json"
    log = json.loads(log_path.read_text()) if log_path.is_file() and not log_path.is_symlink() else None
    if not ckpt.exists() and log is None:
        return [f"neither {ckpt.name} nor a training log of {name!r} exists: nothing confirms the run's ownership"]
    listed = set()
    if log is not None:
        if "yaml_config" not in log:
            problems.append(f"{log_path.relative_to(root)} predates YAML configs: cannot confirm it came from this config")
        elif not comparable(log["yaml_config"], as_dict(cfg), set()):
            problems.append(f"{log_path.relative_to(root)} was produced by a different config than the one given")
        listed = {Path(e.get("checkpoint") or "").name for e in log.get("evaluations", [])}
        listed |= {Path(log.get("final_checkpoint", {}).get("path", "")).name, Path(log.get("config", {}).get("initial_checkpoint", "")).name}
    found = [p for p in found if not p.is_symlink()]  # symlinks are refused by plan(); never read through them
    for c in [p for p in found if p.parent == d["checkpoints"]]:
        if not (c / "adapter_config.json").is_file():
            problems.append(f"{c.relative_to(root)} is not a LoRA checkpoint (no adapter_config.json)")
            continue
        try:
            check_adapter_matches(cfg, c)
        except SystemExit as e:
            problems.append(f"{c.relative_to(root)}: {str(e).splitlines()[0]}")
        if log is not None and c.name not in listed and not (c / "training_state.pt").exists():
            problems.append(f"{c.relative_to(root)} is not listed in {log_path.relative_to(root)}")
    for kind in ("eval", "verification"):  # outputs of validate/test/verify: must name a checkpoint of this run
        for out in [p for p in found if p.parent == d[kind]]:
            docs = [json.loads(f.read_text()) for f in out.glob("*.json") if f.is_file() and not f.is_symlink()]
            named = {Path(x["checkpoint"]).name for x in docs if isinstance(x, dict) and x.get("checkpoint")}
            if named - {out.name}:
                problems.append(f"{out.relative_to(root)} names a checkpoint of another run: {sorted(named)}")
    return problems


def plan(cfg, root, checkpoint):
    """(paths with sizes) to delete for the run of `checkpoint`; raises SystemExit listing every reason to refuse."""
    root = Path(os.path.abspath(root))
    d = output_dirs(root)
    real_root = Path(os.path.realpath(root))
    if bad := [k for k, p in d.items() if p.exists() and p.resolve() != real_root / p.relative_to(root)]:
        raise SystemExit(f"refusing: output directories {bad} are or pass through symbolic links")
    found = artifacts(root, cfg.run.name)
    problems = ownership_problems(cfg, root, checkpoint, found)
    if (pid := lock_owner(lock_path(cfg.run.name, d["results"]))) is not None:
        problems.append(f"run {cfg.run.name!r} is being trained by process {pid}")
    allowed = {p.resolve() for p in d.values()}
    problems += [f"{p} is outside the experiment output directories" for p in found if p.parent.resolve() not in allowed]
    problems += [f"symbolic link {s.relative_to(root)}" for p in found for s in symlinks(p)]
    problems += [f"{p.relative_to(root)} is committed to git (deleting it frees no space; use git rm)" for p in git_tracked(root, found)]
    problems += [f"{f} references {', '.join(r)}" for f, r in references(root, found).items()] if found else []
    if problems:
        raise SystemExit("refusing to remove run {!r}:\n  ".format(cfg.run.name) + "\n  ".join(problems))
    return [(p, size(p)) for p in found]


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def execute(name, root, items, lock, confirm):
    """Print the plan; delete it only with confirm=True (re-checking that no training started meanwhile)."""
    print(f"run {name!r} ({'deleting' if confirm else 'dry run'}):")
    for p, n in items:
        print(f"  {human(n):>10}  {p.relative_to(root)}{'/' if p.is_dir() else ''}")
    print(f"  {human(sum(n for _, n in items)):>10}  total, {len(items)} paths" if items else "  nothing to remove")
    if not confirm:
        print("dry run: nothing deleted; add --confirm to delete these paths")
        return items
    if (pid := lock_owner(lock)) is not None:
        raise SystemExit(f"refusing: run {name!r} started training (process {pid})")
    for p, _ in items:
        shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
    print(f"deleted {len(items)} paths")
    return items


def remove(cfg, root, checkpoint, confirm=False):
    """A run from before runs/ existed. Returns the planned [(path, bytes)]."""
    root = Path(os.path.abspath(root))
    items = plan(cfg, root, checkpoint)
    return execute(cfg.run.name, root, items, lock_path(cfg.run.name, output_dirs(root)["results"]), confirm)


def plan_run(name):
    """[(path, bytes)] to delete for runs/<name>/; raises SystemExit listing every reason to refuse."""
    root = Path(os.path.abspath(runs.ROOT))
    if not re.fullmatch(RUN_NAME, name):
        raise SystemExit(f"refusing: {name!r} is not a run name")
    top, d, lock = runs.runs_dir(), runs.run_dir(name), runs.lock_path(name)
    if top.exists() and top.resolve() != Path(os.path.realpath(root)) / "runs":
        raise SystemExit(f"refusing: {top} is or passes through a symbolic link")
    if d.is_symlink():
        raise SystemExit(f"refusing: {d.relative_to(root)} is a symbolic link")
    if not d.is_dir():
        hint = (" (a run from before runs/ existed: use --config and --checkpoint)" if runs.legacy_outputs(name) else "")
        raise SystemExit(f"no run {name!r}: {d.relative_to(root)} does not exist{hint}")
    problems = []
    m = runs.read_json(d / "manifest.json") if (d / "manifest.json").is_file() and not (d / "manifest.json").is_symlink() else None
    if m is None or m.get("run") != name:
        problems.append(f"{d.relative_to(root)}/manifest.json is missing or names another run: ownership cannot be confirmed")
    if (pid := lock_owner(lock)) is not None:
        problems.append(f"run {name!r} is being trained by process {pid}")
    found = [d] + ([lock] if lock.exists() and not lock.is_symlink() else [])
    problems += [f"symbolic link {s.relative_to(root)}" for p in found for s in symlinks(p)]
    problems += [f"{p.relative_to(root)} is committed to git (deleting it frees no space; use git rm)" for p in git_tracked(root, found)]
    problems += [f"{f} references {', '.join(r)}" for f, r in references(root, found).items()]
    if problems:
        raise SystemExit(f"refusing to remove run {name!r}:\n  " + "\n  ".join(problems))
    return [(p, size(p)) for p in found]


def remove_run(name, confirm=False):
    """runs/<name>/: dry run unless confirm. Returns the planned [(path, bytes)]."""
    return execute(name, Path(os.path.abspath(runs.ROOT)), plan_run(name), runs.lock_path(name), confirm)
