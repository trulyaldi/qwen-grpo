"""Crash-consistent checkpoint publication and validation (Ticket 09).

publish(final, write): write(tmp) fills a temporary sibling directory (checkpoints/.tmp-step<N>-<pid>/); every file
is fsynced, COMMIT.json (the step and the sha256 and size of every file) is written and fsynced last, then the
directory is renamed to its final name (atomic on one file system) and the parent directory is fsynced. So a
checkpoint directory either does not exist under its final name or is complete: an interrupted publication leaves only
a .tmp-* directory, which runs.checkpoints never lists and prepare() removes.
validate(path): a checkpoint is valid iff COMMIT.json exists and every listed file exists with its recorded size and
sha256 (and nothing is missing); a resumable one must also hold training_state.pt.
Commit semantics: an optimizer update is committed when the checkpoint that contains it is published. A resumed run
restarts from the latest valid checkpoint, so a committed update is never applied twice (its effects are in the
restored adapter and optimizer state and the step counter), and an uncommitted one (after that checkpoint) is replayed.
prepare(run): before resuming, removes interrupted publications and quarantines invalid checkpoints (renamed to
.invalid-step<N>-<stamp>, kept for inspection), so they are ignored and the replay can publish those steps again.
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from qwen_grpo import runs
from qwen_grpo.checkpointing import STATE
from qwen_grpo.streaming.dataset import fsync_dir

COMMIT = "COMMIT.json"


def _file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _crash_point(name):
    """Crash-injection hook inside publication (tests replace it); a no-op otherwise."""


def publish(final, write, step):
    final = Path(final)
    if final.exists():
        raise RuntimeError(f"{final} already exists: a published checkpoint is never overwritten (prepare() first)")
    tmp = final.with_name(f".tmp-{final.name}-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    write(tmp)
    files = {}
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            with open(p, "rb") as f:
                os.fsync(f.fileno())
            files[str(p.relative_to(tmp))] = {"bytes": p.stat().st_size, "sha256": _file_sha(p)}
    _crash_point("files_written")
    with open(tmp / COMMIT, "w") as f:
        json.dump({"step": step, "files": files}, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    fsync_dir(tmp)
    _crash_point("committed_before_rename")
    os.rename(tmp, final)
    fsync_dir(final.parent)
    return final


def validate(path, resumable=True):
    """(ok, reason)."""
    path = Path(path)
    commit = path / COMMIT
    if not commit.is_file():
        return False, "no COMMIT.json (incomplete or legacy checkpoint)"
    try:
        meta = json.loads(commit.read_text())
    except json.JSONDecodeError as e:
        return False, f"COMMIT.json unreadable: {e}"
    present = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p.name != COMMIT}
    if missing := sorted(set(meta["files"]) - present):
        return False, f"missing files {missing}"
    for name, f in meta["files"].items():
        p = path / name
        if p.stat().st_size != f["bytes"] or _file_sha(p) != f["sha256"]:
            return False, f"{name} does not match its recorded size / sha256"
    if resumable and STATE not in meta["files"]:
        return False, f"no {STATE}: not resumable"
    return True, None


def latest_valid(name):
    """(path of the highest-step valid resumable checkpoint or None, [(path, reason)] of the invalid ones)."""
    best, rejected = None, []
    for step, p in runs.checkpoints(name):
        ok, why = validate(p, resumable=step > 0)  # step0 is the initial adapter: valid, but never resumed from
        if not ok:
            rejected.append((p, why))
        elif step > 0:
            best = p
    return best, rejected


def prepare(name):
    """Remove interrupted publications and quarantine invalid checkpoints.
    -> {"removed_tmp": [...], "quarantined": [(path, reason)]}."""
    d = runs.run_dir(name) / "checkpoints"
    out = {"removed_tmp": [], "quarantined": []}
    if not d.is_dir():
        return out
    for p in d.glob(".tmp-step*"):
        shutil.rmtree(p)
        out["removed_tmp"].append(runs.rel(p))
    stamp = time.strftime("%Y%m%dT%H%M%S")
    for step, p in runs.checkpoints(name):
        ok, why = validate(p, resumable=step > 0)
        if not ok:
            dest = p.with_name(f".invalid-{p.name}-{stamp}")
            os.rename(p, dest)
            out["quarantined"].append((runs.rel(dest), why))
    fsync_dir(d)
    return out
