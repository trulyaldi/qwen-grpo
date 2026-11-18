"""Reading recorded experiment results under research/l2l/experiments/, compressed or not (no torch import).

A lab case writes <name>/result.json. Large historical results are stored losslessly as <name>/result.json.gz
(`scripts/experiment.py index --compress-over-mib N`): the decompressed bytes are exactly the original file, whose
sha256 research/l2l/evidence_index.json records. Every reader goes through read_result(), so a compressed result is
judged exactly like an uncompressed one.
"""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "research" / "l2l" / "experiments"
RESULT_FILES = ("result.json", "result.json.gz")


def result_path(directory):
    """The result file of a lab experiment directory (plain or gzip), or None."""
    return next((Path(directory) / n for n in RESULT_FILES if (Path(directory) / n).exists()), None)


def read_bytes(path):
    path = Path(path)
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def read_result(directory):
    """The parsed result of a lab experiment directory, or None if it has none."""
    path = result_path(directory)
    return None if path is None else json.loads(read_bytes(path))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


# --- code identity -------------------------------------------------------------------------------------------------
# What an experiment's result depends on, by content rather than by commit: every tracked or untracked (not ignored)
# file under these paths. A commit alone misses uncommitted edits and untracked modules; a "dirty" flag alone cannot
# tell two different dirty trees apart. The report generator is excluded: changing it changes how evidence is
# summarised, never the evidence (report_code_sha256 identifies it separately).
EXPERIMENT_PATHS = ("qwen_grpo", "scripts", "configs", "data", "tests", "research/l2l/hypotheses.yaml",
                    "research/l2l/tolerances.yaml", "requirements.txt", "pyproject.toml")
REPORT_PATHS = ("scripts/assignment_report.py",)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_files(root, paths):
    """Tracked and untracked, not git-ignored files under `paths` of the repository at `root` (None without git)."""
    import subprocess
    r = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                        *paths], capture_output=True)
    return None if r.returncode else sorted(set(p for p in r.stdout.decode().split("\0") if p))


def _content_sha256(root, files):
    h = hashlib.sha256()
    for f in files:
        p = Path(root) / f
        h.update(f.encode() + b"\0" + (file_sha256(p).encode() if p.is_file() else b"<deleted>") + b"\n")
    return h.hexdigest()


def code_sha256(root=ROOT):
    """Content identity of the experiment code at `root` (None if it cannot be established)."""
    files = code_files(root, EXPERIMENT_PATHS)
    return None if files is None else _content_sha256(root, [f for f in files if f not in REPORT_PATHS])


def report_code_sha256(root=ROOT):
    return _content_sha256(root, list(REPORT_PATHS))
