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
