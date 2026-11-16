"""Ticket 00 resident baseline: python scripts/l2l_baseline.py --junit research/l2l/baseline_tests.xml

Records, in research/l2l/baseline.json: the source commit and worktree state, library/hardware versions, model and
tokenizer revisions of the local HF cache, config hashes, the CPU test-suite outcome parsed from pytest's JUnit XML
(every failure must have a documented cause, else it is reported UNCLASSIFIED), and the measured contents of the
resident reference fixtures. Loads no pretrained weights and does not initialize CUDA.

The suite is run separately, so its raw log is kept as produced:
CUDA_VISIBLE_DEVICES="" python -m pytest -q -rA --continue-on-collection-errors \
    --junitxml=research/l2l/baseline_tests.xml 2>&1 | tee research/l2l/baseline_tests.log
"""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from qwen_grpo import config  # noqa: E402
from qwen_grpo.execution import fixtures as F  # noqa: E402

OUT = ROOT / "research" / "l2l" / "baseline.json"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B")
CONFIGS = ("configs/default.yaml", "configs/qwen.yaml")

# Failures that exist before Ticket 00's code, with their cause (docs/l2l/resident_reference.md, "Baseline tests").
KNOWN_FAILURES = {
    "tests/test_trl_reference.py": "collection error: imports research/trl_reference, deleted in 4424e0f",
    "tests.test_config::test_new_runs_reject_the_historical_schema_with_the_offending_keys":
        "48b2028 rewrote configs/default.yaml in the current schema; the test still expects the historical one",
    "tests.test_config::test_historical_configs_in_the_repository_still_load_as_records":
        "48b2028 rewrote configs/default.yaml; the test still expects its historical values (balanced, B=2, 32 samples)",
    "tests.test_config::test_current_configs_load_with_distinct_run_names":
        "pre-existing before 48b2028: expects >= 2 current-schema configs besides default.yaml, finds only configs/qwen.yaml",
    "tests.test_historical::test_saved_log_and_checkpoint_configs_are_the_reference_yaml":
        "environment: checkpointing.load_training_state calls torch.load without map_location; the saved state holds "
        "CUDA tensors and CUDA is hidden (CPU suite). Not run with a GPU.",
    "tests.test_historical::test_reference_run_resumes_under_the_current_code":
        "environment: same torch.load without map_location on CUDA-saved state, CUDA hidden. Not run with a GPU.",
}


def run(*args):
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git():
    tracked = run("git", "status", "--porcelain", "--untracked-files=no")
    return {"commit": run("git", "rev-parse", "HEAD"), "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "tracked_dirty": None if tracked is None else bool(tracked),
            "untracked": (run("git", "status", "--porcelain") or "").count("??")}


def environment():
    gpu = run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    mem = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal"))
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "datasets", "pyyaml")},
            "torch_cuda": torch.version.cuda, "nvidia_smi_gpu": gpu, "cpu_count": torch.multiprocessing.cpu_count(),
            "mem_total": mem.split(":")[1].strip(), "torch_default_threads": torch.get_num_threads()}


def revisions():
    """Snapshot hash the local HF cache resolves `main` to (model weights and tokenizer share the repo)."""
    out = {}
    for m in MODELS:
        ref = HUB / f"models--{m.replace('/', '--')}" / "refs" / "main"
        out[m] = ref.read_text().strip() if ref.exists() else None
    return out


def configs():
    return {p: {"sha256": config.sha256(config.load(ROOT / p)), "model": config.load(ROOT / p).model.name} for p in CONFIGS}


def tests(junit):
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    totals = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    failed = []
    for case in root.iter("testcase"):
        for kind in ("failure", "error"):
            if (el := case.find(kind)) is not None:
                cls, name = case.get("classname", ""), case.get("name", "")
                key = f"{cls}::{name}" if name else cls
                if cls == "" or name in ("", cls.rsplit(".", 1)[-1]):  # collection error: report the file
                    key = cls.replace(".", "/") + ".py"
                cause = KNOWN_FAILURES.get(key)
                failed.append({"test": key, "kind": kind, "message": (el.get("message") or "").splitlines()[0][:300],
                               "cause": cause or "UNCLASSIFIED"})
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return {"junit": str(Path(junit).resolve().relative_to(ROOT)), **totals, "failed": failed,
            "unclassified": [f["test"] for f in failed if f["cause"] == "UNCLASSIFIED"]}


def fixture(shape):
    """What the saved fixture contains, measured from the file (no recomputation)."""
    fx = torch.load(F.path(shape), weights_only=True)
    r, ref = fx["rollouts"], fx["reference"]
    grads = {n: g.norm().item() for n, g in ref["grads"].items()}
    return {"file": str(F.path(shape).relative_to(ROOT)), "sha256": F.file_sha256(F.path(shape)), "meta": fx["meta"],
            "n_state_tensors": len(fx["state"]), "n_lora_tensors": len(ref["grads"]),
            "input_ids_shape": list(r["input_ids"].shape), "completion_lengths": r["completion_mask"].sum(1).tolist(),
            "advantages": fx["advantages"].tolist(), "old_logp_range": [fx["old_logp"].min().item(), fx["old_logp"].max().item()],
            "iteration0_loss": ref["loss"].item(),
            "lora_grad_norm": {"min": min(grads.values()), "max": max(grads.values())},
            "updates": [{k: u[k] for k in ("iteration", "loss", "init_diff", "grad_norm", "update_norm", "ratio_min",
                                           "ratio_max", "clip_frac", "outside_frac", "approx_kl") if k in u}
                        for u in ref["updates"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--junit", required=True)
    ap.add_argument("--log", default="research/l2l/baseline_tests.log")
    a = ap.parse_args()
    t = tests(ROOT / a.junit)
    out = {"ticket": "00", "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "source": git(), "environment": environment(), "model_revisions": revisions(), "configs": configs(),
           "tests": {**t, "log": a.log,
                     "command": 'CUDA_VISIBLE_DEVICES="" python -m pytest -q -rA --continue-on-collection-errors '
                                "--junitxml=research/l2l/baseline_tests.xml"},
           "fixtures": {s: fixture(s) for s in F.SHAPES},
           "tolerances": {"fixture_inputs": 0, "fixture_reference_outputs": 0,
                          "why": "same code, kernels, thread count (fixtures.THREADS) and reduction order on CPU"},
           "status": "PASS" if not t["unclassified"] else "FAIL"}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{OUT.relative_to(ROOT)}: {t['passed']} passed, {t['failures']} failed, {t['errors']} errors, "
          f"unclassified {t['unclassified']} -> {out['status']}")


if __name__ == "__main__":
    main()
