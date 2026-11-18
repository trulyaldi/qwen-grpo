"""Step 0 theoretical memory and PCIe budget: python scripts/budget.py --config configs/budget/qwen3-4b.yaml

Reads config.json and safetensors headers only (cached files, else the Hub's header metadata); loads no weights and
touches no GPU. Prints the report and writes results/budget/predictions/<run_name>.{json,md}, never overwriting.
"""
import argparse
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo.budget import architecture, calculator  # noqa: E402


def draft_bytes(cfg):
    if not (cfg["dflash"]["estimate"] or cfg["dflash"]["enabled"]):
        return None
    try:
        tensors, _, _ = architecture.checkpoint_tensors(cfg["dflash"]["draft_model_id"], None)
    except architecture.MetadataError as e:
        print(f"DFlash draft metadata unavailable, reported as unknown: {e}", file=sys.stderr)
        return None
    return sum(t.stored_bytes for t in tensors.values())


def draft_architecture(cfg):
    """The drafter's config.json (cached, else downloaded: a few KB) as calculator.DRAFT_KEYS; None if DFlash is off or
    the file is unavailable (the DFlash rollout is then reported with critical unknowns)."""
    if not cfg["dflash"]["enabled"]:
        return None
    import json
    from huggingface_hub import hf_hub_download
    try:
        raw = json.loads(Path(hf_hub_download(cfg["dflash"]["draft_model_id"], "config.json")).read_text())
    except Exception as e:  # noqa: BLE001 - reported, the prediction then says unknown
        print(f"DFlash drafter config unavailable, its activations reported as unknown: {e}", file=sys.stderr)
        return None
    return calculator.draft_config(raw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=ROOT / "results/budget/predictions")
    args = ap.parse_args()
    cfg = calculator.load_config(args.config)
    arch, _ = architecture.load(cfg["model"]["id"], cfg["lora"]["targets"])
    report = calculator.build_report(cfg, arch, draft_bytes(cfg), draft_architecture(cfg))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    report["provenance"] = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "config": str(args.config),
                            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                            "git_commit": commit or None, "kind": "theoretical prediction (no measurement)"}
    md = calculator.render_markdown(report)
    print(md)
    for p in calculator.write_prediction(report, md, args.out_dir):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
