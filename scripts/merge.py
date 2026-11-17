"""Merge a LoRA checkpoint into its base model tensor by tensor (Ticket 10), and optionally evaluate the merged model
through the L2L engine (never model.generate(), never the whole model on the GPU).

    python scripts/merge.py --adapter runs/qwen/checkpoints/step90 --out merged/qwen_step90
    python scripts/merge.py --adapter runs/qwen/checkpoints/step90 --out merged/qwen_step90 --evaluate --device cuda --n 64

--base defaults to the adapter's base_model_name_or_path (resolved to its local cached snapshot). The merged directory
is a Hugging Face checkpoint (shards + index + config + tokenizer files) with merge_report.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base")
    ap.add_argument("--shard-gib", type=float, default=1.0)
    ap.add_argument("--evaluate", action="store_true", help="greedy validation predictions of the merged model (L2L)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=64, help="validation examples to evaluate")
    ap.add_argument("--storage", choices=["pinned", "pageable"], default=None)
    a = ap.parse_args(argv)
    from qwen_grpo.export.merge import load_merged_l2l, stream_merge
    base = a.base or json.loads((a.adapter / "adapter_config.json").read_text())["base_model_name_or_path"]
    rep = stream_merge(base, a.adapter, a.out, shard_bytes=int(a.shard_gib * 2**30))
    worst = max(v["max_residual"] for v in rep["merged"].values())
    print(f"merged {len(rep['merged'])} tensors (scale {rep['scale']}), copied {rep['copied']}, {len(rep['shards'])} "
          f"shards, {rep['total_bytes'] / 2**20:.0f} MiB; worst residual {worst:.3e}, all within half an ulp: "
          f"{all(v['within_half_ulp'] for v in rep['merged'].values())} -> {a.out}")
    if a.evaluate:
        import torch
        from transformers import AutoTokenizer
        from qwen_grpo.evaluation import evaluate, metrics, summary
        from qwen_grpo.sentiment import load_splits
        storage = a.storage or ("pinned" if torch.device(a.device).type == "cuda" else "pageable")
        model, backend = load_merged_l2l(a.out, a.device, None, storage, 2)
        tok = AutoTokenizer.from_pretrained(a.out, padding_side="left")
        examples = [{"id": i, **r} for i, r in enumerate(load_splits()["validation"][: a.n])]
        preds = evaluate(model, tok, examples, generator=backend.greedy)
        print(f"merged model, {a.n} validation examples (L2L greedy): {summary(metrics(preds))}")
        if torch.device(a.device).type == "cuda":
            print(f"peak GPU allocated {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB (model "
                  f"{sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20:.0f} MiB)")
        backend.close()


if __name__ == "__main__":
    main()
