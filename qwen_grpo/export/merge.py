"""Streaming LoRA merge (Ticket 10): W_merged = W0 + (alpha / r) B A, one tensor at a time, on the CPU, into a
Hugging Face checkpoint of bounded shards. No model is built and nothing touches a GPU.

stream_merge(base, adapter, out):
  base     a Hugging Face checkpoint directory (config.json + model.safetensors or a sharded index), or a hub name
           resolved to its local cached snapshot (never downloaded here)
  adapter  a PEFT checkpoint directory (adapter_config.json + adapter_model.safetensors)
  out      written crash-consistently: a temporary sibling directory renamed into place at the end
Per base tensor (read lazily from the safetensors files, so only the current tensor is materialized):
  - a LoRA target "<module>.weight": W = round_to_base_dtype(fp32(W0) + (alpha / r) * fp32(B) @ fp32(A)), one rounding
    of the exact fp32 sum (PEFT's merge_and_unload instead rounds the delta to the base dtype and then adds in it: two
    roundings); its residual against the fp32 sum, the realized delta and the intended delta are recorded;
  - every other tensor is copied unchanged (same name, dtype and bytes), so tied embeddings / LM head stay exactly
    as the base stores them (an adapter on embed_tokens or lm_head is refused).
Tensors are written in shards of at most `shard_bytes` (plus one tensor), each released after writing, with
model.safetensors.index.json; config.json, generation_config.json and the tokenizer files are copied. Peak memory:
one shard buffer + one tensor + the adapter (small), never a full model copy.
Refused (never silently merged wrong): rsLoRA / DoRA / fan_in_fan_out / bias / dropout != 0 adapters, adapter keys
without a matching base tensor, a base tensor of an unexpected shape, an unused adapter module.
"""
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json", "generation_config.json")


def resolve_base(base):
    """A local checkpoint directory: `base` itself, or the cached hub snapshot of the model name `base`."""
    p = Path(base)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(base), local_files_only=True,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))


def base_files(base_dir):
    """{tensor name: safetensors file} of a checkpoint (single file or index)."""
    index = base_dir / "model.safetensors.index.json"
    if index.exists():
        wm = json.loads(index.read_text())["weight_map"]
        return {k: base_dir / v for k, v in wm.items()}
    f = base_dir / "model.safetensors"
    if not f.exists():
        raise FileNotFoundError(f"{base_dir}: no model.safetensors or model.safetensors.index.json")
    with safe_open(f, framework="pt") as h:
        return {k: f for k in h.keys()}


def read_adapter(adapter_dir):
    """(scale = alpha / r, {base weight name: (A fp32 [r, in], B fp32 [out, r])}, adapter config)."""
    adapter_dir = Path(adapter_dir)
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    for key, ok in (("use_rslora", False), ("use_dora", False), ("fan_in_fan_out", False), ("bias", "none")):
        if cfg.get(key, ok) != ok:
            raise ValueError(f"adapter {key}={cfg.get(key)!r} is not supported by the streaming merge")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError("per-module rank_pattern / alpha_pattern adapters are not supported")
    scale = cfg["lora_alpha"] / cfg["r"]
    pairs = {}
    with safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as h:
        for k in h.keys():
            if not k.endswith(".lora_A.weight"):
                if not k.endswith(".lora_B.weight"):
                    raise ValueError(f"unexpected adapter tensor {k}")
                continue
            module = k[: -len(".lora_A.weight")].removeprefix("base_model.model.")
            A = h.get_tensor(k).float()
            B = h.get_tensor(k.replace(".lora_A.weight", ".lora_B.weight")).float()
            if A.shape[0] != cfg["r"] or B.shape[1] != cfg["r"]:
                raise ValueError(f"{module}: A {tuple(A.shape)}, B {tuple(B.shape)} do not have rank {cfg['r']}")
            pairs[f"{module}.weight"] = (A, B)
    if any(n.startswith(("model.embed_tokens", "lm_head")) for n in pairs):
        raise ValueError("adapters on embed_tokens / lm_head are not supported (tied weights)")
    return scale, pairs, cfg


def _ulp_bound(W):
    """Half an ulp of every element of W's dtype at its magnitude: the largest error of one correct rounding."""
    eps = torch.finfo(W.dtype).eps
    return 0.5 * eps * W.float().abs()


def stream_merge(base, adapter, out, shard_bytes=1 << 30, dtype=None):
    """Merge `adapter` into `base`, writing a Hugging Face checkpoint to `out`. Returns the merge report:
    {"scale", "merged": {name: residual stats}, "copied": n, "shards": [...], "total_bytes"}."""
    base_dir, out = resolve_base(base), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: a merged checkpoint is never overwritten")
    scale, pairs, acfg = read_adapter(adapter)
    files = base_files(base_dir)
    if missing := sorted(set(pairs) - set(files)):
        raise ValueError(f"adapter modules without a base tensor: {missing[:5]}")
    tmp = out.with_name(f".tmp-{out.name}-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    report = {"scale": scale, "r": acfg["r"], "alpha": acfg["lora_alpha"], "base": str(base_dir), "adapter": str(adapter),
              "merged": {}, "copied": 0, "shards": [], "total_bytes": 0}
    weight_map, shard, shard_size, n_shards = {}, {}, 0, 0
    handles = {}

    def flush():
        nonlocal shard, shard_size, n_shards
        if not shard:
            return
        n_shards += 1
        name = f"shard-{n_shards:05d}.safetensors"
        save_file(shard, str(tmp / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        shard, shard_size = {}, 0

    try:
        for name in sorted(files):
            f = files[name]
            if f not in handles:
                handles[f] = safe_open(f, framework="pt").__enter__()
            W0 = handles[f].get_tensor(name)
            if name in pairs:
                A, B = pairs[name]
                if tuple(W0.shape) != (B.shape[0], A.shape[1]):
                    raise ValueError(f"{name}: base {tuple(W0.shape)} vs adapter B @ A {(B.shape[0], A.shape[1])}")
                delta = scale * (B @ A)                                            # fp32 [out, in]
                exact = W0.float() + delta
                W = exact.to(dtype or W0.dtype)
                residual = (W.float() - exact).abs()
                realized = W.float() - W0.float()
                report["merged"][name] = {
                    "dtype": str(W.dtype), "max_residual": residual.max().item(),
                    "within_half_ulp": bool((residual <= _ulp_bound(W) + 1e-30).all()),
                    "delta_norm": delta.norm().item(), "delta_max": delta.abs().max().item(),
                    "realized_delta_norm": realized.norm().item(),
                    "delta_relative_error": ((realized - delta).norm() / delta.norm()).item() if delta.norm() > 0 else None,
                    "changed_elements": int((W != W0.to(W.dtype)).sum())}
                del A, B, delta, exact, residual, realized
            else:
                W = W0 if dtype is None else W0.to(dtype)
                report["copied"] += 1
            shard[name] = W.contiguous()
            shard_size += W.numel() * W.element_size()
            report["total_bytes"] += W.numel() * W.element_size()
            del W0, W
            if shard_size >= shard_bytes:
                flush()
        flush()
    finally:
        for h in handles.values():
            h.__exit__(None, None, None)
    if unused := sorted(set(pairs) - set(report["merged"])):
        raise ValueError(f"adapter modules not merged: {unused}")
    # final shard names (model-0000k-of-0000n) and the index
    renames = {f"shard-{i:05d}.safetensors": f"model-{i:05d}-of-{n_shards:05d}.safetensors" for i in range(1, n_shards + 1)}
    for a, b in renames.items():
        os.rename(tmp / a, tmp / b)
    weight_map = {k: renames[v] for k, v in weight_map.items()}
    (tmp / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": report["total_bytes"]}, "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    report["shards"] = sorted(renames.values())
    config = json.loads((base_dir / "config.json").read_text())
    if dtype is not None:
        config["torch_dtype"] = str(dtype).removeprefix("torch.")
    (tmp / "config.json").write_text(json.dumps(config, indent=2))
    for name in TOKENIZER_FILES:
        if (base_dir / name).exists():
            shutil.copy(base_dir / name, tmp / name)
    (tmp / "merge_report.json").write_text(json.dumps({k: v for k, v in report.items()}, indent=1))
    os.rename(tmp, out)
    return report


def load_merged_l2l(path, device, dtype=None, weight_storage="pinned", weight_buffers=2, kv_device=None):
    """The merged checkpoint as a plain Hugging Face model (no PEFT), loaded on the CPU and run by the L2L engine:
    frozen decoder weights in a WeightStore, the rest on `device` (execution.backend.L2LBackend). Never resident on the
    GPU as a whole; generation through execution.generator, never model.generate()."""
    from transformers import AutoModelForCausalLM

    from qwen_grpo.execution.backend import L2LBackend
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype or "auto").eval().requires_grad_(False)
    return model, L2LBackend(model, device, weight_storage, weight_buffers, kv_device=kv_device)
