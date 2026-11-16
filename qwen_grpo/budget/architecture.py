"""Exact tensor metadata of a Hugging Face checkpoint without loading weights: config.json + safetensors headers.

Every parameter is assigned to one group: the embedding, decoder layer i, the final norm, or the LM head. With
tie_word_embeddings the LM head is the embedding matrix; a checkpoint may still store a copy (Qwen3-0.6B does), which is
counted on disk but not in live memory.
"""
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from qwen_grpo.model import PRESETS, target_modules

SUPPORTED = ("qwen3",)  # dense Qwen3 only: the layer/LoRA naming and the shape checks below are written for it
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8,
               "F8_E4M3": 1, "F8_E5M2": 1, "bfloat16": 2, "float16": 2, "float32": 4}
REQUIRED = ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "vocab_size", "tie_word_embeddings")
LAYER = re.compile(r"model\.layers\.(\d+)\.(.+)")
GROUPS = {"model.embed_tokens.weight": "embed", "model.norm.weight": "norm", "lm_head.weight": "lm_head"}


class MetadataError(ValueError):
    """Required metadata is missing or inconsistent."""


class UnsupportedArchitecture(ValueError):
    """The calculator's accounting is not valid for this model family."""


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str  # as stored in the checkpoint

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def stored_bytes(self) -> int:
        return self.numel * DTYPE_BYTES[self.dtype]


@dataclass
class Architecture:
    model_id: str
    revision: str | None
    source: str                       # where the tensor metadata came from: "local safetensors" | "remote safetensors headers"
    config: dict                      # the REQUIRED fields
    embed: Tensor
    norm: Tensor
    lm_head: Tensor | None            # None when tied: the embedding serves as the LM head
    layers: list[list[Tensor]]        # layers[i] = tensors of decoder layer i
    tied_duplicates: list[Tensor] = field(default_factory=list)  # stored in the checkpoint, aliased once loaded
    shard_bytes: dict[str, int] = field(default_factory=dict)    # file -> stored tensor bytes
    lora_targets: list[Tensor] = field(default_factory=list)     # base weights adapted by LoRA

    def __getattr__(self, name):  # hidden_size, num_hidden_layers, ... read straight from the config
        if name in REQUIRED:
            return self.config[name]
        raise AttributeError(name)

    def layer_numel(self, i: int) -> int:
        return sum(t.numel for t in self.layers[i])

    @property
    def non_decoder(self) -> list[Tensor]:
        return [self.embed, self.norm] + ([self.lm_head] if self.lm_head else [])

    @property
    def live_numel(self) -> int:
        return sum(t.numel for t in self.non_decoder) + sum(map(self.layer_numel, range(len(self.layers))))

    @property
    def checkpoint_bytes(self) -> int:
        return sum(self.shard_bytes.values())

    @property
    def first_adapted_layer(self) -> int:
        return min(int(LAYER.fullmatch(t.name).group(1)) for t in self.lora_targets)


def _config_dict(config) -> dict:
    """Validate the model family and return the REQUIRED fields; `config` is a transformers config or a plain dict."""
    get = config.get if isinstance(config, dict) else lambda k, d=None: getattr(config, k, d)
    model_type = get("model_type")
    if model_type not in SUPPORTED:
        raise UnsupportedArchitecture(f"model_type {model_type!r}: supported {SUPPORTED}")
    if get("num_experts") or get("num_local_experts"):
        raise UnsupportedArchitecture("mixture-of-experts layers are not modelled")
    if missing := [k for k in REQUIRED if get(k) is None]:
        raise MetadataError(f"config lacks {missing}")
    return {k: get(k) for k in REQUIRED}


def from_metadata(model_id: str, config, tensors: dict[str, Tensor], shard_bytes: dict[str, int], source: str,
                  revision: str | None = None) -> Architecture:
    """Group checkpoint tensors and check them against the config. Pure: no I/O."""
    cfg = _config_dict(config)
    h, L, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    layers, named = [[] for _ in range(L)], {}
    for name, t in tensors.items():
        if m := LAYER.fullmatch(name):
            i = int(m.group(1))
            if i >= L:
                raise MetadataError(f"{name}: layer index >= num_hidden_layers={L}")
            layers[i].append(t)
        elif name in GROUPS:
            named[GROUPS[name]] = t
        else:
            raise MetadataError(f"unclassified checkpoint tensor {name!r}")
    if missing := [g for g in ("embed", "norm") if g not in named] + [i for i in range(L) if not layers[i]]:
        raise MetadataError(f"checkpoint lacks {missing}")
    if named["embed"].shape != (V, h):
        raise MetadataError(f"embed_tokens {named['embed'].shape} != (vocab_size, hidden_size) = {(V, h)}")
    _check_attention(cfg, layers)

    lm_head, duplicates = named.get("lm_head"), []
    if cfg["tie_word_embeddings"]:
        if lm_head is not None:
            if lm_head.shape != named["embed"].shape:
                raise MetadataError(f"tied lm_head {lm_head.shape} != embed_tokens {named['embed'].shape}")
            duplicates, lm_head = [lm_head], None
    elif lm_head is None:
        raise MetadataError("untied model without lm_head.weight in the checkpoint")
    return Architecture(model_id, revision, source, cfg, named["embed"], named["norm"], lm_head,
                        [sorted(ts, key=lambda t: t.name) for ts in layers], duplicates, shard_bytes)


def _check_attention(cfg: dict, layers: list[list[Tensor]]):
    """q/k/v projections must match the head counts the KV-cache and activation formulas use."""
    h, d = cfg["hidden_size"], cfg["head_dim"]
    expect = {"q_proj": cfg["num_attention_heads"] * d, "k_proj": cfg["num_key_value_heads"] * d,
              "v_proj": cfg["num_key_value_heads"] * d}
    for i, ts in enumerate(layers):
        for t in ts:
            proj = t.name.rsplit(".", 2)[-2]
            if t.name.endswith(".weight") and proj in expect and t.shape != (expect[proj], h):
                raise MetadataError(f"layer {i} {proj} {t.shape} != {(expect[proj], h)} from the config")


def resolve_lora_targets(arch: Architecture, config, targets) -> list[Tensor]:
    """Base weights adapted by `targets` (a preset name or {modules, num_layers}), resolved with the training code's own
    model.target_modules on a meta-device skeleton (no weights)."""
    modules, num_layers = PRESETS[targets] if isinstance(targets, str) else (targets["modules"], targets["num_layers"])
    with torch.device("meta"):
        skeleton = AutoModelForCausalLM.from_config(config)
    names = {n + ".weight" for n in target_modules(skeleton, tuple(modules), num_layers)}
    by_name = {t.name: t for ts in arch.layers for t in ts}
    if missing := sorted(names - by_name.keys()):
        raise MetadataError(f"LoRA targets absent from the checkpoint: {missing}")
    return [by_name[n] for n in sorted(names)]


def read_header(path: Path) -> dict[str, Tensor]:
    """Tensor names, shapes and dtypes from a .safetensors file's JSON header (first 8 bytes: header length)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return {k: Tensor(k, tuple(v["shape"]), v["dtype"]) for k, v in header.items() if k != "__metadata__"}


def checkpoint_tensors(model_id: str, revision: str | None) -> tuple[dict[str, Tensor], dict[str, int], str]:
    """All checkpoint tensors and per-shard bytes: from cached safetensors files if every shard is cached, otherwise
    from the Hub's safetensors headers (a few KB per shard; no weights are downloaded)."""
    from huggingface_hub import get_safetensors_metadata, try_to_load_from_cache

    def cached(name):
        p = try_to_load_from_cache(model_id, name, revision=revision)
        return Path(p) if isinstance(p, str) else None

    if index := cached("model.safetensors.index.json"):
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    paths = [cached(f) for f in files]
    if all(paths):
        headers = {f: read_header(p) for f, p in zip(files, paths)}
        source = "local safetensors headers"
    else:
        try:
            meta = get_safetensors_metadata(model_id, revision=revision)
        except Exception as e:
            raise MetadataError(f"{model_id}: no cached checkpoint and remote safetensors metadata failed: {e}") from e
        headers = {f: {k: Tensor(k, tuple(v.shape), v.dtype) for k, v in m.tensors.items()}
                   for f, m in meta.files_metadata.items()}
        source = "remote safetensors headers"
    tensors = {k: t for h in headers.values() for k, t in h.items()}
    return tensors, {f: sum(t.stored_bytes for t in h.values()) for f, h in headers.items()}, source


def load(model_id: str, lora_targets) -> tuple[Architecture, object]:
    """Architecture of `model_id` with its LoRA targets resolved, and its transformers config."""
    from huggingface_hub import hf_hub_download

    try:
        config_path = Path(hf_hub_download(model_id, "config.json"))
    except Exception as e:
        raise MetadataError(f"{model_id}: config.json unavailable: {e}") from e
    revision = config_path.parent.name  # .../snapshots/<commit sha>/config.json
    config = AutoConfig.from_pretrained(config_path.parent)
    tensors, shards, source = checkpoint_tensors(model_id, revision)
    arch = from_metadata(model_id, config, tensors, shards, source, revision)
    arch.lora_targets = resolve_lora_targets(arch, config, lora_targets)
    return arch, config
