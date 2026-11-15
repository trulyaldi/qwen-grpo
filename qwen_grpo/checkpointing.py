"""Checkpoints: PEFT adapters (what evaluation loads) and the training state that makes a run resumable.

checkpoints/<run>_step<N>/ holds adapter_config.json + adapter_model.safetensors and, for runs since resumable
checkpoints were added, training_state.pt: optimizer state, progress counters, the training log so far and the RNG
states at the moment of saving. Older checkpoints hold the adapter only; they evaluate as before but cannot resume.
"""
import json
import random
import re
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict

from qwen_grpo.model import matched_modules, skeleton, target_modules

ROOT = Path(__file__).resolve().parents[1]
STATE = "training_state.pt"


def rel(path):
    """Repository-relative path string if `path` is inside the repository, else the absolute path."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def run_of(checkpoint):
    """checkpoints/<run>_step<N> -> (<run>, N)."""
    m = re.fullmatch(r"(.+)_step(\d+)", Path(checkpoint).name)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def check_adapter_matches(cfg, checkpoint):
    """The adapter's saved config must be the one the YAML describes; otherwise loading would silently mismatch.
    Target modules are compared as resolved module names on the model's weightless skeleton, so a historical regex and
    an equivalent list of names agree."""
    saved = json.loads((Path(checkpoint) / "adapter_config.json").read_text())
    expected = {"r": cfg.lora.rank, "lora_alpha": cfg.lora.alpha, "lora_dropout": cfg.lora.dropout,
                "base_model_name_or_path": cfg.model.name}
    got = {k: saved.get(k) for k in expected}
    if got == expected:
        s = skeleton(cfg.model.name)
        got["modules"] = sorted(matched_modules(s, saved["target_modules"]))
        expected["modules"] = sorted(target_modules(s, *cfg.lora.spec()))
    if got != expected:
        raise SystemExit(f"{checkpoint} does not match the config:\n  checkpoint {got}\n  config     {expected}")


def adapter_state(model):
    return {k: v.detach().float().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def load_adapter(model, checkpoint):
    """Load adapter weights into an existing PEFT model; refuses missing/extra tensors and checks they took effect."""
    loaded = load_peft_weights(str(checkpoint))
    live = get_peft_model_state_dict(model)
    if live.keys() != loaded.keys():
        raise SystemExit(f"adapter tensors differ: missing {sorted(live.keys() - loaded.keys())}, extra {sorted(loaded.keys() - live.keys())}")
    set_peft_model_state_dict(model, loaded)
    assert all(torch.equal(p, loaded[k].to(p.device)) for k, p in get_peft_model_state_dict(model).items()), "adapter not loaded"
    return {k: v.float().cpu() for k, v in loaded.items()}


def rng_state():
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    random.setstate(s["python"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(model, optimizer, path, state):
    """Adapter + training_state.pt = {**state, optimizer, rng}. `state` must hold only tensors and plain Python data."""
    path = Path(path)
    model.save_pretrained(path)
    torch.save({**state, "optimizer": optimizer.state_dict(), "rng": rng_state()}, path / STATE)


def load_training_state(checkpoint):
    path = Path(checkpoint) / STATE
    if not path.exists():
        raise SystemExit(f"{checkpoint} has no {STATE}: it holds adapter weights only and cannot resume training")
    return torch.load(path, weights_only=True)
