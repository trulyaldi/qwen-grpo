"""Checkpoints: PEFT adapters (what evaluation loads) and the training state that makes a run resumable.

checkpoints/<run>_step<N>/ holds adapter_config.json + adapter_model.safetensors and, for runs since resumable
checkpoints were added, training_state.pt: optimizer state, progress counters, the training log so far and the RNG
states at the moment of saving. Older checkpoints hold the adapter only; they evaluate as before but cannot resume.
"""
import json
import os
import random
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict

from qwen_grpo.model import matched_modules, skeleton, target_modules
from qwen_grpo.runs import lock_path, rel  # noqa: F401 (re-exported)

STATE = "training_state.pt"


def check_adapter_matches(cfg, checkpoint):
    """The adapter's saved config must be the one the YAML describes; otherwise loading would silently mismatch.
    Target modules are compared as resolved module names on the model's weightless skeleton, so a historical regex and
    an equivalent list of names agree."""
    saved = json.loads((Path(checkpoint) / "adapter_config.json").read_text())
    expected = {"r": cfg.lora.rank, "lora_alpha": cfg.lora.alpha, "lora_dropout": 0.0,
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


def lock_owner(path):
    """PID of the live process holding the lock at `path`, or None (no lock, unreadable or stale lock)."""
    try:
        pid = int(Path(path).read_text())
        os.kill(pid, 0)
    except PermissionError:
        return pid  # alive, owned by another user
    except (OSError, ValueError):
        return None
    return pid if pid > 0 else None


@contextmanager
def training_lock(path):
    """One training process per run: refuses while a live process holds the lock; a stale lock is taken over."""
    path = Path(path)
    if (pid := lock_owner(path)) is not None:
        raise SystemExit(f"run is already being trained by process {pid} (lock {path})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


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
