"""YAML run configuration, the single source of truth for scripts/run.py.

Every key is required; unknown, missing or ill-typed keys are errors. Some keys accept one value only: they record a
choice the code does not yet support changing, and validate() says why instead of silently ignoring a different value.
"""
import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from qwen_grpo.model import MODEL, TARGETS

DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
BATCH_MODES = ("cycle", "balanced", "unconstrained")


@dataclass(frozen=True)
class Run:
    name: str  # outputs: results/<name>/, results/benchmarks/<name>.*, checkpoints/<name>_step<N>/


@dataclass(frozen=True)
class Model:
    name: str
    dtype: str  # frozen base weights; LoRA adapters are always fp32
    thinking: bool


@dataclass(frozen=True)
class Lora:
    rank: int
    alpha: int
    dropout: float
    targets: str  # a key of model.TARGETS


@dataclass(frozen=True)
class Data:
    train_samples: int  # balanced subset of the training split
    val_subset: int  # balanced subset of validation, evaluated every eval_every steps
    subset_seed: int  # selects both subsets; kept apart from training.seed so seed studies see the same data


@dataclass(frozen=True)
class Training:
    batch_size: int  # B prompts per optimizer step
    batch_mode: str
    group_size: int  # G completions per prompt
    learning_rate: float
    max_steps: int  # optimizer updates
    max_attempts: int  # rollout batches, including zero-signal batches that are skipped
    clip_eps: float
    temperature: float
    max_new_tokens: int  # rollout length; evaluation length is fixed (evaluation.MAX_NEW) to match the baseline
    seed: int  # LoRA init, rollout sampling, batch order
    eval_every: int
    full_val: bool  # full-validation evaluation, label scores and benchmark files after training


@dataclass(frozen=True)
class Config:
    run: Run
    model: Model
    lora: Lora
    data: Data
    training: Training


def _build(cls, raw, where):
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    if unknown := sorted(set(raw) - names):
        raise ValueError(f"{where}: unknown keys {unknown}")
    if missing := sorted(names - set(raw)):
        raise ValueError(f"{where}: missing keys {missing}")
    values = {}
    for f in dataclasses.fields(cls):
        v, key = raw[f.name], f"{where}.{f.name}"
        if dataclasses.is_dataclass(f.type):
            values[f.name] = _build(f.type, v, key)
        elif f.type is float and isinstance(v, (int, float)) and not isinstance(v, bool):
            values[f.name] = float(v)
        elif type(v) is f.type:  # exact: rejects True for an int, 1 for a bool, "1e-4" for a float
            values[f.name] = v
        else:
            raise ValueError(f"{key}: expected {f.type.__name__}, got {type(v).__name__} {v!r}")
    return cls(**values)


def validate(cfg: Config) -> Config:
    m, lo, d, t = cfg.model, cfg.lora, cfg.data, cfg.training
    checks = [
        (re.fullmatch(r"[A-Za-z0-9_.-]+", cfg.run.name) is not None, "run.name: letters, digits, '_', '.', '-' only"),
        (m.name == MODEL, f"model.name: only {MODEL} is supported (the LoRA target regexes name its layers 26-27)"),
        (m.dtype in DTYPES, f"model.dtype: one of {list(DTYPES)}"),
        (m.thinking is False, "model.thinking: only false (non-thinking mode; sentiment.build_prompts asserts it)"),
        (lo.targets in TARGETS, f"lora.targets: one of {list(TARGETS)}"),
        (lo.rank >= 1 and lo.alpha >= 1, "lora.rank and lora.alpha must be >= 1"),
        (lo.dropout == 0.0, "lora.dropout: only 0.0 (the model is never put in train mode, so dropout would be "
                            "silently inactive; enabling it would also break the old/current log-prob equality)"),
        (d.train_samples >= 2 and d.train_samples % 2 == 0, "data.train_samples: even and >= 2 (balanced by label)"),
        (d.val_subset >= 2 and d.val_subset % 2 == 0, "data.val_subset: even and >= 2 (balanced by label)"),
        (t.batch_mode in BATCH_MODES, f"training.batch_mode: one of {list(BATCH_MODES)}"),
        ((t.batch_mode == "cycle") == (t.batch_size == 1), "training: batch_mode cycle <=> batch_size 1"),
        (t.batch_mode != "balanced" or t.batch_size == 2, "training: batch_mode balanced needs batch_size 2"),
        (t.batch_size <= d.train_samples, "training.batch_size must not exceed data.train_samples"),
        (t.group_size >= 2, "training.group_size: >= 2 (a group std needs two samples)"),
        (t.learning_rate >= 0, "training.learning_rate: >= 0 (0 is the negative control)"),
        (1 <= t.max_steps <= t.max_attempts, "training: 1 <= max_steps <= max_attempts"),
        (0 < t.clip_eps < 1, "training.clip_eps: in (0, 1)"),
        (t.temperature == 1.0, "training.temperature: only 1.0 (log-probs are computed from untempered logits, so "
                               "any other value would make the sampler differ from the policy being optimized)"),
        (t.max_new_tokens >= 1 and t.eval_every >= 1, "training.max_new_tokens and training.eval_every: >= 1"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))
    return cfg


def load(path) -> Config:
    return validate(_build(Config, yaml.safe_load(Path(path).read_text()), Path(path).name))


def as_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
