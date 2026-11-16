"""YAML run configuration, the single source of truth for scripts/run.py.

Every key without a default is required; unknown, missing or ill-typed keys are errors. Defaults are the values every
run so far used, so a YAML may omit those keys and older configs and saved training logs load unchanged (normalized()
fills them in). Some keys accept one value only: they record a choice the code does not yet support changing, and
validate() says why instead of silently ignoring a different value.

A YAML file may start with `extends: <path of another YAML, relative to this file>`: the parent is loaded first (itself
possibly extending another) and this file's keys are merged onto it by merge(). The resolved config has no `extends`.
"""
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

from qwen_grpo.evaluation import BATCH, BOOTSTRAP, MAX_NEW, SCORE_BATCH
from qwen_grpo.model import PRESETS

RUN_NAME = r"[A-Za-z0-9][A-Za-z0-9_.-]*"  # no path separators, cannot start with "." (hidden, "..") or "-"
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
BATCH_MODES = ("cycle", "balanced", "unconstrained")


@dataclass(frozen=True)
class Run:
    name: str  # outputs: results/<name>/, results/benchmarks/<name>.*, checkpoints/<name>_step<N>/
    gpu_budget_gib: float = 4.0  # peak reserved memory the run is checked against (reported, not enforced)
    overwrite: bool = False  # true: train deletes this run's earlier results/checkpoints first (never on --resume)


@dataclass(frozen=True, kw_only=True)
class Model:
    name: str  # a Qwen3 causal LM (the prompt asserts Qwen3's non-thinking chat template)
    dtype: str = "bfloat16"  # frozen base weights; LoRA adapters are always fp32
    thinking: bool = False


@dataclass(frozen=True)
class LoraTargets:
    modules: list  # Linear module names inside each decoder layer, e.g. [q_proj, v_proj, o_proj]
    num_layers: int  # adapt the last num_layers decoder layers


@dataclass(frozen=True, kw_only=True)
class Lora:
    rank: int
    alpha: int
    dropout: float = 0.0
    targets: str | LoraTargets  # a key of model.PRESETS, or {modules: [...], num_layers: N}

    def spec(self):
        """(modules, num_layers), resolved against the loaded model by model.target_modules()."""
        if isinstance(self.targets, str):
            return PRESETS[self.targets]
        return tuple(self.targets.modules), self.targets.num_layers


@dataclass(frozen=True, kw_only=True)
class Data:
    train_samples: int  # balanced subset of the training split
    val_subset: int  # balanced subset of validation, evaluated every eval_every steps
    subset_seed: int = 42  # selects both subsets; kept apart from training.seed so seed studies see the same data


@dataclass(frozen=True, kw_only=True)
class Training:
    batch_size: int  # B prompts per optimizer step
    batch_mode: str
    group_size: int  # G completions per prompt
    learning_rate: float
    max_steps: int  # optimizer updates
    max_attempts: int  # rollout batches, including zero-signal batches that are skipped
    clip_eps: float = 0.2
    num_iterations: int = 1  # optimizer updates per rollout batch, all against its sampling-time log-probs
    temperature: float = 1.0
    max_new_tokens: int  # rollout length; evaluation length is evaluation.max_new_tokens
    seed: int  # LoRA init, rollout sampling, batch order
    eval_every: int
    full_val: bool  # full-validation evaluation, label scores and benchmark files after training


@dataclass(frozen=True)
class NegativeControl:
    checkpoint: str | None = None  # an LR=0 checkpoint of this config (provenance checked), or null
    train: bool = False  # true: run the LR=0 control with train() before verifying (run.name + "_lr0"); never implicit


@dataclass(frozen=True)
class Statistics:
    confidence_level: float = 0.95
    bootstrap_samples: int = 2000
    seed: int = 42


@dataclass(frozen=True)
class Verification:
    split: str = "validation"  # held-out split: validation (model selection) or test (final report only)
    merge_subset: int = 8  # examples for the full-logit adapter-vs-merged comparison
    negative_control: NegativeControl = field(default_factory=NegativeControl)
    statistics: Statistics = field(default_factory=Statistics)


@dataclass(frozen=True)
class Evaluation:
    """Greedy evaluation and the bootstrap of train's full-validation and validate/test (verify uses its statistics).
    Defaults are the settings of the saved baseline; bf16 outputs depend on batch shape, so batch sizes matter too."""
    max_new_tokens: int = MAX_NEW
    batch_size: int = BATCH
    score_batch_size: int = SCORE_BATCH
    bootstrap_samples: int = BOOTSTRAP
    bootstrap_seed: int = 42


@dataclass(frozen=True, kw_only=True)
class Config:
    run: Run
    model: Model
    lora: Lora
    data: Data
    training: Training
    verification: Verification = field(default_factory=Verification)
    evaluation: Evaluation = field(default_factory=Evaluation)


def _build(cls, raw, where):
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    names = {f.name for f in dataclasses.fields(cls)}
    if unknown := sorted(set(raw) - names):
        raise ValueError(f"{where}: unknown keys {unknown}")
    required = {f.name for f in dataclasses.fields(cls) if f.default is f.default_factory is dataclasses.MISSING}
    if missing := sorted(required - set(raw)):
        raise ValueError(f"{where}: missing keys {missing}")
    values = {}
    for f in dataclasses.fields(cls):
        if f.name not in raw:
            continue
        v, key = raw[f.name], f"{where}.{f.name}"
        if dataclasses.is_dataclass(f.type):
            values[f.name] = _build(f.type, v, key)
        elif f.type == str | LoraTargets and isinstance(v, dict):
            values[f.name] = _build(LoraTargets, v, key)
        elif f.type == str | LoraTargets and type(v) is str:
            values[f.name] = v
        elif f.type == str | None and (v is None or type(v) is str):
            values[f.name] = v
        elif f.type is float and isinstance(v, (int, float)) and not isinstance(v, bool):
            values[f.name] = float(v)
        elif type(v) is f.type:  # exact: rejects True for an int, 1 for a bool, "1e-4" for a float
            values[f.name] = v
        else:
            raise ValueError(f"{key}: expected {getattr(f.type, '__name__', f.type)}, got {type(v).__name__} {v!r}")
    return cls(**values)


def validate(cfg: Config) -> Config:
    m, lo, d, t, v, e = cfg.model, cfg.lora, cfg.data, cfg.training, cfg.verification, cfg.evaluation
    nc, st = v.negative_control, v.statistics
    custom = lo.targets if isinstance(lo.targets, LoraTargets) else None
    checks = [
        (re.fullmatch(RUN_NAME, cfg.run.name) is not None,
         "run.name: letters, digits, '_', '.', '-', starting with a letter or digit (\"..\" would name the parent directory)"),
        (cfg.run.gpu_budget_gib > 0, "run.gpu_budget_gib: > 0"),
        (re.fullmatch(r"Qwen/Qwen3-[0-9.]+B", m.name) is not None,
         "model.name: a dense Qwen3 model 'Qwen/Qwen3-<size>B' (prompt, EOS ids and LoRA presets assume Qwen3)"),
        (m.dtype in DTYPES, f"model.dtype: one of {list(DTYPES)}"),
        (m.thinking is False, "model.thinking: only false (non-thinking mode; sentiment.build_prompts asserts it)"),
        (custom is not None or lo.targets in PRESETS, f"lora.targets: one of {list(PRESETS)} or {{modules, num_layers}}"),
        (custom is None or (len(custom.modules) >= 1 and all(type(x) is str and x for x in custom.modules)
                            and len(set(custom.modules)) == len(custom.modules)),
         "lora.targets.modules: a non-empty list of distinct module names"),
        (custom is None or custom.num_layers >= 1, "lora.targets.num_layers: >= 1"),
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
        (t.num_iterations >= 1, "training.num_iterations: >= 1"),
        (t.temperature == 1.0, "training.temperature: only 1.0 (log-probs are computed from untempered logits, so "
                               "any other value would make the sampler differ from the policy being optimized)"),
        (t.max_new_tokens >= 1 and t.eval_every >= 1, "training.max_new_tokens and training.eval_every: >= 1"),
        (v.split in ("validation", "test"), "verification.split: validation or test"),
        (v.merge_subset >= 1, "verification.merge_subset: >= 1"),
        (not (nc.train and nc.checkpoint), "verification.negative_control: set checkpoint or train, not both"),
        (0 < st.confidence_level < 1, "verification.statistics.confidence_level: in (0, 1)"),
        (min(e.max_new_tokens, e.batch_size, e.score_batch_size) >= 1, "evaluation: token and batch sizes >= 1"),
        (e.bootstrap_samples >= 1000 and e.bootstrap_samples % 1000 == 0,
         "evaluation.bootstrap_samples: a multiple of 1000, >= 1000 (resampled in chunks of 1000)"),
        (st.bootstrap_samples >= 100 and st.bootstrap_samples % 100 == 0,
         "verification.statistics.bootstrap_samples: a multiple of 100, >= 100 (resampled in chunks of 100)"),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))
    return cfg


def merge(base: dict, override: dict) -> dict:
    """override's keys onto base: mappings merge key by key, any other value (list, scalar, null) replaces."""
    out = dict(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read(path, chain=()) -> dict:
    """The raw YAML mapping of `path` with its `extends` chain merged in (parents first)."""
    path = Path(path).resolve()
    if path in chain:
        raise ValueError("circular extends: " + " -> ".join(p.name for p in (*chain, path)))
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: expected a mapping, got {type(raw).__name__}")
    if "extends" not in raw:
        return raw
    parent = raw.pop("extends")
    if type(parent) is not str or not (path.parent / parent).is_file():
        raise ValueError(f"{path.name}: extends {parent!r} is not a YAML file (paths are relative to {path.parent})")
    base = read(path.parent / parent, (*chain, path))
    name = raw.get("run", {}).get("name") if isinstance(raw.get("run"), dict) else None
    if name is None or name == base.get("run", {}).get("name"):
        raise ValueError(f"{path.name}: a config that extends another must set its own run.name (outputs are named by it)")
    return merge(base, raw)


def load(path) -> Config:
    return validate(_build(Config, read(path), Path(path).name))


def as_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


def sha256(cfg: Config) -> str:
    """Hash of the fully resolved config (defaults included), independent of key order and of how the YAML was split
    across files: a config and an equivalent one written differently hash equally."""
    return hashlib.sha256(json.dumps(as_dict(cfg), sort_keys=True).encode()).hexdigest()


def differences(a: dict, b: dict, prefix="") -> list:
    """Dotted keys whose values differ between two config dicts."""
    keys = sorted(set(a) | set(b))
    return [d for k in keys for d in (differences(a[k], b[k], f"{prefix}{k}.")
                                       if isinstance(a.get(k), dict) and isinstance(b.get(k), dict)
                                       else [f"{prefix}{k}"] if a.get(k, ...) != b.get(k, ...) else [])]


def normalized(raw: dict) -> dict:
    """A saved config dict (e.g. a training log's yaml_config) in the current schema, defaults filled in."""
    return as_dict(_build(Config, raw, "saved config"))
