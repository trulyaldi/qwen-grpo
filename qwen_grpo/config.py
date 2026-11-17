"""YAML run configuration, the single source of truth for scripts/run.py.

Sections: run, model, lora, data, training, evaluation, and optionally verification, execution, streaming and dflash. Every key without a default is
required; unknown, missing or ill-typed keys are errors. A YAML may start with `extends: <path of another YAML, relative
to this file>`: the parent is loaded first and this file's keys are merged onto it by merge().

Historical configs (written before this schema) name keys that were removed or moved (HISTORICAL). New experiments
must not use them: load() rejects them unless historical=True, which upgrade()s the mapping explicitly. A removed key
is accepted only with the one value the current code implements; training.batch_mode cycle/balanced is kept as
training.legacy_batch_mode, so such a config stays readable (validate, test, verify, report) but cannot be trained or
resumed.
"""
import copy
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
LEGACY_BATCH_MODES = {"cycle": 1, "balanced": 2}  # historical samplers the current code cannot reproduce -> their B

# Historical keys. REMOVED: the only value the current code implements (that value is dropped, any other is an error).
# MOVED: the key's current name. training.batch_mode "unconstrained" is the shuffled sampler (training.shuffled_batches).
REMOVED = {"model.thinking": False, "lora.dropout": 0.0, "training.temperature": 1.0, "evaluation.bootstrap_seed": 42,
           "verification.merge_subset": 8, "verification.statistics.confidence_level": 0.95,
           "verification.statistics.bootstrap_samples": 2000, "verification.statistics.seed": 42}
MOVED = {"training.eval_every": "evaluation.every", "training.full_val": "evaluation.full_validation",
         "verification.negative_control.checkpoint": "verification.control_checkpoint",
         "verification.negative_control.train": "verification.train_control"}
HISTORICAL = (*REMOVED, *MOVED, "training.batch_mode", "training.legacy_batch_mode")


@dataclass(frozen=True)
class Run:
    name: str  # outputs: runs/<name>/
    gpu_budget_gib: float = 4.0  # peak reserved memory the run is checked against (reported, not enforced)
    overwrite: bool = False  # true: train deletes this run's earlier outputs first (never on --resume)


@dataclass(frozen=True, kw_only=True)
class Model:
    name: str  # a dense Qwen3 causal LM, always in non-thinking mode (sentiment.build_prompts asserts it)
    dtype: str = "bfloat16"  # frozen base weights; LoRA adapters are always fp32


@dataclass(frozen=True)
class LoraTargets:
    modules: list  # Linear module names inside each decoder layer, e.g. [q_proj, v_proj, o_proj]
    num_layers: int  # adapt the last num_layers decoder layers


@dataclass(frozen=True, kw_only=True)
class Lora:
    rank: int
    alpha: int  # scale alpha / rank; no dropout (the model is never in train mode)
    targets: str | LoraTargets  # a key of model.PRESETS, or {modules: [...], num_layers: N}

    def spec(self):
        """(modules, num_layers), resolved against the loaded model by model.target_modules()."""
        if isinstance(self.targets, str):
            return PRESETS[self.targets]
        return tuple(self.targets.modules), self.targets.num_layers


@dataclass(frozen=True, kw_only=True)
class Data:
    train_samples: int  # balanced subset of the training split
    val_subset: int  # balanced subset of validation, evaluated every evaluation.every steps
    subset_seed: int = 42  # selects both subsets; kept apart from training.seed so seed studies see the same data


@dataclass(frozen=True, kw_only=True)
class Training:
    batch_size: int  # B prompts per rollout batch, drawn by training.shuffled_batches
    group_size: int  # G completions per prompt, sampled at temperature 1
    learning_rate: float
    max_steps: int  # optimizer updates
    max_attempts: int  # rollout batches, including zero-signal batches that are skipped
    max_new_tokens: int  # rollout length; evaluation length is evaluation.max_new_tokens
    seed: int  # LoRA init, rollout sampling, batch order
    clip_eps: float = 0.2
    num_iterations: int = 1  # optimizer updates per rollout batch, all against its sampling-time log-probs
    micro_batch_size: int | None = None  # completions per forward/backward chunk; None: all B*G at once
    legacy_batch_mode: str | None = None  # set by upgrade() only: a historical sampler that cannot be trained


@dataclass(frozen=True, kw_only=True)
class Evaluation:
    """Greedy evaluation and the bootstrap of train's full validation and of validate/test. Defaults are the settings of
    the saved baseline; bf16 outputs depend on batch shape, so batch sizes matter too."""
    every: int  # validation-subset evaluation and checkpoint every `every` optimizer steps
    full_validation: bool  # full-validation evaluation and label scores of the final checkpoint after training
    max_new_tokens: int = MAX_NEW
    batch_size: int = BATCH
    score_batch_size: int = SCORE_BATCH
    bootstrap_samples: int = BOOTSTRAP


@dataclass(frozen=True)
class Verification:
    split: str = "validation"  # held-out split: validation (model selection) or test (final report only)
    control_checkpoint: str | None = None  # an LR=0 checkpoint of this config (provenance checked), or null
    train_control: bool = False  # true: train the LR=0 control (run <name>_lr0) before verifying; never implicit


BACKENDS = ("resident", "l2l_sync", "l2l_async")
WEIGHT_STORAGE = ("pinned", "pageable")
KV_CACHE = ("gpu", "cpu")
L2L_ONLY = ("weight_storage", "resident_layers", "weight_buffers", "prefetch_depth", "activation_checkpointing")
DRAFTERS = {"z-lab/Qwen3-4B-DFlash-b16": "Qwen/Qwen3-4B"}  # DFlash drafter -> the target model it was trained for


@dataclass(frozen=True, kw_only=True)
class Execution:
    """Where frozen base weights live and how layers run (docs/l2l/). The L2L_ONLY keys are null under resident, which
    keeps the whole model on the GPU, and required under l2l_*. resident_layers is checked against the model depth when
    the model is loaded."""
    backend: str = "resident"  # resident | l2l_sync | l2l_async
    weight_storage: str | None = None  # frozen base weights in CPU RAM: pinned (assignment) | pageable (explicit opt-out)
    resident_layers: int | None = None  # decoder layers kept on the GPU for the whole run
    weight_buffers: int | None = None  # GPU buffers for streamed layers
    prefetch_depth: int | None = None  # streamed layers in flight ahead of the one computing; l2l_sync: 0
    activation_checkpointing: bool | None = None  # keep layer-boundary inputs only, recompute layers in backward
    vocab_chunk_size: int | None = None  # output head over vocabulary chunks; null: the whole vocabulary at once
    kv_cache: str = "gpu"  # rollout KV cache placement


@dataclass(frozen=True, kw_only=True)
class Streaming:
    """Disk-backed prompts and rollout records (Ticket 09). Provisional: fields will follow the implementation."""
    enabled: bool = False
    directory: str | None = None  # prompt shards, rollout records and resume state


@dataclass(frozen=True, kw_only=True)
class DFlash:
    """DFlash speculative decoding (Ticket 12). Provisional: only the drafter's identity is checked here; drafting
    parameters wait until the paper and code have been read."""
    enabled: bool = False
    draft_model: str | None = None  # a key of DRAFTERS whose target is model.name


@dataclass(frozen=True, kw_only=True)
class Config:
    run: Run
    model: Model
    lora: Lora
    data: Data
    training: Training
    evaluation: Evaluation
    verification: Verification = field(default_factory=Verification)
    execution: Execution = field(default_factory=Execution)
    streaming: Streaming = field(default_factory=Streaming)
    dflash: DFlash = field(default_factory=DFlash)


OPTIONAL_SECTIONS = {"execution": Execution, "streaming": Streaming, "dflash": DFlash}  # added after runs were saved


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
        elif f.type in (str | None, int | None, bool | None) and (v is None or type(v) is f.type.__args__[0]):
            values[f.name] = v
        elif f.type is float and isinstance(v, (int, float)) and not isinstance(v, bool):
            values[f.name] = float(v)
        elif type(v) is f.type:  # exact: rejects True for an int, 1 for a bool, "1e-4" for a float
            values[f.name] = v
        else:
            raise ValueError(f"{key}: expected {getattr(f.type, '__name__', f.type)}, got {type(v).__name__} {v!r}")
    return cls(**values)


def validate(cfg: Config) -> Config:
    m, lo, d, t, e, v = cfg.model, cfg.lora, cfg.data, cfg.training, cfg.evaluation, cfg.verification
    custom = lo.targets if isinstance(lo.targets, LoraTargets) else None
    rollouts = t.batch_size * t.group_size
    checks = [
        (re.fullmatch(RUN_NAME, cfg.run.name) is not None,
         "run.name: letters, digits, '_', '.', '-', starting with a letter or digit (\"..\" would name the parent directory)"),
        (cfg.run.gpu_budget_gib > 0, "run.gpu_budget_gib: > 0"),
        (re.fullmatch(r"Qwen/Qwen3-[0-9.]+B", m.name) is not None,
         "model.name: a dense Qwen3 model 'Qwen/Qwen3-<size>B' (prompt, EOS ids and LoRA presets assume Qwen3)"),
        (m.dtype in DTYPES, f"model.dtype: one of {list(DTYPES)}"),
        (custom is not None or lo.targets in PRESETS, f"lora.targets: one of {list(PRESETS)} or {{modules, num_layers}}"),
        (custom is None or (len(custom.modules) >= 1 and all(type(x) is str and x for x in custom.modules)
                            and len(set(custom.modules)) == len(custom.modules)),
         "lora.targets.modules: a non-empty list of distinct module names"),
        (custom is None or custom.num_layers >= 1, "lora.targets.num_layers: >= 1"),
        (lo.rank >= 1 and lo.alpha >= 1, "lora.rank and lora.alpha must be >= 1"),
        (d.train_samples >= 2 and d.train_samples % 2 == 0, "data.train_samples: even and >= 2 (balanced by label)"),
        (d.val_subset >= 2 and d.val_subset % 2 == 0, "data.val_subset: even and >= 2 (balanced by label)"),
        (1 <= t.batch_size <= d.train_samples, "training.batch_size: between 1 and data.train_samples"),
        (t.group_size >= 2, "training.group_size: >= 2 (a group std needs two samples)"),
        (t.learning_rate >= 0, "training.learning_rate: >= 0 (0 is the negative control)"),
        (1 <= t.max_steps <= t.max_attempts * t.num_iterations,
         "training: 1 <= max_steps <= max_attempts * num_iterations (each rollout batch gives at most num_iterations updates)"),
        (t.max_new_tokens >= 1, "training.max_new_tokens: >= 1"),
        (0 < t.clip_eps < 1, "training.clip_eps: in (0, 1)"),
        (t.num_iterations >= 1, "training.num_iterations: >= 1"),
        (t.micro_batch_size is None or 1 <= t.micro_batch_size <= rollouts,
         f"training.micro_batch_size: null or between 1 and batch_size * group_size = {rollouts}"),
        (t.legacy_batch_mode is None or LEGACY_BATCH_MODES.get(t.legacy_batch_mode) == t.batch_size,
         f"training.legacy_batch_mode: one of {LEGACY_BATCH_MODES} (mode: batch_size)"),
        (e.every >= 1, "evaluation.every: >= 1"),
        (min(e.max_new_tokens, e.batch_size, e.score_batch_size) >= 1, "evaluation: token and batch sizes >= 1"),
        (e.bootstrap_samples >= 1000 and e.bootstrap_samples % 1000 == 0,
         "evaluation.bootstrap_samples: a multiple of 1000, >= 1000 (resampled in chunks of 1000)"),
        (v.split in ("validation", "test"), "verification.split: validation or test"),
        (not (v.train_control and v.control_checkpoint), "verification: set control_checkpoint or train_control, not both"),
        *execution_checks(cfg),
    ]
    errors = [msg for ok, msg in checks if not ok]
    if errors:
        raise ValueError("invalid config:\n  " + "\n  ".join(errors))
    return cfg


def execution_checks(cfg: Config) -> list:
    """(ok, message) pairs for the execution, streaming and dflash sections."""
    x, s, f = cfg.execution, cfg.streaming, cfg.dflash
    l2l = x.backend in ("l2l_sync", "l2l_async")
    unset = [k for k in L2L_ONLY if getattr(x, k) is None]
    checks = [
        (x.backend in BACKENDS, f"execution.backend: one of {list(BACKENDS)}"),
        (x.kv_cache in KV_CACHE, f"execution.kv_cache: one of {list(KV_CACHE)}"),
        (x.vocab_chunk_size is None or x.vocab_chunk_size >= 1, "execution.vocab_chunk_size: null or >= 1"),
        (s.enabled == (s.directory is not None), "streaming: enabled requires directory, and directory requires enabled"),
        (s.directory is None or s.directory != "", "streaming.directory: a non-empty path"),
        (f.enabled == (f.draft_model is not None), "dflash: enabled requires draft_model, and draft_model requires enabled"),
        (f.draft_model is None or f.draft_model in DRAFTERS, f"dflash.draft_model: one of {list(DRAFTERS)}"),
        (f.draft_model not in DRAFTERS or DRAFTERS[f.draft_model] == cfg.model.name,
         f"dflash.draft_model {f.draft_model!r} drafts for {DRAFTERS.get(f.draft_model)}, not model.name {cfg.model.name!r}"),
    ]
    if x.backend == "resident":
        checks += [(not (set(L2L_ONLY) - set(unset)),
                    f"execution: {sorted(set(L2L_ONLY) - set(unset))} only apply to l2l backends; resident keeps the "
                    "whole model on the GPU, so leave them null"),
                   (x.kv_cache == "gpu", "execution.kv_cache: resident generation (HF generate) keeps the cache on the GPU")]
    elif l2l:
        checks += [(not unset, f"execution: backend {x.backend} needs {unset}")]
        if not unset:
            checks += [
                (x.weight_storage in WEIGHT_STORAGE, f"execution.weight_storage: one of {list(WEIGHT_STORAGE)}"),
                (x.resident_layers >= 0, "execution.resident_layers: >= 0"),
                (x.weight_buffers >= 1, "execution.weight_buffers: >= 1"),
                (x.backend != "l2l_sync" or x.prefetch_depth == 0,
                 "execution.prefetch_depth: 0 for l2l_sync (each layer is transferred when it is needed)"),
                (x.backend != "l2l_async" or x.prefetch_depth >= 1, "execution.prefetch_depth: >= 1 for l2l_async"),
                (x.backend != "l2l_async" or x.weight_buffers >= x.prefetch_depth + 1,
                 "execution.weight_buffers: >= prefetch_depth + 1 (one buffer is being computed on while prefetch_depth "
                 "others are being filled)"),
            ]
    return checks


def unimplemented(cfg: Config) -> list:
    """Settings this code base accepts in a config but cannot execute yet. Training and evaluation refuse them (no
    silent resident fallback); the L2L lab reports them as BLOCKED."""
    x = cfg.execution
    sync = x.backend in ("l2l_sync", "l2l_async")  # Tickets 03-08: execution.backend.L2LBackend
    return ([f"execution.backend {x.backend!r}"] * (x.backend not in ("resident", "l2l_sync", "l2l_async"))
            + ["execution.resident_layers > 0 (L2L streams every decoder layer)"] * bool(sync and x.resident_layers)
            + ["execution.vocab_chunk_size (H03 unresolved)"] * (x.vocab_chunk_size is not None)
            + ["execution.kv_cache 'cpu'"] * (x.kv_cache != "gpu" and not sync)
            + ["streaming"] * cfg.streaming.enabled + ["dflash"] * cfg.dflash.enabled)


def require_implemented(cfg: Config):
    if missing := unimplemented(cfg):
        raise SystemExit(f"config {cfg.run.name!r} uses settings that are not implemented yet: {missing}. Refusing to "
                         "run instead of falling back to the resident backend.")


# --- historical configs ------------------------------------------------------------------------------------------

_MISSING = object()


def _pop(raw, dotted):
    """Remove raw[a][b][c] for "a.b.c" and return it (_MISSING if absent); parents left empty are removed too."""
    *parents, key = dotted.split(".")
    chain = [raw]
    for p in parents:
        if not isinstance(chain[-1].get(p), dict):
            return _MISSING
        chain.append(chain[-1][p])
    value = chain[-1].pop(key, _MISSING)
    for p, parent in zip(reversed(parents), reversed(chain[:-1])):
        if parent[p] == {}:
            del parent[p]
    return value


def _put(raw, dotted, value):
    *parents, key = dotted.split(".")
    for p in parents:
        raw = raw.setdefault(p, {})
    raw[key] = value


def historical_keys(raw: dict) -> list:
    """Dotted keys of `raw` that belong to the historical schema."""
    return [k for k in HISTORICAL if _pop(copy.deepcopy(raw), k) is not _MISSING]


def upgrade(raw: dict) -> dict:
    """A historical config mapping in the current schema (a current one is returned unchanged). Moved keys are renamed;
    a removed key must hold the value the current code implements, so nothing is reinterpreted silently."""
    raw = copy.deepcopy(raw)
    for old, new in MOVED.items():
        if (v := _pop(raw, old)) is not _MISSING:
            _put(raw, new, v)
    for key, implemented in REMOVED.items():
        v = _pop(raw, key)
        same_type = type(v) is type(implemented) or (type(implemented) is float and type(v) is int)  # YAML 0 for 0.0
        if v is not _MISSING and not (same_type and v == implemented):
            raise ValueError(f"{key}: {v!r} is a historical setting the current code does not implement (only "
                             f"{implemented!r}); this config cannot be loaded")
    mode = _pop(raw, "training.batch_mode")
    if mode not in (_MISSING, "unconstrained"):  # unconstrained is the shuffled sampler, bitwise (tests/test_config.py)
        _put(raw, "training.legacy_batch_mode", mode)
    return raw


# --- loading -----------------------------------------------------------------------------------------------------

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


def load(path, historical=False) -> Config:
    """The validated config of a YAML file. historical=True also accepts a config written in the historical schema
    (a saved runs/<name>/config.yaml, a checkpoint's config): needed to resume or evaluate older runs."""
    raw = read(path)
    if old := historical_keys(raw):
        if not historical:
            raise ValueError(f"{Path(path).name}: {old} belong to the historical config schema (removed or renamed, see "
                             "README); a new run must use the current schema")
        raw = upgrade(raw)
    return validate(_build(Config, raw, Path(path).name))


def as_dict(cfg: Config) -> dict:
    """The resolved config as plain data; training.legacy_batch_mode appears only when it is set."""
    out = dataclasses.asdict(cfg)
    if out["training"]["legacy_batch_mode"] is None:
        del out["training"]["legacy_batch_mode"]
    return out


def sha256(cfg: Config) -> str:
    """Hash of the fully resolved config (defaults included), independent of key order and of how the YAML was split
    across files. Configs of runs trained before the current schema hashed their historical form, so their recorded
    hash differs from this one. A section added later (OPTIONAL_SECTIONS) is left out while it equals its defaults
    exactly, so configs that do not use it keep the hash they had before it existed."""
    out = as_dict(cfg)
    for name, cls in OPTIONAL_SECTIONS.items():
        if getattr(cfg, name) == cls():
            del out[name]
    return hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()


def differences(a: dict, b: dict, prefix="") -> list:
    """Dotted keys whose values differ between two config dicts."""
    keys = sorted(set(a) | set(b))
    return [d for k in keys for d in (differences(a[k], b[k], f"{prefix}{k}.")
                                       if isinstance(a.get(k), dict) and isinstance(b.get(k), dict)
                                       else [f"{prefix}{k}"] if a.get(k, ...) != b.get(k, ...) else [])]


def normalized(raw: dict) -> dict:
    """A saved config dict (a training log's yaml_config, a checkpoint's config) in the current schema, defaults filled
    in. A historical batch mode survives as training.legacy_batch_mode, so it never compares equal to the shuffled one."""
    return as_dict(_build(Config, upgrade(raw), "saved config"))
