"""Multi-step GRPO + LoRA: fp32 adapters on selected projections of a frozen Qwen3 base model.

Run through scripts/run.py train --config <yaml> [--resume runs/<run>/checkpoints/step<N>]; every setting comes from the config
(qwen_grpo/config.py). Rollouts: sampled at temperature 1, training.max_new_tokens. Evaluation: greedy, in chunks,
with the YAML evaluation settings (defaults: those of the saved baseline, so runs stay comparable with it).
Checkpoints hold the adapter and the training state (optimizer, progress, log so far, RNG), so a run resumes from any
checkpoint it saved and continues as the uninterrupted run would have (tests/test_trainer.py checks this bitwise on CPU).
With memory.guard on CUDA the whole run executes under execution.profiler.Profiler: nvidia-smi / cudaMemGetInfo samples
are kept raw in training/memory/<session>.json and the run stops (status failed) at the first sample above
run.gpu_budget_gib. training.log lines carry a UTC timestamp (the console output does not).
Outputs (paths: qwen_grpo/runs.py): runs/<run>/config.yaml (resolved config), manifest.json (identity, status),
training/training.json (config and its sha256, environment, per-step log, subset evaluations, final full-validation
comparison, memory), training/training.log (console output), training/validation.json (raw full-validation predictions
and label scores of the final checkpoint), checkpoints/step<N>/; then summary.json, plots/ and report.md (reporting.py).
"""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
from transformers import AutoTokenizer

from qwen_grpo import runs
from qwen_grpo.checkpointing import (load_adapter, load_training_state, rel, save_checkpoint, set_rng_state,
                                     training_lock)
from qwen_grpo.evidence import code_sha256
from qwen_grpo.config import DTYPES, Config, as_dict, differences, normalized, require_implemented, sha256
from qwen_grpo.evaluation import (BOOTSTRAP_SEED, evaluate, label_scores, metrics, paired_auc_bootstrap, paired_bootstrap,
                                  paired_changes, paired_compare, predictions_path, settings, summary, win_matrix)
from qwen_grpo.model import EOS_IDS, digest, expected_lora_params, lora_model, lora_modules
from qwen_grpo.execution.backend import make_backend
from qwen_grpo.streaming import resume as stream_resume
from qwen_grpo.streaming.dataset import PromptStream
from qwen_grpo.streaming.rollouts import RolloutStore
from qwen_grpo.reporting import update_run
from qwen_grpo.sentiment import SPLITS_PATH, build_prompts, load_splits, parse_label
from qwen_grpo.training import (balanced_subset, check_grouping, decode, group_rewards, grpo_update, prepare,
                                shuffled_batches)

ROOT = Path(__file__).resolve().parents[1]  # the git repository (provenance); outputs: runs.ROOT
BASE_FILE = predictions_path("validation")  # saved base predictions; the final evaluation compares with them if they match
SEED = BOOTSTRAP_SEED  # kept: research/trl_reference (removed in 4424e0f) imported it
PACKAGES = ("torch", "transformers", "peft", "datasets", "pyyaml")


def l2(norms):
    """Aggregate L2 norm sqrt(sum_k ||theta_k||^2) from per-tensor norms."""
    return math.sqrt(sum(x * x for x in norms))


def to(x, device):
    """x with every tensor in it (nested dicts) moved to device."""
    if isinstance(x, dict):
        return {k: to(v, device) for k, v in x.items()}
    return x.to(device) if isinstance(x, torch.Tensor) else x


class InjectedCrash(KeyboardInterrupt):
    """A simulated crash (Ticket 09 crash injection); a KeyboardInterrupt, so the run is marked interrupted."""


_CRASH_SEEN = Counter()


def crash_point(point):
    """Crash injection: raises InjectedCrash at the N-th occurrence of `point` in this process when the environment
    variable QWEN_GRPO_CRASH is "point" (N = 1) or "point@N". Points: prompt_read, rollouts_generated, update (an
    applied but uncommitted optimizer update), files_written / committed_before_rename (inside checkpoint publication),
    after_checkpoint. Tests may replace this function. A no-op when the variable is unset."""
    spec = os.environ.get("QWEN_GRPO_CRASH")
    if not spec:
        return
    name, _, nth = spec.partition("@")
    if name == point:
        _CRASH_SEEN[point] += 1
        if _CRASH_SEEN[point] == int(nth or 1):
            raise InjectedCrash(f"injected crash at {point} (occurrence {_CRASH_SEEN[point]})")


stream_resume._crash_point = lambda name: crash_point(name)  # publication points share the trainer's hook


def streaming_dir(cfg):
    """streaming.directory, relative paths resolved inside the run directory (runs/<name>/<directory>)."""
    d = Path(cfg.streaming.directory)
    return d if d.is_absolute() else runs.run_dir(cfg.run.name) / d


def git_state():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def model_revision(model_id):
    """The commit of the locally cached snapshot of a Hub model (what from_pretrained loads), else None. Local only."""
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(model_id, "config.json")
    except Exception:  # noqa: BLE001 - provenance must never stop a run
        return None
    return Path(path).parent.name if isinstance(path, str) else None


def host():
    """CPU count and total host RAM (bytes, /proc/meminfo); no host name or user data."""
    out = {"cpu_count": os.cpu_count(), "mem_total_bytes": None}
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                out["mem_total_bytes"] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return out


def environment(splits, device, models=()):
    """Provenance of a training session: code version, library versions, dataset identity, hardware and the cached
    revision of each model in `models` (target, drafter)."""
    env = {**git_state(),
           "python": sys.version.split()[0], "platform": platform.platform(),
           "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
           "dataset": {"name": splits["dataset"], "split_seed": splits["seed"], "file": rel(SPLITS_PATH),
                       "sha256": hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest(),
                       "sizes": {s: len(splits[s]) for s in ("train", "validation", "test")}},
           "device": str(device), "host": host(),
           "models": {m: {"revision": model_revision(m)} for m in models},
           "code_sha256": code_sha256()}  # content identity of the experiment code (qwen_grpo/evidence.py)
    if torch.device(device).type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_gib": p.total_memory / 2**30,
                      "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    return env


def check_resume(cfg, checkpoint, state):
    """Resume only the run this checkpoint belongs to, with the same settings, without overwriting later checkpoints."""
    name, step, layout = runs.parse_checkpoint(checkpoint)
    if layout != "run":
        raise SystemExit(f"{checkpoint} is not a checkpoint under runs/: runs from before runs/ existed cannot be resumed")
    if name != cfg.run.name or step != state["step"]:
        raise SystemExit(f"{checkpoint} is not a checkpoint of run {cfg.run.name!r}")
    strip = lambda c: {k: ({**v, "overwrite": None} if k == "run" else v) for k, v in c.items() if k != "verification"}
    diff = differences(strip(normalized(state["config"])), strip(as_dict(cfg)))
    if diff and set(diff) <= CONTINUATION_KEYS and continuation_of(name, checkpoint) is not None:
        pass  # the copied source checkpoint of a recorded continuation (continue_run): only the step budget changed
    elif diff:
        raise SystemExit(f"{checkpoint} was trained with a different config ({', '.join(diff)}); resume needs the same "
                         f"settings, e.g. --config {rel(runs.run_dir(name) / 'config.yaml')}")
    if later := [rel(p) for s, p in runs.checkpoints(name) if s > step]:
        raise SystemExit(f"resuming from step {step} would overwrite later checkpoints {later}")


# --- continuation: a longer run that continues another run's checkpoint ----------------------------------------------
# A run's identity includes training.max_steps, so a 60-update run cannot simply be resumed to 210 updates (check_resume
# refuses the different config). continue_run creates a NEW run that starts from a byte-for-byte copy of the source
# run's checkpoint (adapter, AdamW state, RNG state, sampler state, counters, log so far) and of its streaming
# directory, records where it came from, and resumes it with the standard resume path. Only CONTINUATION_KEYS may
# differ from the source config, and only for that recorded copied checkpoint: the continued trajectory is the one an
# uninterrupted run with the larger max_steps would have produced. The source run is only read.
CONTINUATION_KEYS = frozenset({"run.name", "training.max_steps"})


def _tree_sha256(path):
    h = hashlib.sha256()
    for p in sorted(Path(path).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode())
    return h.hexdigest()


def continuation_of(name, checkpoint):
    """The manifest's continued_from record if `checkpoint` is the copied source checkpoint of run `name` and still has
    the recorded content, else None."""
    rec = (runs.manifest(name) or {}).get("continued_from")
    if not rec:
        return None
    copied = runs.ROOT / rec["copied_checkpoint"]
    if Path(checkpoint).resolve() != copied.resolve():
        return None
    if _tree_sha256(copied) != rec["checkpoint_tree_sha256"]:
        raise SystemExit(f"{checkpoint}: the copied source checkpoint changed since the continuation was set up")
    return rec


def continue_run(cfg, source, device="cuda"):
    """Train run cfg.run.name as the continuation of `source` (a checkpoint runs/<src>/checkpoints/step<N> of a
    completed streaming run) up to cfg.training.max_steps updates. Refuses: an existing target run, a source that is not
    a valid resumable checkpoint of a completed run, a config differing from the source's in anything but
    CONTINUATION_KEYS, or a step budget not above N."""
    source = Path(source).resolve()
    src_name, step, layout = runs.parse_checkpoint(source)
    name = cfg.run.name
    if layout != "run" or not cfg.streaming.enabled:
        raise SystemExit("continuation needs a streaming run's checkpoint under runs/ (crash-consistent, with its "
                         "sampler and rollout records)")
    if src_name == name:
        raise SystemExit(f"--continue-from: {source} belongs to run {name!r} itself; use --resume for that")
    if runs.run_dir(name).exists():
        raise SystemExit(f"run {name!r} already exists ({rel(runs.run_dir(name))}/): a continuation is set up once; "
                         f"resume it with --resume latest")
    ok, why = stream_resume.validate(source)
    if not ok:
        raise SystemExit(f"{source} is not a valid resumable checkpoint: {why}")
    src_manifest = runs.manifest(src_name) or {}
    if src_manifest.get("status") not in ("completed", "complete"):
        raise SystemExit(f"run {src_name!r} is {src_manifest.get('status')!r}: only a completed run is continued")
    state = load_training_state(source)
    strip = lambda c: {k: ({**v, "overwrite": None} if k == "run" else v) for k, v in c.items() if k != "verification"}
    diff = set(differences(strip(normalized(state["config"])), strip(as_dict(cfg))))
    if not diff <= CONTINUATION_KEYS:
        raise SystemExit(f"{name} differs from {src_name} in {sorted(diff - CONTINUATION_KEYS)}: a continuation may "
                         f"change only {sorted(CONTINUATION_KEYS)}")
    if state["step"] != step or cfg.training.max_steps <= step:
        raise SystemExit(f"continuation from step {step} needs training.max_steps > {step} "
                         f"(got {cfg.training.max_steps})")
    src_dir, dst = runs.run_dir(src_name), runs.run_dir(name)
    try:  # setup only; a failure here removes the half-built target (the source is never touched)
        src_stream = src_dir / Path(normalized(state["config"])["streaming"]["directory"])
        (dst / "checkpoints").mkdir(parents=True)
        runs.training_dir(name).mkdir(parents=True)
        for s in (0, step):  # the initial adapter (parameter-delta reference) and the checkpoint continued from
            shutil.copytree(runs.checkpoint_path(src_name, s), runs.checkpoint_path(name, s))
        shutil.copytree(src_stream, streaming_dir(cfg))
        copied = runs.checkpoint_path(name, step)
        for s in (0, step):
            ok, why = stream_resume.validate(runs.checkpoint_path(name, s), resumable=s > 0)
            if not ok:
                raise SystemExit(f"copy of {src_name} step{s} is not valid: {why}")
        (dst / "config.yaml").write_text(f"# resolved config of run {name}, sha256 {sha256(cfg)}\n"
                                         + yaml.safe_dump(as_dict(cfg), sort_keys=False))
        runs.create(name, sha256(cfg), git_state())
        record = {"run": src_name, "run_id": src_manifest.get("id"), "checkpoint": rel(source), "step": step,
                  "config_sha256": src_manifest.get("config_sha256"), "copied_checkpoint": rel(copied),
                  "checkpoint_tree_sha256": _tree_sha256(copied), "source_checkpoint_tree_sha256": _tree_sha256(source),
                  "streaming_tree_sha256": _tree_sha256(streaming_dir(cfg)), "changed": sorted(diff)}
        if record["checkpoint_tree_sha256"] != record["source_checkpoint_tree_sha256"]:
            raise SystemExit("the copied checkpoint differs from its source")
        m = runs.manifest(name)
        m["continued_from"] = record  # identity: a new run (own id and config hash) whose trajectory starts at `source`
        runs.write_json(runs.run_dir(name) / "manifest.json", m)
    except BaseException:
        shutil.rmtree(dst, ignore_errors=True)
        raise
    print(f"continuation: run {name} continues {rel(source)} (step {step}) to {cfg.training.max_steps} updates")
    return train(cfg, resume=copied, device=device)


def coverage(result, train_ids, attempts, steps, G):
    """What the run actually trained on, from its own records: every rollout attempt (updates and zero-signal batches)
    with the prompt ids it drew. A full pass over the pool is claimed only when every pool prompt was visited."""
    by_attempt = {r["attempt"]: r["prompt_ids"] for r in result.get("training", [])}
    by_attempt.update({z["attempt"]: z["prompt_ids"] for z in result.get("zero_signal", [])})
    selections = [i for a in sorted(by_attempt) for i in by_attempt[a]]
    pool = set(train_ids)
    visited = set(selections) & pool
    return {"train_pool_prompts": len(pool), "rollout_attempts": attempts, "successful_updates": steps,
            "attempts_with_records": len(by_attempt), "prompt_selections": len(selections),
            "unique_prompts_visited": len(visited), "coverage_fraction": len(visited) / len(pool) if pool else None,
            "zero_signal_batches": len(result.get("zero_signal", [])),
            "skipped_batches": len(result.get("zero_signal", [])), "generated_completions": len(selections) * G,
            "full_pass_over_pool": bool(pool) and visited == pool}


def start(cfg, resume):
    """A new run: refuse a name already in use, then create runs/<name>/ with config.yaml and manifest.json. A resumed
    run: check the checkpoint belongs to it and to this config. Returns the checkpoint's training state, or None."""
    name = cfg.run.name
    if resume is not None:
        if cfg.streaming.enabled:  # crash-consistent checkpoints: only a valid one may be resumed
            ok, why = stream_resume.validate(resume)
            if not ok:
                raise SystemExit(f"{resume} is not a valid checkpoint ({why}); latest valid: "
                                 f"{stream_resume.latest_valid(name)[0]}")
            done = stream_resume.prepare(name)
            if done["removed_tmp"] or done["quarantined"]:
                print(f"streaming resume: removed interrupted publications {done['removed_tmp']}, quarantined invalid "
                      f"checkpoints {done['quarantined']}")
        state = load_training_state(resume)
        check_resume(cfg, resume, state)
        runs.set_status(name, "running", resumed_from=rel(resume))
        return state
    if legacy := runs.legacy_outputs(name):
        raise SystemExit(f"run.name {name!r} belongs to a run from before runs/ existed: {[rel(p) for p in legacy]}\n"
                         "  choose another run.name (those artifacts are kept; `remove --config --checkpoint` deletes them)")
    if runs.run_dir(name).exists():
        if not cfg.run.overwrite:
            raise SystemExit(f"run {name!r} already exists ({rel(runs.run_dir(name))}/), refusing to overwrite it\n"
                             "  choose another run.name, or set run.overwrite: true to delete it first")
        shutil.rmtree(runs.run_dir(name))
        print(f"run.overwrite: deleted {rel(runs.run_dir(name))}/")
    runs.training_dir(name).mkdir(parents=True)
    (runs.run_dir(name) / "config.yaml").write_text(f"# resolved config of run {name}, sha256 {sha256(cfg)}\n"
                                                    + yaml.safe_dump(as_dict(cfg), sort_keys=False))
    runs.create(name, sha256(cfg), git_state())
    return None


@contextmanager
def tee(path):
    """Copy everything printed to `path`, appended (a resumed run continues the same log); every line in the file starts
    with its UTC time (the console keeps the plain text)."""
    with open(path, "a") as f:
        out = sys.stdout
        line_start = [True]

        class Tee:
            def write(self, s):
                for part in s.splitlines(keepends=True):
                    if line_start[0]:
                        f.write(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ "))
                    f.write(part)
                    line_start[0] = part.endswith("\n")
                return out.write(s)

            def flush(self):
                f.flush()
                out.flush()

            def __getattr__(self, attr):
                return getattr(out, attr)
        sys.stdout = Tee()
        try:
            yield
        finally:
            sys.stdout = out


def train(cfg: Config, resume=None, device="cuda"):
    """Train run cfg.run.name in runs/<name>/ under runs/.<name>.lock (so `remove` cannot delete it meanwhile). The
    manifest records how training ended; summary.json, plots/ and report.md are then rebuilt from the saved files."""
    name = cfg.run.name
    require_implemented(cfg)  # before start(): a refused config leaves nothing behind
    if mode := cfg.training.legacy_batch_mode:
        raise SystemExit(f"run {name!r} uses the historical batch_mode {mode!r}, which the current code does not implement "
                         "(it has only the shuffled sampler): it cannot be trained or resumed. Its saved outputs stay "
                         "readable; train it with the code of commit 9bd4500 if needed")
    if resume is not None and str(resume) == "latest":  # the highest-step valid checkpoint (streaming runs)
        resume, rejected = stream_resume.latest_valid(name)
        if resume is None:
            raise SystemExit(f"run {name!r} has no valid resumable checkpoint (rejected: {rejected})")
    with training_lock(runs.lock_path(name)):
        state = start(cfg, resume)
        guard = None
        _HELD.clear()  # resources _train acquires (the backend), released below whatever happens
        try:
            guard = memory_guard(cfg, device)  # inside: a CUDA error here marks the run failed, not "running"
            with tee(runs.training_dir(name) / "training.log"), guard or nullcontext():
                result = _train(cfg, resume, state, device)
        except BaseException as e:
            exceeded = guard is not None and guard.violations
            status = "interrupted" if isinstance(e, KeyboardInterrupt) and not exceeded else "failed"
            error = (f"GPU memory budget exceeded: {guard.violations[0]}" if exceeded else f"{type(e).__name__}: {e}")
            runs.set_status(name, status, error=error)
            save_memory(cfg, guard)
            try:
                update_run(name)
            except Exception as r:  # the original error matters more than a report that could not be built
                print(f"report not rebuilt ({type(r).__name__}: {r}); run `scripts/run.py report --run {name}`")
            if exceeded:
                raise SystemExit(f"run {name!r} stopped: {error} (raw samples: "
                                 f"{rel(runs.training_dir(name) / 'memory')}/)") from e
            raise
        finally:
            while _HELD:  # the L2L weight store unregisters its pinned memory (GBs for Qwen3-4B)
                _HELD.pop().close()
            if guard is not None and cfg.memory.allocator_cap:
                torch.cuda.set_per_process_memory_fraction(1.0, torch.device(device))
        save_memory(cfg, guard)
        runs.set_status(name, "completed")
    report = update_run(name)
    print(f"\nrun {name}: {rel(runs.run_dir(name))}/  report: {rel(report)}")
    return result


def models(cfg):
    """The Hub models a run loads: the target, and the DFlash drafter if enabled."""
    return (cfg.model.name, *([cfg.dflash.draft_model] if cfg.dflash.enabled else []))


def memory_guard(cfg, device):
    """The memory.guard Profiler for this run (started by entering it), or None: guard off, or not on CUDA (then nothing
    is enforced and the run's memory record says so). allocator_cap is applied here, before any model is loaded."""
    if not cfg.memory.guard or torch.device(device).type != "cuda":
        if cfg.memory.guard:
            print(f"memory.guard: not enforced on device {device!r} (no CUDA)")
        return None
    from qwen_grpo.execution.profiler import Profiler
    budget = int(cfg.run.gpu_budget_gib * 2**30)
    dev = torch.device(device)
    if cfg.memory.allocator_cap:
        total = torch.cuda.get_device_properties(dev).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget / total), dev)
    return Profiler(budget_bytes=budget, interval=cfg.memory.interval_s, device=device)


def save_memory(cfg, guard):
    """The guard's raw samples, violations and evidence grade: training/memory/<UTC session start>.json (one file per
    training session, so a resumed run keeps every earlier session's samples)."""
    if guard is None:
        return None
    out = runs.training_dir(cfg.run.name) / "memory"
    stamp = datetime.fromtimestamp(guard.samples[0]["t"] if guard.samples else time.time(), timezone.utc)
    rep = {"budget_gib": cfg.run.gpu_budget_gib, "memory": as_dict(cfg)["memory"], **guard.report(raw=True)}
    return runs.write_json(out / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}.json", rep)


def check_device(cfg, device):
    """Refuse a device the run cannot use, before anything is loaded: CUDA requested but unavailable, a torch build
    without kernels for the GPU (e.g. sm_61 on a cu130 wheel), or bfloat16 weights on a GPU without bf16 support. No
    fallback to the CPU or to float32 is ever taken."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise SystemExit(f"device {device!r} requested but torch.cuda.is_available() is False (torch {torch.__version__}, "
                         f"built for CUDA {torch.version.cuda}); refusing to fall back to the CPU")
    p = torch.cuda.get_device_properties(dev)
    code = p.major * 10 + p.minor
    arches = torch.cuda.get_arch_list()
    if f"sm_{code}" not in arches and not any(a.startswith("compute_") and int(a.split("_")[1]) <= code for a in arches):
        raise SystemExit(f"{p.name} (sm_{code}) is not supported by this torch build ({torch.__version__}: {arches}); "
                         "install a torch wheel built for this GPU")
    if cfg.model.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise SystemExit(f"model.dtype bfloat16 but {p.name} does not support bf16; refusing to switch precision "
                         "silently (set model.dtype: float32 explicitly if that experiment is intended)")


def rss_bytes():
    """This process's resident set size (Linux /proc), or None."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def rollout_evidence(backend):
    """What generated the last rollout batch (kept in the step's timings, outside the bitwise-compared training rows):
    the generator class and, for DFlash, its per-call statistics, i.e. proof that DFlash actually ran rather than that
    the config asked for it. None for the resident backend's model.generate()."""
    gen = getattr(backend, "sampler", None)
    if gen is None:
        return None
    out = {"generator": type(gen).__name__}
    last = getattr(gen, "last", None)
    if out["generator"] == "DFlashGenerator" and last is not None:
        acc = list(last["accepted_per_pass"])
        out.update(draft_model=getattr(backend, "draft_model", None), drafter_bytes=last.get("drafter_bytes"),
                   steps=int(last["steps"]), target_passes=int(last["target_passes"]), accepted_per_pass=acc,
                   accepted_drafts=int(sum(acc)), calls=len(getattr(gen, "history", [])))
    return out


def weight_bytes_copied(backend):
    """Frozen weight bytes the L2L executor has copied host -> GPU so far (None for the resident backend)."""
    ex = getattr(backend, "executor", None)
    if ex is None:
        return None
    if hasattr(ex, "stats"):  # async prefetching executor: counts every copy it issued
        return ex.stats["bytes_copied"]
    return sum(t["slab_bytes"] for t in getattr(ex, "trace", []) if t.get("loaded", True))


_HELD = []  # closeable resources of the training session in progress (one per process; train() empties it)


def _train(cfg: Config, resume, state, device):
    check_device(cfg, device)  # before any tokenizer or model is loaded; the run is recorded as failed
    t, lo, d, ev = cfg.training, cfg.lora, cfg.data, cfg.evaluation
    B, G, name = t.batch_size, t.group_size, cfg.run.name
    gen_kw, score_kw = settings(ev)
    on_cuda = torch.device(device).type == "cuda"
    gib = lambda f: f() / 2**30 if on_cuda else None
    dtype = DTYPES[cfg.model.dtype]
    decoding = {"do_sample": False, "max_new_tokens": ev.max_new_tokens}

    splits = load_splits()
    train_set = balanced_subset(splits["train"], d.train_samples, d.subset_seed)
    val_all = [{"id": i, **r} for i, r in enumerate(splits["validation"])]
    val_sub = sorted(balanced_subset(splits["validation"], d.val_subset, d.subset_seed), key=lambda e: e["id"])

    def saved_predictions(path):
        """Saved predictions of the same model under the same decoding, aligned with validation, else None."""
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        if saved.get("model") != cfg.model.name or saved.get("decoding") != decoding:
            return None
        preds = saved["predictions"]
        assert [(p["id"], p["text"], p["label"]) for p in preds] == [(e["id"], e["text"], e["label"]) for e in val_all], path
        return preds

    torch.manual_seed(t.seed)  # LoRA A init
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    assert tok.pad_token_id == EOS_IDS[1]  # baseline.py pads with tok.pad_token_id, lora_model with EOS_IDS[1]
    # L2L: load on the CPU; the backend keeps the frozen decoder weights there (WeightStore) and moves the rest
    load_device = device if cfg.execution.backend == "resident" else "cpu"
    model = lora_model(dtype, load_device, lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
    backend = make_backend(cfg, model, device)  # generation, log-probs and evaluation run through it
    _HELD.append(backend)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p for n, p in model.named_parameters() if not p.requires_grad}
    modules = lora_modules(model)
    module_names = [n for n, _ in modules]
    assert sorted(module_names) == sorted(model.peft_config["default"].target_modules), module_names
    n_trainable = sum(p.numel() for p in params.values())
    assert n_trainable == expected_lora_params(modules, lo.rank) and len(params) == 2 * len(modules)
    assert all(m.r["default"] == lo.rank and m.scaling["default"] == lo.alpha / lo.rank for _, m in modules)
    assert all(".lora_A." in n or ".lora_B." in n for n in params)
    assert all(p.dtype == torch.float32 for p in params.values())
    assert all(p.dtype == dtype for p in frozen.values())
    names_A, names_B = [n for n in params if ".lora_A." in n], [n for n in params if ".lora_B." in n]
    pairs = [(m.lora_A["default"].weight, m.lora_B["default"].weight) for _, m in modules]
    assert all(params[n].abs().max() == 0 for n in names_B)  # adapter contributes exactly zero: initial model = base
    n_total = n_trainable + sum(p.numel() for p in frozen.values())
    print(f"LoRA modules ({len(modules)}): {module_names}")
    print(f"trainable parameters: {n_trainable:,} of {n_total:,} ({100 * n_trainable / n_total:.4f}%)")
    init = {n: p.detach().clone() for n, p in params.items()}  # the seeded init, equal to checkpoints/<run>_step0
    frozen_digest = {n: digest(p) for n, p in frozen.items()}
    optimizer = torch.optim.AdamW(params.values(), lr=t.learning_rate, weight_decay=0.0)

    def eval_subset(step, ckpt=None):
        preds = evaluate(model, tok, val_sub, generator=backend.greedy, **gen_kw)
        result["evaluations"].append({"step": step, "checkpoint": ckpt, "split": "validation", "metrics": metrics(preds), "predictions": preds})
        print(f"  [val subset, step {step}] {summary(metrics(preds))}")

    if resume is None:
        if cfg.streaming.enabled:  # initial adapter (reference for parameter deltas), published crash-consistently
            stream_resume.publish(runs.checkpoint_path(name, 0), lambda tmp: model.save_pretrained(tmp), 0)
        else:
            model.save_pretrained(runs.checkpoint_path(name, 0))  # initial adapter: reference for parameter deltas
        result = {
            "model": cfg.model.name, "method": "GRPO+LoRA", "seed": t.seed, "yaml_config": as_dict(cfg),
            "config_sha256": sha256(cfg),
            "environment": environment(splits, device, models(cfg)),
            # how rollouts are produced (per-step proof: timings[].rollout_generator)
            "rollout_backend": {"backend": backend.name,
                                "generator": type(getattr(backend, "sampler", None)).__name__
                                if getattr(backend, "sampler", None) is not None else "model.generate",
                                "draft_model": getattr(backend, "draft_model", None),
                                "drafter_bytes": backend.drafter.nbytes() if getattr(backend, "drafter", None) else None},
            "config": {"lr": t.learning_rate, "steps": t.max_steps, "max_attempts": t.max_attempts, "train_samples": d.train_samples,
                       "train_ids": [e["id"] for e in train_set], "val_subset_ids": [e["id"] for e in val_sub],
                       "B": B, "G": G, "micro_batch_size": t.micro_batch_size,
                       "batch_policy": "shuffled (training.shuffled_batches); a zero-signal batch is consumed, not retried",
                       "rollout_max_new_tokens": t.max_new_tokens, "eval_max_new_tokens": ev.max_new_tokens,
                       "eval_batch": ev.batch_size, "sampling": {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0},
                       "clip_eps": t.clip_eps, "num_iterations": t.num_iterations, "kl_beta": 0.0, "optimizer": "AdamW(weight_decay=0)", "grad_clipping": None,
                       "lora": {"targets": as_dict(cfg)["lora"]["targets"], "target_modules": module_names,
                                "trainable_params": n_trainable, "r": lo.rank, "alpha": lo.alpha, "dropout": 0.0},
                       "dtypes": {"base": str(dtype), "lora": "torch.float32"}, "gpu_budget_gib": cfg.run.gpu_budget_gib,
                       "checkpoint_contents": "adapter weights + adapter_config + training_state.pt (optimizer, progress, log, RNG)",
                       "initial_checkpoint": rel(runs.checkpoint_path(name, 0))},
            "training": [], "zero_signal": [], "evaluations": [],
        }
        base_full = None
        if ev.full_validation:
            base_full = evaluate(model, tok, val_all, generator=backend.greedy, **gen_kw)
            saved = saved_predictions(BASE_FILE)
            mism = None if saved is None else [p["id"] for p, q in zip(base_full, saved) if p["response"] != q["response"]]
            result["base_reproduction"] = {"n": len(base_full), "mismatched_ids": mism, "metrics": metrics(base_full)}
            print(f"base full validation (in-process): {summary(metrics(base_full))}\n  mismatches vs {BASE_FILE.name}: "
                  + ("not compared (different model or decoding)" if mism is None else str(len(mism))))
        eval_subset(0)
        step = attempts = 0
        ckpt, mixed_counts, elapsed, pending = None, Counter(), 0.0, None
        torch.manual_seed(t.seed)  # rollout sampling
    else:
        load_adapter(model, resume)
        optimizer.load_state_dict(state["optimizer"])
        result, base_full = state["result"], state["base_full"]
        step, attempts, ckpt, elapsed = state["step"], state["attempts"], state["checkpoint"], state["train_seconds"]
        mixed_counts = Counter(state["mixed_counts"])
        pending = to(state.get("pending"), model.device)  # a batch whose remaining updates were cut by this checkpoint
        result.setdefault("resumed", []).append({"from": rel(resume), "step": step, "attempts": attempts,
                                                 "environment": environment(splits, device, models(cfg))})
        if (cont := continuation_of(name, resume)) is not None:  # first start of a continuation run
            result.update(yaml_config=as_dict(cfg), config_sha256=sha256(cfg), continued_from=cont)
            result["config"]["steps"] = t.max_steps
            ckpt = rel(resume)
            print(f"continuing run {cont['run']} from step {step} as run {name} (to {t.max_steps} updates)")
        print(f"resumed from {rel(resume)}: step {step}, attempt {attempts}")

    def stream_restore(s, step, attempts):
        """The pending group of a streaming checkpoint, read back from its on-disk record (policy version checked)."""
        ref = s.get("pending")
        if ref is None:
            return None
        store = RolloutStore(streaming_dir(cfg) / "rollouts", s["run_id"])
        rec = store.read(ref["group_id"], ref["policy_version"], model.device)
        keys = ("input_ids", "attention_mask", "response_ids", "completion_mask", "prompt_len")
        return {"rollouts": {k: rec[k] for k in keys}, "adv": rec["advantages"], "old_logp": rec["old_logp"],
                "group": ref["group"], "iteration": ref["iteration"], "policy_version": ref["policy_version"]}

    def save(step):
        """Adapter + resumable state; the subset evaluation runs first so the saved log and RNG state include it.
        Streaming: the pending group is a reference to its on-disk record, and the checkpoint is published
        crash-consistently (streaming/resume.py); publication is what commits the updates up to `step`."""
        path = runs.checkpoint_path(name, step)
        eval_subset(step, rel(path))
        st = {"config": as_dict(cfg), "step": step, "attempts": attempts, "checkpoint": rel(path),
              "mixed_counts": dict(mixed_counts), "train_seconds": time.time() - t0, "result": result,
              "base_full": base_full, "pending": to(pending, "cpu")}
        if not cfg.streaming.enabled:
            save_checkpoint(model, optimizer, path, st)
            return rel(path)
        st["pending"] = None
        st["streaming"] = {"sampler": prompts_on_disk.state(), "rollout_group": attempts, "policy_version": step,
                           "optimizer_steps": step, "run_id": run_id, "dataset_sha256": prompts_on_disk.sha256,
                           "pending": None if pending is None else {"group_id": pending["group"]["attempt"],
                                                                    "policy_version": pending["policy_version"],
                                                                    "iteration": pending["iteration"],
                                                                    "group": pending["group"]}}
        stream_resume.publish(path, lambda tmp: save_checkpoint(model, optimizer, tmp, st), step)
        crash_point("after_checkpoint")
        return rel(path)

    if cfg.streaming.enabled:  # disk-backed prompts with a saved sampler state: no replay needed
        sdir = streaming_dir(cfg)
        run_id = runs.manifest(name)["id"]
        prompts_on_disk = PromptStream.build(sdir / "prompts", train_set, t.seed, B)
        rollout_store = RolloutStore(sdir / "rollouts", run_id)
        if resume is not None:
            s = state["streaming"]
            prompts_on_disk.restore(s["sampler"])
            if prompts_on_disk.batches != attempts or s["rollout_group"] != attempts or s["optimizer_steps"] != step:
                raise SystemExit(f"inconsistent checkpoint: sampler drew {prompts_on_disk.batches} batches, rollout group "
                                 f"{s['rollout_group']}, optimizer steps {s['optimizer_steps']} vs attempts {attempts}, "
                                 f"step {step}")
            if orphans := rollout_store.discard_after(attempts):
                print(f"streaming resume: uncommitted rollout groups {orphans} moved aside (they are regenerated)")
            pending = stream_restore(s, step, attempts)  # the pending group: from its record on disk
            if pending is not None:
                print(f"streaming resume: pending group {pending['group']['attempt']} restored from disk at iteration "
                      f"{pending['iteration']}")
        stream = None
    else:
        stream = shuffled_batches(train_set, B, seed=t.seed)
        for _ in range(attempts):  # replay: every attempt, zero-signal or not, consumed one batch
            next(stream)
    if resume is not None:
        set_rng_state(state["rng"])  # last: nothing between here and the next rollout may draw random numbers
    t0 = time.time() - elapsed
    # pending: the current rollout batch while it still has updates to run (rollouts, advantages, the old_logp computed
    # once at sampling time, its log record, next iteration); a checkpoint saves it, so resuming continues it exactly.
    while step < t.max_steps and (pending is not None or attempts < t.max_attempts):
        t_step = time.time()
        timing = {"rollout_s": None, "prepare_s": None}
        sync = torch.cuda.synchronize if on_cuda else (lambda: None)  # phase times include the queued GPU work
        copied0 = weight_bytes_copied(backend)
        if pending is None:
            if stream is None:
                batch, occurrences = prompts_on_disk.next_batch()
                crash_point("prompt_read")
            else:
                batch = next(stream)
            attempts += 1
            prompts = build_prompts(tok, [e["text"] for e in batch])
            t0_phase = time.time()
            rollouts = backend.generate_rollouts(tok, prompts, G=G, max_new_tokens=t.max_new_tokens)
            sync()
            timing["rollout_s"] = time.time() - t0_phase
            timing["rollout_tokens"] = int(rollouts["completion_mask"].sum())
            timing["rollout_generator"] = rollout_evidence(backend)
            check_grouping(rollouts, tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"].to(model.device), G)
            responses = decode(tok, rollouts)
            rewards = group_rewards(responses, [e["label"] for e in batch], G)  # [B, G]
            mixed = int((rewards != rewards[:, :1]).any(dim=1).sum())
            mixed_counts[mixed] += 1
            group = {"attempt": attempts, "prompt_ids": [e["id"] for e in batch], "labels": [e["label"] for e in batch],
                     "mixed_groups": mixed, "responses": responses, "rewards": rewards.tolist(), "mean_reward": rewards.mean().item(),
                     "invalid_responses": sum(parse_label(r) is None for r in responses),
                     "rollout_sha256": hashlib.sha256(rollouts["response_ids"].cpu().numpy().tobytes()).hexdigest()[:16],
                     "mean_completion_tokens": rollouts["completion_mask"].sum(1).float().mean().item()}  # through the first EOS
            t0_phase = time.time()
            targets = prepare(model, rollouts, rewards, t.micro_batch_size, logprobs=backend.logprobs)
            sync()
            timing["prepare_s"] = time.time() - t0_phase
            if stream is None:  # the group's record on disk, before any update uses it
                group["occurrences"] = occurrences
                rollout_store.write({"group_id": attempts, "policy_version": step, "occurrences": occurrences,
                                     "prompt_ids": group["prompt_ids"], "labels": group["labels"], **rollouts,
                                     "rewards": rewards, "zero_signal": targets is None,
                                     "advantages": None if targets is None else targets[0],
                                     "old_logp": None if targets is None else targets[1]})
                crash_point("rollouts_generated")
            if targets is None:
                del rollouts
                result["zero_signal"].append(group)
                # kept apart from the zero_signal records, which runs compare bitwise
                result.setdefault("zero_signal_rollout_generators", []).append(
                    {"attempt": attempts, "rollout_generator": timing["rollout_generator"]})
                continue
            pending = {"rollouts": rollouts, "adv": targets[0], "old_logp": targets[1], "group": group, "iteration": 0,
                       "policy_version": step}
            del rollouts, targets
        group = pending["group"]
        t0_phase = time.time()
        out = grpo_update(model, params, optimizer, pending["rollouts"], pending["adv"], pending["old_logp"],
                          clip_eps=t.clip_eps, iteration=pending["iteration"], micro_batch_size=t.micro_batch_size,
                          logprobs=backend.logprobs)
        sync()
        timing["update_s"] = time.time() - t0_phase  # forward + backward (+ recomputation) + AdamW step
        crash_point("update")  # the update is applied in memory but not committed (no checkpoint contains it yet)
        pending["iteration"] += 1
        if pending["iteration"] == t.num_iterations:
            pending = None
        step += 1
        with torch.no_grad():  # norms over several modules aggregate as sqrt(sum of squares)
            row = {"step": step, **group, **{k: v for k, v in out.items() if k != "grad_norms"},
                   "grad_norm_A": l2(out["grad_norms"][n] for n in names_A), "grad_norm_B": l2(out["grad_norms"][n] for n in names_B),
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "step_seconds": time.time() - t_step,  # GRPO update (+ rollout generation and rewards at iteration 0)
                   "lora_A_norm": l2(params[n].norm().item() for n in names_A),
                   "lora_B_norm": l2(params[n].norm().item() for n in names_B),
                   "effective_adapter_norm": l2(((lo.alpha / lo.rank) * Bm @ A).norm().item() for A, Bm in pairs),
                   "change_from_init": math.sqrt(sum((p - init[n]).pow(2).sum().item() for n, p in params.items())),
                   "cuda_allocated_gib": gib(torch.cuda.memory_allocated),
                   "peak_allocated_gib": gib(torch.cuda.max_memory_allocated),
                   "peak_reserved_gib": gib(torch.cuda.max_memory_reserved)}
        result["training"].append(row)
        sync()
        copied = weight_bytes_copied(backend)
        # wall-clock phases of this step, kept outside the training rows (which runs compare bitwise); rollout_s and
        # prepare_s are None on later iterations of a reused batch. Allocator figures are in the row; nvidia-smi
        # samples (CUDA context included) in training/memory/ when memory.guard is on
        result.setdefault("timings", []).append({
            "step": step, **timing, "step_s": time.time() - t_step,
            "cpu_rss_bytes": rss_bytes(), "backend": backend.name,
            "weight_bytes_copied": None if copied is None else copied - copied0})
        ids = ",".join(f"{i}{label[0]}" for i, label in zip(group["prompt_ids"], group["labels"]))
        mem = f"alloc {row['cuda_allocated_gib']:.2f}  peak res {row['peak_reserved_gib']:.2f} GiB" if on_cuda else ""
        clip = (f"it {out['iteration']}  r [{out['ratio_min']:.3f}, {out['ratio_max']:.3f}]  clip {out['clip_frac']:.3f}  "
                f"kl {out['approx_kl']:.2e}  ") if t.num_iterations > 1 else ""
        print(f"step {step:2d} (attempt {group['attempt']:3d}) {ids:12} mixed {group['mixed_groups']}  R {row['mean_reward']:.2f}  {clip}"
              f"loss {row['loss']:+.4f}  |g| {row['grad_norm']:.3e} (A {row['grad_norm_A']:.2e}, B {row['grad_norm_B']:.2e})  "
              f"|dtheta| {row['update_norm']:.2e}  |(a/r)BA| {row['effective_adapter_norm']:.3e}  {mem}")
        if step % ev.every == 0 or step == t.max_steps:
            ckpt = save(step)
    if step and ckpt != rel(runs.checkpoint_path(name, step)):  # attempt budget ran out between checkpoints
        ckpt = save(step)

    result["train_seconds"] = time.time() - t0
    result["groups_generated"], result["successful_steps"] = attempts, step  # groups_generated counts batches
    result["prompt_groups"], result["responses_generated"] = attempts * B, attempts * B * G
    result["batches_by_mixed_groups"] = {str(k): mixed_counts[k] for k in range(B + 1)}
    result["zero_signal_groups"] = len(result["zero_signal"])  # batches skipped: no group had mixed rewards
    result["coverage"] = coverage(result, [e["id"] for e in train_set], attempts, step, G)
    if step < t.max_steps:
        print(f"Attempt budget exhausted: {step} of {t.max_steps} steps after {attempts} groups")

    result["frozen_changed"] = [n for n, p in frozen.items() if digest(p) != frozen_digest[n]]
    result["params_equal_init"] = all(torch.equal(p, init[n]) for n, p in params.items())
    if t.learning_rate == 0:
        subset = [e["predictions"] for e in result["evaluations"]]
        result["control"] = {"params_equal_init": result["params_equal_init"],
                             "subset_predictions_unchanged": all(s == subset[0] for s in subset)}

    if ckpt is not None:  # evaluate the saved artifact, not just the in-memory adapter
        live = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        loaded = load_peft_weights(str(runs.ROOT / ckpt))
        assert live.keys() == loaded.keys() and all(torch.equal(live[k], loaded[k].to(live[k].device)) for k in live), ckpt
        set_peft_model_state_dict(model, loaded)
        assert all(torch.equal(p, live[k]) for k, p in get_peft_model_state_dict(model).items())
        result["final_checkpoint"] = {"path": ckpt, "step": step, "reloaded_bitwise_equal": True}

    def save_log(status):
        """training/training.json; written when training ends and again after the final evaluation."""
        mem = {"peak_allocated_gib": None, "peak_reserved_gib": None, "device_gib": None, "within_budget": None}
        if on_cuda:
            torch.cuda.synchronize()
            mem.update(peak_allocated_gib=gib(torch.cuda.max_memory_allocated), peak_reserved_gib=gib(torch.cuda.max_memory_reserved),
                       device_gib=torch.cuda.mem_get_info()[1] / 2**30)
            mem["within_budget"] = mem["peak_reserved_gib"] < cfg.run.gpu_budget_gib
        result["status"], result["memory"] = status, mem
        runs.write_json(runs.training_dir(name) / "training.json", result)
        return mem

    save_log("training complete; final evaluation pending")  # survives a failure in the evaluation below

    if ev.full_validation and ckpt is not None:
        boot = {"n": ev.bootstrap_samples, "seed": BOOTSTRAP_SEED}
        trained_full = evaluate(model, tok, val_all, generator=backend.greedy, **gen_kw)
        tm, bm = metrics(trained_full), metrics(base_full)
        diff, ci_lo, ci_hi = paired_bootstrap([p["correct"] for p in base_full], [p["correct"] for p in trained_full], **boot)
        out_path = runs.training_dir(name) / "validation.json"
        saved_base = saved_predictions(BASE_FILE)
        result["final"] = {"base": bm, "trained": tm, "accuracy_diff": diff, "bootstrap_ci95": [ci_lo, ci_hi],
                           "bootstrap": {"resamples": ev.bootstrap_samples, "seed": BOOTSTRAP_SEED, "paired_by": "id"},
                           "paired_changes": paired_changes(base_full, trained_full),
                           "vs_saved": {} if saved_base is None else {"base": paired_compare(saved_base, trained_full, **boot)}}
        scores = {"trained": label_scores(model, tok, val_all, logprobs=backend.logprobs, **score_kw)}
        with model.disable_adapter():
            scores["base"] = label_scores(model, tok, val_all, logprobs=backend.logprobs, **score_kw)
        W = {k: win_matrix(v, [e["label"] for e in val_all]) for k, v in scores.items()}
        result["final"]["auc"] = {k: w.mean().item() for k, w in W.items()}
        result["final"]["auc_vs"] = {k: paired_auc_bootstrap(W["trained"], W[k], **boot) for k in W if k != "trained"}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "model": cfg.model.name, "checkpoint": ckpt, "step": step, "split": "validation", "decoding": decoding,
            "label_scores": {"ids": [e["id"] for e in val_all], **scores}, "predictions": trained_full,
        }, indent=1))

    mem = save_log("complete")  # memory peaks now include the final evaluation

    rows = result["training"]
    print(f"\nbatches by number of mixed groups: {result['batches_by_mixed_groups']}, prompt groups {attempts * B}, "
          f"responses {attempts * B * G}, train time {result['train_seconds']:.0f} s")
    print(f"steps {step}/{t.max_steps}, batches {attempts}, zero-signal batches {result['zero_signal_groups']}, "
          f"frozen changed {len(result['frozen_changed'])}/{len(frozen)}, params equal init: {result['params_equal_init']}")
    if rows:
        print(f"grad norm range [{min(r['grad_norm'] for r in rows):.3e}, {max(r['grad_norm'] for r in rows):.3e}]  "
              f"change from init {rows[-1]['change_from_init']:.3e}  |(a/r)BA| {rows[-1]['effective_adapter_norm']:.3e}")
    if "control" in result:
        print(f"LR=0 control: {result['control']}")
    if "final" in result:
        f = result["final"]
        print(f"\nfull validation  base:    {summary(f['base'])}\nfull validation  trained: {summary(f['trained'])}")
        print(f"accuracy diff {f['accuracy_diff']:+.4f}  paired bootstrap 95% CI [{f['bootstrap_ci95'][0]:+.4f}, {f['bootstrap_ci95'][1]:+.4f}]")
        print(f"paired changes: {f['paired_changes']}")
        for ref, c in f["vs_saved"].items():
            print(f"\nvs saved {ref} predictions: {summary(c['ref_metrics'])}")
            print(f"  accuracy diff {c['accuracy_diff']:+.4f}  paired bootstrap 95% CI [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]")
            print(f"  paired changes: {c['paired_changes']}")
        for k, a in f["auc"].items():
            vs = f["auc_vs"].get(k)
            print(f"ROC AUC {k:24} {a:.4f}" + (f"   trained - this: {vs['auc_diff']:+.4f}  CI [{vs['ci95'][0]:+.4f}, {vs['ci95'][1]:+.4f}]" if vs else ""))
    if on_cuda:
        print(f"peak allocated {mem['peak_allocated_gib']:.2f} GiB, peak reserved {mem['peak_reserved_gib']:.2f} GiB, "
              f"device {mem['device_gib']:.2f} GiB; allocator peak below the {cfg.run.gpu_budget_gib} GiB budget: "
              f"{'yes' if mem['within_budget'] else 'NO'} (allocator stats exclude the CUDA context: not budget evidence; "
              + ("nvidia-smi samples in training/memory/)" if cfg.memory.guard else "memory.guard off: no nvidia-smi samples)"))
    if result["frozen_changed"] or ("control" in result and not all(result["control"].values())):
        raise SystemExit("FAILED: frozen weights changed or LR=0 control changed")
    return result
