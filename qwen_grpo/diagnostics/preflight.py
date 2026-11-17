"""Preflight: everything that can be checked before paying for GPU time, and an explicit GPU mode for the rented host.

Every check has one of four statuses, and nothing is upgraded on weaker evidence:
  PASS        checked and satisfied
  FAIL        checked and violated (training must not start)
  BLOCKED     cannot be checked here (missing local metadata, no CUDA, no per-process nvidia-smi ...)
  NOT TESTED  not attempted in this mode, or the estimate is inconclusive (inside the safety margin, unknown terms)
Estimates (memory, transfer volume and time, disk) are reported with their assumptions in `estimates` and only feed
checks named `*.estimate`; the measured GPU budget is a separate check (`gpu.measured`) that only GPU mode can PASS.

Offline mode needs no GPU and no network: Hugging Face files are read from the local cache only (huggingface_hub is
patched to local_files_only for the duration); --allow-metadata-download permits config.json and safetensors *headers*
(kilobytes), never weights. GPU mode adds the hardware checks and, unless hardware_only, one profiled pass of every
workflow phase through scripts/profile_run.py (and the DFlash path through scripts/dflash_bench.py), under the
nvidia-smi memory guard. Reports: runs/_preflight/<run>/<UTC>_<mode>.json. `train` reads them through gate().
"""
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "_preflight"
PASS, FAIL, BLOCKED, NOT_TESTED = "PASS", "FAIL", "BLOCKED", "NOT TESTED"
GiB = 2**30
# Runtime assumptions of the Step-0 budget (configs/budget/*.yaml); unmeasured until GPU mode replaces them
ASSUMED = {"pcie_gb_per_s": 12.0, "cuda_context_gib": 0.5, "allocator_overhead": 0.10, "cpu_process_gib": 2.0,
           "safety_margin": 0.10}
LONGEST_PROMPT_FALLBACK = 100  # results/budget: longest tokenized prompt over all splits with the Qwen3 tokenizer
AMPERE = (8, 0)


class Offline(RuntimeError):
    """A network access offline preflight does not allow."""


def show(path):
    """A path for messages: repository-relative when inside it."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check(cid, stage, status, detail, **data):
    assert status in (PASS, FAIL, BLOCKED, NOT_TESTED), status
    return {"id": cid, "stage": stage, "status": status, "detail": detail, **({"data": data} if data else {})}


def overall(checks):
    statuses = [c["status"] for c in checks]
    for s in (FAIL, BLOCKED, NOT_TESTED):
        if s in statuses:
            return s
    return PASS


@contextmanager
def hub(allow_network):
    """huggingface_hub restricted to the local cache unless allow_network (metadata only: the callers never fetch
    weights)."""
    import huggingface_hub as hh
    if allow_network:
        yield
        return
    saved = hh.hf_hub_download, hh.get_safetensors_metadata, hh.snapshot_download

    def local(*a, **k):
        return saved[0](*a, **{**k, "local_files_only": True})

    def refuse(*a, **k):
        raise Offline("not in the local Hugging Face cache (offline preflight; --allow-metadata-download fetches "
                      "config.json / safetensors headers only)")
    hh.hf_hub_download, hh.get_safetensors_metadata, hh.snapshot_download = local, refuse, refuse
    try:
        yield
    finally:
        hh.hf_hub_download, hh.get_safetensors_metadata, hh.snapshot_download = saved


def memory_signature(cfg):
    """Hash of the settings that determine GPU memory: a GPU preflight of one config vouches for another config only
    if this matches (e.g. the smoke run and the final run share it: same model, execution, batch shape, budget)."""
    from qwen_grpo.config import as_dict
    d = as_dict(cfg)
    keep = {"model": d["model"], "lora": d["lora"], "execution": d["execution"], "dflash": d["dflash"],
            "memory": d["memory"], "budget": d["run"]["gpu_budget_gib"],
            "training": {k: d["training"][k] for k in ("batch_size", "group_size", "max_new_tokens", "micro_batch_size")},
            "evaluation": {k: d["evaluation"][k] for k in ("max_new_tokens", "batch_size", "score_batch_size")}}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


# --- offline checks ---------------------------------------------------------------------------------------------

def check_config(path):
    from qwen_grpo import config
    try:
        cfg = config.load(path)
    except (ValueError, OSError) as e:
        return None, [check("config.valid", "config", FAIL, str(e))]
    out = [check("config.valid", "config", PASS, f"{path} loads in the current schema (strict keys, types, combinations)",
                 sha256=config.sha256(cfg), run=cfg.run.name)]
    missing = config.unimplemented(cfg)
    out.append(check("config.implemented", "config", FAIL if missing else PASS,
                     f"settings this code cannot execute: {missing}" if missing else "every setting is implemented"))
    return cfg, out


def check_run_name(cfg):
    from qwen_grpo import runs
    if runs.legacy_outputs(cfg.run.name):
        return check("run.name", "checkpoints", FAIL, f"run.name {cfg.run.name!r} belongs to a run from before runs/")
    if runs.run_dir(cfg.run.name).exists() and not cfg.run.overwrite:
        ckpts = [s for s, _ in runs.checkpoints(cfg.run.name)]
        return check("run.name", "checkpoints", FAIL,
                     f"runs/{cfg.run.name}/ exists (checkpoints at steps {ckpts}): `train` refuses it; continue it with "
                     f"`qwen-grpo resume --run {cfg.run.name}` or choose another run.name", checkpoints=ckpts)
    return check("run.name", "checkpoints", PASS, f"new run directory runs/{cfg.run.name}/"
                 + (" (run.overwrite: an existing one would be deleted)" if cfg.run.overwrite else ""))


def check_dependencies():
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    found, bad = {}, []
    for name, want in pins.items():
        try:
            have = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            have = None
        found[name] = have
        if have is None or have.split("+")[0] != want:
            bad.append(f"{name} {have} (requirements.txt: {want})")
    return check("dependencies", "environment", FAIL if bad else PASS,
                 f"version mismatches: {bad}" if bad else "every requirements.txt pin is installed", installed=found)


def check_data(cfg):
    from qwen_grpo.diagnostics import invariants
    from qwen_grpo.sentiment import SPLITS_PATH, load_splits
    if not SPLITS_PATH.exists():
        return [check("data.present", "data", FAIL, f"{SPLITS_PATH} missing")]
    splits = load_splits()
    sha = hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest()
    problems = invariants.split_problems(splits)
    out = [check("data.splits", "data", FAIL if problems else PASS,
                 "; ".join(problems) if problems else "labels valid, no duplicate text within or across train / "
                 "validation / test", sha256=sha, sizes={s: len(splits[s]) for s in ("train", "validation", "test")})]
    per_label = {s: min(sum(r["label"] == lab for r in splits[s]) for lab in ("NEGATIVE", "POSITIVE"))
                 for s in ("train", "validation")}
    ok = cfg.data.train_samples // 2 <= per_label["train"] and cfg.data.val_subset // 2 <= per_label["validation"]
    out.append(check("data.subsets", "data", PASS if ok else FAIL,
                     f"balanced subsets: train {cfg.data.train_samples} of 2 x {per_label['train']}, validation "
                     f"{cfg.data.val_subset} of 2 x {per_label['validation']}"))
    return out


def check_reward():
    from qwen_grpo.diagnostics import invariants
    problems = invariants.reward_problems()
    return check("reward.contract", "reward", FAIL if problems else PASS,
                 "; ".join(problems) if problems else "strict exact-label reward: 1 only for one canonical label "
                 "(whitespace/case-insensitive); empty, multi-label, punctuated and padded answers score 0")


def tokenizer_checks(cfg):
    """Chat template, EOS / pad ids, and the longest prompt (for the budget). Local cache only."""
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    from qwen_grpo.model import EOS_IDS
    from qwen_grpo.sentiment import build_prompts, load_splits
    if not isinstance(try_to_load_from_cache(cfg.model.name, "tokenizer.json"), str):
        return None, [check("tokenizer", "tokenizer", BLOCKED, f"{cfg.model.name}: tokenizer.json is not in the local "
                            "Hugging Face cache (offline preflight does not download it)")]
    try:
        tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left", local_files_only=True)
    except Exception as e:  # noqa: BLE001
        return None, [check("tokenizer", "tokenizer", BLOCKED, f"{cfg.model.name} tokenizer not cached: "
                            f"{type(e).__name__}")]
    out = []
    try:
        prompts = build_prompts(tok, ["a fine film .", "dull ."])
        out.append(check("tokenizer.template", "tokenizer", PASS, "non-thinking chat template: the prompt ends with "
                         "the empty <think></think> block (sentiment.build_prompts asserts it)"))
    except AssertionError as e:
        out.append(check("tokenizer.template", "tokenizer", FAIL, f"chat template is not non-thinking: {e}"))
        prompts = None
    ids = {"eos": tok.convert_tokens_to_ids("<|im_end|>"), "endoftext": tok.convert_tokens_to_ids("<|endoftext|>"),
           "pad": tok.pad_token_id}
    ok = [ids["eos"], ids["endoftext"]] == EOS_IDS and ids["pad"] == EOS_IDS[1]
    out.append(check("tokenizer.eos", "tokenizer", PASS if ok else FAIL,
                     f"<|im_end|> {ids['eos']}, <|endoftext|> {ids['endoftext']}, pad {ids['pad']} "
                     f"(model.EOS_IDS {EOS_IDS}: generation stops and pads with these)", **ids))
    longest = None
    if prompts is not None:
        splits = load_splits()
        texts = [r["text"] for s in ("train", "validation", "test") for r in splits[s]]
        longest = max(len(x) for x in tok(build_prompts(tok, texts), add_special_tokens=False)["input_ids"])
    return longest, out


def architecture(cfg, allow_network):
    from qwen_grpo.budget import architecture as A
    targets = cfg.lora.targets if isinstance(cfg.lora.targets, str) else \
        {"modules": list(cfg.lora.targets.modules), "num_layers": cfg.lora.targets.num_layers}
    with hub(allow_network):
        try:
            arch, hf_config = A.load(cfg.model.name, targets)
        except (A.MetadataError, Offline, OSError) as e:
            return None, None, targets, str(e)
    return arch, hf_config, targets, None


def check_model(cfg, arch, hf_config, error):
    if arch is None:
        return [check("model.architecture", "model", BLOCKED, f"{cfg.model.name}: {error}")]
    x = cfg.execution
    out = [check("model.architecture", "model", PASS if hf_config.model_type == "qwen3" else FAIL,
                 f"{cfg.model.name} ({arch.revision[:12]}): model_type {hf_config.model_type}, "
                 f"{arch.num_hidden_layers} layers, hidden {arch.hidden_size}, vocab {arch.vocab_size}, "
                 f"tied embeddings {arch.tie_word_embeddings}, {arch.checkpoint_bytes / GiB:.2f} GiB on disk",
                 revision=arch.revision, source=arch.source, layers=arch.num_hidden_layers)]
    adapted = sorted({int(re.match(r"model\.layers\.(\d+)\.", t.name).group(1)) for t in arch.lora_targets})
    out.append(check("lora.targets", "lora", PASS if adapted else FAIL,
                     f"{len(arch.lora_targets)} target tensors in layers {adapted}", layers=adapted))
    if x.backend != "resident" and x.resident_layers is not None and x.resident_layers > arch.num_hidden_layers:
        out.append(check("execution.residency", "execution", FAIL,
                         f"resident_layers {x.resident_layers} > {arch.num_hidden_layers} decoder layers"))
    return out


def check_dflash(cfg, arch, allow_network):
    """Drafter <-> target compatibility from the drafter's config.json (the weights are never fetched here)."""
    if not cfg.dflash.enabled:
        return [], None
    from huggingface_hub import hf_hub_download
    with hub(allow_network):
        try:
            raw = json.loads(Path(hf_hub_download(cfg.dflash.draft_model, "config.json")).read_text())
        except Exception as e:  # noqa: BLE001
            return [check("dflash.compat", "dflash", BLOCKED, f"{cfg.dflash.draft_model} config.json: {e}")], None
    if arch is None:
        return [check("dflash.compat", "dflash", BLOCKED, "target architecture unavailable")], raw
    ids = raw.get("dflash_config", {}).get("target_layer_ids", [])
    problems = [p for ok, p in (
        (raw.get("hidden_size") == arch.hidden_size, f"drafter hidden_size {raw.get('hidden_size')} != target "
                                                     f"{arch.hidden_size} (it uses the target's embedding and head)"),
        (raw.get("vocab_size") == arch.vocab_size, f"vocab {raw.get('vocab_size')} != {arch.vocab_size}"),
        (bool(ids) and max(ids) < arch.num_hidden_layers, f"feature layers {ids} outside {arch.num_hidden_layers}"),
        ("block_size" in raw and "mask_token_id" in raw.get("dflash_config", {}), "block_size / mask_token_id missing"),
    ) if not ok]
    return [check("dflash.compat", "dflash", FAIL if problems else PASS,
                  "; ".join(problems) if problems else f"drafter for {cfg.model.name}: feature layers {ids}, block "
                  f"size {raw['block_size']}, hidden {raw['hidden_size']}", block_size=raw.get("block_size"),
                  feature_layers=ids)], raw


def budget_config(cfg, arch, prompt_tokens, host_ram_gib, pcie, loading_copy):
    """The Step-0 calculator's schema (qwen_grpo/budget/calculator.py) filled from a training config. The executed
    schedule is micro-batch-major (training.py: every micro-batch streams every layer)."""
    x, t, ev = cfg.execution, cfg.training, cfg.evaluation
    l2l = x.backend != "resident"
    n = t.batch_size * t.group_size
    targets = cfg.lora.targets if isinstance(cfg.lora.targets, str) else \
        {"modules": list(cfg.lora.targets.modules), "num_layers": cfg.lora.targets.num_layers}
    return {
        "run_name": f"preflight_{cfg.run.name}",
        "model": {"id": cfg.model.name, "weight_dtype": cfg.model.dtype},
        "hardware": {"gpu_vram_gib": cfg.run.gpu_budget_gib, "cpu_ram_gib": host_ram_gib,
                     "cpu_ram_source": "preflight", "pcie_gb_per_s": pcie, "pcie_source": "preflight",
                     "copy_compute_overlap": x.backend == "l2l_async"},
        "runtime": {k: ASSUMED[k] for k in ("cuda_context_gib", "allocator_overhead", "cpu_process_gib",
                                            "safety_margin")},
        "grpo": {"prompt_batch_size": t.batch_size, "group_size": t.group_size, "rollout_batch_size": n,
                 "micro_batch_size": t.micro_batch_size or n, "prompt_tokens": prompt_tokens,
                 "generation_tokens": t.max_new_tokens, "num_iterations": t.num_iterations, "reference_logprobs": False},
        "lora": {"rank": cfg.lora.rank, "targets": targets, "dtype": "float32", "optimizer": "adamw"},
        "streaming": {"resident_decoder_layers": x.resident_layers if l2l else arch.num_hidden_layers,
                      "gpu_layer_buffers": x.weight_buffers if l2l else 2,  # resident: unused (no streamed layer)
                      "weight_placement": "pageable_cpu" if l2l and x.weight_storage == "pageable" else "pinned_cpu",
                      "pinned_allocation": "per_layer", "non_decoder_placement": "gpu", "loading_copy": loading_copy,
                      "schedule": "microbatch_major", "activation_checkpoints": "gpu", "kv_cache": x.kv_cache,
                      "logits": "response_only"},
        "evaluation": {"examples": cfg.data.val_subset, "batch_size": ev.batch_size,
                       "score_batch_size": ev.score_batch_size, "generation_tokens": ev.max_new_tokens,
                       "during_training": True},
        "merge": {"strategy": "tensorwise", "device": "cpu"},
        "dflash": {"enabled": False, "estimate": cfg.dflash.enabled,
                   "draft_model_id": cfg.dflash.draft_model or "none"},
    }


def resource_estimates(cfg, arch, prompt_tokens, host_ram_gib, pcie, draft_bytes):
    from qwen_grpo.budget import calculator as C
    reps = {}
    for copy in ("tensor", "model"):  # the loader's pageable copy: one tensor alive (per-layer adoption) .. whole model
        bcfg = C.validate(budget_config(cfg, arch, prompt_tokens, host_ram_gib, pcie, copy))
        reps[copy] = C.build_report(bcfg, arch, draft_bytes)
    rep = reps["model"]
    phases = {p["phase"]: {"gpu_device_bytes": p["gpu_peak"]["gpu_device"], "gpu_allocated_bytes":
                           p["gpu_peak"]["gpu_allocated"], "gpu_verdict": p["gpu_verdict"],
                           "cpu_peak_bytes": p["cpu_peak_bytes"], "cpu_verdict": p["cpu_verdict"]}
              for p in rep["phases"]}
    cpu_low = max(p["cpu_peak_bytes"] for p in reps["tensor"]["phases"] if p["phase"] in C.OVERALL_PHASES)
    cpu_high = max(p["cpu_peak_bytes"] for p in rep["phases"] if p["phase"] in C.OVERALL_PHASES)
    gpu = [phases[p] for p in phases if p in C.OVERALL_PHASES]
    return {"assumptions": {**ASSUMED, "pcie_gb_per_s": pcie, "host_ram_gib": host_ram_gib,
                            "prompt_tokens": prompt_tokens, "schedule": "microbatch_major",
                            "loading_copy": "range: one tensor (per-layer adoption) .. whole checkpoint"},
            "phases": phases,
            "gpu_peak_device_bytes": max(p["gpu_device_bytes"] for p in gpu),
            "gpu_verdict": C.combine(p["gpu_verdict"] for p in gpu),
            "cpu_peak_bytes_range": [cpu_low, cpu_high],
            "pinned_bytes": rep["residency"]["cpu_weight_bytes"],
            "transfers_per_grpo_step": rep["transfers_per_grpo_step"],
            "transfers": rep["transfers"],
            "uncertainties": rep["uncertainties"],
            "residency_recommendation": rep["residency"]["recommendation"]}


def check_estimates(cfg, est, host_ram_gib):
    budget = cfg.run.gpu_budget_gib * GiB
    peak, verdict = est["gpu_peak_device_bytes"], est["gpu_verdict"]
    status = {"PASS": PASS, "FAIL": FAIL}.get(verdict, NOT_TESTED)
    out = [check("gpu.estimate", "memory", status,
                 f"estimated device peak {peak / GiB:.2f} GiB vs budget {cfg.run.gpu_budget_gib} GiB (calculator "
                 f"verdict {verdict}; 'UNKNOWN' = inside the {ASSUMED['safety_margin']:.0%} margin or an unmodelled "
                 "term). An estimate is never evidence of the budget: see gpu.measured", peak_bytes=peak,
                 budget_bytes=budget)]
    lo, hi = est["cpu_peak_bytes_range"]
    limit = host_ram_gib * GiB * (1 - ASSUMED["safety_margin"])
    cpu_status = PASS if hi <= limit else FAIL if lo > host_ram_gib * GiB else NOT_TESTED
    out.append(check("cpu.estimate", "memory", cpu_status,
                     f"estimated host RAM peak {lo / GiB:.1f}..{hi / GiB:.1f} GiB (pinned store "
                     f"{est['pinned_bytes'] / GiB:.2f} GiB) vs {host_ram_gib:.1f} GiB host RAM"
                     + ("" if cpu_status != NOT_TESTED else "; depends on the loader's transient copy: measure it "
                        "with preflight --gpu"), range_bytes=[lo, hi]))
    return out


def disk_estimates(cfg, arch, draft_bytes):
    from qwen_grpo.budget import calculator as C
    lora = C.lora_numel(arch.lora_targets, cfg.lora.rank) * 4
    checkpoints = cfg.training.max_steps // cfg.evaluation.every + 2  # step0, every N, final
    n = cfg.training.batch_size * cfg.training.group_size
    groups = cfg.training.max_steps // cfg.training.num_iterations + 50  # zero-signal groups are also written
    per_group = n * (LONGEST_PROMPT_FALLBACK + cfg.training.max_new_tokens) * 8 * 4  # ids, masks, logp (int64 bound)
    out = {"model_cache_bytes": arch.checkpoint_bytes + (draft_bytes or 0),
           "checkpoints_bytes": checkpoints * lora * 4,  # adapter + AdamW (2x) + gradients/log margin
           "streaming_bytes": groups * per_group if cfg.streaming.enabled else 0,
           "merged_checkpoint_bytes": arch.checkpoint_bytes}
    out["run_total_bytes"] = out["checkpoints_bytes"] + out["streaming_bytes"] + out["merged_checkpoint_bytes"]
    return out


def check_disk(est):
    from huggingface_hub import constants
    cache = Path(constants.HF_HUB_CACHE)
    cache_probe = cache if cache.exists() else Path.home()
    free_root, free_cache = shutil.disk_usage(ROOT).free, shutil.disk_usage(cache_probe).free
    same = os.stat(ROOT).st_dev == os.stat(cache_probe).st_dev
    need_root = est["run_total_bytes"] + (est["model_cache_bytes"] if same else 0)
    ok = need_root <= free_root and (same or est["model_cache_bytes"] <= free_cache)
    return check("disk", "environment", PASS if ok else FAIL,
                 f"needs ~{need_root / GiB:.1f} GiB on the repository's filesystem ({free_root / GiB:.1f} GiB free)"
                 + ("" if same else f" and {est['model_cache_bytes'] / GiB:.1f} GiB in the HF cache "
                    f"({free_cache / GiB:.1f} GiB free)") + " (model cache, checkpoints, streaming records, one merged "
                 "checkpoint; estimates)", free_root_bytes=free_root, free_cache_bytes=free_cache)


def host_ram():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / GiB
    except OSError:
        pass
    return None


# --- GPU mode -----------------------------------------------------------------------------------------------------

def smi(args):
    try:
        r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def gpu_hardware(cfg, est):
    """Hardware checks on the GPU host: architecture, memory, ownership, per-process accounting, pinned memory and
    the measured H2D bandwidth. Allocates the planned pinned bytes once (then frees them)."""
    import torch

    out, measured = [], {}
    if not torch.cuda.is_available():
        return [check("gpu.cuda", "gpu", BLOCKED, "torch.cuda.is_available() is False")], measured
    dev = torch.device("cuda")
    p = torch.cuda.get_device_properties(dev)
    cap = (p.major, p.minor)
    out.append(check("gpu.cuda", "gpu", PASS, f"{p.name}, compute capability {p.major}.{p.minor}, torch CUDA "
                     f"{torch.version.cuda}, arch list {torch.cuda.get_arch_list()}", name=p.name))
    out.append(check("gpu.architecture", "gpu", PASS if cap >= AMPERE else FAIL,
                     f"compute capability {p.major}.{p.minor}: the assignment requires Ampere (8.0) or newer"))
    arches = torch.cuda.get_arch_list()
    if not kernels_for(cap, arches):
        out.append(check("gpu.kernels", "gpu", FAIL, f"this torch build ({torch.__version__}) has no kernels for "
                         f"sm_{p.major}{p.minor} (it has {arches}): nothing can run on this GPU"))
        return out, measured
    out.append(check("gpu.kernels", "gpu", PASS, f"torch {torch.__version__} has kernels for sm_{p.major}{p.minor}"))
    try:
        _gpu_runtime(cfg, est, dev, p, out, measured)
    except Exception as e:  # noqa: BLE001 - a broken GPU runtime is a FAIL of preflight, never a crash
        out.append(check("gpu.runtime", "gpu", FAIL, f"{type(e).__name__}: {(str(e).splitlines() or [''])[0]}"))
    return out, measured


def kernels_for(cap, arches):
    """Whether a torch build with `arches` (e.g. ['sm_80', 'sm_86', 'compute_90']) can run on compute capability `cap`:
    an exact sm_XY binary, or PTX (compute_XY) for an older or equal capability that the driver can JIT-compile."""
    code = cap[0] * 10 + cap[1]
    return f"sm_{code}" in arches or any(a.startswith("compute_") and int(a.split("_")[1]) <= code for a in arches)


def _gpu_runtime(cfg, est, dev, p, out, measured):
    """The checks that execute on the GPU (context, accounting, exclusivity, pinned memory, H2D bandwidth), appended to
    `out` as they complete, so the checks before a failure are kept."""
    import torch

    from qwen_grpo.execution.profiler import h2d_bandwidth, is_wsl
    total = p.total_memory
    budget = cfg.run.gpu_budget_gib * GiB
    out.append(check("gpu.capacity", "gpu", PASS if total >= budget else FAIL,
                     f"{total / GiB:.2f} GiB on the device vs a {cfg.run.gpu_budget_gib} GiB budget"
                     + (" (larger card: the budget is enforced by memory.guard, not by the hardware)"
                        if total > budget * 1.05 else ""), total_bytes=total))
    torch.zeros(1, device=dev)  # create the CUDA context, then look at it from nvidia-smi
    torch.cuda.synchronize()
    apps = smi(["--query-compute-apps=pid,used_memory"])
    driver = smi(["--query-gpu=driver_version,memory.used,memory.total"])
    mine = [ln for ln in (apps or "").splitlines() if ln.split(",")[0].strip() == str(os.getpid())]
    numeric = bool(mine) and mine[0].split(",")[1].strip().isdigit()
    others = [ln for ln in (apps or "").splitlines() if ln.split(",")[0].strip() != str(os.getpid())]
    out.append(check("gpu.accounting", "gpu", PASS if numeric and not is_wsl() else BLOCKED,
                     "nvidia-smi reports this process's memory: budget evidence can be assignment-grade" if numeric
                     and not is_wsl() else f"no numeric per-process nvidia-smi figure (WSL2: {is_wsl()}, raw {mine!r}): "
                     "the budget can only be guarded device-wide, never shown as PASS", raw_apps=apps, raw_gpu=driver))
    out.append(check("gpu.exclusive", "gpu", PASS if not others else FAIL,
                     "no other compute process on the GPU" if not others else f"other GPU processes {others}: "
                     "device-wide figures include them and the budget cannot be attributed"))
    if numeric:
        ctx = int(mine[0].split(",")[1]) * 2**20 - torch.cuda.memory_reserved(dev)
        measured["cuda_context_bytes"] = ctx
    try:
        pinned = torch.empty(est["pinned_bytes"], dtype=torch.uint8, pin_memory=True)
        ok = pinned.is_pinned()
        del pinned
        out.append(check("cpu.pinned", "memory", PASS if ok else FAIL,
                         f"page-locked {est['pinned_bytes'] / GiB:.2f} GiB (the planned weight store) and released it"))
    except RuntimeError as e:
        out.append(check("cpu.pinned", "memory", FAIL, f"cannot page-lock {est['pinned_bytes'] / GiB:.2f} GiB: {e}"))
    bw = h2d_bandwidth(256 * 2**20)
    measured["h2d_bytes_per_s"] = bw
    out.append(check("pcie.bandwidth", "gpu", PASS, f"pinned H2D {bw / 1e9:.1f} GB/s (median of 10 x 256 MiB)",
                     bytes_per_s=bw))


def gpu_model_steps(config_path, cfg, stamp):
    """One profiled pass of every workflow phase on the real model (scripts/profile_run.py) and, with DFlash, one
    DFlash rollout check (scripts/dflash_bench.py), both under the nvidia-smi guard at the config's budget."""
    out_dir = OUT / cfg.run.name / f"{stamp}_gpu"
    checks = []
    budget = str(cfg.run.gpu_budget_gib)
    prof = [sys.executable, "scripts/profile_run.py", "--config", str(config_path), "--budget-gib", budget, "--steps",
            "1", "--eval-n", "8", "--out", "profile", "--out-dir", str(out_dir)]
    if cfg.execution.backend == "resident":
        return [check("gpu.measured", "memory", NOT_TESTED, "profile_run.py profiles L2L backends only; a resident "
                      "config is measured by its training run (memory.guard)")]
    r = subprocess.run(prof, cwd=ROOT)
    path = out_dir / "profile.json"
    if not path.exists():
        return [check("gpu.measured", "memory", FAIL, f"profile_run.py exited {r.returncode} without a result")]
    rep = json.loads(path.read_text())
    status = {"PASS": PASS, "FAIL": FAIL}.get(rep["verdict"], BLOCKED)
    peak = rep["profile"]["peak"]
    checks.append(check("gpu.measured", "memory", status,
                        f"setup, rollout, old log-probs, forward, backward, optimizer, merge and evaluation under the "
                        f"guard: verdict {rep['verdict']} (evidence {rep['profile']['evidence']}); peak nvidia-smi "
                        f"process {(peak.get('process_used') or 0) / GiB:.2f} GiB, device {(peak.get('device_used') or 0) / GiB:.2f} "
                        f"GiB, reserved {(peak.get('reserved') or 0) / GiB:.2f} GiB", result=show(path)))
    if cfg.dflash.enabled:
        dfl = [sys.executable, "scripts/dflash_bench.py", "--config", str(config_path), "--budget-gib", budget,
               "--max-new", str(cfg.training.max_new_tokens), "--b-scales", "0.0", "--out", "dflash",
               "--out-dir", str(out_dir)]
        subprocess.run(dfl, cwd=ROOT)
        p = out_dir / "dflash.json"
        if not p.exists():
            checks.append(check("dflash.measured", "dflash", FAIL, "dflash_bench.py produced no result"))
        else:
            d = json.loads(p.read_text())
            ok = d["all_greedy_equal"] and d["budget_verdict"] != "FAIL"
            checks.append(check("dflash.measured", "dflash", PASS if ok and d["budget_verdict"] == "PASS" else
                                FAIL if not ok else BLOCKED,
                                f"greedy DFlash == plain decoding: {d['all_greedy_equal']}; budget "
                                f"{d['budget_verdict']}", result=show(p)))
    return checks


# --- driver --------------------------------------------------------------------------------------------------------

def run(config_path, gpu=False, allow_network=False, pcie_gbps=None, host_ram_gib=None, model_steps=True):
    from qwen_grpo.trainer import git_state
    config_path = Path(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rep = {"schema": 1, "created_utc": stamp, "mode": "gpu" if gpu else "offline",
           "config": show(config_path), **git_state(), "checks": [], "estimates": {}}
    cfg, checks = check_config(config_path)
    rep["checks"] += checks
    if cfg is None:
        rep["overall"] = FAIL
        return rep
    rep.update(run=cfg.run.name, config_sha256=checks[0]["data"]["sha256"], memory_signature=memory_signature(cfg))
    rep["checks"] += [check_run_name(cfg), check_dependencies(), *check_data(cfg), check_reward()]
    longest, tok_checks = tokenizer_checks(cfg)
    rep["checks"] += tok_checks
    arch, hf_config, _, error = architecture(cfg, allow_network)
    rep["checks"] += check_model(cfg, arch, hf_config, error)
    dflash_checks, _ = check_dflash(cfg, arch, allow_network)
    rep["checks"] += dflash_checks
    ram = host_ram_gib or host_ram()
    pcie = pcie_gbps or ASSUMED["pcie_gb_per_s"]
    prompt_tokens = longest or LONGEST_PROMPT_FALLBACK
    rep["estimates"]["prompt_tokens"] = {"value": prompt_tokens, "source": "tokenized splits" if longest else
                                         "fallback (tokenizer not cached): results/budget value"}
    draft_bytes = None
    if cfg.dflash.enabled and arch is not None:
        from qwen_grpo.budget import architecture as A
        with hub(allow_network):
            try:
                tensors, _, _ = A.checkpoint_tensors(cfg.dflash.draft_model, None)
                draft_bytes = sum(t.stored_bytes for t in tensors.values())
            except (A.MetadataError, Offline) as e:
                rep["checks"].append(check("dflash.size", "dflash", BLOCKED, f"drafter size unknown: {e}"))
    if arch is not None:
        est = resource_estimates(cfg, arch, prompt_tokens, ram, pcie, draft_bytes)
        rep["estimates"]["resources"] = est
        rep["estimates"]["disk"] = disk_estimates(cfg, arch, draft_bytes)
        rep["checks"] += check_estimates(cfg, est, ram)
        rep["checks"].append(check_disk(rep["estimates"]["disk"]))
        if cfg.dflash.enabled:
            dph = est["phases"].get("dflash (optional estimate)")
            rep["checks"].append(check("dflash.estimate", "dflash",
                                       FAIL if dph and dph["gpu_verdict"] == "FAIL" else NOT_TESTED,
                                       "drafter weights sized" + (f" ({draft_bytes / GiB:.2f} GiB)" if draft_bytes else "")
                                       + "; its KV cache and feature buffers are not modelled by the calculator: "
                                       "only GPU mode can establish DFlash memory"))
    else:
        rep["checks"] += [check(c, "memory", BLOCKED, "no architecture metadata: nothing to estimate")
                          for c in ("gpu.estimate", "cpu.estimate", "disk")]
    if gpu:
        if arch is None:
            rep["checks"].append(check("gpu.hardware", "gpu", BLOCKED, "needs the architecture metadata first"))
        else:
            hw, measured = gpu_hardware(cfg, rep["estimates"]["resources"])
            rep["checks"] += hw
            rep["measured"] = measured
            if "h2d_bytes_per_s" in measured:  # the time estimate again, at the measured bandwidth
                rep["estimates"]["resources_measured_pcie"] = resource_estimates(
                    cfg, arch, prompt_tokens, ram, measured["h2d_bytes_per_s"] / 1e9, draft_bytes)["transfers_per_grpo_step"]
            fatal = [c for c in hw if c["status"] == FAIL]
            if model_steps and not fatal and all(c["status"] == PASS for c in rep["checks"]
                                                     if c["id"] in ("config.valid", "config.implemented")):
                rep["checks"] += gpu_model_steps(config_path, cfg, stamp)
            else:
                rep["checks"].append(check("gpu.measured", "memory", NOT_TESTED,
                                           "model steps not run" + (f": hardware FAIL {[c['id'] for c in fatal]}"
                                                                   if fatal else " (--hardware-only)")))
    else:
        rep["checks"].append(check("gpu.measured", "memory", NOT_TESTED,
                                   "offline mode: the GPU budget is only established by `preflight --gpu` "
                                   "(nvidia-smi, CUDA context included) on the target GPU"))
    rep["overall"] = overall(rep["checks"])
    rep["overall_offline"] = overall([c for c in rep["checks"] if c["id"] != "gpu.measured"
                                      and c["stage"] != "gpu" and not c["id"].endswith(".measured")])
    return rep



def save(rep):
    out = OUT / (rep.get("run") or "_invalid")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rep['created_utc']}_{rep['mode']}.json"
    path.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    return path


def render(rep):
    lines = [f"preflight ({rep['mode']}) of {rep['config']}: {rep['overall']}"
             + (f"  [offline checks: {rep['overall_offline']}]" if "overall_offline" in rep else "")]
    for c in rep["checks"]:
        lines.append(f"  {c['status']:10} {c['id']:20} {c['detail']}")
    res = rep.get("estimates", {}).get("resources")
    if res:
        t = res["transfers_per_grpo_step"]
        lines.append(f"  estimates: GPU device peak {res['gpu_peak_device_bytes'] / GiB:.2f} GiB, host RAM "
                     f"{res['cpu_peak_bytes_range'][0] / GiB:.1f}..{res['cpu_peak_bytes_range'][1] / GiB:.1f} GiB, "
                     f"pinned {res['pinned_bytes'] / GiB:.2f} GiB, H2D per GRPO step {t['bytes'] / 1e9:.1f} GB "
                     f"(~{t['transfer_s']:.1f} s at {res['assumptions']['pcie_gb_per_s']:.1f} GB/s, assumed "
                     "unless measured)")
    return "\n".join(lines)


def latest(run, mode=None):
    d = OUT / run
    files = sorted(d.glob(f"*_{mode}.json" if mode else "*.json")) if d.is_dir() else []
    return (files[-1], json.loads(files[-1].read_text())) if files else (None, None)


def gate(config_path, cfg, skip=False):
    """Whether `train` may start: the offline checks must not FAIL (BLOCKED / NOT TESTED are allowed for unguarded
    configs: they cannot be checked here), and a config with memory.guard (the GPU-budget runs) also needs a GPU
    preflight record whose overall status is PASS for a config with the same memory signature. skip=True starts
    anyway and says so (the run's log keeps the message)."""
    rep = run(config_path)
    path = save(rep)
    fails = [c["id"] for c in rep["checks"] if c["status"] == FAIL]
    if fails:
        return {"ok": False, "message": f"preflight FAIL {fails} ({show(path)}); refusing to train"}
    if not cfg.memory.guard:
        return {"ok": True, "message": f"preflight (offline): no FAIL ({show(path)})"}
    sig = memory_signature(cfg)
    records = sorted(p for p in OUT.glob("*/*_gpu.json"))
    passing = [p for p in records if (r := json.loads(p.read_text())).get("memory_signature") == sig
               and r.get("overall") == PASS]
    if passing:
        return {"ok": True, "message": f"GPU preflight PASS for this memory signature: {show(passing[-1])}"}
    if skip:
        return {"ok": True, "message": "WARNING --skip-preflight: no passing GPU preflight for this config's memory "
                                       "signature; the 8 GB budget is enforced only by memory.guard during the run"}
    return {"ok": False, "message": "this config enforces a GPU budget (memory.guard) but no GPU preflight with the "
                                    f"same memory signature passed: run `qwen-grpo preflight --config {config_path} "
                                    "--gpu` on the target GPU first (or --skip-preflight to accept the risk)"}
