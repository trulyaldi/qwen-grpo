"""Ticket 02 lab cases: the pinned CPU weight store (qwen_grpo/execution/weight_store.py). Registered in lab.CASES.

store_equivalence  correctness: stored tensors and the unchanged resident code on store-backed parameters, against the
                   Ticket 00 fixtures, tolerance 0; plus verify() and an unchanged PyTorch CUDA allocator.
store_depth        memory (H07): PyTorch-allocated GPU memory with every layer copied through a fixed buffer pool vs
                   the same decoder layers resident, over increasing depth at a fixed layer shape.
store_build        memory: the real model of the config adopted into the store; every stored tensor against the
                   checkpoint's safetensors (an independent reference); RSS and MemAvailable traced per layer.
pinned_transfer    performance: host-to-device copy of one slab from pageable vs page-locked memory, interleaved trials,
                   identical copy settings; plus pin_memory=True rounding, RLIMIT_MEMLOCK, an optional capacity probe.
"""
import gc
import resource
from pathlib import Path

import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import lab
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.memory_plan import plan
from qwen_grpo.execution.weight_store import PinningError, WeightStore, decoder_layers, digest, gpu_buffers
from qwen_grpo.model import PRESETS, lora_config, target_modules

MiB = 2 ** 20
DEPTHS = (2, 4, 8, 16, 32)
DEPTH_SHAPE = dict(hidden_size=256, intermediate_size=768, num_attention_heads=4, num_key_value_heads=2, head_dim=64)


def meminfo():
    info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    kib = lambda d, k: int(d[k].split()[0]) * 1024
    return {"mem_available": kib(info, "MemAvailable"), "rss": kib(status, "VmRSS"), "peak_rss": kib(status, "VmHWM")}


def cuda_state():
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {"allocated": torch.cuda.memory_allocated(), "device_used": total - free}


def delta(before, after):
    return None if before is None else {k: after[k] - before[k] for k in before}


# --- store_equivalence ---------------------------------------------------------------------------------------------------

def store_equivalence(cfg, result, perturbation=None, **_):
    for shape in F.SHAPES:
        lab.check_fixture(shape, cfg)
    tol = lab.tolerances_for("store_equivalence", cfg)
    result.update(comparisons=[], reference_integrity=[], measurements={})
    for shape in F.SHAPES:
        ref_out, saved = lab.execute("resident", shape)          # fresh resident reference, checked against the saved
        result["reference_integrity"] += [f"{shape}:{m}" for m in F.mismatches(ref_out, saved["reference"])]
        model, _ = F.load(shape)
        before = cuda_state()
        store = WeightStore.build(model, cfg.execution.weight_storage)
        try:
            built = delta(before, cuda_state())
            with F.isolated():
                got_out = F.reference(model, saved)               # the unchanged resident code, store-backed weights
            problems = store.verify()                             # after execution too: nothing was moved or changed
            ref, got = lab.flatten(saved["reference"]), lab.flatten(got_out)
            for s in store.slabs:
                for e in s.entries:
                    key = f"weights.{s.index}.{e.name}"
                    ref[key] = saved["state"][f"base_model.model.model.layers.{s.index}.{e.name}"]
                    got[key] = store.views(s.index)[e.name]
            if perturbation and shape == next(iter(F.SHAPES)):
                result["perturbation"] = lab.perturb(got, perturbation)
            result["comparisons"] += [{"fixture": shape, **row} for row in lab.compare_all(ref, got, tol)]
            result["measurements"][shape] = {"store": store.summary(), "verify_problems": problems,
                                             "build_delta": built,
                                             "pinned_views": sum(e.param.is_pinned() for s in store.slabs for e in s.entries)}
            result["errors"] += [f"{shape}: {p}" for p in problems]
            if built and built["allocated"] != 0:
                result["errors"].append(f"{shape}: building the store changed PyTorch-allocated CUDA memory by "
                                        f"{built['allocated']} B")
        finally:
            store.close()


# --- store_depth (H07) -----------------------------------------------------------------------------------------------------

def depth_model(layers, seed=0):
    """bf16 Qwen3 of DEPTH_SHAPE with `layers` decoder layers and fp32 LoRA on the last two (as in training)."""
    with F.isolated(seed):
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, tie_word_embeddings=True, num_hidden_layers=layers,
                                            **DEPTH_SHAPE)).to(torch.bfloat16).eval()
        return get_peft_model(base, lora_config(target_modules(base, *PRESETS["last2-attention"]))).eval()


def store_depth(cfg, result, depths=DEPTHS, **_):
    x = cfg.execution
    if not torch.cuda.is_available():
        raise lab.Blocked("store_depth measures GPU memory; no CUDA device is available")
    rows = []
    gc.collect()
    torch.cuda.empty_cache()
    start = torch.cuda.memory_allocated()
    for depth in depths:
        model = depth_model(depth)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        store = WeightStore.build(model, x.weight_storage)
        build_delta = torch.cuda.memory_allocated() - base
        buffers = gpu_buffers(x.weight_buffers, store.max_slab_bytes)
        problems = []
        for i in range(depth):                                                   # every layer through the fixed pool
            problems += store.check_views(i, store.copy_to(i, buffers[i % x.weight_buffers]))
        torch.cuda.synchronize()
        streamed = torch.cuda.max_memory_allocated() - base
        p = plan(model, x)
        del buffers
        store.close()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        decoder_layers(model).to("cuda")                                        # the same layers, resident
        torch.cuda.synchronize()
        resident = torch.cuda.max_memory_allocated() - base
        rows.append({"depth": depth, "max_slab_bytes": p["max_slab_bytes"], "store_bytes": sum(p["slab_bytes"]),
                     "planned_buffer_bytes": p["items"]["gpu.weight_buffers"]["bytes"],
                     "streamed_peak_allocated": streamed, "resident_peak_allocated": resident,
                     "build_allocated_delta": build_delta, "view_problems": len(problems)})
        result["errors"] += [f"depth {depth}: {q}" for q in problems[:5]]
        if build_delta:
            result["errors"].append(f"depth {depth}: building the store allocated {build_delta} B of CUDA memory")
        del model, store, p  # the store holds the Parameter objects: keeping it would carry this depth's GPU layers over
        gc.collect()
        torch.cuda.empty_cache()
        if leftover := torch.cuda.memory_allocated() - start:
            result["errors"].append(f"depth {depth}: {leftover} B still allocated after cleanup; later depths would be biased")
    result["measurements"] = {"depths": rows, "layer_shape": DEPTH_SHAPE, "dtype": "bfloat16",
                              "weight_buffers": x.weight_buffers, "storage": x.weight_storage,
                              "reading": "torch.cuda.max_memory_allocated() minus the allocation before the phase "
                                         "(PyTorch allocator only; diagnostic grade)"}
    result["memory"] = lab.memory("cuda")
    result["required_measurements"] = ["gpu_memory"]


# --- store_build (real model) ------------------------------------------------------------------------------------------------

def checkpoint_digests(name):
    """{tensor name: sha256} straight from the cached safetensors shards, one tensor in memory at a time."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    out = {}
    for shard in sorted(Path(snapshot_download(name, local_files_only=True)).glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if ".layers." in k:
                    out[k] = digest(f.get_tensor(k))
    return out


def store_build(cfg, result, margin_mib=768, floor_mib=512, **_):
    from qwen_grpo.config import DTYPES
    from qwen_grpo.model import lora_model, skeleton

    x, lo = cfg.execution, cfg.lora
    sk = skeleton(cfg.model.name).requires_grad_(False)
    elem = torch.empty(0, dtype=DTYPES[cfg.model.dtype]).element_size()
    model_bytes = sum(p.numel() for p in sk.parameters()) * elem
    layer_bytes = [sum(p.numel() for _, p in layer.named_parameters()) * elem for layer in decoder_layers(sk)]
    start = meminfo()
    # Projection if adopting a layer releases its loaded storage: model + one slab + margin. Whether it does is what
    # the per-layer trace measures; the floor guard stops the build (store closed) if it does not and RAM runs out.
    projected = model_bytes + max(layer_bytes) + margin_mib * MiB
    result["measurements"] = {"start": start, "model_bytes": model_bytes, "store_bytes_unaligned": sum(layer_bytes),
                              "projected_peak_bytes": projected, "margin_mib": margin_mib, "floor_mib": floor_mib,
                              "projection": "model + largest slab + margin (assumes adopted layers release storage)"}
    if projected > start["mem_available"]:
        raise lab.Blocked(f"projected host peak {projected / MiB:.0f} MiB (model + largest slab + margin) exceeds "
                          f"MemAvailable {start['mem_available'] / MiB:.0f} MiB; refusing to start")
    del sk
    torch.manual_seed(cfg.training.seed)
    model = lora_model(DTYPES[cfg.model.dtype], "cpu", lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
    loaded = meminfo()
    before = cuda_state()
    trace = []

    def guard(i):
        trace.append({"layer": i, **meminfo()})
        if trace[-1]["mem_available"] < floor_mib * MiB:
            raise lab.SetupError(f"MemAvailable fell to {trace[-1]['mem_available'] / MiB:.0f} MiB after layer {i} "
                                 f"(floor {floor_mib} MiB): adopted layers are not releasing their storage; build stopped")

    try:
        store = WeightStore.build(model, x.weight_storage, on_layer=guard)
    finally:
        result["measurements"].update(loaded=loaded, per_layer=trace)
    try:
        built = delta(before, cuda_state())
        problems = store.verify()
        reference = checkpoint_digests(cfg.model.name)
        mismatched, missing = [], []
        for s in store.slabs:
            for e in s.entries:
                key = f"model.layers.{s.index}.{e.name.replace('.base_layer', '')}"
                if key not in reference:
                    missing.append(key)
                elif reference[key] != e.sha256:
                    mismatched.append(key)
        stored = {f"model.layers.{s.index}.{e.name.replace('.base_layer', '')}" for s in store.slabs for e in s.entries}
        result["measurements"].update(
            end=meminfo(), store=store.summary(),
            plan=plan(model, x), build_delta=built, verify_problems=problems,
            checkpoint_tensors_compared=len(stored), checkpoint_mismatches=mismatched, checkpoint_missing=missing,
            checkpoint_layer_tensors_not_stored=sorted(set(reference) - stored),
            pinned_views=sum(e.param.is_pinned() for s in store.slabs for e in s.entries),
            stored_tensors=sum(len(s.entries) for s in store.slabs),
            tied=model.get_base_model().lm_head.weight is model.get_base_model().model.embed_tokens.weight)
        result["errors"] += problems[:20] + [f"checkpoint mismatch: {k}" for k in mismatched[:20]]
        result["errors"] += [f"not in the checkpoint: {k}" for k in missing[:20]]
        result["errors"] += [f"checkpoint layer tensor not stored: {k}" for k in sorted(set(reference) - stored)[:20]]
        if built and built["allocated"] != 0:
            result["errors"].append(f"building the store changed PyTorch-allocated CUDA memory by {built['allocated']} B")
    finally:
        store.close()
    result["memory"] = lab.memory("cuda" if torch.cuda.is_available() else "cpu")


# --- pinned_transfer ---------------------------------------------------------------------------------------------------------

def registered(nbytes):
    """A page-locked allocation of exactly nbytes (as the store makes them) and a release function."""
    raw = torch.empty(nbytes, dtype=torch.uint8)
    rc = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), nbytes, 0)
    if rc != torch.cuda.cudart().cudaError.success:
        raise PinningError(f"cudaHostRegister of {nbytes} bytes failed: {rc}")
    return raw, lambda: torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())


def pin_memory_rounding(sizes):
    """Bytes the CachingHostAllocator holds for each live pin_memory=True request (all kept alive: no reuse)."""
    live, rows = [], []
    for n in sizes:
        before = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
        live.append(torch.empty(n, dtype=torch.uint8, pin_memory=True))
        rows.append({"requested": n, "held": torch.cuda.host_memory_stats().get("allocated_bytes.current", 0) - before})
    return rows


def capacity_probe(cap_bytes, chunk=256 * MiB):
    held, rc = [], None
    try:
        while (len(held) + 1) * chunk <= cap_bytes:
            raw = torch.empty(chunk, dtype=torch.uint8)
            rc = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), chunk, 0)
            if rc != torch.cuda.cudart().cudaError.success:
                break
            held.append(raw)
            rc = None
        return {"cap_bytes": cap_bytes, "chunk_bytes": chunk, "registered_bytes": len(held) * chunk,
                "failed_with": None if rc is None else str(rc), "mem_available_while_held": meminfo()["mem_available"]}
    finally:
        for raw in held:
            torch.cuda.cudart().cudaHostUnregister(raw.data_ptr())


def pinned_transfer(cfg, result, trials=20, warmup=3, nbytes=None, capacity_gib=0.0, **_):
    import time

    from qwen_grpo.model import skeleton

    if not torch.cuda.is_available():
        raise lab.Blocked("host-to-device transfers need a CUDA device")
    slab_4b = plan(skeleton("Qwen/Qwen3-4B").requires_grad_(False).to(torch.bfloat16), cfg.execution)["max_slab_bytes"]
    nbytes = int(nbytes or slab_4b)
    pageable = torch.empty(nbytes, dtype=torch.uint8).fill_(1)
    pinned, release = registered(nbytes)
    pinned.fill_(1)
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    sides = {"reference": pageable, "candidate": pinned}  # reference pageable, candidate page-locked; same copy call

    def timed(src):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=False)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    try:
        assert not pageable.is_pinned() and pinned.is_pinned()
        result["timing"] = {s: {"warmup": warmup, "trials_s": [], "source": "pageable" if s == "reference" else "pinned"}
                            for s in sides}
        for _ in range(warmup):
            for src in sides.values():
                timed(src)
        for _ in range(trials):
            for s, src in sides.items():
                result["timing"][s]["trials_s"].append(timed(src))
        for s, t in result["timing"].items():
            t.update(lab.summarize(t["trials_s"]))
            t["median_gb_per_s"] = nbytes / t["median_s"] / 1e9
        r = lab.ratio_ci(result["timing"]["candidate"]["trials_s"], result["timing"]["reference"]["trials_s"])
        lo, hi = r["ci95"]
        result["conclusion"] = {**r, "statement": "candidate faster" if hi < 1 else "candidate slower" if lo > 1
                                else "no detectable difference"}
    finally:
        release()
    del pinned, pageable
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    sizes = [MiB + 1, 31_457_280, slab_4b]
    probe = None
    if capacity_gib:
        cap = min(int(capacity_gib * 2 ** 30), meminfo()["mem_available"] - 2 ** 30)
        probe = capacity_probe(cap)
    result["measurements"] = {"nbytes": nbytes, "qwen3_4b_max_slab_bytes": slab_4b,
                              "pin_memory_rounding": pin_memory_rounding(sizes),
                              "rlimit_memlock": {"soft": soft, "hard": hard}, "meminfo": meminfo(),
                              "capacity_probe": probe}
    result["memory"] = lab.memory("cuda")
    result["required_measurements"] = ["gpu_memory"]
