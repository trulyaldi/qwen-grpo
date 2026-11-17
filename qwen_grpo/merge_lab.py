"""Ticket 10 lab case: merge_equivalence (H09), the streaming LoRA merge against the adapter-attached model.

Variants: a tiny random Qwen3 (real tokenizer vocabulary) in fp32 and in bf16, with LoRA (last2-attention, B ~ N(0,
0.5) so the adapter changes logits strongly); with --real, the config's model (Qwen3-0.6B, bf16) with a trained
adapter (default runs/qwen/checkpoints/step90). Per variant:
  merge      export.merge.stream_merge of the base checkpoint and the adapter, in a child process (so its peak CPU
             memory is its own; RssAnon / VmRSS sampled every 10 ms; the child must never initialize CUDA)
  evaluate   three models through the L2L engine on --device, identical inputs (validation prompts):
               adapter  the PEFT model with the adapter (L2LBackend), base = the same with disable_adapter()
               merged   the merged checkpoint loaded on its own as a plain Hugging Face model (L2LBackend)
             full logits per prompt (explicit forward through the streaming executor) and greedy generation
             (execution.generator, never model.generate), with the peak GPU memory of each evaluation
Comparisons (tolerances.yaml, merge_equivalence, by dtype):
  weights.<variant>.<tensor>       merged tensor vs the exact fp32 W0 + (alpha/r) B A; and every element must lie
                                   within half an ulp of it (one correct rounding: an error, not a threshold, if not)
  logits.<variant>.<i>             merged vs adapter-attached logits at the prompt's real positions
  greedy                           greedy tokens of merged vs adapter-attached, row by row; a row may differ only at
                                   a near-tie (top-1 / top-2 margin of the adapter's logits <= 2 x the logit bound)
Execution requirements: the adapter's own effect (adapter vs base) is at least POWER[dtype] times the merge
discrepancy (power), every adapted tensor actually changed (adapter changes survive), the merged checkpoint loads
independently (plain HF, no LoRA tensors, the index covers every base tensor, tied embedding / head preserved), and
on a GPU the peak allocated memory of the merged evaluation stays below the model's size (never fully resident).
"""
import json
import multiprocessing as mp
import tempfile
import threading
import time
from pathlib import Path

import torch

from qwen_grpo import lab
from qwen_grpo.config import DTYPES
from qwen_grpo.execution import generator as gen
from qwen_grpo.execution.backend import L2LBackend
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.model import EOS_IDS, lora_config, lora_modules, lora_model, target_modules
from qwen_grpo.export.merge import base_files, load_merged_l2l, resolve_base, stream_merge

POWER = {"float32": 100.0, "bfloat16": 10.0}  # adapter effect / merge discrepancy, at least
REAL_ADAPTER = Path(__file__).resolve().parents[1] / "runs" / "qwen" / "checkpoints" / "step90"  # (lab imports us)


def _sample_memory(stop, peaks):
    while not stop.is_set():
        try:
            for line in open("/proc/self/status"):
                k, _, v = line.partition(":")
                if k in ("RssAnon", "VmRSS"):
                    peaks[k] = max(peaks.get(k, 0), int(v.split()[0]) * 1024)
        except OSError:
            pass
        time.sleep(0.01)


def _merge_child(base, adapter, out, shard_bytes, q):
    peaks = {}
    stop = threading.Event()
    t = threading.Thread(target=_sample_memory, args=(stop, peaks), daemon=True)
    start_anon = None
    try:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon"):
                start_anon = int(line.split()[1]) * 1024
    except OSError:
        pass
    t.start()
    t0 = time.perf_counter()
    try:
        rep = stream_merge(base, adapter, out, shard_bytes=shard_bytes)
        err = None
    except Exception as e:  # reported to the parent as an error
        rep, err = None, f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - t0
    stop.set()
    t.join()
    q.put({"report": rep, "error": err, "seconds": seconds, "peak_rss_anon_bytes": peaks.get("RssAnon"),
           "peak_vm_rss_bytes": peaks.get("VmRSS"), "start_rss_anon_bytes": start_anon,
           "cuda_initialized": torch.cuda.is_initialized()})


def merge_in_child(base, adapter, out, shard_bytes, timeout_s=3600):
    """Run the merge in a fresh (spawned) process; never blocks on a child that died without reporting."""
    import queue
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_merge_child, args=(str(base), str(adapter), str(out), shard_bytes, q))
    p.start()
    deadline = time.time() + timeout_s
    while True:
        try:
            res = q.get(timeout=1.0)
            break
        except queue.Empty:
            if not p.is_alive() or time.time() > deadline:
                p.terminate()
                p.join()
                return {"report": None, "error": f"merge process ended without a result (exit code {p.exitcode})",
                        "seconds": None, "peak_rss_anon_bytes": None, "peak_vm_rss_bytes": None,
                        "start_rss_anon_bytes": None, "cuda_initialized": None}
    p.join()
    return res


def tiny_variant(work, dtype):
    """Save a tiny random Qwen3 base and a LoRA adapter for it; return (base dir, adapter dir, PEFT model)."""
    from peft import get_peft_model
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    with torch.random.fork_rng():
        torch.manual_seed(0)
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                            tie_word_embeddings=True)).to(dtype).eval()
        base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
        bdir = work / "base"
        base.save_pretrained(bdir)
        model = get_peft_model(base, lora_config(target_modules(base, ("q_proj", "v_proj", "o_proj"), 2), 8, 16)).eval()
        for _, m in lora_modules(model):
            torch.nn.init.normal_(m.lora_B["default"].weight, std=0.5)
    adir = work / "adapter"
    model.save_pretrained(adir)
    return bdir, adir


def prompts(tok, n):
    from qwen_grpo.sentiment import build_prompts, load_splits
    ex = load_splits()["validation"][:n]
    return build_prompts(tok, [e["text"] for e in ex])


def logits_one(model, ex, ids, mask):
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    with torch.no_grad():
        return explicit_forward(model, ids, mask, pos, 0, layer_executor=ex)[0, mask[0].bool()].float()  # [T_real, V]


def run_variant(cfg, result, label, dt, bdir, adir, adapter_model, tok, device, tol, n_logits, n_greedy, storage, nbuf,
                work):
    dtype = DTYPES[dt]
    out = work / "merged"
    child = merge_in_child(bdir, adir, out, shard_bytes=64 << 20)
    meas = {k: v for k, v in child.items() if k != "report"}
    if child["error"]:
        result["errors"].append(f"{label}: merge failed: {child['error']}")
        return
    rep = child["report"]
    if child["cuda_initialized"]:
        result["errors"].append(f"{label}: the merge process initialized CUDA")
    # weights: merged tensors against the exact fp32 sum
    files = base_files(Path(rep["base"]))
    from safetensors import safe_open
    from qwen_grpo.export.merge import read_adapter
    scale, pairs, _ = read_adapter(adir)
    merged_files = base_files(out)
    ref_t, got_t = {}, {}
    for name, (A, B) in pairs.items():
        with safe_open(files[name], framework="pt") as h0, safe_open(merged_files[name], framework="pt") as hm:
            W0, Wm = h0.get_tensor(name), hm.get_tensor(name)
        exact = W0.float() + scale * (B @ A)
        ref_t[f"weights.{label}.{name}"], got_t[f"weights.{label}.{name}"] = exact, Wm.float()
        if not rep["merged"][name]["within_half_ulp"]:
            result["errors"].append(f"{label}: {name} is not within half an ulp of W0 + (a/r)BA")
        if rep["merged"][name]["changed_elements"] == 0:
            result["errors"].append(f"{label}: {name} unchanged by the merge: the adapter did not survive")
    if set(merged_files) != set(files):
        result["errors"].append(f"{label}: the merged index does not hold exactly the base tensors")
    if any("lora_" in k for k in merged_files):
        result["errors"].append(f"{label}: LoRA tensors in the merged checkpoint")
    # three models through L2L
    adapter_backend = L2LBackend(adapter_model, device, storage, nbuf)
    merged, merged_backend = load_merged_l2l(out, device, dtype, storage, nbuf)
    if merged.config.tie_word_embeddings and merged.lm_head.weight is not merged.model.embed_tokens.weight:
        result["errors"].append(f"{label}: the merged model's LM head is not tied to its embedding")
    if any("lora" in n for n, _ in merged.named_parameters()):
        result["errors"].append(f"{label}: the merged model still has LoRA parameters")
    texts = prompts(tok, max(n_logits, n_greedy))
    peak = {}
    adapter_vs_base, adapter_vs_merged = 0.0, 0.0
    for i, text in enumerate(texts[:n_logits]):
        enc = tok([text], return_tensors="pt", add_special_tokens=False).to(device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        za = logits_one(adapter_model, adapter_backend.executor, ids, mask)
        with adapter_model.disable_adapter():
            zb = logits_one(adapter_model, adapter_backend.executor, ids, mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        zm = logits_one(merged, merged_backend.executor, ids, mask)
        if device.type == "cuda":
            peak["merged_logits"] = max(peak.get("merged_logits", 0), torch.cuda.max_memory_allocated())
        adapter_vs_base = max(adapter_vs_base, (za - zb).abs().max().item())
        adapter_vs_merged = max(adapter_vs_merged, (za - zm).abs().max().item())
        ref_t[f"logits.{label}.{i}"], got_t[f"logits.{label}.{i}"] = za.cpu(), zm.cpu()
        del za, zb, zm
    power = adapter_vs_base / max(adapter_vs_merged, 1e-30)
    if power < POWER[dt]:
        result["errors"].append(f"{label}: adapter effect / merge discrepancy {power:.1f} < {POWER[dt]}: no power")
    # greedy generation, batched, left-padded
    enc = tok(texts[:n_greedy], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    pad = tok.pad_token_id
    with torch.no_grad():
        ga = gen.generate(adapter_model, enc["input_ids"], enc["attention_mask"], EOS_IDS, pad, 4, do_sample=False,
                          layer_executor=adapter_backend.executor, keep_logits=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        gm = gen.generate(merged, enc["input_ids"], enc["attention_mask"], EOS_IDS, pad, 4, do_sample=False,
                          layer_executor=merged_backend.executor)
        if device.type == "cuda":
            peak["merged_generation"] = torch.cuda.max_memory_allocated()
        with adapter_model.disable_adapter():
            gb = gen.generate(adapter_model, enc["input_ids"], enc["attention_mask"], EOS_IDS, pad, 4, do_sample=False,
                              layer_executor=adapter_backend.executor)
    P = enc["input_ids"].shape[1]
    a_tok, m_tok, b_tok = ga["sequences"][:, P:], gm["sequences"][:, P:], gb["sequences"][:, P:]
    R = min(a_tok.shape[1], m_tok.shape[1])
    near, bad = 0, 0
    for row in range(a_tok.shape[0]):
        diff = [k for k in range(R) if a_tok[row, k] != m_tok[row, k]]
        if not diff and a_tok.shape[1] == m_tok.shape[1]:
            continue
        k = diff[0] if diff else R
        z = ga["step_logits"][min(k, len(ga["step_logits"]) - 1)][row]
        top2 = z.topk(2).values
        if top2[0] - top2[1] <= 2 * tol["logits"]["rtol"] * z.abs().max():
            near += 1
        else:
            bad += 1
    result["comparisons"].append({"quantity": "greedy", "name": f"greedy.{label}", "model": label, "pass": bad == 0,
                                  "reason": None if bad == 0 else f"{bad} rows differ without a near-tie"})
    dec = lambda t: [tok.decode(r[r != pad]) for r in t]
    agree = sum(x == y for x, y in zip(dec(a_tok), dec(m_tok))) / a_tok.shape[0]
    model_bytes = sum(p.numel() * p.element_size() for p in merged.parameters())
    if device.type == "cuda" and max(peak.values()) >= model_bytes:
        result["errors"].append(f"{label}: peak GPU memory {max(peak.values())} B >= model size {model_bytes} B")
    meas.update(report_summary={k: rep[k] for k in ("scale", "r", "alpha", "copied", "shards", "total_bytes")},
                per_tensor={n: rep["merged"][n] for n in rep["merged"]},
                adapter_vs_base_max_abs=adapter_vs_base, adapter_vs_merged_max_abs=adapter_vs_merged, power_ratio=power,
                greedy_rows=int(a_tok.shape[0]), greedy_near_ties=near, greedy_disagreements_without_near_tie=bad,
                decoded_agreement_adapter_vs_merged=agree,
                decoded_agreement_adapter_vs_base=sum(x == y for x, y in zip(dec(a_tok), dec(b_tok))) / a_tok.shape[0],
                peak_gpu_allocated_bytes=peak or None, model_bytes=model_bytes,
                base_checkpoint_bytes=sum(f.stat().st_size for f in set(files.values())))
    result["measurements"][label] = meas
    result["comparisons"] += [{"model": label, **row} for row in lab.compare_all(ref_t, got_t, tol)]
    adapter_backend.close()
    merged_backend.close()


def merge_equivalence(cfg, result, perturbation=None, device="cpu", real=False, adapter=None, n_logits=8,
                      n_greedy=32, **_):
    if perturbation:
        raise lab.SetupError("merge_equivalence has no fault injection: a wrong merge is the adapter-vs-base distance "
                             "(the power requirement)")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise lab.Blocked("merge_equivalence --device cuda: no CUDA device is available")
    device = torch.empty(0, device=device).device
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
    result.update(comparisons=[], reference_integrity=[], device=str(device), measurements={})
    storage, nbuf = cfg.execution.weight_storage, cfg.execution.weight_buffers
    with tempfile.TemporaryDirectory(prefix="merge_lab_") as tmp:
        tmp = Path(tmp)
        variants = [("tiny_float32", "float32"), ("tiny_bfloat16", "bfloat16")] if not real else []
        for label, dt in variants:
            work = tmp / label
            work.mkdir()
            bdir, adir = tiny_variant(work, DTYPES[dt])
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, GenerationConfig
            base = AutoModelForCausalLM.from_pretrained(bdir, dtype=DTYPES[dt]).eval()
            base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
            model = PeftModel.from_pretrained(base, adir, is_trainable=True).eval()  # LoRA stays out of the store
            run_variant(cfg, result, label, dt, bdir, adir, model, tok, device,
                        lab.tolerances_for("merge_equivalence", cfg, dt), n_logits, n_greedy, storage, nbuf, work)
        if real:
            dt = cfg.model.dtype
            adir = Path(adapter) if adapter else REAL_ADAPTER
            lo = json.loads((adir / "adapter_config.json").read_text())
            bdir = resolve_base(cfg.model.name)
            model = lora_model(DTYPES[dt], "cpu", lo["target_modules"], lo["r"], lo["lora_alpha"], name=cfg.model.name)
            from qwen_grpo.checkpointing import load_adapter
            load_adapter(model, adir)
            work = tmp / "real"
            work.mkdir()
            run_variant(cfg, result, f"real_{dt}", dt, bdir, adir, model, tok, device,
                        lab.tolerances_for("merge_equivalence", cfg, dt), n_logits, n_greedy, storage, nbuf, work)
    result["memory"] = lab.memory(device)
