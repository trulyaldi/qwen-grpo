"""Ticket 08 lab case: prefetch_timing (H06), asynchronous vs synchronous L2L on the real model, on a CUDA device.

One model (config model.name / dtype / LoRA; LoRA B set to a seeded N(0, 1e-2) so every gradient is nonzero), one
pinned WeightStore, two executors with their own pools of execution.weight_buffers GPU buffers (both pools exist
throughout, so memory baselines are equal): the synchronous CheckpointedLayerExecutor (reference) and the
PrefetchingCheckpointedExecutor with execution.prefetch_depth (candidate). Model, inputs, batch, precision and
residency are identical; only the executor differs.

Per trial, for each side (sides alternate their order trial by trial, after `warmup` untimed trials):
  generation   greedy decode of training.batch_size prompts x training.group_size, `max_new` tokens
  forward      no-grad log-probs of the rollouts in micro-batches (prepare's pass)
  step         the GRPO step: forward with grad, GRPO loss, backward, AdamW (each part timed); LoRA weights and AdamW
               state are reset before every step, so every trial computes the same thing
Each part is timed by wall clock between torch.cuda.synchronize() calls (measurement boundaries, outside the
pipeline). The executors also record CUDA events around every weight copy and every layer block: from them, per part,
H2D copy time, compute time, bytes copied and the measured overlap (time during which a copy and a compute block run
at once on the GPU; 0 for the synchronous executor by construction, measured anyway).
Primary timing (timing.reference / candidate): the whole GRPO step. The other parts are measurements with their own
medians, IQRs and candidate/reference ratio CIs. Correctness: the first trial's generated tokens, forward log-probs,
step loss and every LoRA gradient must be bitwise equal between the sides (errors otherwise).
"""
import time

import torch
from transformers import AutoTokenizer

from qwen_grpo import lab
from qwen_grpo.config import DTYPES
from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.generator import generate
from qwen_grpo.execution.layer_stream import place_resident
from qwen_grpo.execution.output_head import streamed_logprobs
from qwen_grpo.execution.prefetch import PrefetchingCheckpointedExecutor, timeline_report
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.grpo import grpo_loss
from qwen_grpo.model import EOS_IDS, lora_model, lora_modules
from qwen_grpo.sentiment import build_prompts, load_splits
from qwen_grpo.training import balanced_subset, micro_batches

PARTS = ("generation", "forward", "step_forward", "step_backward", "step_optimizer", "step")


class TimedSync(CheckpointedLayerExecutor):
    """The synchronous executor with the same CUDA-event timeline as the prefetching one (copies, compute blocks)."""

    def __init__(self, store, buffers):
        super().__init__(store, buffers)
        self._events = []

    def _acquire(self, index, phase):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        out = super()._acquire(index, phase)                                       # the copy (synchronous)
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._events.append(("copy", index, phase, start, end))
        self._block = (index, phase, end)
        return out

    def _release(self, index, phase):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        i, ph, start = self._block
        self._events.append(("compute", i, ph, start, end))
        super()._release(index, phase)

    def report(self):
        out = timeline_report(self._events)
        self._events = []
        return out


def timeline_summary(rep):
    if rep is None:  # CPU buffers (smoke test): no CUDA timeline
        return {"copy_ms": None, "compute_ms": None, "overlap_ms": None, "overlap_fraction_of_copy": None, "copies": None}
    return {k: rep[k] for k in ("copy_ms", "compute_ms", "overlap_ms", "overlap_fraction_of_copy", "copies")}


def prefetch_timing(cfg, result, trials=7, warmup=2, device="cuda", max_new=16, perturbation=None, smoke=False, **_):
    """smoke=True (tests only): allow a CPU device, to exercise this function's code paths without CUDA."""
    if perturbation:
        raise lab.SetupError("prefetch_timing has no fault injection (see gradient_equivalence unsafe_reuse=<i>)")
    if not smoke and (torch.device(device).type != "cuda" or not torch.cuda.is_available()):
        raise lab.Blocked("prefetch_timing measures CUDA streams: it needs --device cuda and a CUDA device")
    if cfg.execution.backend != "l2l_async":
        raise lab.Blocked("prefetch_timing compares l2l_async (candidate) with l2l_sync: use an l2l_async config")
    device = torch.empty(0, device=device).device
    x, t, lo = cfg.execution, cfg.training, cfg.lora
    B, G, mb = t.batch_size, t.group_size, t.micro_batch_size

    torch.manual_seed(t.seed)
    model = lora_model(DTYPES[cfg.model.dtype], "cpu", lo.spec(), lo.rank, lo.alpha, name=cfg.model.name)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _, m in lora_modules(model):
            w = m.lora_B["default"].weight
            w.copy_((torch.randn(w.shape, generator=g) * 1e-2).to(w.dtype))
    store = WeightStore.build(model, x.weight_storage)
    try:
        place_resident(model, store, device)
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        init = {n: p.detach().clone() for n, p in params.items()}
        sides = {"reference": TimedSync(store, gpu_buffers(x.weight_buffers, store.max_slab_bytes, device)),
                 "candidate": PrefetchingCheckpointedExecutor(store, gpu_buffers(x.weight_buffers, store.max_slab_bytes,
                                                                                 device), x.prefetch_depth, timeline=True)}
        tok = AutoTokenizer.from_pretrained(cfg.model.name, padding_side="left")
        examples = balanced_subset(load_splits()["train"], max(2, B + B % 2), cfg.data.subset_seed)[:B]
        batch = tok(build_prompts(tok, [e["text"] for e in examples]), return_tensors="pt", padding=True,
                    add_special_tokens=False).to(device)
        P = batch["input_ids"].shape[1]
        with torch.no_grad():
            seq = generate(model, batch["input_ids"], batch["attention_mask"], EOS_IDS, tok.pad_token_id, max_new, G,
                           do_sample=False, layer_executor=sides["reference"])["sequences"]
        sides["reference"].report()
        R = seq.shape[1] - P
        resp = seq[:, P:]
        is_eos = torch.isin(resp, torch.tensor(EOS_IDS, device=device))
        completion = (torch.cumsum(is_eos.long(), 1) - is_eos.long()) == 0
        rollouts = {"input_ids": seq, "prompt_len": P, "response_ids": resp, "completion_mask": completion,
                    "attention_mask": torch.cat([batch["attention_mask"].repeat_interleave(G, 0), completion.long()], 1)}
        adv = torch.linspace(-1.0, 1.0, B * G, device=device)
        result["measurements"] = {"rows": B * G, "prompt_len": P, "response_len": R, "max_new": max_new,
                                  "micro_batch_size": mb, "layers": len(store.slabs), "depth": x.prefetch_depth,
                                  "weight_buffers": x.weight_buffers, "slab_bytes": store.max_slab_bytes,
                                  "streamed_bytes_per_pass": sum(s.nbytes for s in store.slabs),
                                  "parts": {s: {p: [] for p in PARTS} for s in sides},
                                  "timelines": {s: {p: [] for p in ("generation", "forward", "step")} for s in sides},
                                  "peak_allocated_bytes": {s: {} for s in sides}}
        meas = result["measurements"]
        with torch.no_grad():
            old = torch.cat([streamed_logprobs(model, c, None, sides["reference"]) for _, c in micro_batches(rollouts, mb)])
        sides["reference"].report()
        first = {}

        def reset():
            with torch.no_grad():
                for n, p in params.items():
                    p.copy_(init[n])
                    p.grad = None

        def run(side, timed):
            ex = sides[side]
            parts, outs = {}, {}
            reset()  # every part of every run starts from the same LoRA weights (the previous step updated them)
            for name, fn in (("generation", lambda: generate(model, batch["input_ids"], batch["attention_mask"], EOS_IDS,
                                                             tok.pad_token_id, max_new, G, do_sample=False,
                                                             layer_executor=ex)["sequences"]),
                             ("forward", lambda: torch.cat([streamed_logprobs(model, c, None, ex)
                                                            for _, c in micro_batches(rollouts, mb)]))):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                with torch.no_grad():
                    outs[name] = fn()
                torch.cuda.synchronize()
                parts[name] = time.perf_counter() - t0
                meas["peak_allocated_bytes"][side][name] = torch.cuda.max_memory_allocated()
                rep = ex.report()
                if timed:
                    meas["timelines"][side][name].append(timeline_summary(rep))
            reset()
            opt = torch.optim.AdamW(params.values(), lr=t.learning_rate, weight_decay=0.0)
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sf = sb = 0.0
            loss_total = 0.0
            N = B * G
            for rows, c in micro_batches(rollouts, mb):
                t0 = time.perf_counter()
                lp = streamed_logprobs(model, c, None, ex)
                loss = grpo_loss(lp, old[rows], adv[rows], c["completion_mask"], t.clip_eps) * (c["input_ids"].shape[0] / N)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                sf, sb, loss_total = sf + (t1 - t0), sb + (t2 - t1), loss_total + loss.item()
            outs["loss"] = loss_total
            outs["grads"] = {n: p.grad.detach().clone() for n, p in params.items()}
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            so = time.perf_counter() - t0
            meas["peak_allocated_bytes"][side]["step"] = torch.cuda.max_memory_allocated()
            reset()
            parts.update(step_forward=sf, step_backward=sb, step_optimizer=so, step=sf + sb + so)
            rep = ex.report()
            if timed:
                meas["timelines"][side]["step"].append(timeline_summary(rep))
                for p in PARTS:
                    meas["parts"][side][p].append(parts[p])
            return outs

        for trial in range(warmup + trials):
            order = ("reference", "candidate") if trial % 2 == 0 else ("candidate", "reference")
            for side in order:
                outs = run(side, timed=trial >= warmup)
                if trial == 0:
                    first[side] = outs
        a, b = first["reference"], first["candidate"]
        if not torch.equal(a["generation"], b["generation"]):
            result["errors"].append("generated tokens differ between l2l_sync and l2l_async")
        if not torch.equal(a["forward"], b["forward"]):
            result["errors"].append("forward log-probs differ between l2l_sync and l2l_async")
        if a["loss"] != b["loss"] or any(not torch.equal(a["grads"][n], b["grads"][n]) for n in a["grads"]):
            result["errors"].append("GRPO step loss or LoRA gradients differ between l2l_sync and l2l_async")
        if any(g.abs().max() == 0 for g in a["grads"].values()):
            result["errors"].append("a LoRA gradient is identically zero: the step comparison has no power")
        if problems := store.verify():
            result["errors"] += [f"store: {p}" for p in problems[:5]]
        result["timing"] = {s: {"warmup": warmup, "trials_s": meas["parts"][s]["step"]} for s in sides}
        for s, tm in result["timing"].items():
            tm.update(lab.summarize(tm["trials_s"]))
        r = lab.ratio_ci(result["timing"]["candidate"]["trials_s"], result["timing"]["reference"]["trials_s"])
        lo_, hi_ = r["ci95"]
        result["conclusion"] = {**r, "statement": "candidate faster" if hi_ < 1 else "candidate slower" if lo_ > 1
                                else "no detectable difference", "what": "whole GRPO step (forward + backward + AdamW)"}
        meas["parts_summary"] = {s: {p: lab.summarize(v) for p, v in meas["parts"][s].items()} for s in sides}
        meas["parts_ratio_ci"] = {p: lab.ratio_ci(meas["parts"]["candidate"][p], meas["parts"]["reference"][p])
                                  for p in PARTS}
        meas["timeline_median"] = {s: {p: {k: _median([row[k] for row in rows if row[k] is not None]) for k in rows[0]}
                                       if rows else None for p, rows in meas["timelines"][s].items()} for s in sides}
        result["memory"] = lab.memory(device)
        if device.type == "cuda":
            result["required_measurements"] = ["gpu_memory"]
    finally:
        store.close()


def _median(x):
    x = sorted(x)
    n = len(x)
    return (x[n // 2] if n % 2 else (x[n // 2 - 1] + x[n // 2]) / 2) if x else None
