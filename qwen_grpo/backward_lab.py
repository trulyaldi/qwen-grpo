"""Ticket 04 lab case: gradient_equivalence (H02), checkpointed L2L backward against resident autograd.

For every model of the grid (fixture shapes x attention x dtype x LoRA placement) a resident reference (the whole
model on the device, explicit_forward with the plain layer call) and a candidate with identical weights (frozen decoder
weights in the WeightStore, backward.CheckpointedLayerExecutor) compute the same GRPO loss on the same rollouts
(fixtures.rollout_batch) in two micro-batches (the two prompt groups), for 1 and 2 weight buffers:
  loss.<tag>.<mb>                 the micro-batch's GRPO loss
  logits.<tag>.<mb>               its logits (the forward under grad mode)
  grads.<tag>.<step>.<param>      every LoRA parameter's .grad after micro-batch 1 (step mb1: single micro-batch) and
                                  after micro-batch 2 was accumulated into it (step acc)
  hidden_grads.<tag>.<mb>.<i>     dL/d(output of decoder layer i), wherever the resident graph has it (layers at or
                                  above the first adapted layer minus one; below it nothing requires grad)
Each LoRA comparison records max abs error, relative error, relative L2 error, cosine and both gradient norms
(lab.compare). Adapters: every projection of the placed layers, B ~ N(0, LORA_B_STD), so A's gradient is nonzero.

reference_integrity: the explicit resident forward equals HF's logits bitwise, and the resident gradients are bitwise
repeatable (a second resident run); the 0 threshold assumes a deterministic reference.
Measurements: saved_bytes / resident_saved_bytes, the bytes autograd saves during the two forwards (not during
backward's recomputation) for L2L and for the resident model with the same placement.
Errors (execution FAIL): a reference LoRA parameter without a gradient, changed Parameter identities or optimizer
references, a store that changed, a saved autograd tensor that lives in a weight buffer, frozen weights not back in
their slabs, backward weight loads for the wrong layers.
Fault injection (perturbation "detach_boundary=<i>"): the boundary activation saved for layer i is detached, so no
gradient reaches anything below layer i. Only the comparisons catch it.
"""
import contextlib

import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import lab
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.layer_stream import _inside, place_resident
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.grpo import grpo_loss, response_logprobs
from qwen_grpo.model import lora_config, lora_modules

ATTN = ("eager", "sdpa")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
BUFFERS = (1, 2)
ATTN_PROJ, MLP_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj"), ("gate_proj", "up_proj", "down_proj")


def placements(n):
    """{name: adapted decoder layers} for a model with n layers (the placements Ticket 04 asks for)."""
    return {"final": [n - 1], "last2": [n - 2, n - 1], "middle": [n // 2], "first_last": [0, n - 1],
            "distributed": sorted({0, n // 3, (2 * n) // 3, n - 1})}


def build(shape, attn, dtype, layers, vocab=F.VOCAB):
    """Seeded tiny Qwen3 with LoRA on every projection of `layers`; B ~ N(0, LORA_B_STD). Call inside isolated()."""
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, tie_word_embeddings=True, attn_implementation=attn,
                                     **F.SHAPES[shape])).to(dtype).eval()
    targets = [f"model.layers.{i}.self_attn.{p}" for i in layers for p in ATTN_PROJ]
    targets += [f"model.layers.{i}.mlp.{p}" for i in layers for p in MLP_PROJ]
    m = get_peft_model(m, lora_config(targets, F.RANK, F.ALPHA)).eval()
    for _, mod in lora_modules(m):
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=F.LORA_B_STD)
    return m


def microbatches(device):
    """The fixture rollouts split by prompt group: [(rollouts, advantages [G], old_logp [G, R])] x 2. old_logp is a
    seeded perturbation of nothing in particular (N(0, 1) - 3), so ratios differ from 1 and some tokens clip."""
    rollouts, _ = F.rollout_batch()
    adv = torch.tensor([1.0, -0.5, -0.5, 0.8, 0.4, -1.2], dtype=torch.float32)
    g = torch.Generator().manual_seed(F.SEED + 2)
    old = torch.randn(F.B * F.G, F.R, generator=g, dtype=torch.float32) - 3.0
    out = []
    for b in range(F.B):
        rows = slice(b * F.G, (b + 1) * F.G)
        r = {k: (v[rows].to(device) if isinstance(v, torch.Tensor) else v) for k, v in rollouts.items()}
        out.append((r, adv[rows].to(device), old[rows].to(device)))
    return out


def step(model, executor, mb, saved=None):
    """Forward + GRPO loss + backward of one micro-batch; .grad accumulates. -> (loss, logits, {i: dL/dhidden_i}).
    saved: a SavedTensors that records what the forward's graph saves (not the backward's own recomputation, whose
    graph is consumed and released inside the layer's weight binding)."""
    rollouts, adv, old = mb
    ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    hidden = {}

    def keep(i, h):
        if h.requires_grad:
            h.retain_grad()
            hidden[i] = h

    with saved.hooks() if saved is not None else contextlib.nullcontext():
        logits = explicit_forward(model, ids, mask, pos, layer_executor=executor, layer_callback=keep)  # [G, T, V]
        logp = response_logprobs(logits, ids, rollouts["prompt_len"])                                   # [G, R] fp32
        loss = grpo_loss(logp, old, adv, rollouts["completion_mask"], F.CLIP_EPS)
    if loss.requires_grad:  # not when a fault cut every path to a LoRA parameter: the comparisons then see no gradient
        loss.backward()
    return loss.detach(), logits.detach(), {i: h.grad for i, h in hidden.items()}


class SavedTensors:
    """saved_tensors_hooks that record every tensor autograd saves: the sum of their sizes (a tensor saved twice counts
    twice; saved parameters count too) and how many lie in a weight buffer (those would be overwritten by reuse)."""

    def __init__(self, buffers):
        self.buffers, self.bytes, self.in_buffer = buffers, 0, 0

    def pack(self, t):
        self.bytes += t.numel() * t.element_size()
        self.in_buffer += any(_inside(t, b) for b in self.buffers)
        return t

    def hooks(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, lambda t: t)


class DetachBoundary(CheckpointedLayerExecutor):
    """Fault injection: the activation saved for layer `at` is detached from the graph below it."""

    def __init__(self, store, buffers, at):
        super().__init__(store, buffers)
        self.at = at

    def boundary(self, index, hidden):
        return hidden.detach() if index == self.at else hidden


def executor(store, buffers, perturbation):
    if perturbation is None:
        return CheckpointedLayerExecutor(store, buffers)
    kind, value = perturbation.split("=")
    if kind != "detach_boundary":
        raise lab.SetupError(f"gradient_equivalence supports perturbation detach_boundary=<i>, not {perturbation!r}")
    return DetachBoundary(store, buffers, int(value))


def run_resident(model, mbs, saved=None):
    model.zero_grad(set_to_none=True)
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    out = []
    for mb in mbs:
        loss, logits, hidden = step(model, None, mb, saved)
        out.append({"loss": loss, "logits": logits, "hidden": hidden,
                    "grads": {n: None if p.grad is None else p.grad.clone() for n, p in params.items()}})
    return out


def run_model(cfg, result, shape, attn, dt, placement, layers, device, perturbation, tol):
    dtype = DTYPES[dt]
    tag0 = f"{shape}.{attn}.{dt}.{placement}"
    with F.isolated():
        ref = build(shape, attn, dtype, layers)
        cand = build(shape, attn, dtype, layers)
    cand.load_state_dict(ref.state_dict(), strict=True)
    ref.to(device)
    mbs = microbatches(device)
    saved = SavedTensors([])
    with F.isolated():
        first, second = run_resident(ref, mbs, saved), run_resident(ref, mbs)
    result["measurements"]["resident_saved_bytes"][tag0] = saved.bytes
    for k, (a, b) in enumerate(zip(first, second)):
        for n, g in a["grads"].items():
            if g is None:
                result["errors"].append(f"{tag0}: reference LoRA parameter {n} has no gradient")
            elif not torch.equal(g, b["grads"][n]):
                result["reference_integrity"].append(f"{tag0}.mb{k + 1}: resident gradient of {n} is not repeatable")
        rollouts = mbs[k][0]
        ids, mask = rollouts["input_ids"], rollouts["attention_mask"]
        with F.isolated(), torch.no_grad():
            hf = ref(input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(1) - 1).clamp(min=0),
                     use_cache=False).logits
        if not torch.equal(hf, a["logits"]):
            result["reference_integrity"].append(f"{tag0}.mb{k + 1}: explicit resident logits != HF")
    store = WeightStore.build(cand, cfg.execution.weight_storage)
    try:
        place_resident(cand, store, device)
        params = {n: p for n, p in cand.named_parameters() if p.requires_grad}
        optimizer = torch.optim.AdamW(params.values(), lr=F.LR, weight_decay=0.0)
        ids_before = {n: id(p) for n, p in params.items()}
        expected_backward = sorted(range(min(layers), cand.config.num_hidden_layers), reverse=True)
        for n_buf in BUFFERS:
            tag = f"{tag0}.b{n_buf}"
            buffers = gpu_buffers(n_buf, store.max_slab_bytes, device)
            ex = executor(store, buffers, perturbation)
            saved = SavedTensors(buffers)  # both micro-batches
            cand.zero_grad(set_to_none=True)
            ref_t, got_t = {}, {}
            for k, mb in enumerate(mbs):
                mbk = f"mb{k + 1}"
                ex.trace.clear()
                with F.isolated():
                    loss, logits, hidden = step(cand, ex, mb, saved)
                loads = [t["layer"] for t in ex.trace if t["phase"] == "backward"]
                if perturbation is None and loads != expected_backward:
                    result["errors"].append(f"{tag}.{mbk}: backward loaded layers {loads}, expected {expected_backward}")
                r = first[k]
                ref_t[f"loss.{tag}.{mbk}"], got_t[f"loss.{tag}.{mbk}"] = r["loss"], loss
                ref_t[f"logits.{tag}.{mbk}"], got_t[f"logits.{tag}.{mbk}"] = r["logits"], logits
                for i, g in r["hidden"].items():
                    ref_t[f"hidden_grads.{tag}.{mbk}.{i}"] = g
                for i, g in hidden.items():
                    if g is not None:
                        got_t[f"hidden_grads.{tag}.{mbk}.{i}"] = g
                stepname = "mb1" if k == 0 else "acc"
                for n, p in params.items():
                    ref_t[f"grads.{tag}.{stepname}.{n}"] = r["grads"][n]
                    if p.grad is not None:
                        got_t[f"grads.{tag}.{stepname}.{n}"] = p.grad.clone()
            result["measurements"]["saved_bytes"][tag] = saved.bytes
            if saved.in_buffer:
                result["errors"].append(f"{tag}: {saved.in_buffer} saved autograd tensors live in a weight buffer")
            result["comparisons"] += [{"model": tag, **row} for row in lab.compare_all(ref_t, got_t, tol)]
            del buffers, ex
        if {n: id(p) for n, p in cand.named_parameters() if p.requires_grad} != ids_before:
            result["errors"].append(f"{tag0}: LoRA Parameter identities changed")
        if [id(p) for g in optimizer.param_groups for p in g["params"]] != list(ids_before.values()):
            result["errors"].append(f"{tag0}: optimizer parameter references changed")
        if moved := [f"{s.index}.{e.name}" for s in store.slabs for e in s.entries
                     if e.param.device.type != "cpu" or e.param.data_ptr() != s.data.data_ptr() + e.offset]:
            result["errors"].append(f"{tag0}: frozen weights not back in their slabs: {moved[:5]}")
        if problems := store.verify():
            result["errors"] += [f"{tag0}: store: {p}" for p in problems[:5]]
    finally:
        store.close()


def gradient_equivalence(cfg, result, perturbation=None, device="cpu", **_):
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise lab.Blocked("gradient_equivalence --device cuda: no CUDA device is available")
    device = torch.empty(0, device=device).device
    result.update(comparisons=[], reference_integrity=[], device=str(device),
                  measurements={"saved_bytes": {}, "resident_saved_bytes": {}})
    if perturbation:
        result["perturbation"] = {"spec": perturbation,
                                  "effect": "the activation saved for layer i is detached: no gradient below layer i"}
    tolerances = {dt: lab.tolerances_for("gradient_equivalence", cfg, dt) for dt in DTYPES}
    for shape in F.SHAPES:
        for attn in ATTN:
            for dt in DTYPES:
                for placement, layers in placements(F.SHAPES[shape]["num_hidden_layers"]).items():
                    run_model(cfg, result, shape, attn, dt, placement, layers, device, perturbation, tolerances[dt])
    result["memory"] = lab.memory(device)
