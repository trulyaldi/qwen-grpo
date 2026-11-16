"""Memory plan of an L2L layout: exact bytes of what the weight store and the GPU will hold, from the model's own
tensors (a meta-device skeleton works: only shapes and dtypes are read), plus the transient terms that can only be
estimated, computed by the Step-0 budget calculator (qwen_grpo/budget/calculator.py) rather than re-derived here.

Every item is tagged `exact` (counted from tensors) or `estimate` (a formula), and nothing measured is filled in: the
CUDA context and allocator overhead are left to measurement (lab.memory), as the protocol requires.

Assumption to be confirmed by the owner: resident_layers keeps the LAST N decoder layers on the GPU (they hold the
LoRA adapters of the default presets and are the first needed in backward).
"""
from qwen_grpo.budget import calculator as budget
from qwen_grpo.execution.weight_store import PAGE, align, decoder_layers, layout


def _bytes(t):
    return t.numel() * t.element_size()


def plan(model, execution, n_seq=None, seq_len=None, scored=None):
    """{item: {bytes, kind, note}} for `model` under the config's execution section (an l2l backend). n_seq, seq_len and
    scored (response tokens) enable the transient estimates."""
    if execution.backend not in ("l2l_sync", "l2l_async"):
        raise ValueError("a memory plan describes an l2l layout")
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = list(decoder_layers(model))
    frozen = [sum(_bytes(p) for _, p in layout(layer)[0]) for layer in layers]
    slabs = [sum(align(_bytes(p)) for _, p in layout(layer)[0]) for layer in layers]
    p_max = max(slabs)
    resident = range(len(layers) - execution.resident_layers, len(layers))
    streamed = [i for i in range(len(layers)) if i not in resident]
    lora = [p for p in model.parameters() if p.requires_grad]
    lora_bytes = sum(_bytes(p) for p in lora)
    embed, head = causal_lm.model.embed_tokens.weight, causal_lm.lm_head.weight
    tied = head is embed
    pinned = execution.weight_storage == "pinned"
    items = {
        "cpu.store": dict(bytes=sum(slabs) + PAGE * len(slabs), kind="exact",
                          note=f"{len(slabs)} slabs, tensors aligned to 256 B, + up to one page each for alignment"),
        "cpu.pinned": dict(bytes=sum(slabs) + PAGE * len(slabs) if pinned else 0, kind="exact",
                           note="cudaHostRegister of each slab allocation (exact size + one page)" if pinned else "pageable store"),
        "cpu.pinned_if_pin_memory": dict(bytes=sum(budget.pow2_ceil(s) for s in slabs) if pinned else 0, kind="exact",
                                         note="what pin_memory=True would hold (CachingHostAllocator rounds to 2^k)"),
        "gpu.weight_buffers": dict(bytes=execution.weight_buffers * p_max, kind="exact",
                                   note=f"{execution.weight_buffers} x largest slab ({p_max} B)"),
        "gpu.resident_layers": dict(bytes=sum(frozen[i] for i in resident), kind="exact",
                                    note=f"last {execution.resident_layers} decoder layers (assumption, see module doc)"),
        "gpu.embedding_and_head": dict(bytes=_bytes(embed) + (0 if tied else _bytes(head)), kind="exact",
                                       note="tied: one tensor" if tied else "untied: embedding + LM head"),
        "gpu.final_norm": dict(bytes=_bytes(causal_lm.model.norm.weight), kind="exact", note=""),
        "gpu.lora": dict(bytes=lora_bytes, kind="exact", note=f"{len(lora)} fp32 tensors"),
        "gpu.lora_grads": dict(bytes=lora_bytes, kind="exact", note="one gradient per LoRA tensor"),
        "gpu.adamw_state": dict(bytes=budget.adamw_state_bytes(sum(p.numel() for p in lora)), kind="exact",
                                note="exp_avg + exp_avg_sq in fp32"),
    }
    if n_seq and seq_len:
        cfg = causal_lm.config
        dtype = str(embed.dtype).removeprefix("torch.")
        items["gpu.kv_cache"] = dict(bytes=budget.kv_bytes(n_seq, seq_len, len(layers), cfg.num_key_value_heads,
                                                           cfg.head_dim, dtype), kind="estimate", note="all layers, full length")
        items["gpu.boundary_checkpoints"] = dict(
            bytes=budget.checkpoint_bytes(len(layers) + 1, n_seq, seq_len, cfg.hidden_size, dtype), kind="estimate",
            note="one hidden state per layer boundary (upper bound: Step 0 keeps only those above the first adapter)")
        if scored:
            items["gpu.logits"] = dict(bytes=budget.logits_bytes(n_seq, scored + 1, scored, cfg.vocab_size, dtype)["backward_peak"],
                                       kind="estimate", note="response-only logits, backward peak (Step 0 formula)")
    return {"layers": len(layers), "streamed_layers": streamed, "resident_layers": list(resident), "max_slab_bytes": p_max,
            "frozen_bytes": frozen, "slab_bytes": slabs, "tied_embeddings": tied, "items": items,
            "gpu_exact_bytes": sum(v["bytes"] for k, v in items.items() if k.startswith("gpu.") and v["kind"] == "exact"),
            "unmeasured": ["CUDA context", "allocator overhead", "non-PyTorch allocations"]}
