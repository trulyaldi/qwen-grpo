"""Pinned CPU weight store (qwen_grpo/execution/weight_store.py) and memory plan (memory_plan.py), on the Ticket 00
fixtures and tiny bf16 models. Tolerance 0 throughout: the store moves bytes, it computes nothing.

Page-locking needs a CUDA device. Tests marked `needs_cuda` are skipped, with that reason, in the CPU-only suite and are
run with CUDA visible as Ticket 02's evidence (research/l2l/experiments/ticket02_tests.log). They use the GPU only for
cudaHostRegister and small buffer copies."""
import dataclasses

import pytest
import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import config
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.memory_plan import plan
from qwen_grpo.execution.weight_store import (ALIGN, PAGE, PinningError, WeightStore, decoder_layers, digest,
                                              gpu_buffers)
from qwen_grpo.model import PRESETS, lora_config, lora_modules, target_modules

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="page-locking (cudaHostRegister) and GPU buffers need a CUDA device")
STORAGES = [pytest.param("pinned", marks=needs_cuda), "pageable"]
L2L = config.Execution(backend="l2l_sync", weight_storage="pinned", resident_layers=0, weight_buffers=2,
                       prefetch_depth=0, activation_checkpointing=True)


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)  # tests/test_grpo.py sets float64 at import


@pytest.fixture
def stores():
    built = []
    yield built
    for s in built:
        s.close()


def build(model, storage, stores):
    s = WeightStore.build(model, storage)
    stores.append(s)
    return s


def frozen_decoder(model):
    return {f"{i}.{n}": p for i, layer in enumerate(decoder_layers(model)) for n, p in layer.named_parameters()
            if not p.requires_grad}


def tiny_bf16(layers=4, seed=0):
    with F.isolated(seed):
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, tie_word_embeddings=True, **F.SHAPES["4b-like"],
                                            )).to(torch.bfloat16).eval()
        base.config.num_hidden_layers = len(base.model.layers)
        return get_peft_model(base, lora_config(target_modules(base, *PRESETS["last2-attention"]))).eval()


# --- ownership -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("shape", list(F.SHAPES))
def test_store_owns_every_frozen_decoder_tensor_and_nothing_else(shape, storage, stores):
    model, saved = F.load(shape)
    lora = {n: (p, p.data_ptr()) for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p, _ in lora.values()], lr=1e-3)            # built before the store
    causal = model.get_base_model()
    outside = {"embed": causal.model.embed_tokens.weight, "norm": causal.model.norm.weight}
    outside_ptrs = {k: v.data_ptr() for k, v in outside.items()}
    before = {k: digest(p) for k, p in frozen_decoder(model).items()}

    s = build(model, storage, stores)
    stored = {f"{sl.index}.{e.name}": e for sl in s.slabs for e in sl.entries}
    assert stored.keys() == before.keys()                                              # every frozen decoder tensor
    assert all(stored[k].sha256 == before[k] for k in before)                          # with its exact bytes
    assert all(stored[k].param is p for k, p in frozen_decoder(model).items())         # same Parameter objects
    assert all(p.data_ptr() == ptr for p, ptr in lora.values())                        # LoRA untouched
    assert [p for g in optimizer.param_groups for p in g["params"]] == [p for p, _ in lora.values()]
    assert {k: v.data_ptr() for k, v in outside.items()} == outside_ptrs               # embedding, norm untouched
    assert causal.lm_head.weight is causal.model.embed_tokens.weight                   # tie preserved
    assert s.verify() == []
    assert s.summary()["dtype"] == "torch.float32" and s.summary()["tensors"] == len(before)


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("shape", list(F.SHAPES))
def test_store_backed_model_reproduces_the_saved_reference_bitwise(shape, storage, stores):
    model, saved = F.load(shape)
    s = build(model, storage, stores)
    for i, sl in enumerate(s.slabs):
        for e in sl.entries:
            key = next(k for k in saved["state"] if k.endswith(f"layers.{i}.{e.name}"))
            assert torch.equal(s.views(i)[e.name], saved["state"][key])
    with F.isolated():
        assert F.mismatches(F.reference(model, saved), saved["reference"]) == []


@needs_cuda
def test_pinned_store_pins_every_view_at_its_exact_size(stores):
    model, _ = F.load("4b-like")
    s = build(model, "pinned", stores)
    assert all(p.is_pinned() for p in frozen_decoder(model).values())
    assert all(v.is_pinned() for i in range(len(s.slabs)) for v in s.views(i).values())
    assert s.summary()["registered_bytes"] == sum(sl.raw.numel() for sl in s.slabs) == s.summary()["store_bytes"] + PAGE * len(s.slabs)


def test_pageable_store_pins_nothing(stores):
    model, _ = F.load("4b-like")
    build(model, "pageable", stores)
    assert not any(p.is_pinned() for p in frozen_decoder(model).values())


@pytest.mark.parametrize("storage", STORAGES)
def test_bf16_layers_store_with_fp32_adapters_left_out(storage, stores):
    model = tiny_bf16()
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    s = build(model, storage, stores)
    assert s.summary()["dtype"] == "torch.bfloat16" and s.verify() == []
    assert all(e.dtype == torch.bfloat16 for sl in s.slabs for e in sl.entries)


def test_a_layer_with_mixed_frozen_dtypes_is_refused():
    model = tiny_bf16()
    decoder_layers(model)[1].input_layernorm.weight.data = decoder_layers(model)[1].input_layernorm.weight.data.float()
    with pytest.raises(ValueError, match="one dtype"):
        WeightStore.build(model, "pageable")


# --- alignment and GPU copies ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("storage", STORAGES)
def test_every_view_is_256_byte_aligned_and_slabs_start_on_a_page(storage, stores):
    s = build(tiny_bf16(), storage, stores)
    assert all(sl.data.data_ptr() % PAGE == 0 for sl in s.slabs)
    assert all(v.data_ptr() % ALIGN == 0 for i in range(len(s.slabs)) for v in s.views(i).values())


@needs_cuda
def test_copy_to_gpu_buffers_reproduces_each_layer_and_stays_aligned(stores):
    s = build(tiny_bf16(), "pinned", stores)
    buffers = gpu_buffers(2, s.max_slab_bytes)
    for i in range(len(s.slabs)):                                                      # round-robin through 2 buffers
        views = s.copy_to(i, buffers[i % 2])
        assert all(v.is_cuda and v.data_ptr() % ALIGN == 0 for v in views.values())
        assert s.check_views(i, views) == []
        assert all(torch.equal(views[n].cpu(), c) for n, c in s.views(i).items())


# --- fault injection: verify() and check_views() must catch each ---------------------------------------------------------

@pytest.mark.parametrize("fault", ["repoint", "to_dtype", "flip_byte"])
def test_verify_detects_ownership_and_content_faults(fault, stores):
    model, _ = F.load("0.6b-like")
    s = build(model, "pageable", stores)
    layer = decoder_layers(model)[1]
    if fault == "repoint":
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data.clone()        # same values, elsewhere
    elif fault == "to_dtype":
        layer.to(torch.float64)
    else:
        s.slabs[2].data[0] ^= 1
    problems = s.verify()
    assert problems, fault
    expected = {"repoint": "layer 1 mlp.up_proj.weight: not a view", "to_dtype": "layer 1",
                "flip_byte": "layer 2 self_attn.q_proj.base_layer.weight: bytes differ"}[fault]
    assert any(p.startswith(expected) for p in problems), problems


@needs_cuda
def test_verify_detects_a_layer_moved_to_the_gpu(stores):
    model, _ = F.load("0.6b-like")
    s = build(model, "pinned", stores)
    decoder_layers(model)[0].to("cuda")
    assert any("on cuda" in p for p in s.verify())


@needs_cuda
def test_check_views_detects_a_wrong_layer_and_a_corrupted_buffer(stores):
    s = build(tiny_bf16(), "pinned", stores)
    buf = gpu_buffers(1, s.max_slab_bytes)[0]
    views = s.copy_to(2, buf)
    other = {n: v for n, v in s.copy_to(3, gpu_buffers(1, s.max_slab_bytes)[0]).items()}
    assert s.check_views(2, other)                                                     # layer 3's bytes, labelled 2
    buf[s.slabs[2].entries[0].offset] ^= 1
    assert s.check_views(2, views)[0].startswith("layer 2 self_attn.q_proj.weight: content differs")


# --- fail closed -------------------------------------------------------------------------------------------------------

def test_pinning_without_cuda_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, _ = F.load("0.6b-like")
    with pytest.raises(PinningError, match="needs a CUDA device"):
        WeightStore.build(model, "pinned")


@needs_cuda
def test_a_failed_registration_raises_and_releases_what_was_registered(monkeypatch):
    real = torch.cuda.cudart()
    calls = {"register": 0, "unregister": 0}

    class Flaky:
        cudaError = real.cudaError

        def cudaHostRegister(self, ptr, n, flags):
            calls["register"] += 1
            return real.cudaHostRegister(ptr, n, flags) if calls["register"] < 3 else "simulated cudaErrorMemoryAllocation"

        def cudaHostUnregister(self, ptr):
            calls["unregister"] += 1
            return real.cudaHostUnregister(ptr)

    monkeypatch.setattr(torch.cuda, "cudart", lambda: Flaky())
    model, _ = F.load("0.6b-like")
    with pytest.raises(PinningError, match="layer 2"):
        WeightStore.build(model, "pinned")
    assert calls == {"register": 3, "unregister": 2}                                   # layers 0-1 released


@needs_cuda
def test_a_store_dropped_without_close_unregisters_its_slabs():
    """Otherwise freed slab memory stays registered and a later allocation at that address reports is_pinned()."""
    import gc
    model, _ = F.load("0.6b-like")
    store = WeightStore.build(model, "pinned")
    views = [e.param for e in store.slabs[0].entries]  # the model keeps the slab memory alive
    assert all(p.is_pinned() for p in views)
    del store
    gc.collect()
    assert not any(p.is_pinned() for p in views)


@needs_cuda
def test_building_allocates_nothing_in_the_cuda_allocator(stores):
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    model, _ = F.load("4b-like")
    build(model, "pinned", stores)
    assert torch.cuda.memory_allocated() == before
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_close_is_idempotent_and_a_closed_store_refuses_use(stores):
    model, _ = F.load("0.6b-like")
    s = WeightStore.build(model, "pageable")
    s.close()
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.views(0)


@needs_cuda
def test_layers_on_the_gpu_cannot_be_adopted():
    model, _ = F.load("0.6b-like")
    decoder_layers(model)[0].self_attn.k_proj.to("cuda")
    with pytest.raises(ValueError, match="must be on the CPU"):
        WeightStore.build(model, "pageable")


# --- memory plan -----------------------------------------------------------------------------------------------------

def test_plan_matches_the_built_store_exactly(stores):
    model, _ = F.load("4b-like")
    p = plan(model, L2L, n_seq=6, seq_len=10, scored=4)
    s = build(model, "pageable", stores)
    assert p["slab_bytes"] == s.summary()["slab_bytes"] and p["max_slab_bytes"] == s.max_slab_bytes
    assert p["items"]["gpu.weight_buffers"]["bytes"] == 2 * s.max_slab_bytes
    lora = sum(q.numel() * 4 for q in model.parameters() if q.requires_grad)
    assert p["items"]["gpu.lora"]["bytes"] == p["items"]["gpu.lora_grads"]["bytes"] == lora
    assert p["items"]["gpu.adamw_state"]["bytes"] == 2 * lora
    assert p["tied_embeddings"] and p["items"]["gpu.embedding_and_head"]["bytes"] == 64 * 40 * 4
    assert {k for k, v in p["items"].items() if v["kind"] == "estimate"} == {"gpu.kv_cache", "gpu.boundary_checkpoints",
                                                                            "gpu.logits"}


def test_resident_layers_are_the_last_ones_and_leave_the_buffers_unchanged():
    model, _ = F.load("4b-like")
    p = plan(model, dataclasses.replace(L2L, resident_layers=2))
    assert p["resident_layers"] == [4, 5] and p["streamed_layers"] == [0, 1, 2, 3]
    assert p["items"]["gpu.resident_layers"]["bytes"] == sum(p["frozen_bytes"][4:])


@pytest.mark.parametrize("name", ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"])
def test_plan_on_the_real_skeleton_agrees_with_the_checkpoint_headers(name):
    """Two independent counts of every decoder layer: the plan (module shapes of a meta skeleton) and the Step-0
    architecture (safetensors headers in the local HF cache)."""
    from qwen_grpo.budget import architecture
    from qwen_grpo.model import skeleton

    try:
        arch, _ = architecture.load(name, "last2-attention")
    except architecture.MetadataError as e:  # only a genuinely unavailable header is a skip; any other error fails
        pytest.skip(f"checkpoint headers of {name} not available: {e}")
    sk = skeleton(name).requires_grad_(False)  # frozen as get_peft_model leaves the base
    p = plan(sk, L2L)
    elem = next(sk.parameters()).element_size()
    assert [b // elem for b in p["frozen_bytes"]] == [arch.layer_numel(i) for i in range(arch.num_hidden_layers)]
    assert p["tied_embeddings"]
