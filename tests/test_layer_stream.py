"""Synchronous L2L forward (qwen_grpo/execution/layer_stream.py) against resident execution, on tiny random Qwen3 models.

Tolerance 0 throughout (research/l2l/tolerances.yaml, forward_equivalence): same device, same weight bytes, same
kernels. CPU tests use a pageable store and CPU buffers (the same executor code); CUDA tests use the pinned store and GPU
buffers and are skipped without a GPU. The full grid with per-layer errors is the lab case forward_equivalence.
"""
import pytest
import torch

from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.qwen3_forward import explicit_forward, unwrap
from qwen_grpo.execution.weight_store import WeightStore, digest, gpu_buffers
from qwen_grpo.forward_lab import INPUTS, Residency, WrongLayer, build, inputs

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


def pair(shape="0.6b-like", attn="eager", dtype=torch.float32, lora="all", device="cpu"):
    """(resident reference on device, L2L candidate with the same weights, its store)."""
    with F.isolated():
        ref, cand = build(shape, attn, dtype, lora), build(shape, attn, dtype, lora)
    cand.load_state_dict(ref.state_dict())
    store = WeightStore.build(cand, "pinned" if device == "cuda" else "pageable")
    place_resident(cand, store, device)
    return ref.to(device), cand, store


def run(model, ex, ids, mask, pos, keep=0):
    with torch.no_grad():
        return explicit_forward(model, ids, mask, pos, keep, layer_executor=ex)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", list(F.SHAPES))
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lora", ["plain", "last2-attention", "all"])
def test_l2l_logits_equal_resident(device, shape, attn, dtype, lora):
    ref, cand, store = pair(shape, attn, dtype, lora, device)
    for n_buf in (1, 2):
        ex = StreamingLayerExecutor(store, gpu_buffers(n_buf, store.max_slab_bytes, device))
        for kind, T, lengths in INPUTS:
            ids, mask, pos = inputs(kind, T, lengths, device)
            for keep in (0, 2, torch.tensor([0, T - 1], device=device)):
                with torch.no_grad():
                    hf = ref(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                             logits_to_keep=keep).logits
                assert torch.equal(run(cand, ex, ids, mask, pos, keep), hf), (n_buf, kind, T, keep)
        assert [t["buffer"] for t in ex.trace][:3] == [0, 1 % n_buf, 2 % n_buf]
    assert store.verify() == []
    store.close()


@pytest.mark.parametrize("device", DEVICES)
def test_adapter_disabled_and_lora_has_effect(device):
    ref, cand, store = pair(device=device)
    ex = StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, device))
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], device)
    with torch.no_grad(), ref.disable_adapter(), cand.disable_adapter():
        off_ref = ref(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
        off = run(cand, ex, ids, mask, pos)
    assert torch.equal(off, off_ref)
    assert (run(cand, ex, ids, mask, pos) - off).abs().max() > 1e-2  # the adapters really run inside streamed layers


@pytest.mark.parametrize("device", DEVICES)
def test_kernels_read_weights_from_the_buffer_pool_only(device):
    _, cand, store = pair(device=device)
    buffers = gpu_buffers(2, store.max_slab_bytes, device)
    residency = Residency(cand, store, buffers)
    ex = StreamingLayerExecutor(store, buffers)
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], device)
    with torch.no_grad():
        explicit_forward(cand, ids, mask, pos, layer_executor=ex, layer_callback=lambda i, h: residency.after_layer(i))
    assert residency.violations == [] and residency.unseen() == []
    assert all(e.param.device.type == "cpu" for s in store.slabs for e in s.entries)  # back in the store afterwards


@CUDA
def test_allocated_memory_is_the_same_before_every_layer_and_after_the_forward():
    _, cand, store = pair(shape="4b-like", device="cuda")
    ex = StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cuda"))
    ids, mask, pos = inputs("left", 17, [17, 3, 11, 1], "cuda")
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = run(cand, ex, ids, mask, pos)
    del out
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert len({t["allocated_before"] for t in ex.trace[1:]}) == 1


# --- fault injection: each must be caught -------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_wrong_layer_weights_are_detected(device):
    ref, cand, store = pair(device=device)
    ex = WrongLayer(store, gpu_buffers(2, store.max_slab_bytes, device), wrong=1)  # passes verify(): loads are later
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], device)
    ref_layers, got_layers = [], []
    with torch.no_grad():
        explicit_forward(ref, ids, mask, pos, layer_callback=lambda i, h: ref_layers.append(h.clone()))
        explicit_forward(cand, ids, mask, pos, layer_executor=ex, layer_callback=lambda i, h: got_layers.append(h.clone()))
    assert torch.equal(got_layers[0], ref_layers[0])
    assert all(not torch.equal(g, r) for g, r in zip(got_layers[1:], ref_layers[1:]))  # localized to layer 1 onward


class CorruptBuffer(StreamingLayerExecutor):
    def load(self, index, buffer):
        views = super().load(index, buffer)
        if index == 2:
            w = views["mlp.down_proj.base_layer.weight"]
            w.view(-1)[0] += 0.1 * w.abs().max()  # visible: a 1-ulp change is not (Ticket 00)
        return views


@pytest.mark.parametrize("device", DEVICES)
def test_corrupted_buffer_after_the_copy_is_detected(device):
    ref, cand, store = pair(device=device)
    ex = CorruptBuffer(store, gpu_buffers(2, store.max_slab_bytes, device))
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], device)
    with torch.no_grad():
        hf = ref(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False).logits
    assert not torch.equal(run(cand, ex, ids, mask, pos), hf)
    assert store.verify() == []  # the store itself was not touched: only the comparison sees this fault


# --- fail closed --------------------------------------------------------------------------------------------------

def cpu_executor(lora="all"):
    _, cand, store = pair(lora=lora)
    return cand, store, StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"))


def test_refuses_grad_mode():
    cand, _, ex = cpu_executor()
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], "cpu")
    with pytest.raises(RuntimeError, match="forward-only"):
        explicit_forward(cand, ids, mask, pos, layer_executor=ex)


def test_refuses_a_changed_store_a_closed_store_and_small_buffers():
    cand, store, _ = cpu_executor()
    with pytest.raises(ValueError, match="largest slab"):
        StreamingLayerExecutor(store, gpu_buffers(1, store.max_slab_bytes - 1, "cpu"))
    layer = unwrap(cand)[0].layers[0]
    layer.to(torch.float64)  # a module conversion re-points its weights out of the store
    with pytest.raises(RuntimeError, match="not intact"):
        StreamingLayerExecutor(store, gpu_buffers(1, store.max_slab_bytes, "cpu"))
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        StreamingLayerExecutor(store, gpu_buffers(1, store.max_slab_bytes, "cpu"))


def test_refuses_a_layer_that_does_not_own_the_slab_and_a_mismatched_layout():
    cand, store, ex = cpu_executor(lora="last2-attention")  # layers 0-1 plain, 2-3 with adapters
    layers = unwrap(cand)[0].layers
    hidden = torch.zeros(1, 2, cand.config.hidden_size)
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="not the module"):
            ex.execute(0, layers[1], hidden)
        with pytest.raises(RuntimeError, match="no such weight"):
            WrongLayer(store, ex.buffers, wrong=1).execute(1, layers[1], hidden)  # slab 2 has LoRA base_layer names
    assert store.verify() == []  # restored after the refusal


def test_weights_return_to_the_store_when_the_layer_raises(monkeypatch):
    cand, store, ex = cpu_executor()
    layer = unwrap(cand)[0].layers[0]
    monkeypatch.setattr(layer, "forward", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with torch.no_grad(), pytest.raises(ValueError, match="boom"):
        ex.execute(0, layer, torch.zeros(1, 2, cand.config.hidden_size))
    assert store.verify() == []


# --- placement and identities -------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_placement_keeps_parameter_identities_and_the_tie(device):
    with F.isolated():
        model = build("4b-like", "eager", torch.float32, "all")
    lora = {n: p for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(lora.values())
    store = WeightStore.build(model, "pinned" if device == "cuda" else "pageable")
    place_resident(model, store, device)
    decoder, head = unwrap(model)
    assert head.weight is decoder.embed_tokens.weight
    assert {n: p for n, p in model.named_parameters() if p.requires_grad} == lora
    assert all(p.device.type == device for p in lora.values())
    assert [p for g in optimizer.param_groups for p in g["params"]] == list(lora.values())
    before = {n: digest(p) for n, p in lora.items()}
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], device)
    run(model, StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, device)), ids, mask, pos)
    assert {n: digest(p) for n, p in lora.items()} == before
    assert store.verify() == []


def test_place_resident_refuses_unstored_frozen_weights():
    with F.isolated():
        model = build("0.6b-like", "eager", torch.float32, "plain")
    store = WeightStore.build(model, "pageable")
    store.slabs.pop()  # the last layer is no longer streamed: its weights are neither stored nor placed
    with pytest.raises(RuntimeError, match="neither streamed nor on"):
        place_resident(model, store, torch.device("meta"))
