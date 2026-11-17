"""Asynchronous L2L prefetching (qwen_grpo/execution/prefetch.py) against the synchronous executor.

CPU tests run the same buffer-ownership logic on CPU buffers (immediate copies, no streams): outputs, gradients and
generated tokens must equal the synchronous executor's bitwise, every use must find its intended layer in the buffer
(verify mode), copies must follow the ownership rules, and an unsafe reuse must be caught. CUDA tests (skipped without
a GPU) add the transfer stream, events and the measured overlap.
"""
import pytest
import torch

from qwen_grpo.backward_lab import build, microbatches, placements, run_resident, step
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.generator import generate
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.prefetch import (PrefetchingCheckpointedExecutor, PrefetchingLayerExecutor, UnsafeReuse,
                                          _union)
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.forward_lab import inputs

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def l2l_model(layers=(0, 2, 5), shape="4b-like", dtype=torch.float32, device="cpu", vocab=64):
    with F.isolated():
        ref = build(shape, "sdpa", dtype, list(layers), vocab=vocab)
        cand = build(shape, "sdpa", dtype, list(layers), vocab=vocab)
    cand.load_state_dict(ref.state_dict())
    store = WeightStore.build(cand, "pinned" if device == "cuda" else "pageable")
    place_resident(cand, store, device)
    return ref.to(device), cand, store


class Audited(PrefetchingCheckpointedExecutor):
    """Checks the ownership rules at every copy: the target is never the slot in use (R3), and a slot is never
    overwritten while it holds a layer of the upcoming window."""

    def _copy(self, index, slot, kind):
        assert slot is not self._in_use or self._in_use is None, f"copy of layer {index} into the slot in use"
        super()._copy(index, slot, kind)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("depth, n_buf", [(1, 2), (1, 3), (2, 3), (3, 5)])
def test_forward_equals_the_synchronous_executor_over_repeated_passes(device, depth, n_buf):
    _, cand, store = l2l_model(device=device)
    sync = StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, device))
    ex = PrefetchingLayerExecutor(store, gpu_buffers(n_buf, store.max_slab_bytes, device), depth, verify=True)
    with torch.no_grad():
        for kind, T, lengths in [("left", 9, [9, 5, 9, 2]), ("unpadded", 5, None), ("left", 17, [17, 3, 11, 1])] * 3:
            ids, mask, pos = inputs(kind, T, lengths, device)
            assert torch.equal(explicit_forward(cand, ids, mask, pos, layer_executor=ex),
                               explicit_forward(cand, ids, mask, pos, layer_executor=sync))
    L = len(store.slabs)
    assert ex.stats["uses"] == 9 * L
    assert ex.stats["hits"] >= 8 * depth  # the wrap prefetch hands every later pass its first layers resident
    assert store.verify() == []


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("placement", ["final", "middle", "first_last", "distributed"])
@pytest.mark.parametrize("depth, n_buf", [(1, 2), (2, 4)])
def test_gradients_equal_the_synchronous_executor_with_micro_batch_accumulation(device, placement, depth, n_buf):
    layers = placements(6)[placement]
    ref, cand, store = l2l_model(layers, device=device)
    mbs = microbatches(device)
    with F.isolated():
        want = run_resident(ref, mbs)
    ex = Audited(store, gpu_buffers(n_buf, store.max_slab_bytes, device), depth, verify=True)
    cand.zero_grad(set_to_none=True)
    for k, mb in enumerate(mbs):
        with F.isolated():
            step(cand, ex, mb)
        for n, p in cand.named_parameters():
            if p.requires_grad:
                assert torch.equal(p.grad, want[k]["grads"][n]), (k, n)
    back = [t for t in ex.trace if t["phase"] == "backward"]
    assert [t["layer"] for t in back][:6 - min(layers)] == list(range(5, min(layers) - 1, -1))
    assert not back[0]["loaded"]  # the last forward layer is the first backward layer: a hit, no copy
    below = {i for i, kind, phase in ex.copy_log if phase == "backward" and i < min(layers)}
    assert below <= set(range(depth))  # below the lowest graph layer: only the next forward's first `depth` layers
    assert store.verify() == []


@pytest.mark.parametrize("device", DEVICES)
def test_generation_equals_the_synchronous_executor_and_decode_steps_find_layer_0_prefetched(device):
    _, cand, store = l2l_model(device=device)
    ids, mask, _ = inputs("left", 9, [9, 5, 9, 2], device)
    ids = ids.clamp(min=3).masked_fill(mask == 0, F.PAD)
    sync = StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, device))
    ex = PrefetchingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, device), 1, verify=True)
    with F.isolated():
        a = generate(cand, ids, mask, F.EOS_IDS, F.PAD, 8, 2, do_sample=False, layer_executor=sync, keep_logits=True)
        b = generate(cand, ids, mask, F.EOS_IDS, F.PAD, 8, 2, do_sample=False, layer_executor=ex, keep_logits=True)
    assert torch.equal(a["sequences"], b["sequences"])
    assert all(torch.equal(x, y) for x, y in zip(a["step_logits"], b["step_logits"]))
    L = len(store.slabs)
    first_uses = [t for t in ex.trace if t["layer"] == 0]
    assert len(first_uses) == b["steps"] and all(not t["loaded"] for t in first_uses[1:])
    assert ex.stats["on_demand_copies"] == 1  # only the very first layer of the very first step waits for its copy
    assert ex.stats["uses"] == L * b["steps"]


def test_a_complete_training_step_through_the_backend_is_bitwise_and_stress_repeated():
    from qwen_grpo.execution.backend import L2LBackend
    from qwen_grpo.training import grpo_update, prepare
    results = {}
    for depth in (0, 1, 2):
        with F.isolated():
            model, fx = F.build("4b-like")
        b = L2LBackend(model, "cpu", "pageable", 3, prefetch_depth=depth, verify=bool(depth))
        params = b.trainable()
        opt = torch.optim.AdamW(params.values(), lr=5e-2, weight_decay=0.0)
        with F.isolated():
            adv, old = prepare(model, fx["rollouts"], fx["rewards"], 2, logprobs=b.logprobs)
            logs = [grpo_update(model, params, opt, fx["rollouts"], adv, old, 0.2, k, 2, logprobs=b.logprobs)
                    for k in range(4)]
        results[depth] = ([log["loss"] for log in logs], {n: p.detach().clone() for n, p in params.items()})
        assert b.store.verify() == []
        b.close()
    for depth in (1, 2):
        assert results[depth][0] == results[0][0]
        assert all(torch.equal(results[depth][1][n], p) for n, p in results[0][1].items())


# --- fault injection ----------------------------------------------------------------------------------------------

def test_unsafe_buffer_reuse_is_caught_by_verify_and_by_the_comparison():
    _, cand, store = l2l_model()
    ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], "cpu")
    with torch.no_grad():
        want = explicit_forward(cand, ids, mask, pos, layer_executor=CheckpointedLayerExecutor(
            store, gpu_buffers(2, store.max_slab_bytes, "cpu")))
        with pytest.raises(RuntimeError, match="does not hold layer 2"):
            explicit_forward(cand, ids, mask, pos, layer_executor=UnsafeReuse(
                store, gpu_buffers(2, store.max_slab_bytes, "cpu"), at=2, verify=True))
        got = explicit_forward(cand, ids, mask, pos, layer_executor=UnsafeReuse(
            store, gpu_buffers(2, store.max_slab_bytes, "cpu"), at=2))
    assert not torch.equal(got, want)
    assert store.verify() == []  # the CPU-owned weights are never touched, whatever happens to a buffer


def test_refuses_too_few_buffers_and_depth_0():
    _, _, store = l2l_model()
    with pytest.raises(ValueError, match="cannot hold"):
        PrefetchingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"), depth=2)
    with pytest.raises(ValueError, match="depth >= 1"):
        PrefetchingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"), depth=0)


def test_interval_union():
    assert _union([(3, 4), (0, 1), (0.5, 2)]) == [(0, 2), (3, 4)]


# --- CUDA: streams, events, measured overlap ------------------------------------------------------------------------

@CUDA
def test_cuda_copies_run_on_the_transfer_stream_and_the_timeline_reports_measured_overlap():
    _, cand, store = l2l_model(shape="4b-like", device="cuda")
    ex = PrefetchingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cuda"), 1, timeline=True)
    ids, mask, pos = inputs("left", 17, [17, 3, 11, 1], "cuda")
    with torch.no_grad():
        for _ in range(3):
            explicit_forward(cand, ids, mask, pos, layer_executor=ex)
    rep = ex.report()
    assert rep["copies"] == ex.stats["on_demand_copies"] + ex.stats["prefetch_copies"]
    assert 0 <= rep["overlap_ms"] <= min(rep["copy_ms"], rep["compute_ms"])
    assert {r["kind"] for r in rep["intervals"]} == {"copy", "compute"}


class FakeEvent:
    """torch.cuda.Event stand-in on the CPU (smoke test of prefetch_timing's code paths: wall-clock timestamps)."""

    def __init__(self, enable_timing=False):
        self.t = None

    def record(self, stream=None):
        import time
        self.t = time.perf_counter()

    def elapsed_time(self, other):
        return (other.t - self.t) * 1000.0


def test_prefetch_timing_code_paths_run_end_to_end_on_cpu_with_fake_events(tmp_path, monkeypatch):
    """Not a measurement: the real case needs CUDA. A tiny Qwen3 (real tokenizer and data) through every part of the
    case with fake CUDA events, so that the GPU run on the laptop does not fail on a Python error."""
    from qwen_grpo import lab, prefetch_lab
    from tests.test_trainer import tiny_lora_model
    monkeypatch.setattr(prefetch_lab, "lora_model", tiny_lora_model)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(lab, "needs_l2l", lambda cfg, case: [])
    cfg = lab.ROOT / "research" / "l2l" / "configs" / "async_cpu.yaml"
    r = lab.run("prefetch_timing", cfg, "smoke", root=tmp_path, device="cpu", trials=5, warmup=1, max_new=3, smoke=True)
    assert r["execution_status"] == "PASS" and r["errors"] == [], r["reasons"][:5]
    m = r["measurements"]
    assert len(m["parts"]["candidate"]["step"]) == 5 and set(m["parts_ratio_ci"]) == set(prefetch_lab.PARTS)
    assert m["timeline_median"]["reference"]["step"]["copies"] > 0  # the synchronous side's (fake) timeline
    assert r["conclusion"]["what"].startswith("whole GRPO step")
