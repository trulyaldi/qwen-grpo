"""H2D byte accounting and the H06 timeline metrics.

executor_bytes counts every weight copy an executor issued, asynchronous prefetches included; the counting that
scripts/profile_run.py used before 2026-11-19 (trace entries whose use was not a hit) misses every layer a prefetch had
already delivered. timeline_metrics derives overlap, stall, idle time, bytes, bandwidth and submit-to-ready latency
from GPU-timestamp intervals; it is pure, so it is tested here on hand-made timelines (the CUDA path that produces the
intervals is tests/test_prefetch.py's CUDA test)."""
import pytest
import torch

from qwen_grpo.execution.layer_stream import StreamingLayerExecutor
from qwen_grpo.execution.prefetch import PrefetchingLayerExecutor, executor_bytes, timeline_metrics
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.execution.weight_store import gpu_buffers
from qwen_grpo.forward_lab import inputs
from tests.test_prefetch import l2l_model


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def old_count(ex, since=0):
    return sum(t["slab_bytes"] for t in ex.trace[since:] if t.get("loaded", True))


def forward(cand, ex, passes):
    with torch.no_grad():
        for _ in range(passes):
            ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], "cpu")
            explicit_forward(cand, ids, mask, pos, layer_executor=ex)


def test_async_bytes_include_prefetches_and_the_old_trace_count_missed_them():
    _, cand, store = l2l_model()
    L, slab = len(store.slabs), [s.nbytes for s in store.slabs]
    ex = PrefetchingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"), 1)
    forward(cand, ex, passes=3)
    issued = ex.stats["prefetch_copies"] + ex.stats["on_demand_copies"]
    assert executor_bytes(ex) == ex.stats["bytes_copied"] > 0
    assert issued >= 3 * L - 1  # with 2 buffers every layer of every pass is copied once (the first may be prefetched)
    assert executor_bytes(ex) == sum(slab[i] for i, _, _ in ex.copy_log)
    assert old_count(ex) < executor_bytes(ex)  # prefetched layers were hits: the old count saw almost nothing
    assert ex.stats["hits"] > 0


def test_phase_deltas_sum_to_the_total_and_sync_counts_match_its_copies():
    _, cand, store = l2l_model()
    ex = PrefetchingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"), 1)
    marks = [executor_bytes(ex)]
    for _ in range(3):
        forward(cand, ex, passes=1)
        marks.append(executor_bytes(ex))
    assert sum(b - a for a, b in zip(marks, marks[1:])) == executor_bytes(ex)
    sync = StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"))
    forward(cand, sync, passes=2)
    assert executor_bytes(sync) == old_count(sync) == 2 * sum(s.nbytes for s in store.slabs)


def row(kind, layer, a, b, submit=None, nbytes=None):
    r = {"kind": kind, "layer": layer, "phase": "forward", "start_ms": a, "end_ms": b}
    if kind == "copy":
        r.update(submit_ms=submit, bytes=nbytes)
    return r


def test_timeline_metrics_overlap_stall_idle_bandwidth_and_latency():
    # layer 0 copied 0-10 then computed 10-20; layer 1 prefetched 10-25 (submitted at 10), so compute waits 20-25
    rows = [row("copy", 0, 0, 10, 0, 1000), row("compute", 0, 10, 20),
            row("copy", 1, 10, 25, 10, 1000), row("compute", 1, 25, 30),
            row("copy", 0, 20, 28, 20, 1000)]  # next pass's layer 0, fully overlapped with compute 1 (25-28)
    m = timeline_metrics(rows)
    assert m["copy_ms"] == 28 and m["compute_ms"] == 15
    assert m["overlap_ms"] == 10 + 3  # 10-20 with copy 1; 25-28 with copy 0
    assert m["compute_idle_ms"] == 5 and m["stall_on_copy_ms"] == 5  # 20-25: waiting for layer 1's weights
    assert m["bytes_copied"] == 3000 and m["h2d_bytes_per_s"] == pytest.approx(3000 / ((10 + 15 + 8) / 1000))
    assert m["submit_to_ready_ms_max"] == 15 and m["submit_to_ready_ms_median"] == 10
    assert m["window_ms"] == 30


def test_a_synchronous_timeline_has_no_overlap_and_all_idle_is_stall():
    rows = [row("copy", 0, 0, 10, 0, 10), row("compute", 0, 10, 20), row("copy", 1, 20, 30, 20, 10),
            row("compute", 1, 30, 35)]
    m = timeline_metrics(rows)
    assert m["overlap_ms"] == 0 and m["compute_idle_ms"] == 10 and m["stall_on_copy_ms"] == 10


def test_missing_byte_counts_give_none_not_zero():
    m = timeline_metrics([row("copy", 0, 0, 1), row("compute", 0, 1, 2)])
    assert m["bytes_copied"] is None and m["h2d_bytes_per_s"] is None and m["submit_to_ready_ms_median"] is None
