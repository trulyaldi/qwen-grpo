"""Asynchronous L2L prefetching (Ticket 08): weights of the next layers are copied on a dedicated CUDA transfer stream
while the current layer computes, into a fixed pool of reusable GPU buffers with explicit ownership.

PrefetchMixin replaces the synchronous executor's _acquire/_release (layer_stream.StreamingLayerExecutor.bound keeps
re-pointing the Parameters, checking that every weight lies in the buffer and restoring the CPU views):

Buffer ownership. Each pool buffer is a Slot: the layer whose bytes it holds (or is receiving), a `ready` event
(recorded on the transfer stream after its copy) and a `free` event (recorded on the compute stream after the last
block that read it). Rules, which together are the only synchronization in the pipeline:
  R1  a copy into a slot is enqueued on the transfer stream after transfer.wait_event(slot.free): it cannot start
      before every kernel that read the slot's previous layer has finished (no overwrite of weights in use);
  R2  compute uses a slot only after compute.wait_event(slot.ready): never before its copy has completed;
  R3  the slot in use is never chosen as a copy target, nor is a slot holding a layer of the upcoming window;
  R4  slot.free is recorded at _release, after the block's kernels (the layer call, or the backward recomputation and
      its autograd.grad) were all enqueued on the compute stream.
R1 and R3 are enforced structurally, always (not only in verify mode): every copy is validated against the slot
ownership state before it is enqueued (_claim) and an illegal target raises BufferReuseError without touching the
buffer. The check is host-side bookkeeping only (no synchronization). It is what catches an overwrite of a layer that
is never recomputed in backward (e.g. any layer below the lowest LoRA layer), where the recomputation checksum of
backward.py has nothing to compare.
No torch.cuda.synchronize() / stream.synchronize() is called in the pipeline (verify mode and timeline reports
synchronize, outside the timed pipeline; they are diagnostics).

Prefetch order. At _acquire(i) the next `depth` layers in execution order are prefetched into the least recently
used slots outside R3:
  forward    i+1, i+2, ... ; past the last layer, wrap to layer 0 (the next forward: a decode step, the next
             micro-batch), except in a pass that builds an autograd graph, which continues with backward
  backward   i-1, i-2, ... down to the lowest layer that has a node (layers below it are never recomputed), then 0, 1, ...
A layer already in a slot is a hit and is not copied again (the last forward layer is the first backward layer).

CPU buffers (tests, no CUDA): the same ownership logic with immediate copies and no streams or events.
verify=True: after R2 each use checks that the buffer's bytes are the intended layer's (an exact int64 checksum of
the slab, computed once on the CPU) and raises otherwise.
timeline=True (CUDA): timing events around every copy (transfer stream) and every block (compute stream); report()
returns the intervals and the measured overlap: the time during which a copy and a compute block are both running,
from the GPU timestamps (not inferred from the existence of two streams).

Shapes: L decoder layers, n pool buffers (n >= depth + 1).
"""
import contextlib
from dataclasses import dataclass

import torch

from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor


class BufferReuseError(RuntimeError):
    """A copy would overwrite a weight buffer that is in use or protected (R1/R3): refused before it is enqueued."""


@dataclass(eq=False)  # slots are compared by identity (they hold a tensor)
class Slot:
    buffer: torch.Tensor
    index: int | None = None          # layer whose bytes are in (or on their way into) the buffer
    ready: object = None              # torch.cuda.Event recorded after the copy (transfer stream)
    free: object = None               # torch.cuda.Event recorded after the last use (compute stream)
    last_use: int = -1


class PrefetchMixin:
    def _init_prefetch(self, depth=1, verify=False, timeline=False):
        if depth < 1:
            raise ValueError("prefetch depth >= 1 (depth 0 is the synchronous executor)")
        if len(self.buffers) < depth + 1:
            raise ValueError(f"{len(self.buffers)} buffers cannot hold the layer in use plus {depth} prefetched ones")
        self.depth, self.verify = depth, verify
        self.cuda = self.buffers[0].is_cuda
        self.copy_stream = torch.cuda.Stream(device=self.buffers[0].device) if self.cuda else None
        self.slots = [Slot(b) for b in self.buffers]
        self.stats = {"uses": 0, "hits": 0, "on_demand_copies": 0, "prefetch_copies": 0, "bytes_copied": 0}
        self._events = [] if (timeline and self.cuda) else None
        self._clock, self._in_use, self._phase = 0, None, None
        self.copy_log = []  # (layer, kind, phase of the use that issued it), in issue order
        self._graph_pass, self._graph_floor = False, None
        self._checksums = {}

    # --- ownership ----------------------------------------------------------------------------------------------------

    def _slot_of(self, index):
        return next((s for s in self.slots if s.index == index), None)

    def _victim(self, exclude):
        """The least recently used slot outside `exclude` (R3), empty slots first; None if there is none."""
        free = [s for s in self.slots if all(s is not x for x in exclude)]
        return min(free, key=lambda s: s.last_use) if free else None

    def _claim(self, index, slot, protected=()):
        """Ownership validation before any copy of layer `index` into `slot` (production; no synchronization):
          - the slot bound to the layer that is running (its forward, or its backward recomputation and autograd.grad)
            is never a target (R3);
          - nor is a slot in `protected`: the prefetch window being filled (R3);
          - on CUDA a slot that has had readers must carry their completion event, or the copy could not wait for them
            (R1).
        Raises BufferReuseError; the slot is left unchanged."""
        k = self.slots.index(slot)
        if slot is self._in_use:
            raise BufferReuseError(f"copy of layer {index} into buffer {k}, which is bound to layer {slot.index} "
                                   f"in use ({self._phase}): its weights would be overwritten while read (R3)")
        if any(slot is p for p in protected):
            raise BufferReuseError(f"copy of layer {index} into buffer {k}, which holds layer {slot.index} of the "
                                   f"prefetch window (R3)")
        if self.cuda and slot.last_use >= 0 and slot.free is None:
            raise BufferReuseError(f"copy of layer {index} into buffer {k}: the completion event of layer "
                                   f"{slot.index}'s readers is missing, the copy cannot wait for them (R1)")

    def _copy(self, index, slot, kind):
        """Copy layer `index` into `slot` on the transfer stream, after the slot's previous readers (R1)."""
        self._claim(index, slot)
        nbytes = self.store.slabs[index].nbytes
        if self.cuda:
            if slot.free is not None:
                self.copy_stream.wait_event(slot.free)
            with torch.cuda.stream(self.copy_stream):
                start = self._event()
                self.store.copy_nonblocking(index, slot.buffer)
                end = self._event()
                slot.ready = torch.cuda.Event()
                slot.ready.record(self.copy_stream)
            if self._events is not None:
                self._events.append(("copy", index, kind, start, end))
        else:
            self.store.copy_nonblocking(index, slot.buffer)
        slot.index = index
        self.copy_log.append((index, kind, self._phase))
        self.stats[f"{kind}_copies"] += 1
        self.stats["bytes_copied"] += nbytes

    def _event(self):
        if self._events is None:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def _upcoming(self, index, phase):
        L = len(self.store.slabs)
        if phase == "forward":
            seq = list(range(index + 1, L)) + ([] if self._graph_pass else list(range(L)))
        else:
            floor = self._graph_floor or 0
            seq = list(range(index - 1, floor - 1, -1)) + list(range(L))
        out = []
        for j in seq:
            if j != index and j not in out:
                out.append(j)
            if len(out) == self.depth:
                break
        return out

    def _prefetch(self, index, phase, current):
        upcoming = self._upcoming(index, phase)
        exclude = [current] + [s for s in self.slots if s.index in upcoming]
        for j in upcoming:
            if self._slot_of(j) is not None:
                continue
            victim = self._target(j, exclude)
            if victim is None:
                break
            self._claim(j, victim, exclude)
            self._copy(j, victim, "prefetch")
            exclude.append(victim)

    def _target(self, index, exclude):
        """The slot a prefetch of `index` goes to (a hook for fault injection)."""
        return self._victim(exclude)

    # --- the executor hooks --------------------------------------------------------------------------------------------

    def _acquire(self, index, phase):
        self._phase = phase
        slab = self.store.slabs[index]
        slot = self._slot_of(index)
        hit = slot is not None
        if not hit:
            upcoming = self._upcoming(index, phase)
            slot = self._victim([s for s in self.slots if s.index in upcoming]) or self._victim([])
            self._copy(index, slot, "on_demand")
        self.stats["uses"] += 1
        self.stats["hits"] += hit
        self._in_use = slot
        try:
            self._prefetch(index, phase, slot)
        except BaseException:  # a refused prefetch (BufferReuseError): bound() never enters its block, so no _release
            self._in_use = None
            raise
        if self.cuda and slot.ready is not None:
            torch.cuda.current_stream().wait_event(slot.ready)                     # R2
        if self.verify:
            self._check(index, slot)
        k = self.slots.index(slot)
        self.trace.append({"layer": index, "phase": phase, "buffer": k, "slab": index, "slab_bytes": slab.nbytes,
                           "loaded": not hit,
                           "allocated_before": torch.cuda.memory_allocated(slot.buffer.device) if self.cuda else None})
        if self._events is not None:
            self._compute_start = self._event()
        return slot.buffer, {e.name: slab.view(slot.buffer, e) for e in slab.entries}

    def _release(self, index, phase):
        slot, self._in_use = self._in_use, None
        if self.cuda:
            slot.free = torch.cuda.Event()
            slot.free.record()                                                      # R4: on the compute stream
            if self._events is not None:
                self._events.append(("compute", index, phase, self._compute_start, self._event()))
        slot.last_use = self._clock
        self._clock += 1

    def execute(self, index, layer, hidden, **kwargs):
        """Records whether this pass builds a graph (decides the forward wrap) and its lowest graph layer (where the
        backward prefetch stops)."""
        if index == 0:
            self._graph_pass, self._graph_floor = torch.is_grad_enabled(), None
        out = super().execute(index, layer, hidden, **kwargs)
        if self._graph_pass and out.requires_grad and self._graph_floor is None:
            self._graph_floor = index
        return out

    # --- diagnostics -----------------------------------------------------------------------------------------------------

    def _check(self, index, slot):
        """The slot's bytes are layer `index`'s (exact int64 checksum; synchronizes: verify mode only)."""
        s = self.store.slabs[index]
        if index not in self._checksums:
            self._checksums[index] = int(s.data.view(torch.int64).sum())
        got = int(slot.buffer[: s.nbytes].view(torch.int64).sum())
        if got != self._checksums[index] or slot.index != index:
            raise RuntimeError(f"buffer {self.slots.index(slot)} does not hold layer {index}'s weights "
                               f"(slot says layer {slot.index}, checksum {got} != {self._checksums[index]})")

    def report(self, clear=True):
        """Timeline (CUDA, timeline=True): see timeline_report. Synchronizes (diagnostic, outside the pipeline)."""
        if self._events is None:
            return None
        out = timeline_report(self._events)
        if clear:
            self._events = []
        return out


def timeline_report(events):
    """[(kind 'copy'|'compute', layer, phase, start event, end event)] -> intervals in ms from the first event, the
    union lengths of copy time and compute time, and the measured overlap: the time during which at least one copy
    and at least one compute block run at once, from the GPU timestamps. Synchronizes."""
    torch.cuda.synchronize()
    events = [e for e in events if e[3] is not None]
    if not events:
        return {"intervals": [], "copy_ms": 0.0, "compute_ms": 0.0, "overlap_ms": 0.0, "copies": 0,
                "overlap_fraction_of_copy": 0.0}
    t0 = min((e[3] for e in events), key=lambda ev: events[0][3].elapsed_time(ev))
    rows = [{"kind": k, "layer": i, "phase": ph, "start_ms": t0.elapsed_time(a), "end_ms": t0.elapsed_time(b)}
            for k, i, ph, a, b in events]
    copies = _union([(r["start_ms"], r["end_ms"]) for r in rows if r["kind"] == "copy"])
    compute = _union([(r["start_ms"], r["end_ms"]) for r in rows if r["kind"] == "compute"])
    overlap = sum(max(0.0, min(a1, b1) - max(a0, b0)) for a0, a1 in copies for b0, b1 in compute)
    out = {"intervals": rows, "copy_ms": sum(b - a for a, b in copies), "compute_ms": sum(b - a for a, b in compute),
           "overlap_ms": overlap, "copies": sum(r["kind"] == "copy" for r in rows)}
    out["overlap_fraction_of_copy"] = overlap / out["copy_ms"] if out["copy_ms"] else 0.0
    return out


def _union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


class PrefetchingLayerExecutor(PrefetchMixin, StreamingLayerExecutor):
    """Forward-only (no_grad) L2L executor with asynchronous prefetching."""

    def __init__(self, store, buffers, depth=1, verify=False, timeline=False):
        StreamingLayerExecutor.__init__(self, store, buffers)
        self._init_prefetch(depth, verify, timeline)


class PrefetchingCheckpointedExecutor(PrefetchMixin, CheckpointedLayerExecutor):
    """Differentiable L2L executor (checkpointed backward, reverse-order prefetch) with asynchronous prefetching."""

    def __init__(self, store, buffers, depth=1, verify=False, timeline=False):
        CheckpointedLayerExecutor.__init__(self, store, buffers)
        self._init_prefetch(depth, verify, timeline)


class UnsafeReuse(PrefetchingCheckpointedExecutor):
    """Fault injection: when layer `at` is acquired, the next layer's prefetch targets the slot in use and drops the
    event of its readers (R1, R3 broken). Without protection the compute stream would wait for that copy, so the
    overwrite would always happen before layer `at` reads its weights (deterministic, unlike a real race). The
    ownership check (_claim) refuses the copy with BufferReuseError before it is enqueued."""

    def __init__(self, store, buffers, at, depth=1, verify=False, timeline=False):
        super().__init__(store, buffers, depth, verify, timeline)
        self.at, self.injected = at, 0  # injected: how often the fault fired (0: this run never exercised it)

    def _target(self, index, exclude):
        current = exclude[0]
        if current is not None and current.index == self.at:
            self.injected += 1
            current.free = None                                                     # forget its readers (R1 broken)
            return current                                                          # the slot in use (R3 broken)
        return self._victim(exclude)


@contextlib.contextmanager
def no_prefetch_timeline(executor):
    """Temporarily disable timeline recording (e.g. during warmup)."""
    saved = getattr(executor, "_events", None)
    executor._events = None
    try:
        yield
    finally:
        executor._events = saved


def is_async(execution):
    return execution.backend == "l2l_async"


def make_executor(execution, store, buffers, checkpointed, verify=False):
    """The layer executor config.Execution selects: l2l_sync -> the synchronous executors (Tickets 03-04), l2l_async ->
    the prefetching ones with execution.prefetch_depth."""
    if is_async(execution):
        cls = PrefetchingCheckpointedExecutor if checkpointed else PrefetchingLayerExecutor
        return cls(store, buffers, execution.prefetch_depth, verify=verify)
    return (CheckpointedLayerExecutor if checkpointed else StreamingLayerExecutor)(store, buffers)


def buffer_counts(execution, sync_counts):
    """Pool sizes a lab case sweeps: the synchronous ones, or depth + 1 and depth + 2 for the prefetching executor."""
    return (execution.prefetch_depth + 1, execution.prefetch_depth + 2) if is_async(execution) else sync_counts
