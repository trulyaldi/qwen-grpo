"""Synchronous L2L forward (Ticket 03): decoder layers run on the GPU with weights streamed from the CPU weight store.

Placement (place_resident): the embedding (= the tied LM head), the final norm, the rotary buffers and every trainable
(LoRA) parameter move to the device; the frozen decoder-layer weights stay in the WeightStore's CPU slabs. Parameters
are moved by replacing .data, so every Parameter object (optimizer references, the embedding/head tie) is unchanged.

StreamingLayerExecutor implements qwen3_forward's layer-executor interface. For decoder layer i it
  1. copies slab i into buffer i mod n of a fixed pool of n GPU buffers (WeightStore.copy_to, synchronous),
  2. re-points the layer's own frozen Parameters at their views of that buffer and checks that every one of them now
     lies inside the buffer (otherwise it raises: no layer ever runs on weights from anywhere else),
  3. calls the unchanged HF/PEFT decoder layer (LoRA runs inside it, on its resident parameters),
  4. re-points the frozen Parameters back at their CPU slab views, also when the call raises.
The layer's output is a new activation tensor, checked not to live in the buffer, so it survives the buffer's reuse.

Buffer reuse is safe because everything runs on one CUDA stream: the copy into buffer i mod n for layer i + n is
queued after layer i's kernels, and copy_to synchronizes after each copy.

Forward only: the executor refuses to run with grad mode enabled. Autograd would save views of a buffer that a later
layer overwrites, and backward would then use the wrong weights without any error (checkpointed recomputation is
Ticket 04).

Shapes: N rows, T columns, H hidden size.
"""
import contextlib

import torch

from qwen_grpo.execution.qwen3_forward import unwrap


def place_resident(model, store, device):
    """Move the non-streamed tensors of `model` to `device` (not the model: its frozen decoder weights stay in
    `store`). Raises if afterwards any tensor is neither on `device` nor a stored decoder weight."""
    decoder, lm_head = unwrap(model)
    for module in (decoder.embed_tokens, lm_head, decoder.norm, decoder.rotary_emb):
        module.to(device)  # in place: Module._apply replaces .data, the Parameter objects (and the tie) stay
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(device)
    device = torch.empty(0, device=device).device  # "cuda" -> "cuda:0", as tensors report it
    stored = {id(e.param) for s in store.slabs for e in s.entries}
    wrong = [n for n, p in model.named_parameters() if id(p) not in stored and p.device != device]
    wrong += [n for n, b in model.named_buffers() if b.device != device]
    if wrong:
        raise RuntimeError(f"tensors neither streamed nor on {device}: {wrong}")


def _inside(t, buffer):
    """t lies entirely inside `buffer` (same device, address range)."""
    start, end = buffer.data_ptr(), buffer.data_ptr() + buffer.numel()
    return t.device == buffer.device and start <= t.data_ptr() and t.data_ptr() + t.numel() * t.element_size() <= end


class StreamingLayerExecutor:
    """Runs decoder layer i on weights copied from store slab i into a fixed pool of buffers (see module docstring).

    trace: one record per weight load: the layer, the phase (forward, or backward recomputation), the buffer used, the
    slab copied and the PyTorch-allocated device memory just before the copy (None on CPU)."""

    def __init__(self, store, buffers):
        if not buffers:
            raise ValueError("at least one weight buffer")
        if any(b.dtype != torch.uint8 or b.numel() < store.max_slab_bytes for b in buffers):
            raise ValueError(f"every buffer must be uint8 with at least {store.max_slab_bytes} bytes (largest slab)")
        if len({b.device for b in buffers}) != 1:
            raise ValueError("all buffers must be on one device")
        if problems := store.verify():
            raise RuntimeError(f"the weight store is not intact: {problems[:5]}")
        self.store, self.buffers, self.trace = store, buffers, []

    def load(self, index, buffer):
        """{name: device view} of layer `index`'s weights in `buffer`."""
        return self.store.copy_to(index, buffer)

    @contextlib.contextmanager
    def bound(self, index, layer, phase="forward"):
        """Layer `index`'s frozen Parameters point into a pool buffer (loaded now) inside the block, and back at their
        CPU slab views after it, also on an exception. Yields the buffer. Anything that reads the weights (the layer
        call, and in backward the autograd.grad of a recomputed layer) must finish inside the block."""
        slab = self.store.slabs[index]
        params = dict(layer.named_parameters())
        if any(params.get(e.name) is not e.param for e in slab.entries):
            raise RuntimeError(f"layer {index} is not the module whose weights slab {index} holds")
        k = index % len(self.buffers)
        buffer = self.buffers[k]
        cuda = buffer.device.type == "cuda"
        self.trace.append({"layer": index, "phase": phase, "buffer": k, "slab": index, "slab_bytes": slab.nbytes,
                           "allocated_before": torch.cuda.memory_allocated(buffer.device) if cuda else None})
        views = self.load(index, buffer)
        try:
            for e in slab.entries:
                v = views.get(e.name)
                if v is None:
                    raise RuntimeError(f"layer {index} {e.name}: no such weight in the loaded buffer")
                if v.shape != e.param.shape or v.dtype != e.param.dtype:
                    raise RuntimeError(f"layer {index} {e.name}: view {v.dtype} {tuple(v.shape)} != parameter")
                e.param.data = v
            if outside := [e.name for e in slab.entries if not _inside(e.param, buffer)]:
                raise RuntimeError(f"layer {index}: weights outside the streaming buffer: {outside}")
            yield buffer
        finally:
            for e in slab.entries:
                e.param.data = slab.view(slab.data, e)

    def execute(self, index, layer, hidden, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError("StreamingLayerExecutor is forward-only: run under torch.no_grad() (autograd would save "
                               "views of a weight buffer that later layers overwrite; backward.py checkpoints instead)")
        with self.bound(index, layer) as buffer:
            out = layer(hidden, **kwargs)                                           # [N, T, H]
        if _inside(out, buffer):
            raise RuntimeError(f"layer {index}: output aliases the weight buffer")
        return out
