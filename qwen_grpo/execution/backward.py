"""Checkpointed L2L backward (Ticket 04): each decoder layer is one autograd node that keeps only its input.

Under grad mode, CheckpointedLayerExecutor runs decoder layer i as _CheckpointedLayer.apply(hidden_i, *lora_i):
  forward   streams the layer's frozen weights in (StreamingLayerExecutor.bound) and runs the unchanged HF/PEFT layer
            with grad disabled, so no graph and no view of a weight buffer is kept. The only tensor saved is the layer
            boundary hidden_i [N, T, H] (plus references to the layer's own LoRA Parameters, which cost nothing).
  backward  streams the same weights in again, recomputes the layer from hidden_i with grad enabled, and inside the
            same binding computes
                (dL/dhidden_i, dL/dlora_i) = autograd.grad(layer(hidden_i), (hidden_i, *lora_i), dL/dhidden_{i+1}).
            The frozen weights take part in dL/dhidden_i (through every projection, norm and residual) but receive
            no gradient. The recomputed graph and the buffer views are released before the next layer reuses a buffer.
The LoRA Parameters are explicit inputs of the node, so autograd accumulates the returned gradients into their .grad,
exactly as for any leaf: the optimizer's Parameter objects are used unchanged, and micro-batch accumulation works.

The recomputation equals the original forward bitwise (same weight bytes, same kernels: Ticket 03), so the gradients
are those of the resident graph whenever the resident backward is itself deterministic.

Fail-closed rules:
  - the recomputation must reproduce the forward: each node keeps an exact checksum of its forward output's bit
    patterns (int64 sum of the elements viewed as integers) and backward raises if the recomputed output's differs.
    This catches state that changed between forward and backward (an adapter disabled or switched, other weights);
    it is a checksum, not a proof of equality;
  - layers in training mode are refused: dropout (LoRA or attention) would draw a different mask in the recomputation.
    The model is never put in train mode and lora_config sets no dropout (model.py), so this costs nothing;
  - double backward raises (once_differentiable);
  - every LoRA Parameter of a layer that is reached by backward must get a gradient (autograd.grad with
    allow_unused=False raises otherwise);
  - layer keyword arguments (masks, rotary cos/sin) must not require grad: they are not differentiated;
  - with grad disabled the executor is the forward-only StreamingLayerExecutor (no node, nothing saved).

Shapes: N rows, T columns, H hidden size.
"""
import torch
from torch.autograd.function import once_differentiable

from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, _inside


def _tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (tuple, list)):
        for y in x:
            yield from _tensors(y)


_INT = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def checksum(t):
    """Exact int64 sum of the bit patterns of `t` (a 0-dim tensor on t's device; no synchronization)."""
    return t.detach().contiguous().view(_INT[t.element_size()]).sum(dtype=torch.int64)


class _CheckpointedLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, executor, index, layer, kwargs, hidden, *lora):
        ctx.executor, ctx.index, ctx.layer, ctx.kwargs = executor, index, layer, kwargs
        ctx.save_for_backward(hidden, *lora)  # the boundary activation, and the leaves to differentiate
        out = StreamingLayerExecutor.execute(executor, index, layer, hidden, **kwargs)  # grad is off in forward
        ctx.checksum = checksum(out)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        hidden, *lora = ctx.saved_tensors
        need_h, need_p = ctx.needs_input_grad[4], ctx.needs_input_grad[5:]
        h = hidden.detach().requires_grad_(need_h)
        wrt = ([h] if need_h else []) + [p for p, need in zip(lora, need_p) if need]
        with torch.enable_grad(), ctx.executor.bound(ctx.index, ctx.layer, phase="backward") as buffer:
            out = ctx.layer(h, **ctx.kwargs)                                           # [N, T, H], recomputed
            if not torch.equal(checksum(out), ctx.checksum):
                raise RuntimeError(f"layer {ctx.index}: the recomputed output differs from the forward's (model state "
                                   "changed between forward and backward?); refusing to return gradients")
            grads = list(torch.autograd.grad(out, wrt, grad_out, allow_unused=False))
            del out
            if aliased := [g for g in grads if _inside(g, buffer)]:
                raise RuntimeError(f"layer {ctx.index}: {len(aliased)} gradients alias the weight buffer")
        grad_h = grads.pop(0) if need_h else None
        grad_p = [grads.pop(0) if need else None for need in need_p]
        return (None, None, None, None, grad_h, *grad_p)


class CheckpointedLayerExecutor(StreamingLayerExecutor):
    """Layer executor for explicit_forward that is differentiable with respect to the LoRA parameters and the hidden
    states (see module docstring). Without grad mode it is the Ticket 03 forward-only executor."""

    def boundary(self, index, hidden):
        """The tensor saved for layer `index` (the input hidden state). A hook for fault injection only."""
        return hidden

    def execute(self, index, layer, hidden, **kwargs):
        if not torch.is_grad_enabled():
            return super().execute(index, layer, hidden, **kwargs)
        if graded := [t for v in kwargs.values() for t in _tensors(v) if t.requires_grad]:
            raise RuntimeError(f"layer {index}: {len(graded)} keyword tensors require grad; only the hidden state and "
                               "the LoRA parameters are differentiated")
        if any(m.training for m in layer.modules()):
            raise RuntimeError(f"layer {index} is in training mode: dropout would differ in the recomputation")
        lora = [p for _, p in layer.named_parameters() if p.requires_grad]
        return _CheckpointedLayer.apply(self, index, layer, kwargs, self.boundary(index, hidden), *lora)
