"""CPU ownership of the frozen decoder-layer weights of a Qwen3 (PEFT) model: one contiguous slab per decoder layer.

build() copies every frozen parameter of decoder layer i into slab i and re-points the parameter's .data at its view
of the slab, so the slab is the only copy the model references (the old storage is released when nothing else holds
it). storage="pinned" page-locks each slab with cudaHostRegister on its exact size, avoiding the CachingHostAllocator's
power-of-two rounding of pin_memory=True (the registered range is the slab's allocation: its exact size plus at
most one page of alignment slack); a registration failure raises and is never replaced by pageable memory.

Invariants (checked by verify(); tests/test_weight_store.py):
  I1  every frozen decoder-layer parameter is a view of its layer's slab, at the recorded offset, on CPU
  I2  its bytes equal the digest recorded at build (one-ulp changes are invisible in outputs: Ticket 00)
  I3  pinned: every parameter view is_pinned(); pageable: none is
  I4  trainable (LoRA) parameters, the embedding / tied LM head, the final norm and buffers are never in the store;
      their Parameter objects and storage are untouched (optimizer references stay valid)
  I5  every tensor offset in a slab is a multiple of ALIGN (256 B, cudaMalloc's alignment), slabs start on a page, so
      GPU views made by copy_to() are aligned like ordinary allocations (kernel selection can depend on alignment)
  I6  building allocates nothing in the PyTorch CUDA allocator (cudaHostRegister may create the CUDA context: that is
      recorded separately by callers, never folded into this invariant)
  I7  one dtype per layer slab (the frozen weights of a Qwen3 layer share the model dtype)
Lifetime: the store keeps each slab alive and unregisters it exactly once, in close() or, if close() is never called,
when the store is garbage-collected (memory freed while still registered stays page-locked in the driver, and a later
allocation at the same address then reports is_pinned(): found in Ticket 03). After close() the store refuses use.
A module moved with .to() or a parameter re-pointed elsewhere breaks I1 and is reported by verify().
"""
import hashlib
import weakref
from dataclasses import dataclass, field

import torch

ALIGN, PAGE = 256, 4096
STORAGE = ("pinned", "pageable")


class PinningError(RuntimeError):
    """Page-locking failed; the store is not built (no silent pageable fallback)."""


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()


def align(n, a=ALIGN):
    return (n + a - 1) // a * a


@dataclass
class Entry:
    name: str  # parameter name inside the decoder layer, e.g. self_attn.q_proj.base_layer.weight
    offset: int  # bytes from the slab start, a multiple of ALIGN
    shape: tuple
    dtype: torch.dtype
    nbytes: int
    sha256: str
    param: torch.nn.Parameter = field(repr=False)


@dataclass
class Slab:
    index: int
    data: torch.Tensor = field(repr=False)  # uint8 [nbytes], page-aligned view of `raw`
    raw: torch.Tensor = field(repr=False)  # the allocation that owns the memory
    entries: list
    dtype: torch.dtype
    registered: bool = False

    @property
    def nbytes(self):
        return self.data.numel()

    def view(self, base, e):
        return base[e.offset: e.offset + e.nbytes].view(e.dtype).view(e.shape)


def decoder_layers(model):
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    return causal_lm.model.layers[: causal_lm.config.num_hidden_layers]


def layout(layer):
    """[(name, param)] of the frozen parameters of one decoder layer, in module order, and their common dtype."""
    frozen = [(n, p) for n, p in layer.named_parameters() if not p.requires_grad]
    dtypes = {p.dtype for _, p in frozen}
    if not frozen:
        raise ValueError("a decoder layer has no frozen parameters: freeze the base model (get_peft_model does) first")
    if len(dtypes) != 1:
        raise ValueError(f"a layer slab holds one dtype; frozen parameters have {dtypes}")
    return frozen, dtypes.pop()


def slab_bytes(layer):
    frozen, _ = layout(layer)
    return sum(align(p.numel() * p.element_size()) for _, p in frozen)


class WeightStore:
    def __init__(self, slabs, storage):
        self.slabs, self.storage, self.closed = slabs, storage, False
        self._registered = []  # host pointers to unregister; the finalizer must not reference self
        self._finalizer = weakref.finalize(self, _unregister, self._registered)

    # --- construction ------------------------------------------------------------------------------------------------

    @classmethod
    def build(cls, model, storage="pinned", on_layer=None):
        """Adopt `model`'s frozen decoder weights (CPU) into per-layer slabs. Never pass the resident reference model:
        its parameters are re-pointed (values unchanged). on_layer(index) runs after each layer, e.g. to log RSS."""
        if storage not in STORAGE:
            raise ValueError(f"storage: one of {STORAGE}")
        if storage == "pinned" and not torch.cuda.is_available():
            raise PinningError("page-locking needs a CUDA device; none is available")
        slabs = []
        store = cls(slabs, storage)
        try:
            for i, layer in enumerate(decoder_layers(model)):
                slabs.append(store._adopt(i, layer))
                if on_layer is not None:
                    on_layer(i)
        except BaseException:
            store.close()
            raise
        return store

    def _adopt(self, i, layer):
        frozen, dtype = layout(layer)
        if any(p.device.type != "cpu" for _, p in frozen):
            raise ValueError(f"layer {i}: frozen weights must be on the CPU to be adopted")
        nbytes = sum(align(p.numel() * p.element_size()) for _, p in frozen)
        raw = torch.empty(nbytes + PAGE, dtype=torch.uint8)
        start = (-raw.data_ptr()) % PAGE
        slab = Slab(i, raw[start: start + nbytes], raw, [], dtype)
        if self.storage == "pinned":
            # the whole allocation: Tensor.is_pinned() looks up the storage's base pointer, not the view's data_ptr, so
            # registering only the page-aligned slab leaves every view reporting is_pinned() False (measured)
            rc = torch.cuda.cudart().cudaHostRegister(raw.data_ptr(), raw.numel(), 0)
            if rc != torch.cuda.cudart().cudaError.success:
                raise PinningError(f"cudaHostRegister of layer {i} ({raw.numel()} bytes) failed: {rc}")
            slab.registered = True
            self._registered.append(raw.data_ptr())
        offset = 0
        for name, p in frozen:
            n = p.numel() * p.element_size()
            e = Entry(name, offset, tuple(p.shape), p.dtype, n, "", p)
            view = slab.view(slab.data, e)
            view.copy_(p.data)
            p.data = view  # same Parameter object, storage now in the slab
            e.sha256 = digest(view)
            slab.entries.append(e)
            offset += align(n)
        return slab

    # --- use ---------------------------------------------------------------------------------------------------------

    def _open(self):
        if self.closed:
            raise RuntimeError("the weight store is closed")

    @property
    def max_slab_bytes(self):
        return max(s.nbytes for s in self.slabs)

    def views(self, i):
        """{name: CPU tensor} of layer i, views of its slab."""
        self._open()
        s = self.slabs[i]
        return {e.name: s.view(s.data, e) for e in s.entries}

    def copy_to(self, i, buffer):
        """Copy slab i into the uint8 GPU `buffer` (synchronous) and return {name: GPU view}. Only buffer[:nbytes]
        is written; the views are only valid until the buffer is reused."""
        self._open()
        s = self.slabs[i]
        if buffer.dtype != torch.uint8 or buffer.numel() < s.nbytes:
            raise ValueError(f"buffer of {buffer.numel()} bytes ({buffer.dtype}) cannot hold layer {i} ({s.nbytes} bytes)")
        buffer[: s.nbytes].copy_(s.data)
        if buffer.is_cuda:
            torch.cuda.synchronize(buffer.device)
        return {e.name: s.view(buffer, e) for e in s.entries}

    def check_views(self, i, views):
        """Problems if `views` (e.g. from copy_to) do not hold layer i's recorded bytes, or are misaligned."""
        s = self.slabs[i]
        out = []
        for e in s.entries:
            v = views.get(e.name)
            if v is None:
                out.append(f"layer {i} {e.name}: missing")
            elif digest(v) != e.sha256:
                out.append(f"layer {i} {e.name}: content differs from the stored digest")
            elif v.data_ptr() % ALIGN:
                out.append(f"layer {i} {e.name}: data_ptr not {ALIGN}-byte aligned")
        return out

    def verify(self):
        """Problems with I1-I3 and I5 for every stored parameter, now (not only at build time)."""
        self._open()
        out = []
        for s in self.slabs:
            base = s.data.data_ptr()
            for e in s.entries:
                p, where = e.param, f"layer {s.index} {e.name}"
                if p.device.type != "cpu":
                    out.append(f"{where}: on {p.device}, not in the CPU store")
                    continue
                if p.data_ptr() != base + e.offset:
                    out.append(f"{where}: not a view of its slab at offset {e.offset}")
                if (p.data_ptr() - base) % ALIGN:
                    out.append(f"{where}: offset not {ALIGN}-byte aligned")
                if digest(p.data) != e.sha256:
                    out.append(f"{where}: bytes differ from the digest recorded at build")
                if p.data.is_pinned() != (self.storage == "pinned"):
                    out.append(f"{where}: is_pinned() is {p.data.is_pinned()} for storage {self.storage}")
        return out

    def summary(self):
        return {"storage": self.storage, "layers": len(self.slabs), "slab_bytes": [s.nbytes for s in self.slabs],
                "store_bytes": sum(s.nbytes for s in self.slabs), "max_slab_bytes": self.max_slab_bytes,
                "registered_bytes": sum(s.raw.numel() for s in self.slabs if s.registered),
                "tensors": sum(len(s.entries) for s in self.slabs), "dtype": str(self.slabs[0].dtype) if self.slabs else None}

    def close(self):
        """Unregister every registered slab exactly once. Parameters keep pointing at (now pageable) slab memory."""
        if self.closed:
            return
        self._finalizer()  # runs _unregister once; never again at garbage collection
        for s in self.slabs:
            s.registered = False
        self.closed = True


def _unregister(pointers):
    for ptr in pointers:
        torch.cuda.cudart().cudaHostUnregister(ptr)
    pointers.clear()


def gpu_buffers(n, nbytes, device="cuda"):
    """n uint8 GPU buffers of nbytes each (the streamed-layer pool; L2L only ever needs n of them, whatever the depth)."""
    return [torch.empty(nbytes, dtype=torch.uint8, device=device) for _ in range(n)]
