"""Ticket 03 lab case: forward_equivalence (H01), the synchronous L2L forward against resident execution.

For every model of the grid (fixture shapes x attention x dtype x LoRA variant) a resident reference (the whole model on
the device) and a candidate with identical weights (frozen decoder weights in the WeightStore, the rest placed by
layer_stream.place_resident) run the same inputs, for 1 and 2 weight buffers:
  logits.<tag>      HF resident logits vs synchronous L2L logits
  layers.<tag>.<i>  explicit resident vs L2L output of decoder layer i
  reference_integrity: the explicit resident forward must equal HF bitwise; on the CPU, L2L on the Ticket 00 fixture
                    models must reproduce the saved fixture logits (ties this case to the frozen reference).
Residency is checked, not logged: forward pre-hooks on every module that owns frozen decoder weights record where the
weights are when the kernels run; violations are errors (execution FAIL). Faults (perturbation "wrong_layer=<i>") load
layer i+1's slab for layer i, after the store's verify(): only the comparisons can catch it. Under l2l_async the fault
"unsafe_reuse=<i>" (prefetch.UnsafeReuse) must be refused by the production ownership check (BufferReuseError,
recorded under "detections"); other exceptions stay errors, and a configuration where the fault was neither refused
nor caught by a comparison is an error ("not detected").
"""
import torch
from peft import get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import lab
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.prefetch import BufferReuseError, UnsafeReuse, buffer_counts, is_async, make_executor
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.execution.weight_store import WeightStore, decoder_layers, gpu_buffers
from qwen_grpo.model import PRESETS, lora_config, lora_modules, target_modules

ATTN = ("eager", "sdpa")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
LORA = ("plain", "disabled", "last2-attention", "all")  # disabled: the "all" adapters under disable_adapter()
ALL_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BUFFERS = (1, 2)
INPUTS = [("unpadded", 1, None), ("unpadded", 5, None), ("unpadded", 9, None), ("left", 9, [9, 5, 9, 2]),
          ("left", 17, [17, 3, 11, 1])]
KEEPS = {("left", 9): [3, [0, 4, 8]]}  # extra logit selections (int: last k columns; list: index tensor)


def build(shape, attn, dtype, lora):
    """Seeded tiny Qwen3; LoRA B ~ N(0, LORA_B_STD) so adapters change the output. Call inside fixtures.isolated()."""
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=F.VOCAB, tie_word_embeddings=True, attn_implementation=attn,
                                     **F.SHAPES[shape])).to(dtype).eval()
    if lora == "plain":
        return m.requires_grad_(False)
    n = m.config.num_hidden_layers
    targets = target_modules(m, *PRESETS["last2-attention"]) if lora == "last2-attention" else target_modules(m, ALL_PROJ, n)
    m = get_peft_model(m, lora_config(targets, F.RANK, F.ALPHA)).eval()
    for _, mod in lora_modules(m):
        torch.nn.init.normal_(mod.lora_B["default"].weight, std=F.LORA_B_STD)
    return m


def inputs(kind, T, lengths, device):
    """(ids [N, T], mask [N, T] or None, position_ids [N, T] or None); left padding uses cumsum(mask) - 1."""
    g = torch.Generator().manual_seed(T)
    if kind == "unpadded":
        return torch.randint(0, F.VOCAB, (3, T), generator=g).to(device), None, None
    mask = torch.stack([torch.arange(T) >= T - n for n in lengths]).long()
    ids = torch.randint(0, F.VOCAB, (len(lengths), T), generator=g).masked_fill(mask == 0, F.PAD)
    return ids.to(device), mask.to(device), (mask.cumsum(1) - 1).clamp(min=0).to(device)


class WrongLayer(StreamingLayerExecutor):
    """Fault injection: layer `wrong` runs on layer wrong + 1's weights (same shapes, so nothing raises)."""

    def __init__(self, store, buffers, wrong):
        super().__init__(store, buffers)
        self.wrong = wrong

    def load(self, index, buffer):
        return self.store.copy_to((index + 1) % len(self.store.slabs) if index == self.wrong else index, buffer)


class Residency:
    """Forward pre-hooks on every module that directly owns frozen decoder weights: at each call, every such weight must
    lie in the buffer pool. After each layer (layer_callback), its weights must be back at their slab offsets."""

    def __init__(self, model, store, buffers):
        self.store, self.pool, self.violations, self.seen = store, buffers, [], set()
        self.handles = []
        for i, layer in enumerate(decoder_layers(model)):
            for name, module in layer.named_modules():
                frozen = [(n, p) for n, p in module.named_parameters(recurse=False) if not p.requires_grad]
                if frozen:
                    self.handles.append(module.register_forward_pre_hook(self._hook(i, name, frozen)))

    def _hook(self, i, name, frozen):
        def check(module, args):
            for n, p in frozen:
                self.seen.add((i, f"{name}.{n}" if name else n))
                if not any(_in(p, b) for b in self.pool):
                    self.violations.append(f"layer {i} {name}.{n}: kernel input on {p.device} outside the buffer pool")
        return check

    def after_layer(self, i):
        s = self.store.slabs[i]
        for e in s.entries:
            if e.param.device.type != "cpu" or e.param.data_ptr() != s.data.data_ptr() + e.offset:
                self.violations.append(f"layer {i} {e.name}: not back in its CPU slab after the call")

    def unseen(self):
        return [f"layer {s.index} {e.name}: never observed at a kernel call" for s in self.store.slabs for e in s.entries
                if (s.index, e.name) not in self.seen]

    def remove(self):
        for h in self.handles:
            h.remove()


def _in(t, buffer):
    lo = buffer.data_ptr()
    return t.device == buffer.device and lo <= t.data_ptr() < lo + buffer.numel()


def executor(store, buffers, perturbation, cfg):
    """l2l_sync: StreamingLayerExecutor (fault wrong_layer=<i>); l2l_async: the prefetching executor with verify mode
    (fault unsafe_reuse=<i>, without verify: the diagnostic checksum is off, so only the production ownership check
    or the comparisons can catch it)."""
    if perturbation is None:
        return make_executor(cfg.execution, store, buffers, checkpointed=False, verify=is_async(cfg.execution))
    kind, value = perturbation.split("=")
    if is_async(cfg.execution) and kind == "unsafe_reuse":
        return UnsafeReuse(store, buffers, int(value), cfg.execution.prefetch_depth)
    if not is_async(cfg.execution) and kind == "wrong_layer":
        return WrongLayer(store, buffers, int(value))
    raise lab.SetupError(f"forward_equivalence supports wrong_layer=<i> (l2l_sync) and unsafe_reuse=<i> (l2l_async), "
                         f"not {perturbation!r} under {cfg.execution.backend}")


def l2l(model, store, ex, residency, ids, mask, pos, keep, layers=None):
    def callback(i, h):
        residency.after_layer(i)
        if layers is not None:
            layers.append(h.clone())
    return explicit_forward(model, ids, mask, pos, keep, layer_executor=ex, layer_callback=callback)


def memory_trace(model, store, ex, residency, device):
    """PyTorch-allocated device memory before each layer's copy and around one whole forward (outputs freed)."""
    ids, mask, pos = inputs("left", 17, [17, 3, 11, 1], device)
    ex.trace.clear()
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    out = l2l(model, store, ex, residency, ids, mask, pos, 0)
    del out
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_allocated(device)
    per_layer = [t["allocated_before"] for t in ex.trace]
    problems = []
    if len(set(per_layer[1:])) > 1:
        problems.append(f"allocated memory before the layer copies differs across layers >= 1: {per_layer}")
    if after != before:
        problems.append(f"allocated memory after the forward {after} != before {before}")
    return {"before_forward": before, "after_forward": after, "before_each_layer": per_layer,
            "buffers": [t["buffer"] for t in ex.trace]}, problems


def run_model(cfg, result, shape, attn, dt, lora, device, perturbation, tol):
    dtype = DTYPES[dt]
    with F.isolated():
        ref = build(shape, attn, dtype, lora)
        cand = build(shape, attn, dtype, lora)
    cand.load_state_dict(ref.state_dict(), strict=True)
    ref.to(device)  # the reference only; the candidate is never moved as a whole
    store = WeightStore.build(cand, cfg.execution.weight_storage)
    try:
        place_resident(cand, store, device)
        lora_ids = {n: id(p) for n, p in cand.named_parameters() if p.requires_grad}
        off = (lambda m: m.disable_adapter()) if lora == "disabled" else (lambda m: torch.no_grad())
        for n_buf in buffer_counts(cfg.execution, BUFFERS):
            buffers = gpu_buffers(n_buf, store.max_slab_bytes, device)
            residency = Residency(cand, store, buffers)
            ex = executor(store, buffers, perturbation, cfg)
            tag = f"{shape}.{attn}.{dt}.{lora}.b{n_buf}"
            try:
                run_buffers(result, ref, cand, store, ex, residency, off, tag, shape, attn, dt, lora, n_buf, device,
                            perturbation, tol)
            except BufferReuseError as e:  # the ownership check: the expected detection of unsafe_reuse only
                lab.record_refusal(result, perturbation, "unsafe_reuse", tag, e)
            residency.remove()
            del buffers, ex, residency
        if {n: id(p) for n, p in cand.named_parameters() if p.requires_grad} != lora_ids:
            result["errors"].append(f"{shape}.{attn}.{dt}.{lora}: LoRA Parameter identities changed")
        if problems := store.verify():
            result["errors"] += [f"{shape}.{attn}.{dt}.{lora}: store: {p}" for p in problems[:5]]
    finally:
        store.close()


def run_buffers(result, ref, cand, store, ex, residency, off, tag, shape, attn, dt, lora, n_buf, device, perturbation,
                tol):
    """One model and pool size: the comparisons, residency checks, memory trace and LoRA-power check."""
    ref_t, got_t = {}, {}
    with torch.no_grad(), off(ref), off(cand):
        for kind, T, lengths in INPUTS:
            ids, mask, pos = inputs(kind, T, lengths, device)
            for keep in [0] + KEEPS.get((kind, T), []):
                k = torch.tensor(keep, device=device) if isinstance(keep, list) else keep
                key = f"{tag}.{kind}{T}.k{keep if isinstance(keep, int) else 'idx'}"
                hf = ref(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False,
                         logits_to_keep=k).logits
                ref_layers, got_layers = [], []
                explicit = explicit_forward(ref, ids, mask, pos, k,
                                            layer_callback=lambda i, h: ref_layers.append(h.clone()))
                if not torch.equal(explicit, hf):
                    result["reference_integrity"].append(f"{key}: explicit resident != HF")
                got = l2l(cand, store, ex, residency, ids, mask, pos, k, got_layers)
                ref_t[f"logits.{key}"], got_t[f"logits.{key}"] = hf, got
                for i, (r, g) in enumerate(zip(ref_layers, got_layers)):
                    ref_t[f"layers.{key}.{i}"], got_t[f"layers.{key}.{i}"] = r, g
                if len(got_layers) != len(ref_layers):
                    result["errors"].append(f"{key}: {len(got_layers)} L2L layer outputs, {len(ref_layers)} resident")
        if device.type == "cuda" and n_buf == BUFFERS[-1]:
            trace, problems = memory_trace(cand, store, ex, residency, device)
            result["measurements"]["residency"][f"{shape}.{attn}.{dt}.{lora}"] = trace
            result["errors"] += [f"{tag}: {p}" for p in problems]
    if lora in ("last2-attention", "all"):  # power: the adapters change the L2L output
        ids, mask, pos = inputs("left", 9, [9, 5, 9, 2], device)
        with torch.no_grad():
            with_lora = l2l(cand, store, ex, residency, ids, mask, pos, 0)
            with cand.disable_adapter():
                without = l2l(cand, store, ex, residency, ids, mask, pos, 0)
        effect = (with_lora.double() - without.double()).abs().max().item()
        result["measurements"]["lora_effect"][tag] = effect
        if effect <= 1e-2:
            result["errors"].append(f"{tag}: LoRA changes L2L logits by only {effect:.2e}: no power")
    result["errors"] += [f"{tag}: {v}" for v in residency.violations[:5]] + [f"{tag}: {u}" for u in residency.unseen()[:5]]
    rows = [{"model": tag, **row} for row in lab.compare_all(ref_t, got_t, tol)]
    result["comparisons"] += rows
    lab.check_fault_detected(result, perturbation, "unsafe_reuse", tag, rows, getattr(ex, "injected", None))


def fixture_tie_back(cfg, result, perturbation, tol):
    """CPU, inside fixtures.isolated(): L2L on each Ticket 00 fixture model reproduces the saved resident logits."""
    for shape in F.SHAPES:
        model, fixture = F.load(shape)
        store = WeightStore.build(model, cfg.execution.weight_storage)
        try:
            place_resident(model, store, "cpu")
            buffers = gpu_buffers(cfg.execution.weight_buffers, store.max_slab_bytes, "cpu")
            ex, residency = executor(store, buffers, perturbation, cfg), Residency(model, store, buffers)
            r = fixture["rollouts"]
            pos = (r["attention_mask"].cumsum(1) - 1).clamp(min=0)
            try:
                with F.isolated(), torch.no_grad():
                    got = l2l(model, store, ex, residency, r["input_ids"], r["attention_mask"], pos, 0)
            except BufferReuseError as e:  # the ownership check: the expected detection of unsafe_reuse only
                lab.record_refusal(result, perturbation, "unsafe_reuse", f"fixture.{shape}", e)
                residency.remove()
                continue
            key = f"logits.fixture.{shape}.cpu"
            rows = [{"model": f"fixture.{shape}", **row}
                    for row in lab.compare_all({key: fixture["reference"]["logits"]}, {key: got}, tol)]
            result["comparisons"] += rows
            lab.check_fault_detected(result, perturbation, "unsafe_reuse", f"fixture.{shape}", rows,
                                     getattr(ex, "injected", None))
            result["errors"] += [f"fixture {shape}: {v}" for v in residency.violations[:5] + residency.unseen()[:5]]
            residency.remove()
        finally:
            store.close()


def forward_equivalence(cfg, result, perturbation=None, device="cuda", **_):
    for shape in F.SHAPES:
        lab.check_fixture(shape, cfg)
    device = torch.empty(0, device=device).device
    result.update(comparisons=[], reference_integrity=[], measurements={"residency": {}, "lora_effect": {}})
    if perturbation:
        result["perturbation"] = {"spec": perturbation, "effect": (
            "the buffer of layer i is targeted by a prefetch while in use (R1, R3 broken)"
            if perturbation.startswith("unsafe_reuse") else "layer i runs on slab i+1 (loaded after verify())")}
    tolerances = {dt: lab.tolerances_for("forward_equivalence", cfg, dt) for dt in DTYPES}
    fixture_tie_back(cfg, result, perturbation, tolerances["float32"])
    for shape in F.SHAPES:
        for attn in ATTN:
            for dt in DTYPES:
                for lora in LORA:
                    run_model(cfg, result, shape, attn, dt, lora, device, perturbation, tolerances[dt])
    result["memory"] = lab.memory(device)
