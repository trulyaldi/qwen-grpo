# Ticket 03: synchronous L2L forward

Code: `qwen_grpo/execution/layer_stream.py` (placement + executor), `qwen_grpo/forward_lab.py` (lab case
`forward_equivalence`), `tests/test_layer_stream.py`. Evidence: `research/l2l/experiments/t03_*`, measured at commit
`c4b391b`. Forward only: training still refuses `l2l_sync` (`config.unimplemented`), because there is no backward yet
(Ticket 04).

## Design

| tensor | where during the L2L forward |
|---|---|
| frozen decoder-layer weights | CPU slab (Ticket 02); for the duration of its own layer call, a view of one buffer of a fixed GPU pool |
| embedding = tied LM head, final norm, rotary `inv_freq` | device-resident (`place_resident`) |
| LoRA A/B | device-resident, same `Parameter` objects (moved by replacing `.data`) |

Per decoder layer `i` (`StreamingLayerExecutor.execute`):
1. Copy slab `i` into buffer `i mod n` (synchronous).
2. Re-point that layer's own frozen `Parameter`s at their views of the buffer. If any of them is not inside the buffer,
   raise.
3. Call the unchanged HF/PEFT decoder layer. LoRA runs inside it.
4. Re-point the frozen weights back at their CPU slab views. This happens in a `finally`, so also when the layer
   raises.

- **Reuse safety.** Everything runs on one stream, and `copy_to` synchronizes. The copy for layer `i + n` is therefore
  ordered after layer `i`'s kernels. The layer output is a new tensor and is checked not to alias the buffer.
- **Grad mode.** The executor refuses to run with grad mode enabled. Autograd would save views of a buffer that a later
  layer overwrites, and backward would then silently use the wrong weights.
- **What the executor refuses.** Each of these raises; none falls back to resident execution:
  - an altered store (`verify()` runs at construction), or a closed one;
  - a buffer smaller than the largest slab;
  - a layer module that does not own the slab;
  - a loaded buffer whose weight names, shapes or dtypes differ from the layer's.

**The `qwen3_forward.py` interface is unchanged.** The ticket asked to extend the executor abstraction there. The
existing `execute(index, layer, hidden, **kwargs)` hook and `layer_callback` were sufficient, so the new executor
implements that interface and nothing was added to the file. The resident path is byte-identical:
`git diff 805a7b1 -- qwen_grpo/execution/qwen3_forward.py` is empty.

## Tolerance

Tolerance 0 for logits and per-layer outputs, fp32 and bf16 (`tolerances.yaml`, `forward_equivalence`). The mechanism
is that the comparison runs on the same device, with the same weight bytes and the same shapes. The kernels are the
same too: weights are bytes copied into 256 B-aligned views, never converted, so alignment-dependent kernel selection
cannot differ.

**Disclosure.** Before the threshold and the H01 criterion were written down, a pilot compared HF and L2L logits only:
both shapes, eager and SDPA, fp32 and bf16, LoRA on the last two attention blocks, CPU and CUDA. Every difference was
0. A second pilot of the full lab case in a scratch directory, not kept as evidence, also gave all 0. The threshold
was not chosen to fit those results, since 0 is the only value the mechanism justifies. Still, the acceptance run is
not blind.

## Evidence (commit `c4b391b`, tracked tree clean)

| run | result |
|---|---|
| `t03_forward_equivalence` (H01) | **PASS, H01 SUPPORTED.** 2,690 comparisons, all max_abs = 0 (details below) |
| `t03_forward_equivalence_wrong_layer` (fault: layer 0 runs on layer 1's slab, loaded after `verify()`) | **FAIL as expected:** 2,690/2,690 comparisons fail, from layer 0 onward; logits differ by ≥ 0.0999, the fixture logits by 0.37 and 0.45 |
| CPU suite (CUDA hidden), `ticket03_tests.log` | 488 passed, 45 skipped (all "needs CUDA"); failing set identical to the Ticket 00 baseline |
| CUDA run of `test_layer_stream`, `test_weight_store`, `test_lab`, `test_qwen3_forward` | 194 passed |

Details of the acceptance run:
- **Logits:** 450 comparisons of HF resident vs L2L.
- **Per-layer outputs:** 2,240 comparisons of explicit resident vs L2L.
- **Grid:**
  - 2 shapes × eager/SDPA × fp32/bf16;
  - LoRA variants: plain model, adapter disabled, last2-attention, every projection of every layer;
  - 1 and 2 buffers;
  - inputs: unpadded T = 1, 5, 9 and left-padded T = 9, 17;
  - logit columns: all, last 3, and an index tensor.
- **Fixture tie-back:** L2L on the CPU reproduces the saved Ticket 00 fixture logits bitwise.
- **Reference integrity:** explicit resident equals HF bitwise everywhere.
- **Residency:**
  - every frozen weight seen by a kernel lies in the buffer pool, and every stored weight was seen;
  - each layer's weights are back in the slab right after its call;
  - allocated memory before each layer copy is identical for every layer, and the same after the forward as before;
  - LoRA effect under L2L is 0.61 to 0.78 (power check);
  - LoRA `Parameter` identities and the store are unchanged.

### Residency, measured

- A forward pre-hook on every module that owns frozen decoder weights records the weight's device and address when the
  kernel is about to run. This shows that the computation reads buffer memory, not only that the executor re-pointed
  the weights.
- PyTorch-allocated device memory just before each layer's copy is the same byte count for every layer, in all 32 grid
  models. After the forward, with outputs freed, it equals its value before the forward. So no layer leaves a weight
  copy or an activation behind.
- These figures include the resident reference model, which lives on the same device as a separate instance, plus the
  cuBLAS workspace. They show constancy across layers, not a budget. Evidence grade: diagnostic (allocator statistics).

## Ticket 02 code changed by this ticket

1. **`copy_to` synchronizes only CUDA buffers,** so the same executor runs with CPU buffers in the CPU test suite.
2. **A store that is garbage-collected without `close()` now unregisters its slabs.**
   - Found by the Ticket 03 tests: a pinned store from one test was dropped without `close()`. Its memory was freed
     while still registered with CUDA, and a later *pageable* store allocated at the same address then reported
     `is_pinned() == True`.
   - `test_a_store_dropped_without_close_unregisters_its_slabs` fails without the fix. This was checked by detaching
     the finalizer: the views stay pinned.

## Not established here

- Backward through streamed layers (Ticket 04). Training refuses `l2l_sync`.
- Real pretrained weights: Qwen3-0.6B was not run through L2L in this ticket. Only tiny random models were.
- Process-level GPU memory, and memory independence from depth with execution included. Only per-layer constancy at
  fixed depth is shown here; the depth claim is H07 and Ticket 11.
- Transfer overlap and performance (Ticket 08). The synchronous executor is not timed here.
- Generation and the KV cache (Ticket 06). Every forward here uses `use_cache=False`.
