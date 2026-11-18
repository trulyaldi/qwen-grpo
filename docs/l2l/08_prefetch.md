# Ticket 08: asynchronous L2L prefetching

## Code

- `qwen_grpo/execution/prefetch.py`: `PrefetchMixin`, `PrefetchingLayerExecutor` (forward only),
  `PrefetchingCheckpointedExecutor` (checkpointed backward), the `UnsafeReuse` fault, `timeline_report`, and
  `make_executor` / `buffer_counts`.
- `layer_stream.StreamingLayerExecutor.bound` now gets its buffer through two hooks, `_acquire` (returns a buffer
  holding the layer, ready for compute) and `_release` (called after the block). The synchronous hooks behave as
  before; the prefetching executor overrides them.
- `WeightStore.copy_nonblocking`: enqueues a slab copy on the current stream without synchronizing.
- **Selection by configuration.** `execution.backend: l2l_async` (with `prefetch_depth ≥ 1` and `weight_buffers ≥
  prefetch_depth + 1`, as already validated) selects the prefetching executor in the trainer (`backend.make_backend`)
  and in the lab cases. `l2l_sync` stays the synchronous executor.
- `qwen_grpo/prefetch_lab.py`: lab case `prefetch_timing`.
- Configs: `research/l2l/configs/async.yaml` (GPU), `async_cpu.yaml`, `async_qwen3_0.6b.yaml`.
- Tests: `tests/test_prefetch.py`, plus cases in `tests/test_backend.py` and `tests/test_lab.py`.

## Design: buffer ownership

Each pool buffer is a slot with three properties:
- the layer it holds (or is receiving);
- a `ready` event, recorded on the transfer stream after its copy;
- a `free` event, recorded on the compute stream after the last block that read it.

| rule | guarantees |
|---|---|
| R1: a copy into a slot waits on the transfer stream for the slot's `free` event | no transfer overwrites weights still being read |
| R2: compute waits on the slot's `ready` event before using it | no layer reads a buffer before its copy finishes |
| R3: the slot in use, and slots holding layers about to be needed, are never copy targets | the next layers are not evicted by their own prefetch |
| R4: `free` is recorded after all of the block's kernels are enqueued, including the backward recomputation and its `autograd.grad` | R1 covers backward too |

**Enforcement (since 2026-11-18).** R1 and R3 are validated on every copy, not only implied by the victim policy.
`PrefetchMixin._claim` raises `BufferReuseError` before a copy into the slot in use, into a slot of the prefetch
window, or (CUDA) into a used slot without a readers' `free` event. It is always on, independently of `verify`, and
host-only, with no synchronization. See `research/l2l/corrections/h02_unsafe_reuse_20261118/`.

**No global synchronization.** The pipeline never calls `torch.cuda.synchronize()` or `stream.synchronize()`. Only
verify mode and timeline reports synchronize, and they are diagnostics.

**What gets prefetched, and when.** When layer `i` is acquired, the next `depth` layers in execution order are copied
into the least recently used eligible slots:
- **Forward:** `i+1, i+2, …`. After the last layer, it wraps to layer 0, because a decode step or the next micro-batch
  starts there. The exception is a pass that builds a graph, where backward comes next.
- **Backward (reverse order):** `i−1, i−2, …`, down to the lowest layer that has a backward node, then layer 0, 1, ….
  Layers below that lowest node are never recomputed, so they are never fetched.
- **Hits:** a layer already in a slot is used without a copy. For example, the last forward layer is the first
  backward layer.

**The CPU-owned weights never change.** Copies only read the slabs, and `store.verify()` passes after every test.

**CPU buffers.** For tests, the same logic runs with immediate copies and no streams or events.

## Correctness evidence (CPU; the GPU runs are pending)

**Unit tests** (`tests/test_prefetch.py`, verify mode on):
- **Forward:** equals the synchronous executor bitwise over repeated passes, for depth and buffer counts (1, 2),
  (1, 3), (2, 3) and (3, 5).
- **Gradients:** equal bitwise with micro-batch accumulation, for 4 LoRA placements × 2 depths. Every copy is audited
  against R3. The first backward use is a hit, and the reverse prefetch stops at the lowest graph layer.
- **Generation:** equals bitwise. Every decode step finds layer 0 already prefetched; only the very first layer of the
  first step waits for its copy.
- **A complete training step:** prepare plus 4 GRPO updates in micro-batches of 2, at depth 0, 1 and 2, gives equal
  losses and parameters (stress of buffer reuse).
- **Unsafe reuse:** caught by verify mode ("buffer … does not hold layer 2"). Without verify mode it produces wrong
  outputs, and the store stays intact. *(Historical. Since the 2026-11-18 correction it is refused by
  `BufferReuseError` with or without verify mode, for every placement, in forward and backward, and when a prefetch
  targets the protected window.)*

**Real trainer** (`tests/test_backend.py`): an `l2l_async` run equals the resident run bitwise in rollouts, losses,
evaluations, adapters and AdamW state. `model.generate` is rigged to fail during it.

**Lab cases under `async_cpu.yaml`** (scratch pilot):

| case | result |
|---|---|
| `gradient_equivalence` | PASS, 5,296 comparisons |
| `generation_equivalence` | PASS, 950 comparisons |
| `update_equivalence` | PASS, 6,528 comparisons |
| `unsafe_reuse=2` fault | FAIL: 1,360 comparisons fail, and the checkpointed backward's recomputation checksum refuses (historical, before the correction; the `lab_fault` judge scored it "not detected") |
| `unsafe_reuse=2` fault, after the 2026-11-18 correction | FAIL with execution PASS: 80/80 configurations refused at runtime by `BufferReuseError`, no silently wrong result |

**Tolerances.** `forward_equivalence`, `gradient_equivalence`, `generation_equivalence` and `update_equivalence`
carry an `l2l_async` entry with `same_as: l2l_sync` and its own justification: streams and events change when bytes
arrive, never which bytes or which arithmetic. `lab.tolerances_for` refuses a `same_as` entry without a `why`.

## Performance experiment (`prefetch_timing`, CUDA only)

**Setup.** One real Qwen3-0.6B in bf16 (the `configs/qwen.yaml` batch of 5 × 6 and LoRA) and one pinned store, run
through two executors, each with its own pool of `weight_buffers` GPU buffers. The synchronous executor is the
reference and the prefetching one the candidate. Both pools exist throughout, so the memory baseline is equal.
Everything else is held constant.

**Per trial**, after warmup, alternating which side goes first:
- **generation:** greedy, 30 rows;
- **forward:** no-grad log-probs;
- **GRPO step:** forward with grad, backward, AdamW. LoRA weights and AdamW state are reset first, so every trial
  computes the same thing.

**Measured, per part and side**
- **Timings:** wall time, as median, IQR and bootstrap CI, plus the async/sync ratio CI.
- **From GPU timestamps:** H2D copy time, compute time, number of copies, and the **overlap**, meaning the time during
  which a copy and a compute block both run. The synchronous side is measured the same way, so its overlap of about
  0 is measured, not assumed.
- **Bytes:** per pass (from the store) and per part (from the copies).
- **Peak allocated memory.**

**Primary metric (H06):** the whole GRPO step.

**Correctness inside the case:** the first trial's generated tokens, log-probs, loss and every gradient must be bitwise
equal between the two sides.

**Smoke test.** `tests/test_prefetch.py` runs the whole case on CPU with a tiny model and fake CUDA events. It found a
bug before any GPU run: the LoRA weights were not reset between the two sides.

## Not established here

- **Everything on CUDA:** real streams, events, measured overlap, and timings. The development host's GPU cannot run
  this PyTorch build.
- **`forward_equivalence` under `l2l_async`** (it needs CUDA).
- **Whether overlap actually happens on the laptop.** With synchronous launches and one small model, the copy of
  layer i+1 can only overlap layer i's compute if the GPU is still busy when the copy is enqueued. The timeline
  measures this directly.
