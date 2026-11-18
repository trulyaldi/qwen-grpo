# 05 — Asynchronous prefetching, CUDA streams/events, and buffer safety (H02)

**What you will be able to explain after this file**
- What a CUDA stream and a CUDA event are, at the level needed to reason about this code, and what must be true for a copy and a kernel to *actually* overlap.
- The lifecycle of Buffer A and Buffer B through layers 32 → 35, backward 35 → 34, and the next micro-batch's layer 0: who writes, who reads, which event each side waits on.
- The four ownership rules R1–R4 and where each is enforced in `prefetch.py`.
- Why, with 2 buffers, the backward recomputation of layers 34–35 costs **zero** weight copies (verified by simulation).
- The H02 unsafe-reuse fault: what it injects, why it produced *silent* wrong gradients before the 2026-11-18 correction, what the correction does, and what is still not guaranteed.

Prerequisites: 04-l2l-weight-streaming.md (slabs, buffers, `bound()`), 06-checkpointed-backward.md (what backward recomputes).

---

## 1. CUDA background, only what is needed

- **Stream**: an in-order queue of GPU work. Operations on one stream execute in issue order; operations on different streams have **no ordering** unless you add one. The host enqueues work and returns immediately (kernel launches and `copy_(non_blocking=True)` are asynchronous to the host), so the CPU can run ahead of the GPU.
- **Event**: a marker placed in a stream with `event.record(stream)`. It "completes" when all work enqueued on that stream before it has finished. `other_stream.wait_event(event)` makes *other_stream* hold all *later* work until the event completes — a GPU-side dependency; the host does not block.
- **Copy engine**: GPUs have DMA engines separate from the SMs. A host→device copy can proceed while kernels run **only if** (a) it is on a different stream from the kernels, (b) the host memory is page-locked (from pageable memory the driver must stage through a pinned bounce buffer and `copy_(non_blocking=True)` behaves synchronously), and (c) nothing in between forces serialization (a `torch.cuda.synchronize()`, a host read of a GPU value such as `.item()`, or a dependency chain).
- **"Asynchronous" ≠ "overlapping".** Issuing two things on two streams only *permits* overlap. Whether it happens depends on the dependency graph, the hardware, and how far ahead the host can enqueue. The repo states the same rule: overlap is reported only from GPU event timestamps (`prefetch.py:34-36`, `timeline_report`, `prefetch.py:242-260`), never inferred from the existence of a second stream.

## 2. The data structures (`prefetch.py:53-77`)

```python
@dataclass(eq=False)
class Slot:
    buffer: torch.Tensor   # one of the two uint8 GPU buffers (201,861,632 B)
    index: int | None      # layer whose bytes are in (or on their way into) the buffer
    ready: Event           # recorded on the COPY stream right after the copy into this buffer
    free: Event            # recorded on the COMPUTE stream right after the last block that read this buffer
    last_use: int          # logical clock for LRU
```

- `copy_stream = torch.cuda.Stream(device)` — one dedicated transfer stream (`prefetch.py:70`).
- The compute stream is whatever `torch.cuda.current_stream()` is — the default stream in training (`prefetch.py:189`, `204`).
- `self._in_use` — the slot bound to the layer currently running (`prefetch.py:74`, set at `182`, cleared at `201`).

## 3. Rules (`prefetch.py:7-22`) and where they are enforced

| rule | statement | enforced at |
|---|---|---|
| **R1** | a copy into a slot waits for every earlier reader of that slot | `copy_stream.wait_event(slot.free)` before enqueuing the copy (`prefetch.py:113-114`); `_claim` refuses a used slot with no `free` event (`prefetch.py:104-106`) |
| **R2** | compute uses a slot only after its copy completed | `torch.cuda.current_stream().wait_event(slot.ready)` in `_acquire` (`prefetch.py:188-189`) |
| **R3** | never copy into the slot in use, nor into a slot holding a layer of the upcoming window | victim selection `_victim(exclude)` (`prefetch.py:84-87`, `154`), **and since the correction** `_claim` (`prefetch.py:89-106`) called from `_prefetch` (`161`) and from every `_copy` (`110`) |
| **R4** | `slot.free` is recorded after all of the block's kernels were enqueued — for backward, after recompute **and** `autograd.grad` | `_release` (`prefetch.py:200-208`), called from `bound()`'s `finally` (`layer_stream.py:97-100`), which encloses the whole backward block (`backward.py:289-297`) |

No `synchronize()` is called in the pipeline (`prefetch.py:21-22`). `verify=True` (a byte checksum of the buffer, `_check`, `prefetch.py:222-230`) synchronizes and is **off in production**: `make_backend` constructs `L2LBackend(..., prefetch_depth=1)` with `verify` defaulting to `False` (`backend.py:62, 97-98`) **[code]**.

## 4. What happens at one layer: `_acquire` → layer → `_release`

`bound(i, layer)` (`layer_stream.py:76-100`) calls `_acquire(i, phase)`, which the prefetch mixin overrides (`prefetch.py:171-198`):

1. **Find the layer.** `slot = _slot_of(i)`. If present: a **hit** (it was prefetched earlier). If not: an **on-demand** copy into the LRU slot outside the upcoming window (`prefetch.py:176-179`).
2. **Mark in use.** `self._in_use = slot`.
3. **Prefetch** the next `depth = 1` layer(s) in execution order (`_prefetch`, `prefetch.py:152-163`): pick a victim slot (LRU, excluding the current slot and slots holding upcoming layers), validate with `_claim`, then `_copy` (`prefetch.py:108-128`):
   ```python
   copy_stream.wait_event(slot.free)                     # R1: after the slot's last readers
   with torch.cuda.stream(copy_stream):
       buffer[:n].copy_(slab.data, non_blocking=True)    # weight_store.copy_nonblocking, weight_store.py:310-319
       slot.ready = Event(); slot.ready.record(copy_stream)
   slot.index = j
   ```
4. **R2**: `compute.wait_event(slot.ready)` for the slot about to be read.
5. Return `(buffer, views)`; `bound()` re-points the layer's `Parameter.data` at the views and the HF layer's kernels are enqueued on the compute stream.
6. `_release` (after the block): `slot.free = Event(); slot.free.record()` on the compute stream; `slot.last_use = clock`.

### Prefetch order (`_upcoming`, `prefetch.py:137-150`)

| phase | sequence considered | why |
|---|---|---|
| forward, **no-grad pass** (rollout, `prepare`, evaluation) | `i+1, …, 35`, then **wrap** to `0, 1, …` | the next pass (decode step, next micro-batch) starts at layer 0 |
| forward, **grad pass** (`grpo_update`) | `i+1, …, 35`, **no wrap** | backward comes next, not layer 0 |
| backward | `i−1, …, floor`, then `0, 1, …` | `floor = _graph_floor` = lowest layer that built an autograd node = 34 here (`prefetch.py:210-218`); layers below it are never recomputed |

`_graph_pass` is set from `torch.is_grad_enabled()` when layer 0 starts a pass; `_graph_floor` is the first layer whose output `requires_grad` (`prefetch.py:213-217`).

## 5. Verified schedule (CPU simulation of the same ownership logic)

A scratch script ran the real `L2LBackend` on the 6-layer tiny "4b-like" fixture (LoRA on q/v/o of layers 4–5, the analogue of 34–35), CPU buffers, **2 buffers, depth 1**, `prepare` and one `grpo_update` with micro-batches of 3 (N = 6). On CPU the ownership logic is identical, copies are immediate, and there are no streams or events (`prefetch.py:31`, `123-124`) **[test-level evidence: CPU simulation, not CUDA]**:

```text
prepare (no grad), micro-batch 1:  L0 buf0 COPY(on-demand) | L1 buf1 hit | L2 buf0 hit | L3 buf1 hit | L4 buf0 hit | L5 buf1 hit
prepare (no grad), micro-batch 2:  L0 buf0 hit (wrap-prefetched during L5) ... L5 buf1 hit       (+ L0 prefetched again)
update (grad),     micro-batch 1:  L0 buf0 hit ... L4 buf0 hit, L5 buf1 hit
                   backward:       L5 buf1 hit,  L4 buf0 hit      -> issued copy: (L0, prefetch, during backward)
update (grad),     micro-batch 2:  L0 buf1 hit, L1 buf0, ..., L4 buf1, L5 buf0
                   backward:       L5 buf0 hit,  L4 buf1 hit      -> (L0, prefetch, backward)
stats: uses 28, hits 27, on_demand_copies 1, prefetch_copies 24; graph_floor 4; init_diff 0.0; ratio exactly 1.0
```

Consequences, transferred to Qwen3-4B (36 layers, floor 34) **[code-derived]**:
- In a grad forward, even layers land in one buffer and odd layers in the other (or swapped, depending on where layer 0 landed). At the end of the forward, **layer 34 and layer 35 are both still resident**, because a grad pass does not prefetch after the last layer.
- The backward recomputation of 35 and 34 are both **hits: zero weight copies in backward**. Backward of the lowest graph layer prefetches layer 0 of the next micro-batch.
- Only the very first layer 0 of the run waits for an on-demand copy; every later use is a hit in the bookkeeping sense (it still costs one copy, issued one layer earlier).
- **Contradiction with the budget calculator:** its transfer table charges "backward recomputation (×1, layers ≥ 34) 5 sweeps, 2.019 GB" per step (`results/budget/predictions/qwen3-4b-final-dflash_8gb.md`). The 2-buffer implementation performs none of those copies. The prediction is conservative, not wrong in direction, but it does not describe the code.

## 6. Two-buffer timeline on the GPU (grad micro-batch, layers 32–35, backward, next micro-batch)

Assume layer 32 is in **A** (prefetched while layer 31 ran in **B**). "Host" is the Python thread issuing work; the GPU columns show what each stream is allowed to do. `F(i)` = forward kernels of layer i; `R(i)` = backward recomputation + `autograd.grad` of layer i; `C(i→X)` = copy of slab i into buffer X.

| # | host action (`prefetch.py`) | copy stream | compute stream | A holds | B holds |
|---|---|---|---|---|---|
| 1 | `_acquire(32)`: hit A; prefetch 33 → victim B (holds 31, `B.free` = after F(31)) | `wait(B.free)` then **C(33→B)**, record `B.ready` | `wait(A.ready)`; **F(32)** | 32 | 31 → 33 (copy starts when F(31) done) |
| 2 | `_release(32)`: record `A.free` after F(32) | | | 32 | 33 |
| 3 | `_acquire(33)`: hit B; prefetch 34 → victim A | `wait(A.free)` then **C(34→A)**, record `A.ready` | `wait(B.ready)`; **F(33)** | 32 → 34 | 33 |
| 4 | `_release(33)`: record `B.free` | | | 34 | 33 |
| 5 | `_acquire(34)`: hit A; prefetch 35 → victim B | `wait(B.free)` then **C(35→B)** | `wait(A.ready)`; **F(34)** (LoRA output requires grad → `_graph_floor = 34`) | 34 | 33 → 35 |
| 6 | `_acquire(35)`: hit B; grad pass → **no prefetch** | idle | `wait(B.ready)`; **F(35)**; record `B.free` | 34 | 35 |
| 7 | final norm, LM head, `log_softmax`, loss, `part.backward()` (all compute stream) | idle | head + loss + head backward | 34 | 35 |
| 8 | autograd calls `_CheckpointedLayer.backward` for 35: `_acquire(35,'backward')` hit B; upcoming `[34]` already in A → nothing | idle | **R(35)**; record `B.free` | 34 | 35 |
| 9 | backward 34: `_acquire(34,'backward')` hit A; upcoming `[0]` → victim B | `wait(B.free)` then **C(0→B)** | **R(34)**; record `A.free` | 34 | 35 → 0 |
| 10 | next micro-batch `F(0)`: hit B; prefetch 1 → victim A | `wait(A.free)` then **C(1→A)** | `wait(B.ready)`; **F(0)** | 34 → 1 | 0 |

What overlaps, and what cannot:
- **C(i+1) can overlap F(i)** (rows 1, 3, 5): different streams, the copy depends only on `F(i−1)` finishing, the compute only on `C(i)`.
- **C(i+1) cannot start before F(i−1) finishes** (R1, same buffer). With 2 buffers the steady state is: `start(C(i+1)) ≥ end(F(i−1))` and `start(F(i+1)) ≥ max(end(F(i)), end(C(i+1)))`. If a copy takes longer than a layer's compute, the compute stream stalls on `ready` every layer and time per layer ≈ copy time; if compute is longer, copies are hidden. Which regime Qwen3-4B is in for a 6-row micro-batch (very little compute per byte) is **[unverified]**; at an assumed 12 GB/s one slab takes ≈ 17 ms.
- **The host must run ahead.** Overlap in row 3 requires the host to reach `_acquire(33)` and enqueue C(34) while F(32) is still running. Kernel launches are asynchronous, so it normally does; but any host synchronization inside a sweep would serialize. In the forward sweep the code avoids them (`checksum()` in the forward is a device-side reduction, `backward.py:268-270`). In backward, `torch.equal(checksum(out), ctx.checksum)` returns a Python bool and therefore **synchronizes the host** once per recomputed layer (`backward.py:291`); between micro-batches, `grpo_update`'s `.item()` / `isfinite` checks also synchronize (`training.py:120, 127`). These bound the run-ahead per micro-batch; their cost is **[unverified]**.
- **Allocator subtlety (why no `record_stream` is needed):** the two buffers are allocated once on the default stream and never freed; the copy stream only writes into them. Activations are produced and consumed on the compute stream only. So the caching allocator never hands memory touched by the copy stream to another tensor **[code reasoning]**.
- **Backward stream:** PyTorch's autograd engine runs a node's backward on the stream that was current when the forward op ran, so `_release` in backward records `free` on the compute stream that ran `R(i)`. This is PyTorch semantics applied to this code **[code reasoning; unverified on this repo's GPU path]**.

```mermaid
sequenceDiagram
  participant H as Host (Python)
  participant T as Transfer stream
  participant C as Compute stream
  H->>T: wait(B.free) ; copy L33 -> B ; record B.ready
  H->>C: wait(A.ready) ; F(32) on A ; record A.free
  H->>T: wait(A.free) ; copy L34 -> A ; record A.ready
  H->>C: wait(B.ready) ; F(33) on B ; record B.free
  Note over T,C: C(34->A) may overlap F(33) only after F(32) completed (A.free)
  H->>T: wait(B.free) ; copy L35 -> B ; record B.ready
  H->>C: wait(A.ready) ; F(34) ; record A.free
  H->>C: wait(B.ready) ; F(35) ; record B.free  (no prefetch: grad pass)
  H->>C: head, loss, backward R(35) on B (hit)
  H->>C: R(34) on A (hit)
  H->>T: wait(B.free) ; copy L0 -> B (next micro-batch)
```

## 7. Measurement status

- **Correctness of async = sync**: CPU tests compare the prefetching executors against the synchronous ones bitwise for forward, gradients with micro-batch accumulation, generation, and a full backend training step (`tests/test_prefetch.py:50-132`) **[test, CPU]**. The CUDA variants and the lab cases `H01/H02/H04/H05.lab_async_cuda` have **not** run (`evidence/study_report.md`: "NOT TESTED") **[unverified on GPU]**.
- **Overlap / speedup (H06)**: `status: untested`, no evidence (`research/l2l/hypotheses.yaml` H06). The timeline machinery exists (`timeline=True`, `report()`); the test that it measures overlap on CUDA (`tests/test_prefetch.py:213`) is skipped on this host.

## 8. Buffer safety: the H02 unsafe-reuse fault

### 8.1 Why a buffer-ownership violation gives wrong gradients

If layer *i* reads buffer A while A already holds layer *j*'s bytes, then `F(i)` computes `hidden_{i+1}` with the wrong weights. Every later activation, the logits, the loss, and every LoRA gradient are computed from that wrong activation: the gradient is the gradient of a *different function*. Shapes are identical across layers, so nothing crashes. For layers ≥ the graph floor there is a second effect: backward recomputes the layer from its saved input, possibly with yet another set of bytes, so `dL/dhidden` and the LoRA gradients are computed for a function that never ran in forward.

### 8.2 The fault (`UnsafeReuse`, `prefetch.py:289-305`)

When layer `at` is acquired, `_target` returns **the slot in use** (R3 broken) and sets its `free` event to `None` (R1 broken). Before the correction, `_copy` trusted that target: it enqueued `C(at+1 → in-use slot)` without waiting, recorded a new `ready` on that slot, and then R2 made the compute stream **wait for that very copy** before running layer `at`. Result: deterministically, layer `at` computed on layer `at+1`'s weights. (A real race would be nondeterministic; the fault was made deterministic on purpose.)

Chronology for `at = 2` (two buffers, before the fix):

| moment | A | B | what runs |
|---|---|---|---|
| `_acquire(2)` | 2 (in use) | 1 | fault: prefetch of 3 targets **A**, no `wait(A.free)` |
| copy enqueued | 2 → 3 | 1 | `A.ready` now means "layer 3 arrived" |
| R2: compute waits `A.ready` | **3** | 1 | `F(2)` reads A = **layer 3's weights** |
| backward (if layer 2 ≥ floor) | … | … | recompute of 2 binds weights again (by then another prefetch, of layer 1, had overwritten the slot): bytes ≠ forward's → checksum mismatch |

### 8.3 Five distinct cases (do not conflate them)

| # | case | what happens | detected by |
|---|---|---|---|
| 1 | **Correct ordinary execution** | R1–R4 hold; async equals sync bitwise | equivalence tests (CPU) **[test]**; GPU lab cases pending |
| 2 | **Deliberately corrupted execution** | `UnsafeReuse(at=i)` forces case 8.2 | — (it is the injection) |
| 3 | **Runtime detection by recomputation checksum** (pre-fix) | layer `i` ≥ `_graph_floor`: backward recomputes it and `backward.py:291-293` raises "the recomputed output differs from the forward's" | 48 of 80 configurations in `before_fix/` |
| 4 | **Detected only by comparing to a resident reference** (pre-fix) | layer `i` < `_graph_floor` (never recomputed: no node, 06): the corrupted forward flows silently into logits, loss and every gradient; the run completes | 32 of 80 configurations (`final` placement on both shapes, `last2` and `middle` on the 6-layer shape) — only the offline lab comparison caught them |
| 5 | **Structural safeguard (since the correction)** | `_claim` validates every copy target against ownership state **before** anything is enqueued and raises `BufferReuseError`; buffer and slot untouched | 80/80 configurations refused at `unsafe_reuse = 0, 1, 2`, with and without `verify` **[test, CPU]** |

Case 4 is the dangerous one, and it is exactly the production configuration's geometry: layers 0–33 are below the floor 34 (06). In the final run, 34 of 36 layers have **no runtime check that compares bytes** other than (now) the ownership bookkeeping.

### 8.4 Root cause and correction (`research/l2l/corrections/h02_unsafe_reuse_20261118/README.md`; diff in `prefetch.py`, uncommitted on 2026-11-18)

Root cause: R1/R3 were enforced only by the *selection policy* (`_victim`). `_copy` accepted any slot. The only remaining runtime guard was the recomputation checksum, which by construction covers only recomputed layers. And the lab harness scored the checksum's refusal as an execution error, so the `lab_fault` judge reported "NOT detected".

Correction:
- `BufferReuseError(RuntimeError)` (`prefetch.py:49-50`).
- `_claim(index, slot, protected)` (`prefetch.py:89-106`), called by **every** `_copy` (prefetch and on-demand, `prefetch.py:110`) and on each prefetch victim with the window as `protected` (`prefetch.py:161`). It refuses: the slot bound to the running layer; a slot in the prefetch window; on CUDA, a slot that has had readers (`last_use ≥ 0`) but no `free` event. Host-side bookkeeping, no synchronization, always on.
- `_acquire` clears `_in_use` if the prefetch is refused, because `bound()` has not yet entered its `try`, so its `finally` (which calls `_release`) would not run (`prefetch.py:183-187`).
- Harness: a `BufferReuseError` under `unsafe_reuse` is a **detection** (status FAIL, execution PASS); a completed fault run with every comparison in tolerance is an error ("undetected corruption" or "never injected (no power)" when `UnsafeReuse.injected == 0`) (`lab.record_refusal`, `lab.check_fault_detected`).

After-fix results (CPU, `async_cpu.yaml`):

| run | status | execution | detail |
|---|---|---|---|
| `ok`: H02, no fault | PASS | PASS | 5,296 comparisons, max_abs 0 |
| `fault`: `unsafe_reuse=2` | FAIL | PASS | 80/80 refused by `BufferReuseError` |
| `fault0`, `fault1` | FAIL | PASS | 80/80 refused each |
| `fault3` | FAIL | FAIL | 48 refused; 32 never injected (layer 3 is the last layer of the 4-layer shape: its window is already resident) |
| `h05_async`, `h05_sync` | PASS | PASS | 6,528 comparisons each |

The `before_fix` result is kept unchanged as recorded (FAIL/FAIL).

### 8.5 What the correction does *not* guarantee

- **`_claim` trusts its own labels.** It checks the bookkeeping (`slot.index`, `_in_use`, presence of `free`). A bug that mislabels a slot, or a copy issued by code that bypasses `_copy`, is not caught. Only `verify=True` compares actual bytes (`_check`), and it is off in production because it synchronizes.
- **It checks that an event exists, not that it is the right event.** R1 is satisfied by "some `free` event was recorded"; that it was recorded on the stream that ran *all* readers (in particular the autograd backward) rests on PyTorch's stream semantics (section 6).
- **No GPU rerun yet.** The corrected fault controls `H01/H02.fault_unsafe_reuse` on CUDA must be rerun on a GPU host; this host cannot run CUDA kernels (`research/l2l/corrections/.../README.md`, last line).
- **The orchestrator still lists H02.fault_unsafe_reuse as a known FAIL / waived limitation in `--experiment` mode** (`scripts/assignment.py` module docstring) — that text predates the correction's GPU evidence.
- A *real* race (as opposed to the injected deterministic overwrite) would require a missing `wait_event`; such a bug is not exercised by any test, because CPU tests have no streams.

## Open questions / unverified
- **Async = resident on GPU**: run `H01/H02/H04/H05.lab_async_cuda` and the short real runs `qwen3_0.6b_{resident,l2l_async}_short` + `scripts/compare_runs.py`.
- **Corrected fault detection on CUDA**: rerun `H01.fault_unsafe_reuse` and `H02.fault_unsafe_reuse` on a GPU host.
- **Genuine overlap and speedup (H06)**: `H06.prefetch_timing` (timeline with CUDA events, async/sync median ratio with bootstrap CI). Until then, "the copies are hidden" is a design intent, not a result.
- **Cost of host synchronizations in backward and per micro-batch**: visible in an `H11.profile` / Nsight Systems trace.
- **Depth > 1 or 3 buffers**: tested on CPU only (`tests/test_prefetch.py` grids); not used by the final config.
