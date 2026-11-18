# 04 — L2L weight streaming: where every weight lives and how it moves

**What you will be able to explain after this file**
- Which tensors are in CPU RAM, which are page-locked, which are permanently on the GPU, and which cross PCIe every forward — with exact byte counts.
- How the 8 GB Qwen3-4B checkpoint is loaded without ever putting the whole model on the GPU.
- What a "slab" is, why it is one contiguous page-aligned block, why each tensor inside it is 256-B aligned, and how pinning is done (and why not with `pin_memory=True`).
- The mechanism by which a Hugging Face layer runs on weights that live in a reusable GPU buffer: re-pointing `Parameter.data`.
- Why the synchronous executor is forward-only, and why the schedule is micro-batch-major (and what the L2L paper does instead).
- What was taken from the L2L, MegaTrain and Soup ELS papers, and what was not.

Prerequisite: 03-qwen3-forward.md (layer structure and sizes). The asynchronous version and buffer safety are in 05-cuda-prefetch-and-buffer-safety.md; backward in 06-checkpointed-backward.md.

---

## 1. Intuition

Qwen3-4B in bf16 is ~8.0 GB of weights; the GPU budget is 8e9 bytes *including* the CUDA context, activations, KV cache, LoRA state and (in the final run) a 1.07 GB DFlash drafter. The weights cannot all be resident. But a forward pass only ever reads **one decoder layer at a time**, in order. So: keep all 36 layers in CPU RAM, and keep on the GPU a fixed pool of 2 buffers, each large enough for one layer. Before layer *i* runs, copy its 192.5 MiB into a buffer; run it; reuse that buffer for a later layer. GPU weight memory becomes `2 × 192.5 MiB` regardless of depth. The price: every forward pass moves 7.27 GB over PCIe.

## 2. Placement table (final run, `configs/experiments/qwen3_4b_final.yaml`)

| tensor | shape | dtype | device | lifetime | bytes | source |
|---|---|---|---|---|---|---|
| embedding = tied LM head | [151936, 2560] | bf16 | GPU | whole run | 777,912,320 (741.9 MiB) | `place_resident`, `layer_stream.py:34-36` |
| final norm weight | [2560] | bf16 | GPU | whole run | 5,120 | same |
| rotary `inv_freq` buffer | [64] | fp32 | GPU | whole run | 256 | same (buffers checked at `layer_stream.py:43`) |
| 36 decoder-layer slabs (all frozen weights of a layer, incl. the `base_layer` weights of the 6 adapted projections) | uint8 [201,861,632] each | bf16 inside | **CPU, page-locked** | whole run (until `store.close()`) | 7,267,018,752 total (6.768 GiB) | `WeightStore.build`, `weight_store.py:235-252` |
| GPU layer buffers ×2 | uint8 [201,861,632] each | (raw bytes) | GPU | whole run | 403,723,264 (385 MiB) | `gpu_buffers`, `weight_store.py:378-380`; `backend.py:69` |
| LoRA A, B (layers 34–35, q/v/o) | see 03 §4 | fp32 | GPU | whole run | 4,325,376 | `place_resident`, `layer_stream.py:37-39` |
| LoRA `.grad` | same | fp32 | GPU | from first backward of an update to `zero_grad(set_to_none=True)` after the step | 4,325,376 | `training.py:115,142` |
| AdamW `exp_avg`, `exp_avg_sq` | same ×2 | fp32 | GPU | from first step on | 8,650,752 | `torch.optim.AdamW`, `trainer.py:546` |
| rollout KV cache | 36 × 2 × [30, 8, P+R_max(+16 DFlash), 128] | bf16 | GPU (`kv_cache: gpu`) | one `generate` call | e.g. 411 MB at P = 73 with DFlash | `generator.KVCache`, `backend.py:96` |
| DFlash drafter | 537,455,360 params | bf16 | GPU | whole run | ~1.07 GB | `backend.py:102` (08) |

Byte counts are computed from `config.json` and match the budget calculator's header-derived figures **[code + predicted]**; nothing here is a GPU measurement.

The decoder weights of layers 34–35 are streamed exactly like the others: PEFT's frozen `base_layer.weight` is just another frozen parameter of the layer (`layout`, `weight_store.py:210-218`, takes every `requires_grad=False` parameter). Only the small LoRA A/B leaves stay on the GPU.

## 3. Loading the 4B model without a GPU copy

```mermaid
sequenceDiagram
  participant Disk as safetensors (disk / page cache)
  participant CPU as CPU RAM
  participant GPU as GPU
  Disk->>CPU: from_pretrained(dtype=bf16).to("cpu") : full model, pageable (model.py:178)
  CPU->>CPU: get_peft_model : LoRA A,B fp32 (B = 0), base frozen (model.py:182)
  Note over CPU: make_backend -> L2LBackend (backend.py:61-76)
  loop for each of 36 layers (WeightStore._adopt, weight_store.py:254-280)
    CPU->>CPU: raw = torch.empty(slab_bytes + 4096, uint8); align start to a page
    CPU->>GPU: cudaHostRegister(raw) : page-lock (driver call, no GPU memory)
    CPU->>CPU: copy each frozen param into its 256-B aligned offset; p.data = view; record sha256
  end
  CPU->>GPU: place_resident: embedding/head, final norm, rotary, LoRA -> GPU (layer_stream.py:31-45)
  CPU->>GPU: gpu_buffers(2, max_slab_bytes) : 2 x 201,861,632 B (weight_store.py:378-380)
```

Step by step, in the order `trainer._train` runs it:

1. `load_device = "cpu"` for L2L (`trainer.py:521`). `model.lora_model` calls `AutoModelForCausalLM.from_pretrained(name, dtype=bf16).to("cpu")` (`model.py:178`): **the full model, ~8.0 GB, in ordinary pageable CPU memory**. PEFT wraps it (`model.py:182`); LoRA A is seeded by `torch.manual_seed(seed)` just before (`trainer.py:517`), B is zero.
2. `make_backend` → `L2LBackend.__init__` → `WeightStore.build(model, "pinned")` (`backend.py:67`, `weight_store.py:235-252`). For each decoder layer `_adopt` (`weight_store.py:254-280`):
   - `layout(layer)`: the frozen parameters in module order; all must share one dtype (I7).
   - `nbytes = Σ align(numel·elsize, 256)`; allocate `raw = torch.empty(nbytes + 4096, uint8)` and take the page-aligned sub-range as `slab.data`.
   - `cudaHostRegister(raw.data_ptr(), raw.numel(), 0)` on the **whole** allocation (the comment at `weight_store.py:263-264` explains: `is_pinned()` looks up the storage base pointer, so registering only the aligned sub-range made every view report unpinned — measured in Ticket 02). Failure raises `PinningError`; there is no pageable fallback.
   - For each parameter: `view.copy_(p.data)`, then **`p.data = view`** — the same `Parameter` object now aliases slab memory; the old storage is freed when no other reference holds it. A sha256 of the bytes is recorded.
3. `place_resident(model, store, device)` moves everything that is *not* in the store to the GPU by replacing `.data` (`layer_stream.py:31-45`), so Parameter identities — the optimizer's references and the embedding/head tie — survive. It then raises if any parameter or buffer is neither stored nor on the device.
4. `gpu_buffers(2, store.max_slab_bytes, device)` allocates the pool (`backend.py:69`).

Peak CPU memory during step 2 is the full pageable model plus the slabs built so far, minus whatever storages were already released. Whether HF's loader leaves per-tensor storages that can be freed one by one (peak ≈ 8 GB + one slab) or larger shared storages (peak up to ≈ 16 GB) is **[unverified]**. The budget calculator models a "temporary loading copy (shard) 3.714 GiB" (`results/budget/predictions/qwen3-4b-final-dflash_8gb.md`) **[predicted]**; the orchestrator demands 32 GB host RAM for 4B stages (`configs/studies/assignment.yaml` `host_requirements`).

### Why `cudaHostRegister` and not `pin_memory=True`

"Pinned" (page-locked) memory cannot be swapped out, so the GPU's DMA engine can read it directly; only from such memory is `copy_(..., non_blocking=True)` truly asynchronous (05). PyTorch's `pin_memory=True` goes through the CachingHostAllocator, which rounds each block up to a power of two: 256 MiB for a 192.5 MiB layer, +33% (Ticket 02, `docs/l2l/02_weight_store.md`) **[measured on the dev host]**. Registering an exact-size allocation costs at most one page of slack per layer.

**Discrepancy to know about:** the budget calculator still assumes per-layer `pin_memory` rounding ("pinned allocator power-of-two rounding 2.232 GiB", `pinned_allocation: per_layer` in the prediction's config) **[predicted]**, while the code uses `cudaHostRegister` on exact sizes (`weight_store.py:131-137, 265`) **[code]**. The CPU prediction is therefore ~2.2 GiB more pessimistic than the implementation; it does not affect the GPU prediction.

### Invariants of the store (`weight_store.py:139-153`, checked by `verify()`, `weight_store.py:335-354`)

| id | invariant | why it matters |
|---|---|---|
| I1 | every frozen decoder parameter is a view of its slab at the recorded offset, on CPU | no hidden second copy (CPU or GPU) |
| I2 | its bytes equal the digest recorded at build | weights are never modified |
| I3 | pinned store ⇒ every view `is_pinned()` | async copies are real |
| I4 | LoRA, embedding/head, final norm, buffers never in the store | optimizer references stay valid |
| I5 | tensor offsets multiples of 256 B, slab starts on a page | GPU views aligned like `cudaMalloc`; kernel choice cannot differ from resident |
| I6 | building allocates nothing in the PyTorch CUDA allocator | no GPU footprint from the store |
| I7 | one dtype per slab | a slab is a flat byte array reinterpreted per tensor |

`close()` unregisters every slab exactly once (also via a `weakref.finalize` if never called), because memory freed while still registered stays page-locked in the driver — a bug found in Ticket 03 (`weight_store.py:150-153, 362-375`). The trainer closes the backend in a `finally` (`trainer.py:383-384`).

## 4. Binding a layer to a buffer: the synchronous executor

`StreamingLayerExecutor` (`layer_stream.py:54-124`) is the whole idea in ~70 lines.

### 4.1 Concrete trace for layer 12, two buffers, synchronous

```text
_acquire(12)                 k = 12 mod 2 = 0 -> buffer 0                       (layer_stream.py:102-111)
  copy_to(12, buffer0)       buffer0[:201,861,632].copy_(slab12.data)           (weight_store.py:298-308)
                             torch.cuda.synchronize()   <- host waits until the copy (and all queued work) is done
                             views = {name: buffer0[off:off+n].view(bf16).view(shape)}
bound(): for each entry      e.param.data = view   (q_proj.weight now points into buffer0, etc.)   (layer_stream.py:86-93)
                             check every param lies inside buffer0 (_inside), else raise           (layer_stream.py:94-95)
execute: out = layer(hidden, **kwargs)                                                             (layer_stream.py:120-121)
finally: e.param.data = slab view on CPU (always, also on exception)                               (layer_stream.py:97-100)
check   out is not inside buffer0 (the output must not alias weights)                             (layer_stream.py:122-123)
```

Then layer 13 goes to buffer 1, layer 14 overwrites buffer 0, and so on.

### 4.2 Why buffer reuse is safe here

Everything (copy and compute) is on **one CUDA stream**, which executes in issue order; and `copy_to` additionally synchronizes the host after each copy. So the copy of layer 14 into buffer 0 cannot start before layer 12's kernels — issued earlier on the same stream — finish (`layer_stream.py:15-16`). With 2 buffers there is no overlap at all in the synchronous executor: the host blocks on every copy. The second buffer only matters for the asynchronous one (05).

### 4.3 Why the forward-only executor refuses grad mode (`layer_stream.py:117-119`)

If autograd recorded `layer(hidden)` normally, it would save, for the backward of each matmul, references to **views of the weight buffer** (e.g. `W` for `dL/dx = Wᵀ δ`). Two layers later that buffer holds another layer's weights. The backward would read the wrong bytes and produce wrong gradients **without any error**. The forward-only executor therefore raises under grad mode, and the differentiable path is the checkpointed executor (06), which runs the layer under `no_grad` and recomputes it in backward with the weights re-bound.

## 5. One sweep, and how many sweeps a GRPO step costs

A **sweep** = streaming all 36 slabs = 7,267,018,752 B host → device. The engine is **micro-batch-major**: the outer loop is over passes (micro-batches, decode steps), the inner loop over layers, so every pass re-streams every layer.

| phase (one rollout batch, final config) | passes | weight copies |
|---|---|---|
| rollout, plain generator | prefill + one per further token: ≤ 4 sweeps (stops early when all 30 rows hit EOS) | ≤ 4 × 36 |
| rollout, DFlash (the final run) | prefill + 1 to 3 verify passes = 2–4 sweeps (08) | 72–144 |
| `prepare` old log-probs, 5 no-grad micro-batches of 6 | 5 sweeps | 180 (first layer of each later pass is a prefetch "hit", still copied once) |
| `grpo_update`, 5 grad micro-batches of 6 | 5 forward sweeps + backward recompute of layers 35, 34 | 180 forward + **0 in backward** (05: both still resident) |
| subset evaluation every 20 steps (128 prompts, batches of 16, ≤ 4 tokens) | 8 batches × ≤ 4 | ≤ 32 sweeps |

So a training step moves roughly 12–14 sweeps ≈ 87–102 GB **[code-derived count; predicted bytes]**. At an *assumed* 12 GB/s that is ≥ 7–8.5 s of transfer per step; the real bandwidth and compute time are **[unverified]** (the calculator's 103.8 GB/step includes 2.0 GB of backward reloads that the 2-buffer code does not perform, 05).

### What the L2L paper does differently

Pudipeddi et al., *Training Large Neural Networks with Constant Memory using a New Execution Algorithm* (2020, arXiv 2002.05645), Algorithm 2 "inverts the minibatch loop and layer loop": a layer is loaded **once** and all micro-batches pass through it before the next layer is loaded. Here that would turn 5 prepare sweeps + 5 update sweeps into 1 + 1 (plus storing 5 micro-batches' boundary activations). **This layer-major schedule is not implemented** (`configs/profiles/qwen3_4b.yaml` header comment; `research/l2l/literature.md` Ticket 03: "Not taken yet") **[code]**. It is the single largest efficiency gap in the design, and the reason DFlash's savings are small in comparison (08).

## 6. What was taken from which paper (`research/l2l/literature.md`)

| source | taken | not taken / different |
|---|---|---|
| **L2L** (arXiv 2002.05645) §2.1 | only the executing layer (+ transit buffers) on the device; only layer-boundary activations stashed in forward; layers recomputed in backward | the host "Eager Param-Server" updating weights on the CPU: here the base is frozen, LoRA and AdamW live on the GPU, nothing flows back to the host; Algorithm 2 (layer-major); cross mixed precision |
| **MegaTrain** (arXiv 2604.05091) §3.3, A.1, §3.4 | one contiguous page-aligned block per layer moved in one copy; tied embedding/head kept as one tensor; binding views of a flat buffer to a layer's weight slots | MegaTrain keeps the master store **pageable** with two pinned staging buffers (O(largest layer) pinned memory); this repo **pins the whole store** (7.27 GB) because the assignment asks for pinned CPU memory — a choice MegaTrain argues against for host capacity |
| **Soup ELS v3** (Zenodo 10.5281/zenodo.21918325) §3.1, §5.1 | the GPU-resident set (buffers, embedding, final norm, adapters + grads + optimizer); fully page-locked decoder store; bit-exactness vs resident as the correctness gate; the warning that L2L is not necessarily transfer-bound (v3 withdrew that claim) and that WSL2/WDDM spills instead of OOMing | NF4 quantization (bf16 here); per-tensor `pin_memory` (exact-size `cudaHostRegister` per layer here); SFT (GRPO here) |

Alignment to 256 B inside the slab (I5) and re-pointing the HF modules' own parameters instead of writing kernel templates are this repository's own adaptations.

## 7. Failure modes

| failure | consequence | guard |
|---|---|---|
| a module `.to(device)`'d after build | parameter leaves the store (and a GPU copy appears) | `verify()` I1 (`weight_store.py:343-347`); executor checks the store at construction (`layer_stream.py:68-69`) |
| page-lock fails / ceiling too low | store not built | `PinningError`, no silent fallback (`weight_store.py:165-166, 266-267`); WSL2 pinned ceiling unmeasured (≥ 512 MiB) |
| wrong slab bound to a layer | wrong outputs, possibly invisible | identity check `params.get(e.name) is e.param` (`layer_stream.py:83-84`); H01 `wrong_layer` fault **[test]** |
| exception inside a layer call | parameters left pointing into a GPU buffer | restore in `finally` (`layer_stream.py:97-100`) |
| output aliasing the weight buffer | activation overwritten by the next copy | `_inside(out, buffer)` check (`layer_stream.py:122-123`) |
| store freed while registered | stale page-locked range in the driver | single unregister in `close()` / finalizer |

## Open questions / unverified
- **Peak host RAM at 4B during `WeightStore.build`** and whether 7.27 GB can be page-locked on the target host. Settle with `H11.smoke` (records `cpu_rss_bytes` per step and the profiler's RSS samples) and `docs/deployment.md`'s host checks.
- **Effective pinned H2D bandwidth on the target GPU** (assumed 12 GB/s in the budget; 6.58 GB/s pinned vs 4.44 GB/s pageable measured on the dev laptop, Ticket 02). Settle with `H11.profile` per-phase timings and `H06.prefetch_timing`.
- **Cost of the micro-batch-major schedule** vs a layer-major one: not measured; would need an implementation of L2L Algorithm 2 and a timing comparison.
- **Bounded residency with execution (H07, Ticket 11 statement)**: only the copy-only version is supported (`t02_store_depth`); `H07.depth_residency_execution` has not run and H07 is dropped from the default protocol.
