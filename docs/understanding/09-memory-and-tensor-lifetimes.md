# 09 — Memory accounting and tensor lifetimes

**What you will be able to explain after this file**
- Every major tensor of the final run: its shape, dtype, device, lifetime, size and purpose.
- How GPU memory rises and falls through one GRPO step, and why the predicted peak is the DFlash prefill, not training.
- The difference between PyTorch *allocated*, *reserved*, and what nvidia-smi shows for the process.
- Why "8 GB" means two different numbers in this repository, and which mechanism enforces which.
- Which numbers are exact arithmetic, which are budget-calculator predictions, and that **none** is a Qwen3-4B measurement.

> [!IMPORTANT]
> Every Qwen3-4B memory figure here is **[predicted]** (exact arithmetic on shapes, or the budget calculator in
> `qwen_grpo/budget/`). No Qwen3-4B run has executed on any GPU (`research/l2l/final_report.md`, R13 NOT TESTED).

Sizes use: L = 36 layers, H = 2560, intermediate 9728, 32 query / 8 KV heads × 128, V = 151,936, bf16 = 2 B.
N = 30 rollout rows, n = 6 rows per micro-batch, P = padded prompt width (73 for the first real batch; median
≈ 72, max 93 over the first 60 batches), R ≤ 4 response columns, T = P + R. See [01](01-batches-groups-microbatches.md).

---

## 1. Intuition: three memories, three different constraints

| memory | binding constraint | what dominates |
|---|---|---|
| GPU (device) | **8e9 B as shown by nvidia-smi, CUDA context included** (the assignment) | embedding 0.78 GB, drafter 1.07 GB, two layer buffers 0.40 GB, transients (prefill logits up to 2.5 GB) |
| CPU RAM | host capacity and the **page-locked ceiling** (pinned memory cannot be swapped) | frozen decoder store 7.27 GB pinned; the full model briefly during loading |
| Disk | durability, not size | checkpoints (a few MB each), rollout records (tens of KB each), logs |

L2L moves the 7.27 GB of decoder weights from the first row to the second; everything in this file is the bookkeeping
of that move.

---

## 2. CPU memory

| Tensor | Shape | Dtype | Device | Lifetime | Approximate bytes | Purpose |
|---|---|---|---|---|---|---|
| Model as loaded by `from_pretrained(..., dtype=bf16).to("cpu")` | 4,022,468,096 params | bf16 | CPU pageable | from `lora_model` (`model.py:178`) until each tensor is adopted or moved | ≈ 8.045 GB | the only way the code obtains the weights; no shard-wise loading |
| Decoder slab i (`WeightStore._adopt`) | uint8 [201,861,632] (+4 KiB alignment slack) | bf16 bytes | CPU, **page-locked by `cudaHostRegister`** on the exact allocation | build → `store.close()` at the end of `train()` (`trainer.py:383-384`) | 201,861,632 × 36 = **7,267,018,752** | the streamed frozen weights; parameters' `.data` are views into it (`weight_store.py:254-280`) |
| Old per-parameter storages of decoder weights | — | bf16 | CPU pageable | freed when `p.data = view` drops the last reference (`weight_store.py:276`) | 0 after build (if nothing else holds them) | — |
| Embedding / LM head, final norm, LoRA (CPU copies) | — | — | CPU | freed after `place_resident` moves `.data` to the GPU (`layer_stream.py:31-45`) | 0 after placement | — |
| Prompt offsets (`PromptStream.offsets`) | 1024 ints | Python | CPU | run | ~ tens of KB | locate records in `prompts.jsonl`; records are read from disk per batch (`dataset.py:92-96`) |
| One batch of 5 records, prompts text | 5 dicts | str | CPU | one attempt | < 10 KB | tokenization input |
| `result` log dict | grows | Python | CPU | run; serialized into every checkpoint | grows with steps (each eval stores 128 predictions) | resumable training log |
| Pending/rollout CPU copies at write time | `RolloutStore.write` moves tensors to CPU (`rollouts.py:199`) | int64/bool/fp32 | CPU | during the write | ~40 KB | disk record |

**Peak CPU during start-up [predicted, unverified]:** while `WeightStore.build` adopts layer i, the full pageable
model (≈ 8.05 GB, minus layers already adopted) coexists with the growing slabs. If each adopted parameter's old
storage is freed immediately, peak ≈ 8.05 GB + one slab; if `from_pretrained` keeps another reference (e.g. a
memory-mapped file or a shared storage), the peak is higher. Not measured for 4B. The budget calculator predicts a
CPU peak of 14.714 GiB at initialization, including a **2.232 GiB "pin_memory power-of-two rounding" term that the
code does not incur**: the store pins with `cudaHostRegister` on exact-size allocations precisely to avoid that
rounding (`weight_store.py:131-137, 259-269`). The calculator's `pinned_allocation: per_layer` models the other
mechanism. So the CPU prediction is conservative by ~2.2 GiB.

**Why pinned matters physically.** A DMA engine can copy host→device asynchronously only from page-locked memory
(the driver must know the physical pages will not move). From pageable memory, `copy_(non_blocking=True)` is staged
through a driver bounce buffer and is effectively synchronous with respect to the host (`weight_store.py:310-313`).
Pinning 7.27 GB means 7.27 GB of RAM the OS can never page out; under WSL2 the pinnable amount is unmeasured
(`research/l2l/literature.md`, Soup ELS §5.1 notes ceilings far below physical RAM). See [04](04-l2l-weight-streaming.md).

---

## 3. GPU memory: the permanent set

| Tensor | Shape | Dtype | Device | Lifetime | Approximate bytes | Purpose |
|---|---|---|---|---|---|---|
| CUDA context | — | — | GPU (outside PyTorch) | process | unknown; budget assumes 0.5 GiB **[predicted]** | driver state, kernels' code, cuBLAS/cuDNN handles |
| Embedding = tied LM head | [151936, 2560] | bf16 | GPU | run | 777,912,320 | token lookup and output projection (one Parameter, `output_head.py:268-276`) |
| Final RMSNorm weight | [2560] | bf16 | GPU | run | 5,120 | |
| Rotary `inv_freq` buffer | [64] | fp32 | GPU | run | 256 | RoPE |
| Weight buffer A, B (`gpu_buffers`) | uint8 [201,861,632] each | raw bytes | GPU | run | 403,723,264 | the only GPU home of decoder weights (`weight_store.py:378-380`) |
| LoRA A, B (12 tensors) | e.g. q: A [32, 2560], B [4096, 32]; v: A [32, 2560], B [1024, 32]; o: A [32, 4096], B [2560, 32]; × layers 34, 35 | fp32 | GPU | run | 4,325,376 | trainable adapters |
| AdamW `exp_avg`, `exp_avg_sq` | same as LoRA | fp32 | GPU | created at the first `optimizer.step()`, then run | 8,650,752 | optimizer moments |
| DFlash drafter | 537,455,360 params | bf16 | GPU | run | ≈ 1.075 GB (budget: 1.001 GiB) | rollout drafting ([08](08-dflash-integration.md)) |
| **Sum of exact PyTorch terms** | | | | | **≈ 2.27 GB (2.11 GiB)** | budget's "initialization" row: 2.106 GiB allocated |

Notably **absent from the GPU**: all 36 decoder layers except the ≤ 2 sitting in the buffers, LoRA gradients between
updates (freed by `zero_grad(set_to_none=True)`, `training.py:142`), any reference model (β = 0), any optimizer
state for frozen weights.

---

## 4. GPU memory: the transient set, per phase

### 4.1 Rollout (DFlash), N = 30 rows

| Tensor | Shape | Dtype | Lifetime | Bytes at P = 73 | Purpose |
|---|---|---|---|---|---|
| `ids`, `mask` | [30, P..P+4] | int64 | rollout | ~18 KB | sequences |
| Target KV cache | 36 × K,V of [30, 8, P+20, 128] | bf16 | rollout call (`dflash.py:198-199`) | 411.4 MB (plain generator, capacity P+4: 340.6 MB) | attention memory |
| KV prefix copy | `KVCache.update` returns a contiguous copy of the filled prefix of **one** layer (`generator.py:63-64`) | bf16 | one layer call | [30,8,P,128]×2 ≈ 9.0 MB | attention input |
| Hidden state, layer intermediates (no grad) | [30, P, 2560], MLP [30, P, 9728] | bf16 | one layer | 11.2 MB; ~43 MB per MLP intermediate | layer compute |
| Captured features | 5 × [30, P, 2560] then cat [30, P, 12800] | bf16 | prefill → first drafter round | 56 + 56 MB | drafter conditioning |
| **Prefill logits** | [30, P, 151936] bf16 → `.float()` fp32 | bf16+fp32 | prefill (`dflash.py:167`); fp32 kept until the next `_target_step` replaces `logits` | 0.666 + 1.331 = **2.0 GB** (2.54 GB at P = 93) | only the last column is used |
| Block logits per verify | [30, q≤4, V] fp32 + softmax | fp32 | one round | ~73 + 73 MB | target draws |

### 4.2 Old log-probs (`prepare`, no grad), 5 chunks of n = 6

| Tensor | Shape | Dtype | Lifetime | Bytes (T = 77) |
|---|---|---|---|---|
| hidden in/out | [6, 77, 2560] | bf16 | one layer | 2.37 MB each |
| MLP intermediates | [6, 77, 9728] | bf16 | one layer | 9.0 MB each |
| head input `h` | [6, R+1, 2560] | bf16 | chunk | 0.15 MB |
| logits, fp32 copy, log_softmax | [6, R+1 = 5, 151936] | bf16, fp32, fp32 | chunk | 9.1 + 18.2 + 18.2 MB |
| `old_logp` (accumulated) | [30, R] | fp32 | rollout batch (`pending`) | 480 B |

### 4.3 Training forward with grad (`grpo_update`), per chunk

Layers 0–33: `_CheckpointedLayer.apply` is called, but none of its inputs requires grad (the embedding is frozen,
no LoRA in these layers), so **no autograd node is recorded** and the saved input is dropped with `ctx`
(`backward.py:273-280, 311-320`; [06](06-checkpointed-backward.md)). Memory ≈ the no-grad forward above.

| Tensor saved for backward | Shape | Dtype | Bytes |
|---|---|---|---|
| Boundary input of layer 34 | [6, 77, 2560] | bf16 | 2.37 MB |
| Boundary input of layer 35 | [6, 77, 2560] | bf16 | 2.37 MB |
| Final-norm input (layer 35 output) | [6, 77, 2560] | bf16 | 2.37 MB |
| LM-head input, logits fp32, log_softmax output | [6, 5, ·] | | ≈ 36 MB |
| LoRA `.grad` (after the first chunk's backward) | 1,081,344 | fp32 | 4.33 MB, until `zero_grad(set_to_none)` after the step |

### 4.4 Backward recomputation of layer 35, then 34

Inside `executor.bound(..., phase="backward")` the layer is re-run **with grad**, so the full set of autograd-saved
activations of **one** layer exists briefly (q/k/v/o inputs, attention output, MLP gate/up/act), then
`autograd.grad` consumes them and `del out` frees the graph before the next layer (`backward.py:289-297`). The budget
estimates 103.8 MiB for this **[predicted]**. With two buffers, layers 34 and 35 are still resident at the end of the
forward, so **no weight copy** happens in backward (verified on a 6-layer CPU simulation: both backward uses are
hits; [05](05-cuda-prefetch-and-buffer-safety.md)). The budget assumes 2 layer reloads per chunk (2.019 GB of PCIe
per step) — a conservative mismatch in transfer, not in memory.

### 4.5 Evaluation (greedy, plain `StreamedGenerator`, batch 16)

KV cache [16, 8, P+4, 128] × 36 × 2: ≈ 180 MB at P = 73; step logits [16, V] fp32 = 9.7 MB. Label scoring
(`evaluation.label_scores`, batch 8) runs `be.logprobs` over the label tokens: small. The drafter stays resident
(1.07 GB) because it is part of the training backend.

---

## 5. Chronological GPU timeline of one run **[predicted]**

PyTorch-allocated, approximate, P = 73. "Base" = permanent set of §3 ≈ 2.27 GB (2.11 GiB).

| # | phase | code | dominant live transients | allocated ≈ |
|---|---|---|---|---|
| 1 | initialization | `make_backend` (`trainer.py:523`) | — (AdamW states not yet created) | 2.26 GB |
| 2 | rollout prefill | `dflash._target_step` over P columns | KV 0.41 GB + features 0.11 GB + **prefill logits 2.0 GB** | **≈ 4.8 GB** (budget at P = 100: 5.329 GiB = 5.72 GB) |
| 3 | DFlash rounds | `dflash.py:211-242` | KV 0.41 GB, fp32 prefill logits until the first verify reassigns `logits`, drafter context | ≈ 4.2 → 2.9 GB |
| 4 | old log-probs | `prepare` (`training.py:84-95`) | one chunk's logits ~36 MB | ≈ 2.33 GB |
| 5 | training forward | `grpo_update` chunk | boundaries + head ~45 MB | ≈ 2.33 GB |
| 6 | backward recompute | `_CheckpointedLayer.backward` | one layer's graph ~100 MB | ≈ 2.4 GB |
| 7 | gradient accumulation | `.grad += ...` across 5 chunks | LoRA grads 4.3 MB | unchanged |
| 8 | optimizer step | `optimizer.step()` (`training.py:141`) | `before` clones of params 4.3 MB (`training.py:140`), AdamW moments 8.65 MB | unchanged |
| 9 | validation (every 20 steps) | `eval_subset` → `backend.greedy` | KV 0.18 GB, logits | ≈ 2.5 GB |
| 10 | checkpoint / merge | `save_checkpoint`: `state_dict()` → CPU via `torch.save`; merge is CPU-only and outside training (`export/merge.py`) | ~0 | unchanged / 0 for merge |

**Where the peak is and why.** The DFlash prefill is the only place a [rows × columns × V] tensor is materialized for
**all** prompt columns: V = 151,936 makes each column of each row cost 304 KB in bf16 and 608 KB in fp32. Training
deliberately avoids this (`logits_to_keep = R+1` / `hidden[:, P-1:]`, `policy.py:287-301`, `output_head.py:296-303`),
plain generation keeps one column (`generator.py:92`). The DFlash code does not, because only the last column is
needed but `lm_head` is applied before slicing (`dflash.py:167-168, 204`). Removing that would cut the predicted
peak by ~2 GB; this is an observation, not a requested change.

**Reserved vs allocated.** The caching allocator keeps freed blocks; `reserved` ≥ the historical maximum of
allocated plus fragmentation. Because the prefill logits are freed and the next allocations are much smaller, those
~2 GB stay *reserved* for the whole run unless `empty_cache()` is called (it never is). So nvidia-smi will show
roughly the rollout peak plus context **for the entire run**, not just during rollouts **[predicted]**.

---

## 6. Allocated, reserved, device: what each number means

| reading | includes | excludes | source in code |
|---|---|---|---|
| `torch.cuda.memory_allocated` | live tensors | cache, context, non-PyTorch allocations | `trainer.py:744`, row `cuda_allocated_gib` |
| `max_memory_allocated` / `max_memory_reserved` | peaks since `reset_peak_memory_stats` (`trainer.py:526`) | context | `trainer.py:745-746` |
| nvidia-smi per process | context + reserved + any non-PyTorch device allocation (cuBLAS/NCCL workspaces if not from the allocator) | — | `execution/profiler.py`, `telemetry.py` |
| `cudaMemGetInfo` total − free | whole device, all processes (and the Windows host under WSL2) | — | profiler fallback |

`trainer.save_log` computes `within_budget = peak_reserved_gib < gpu_budget_gib` (`trainer.py:800`) and prints
explicitly that allocator stats exclude the CUDA context and are "not budget evidence" (`trainer.py:854-858`).
Evidence grades (`research/l2l/protocol.md`): only **assignment** grade — a numeric per-process nvidia-smi reading in
every sample, outside WSL2 — can support a compliance claim. This host (WSL2) can reach at most `diagnostic`.

---

## 7. 8e9 bytes vs 8 GiB, and who enforces which

| quantity | bytes | MiB | GiB |
|---|---|---|---|
| assignment cap (orchestrator reading) | 8,000,000,000 | 7,629.4 | 7.4506 |
| `run.gpu_budget_gib: 8.0` | 8,589,934,592 | 8,192 | 8.0 |
| gap | 589,934,592 | 562.6 | 0.549 |

| mechanism | threshold | measures | where | when active |
|---|---|---|---|---|
| in-process guard (`Profiler`) | `int(gpu_budget_gib * 2**30)` = **8 GiB** (`trainer.py:408`) | nvidia-smi per-process, else `cudaMemGetInfo` (an over-estimate), every 0.5 s; interrupts the main thread on violation | `execution/profiler.py`, `trainer.memory_guard` (`trainer.py:400-415`) | every `train` with `memory.guard` on CUDA |
| allocator cap | `set_per_process_memory_fraction(8 GiB / total)` (`trainer.py:412-414`) | PyTorch reserved only; an allocation beyond it fails immediately | same | with `allocator_cap: true` |
| external telemetry watchdog | **8e9 B** (`telemetry.py`, `configs/studies/assignment.yaml` `gpu_memory_cap.bytes`) | raw nvidia-smi every 200 ms (`telemetry.py:38`), per process group; stops the workload; a sample above the cap makes the item FAIL | `scripts/assignment.py` | stages D–G of the orchestrated protocol only |

Consequences an interviewer will probe:
1. A direct `python scripts/run.py train --config configs/experiments/qwen3_4b_final.yaml` is guarded at **8 GiB**,
   not 8e9 B. A run using 7.8 GiB would pass its own guard and fail the assignment's reading.
2. The allocator cap excludes the context, so with an 8 GiB cap and ~0.5 GiB context the process could reach
   ~8.5 GiB device memory before the allocator refuses anything; only the samplers can catch that.
3. Sampling is periodic (0.5 s in-process, 0.2 s external): a spike shorter than the interval can be missed. The
   allocator cap is the only instantaneous limit, and it is the looser one.
4. The prediction's device figure for the peak (6.362 GiB at P = 100) leaves ~1.09 GiB headroom to 7.4506 GiB **[predicted]**;
   the margin is real only if the context and workspaces are near the assumed 0.5 GiB.

---

## 8. Disk

| Item | Shape / content | Size | Lifetime | Written by |
|---|---|---|---|---|
| `streaming/prompts/prompts.jsonl` + `prompts.meta.json` | 1024 JSON lines; sha256 + byte offsets | ~150 KB | run | `PromptStream.build` (`dataset.py:61-87`) |
| `streaming/rollouts/group_<id>.pt` | ids [30, T] int64, masks, rewards [5, 6], advantages [30], old_logp [30, R] (None for zero-signal) | ~40 KB each | run (never deleted; uncommitted ones moved to `orphaned/`) | `RolloutStore.write` (`rollouts.py:197-219`) |
| `checkpoints/step<N>/adapter_model.safetensors` | 12 LoRA tensors fp32 | 4.33 MB | run | `model.save_pretrained` |
| `checkpoints/step<N>/training_state.pt` | config, counters, result log, AdamW state (8.65 MB), RNG states, streaming state | ~9 MB + log | run | `save_checkpoint` (`checkpointing.py:456-460`) |
| `COMMIT.json` | file sizes + sha256 | < 1 KB | run | `stream_resume.publish` |
| `training/training.json`, `training.log`, `memory/<stamp>.json` | logs, raw memory samples | MB | run | trainer |
| merged model (optional, after training) | full bf16 checkpoint | ≈ 8 GB | — | `export/merge.py`, CPU only |

Details of the disk protocols: [12](12-streaming-and-resume.md).

---

## 9. Failure modes

| failure | symptom | guard |
|---|---|---|
| pinned registration fails (page-locked ceiling) | `PinningError` at start; never a silent pageable fallback (`weight_store.py:165-166, 265-267`) | fail-closed |
| CPU RAM exhausted while the full model and slabs coexist | OOM kill during `WeightStore.build` | none; host requirement 32e9 B in the study (`configs/studies/assignment.yaml` `host_requirements`) |
| DFlash prefill at a long batch | reserved jumps by ~2.5 GB | guard / watchdog; P ≤ 94 tokens in this pool bounds it |
| allocator fragmentation | reserved ≫ allocated | visible in `peak_reserved_gib`; no `empty_cache` |
| WSL2 nvidia-smi `[N/A]` | guard falls back to device-wide `cudaMemGetInfo` (stops early, never late) | evidence graded `diagnostic` |

## Open questions / unverified

- **Actual per-process GPU peak** of the final configuration and the phase where it occurs: `H11.profile` (`scripts/profile_run.py`, plain rollouts) and `H10.smoke_dflash` with assignment-grade evidence.
- **Real CUDA context + workspace size** on the target GPU (the budget assumes 0.5 GiB): the profiler's setup phase.
- **CPU peak during `WeightStore.build`** and whether old storages are freed per tensor: log RSS via `WeightStore.build(on_layer=...)` on the target host.
- **Pinnable host memory** on the target machine: `WeightStore.build` either succeeds or raises `PinningError`.
- **Whether reserved memory stays at the rollout peak** for the rest of the run: compare `max_memory_reserved` with nvidia-smi samples in `training/memory/`.
