# Ticket 02: pinned CPU weight store

Code: `qwen_grpo/execution/weight_store.py` (store), `qwen_grpo/execution/memory_plan.py` (accounting),
`qwen_grpo/store_lab.py` (experiments). Literature and how we differ from it: `research/l2l/literature.md`.
Evidence: `research/l2l/experiments/t02_*`, measured at commit `721d064`. No forward or execution code changed. The
resident backend is untouched.

## Ownership

| tensor | where | why |
|---|---|---|
| frozen decoder-layer weights (bf16 in training) | **CPU slab per layer, page-locked**; the module's `Parameter.data` is a view of it | read-only during training; streamed to the GPU in Ticket 03 |
| LoRA A/B (fp32, trainable), their gradients, AdamW state | GPU (Ticket 03), never in the store | tiny (≈4.1 MiB × 4 for 4B); stable `Parameter` identities for the optimizer |
| embedding = LM head (tied, one tensor: `lm_head.weight is embed_tokens.weight`, checked for 0.6B and 4B) | not in the store; GPU-resident from Ticket 03 | used at both ends of every forward and every decode step |
| final norm, `rotary_emb.inv_freq` | not in the store; GPU-resident | tiny |

Slab layout: one contiguous `uint8` block per layer, base aligned to a 4 KB page, every tensor offset aligned to 256 B.
`WeightStore.verify()` checks the invariants at any time (I1–I5 in the module docstring):
- each parameter is a view at its recorded offset, on the CPU;
- its bytes match the digest recorded at build;
- its view is `is_pinned()`;
- offsets are aligned.

`check_views()` checks GPU copies the same way. A layer moved with `.to()`, a re-pointed parameter, a flipped byte, a
wrong layer in a buffer and a corrupted buffer byte are each detected (`tests/test_weight_store.py`).

## Two implementation findings

1. **`pin_memory=True` rounds up to powers of two on this host.** The CachingHostAllocator held 2 MiB for a 1 MiB + 1
   request, 32 MiB for a 30 MiB request (one 0.6B slab), and 256 MiB for 192.5 MiB (one 4B slab, +33%)
   (`t02_pinned_transfer`). The store therefore page-locks with `cudaHostRegister` on each slab's exact allocation.
   For the real 0.6B, 881,047,552 B are registered, against 939,524,096 B with `pin_memory` (+6.6%).
2. **`Tensor.is_pinned()` follows the storage's base pointer, not the view's address.** Registering only the
   page-aligned slab inside a larger allocation left **every** parameter view reporting `is_pinned() == False`, even
   though the registration succeeded and DMA would use it. Registering the whole allocation fixes it, and this was
   verified directly. Without the owner's `is_pinned()` requirement this would have gone unnoticed.

## Predicted vs observed

| quantity | predicted | observed | run |
|---|---|---|---|
| stored tensors equal the reference | bitwise (bytes are only copied) | 0.6b-like 44/44, 4b-like 66/66 weights bitwise; resident code on store-backed weights reproduces all 168 saved outputs (logits, log-probs, loss, 12 LoRA grads, 3 AdamW updates) bitwise | `t02_store_equivalence` PASS |
| real Qwen3-0.6B stored tensors equal the checkpoint | bitwise | 308/308 equal to the safetensors shards on disk (independent reference); 0 missing, 0 unstored | `t02_store_build_qwen3_0.6b` PASS |
| pinned views | all | 308/308 `is_pinned()` (0.6B); 44/44 and 66/66 (fixtures) | same |
| CUDA allocator change from building the store | 0 B | 0 B in every run | same |
| controlled fault | detected | perturbed weight (1e-3 rel.) fails exactly that tensor, max_abs 3.9e-7 | `t02_store_equivalence_perturbed_weight` FAIL (expected) |
| streamed GPU memory vs depth (H07) | constant = 2 buffers × largest slab = 3,148,800 B | **3,148,800 B at depths 2, 4, 8, 16, 32**; resident 3.2 → 6.4 → 12.7 → 25.3 → 50.5 MB | `t02_store_depth` **H07 SUPPORTED** (criterion fixed in `hypotheses.yaml` beforehand) |
| pinned vs pageable H2D, one 4B slab (201,861,632 B), same synchronous copy | pinned faster (no staging copy by the driver) | 6.58 vs 4.44 GB/s; time ratio 0.675, 95% CI [0.655, 0.688] | `t02_pinned_transfer` |
| 0.6B store size | 28 × 31.46 MB (exact, from the plan) | 880,932,864 B in slabs, 881,047,552 B registered | `t02_store_build_qwen3_0.6b` |

All GPU numbers are PyTorch-allocator statistics: evidence grade `diagnostic`, not process-level. Transfer rates are
host-specific (WSL2, RTX 3050 Ti laptop).

## Host memory during the 0.6B build

- `from_pretrained` left the weights file-backed (mmap): RSS rose only from 695 to 714 MiB on loading.
- As layers were adopted, RSS grew to 2,188 MiB and `MemAvailable` fell from 2,013 MiB to a minimum of about
  1,347 MiB, then recovered to 1,517 MiB.
- RSS counts file-backed pages that the copy touched, so it overstates memory pressure. The peak RSS of 3,017 MiB came
  afterwards, while the checkpoint was re-read for verification.
- `MemAvailable` is the safer gauge. No second anonymous full-model copy was observed, and the floor guard (512 MiB)
  was never reached.
- Exactly when the source pages are released was not isolated. That is a measured trace, not a mechanism.

## WSL2 page-locked memory

- `ulimit -l` is 64 MiB, but `cudaHostRegister` locked 192.5 MiB (one slab) and 512 MiB (probe) without error, so
  `RLIMIT_MEMLOCK` does not bound CUDA registration here. On native Linux, Soup ELS §5.1 found `ulimit -l` to be the
  binding limit.
- The capacity probe was capped at min(1 GiB, MemAvailable − 1 GiB) = 545 MiB for safety. It registered 512 MiB and
  **did not find a ceiling**. The WSL2 page-locked ceiling is therefore **not measured**; only "≥ 512 MiB, and pinned
  pages come out of MemAvailable" is established.
- The 0.6B store (840 MiB) was pinned successfully. The 4B store (≈ 7.3 GB + one page per layer) exceeds this host's
  6.7 GiB of RAM: **BLOCKED on this host**, unchanged from `00_budget.md`.

## Not established here

- `memory_plan` counts the store, buffers and resident tensors exactly, and reuses the Step-0 calculator for the KV
  cache, boundary checkpoints and logits. It does **not** yet estimate per-layer activation transients or the CUDA
  context. The reuse path is `budget.calculator.Budget.layer_transient` and the measured context (`lab.memory`), to be
  added in Tickets 03–04 and 11.
- "Never adopt the resident reference model" is a documented rule, not enforced by code. Comparison harnesses build
  separate instances (`store_equivalence` does).
- Streaming into GPU buffers is tested only as copies. Running a layer on its GPU views is Ticket 03.
- Process-level GPU memory (WSL2 `nvidia-smi` is unreliable; Ticket 11).
- Behaviour on the target Linux machine (page-locked ceiling, bandwidth).
- Asynchronous transfers (Ticket 08).

## Owner decisions

1. `resident_layers` is implemented in the plan as **the last N decoder layers** (they hold the default LoRA presets
   and are needed first in backward). Confirm or choose otherwise.
2. A fully pinned 4B store does not fit this host. A MegaTrain-style pageable store with two pinned staging slabs would
   fit, but it may not meet the assignment's "pinned" requirement. The assignment text was not available to check.
