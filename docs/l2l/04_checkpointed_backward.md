# Ticket 04: checkpointed L2L backward

Code:
- `qwen_grpo/execution/backward.py`: the checkpointed executor.
- `qwen_grpo/execution/layer_stream.py`: `bound` was factored out of `execute`, so forward and backward share the
  weight binding.
- `qwen_grpo/backward_lab.py`: lab case `gradient_equivalence`.
- `tests/test_backward.py`, plus two cases in `tests/test_lab.py`.

Config: `research/l2l/configs/store_cpu.yaml` (pageable store, for `--device cpu`).

Status: implemented, and a pilot run passes (below). **The acceptance evidence has not been recorded yet.** It needs a
commit with a clean tracked tree (see "Pending").

Training still refuses `l2l_sync` (`config.unimplemented`). Wiring the executor into the trainer is not part of this
ticket.

## Design

Under grad mode, `CheckpointedLayerExecutor` runs decoder layer `i` as a single autograd node:
`_CheckpointedLayer.apply(hidden_i, *lora_i)`.

**Forward**
- Streams the layer's frozen weights into a pool buffer (`StreamingLayerExecutor.bound`).
- Runs the unchanged HF/PEFT layer with grad disabled. No graph is built and no weight-buffer view is kept.
- Saves only the boundary activation `hidden_i` [N, T, H], plus references to the layer's own LoRA `Parameter`s,
  which cost nothing.

**Backward**
1. Streams the same weights in again.
2. Recomputes the layer from `hidden_i` with grad enabled.
3. Inside the same binding, computes
   `(dL/dhidden_i, dL/dlora_i) = autograd.grad(layer(hidden_i), (hidden_i, *lora_i), dL/dhidden_{i+1})`.
4. Releases the recomputed graph and the buffer views before the binding ends.

**Frozen weights still carry the input gradient.** They take part in `dL/dhidden_i` through every projection, norm and
residual. They receive no gradient themselves.

**Optimizer identities are preserved.** The LoRA `Parameter`s are explicit inputs of the node, so autograd
accumulates the returned gradients into their `.grad` as it would for any leaf. The optimizer's `Parameter` objects
are used unchanged, and micro-batch accumulation works without anything extra.

**Buffer reuse is safe.**
- Nothing the forward saves lives in a weight buffer. The lab and the tests check this with `saved_tensors_hooks`.
- Everything that reads a buffer (the recomputation and its `autograd.grad`) finishes inside `bound`.
- Gradients are checked not to alias the buffer.

**Which layers are reloaded.** Below the first adapted layer nothing requires grad (the embedding is frozen), so those
layers create no node and are not reloaded. Backward reloads exactly layers `n-1 … first adapted`, in that order.

**Fail-closed rules.** Each of these raises:
- **Recomputation check:** the recomputed output must match a checksum of the forward output's bit patterns. This
  catches model state that changed between forward and backward (for example a disabled adapter).
- **No training mode:** layers in training mode are refused, because dropout would draw a different mask in the
  recomputation.
- **No double backward:** it is refused (`once_differentiable`).
- **No missing LoRA gradient:** a LoRA parameter that backward reaches but that receives no gradient raises
  (`allow_unused=False`).
- **No grad-requiring keyword tensors:** masks and rotary embeddings that require grad are refused.

Under `no_grad` the executor is the Ticket 03 forward-only path.

## Tolerance

Tolerance 0 for loss, logits, LoRA gradients and hidden-state gradients, in fp32 and bf16 (`tolerances.yaml`,
`gradient_equivalence`).

Why 0 is justified:
- The recomputed layer graph is bitwise equal to the resident graph's segment (same bytes, same kernels; the checksum
  checks it).
- `autograd.grad` over that segment runs the same backward kernels, on the same operands, in the same order as the
  resident backward.
- Within a layer, each LoRA parameter is used once.
- The incoming `dL/dhidden` equals the resident one by induction from the last layer.

This assumes a deterministic resident backward, and the case checks that: it runs the reference twice, and a
non-repeatable reference fails the run (`reference_integrity`) rather than passing it.

**Disclosure.** Before the threshold and the H02 criterion were written down, a scratch pilot on CPU compared resident
and L2L LoRA gradients for the five placements, in fp32 and bf16. Every difference was 0. As with Ticket 03, 0 is the
only value the mechanism justifies, but the acceptance run is not blind.

## Experiment grid (`gradient_equivalence`)

**Grid**
- Both fixture shapes (4 and 6 layers) × eager/SDPA × fp32/bf16 × 1 or 2 weight buffers.
- LoRA placements, on every projection (attention and MLP) of the placed layers:

  | placement | 4-layer model | 6-layer model |
  |---|---|---|
  | final | 3 | 5 |
  | last2 | 2, 3 | 4, 5 |
  | middle | 2 | 3 |
  | first_last | 0, 3 | 0, 5 |
  | distributed | 0, 1, 2, 3 | 0, 2, 4, 5 |

- Adapters are made nontrivial with LoRA B ~ N(0, 0.5), so the gradients of A are nonzero. `require_nonzero` holds
  for every reference gradient.

**Loss.** The clipped GRPO loss on the Ticket 00 rollouts (`fixtures.rollout_batch`), with seeded advantages and old
log-probs. There are two micro-batches, one per prompt group.

**Compared quantities**
- every LoRA `.grad` after micro-batch 1 (single) and after micro-batch 2 (accumulated);
- every available `dL/d(layer output)`;
- loss and logits.

Each comparison records max abs error, relative error, relative L2, cosine, and both gradient norms.

**Fault injection.** `detach_boundary=<i>` detaches the activation saved for layer `i`. With the fault in place, no
gradient reaches anything below layer `i`.

## Pilot (scratch directory, uncommitted tree; not evidence)

| run | result |
|---|---|
| `gradient_equivalence`, CPU | PASS: 5,296 comparisons, all max_abs = 0 (4,480 LoRA gradients, 496 hidden-state gradients, 160 losses, 160 logits). Reference gradient norms range from 1.2e-3 to 0.90 |
| same, `detach_boundary=3` | FAIL as expected, execution PASS: 2,304 comparisons fail, all as "missing in candidate" (no silent zeros). The placements that pass are exactly those with no adapter below layer 3: 4-layer final; 6-layer final, last2 and middle |
| saved bytes, forward graph | L2L saves 8–25 % of the resident model's bytes for the same placement |
| CPU suite (CUDA hidden) | 527 passed, 87 skipped, 23 failed, 1 collection error. The failing set is identical to `HEAD` (aa2f074) on this host |

The 23 failures and the collection error were already there before this ticket. They are specific to this host:
- **Fixture-bitwise tests and lab cases (18 failures).** The Ticket 00 fixtures were recorded with torch `2.14.1+cu126`
  on another CPU. Here torch is `+cu130` on a Xeon W-2145, so `check_fixture` rejects the package versions and the
  bitwise reproductions differ.
- **Config/historical tests (5 failures).** These already failed in the Ticket 03 log.
- **`test_trl_reference` (collection error).** `research/trl_reference` is absent.

**CUDA was not exercised.** The local GPU (Quadro P1000, sm_61) is not supported by the installed PyTorch build.

## Pending

1. Commit. Then record the acceptance runs, each with `--hypothesis H02`:
   ```
   python scripts/l2l_lab.py run --case gradient_equivalence --config research/l2l/configs/store_cpu.yaml --device cpu --name t04_gradient_equivalence_cpu --hypothesis H02
   python scripts/l2l_lab.py run --case gradient_equivalence --config research/l2l/configs/store_cpu.yaml --device cpu --name t04_gradient_equivalence_detach --perturb detach_boundary=3
   ```
   On a supported GPU, also run `--config research/l2l/configs/store.yaml --device cuda` and the CUDA tests.
2. **CUDA determinism.** On a GPU, some backward kernels may be nondeterministic, for example SDPA's
   memory-efficient backward. If the resident reference is not repeatable, the CUDA run will FAIL on reference
   integrity rather than pass. In that case a nonzero tolerance would have to be justified and recorded before the
   run, not fitted to it.

## Not established here

- Real pretrained weights (Qwen3-0.6B through L2L backward).
- Training integration (`l2l_sync` in the trainer), and the KV cache.
- Memory independence from depth (H07, Ticket 11). The saved-bytes figures are allocator-free tensor accounting, not
  device memory.
- Overlap and performance (Ticket 08).
