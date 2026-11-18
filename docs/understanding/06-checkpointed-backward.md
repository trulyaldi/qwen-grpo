# 06 — Checkpointed backward: correct LoRA gradients without resident weights

**What you will be able to explain after this file**
- The central puzzle: decoder weights are overwritten during the forward, yet backward needs them. How the code resolves it.
- Exactly which tensors survive from the forward to the backward of a training micro-batch, with shapes and bytes.
- Why layers 0–33 build **no** autograd node (nothing saved, nothing recomputed) and why that is correct, not an approximation.
- The chain rule from the GRPO loss down to `lora_A` and `lora_B` of layer 34, including the frozen weights' role.
- Why the first optimizer step moves only `B`, never `A`.
- How gradients accumulate across the 5 micro-batches into one AdamW step.
- What the recomputation checksum proves and what it does not.

Prerequisites: 03-qwen3-forward.md (layer math, LoRA placement), 04-l2l-weight-streaming.md (`bound()`), 05-cuda-prefetch-and-buffer-safety.md (buffers, events). The loss itself is derived in 07-grpo-mathematics.md.

---

## 1. The puzzle and the idea

Ordinary autograd, run on a layer whose weights live in a reusable buffer, would save **views of that buffer** for its backward (e.g. `W` for `∂L/∂x = Wᵀδ`). Two layers later the buffer holds other bytes, and backward would silently use them. That is why the plain streaming executor refuses grad mode (`layer_stream.py:117-119`).

Activation checkpointing (Chen et al., 2016, cited by the tickets but not read: `research/l2l/literature.md`) gives the way out, combined here with weight streaming as in L2L §2.1 (arXiv 2002.05645):

1. In the forward, run each layer **without** building a graph; save only its **input** `hidden_i`.
2. In the backward, when autograd reaches layer *i*: stream layer *i*'s weights in **again**, rerun the layer from `hidden_i` **with** grad enabled, differentiate that local graph, release it, release the buffer.

The weights the backward needs are therefore never "kept"; they are **re-acquired** from the CPU slab (or found still resident; 05 §5).

## 2. The mechanism (`qwen_grpo/execution/backward.py`)

Under grad mode, `CheckpointedLayerExecutor.execute` (`backward.py:311-320`) replaces the plain call `layer(hidden)` by

```python
lora = [p for _, p in layer.named_parameters() if p.requires_grad]   # [] for layers 0..33, 6 tensors for 34, 35
return _CheckpointedLayer.apply(self, index, layer, kwargs, hidden, *lora)
```

### 2.1 Forward of the node (`backward.py:274-280`)

```python
ctx.save_for_backward(hidden, *lora)                 # the boundary activation and the LoRA leaves
out = StreamingLayerExecutor.execute(executor, ...)  # bound(): weights -> buffer; layer(hidden); unbind
ctx.checksum = checksum(out)                         # int64 sum of the output's bit patterns, on device
```

Inside `autograd.Function.forward` grad mode is off, so the HF layer builds no graph and holds no buffer views after it returns. `StreamingLayerExecutor.execute` checks that the output does not alias the buffer (`layer_stream.py:122-123`).

### 2.2 Backward of the node (`backward.py:282-300`)

```python
hidden, *lora = ctx.saved_tensors
need_h, need_p = ctx.needs_input_grad[4], ctx.needs_input_grad[5:]
h = hidden.detach().requires_grad_(need_h)
wrt = ([h] if need_h else []) + [p for p, need in zip(lora, need_p) if need]
with torch.enable_grad(), ctx.executor.bound(ctx.index, ctx.layer, phase="backward") as buffer:
    out = ctx.layer(h, **ctx.kwargs)                                  # recomputation, builds a local graph
    if not torch.equal(checksum(out), ctx.checksum): raise ...        # must reproduce the forward
    grads = torch.autograd.grad(out, wrt, grad_out, allow_unused=False)
    del out
    if any gradient lies inside `buffer`: raise ...
return (None, None, None, None, grad_h, *grad_lora)
```

Every read of the weights — the recomputation **and** `autograd.grad`, which multiplies by `Wᵀ` — happens inside `bound()`. Only when that block exits does `_release` record the buffer's `free` event (05, R4), after which the transfer stream may overwrite it. The returned gradients are new tensors outside the buffer (checked).

The LoRA parameters are **explicit inputs** of the node. Returning their gradients from `backward` makes autograd accumulate them into `p.grad` exactly as for any leaf, so the optimizer's `Parameter` objects are used unchanged and micro-batch accumulation works without special code (`backward.py:230-231`).

## 3. Which layers get a node

`torch.autograd.Function.apply` records a node only if at least one tensor input requires grad.

| layer(s) | `hidden` requires grad? | LoRA inputs | node? | saved for backward | recomputed in backward |
|---|---|---|---|---|---|
| 0–33 | no (embedding output; embedding frozen) | none | **no** | nothing (ctx discarded) | **never** |
| 34 | no (output of layer 33) | 6 tensors (q/v/o A, B) | yes | `hidden_34`, 6 leaves | yes, `need_h = False` |
| 35 | **yes** (output of node 34) | 6 tensors | yes | `hidden_35`, 6 leaves | yes, `need_h = True` |

This is exact, not an approximation: the loss depends on the trainable parameters only through layers 34 and 35. The gradient with respect to anything in layers 0–33 is not wanted (frozen), and `∂L/∂hidden_34` is not needed because nothing below layer 34 is trainable. So backward stops at layer 34. The prefetch executor records this lowest node as `_graph_floor` (`prefetch.py:216-217`) and never prefetches below it in backward (05).

Price of this asymmetry: layers 0–33 are never recomputed, so the recomputation checksum gives **no protection** for them — the root of the silent half of the H02 fault (05 §8).

The same code is general: the lab grid places LoRA on the first and last layer, a middle one, several distributed layers (`research/l2l/hypotheses.yaml` H02 criterion); then backward recomputes every layer from the lowest adapted one upward.

## 4. What survives from forward to backward (one grad micro-batch, n = 6)

Example shape `P = 73`, `R = 4`, `T = 77`; `H = 2560`, `V = 151,936`.

| tensor | shape | dtype | device | bytes | saved by |
|---|---|---|---|---|---|
| `hidden_34` (input of layer 34) | [6, 77, 2560] | bf16 | GPU | 2,365,440 | `_CheckpointedLayer` ctx |
| `hidden_35` (input of layer 35) | [6, 77, 2560] | bf16 | GPU | 2,365,440 | `_CheckpointedLayer` ctx |
| layer 35 output = final-norm input | [6, 77, 2560] | bf16 | GPU | 2,365,440 | RMSNorm backward |
| final-norm output columns `P−1…T−1` (head input `h`) | [6, 5, 2560] | bf16 | GPU | 153,600 | `F.linear` backward (needs `h` and the resident `E`) |
| logits upcast + `log_softmax` output | [6, 5, 151936] | fp32 | GPU | 18,232,320 | `log_softmax` backward |
| `logp` → loss intermediates | [6, 4] | fp32 | GPU | tiny | `grpo_loss` ops |
| decoder weights of any layer | — | — | **not saved** | 0 | — |

Total saved activations per micro-batch ≈ 25 MB **[code-derived; predicted]**. The calculator's figures for the same phase are similar ("boundary checkpoints [3 × 6 seq × 104 tok] 9.1 MiB", "logits + fp32 copy + log_softmax 36.5 MiB", `results/budget/predictions/qwen3-4b-final-dflash_8gb.md`) **[predicted]**. The largest *transient* in backward is the recomputed layer's own local graph (q/k/v, attention, the `[6, 77, 9728]` MLP intermediates) — the calculator estimates ~104 MiB (09-memory-and-tensor-lifetimes.md).

After `part.backward()` returns, all of this is freed; `grpo_update` also drops `logp` and `part` (`training.py:130-133`) before the next micro-batch's forward.

## 5. The chain rule, explicitly

Per micro-batch `c` the code backpropagates `part_c = (n_c/N) · grpo_loss_c` (`training.py:124-130`). Writing `ℓ` for that scalar, the reverse path is

```text
ℓ ──► logp[i,t] ──► log_softmax(z)[o_t] ──► z = h Eᵀ ──► h = RMSNorm(y35)[cols P-1..T-2]
  ──► y35 = Layer35(y34; W35, θ35) ──► y34 = Layer34(x34; W34, θ34)        (x34 = hidden_34, no grad)
```

- `∂ℓ/∂logp[i,t]` comes from the clipped surrogate (07); at ratio 1 it is `−(n_c/N)·(1/n_c)·A_i/|o_i|` for valid tokens.
- `∂logp/∂z = onehot(o_t) − softmax(z)` (fp32), then `∂z/∂h = E` (the frozen, resident, tied embedding; `E` itself gets no gradient).
- RMSNorm backward gives `∂ℓ/∂y35` = `grad_out` of node 35.
- Node 35's backward returns `∂ℓ/∂y34 = grad_out` of node 34 and `∂ℓ/∂θ35`; node 34 returns `∂ℓ/∂θ34` only.

Inside a layer, LoRA parameters sit in q, v (before attention) and o (after attention), so the gradient for θ34's q/v adapters must pass back through layer 34's MLP, o_proj, attention softmax, and RoPE — every one of which needs layer 34's **frozen** weights. This is why the recomputation must happen with the real weights bound, even though none of them is updated.

## 6. Gradients through one LoRA projection

For an adapted projection with frozen `W ∈ R^{out×in}`, `A ∈ R^{r×in}`, `B ∈ R^{out×r}`, `s = α/r = 64/32 = 2`, and one token's input `x ∈ R^{in}`:

```
u = A x               ∈ R^r          (fp32)
y = W x + s · B u     ∈ R^out        (base in bf16, delta in fp32, sum cast to bf16; 03 §4)
```

Let `δ = ∂ℓ/∂y` (per token). Summing over all tokens (rows × columns) of the micro-batch:

```
∂ℓ/∂B = s · Σ_tokens δ uᵀ            = s · Σ δ (A x)ᵀ          shape [out, r]
∂ℓ/∂A = s · Σ_tokens (Bᵀ δ) xᵀ                                  shape [r, in]
∂ℓ/∂x = Wᵀ δ + s · Aᵀ (Bᵀ δ)                                    shape [in]   (W used, W gets no gradient)
```

`W` is never formed into `W + sBA`; the effective-weight view is only a way to write the function. The repository computes `‖sBA‖` only as a diagnostic (`trainer.py:742`).

Concrete dimensions (layer 34 or 35):

| projection | x (in) | A | u | B | δ (out) |
|---|---|---|---|---|---|
| q_proj | 2560 | [32, 2560] | 32 | [4096, 32] | 4096 |
| v_proj | 2560 | [32, 2560] | 32 | [1024, 32] | 1024 |
| o_proj | 4096 | [32, 4096] | 32 | [2560, 32] | 2560 |

For one micro-batch of q_proj: the tokens form `X ∈ R^{(6·77)×2560}`, `Δ ∈ R^{(6·77)×4096}`; `∂ℓ/∂B = s Δᵀ (X Aᵀ)` is a `[4096, 462] × [462, 32]` product; `∂ℓ/∂A = s (Δ B)ᵀ X` is `[32, 462] × [462, 2560]`. Note that every token position contributes, including prompt positions: the loss only scores response tokens, but attention lets response positions depend on q/v at prompt positions, so `δ` is nonzero there too.

### 6.1 Trace of one adapted projection, end to end (q_proj of layer 34)

| stage | what happens | where |
|---|---|---|
| init | `A` seeded random (PEFT's init), `B = 0`; both fp32 on GPU | `trainer.py:517, 540`; `layer_stream.py:37-39` |
| forward (micro-batch c) | layer 34 bound to its buffer; `y = W_q x + 2·B(Ax)` computed under no-grad inside node 34 | `backward.py:278` |
| backward | node 34 re-binds layer 34 (hit), recomputes, `autograd.grad(out, [A_q, B_q, A_v, …], grad_out)` | `backward.py:289-294` |
| accumulate | autograd adds the returned `[32,2560]`, `[4096,32]` tensors into `A_q.grad`, `B_q.grad` | autograd leaf accumulation |
| ×5 micro-batches | `.grad` holds `Σ_c ∂part_c/∂θ` = full-batch gradient | `training.py:116-133` |
| check | per-parameter grad norms must be finite | `training.py:135-138` |
| step | `optimizer.step()` (AdamW, lr 2e-4, wd 0); then `zero_grad(set_to_none=True)` | `training.py:141-142` |

### 6.2 Why the first step moves only `B`

At step 1, `B = 0`, so `Bᵀδ = 0` and **`∂ℓ/∂A = 0` exactly** — for all six adapters. AdamW with `g = 0` has `m = v = 0`, update `m̂/(√v̂ + ε) = 0`, and weight decay is 0 (`trainer.py:546`): `A` does not move. `B` receives a nonzero gradient `s Σ δ (Ax)ᵀ`. On Adam's first step `m̂ = g`, `v̂ = g²`, so each element moves by `−lr · g/(|g| + ε) ≈ −2·10⁻⁴ · sign(g)` regardless of the gradient's magnitude. From step 2 on, `B ≠ 0`, so `A` starts receiving gradient. **[code + algebra]**; the trainer logs `grad_norm_A` and `grad_norm_B` separately (`trainer.py:737`), so step 1 of a real run should show `grad_norm_A = 0` — a cheap sanity check to look for in `training.json`.

## 7. Accumulation across micro-batches and when buffers may be reused

```mermaid
sequenceDiagram
  participant O as optimizer
  participant G as LoRA .grad (GPU fp32)
  participant E as executor / buffers
  O->>G: zero_grad(set_to_none=True)  (training.py:115)
  loop c = 1..5 (rows 6c-6 .. 6c-1)
    E->>E: forward sweep, 36 layers (nodes only at 34, 35)
    E->>E: part_c = (6/30) * grpo_loss_c ; part_c.backward()
    E->>G: node 35 then node 34 return grads; autograd ADDS into .grad
    Note over E: graph of micro-batch c freed before c+1 (training.py:130)
  end
  O->>O: finite check, clip_stats, optimizer.step()
  O->>G: zero_grad(set_to_none=True)  (training.py:142)
```

- Gradients are cleared only at the start of `grpo_update` and after the step; between micro-batches they accumulate (`training.py:115, 142`). Because `Σ_c (n_c/N)·loss_c` equals the full-batch loss, the accumulated gradient is the full-batch gradient (07). `tests/test_multistep.py` checks one AdamW step per rollout batch with micro-batching **[test]**.
- A buffer used by node *i*'s backward may be overwritten only after `_release` records its `free` event, which happens after `autograd.grad` returns (R4, 05). Since `del out` happens inside the block and gradients are checked not to alias the buffer, nothing created in backward refers to buffer memory afterwards.
- With 2 buffers, backward of 35 and 34 finds both layers still resident (05 §5), so backward costs no weight copies; the last backward prefetches layer 0 for micro-batch c+1.

## 8. Why the recomputation equals the forward, and what the checksum proves

Bitwise equality of recomputation and forward holds when the same weight bytes, the same input bytes, the same kernels, the same shapes and the same reduction order are used. The design ensures the first four: identical slab bytes (I2, checked by digest at build), the saved input itself, the same HF module, the same micro-batch. Reduction order depends on kernel determinism (e.g. some cuBLAS or SDPA backward paths are nondeterministic; the forward kernels used here are deterministic in practice, which is what the bitwise lab results show) **[test on tiny CPU/CUDA; measured at 0.6B]**.

The checksum `checksum(t) = Σ (bits of t as int64)` (`backward.py:268-270`) is cheap and exact on equal inputs, but it is a **checksum, not a proof**: two different outputs whose bit patterns sum to the same integer would pass. It catches the realistic failures — state changed between forward and backward (adapter disabled or switched, wrong weights bound, a different buffer content) — with overwhelming probability, and it raised in 48 of 80 pre-fix H02 fault configurations (05 §8). It runs `torch.equal(...)`, which synchronizes the host once per recomputed layer.

## 9. Evidence

| claim | evidence | label |
|---|---|---|
| checkpointed backward (sync) gives the same LoRA gradients as resident autograd, bitwise | H02 `t04_gradient_equivalence_cpu`, `t04_gradient_equivalence_cuda` (RTX 3050 Ti): 5,296 comparisons each, threshold 0 | **[test]** tiny random Qwen3 |
| fault: detaching the boundary is caught | `t04_gradient_equivalence_detach_boundary` (status FAIL, execution PASS) | **[test]** |
| one real Qwen3-0.6B GRPO step through L2L equals resident bitwise | `research/l2l/experiments/real_step_20261117T063325Z.json`, `…063548Z.json` | **[measured]** (CUDA, 0.6B) |
| 100-step Qwen3-0.6B training, resident vs `l2l_sync`, identical training rows and evaluations | `research/l2l/experiments/compare_qwen_resident_vs_qwen_l2l.json` (`"equal": true`, `first_differing_step: null`) | **[measured]** (0.6B, sync only) |
| async (prefetching) backward equals sync | `tests/test_prefetch.py:67-90` CPU, `h05_async` after-fix CPU run | **[test]** CPU only |
| anything at 4B | none | **[unverified]** |

Caveat on freshness: `research/l2l/final_report.md` "Evidence freshness" lists engine files changed since `t04_gradient_equivalence_cuda` was produced; the result predates the current code.

## 10. Failure modes and their guards

| failure | what would go wrong | guard |
|---|---|---|
| model or a layer in train mode | dropout draws a different mask in recomputation → different function | refused (`backward.py:317-318`); `lora_config` sets dropout 0 (`model.py:171-173`) |
| double backward / higher-order grads | recomputation graph not built for it | `@once_differentiable` (`backward.py:283`) |
| a LoRA parameter receives no gradient | silent missing update | `allow_unused=False` (`backward.py:294`) |
| mask / RoPE tensors requiring grad | would need differentiating, not supported | refused (`backward.py:314-316`) |
| boundary detached (bug in saving `hidden`) | gradient chain cut below layer 35 | H02 `detach_boundary` fault, detected **[test]**; `boundary()` hook exists only for this injection (`backward.py:307-309`) |
| weights changed between forward and backward (wrong buffer, adapter switched) | gradients of a different function | checksum refusal (`backward.py:291-293`) — only for layers ≥ the floor |
| gradient aliasing a weight buffer | gradient overwritten by the next copy | `_inside` check (`backward.py:296-297`) |
| `grpo_update` retried after a partial accumulation | stale `.grad` | `zero_grad(set_to_none=True)` at entry and on every error path (`training.py:115, 122, 128, 137`) |

## Open questions / unverified
- **Bitwise gradient equality at 4B on the target GPU, sync and async**: `H02.lab_sync_cuda`, `H02.lab_async_cuda` (tiny fixtures on that GPU) and `H05.real_0.6b_bf16` / short real runs with `compare_runs.py`. No 4B gradient comparison is planned in the protocol; a one-step 4B L2L-vs-resident comparison is impossible under 8 GB (resident 4B does not fit), so 4B correctness rests on transfer from tiny/0.6B evidence plus the runtime checks.
- **Determinism of the SDPA/cuBLAS kernels used in the recomputation at 4B shapes** (a nondeterministic kernel would make the checksum fire spuriously and the run fail closed): visible in `H11.smoke` (any checksum refusal stops the run).
- **`grad_norm_A == 0` at step 1** of the real run: a prediction from the algebra above, readable from `runs/qwen3_4b_final/training/training.json` once it exists.
- **Recompute memory peak** (~104 MiB estimate for one layer's local graph): `H11.profile` phase "backward".
