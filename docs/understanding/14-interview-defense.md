# 14: Interview defense

**What you will be able to explain after this file**
- A correct short answer, the mechanism behind it, the code that proves it, and the tempting wrong answer, for ~30 questions a senior engineer is likely to ask.
- Where the system is weak, and how to say so precisely instead of defensively.
- A one-page whiteboard walk-through of one GRPO step at the level of tensors, buffers, streams and function calls.

Evidence labels: **[code]**, **[test]**, **[measured]**, **[predicted]**, **[unverified]**. Line numbers refer to the working tree of 2026-11-18.

Each question has four parts: **Short**, **Deeper**, **Evidence**, **Wrong answer to avoid**.

> The assignment's live-discussion section says: "Don't pre-write answers; we want to think it through with you." Use this file to understand, not to recite. The best answers name a mechanism and a number, and then name the limit of your evidence.

---

## Part A: Architecture choices

### A1. Why not Hugging Face Trainer or TRL's GRPOTrainer?

**Short.** Both assume the model is resident and call `model.generate()` / `model(...)` as black boxes. The assignment requires our own L2L engine, with control over where each layer's weights are at every kernel, a separate copy stream, layer-boundary checkpoints that re-stream weights in backward, and DFlash that needs hidden features from target layers 1/9/17/25/33 mid-forward. None of that is reachable through Trainer's hooks.

**Deeper.** The training math is small (≈150 lines: `qwen_grpo/training.py`, `qwen_grpo/grpo.py`). The hard part is execution. The repo separates them: `backend.logprobs` and `backend.generate_rollouts` are the only things the GRPO code needs from the model (`execution/backend.py:1-17`), so the same `grpo_update` runs on the resident backend (the reference) and on L2L. That separation is what makes the bitwise equivalence tests possible. TRL also runs a reference model for KL by default, which we don't need (β = 0).

**Evidence.** `execution/backend.py:58-87`; `training.grpo_update` takes a `logprobs` callable (`training.py:98-110`); the resident path uses `model.generate` (`policy.py:24-30`), the L2L path never does (`generator.py` docstring; tests patch `model.generate` to fail in `tests/test_dflash.py`).

**Wrong answer.** "Trainer is slow / bloated." Speed isn't the reason; control over weight placement and the autograd graph is.

### A2. Why L2L instead of QLoRA?

**Short.** The assignment says "No quantization is required", and bf16 L2L keeps the base exactly the model being evaluated and merged. QLoRA would train against a 4-bit approximation of W and then merge into bf16 W — a train/serve mismatch — and adds per-layer dequantisation cost.

**Deeper.** Qwen3-4B in NF4 would be ≈2.3 GB and would likely fit resident, so QLoRA is a valid alternative engineering answer; it just answers a different question. Soup ELS v3 (the repo's cited prior work) measured NF4 dequantisation as its largest streaming-specific cost (9.8%). L2L's cost is PCIe traffic: 7.27 GB per full weight sweep. Our design is honest about that cost (≈ 12–14 sweeps per GRPO step, [04](04-l2l-weight-streaming.md)).

**Evidence.** `research/l2l/literature.md` (Soup ELS section); `model.dtype: bfloat16` (`configs/profiles/shared.yaml`).

**Wrong answer.** "QLoRA loses accuracy." That's not established here; say "different trade-off, and the assignment rules it out".

### A3. Why keep frozen weights in pinned CPU memory?

**Short.** A `non_blocking` host-to-device copy is only asynchronous (DMA by the copy engine while the CPU and compute stream continue) if the host source is page-locked. From pageable memory CUDA must first stage through a driver-pinned bounce buffer and the copy is effectively synchronous, so there is no overlap.

**Deeper.** `WeightStore._adopt` allocates one `uint8` slab per layer (201,861,632 B for Qwen3-4B), page-aligns it, page-locks the *whole allocation* with `cudaHostRegister` (not `pin_memory=True`, which rounds each block up to a power of two: +33% measured for one 4B slab), copies the layer's 11 frozen tensors in at 256-byte-aligned offsets, and re-points each `Parameter.data` at its view of the slab. Total pinned ≈ 6.77 GiB. Pinning has a cost: page-locked memory cannot be swapped, and some hosts cap it (`ulimit -l`, WSL2). MegaTrain deliberately does **not** pin the whole model for that reason (pageable store + two pinned staging buffers); this repo pins everything because the assignment says "Weights live in pinned CPU memory".

**Evidence.** `execution/weight_store.py:124-150` (`_adopt`, `cudaHostRegister` at :135); invariants I1–I7 in its docstring; measured pinned 6.58 vs pageable 4.44 GB/s on the laptop (`t02_pinned_transfer`); `copy_nonblocking` docstring ("asynchronous from a pinned slab, synchronous from a pageable one").

**Wrong answer.** "Pinned memory is faster RAM." It's the same RAM; the point is DMA eligibility and asynchrony.

### A4. Why two GPU buffers?

**Short.** With prefetch depth 1, one buffer is being read by the current layer's kernels while the other receives the next layer. `weight_buffers ≥ prefetch_depth + 1` is enforced by config validation. Two is the minimum that allows any overlap; more buffers only help if transfers are bursty relative to compute.

**Deeper.** Each buffer is 201,861,632 B (192.5 MiB), so the pool costs 385 MiB of an 8 GB budget. A third buffer (depth 2) would cost another 192.5 MiB and can't make transfers faster than PCIe bandwidth: in steady state the step is bounded by max(Σ copy, Σ compute). Depth matters only if copy and compute times fluctuate layer to layer. All 36 layers are the same size, so they shouldn't.

**Evidence.** `config.py:303-305` (`weight_buffers >= prefetch_depth + 1`); `configs/profiles/qwen3_4b.yaml` (`weight_buffers: 2`, `prefetch_depth: 1`); `weight_store.gpu_buffers` (:248).

**Wrong answer.** "Double buffering doubles speed." It removes serialization; it cannot exceed the bandwidth bound.

### A5. Why is asynchronous prefetching safe?

**Short.** Four rules, implemented with CUDA events and host-side ownership bookkeeping: a copy into a slot waits for the slot's `free` event (R1); compute waits for the slot's `ready` event (R2); the slot in use and the slots of the upcoming window are never copy targets (R3, checked by `_claim` before anything is enqueued); `free` is recorded after all kernels that read the slot were enqueued (R4).

**Deeper.** The ordering is entirely on the GPU: `copy_stream.wait_event(slot.free)` makes the transfer stream wait for the compute stream's earlier kernels without blocking the CPU; `current_stream().wait_event(slot.ready)` makes compute wait for the copy. The CPU never calls `synchronize` in the pipeline. The buffers are allocated once and never freed, so there is no caching-allocator cross-stream reuse hazard (no `record_stream` needed). Safety of *backward*: in a grad pass, no prefetch is issued after the last forward layer (`_upcoming` returns nothing past layer 35 when `_graph_pass`), so layers 34 and 35 are still in the two buffers when backward starts, and the recomputation reads exactly the bytes the forward read.

**Evidence.** `execution/prefetch.py:7-21` (rules), `_claim` :89-106, `_copy` :108-128, `_acquire` :171-198 (R2 at :189), `_release` :200-208 (R4); `tests/test_prefetch.py` (bitwise vs synchronous executor, CPU); simulation in [05](05-cuda-prefetch-and-buffer-safety.md).

**Wrong answer.** "It's safe because it uses a separate stream." Issuing on two streams creates the *possibility* of a race; the events and ownership rules are what prevent it. Also: the CPU tests can't exercise real stream concurrency.

### A6. What happens if the next layer's weights overwrite the current buffer?

**Short.** The current layer computes with the wrong weights, no error is raised by PyTorch, and the logits and gradients are wrong. Today `_claim` raises `BufferReuseError` before such a copy is enqueued.

**Deeper.** Two cases, which the H02 investigation separated:
- The corrupted layer is ≥ the lowest LoRA layer (34): backward recomputes it with whatever is in the buffer then; the recomputed output's int64 bit-pattern checksum differs from the forward's, and backward refuses.
- The corrupted layer is < 34: it is computed once, never recomputed, so the checksum has nothing to compare. The wrong hidden state flows into the logits, the loss and every LoRA gradient **silently**. Before the fix, 32/80 injected configurations did exactly that, caught only offline by comparison with a resident run.

**Evidence.** `research/l2l/corrections/h02_unsafe_reuse_20261118/README.md`; `prefetch.UnsafeReuse` (:289-305); `backward.py` checksum (:66-75 in the file).

**Wrong answer.** "Backward would detect it." Only for recomputed layers — the whole point of the H02 lesson.

### A7. Why does activation checkpointing help here?

**Short.** Without it, autograd would keep every intermediate of every layer *and references to the weight views*. With it, each layer keeps only its input hidden state, and only layers that have trainable parameters (34, 35) create an autograd node at all. Per micro-batch of 6 rows that's 2 × [6, T, 2560] bf16 ≈ 2 × 2.4 MB.

**Deeper.** In grad mode every layer goes through `_CheckpointedLayer.apply(hidden, *lora)`. For layers 0–33, `hidden` comes from the frozen embedding (no grad) and the layer has no LoRA params, so no input requires grad: PyTorch creates no graph node, and `ctx` (with its saved hidden) is dropped immediately. Layer 34 has LoRA inputs → node; its output requires grad → layer 35 also has a node and needs `dL/dhidden_35`. So the "activation memory" problem nearly disappears for this LoRA placement. More importantly than memory, checkpointing is a **correctness** requirement: the forward runs with grad off, so autograd never saves a view of a buffer that will be overwritten.

**Evidence.** `execution/backward.py` docstring and `CheckpointedLayerExecutor.execute` (:93-102); `layer_stream.StreamingLayerExecutor.execute` refuses grad mode (:116-119 in that file); budget lists boundary checkpoints at ≈ 9 MiB.

**Wrong answer.** "Checkpointing saves activations for all 36 layers and recomputes all of them." Here 34 layers never get recomputed.

### A8. How does backward work without retaining decoder weights on the GPU?

**Short.** Each node's backward re-binds its layer's weights to a pool buffer (`executor.bound(index, layer, phase="backward")`), recomputes the layer from the saved input with grad enabled, and calls `autograd.grad(out, (h, *lora), grad_out)` inside that binding. With two buffers and this LoRA placement, layers 35 and 34 are still resident from the forward, so backward costs **zero** copies.

**Deeper.** Chain rule for one layer f_i with frozen W_i and LoRA θ_i: given g_{i+1} = ∂L/∂h_{i+1}, the node returns g_i = J_h f_i(h_i)ᵀ g_{i+1} and ∂L/∂θ_i = J_θ f_i(h_i)ᵀ g_{i+1}. Both need W_i (every projection's transpose appears in J_h), which is why the weights must be on the GPU during backward even though they receive no gradient. Verified schedule (6-layer analogue, CPU): forward L0…L5 in buf0/buf1 alternating, backward L5 (hit, buf1), L4 (hit, buf0), and during backward of L4 the next micro-batch's layer 0 is prefetched into buf1.

**Evidence.** `backward.py` (`_CheckpointedLayer.backward`); `prefetch._upcoming` (:137-150); simulation in [05](05-cuda-prefetch-and-buffer-safety.md) / [06](06-checkpointed-backward.md). The budget calculator assumes two layer reloads per micro-batch in backward (2.02 GB per step) — the code does not do them; the prediction is conservative there.

**Wrong answer.** "Backward streams all 36 layers in reverse." It streams none here.

---

## Part B: Batch shapes and hyperparameters

### B1. Why G = 6?

**Short.** G must be ≥ 2 for a group std. Larger G means more completions share a baseline (lower-variance advantage) and a higher chance that a group has mixed rewards, but each extra completion costs rollout and two teacher-forced passes. B·G = 30 was inherited from the 0.6B reference run.

**Deeper.** If a prompt's base accuracy is p, a group is zero-signal with probability pᴳ + (1−p)ᴳ. At p = 0.75 (base 0.6B validation accuracy): G = 6 → 0.178 + 0.0002 ≈ 18%; G = 2 → 63%. A batch is skipped only if all 5 groups are uniform. On 0.6B, 42 of 142 attempts (30%) were skipped **[measured]**, so in practice the signal rate is lower than the independent-group estimate (easy prompts have p near 1).

**Evidence.** `config.py:243` (`group_size >= 2`); `runs/qwen/report.md`; `training.prepare` skip rule (`training.py:90`).

**Wrong answer.** "6 was tuned." It wasn't tuned at 4B.

### B2. Why micro-batches of 6? Does it reduce the rollout batch?

**Short.** Memory only. The 30 completions are split into 5 chunks for the teacher-forced old-log-prob pass and the update; each chunk's loss is scaled by 6/30 and gradients accumulate, so the gradient is the full-batch gradient and there is exactly one AdamW step. It does **not** change rollout generation, which still runs all 30 rows at once.

**Deeper.** `grpo_loss` on a chunk returns −(1/n_c) Σ_{i∈c} J_i; multiplying by n_c/N and summing gives −(1/N) Σ_i J_i. The cost is in streaming: the engine is micro-batch-major, so each chunk streams all 36 layers again: 5 sweeps for old log-probs + 5 for the update = 10 × 7.27 GB = 72.7 GB of H2D per step, versus 2 sweeps with `micro_batch_size: null`. The old log-probs are computed in the same chunks as the update because bf16 results depend on batch shape, and iteration 0 needs ratio exactly 1.

**Evidence.** `training.py:74-95` (`micro_batches`, `prepare`), :116-133 (scaling, `part.backward()` per chunk); `tests/test_multistep.py`; `qwen3_4b_final.yaml` comments.

**Wrong answer.** "Smaller micro-batches mean smaller rollouts / a different effective batch / more optimizer steps."

### B3. Why LoRA only on the final two layers (q, v, o)?

**Short.** (a) Backward cost: the lowest adapted layer sets the recomputation floor; adapting layer 0 would mean recomputing all 36 layers in backward. (b) It reproduces the 0.6B reference placement (`last2-attention`). 1.08M trainable parameters (0.027%).

**Deeper.** The cost is capacity: the adapter can only change the last two layers' attention, i.e. it can re-read the final representations but not change how the sentence is encoded. For sentiment, where the base already gets ~75% (0.6B), steering the output head may be enough; at 4B unknown. Dimension check per layer (r = 32): q A[32,2560] B[4096,32]; v A[32,2560] B[1024,32]; o A[32,4096] B[2560,32] → 540,672 per layer.

**Evidence.** `configs/profiles/qwen3_4b.yaml` (`num_layers: 2`); `model.target_modules`; trainer asserts the count (`trainer.py:533`).

**Wrong answer.** "Last layers matter most for classification" (as a fact). It's a cost decision with a plausible but untested capacity assumption.

### B4. Why GRPO for binary sentiment classification?

**Short.** Honestly: because the assignment asks for GRPO + LoRA on a task with an automatic verifiable reward that moves within a few dozen steps. Exact-match label reward is the simplest such reward. As a classifier-training method, supervised cross-entropy on the labels would be the obvious, cheaper, lower-variance baseline.

**Deeper.** With a binary reward and a 2-token label, GRPO is essentially REINFORCE on a 2-arm bandit per prompt, with a group baseline. Its information per sample is 1 bit, versus log-loss on the gold label. The interesting part of the project is the systems work and the verification, not the learning algorithm's fit to the task. Reward subtleties: `parse_label` uppercases and strips, so "Positive" (1 token) is as valid as "POSITIVE" (2 tokens: `POS`, `ITIVE`).

**Evidence.** `sentiment.py:35-42`; assignment PDF.

**Wrong answer.** "GRPO is better than supervised learning for this."

### B5. Why KL β = 0?

**Short.** No reference model forward is needed (saves one more set of weight sweeps per step, ≈ 5 × 7.27 GB), and with one update per batch, few steps and a small LoRA, the policy stays close to the base anyway. Drift is monitored instead (`approx_kl`, prediction bias, held-out accuracy).

**Deeper.** Recent GRPO variants commonly drop the KL term. The risk is reward over-optimization and collapse, which is exactly what the 0.6B run showed at lr 5e-4 (predicted-POSITIVE rate 0.968 at step 100). Note the logged `approx_kl` is computed at the pre-step parameters for iteration 0 and is therefore 0 in this config — it is not monitoring anything.

**Evidence.** `grpo.grpo_loss` (no KL term); `trainer.py:574` (`kl_beta: 0.0`); `grpo.clip_stats`.

**Wrong answer.** "KL is unnecessary because LoRA can't drift." LoRA can collapse a classifier; 0.6B did.

### B6. Why does the clip potentially have no effect with `num_iterations: 1`?

**Short.** The only update happens at parameters equal to those that produced `old_logp`, so every ratio r = exp(logp − old_logp) is exactly 1 (checked: `init_diff ≤ 1e-6`, in practice 0). At r = 1 both branches of min(r·A, clip(r)·A) have the same value and gradient, so the clip never binds and the gradient is the length-normalized REINFORCE gradient with the group baseline, scaled by 1/std.

**Deeper.**
- ∇J = (1/N) Σ_i (A_i/|o_i|) Σ_t ∇ log π(o_{i,t}). Verified numerically: the gradient of `grpo_loss` at ratio 1 equals this expression to 4e-9.
- Mechanism: `torch.minimum` sends half the gradient to each input at a tie; `clamp` passes the gradient through for 0.8 ≤ r ≤ 1.2. The two halves add back up to A·∇r.
- "Implementing a clipped objective" vs "exercising it": the clip would only act on iterations ≥ 1 (num_iterations > 1), where the policy has moved. The lab tests exercise that (3 iterations, "some tokens must clip"); training never does.
- The logged `clip_frac`/`ratio_*` are computed from log-probs gathered *before* `optimizer.step()`, so at iteration 0 they are exactly 1 / 0 by construction.
- Consequence: the loss value is ≈ 0 every step (Σ_i A_i = 0 per group, per-sequence mean of a constant), e.g. `1.19e-08` in `real_step_*`. The assignment says the same: "GRPO loss sits near 0 by construction".

**Evidence.** `training.py:118-123` (INIT_TOL), :139 (`clip_stats` on `new_logp` collected before the step); `grpo.py` `grpo_loss`; scratch check `clip_check.py` (described in [07](07-grpo-mathematics.md)).

**Wrong answer.** "Clipping keeps the update stable." Not in this configuration; stability comes from the lr and Adam.

### B7. What is the effective learning signal per step with binary rewards?

**Short.** Advantages per group depend only on k = number correct out of 6:

| k | correct A | wrong A |
|---|---|---|
| 0 or 6 | 0 | 0 |
| 1 | +2.235 | −0.447 |
| 2 | +1.414 | −0.707 |
| 3 | +1.0 | −1.0 |
| 4 | +0.707 | −1.414 |
| 5 | +0.447 | −2.235 |

**Deeper.** Rare successes get large positive advantage; rare failures get large negative advantage. Std normalization makes every mixed group contribute comparable magnitude regardless of how informative it is (a k = 5 group from an easy prompt gets as much "push" as a k = 3 group). Uniform groups inside a mixed batch contribute zero gradient but still count in the 1/N mean, so the step size shrinks with the fraction of uniform groups. Then Adam rescales everything per element anyway (B9).

**Evidence.** `grpo.group_advantages` (population std, eps 1e-4); scratch output in [07](07-grpo-mathematics.md).

**Wrong answer.** "Advantages are rewards minus mean." They are also divided by the group std.

### B8. Is 60 steps × 5 prompts meaningful training?

**Short.** It's a demonstration-scale run: at least 300 prompt selections from a 1024-prompt pool (≤ 29% coverage, fewer prompts per update if zero-signal batches occur), one update per batch. The config itself says "The goal is a correct, memory-bounded end-to-end run, not maximal accuracy."

**Deeper.** On 0.6B, validation accuracy changed within ±4 points over 100 steps before collapsing. Whether 60 steps at lr 2e-4 moves 4B measurably is exactly H12, untested. Training also records actual coverage (`coverage()`), so no "epoch" claim is made.

**Evidence.** `configs/experiments/qwen3_4b_final.yaml` header; `trainer.coverage` (:267-280).

**Wrong answer.** "We trained for an epoch."

### B9. What does AdamW do on step 1?

**Short.** LoRA B is initialized to 0 and A randomly, so at step 1 ∂L/∂A = (α/r)·Bᵀ·δ·xᵀ = 0: A does not move. B gets m̂/(√v̂+ε) = g/|g| ≈ sign(g), so every B element moves by ≈ lr = 2e-4 regardless of the gradient's magnitude.

**Deeper.** Adam's bias correction makes step-1 updates pure sign updates. The 1/std advantage scaling and the 6/30 micro-batch scaling are irrelevant to step size at step 1 and largely irrelevant later (Adam is scale-invariant per element up to ε). Weight decay is 0, no gradient clipping.

**Evidence.** `trainer.py:546` (`AdamW(..., weight_decay=0.0)`); `trainer.py:540` asserts B = 0 at init.

**Wrong answer.** "The first step changes both A and B."

### B10. Why did 0.6B collapse at lr 5e-4, and is 2e-4 safe?

**Short.** Observed, not explained: at lr 5e-4 the 0.6B predicted-POSITIVE rate oscillated (0.36 → 0.76 → 0.37) and ended at 0.968. 2e-4 is a reasonable guess (smaller sign-sized steps) but **untested at any scale**.

**Deeper.** Plausible mechanism: with binary rewards, if the model slightly favors one label, groups on prompts of that label become k = 6 (zero signal) while groups on the other label have rare successes. Adam's sign-like steps on a 2-way output can overshoot. There is no KL anchor. An lr sweep at 0.6B (cheap) would have been the right experiment before 4B.

**Evidence.** `runs/qwen/report.md` §6; `qwen3_4b_final.yaml` comment.

**Wrong answer.** "2e-4 fixes the collapse."

---

## Part C: Correctness and evidence

### C1. How do you know the gradients are correct?

**Short.** Bitwise equality with a resident reference running the same GRPO code: on tiny random Qwen3 models across shapes/dtypes/LoRA placements/buffer counts (H02, H05), on two real Qwen3-0.6B bf16 updates, and over a 100-step 0.6B training run (sync L2L). Plus runtime guards: the iteration-0 `INIT_TOL` check, the recomputation checksum, ownership checks.

**Deeper.** Bitwise is possible because the L2L path calls the same HF/PEFT modules with the same bytes at the same alignment. Fault controls show the comparisons have power (wrong layer, detached boundary, corrupted advantage all caught). Limits: async L2L is only tested on CPU (no streams); nothing at 4B; the evidence predates later engine edits (freshness table); bitwise equality to resident proves L2L = resident, not that the resident GRPO math is right — that is covered by unit tests of `grpo_loss` and the REINFORCE identity.

**Evidence.** `t04_*`, `t07_*`, `real_step_*`, `compare_qwen_resident_vs_qwen_l2l`; `tests/test_prefetch.py`; [11](11-research-hypotheses.md).

**Wrong answer.** "The loss went down." (It's ≈ 0 by construction.)

### C2. Why bitwise equivalence as the gate rather than a tolerance?

**Short.** Because the mechanism predicts exact equality, and anything weaker hides real bugs: Ticket 00 measured that a one-ulp change of a single weight changes no output element, so small tolerances can't distinguish "right" from "slightly wrong".

**Deeper.** Where kernel shapes legitimately differ (chunked head, KV-cached vs full-prefix logits, merge rounding), the repo derives non-zero tolerances in `tolerances.yaml` with a written `why`. The cost of bitwise gates: they are brittle across hardware (a different GPU may pick different kernels), which is fine because both sides run on the same device.

**Evidence.** `research/l2l/tolerances.yaml`; `research/l2l/literature.md` ("Risk" under L2L).

**Wrong answer.** "Bitwise is overkill."

### C3. Why is `old_logp` not literally the sampling distribution?

**Short.** Tokens are sampled from logits computed incrementally with the KV cache (or in DFlash 4-token verify blocks). `old_logp` is a teacher-forced recomputation of the whole sequence in micro-batches of 6. In bf16 those are different kernel shapes, so they can differ in low bits. The `INIT_TOL` check only compares `old_logp` with the update's teacher-forced log-probs.

**Deeper.** This means the importance ratio is between two teacher-forced evaluations of the same parameters (exactly 1), not between the update policy and the true behaviour policy. The mismatch is a small off-policy error, the same one TRL/vLLM setups have. H04 bounds the cached-vs-prefix difference on tiny models (`prefix_logits` rtol 2⁻⁶ for bf16). At 4B it is unmeasured.

**Evidence.** `generator._step` vs `output_head.streamed_logprobs`; `training.prepare` docstring; `tolerances.yaml generation_equivalence`.

**Wrong answer.** "old_logp are the probabilities the sampler used."

### C4. How can DFlash preserve the target distribution?

**Short.** At every block position the target draws its own token y_j from its own softmax given the prefix (which includes the drafts before j). The row accepts drafts while draft_j = y_j and then emits y at the first mismatch. Every emitted token is a target draw given the emitted prefix, so the sequence has the target's law, whatever the drafter proposes.

**Deeper.** For an argmax drafter this equals standard speculative sampling (accept with probability p(draft), otherwise sample from p restricted to ≠ draft). Batching: rows advance by m = min over unfinished rows; a row that matched further is cut at m and emits y_m, which equals its draft — still its own draw. The stopping index depends only on draws already made, so the sequence remains a product of correct conditionals. Caveats: the target logits come from a 4-column block pass, a different kernel shape than plain decoding (same law in exact arithmetic, low-bit differences in bf16); the RNG stream differs, so samples aren't bitwise those of plain sampling. Tested by chi-square on tiny models; nothing on 4B.

**Evidence.** `dflash.py:14-22`, :211-242; `tests/test_dflash.py`.

**Wrong answer.** "The drafter is trained to match the target, so it's close enough." Exactness comes from the accept rule, not drafter quality.

### C5. Why might DFlash give little or no benefit here?

**Short.** At most 4 new tokens. Plain decoding needs up to R sweeps (3 for "POSITIVE<|im_end|>"); DFlash needs ≥ 2 (prefill + one verify). That saves ≈ 1 of ≈ 13 weight sweeps per GRPO step at best **[predicted]**, while adding a 1.07 GB resident drafter and an LM head over all prompt positions in the prefill.

**Deeper.** The prefill `_target_step` applies `lm_head` to all P columns of 30 rows: [30, P, 151,936] in bf16 plus its fp32 copy ≈ 2.0 GB at P = 73; the budget predicts this as the rollout peak (≈ 6.36 GiB device). Only the last column is used for the first token. Acceptance is the batch minimum over 30 rows, so a single disagreeing row forces m = 0 — though here the label suffix (`ITIVE`, `<|im_end|>`) is nearly deterministic once `POS` is drawn, so m ≥ 2 is plausible **[unverified]**.

**Evidence.** `dflash.py:167` (`lm_head(decoder.norm(hidden)).float()` on all columns); budget prediction file; [08](08-dflash-integration.md).

**Wrong answer.** "DFlash makes rollouts 3× faster."

### C6. What did H02 teach you?

**Short.** That a runtime check comparing recomputation to forward protects only recomputed layers, and that the layers below the lowest LoRA layer — 34 of 36 here — would be corrupted silently. Protection must be structural: refuse an illegal copy before it is enqueued.

**Deeper.** Also a harness lesson: the lab judged a correct refusal as "not detected" because it landed in `errors`. A test harness can be wrong in the direction of hiding a fix as well as hiding a bug.

**Evidence.** corrections README; `prefetch._claim`; `lab.record_refusal` / `check_fault_detected`.

**Wrong answer.** "H02 failed, so the backward is wrong." The ordinary backward was never wrong; the fault-injection control exposed missing protection against a hypothetical scheduler bug.

---

## Part D: Memory

### D1. What exactly is in the 8 GB budget?

**Short.** Everything nvidia-smi attributes to the process: CUDA context (driver + kernels, assumed 0.5 GiB), the PyTorch caching allocator's **reserved** memory (not just allocated), cuBLAS/cuDNN workspaces, and any non-PyTorch allocations. Inside the allocator: embedding/tied head 741.9 MiB, two weight buffers 385 MiB, DFlash drafter ≈ 1.0 GiB, LoRA + grads + AdamW ≈ 16 MiB, KV cache ≈ 0.3–0.5 GB, activations and logits.

**Deeper.** Two thresholds exist: the orchestrator's telemetry watchdog judges **8e9 B** (7.45 GiB), sampling nvidia-smi every 200 ms; the in-process `memory.guard` and H11's criterion use **8 GiB = 8.59e9 B** (`trainer.py:408`), and `allocator_cap` limits only the allocator to 8 GiB of the device (excluding context). So the strict reading is enforced only from outside the process, and a spike shorter than a sampling interval can be missed by both. Predicted device peak 6.36 GiB (DFlash prefill) **[predicted]**, never measured.

**Evidence.** `configs/studies/assignment.yaml` `gpu_memory_cap`; `trainer.memory_guard` (:400-415); `execution/profiler.py` docstring; `telemetry.py`.

**Wrong answer.** "torch.cuda.max_memory_allocated() is below 8 GB, so we fit."

### D2. Where is the peak and why?

**Short.** Predicted at the DFlash rollout prefill: the full-sequence LM head [30, P, V] in bf16 and fp32 on top of the drafter, the KV cache and the resident set. Training is predicted far lower (≈ 3 GiB) because micro-batches of 6 and response-only logits keep everything small.

**Deeper.** See [09](09-memory-and-tensor-lifetimes.md). The fix would be one line in spirit — apply the head to the last column in the prefill and keep features only — but that would change the code under test, which this document set doesn't do.

**Evidence.** budget prediction "rollout (DFlash): prefill LM head over all prompt positions".

**Wrong answer.** "The peak is in backward."

---

## Part E: Systems costs

### E1. Why is the schedule micro-batch-major, and what does it cost?

**Short.** Because the executor is called per forward pass: each chunk of 6 rows runs the full 36-layer loop, so every chunk re-streams every layer. Per step: 5 sweeps (old log-probs) + 5 (update) = 72.7 GB, plus 2–4 sweeps for rollouts.

**Deeper.** L2L's Algorithm 2 inverts the loops (layer outer, micro-batches inner), so each layer is loaded once per pass for all micro-batches; that cuts 10 sweeps to 2, at the price of holding all 5 micro-batches' boundary activations at once and restructuring autograd (backward also layer-major). Not implemented. At an assumed 12 GB/s (unmeasured), 72.7 GB is ≥ 6 s of transfer per step **[predicted]**.

**Evidence.** `qwen3_4b_final.yaml` comment "Scheduling is micro-batch-major"; `literature.md` (L2L §2.2).

**Wrong answer.** "Micro-batching is free."

### E2. Why is the prompt prefill duplicated 6×?

**Short.** `generate` / `dflash_generate` expand the 5 prompts to 30 rows with `repeat_interleave` *before* the prefill, so each prompt is run through all 36 layers 6 times and its KV is stored 6 times. No prefix sharing.

**Deeper.** Weights are streamed once per sweep regardless of row count, so duplication costs compute and KV memory, not PCIe. Prefix sharing (prefill 5 rows, copy KV to 30) would cut prefill compute 6× and KV memory for the prompt part. HF `generate(num_return_sequences)` does the same duplication, which keeps H04's bitwise comparison simple.

**Evidence.** `generator.py:115-116`; `dflash.py:193-194`.

**Wrong answer.** "The six responses share the prompt's KV cache."

### E3. What happens to all-equal groups inside a mixed batch?

**Short.** Their advantages are exactly 0 ((r − mean) = 0, divided by 0 + 1e-4), so they contribute no gradient, but they are still generated, scored twice, and counted in the 1/N mean, diluting the step.

**Evidence.** `grpo.group_advantages`; `training.prepare` skips only when *every* group is uniform (`(rewards == rewards[:, :1]).all()`).

**Wrong answer.** "Zero-signal groups are filtered out."

### E4. What would you change first with more time / three days on H100s?

**Short.** In order: (1) run stages B–D on the target GPU and get assignment-grade nvidia-smi evidence; (2) measure async overlap and the step-time split between rollout, old log-probs and update; (3) remove the DFlash full-prompt LM head; (4) layer-major scheduling; (5) an lr sweep and a shuffled-reward control at 0.6B; (6) per-row ragged DFlash acceptance.

**Wrong answer.** Listing features before measurements.

---

## Part F: What remains unverified

| Assumption | Status | What would settle it |
|---|---|---|
| Async L2L is exact on a GPU | **[unverified]** (CPU only) | `H0{1,2,4,5}.lab_async_cuda` |
| Copy and compute actually overlap | **[unverified]** | `H06.prefetch_timing` timeline events |
| 4B fits 8e9 B per nvidia-smi | **[predicted]** 6.36 GiB | `H11.smoke`, `H11.profile`, `H10.smoke_dflash` |
| DFlash greedy = plain on 4B; acceptance; speed | **[unverified]** | `H10.dflash_bench`, `H10.dflash_drift` |
| Host RAM suffices for pinning 6.77 GiB + loading copy | **[unverified]** | stage D RSS samples |
| lr 2e-4 avoids collapse / learns | **[unverified]** | `H12.train_final`, `H12.final_eval` |
| Real-model merge equivalence | **[unverified]** | `H09.merge_qwen3_0.6b` |
| Current code equals the code behind cited evidence | **false** (freshness table) | fresh stage B |
| Behaviour-policy vs teacher-forced mismatch is negligible at 4B | **[unverified]** | compare sampler step logits with `old_logp` on real rollouts |

---

## The whiteboard script: one GRPO step

1. **Sample.** `PromptStream.next_batch()` reads 5 JSON records by byte offset from `streaming/prompts.jsonl`, following an epoch permutation from `random.Random(42)`; returns records + `(epoch, position, id)` occurrences. `attempts += 1`.
2. **Template.** `build_prompts` applies the Qwen3 chat template with `enable_thinking=False` (ends `<think>\n\n</think>\n\n`). Tokenizer, left padding: `input_ids [5, P]` int64, `attention_mask [5, P]` (P ≈ 51–93), moved to GPU.
3. **Expand.** `dflash_generate`: `repeat_interleave(6)` → `[30, P]`; rows 6b…6b+5 belong to prompt b. KV cache: 36 layers × K and V `[30, 8, P+20, 128]` bf16 on GPU.
4. **Prefill.** `_target_step` over all P columns: embedding (resident) → 36 layers via `PrefetchingCheckpointedExecutor` under `no_grad`: for layer i, compute waits on `slot.ready`; meanwhile the copy stream copies slab i+1 (192.5 MiB, pinned → GPU) into the other buffer after waiting on its `free` event. Features after layers 1/9/17/25/33 kept. LM head on all P columns → fp32 logits; first token = `multinomial(softmax(last column))`.
5. **Draft and verify.** Block `[tok1, mask, mask, mask]`; drafter (resident, 5 layers) proposes 3 tokens via the target's LM head argmax; target verifies the 4-column block in one more 36-layer sweep; accept the common prefix m (min over rows), emit drafts + y_m. Repeat until 4 tokens or all rows hit EOS. Output `[30, P+R]`, R ≤ 4.
6. **Reward.** Decode up to EOS; `reward = 1` iff the stripped, uppercased text is exactly POSITIVE/NEGATIVE and equals the label → `rewards [5, 6]`. If every group is uniform: write the record, skip, next attempt.
7. **Advantages.** `(r − mean_g)/(std_g + 1e-4)` → `[5, 6]` → `adv [30]` fp32.
8. **Old log-probs.** `prepare`: under `no_grad`, 5 chunks of 6 rows, each a full 36-layer streamed forward (`explicit_hidden`), final norm, LM head on the last R+1 columns `[6, R+1, 151936]`, `log_softmax` fp32, gather → `old_logp [30, R]`. Record written to `streaming/rollouts/group_N.pt` (fsync + rename).
9. **Update.** `grpo_update`: `zero_grad`; for each chunk of 6: grad-mode forward — layers 0–33 build no node; layers 34, 35 are `_CheckpointedLayer` nodes saving `[6, T, 2560]` inputs; no prefetch after layer 35. Logp `[6, R]`; check `|logp − old_logp| ≤ 1e-6`; loss = `grpo_loss × 6/30`; `backward()`: head → norm → node 35 (buffer hit: recompute, `autograd.grad`) → node 34 (hit; layer 0 of the next chunk prefetched into the freed buffer) → LoRA `.grad` accumulates (fp32, 1.08M params).
10. **Step.** Grad norms checked finite; one `AdamW.step()` (lr 2e-4, wd 0); `zero_grad(set_to_none=True)`. `step += 1`.
11. **Every 20 steps.** Greedy evaluation on the 128-prompt validation subset (plain streamed greedy, batch 16, no DFlash); then `stream_resume.publish`: adapter + `training_state.pt` (AdamW, sampler state, RNG, log) into a temp dir, fsync, `COMMIT.json` with sha256 of every file, atomic rename. Only now are steps up to here committed.
