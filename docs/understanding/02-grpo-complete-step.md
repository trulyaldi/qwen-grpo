# 02 — One complete GRPO iteration, followed end to end

**What you will be able to explain after this file**
- The exact sequence of function calls of one training step in `trainer._train`, from reading five prompts to publishing a checkpoint.
- For each stage: inputs, outputs, shapes, dtypes, device, whether autograd records anything, and what is allocated, recomputed, freed and written to disk.
- Where the step can abort (zero-signal batch, log-prob mismatch, non-finite loss or gradient), and what state is left behind.

This file follows **one** rollout batch, the first batch of the final run (prompt ids 1373, 35, 882, 558, 1003; [01-batches-groups-microbatches.md](01-batches-groups-microbatches.md) shows how they were drawn). Token-level facts (P = 73, label tokenization) are real **[code]**; the sampled responses and rewards are **illustrative**, chosen to exercise every branch, and are marked as such. The math behind stages 6–12 is derived in [07-grpo-mathematics.md](07-grpo-mathematics.md); the engine behind every "36-layer sweep" is in [04](04-l2l-weight-streaming.md)–[06](06-checkpointed-backward.md).

---

## Before the first step (run start, once)

`trainer.train` → `_train` (`qwen_grpo/trainer.py:347-392`, `491-668`):

1. `check_device` refuses a GPU without kernels or without bf16 (`trainer.py:429-447`). No fallback.
2. `torch.manual_seed(42)`, then `lora_model(bfloat16, "cpu", ...)` (`trainer.py:517-522`, `qwen_grpo/model.py:176-182`): the **whole** Qwen3-4B is loaded on the CPU (≈ 8.0 GB bf16, pageable), PEFT wraps q/v/o of layers 34, 35 with LoRA A ~ PEFT's default init (seeded), B = 0.
3. `make_backend` (`qwen_grpo/execution/backend.py:90-111`): `L2LBackend` adopts the 36 decoder layers' frozen weights into a pinned CPU `WeightStore` (one 201,861,632-byte slab per layer), moves embedding/tied head, final norm, rotary buffers and LoRA tensors to the GPU, allocates two 201,861,632-byte GPU buffers, builds the `PrefetchingCheckpointedExecutor`, then loads the DFlash drafter onto the GPU and installs `DFlashGenerator` as the rollout sampler ([04](04-l2l-weight-streaming.md), [08](08-dflash-integration.md)).
4. Assertions: 1,081,344 trainable parameters, all fp32, all `lora_A`/`lora_B`; every B is exactly zero, so the initial policy **is** the base model (`trainer.py:527-540`).
5. `AdamW(lr=2e-4, weight_decay=0.0)` (`trainer.py:546`); default betas (0.9, 0.999), eps 1e-8; no gradient clipping.
6. Step-0 adapter published crash-consistently (`trainer.py:555`), validation-subset evaluation at step 0 (`trainer.py:590`), `torch.manual_seed(42)` again for rollout sampling (`trainer.py:593`).
7. `PromptStream.build` and `RolloutStore` under `runs/qwen3_4b_final/streaming/` (`trainer.py:644-648`).

State on the GPU now: embedding 741.9 MiB, two buffers 385 MiB, drafter ≈ 1.0 GiB, LoRA 4.1 MiB, no `.grad`, no AdamW moments yet (PyTorch's AdamW creates them lazily at the first `step()`). Full accounting: [09-memory-and-tensor-lifetimes.md](09-memory-and-tensor-lifetimes.md).

---

## The step, stage by stage

The loop body is `trainer.py:672-767`. `step = 0`, `attempts = 0`, `pending = None`.

### 1. Read training examples

```python
batch, occurrences = prompts_on_disk.next_batch()      # trainer.py:679
attempts += 1                                           # trainer.py:683
```

- **Function:** `PromptStream.next_batch` (`streaming/dataset.py:105-116`).
- **Out:** five dicts `{"id", "label", "text"}` read from `prompts.jsonl` by byte offset; `occurrences = [(0,0,1373), (0,1,35), (0,2,882), (0,3,558), (0,4,1003)]`.
- **Device:** CPU, Python objects. **Disk:** read only. **Crash point** `prompt_read` (`trainer.py:680`).

### 2. Construct tokenized prompts

```python
prompts = build_prompts(tok, [e["text"] for e in batch])                 # trainer.py:684
```

`build_prompts` (`sentiment.py:23-32`) returns five chat-formatted strings (non-thinking). Tokenization happens inside the generator wrapper `policy.generate_rollouts` (`policy.py:19-20`): `input_ids [5, 73]` int64, `attention_mask [5, 73]` int64, moved to the GPU (`.to(model.device)`; the model's device is the GPU because the embedding lives there).

### 3. Generate sampled completions

```python
rollouts = backend.generate_rollouts(tok, prompts, G=6, max_new_tokens=4)   # trainer.py:686
```

`L2LBackend.generate_rollouts` → `policy.generate_rollouts(..., generator=self.sampler)` → `DFlashGenerator.__call__` → `dflash_generate` (`execution/dflash.py:179-245`). Under `@torch.no_grad`:

| sub-step | target input | weight sweeps | output |
|---|---|---|---|
| expand | `[5,73]` → `repeat_interleave(6)` → `[30, 73]` | 0 | prompt-major rows |
| prefill | `[30, 73]` through 36 streamed layers; features after layers 1, 9, 17, 25, 33 kept | 1 | first token per row, drawn by the target; KV cache filled to 73 |
| draft + verify (per round) | drafter proposes ≤ 3 tokens; target verifies a block `[30, q]` | 1 per round | accepted prefix + one target draw per row |

Rounds continue until 4 tokens or all rows hit EOS. Every token in the output is the **target's** own draw ([08-dflash-integration.md](08-dflash-integration.md)). After return, `policy.generate_rollouts` builds `response_ids`, `completion_mask`, `attention_mask` (`policy.py:32-41`).

- **Autograd:** off. **Allocated:** KV cache `[30, 8, 73+4+16, 128]` bf16 × 2 × 36 layers ≈ 411 MB at P = 73 (freed when the generator returns), the prefill LM-head output over **all** 73 positions in bf16 and fp32 (≈ 2.0 GB transient, the predicted peak of the whole run: [09](09-memory-and-tensor-lifetimes.md)).
- **Kept:** `rollouts` dict on the GPU (a few kB). `timing["rollout_generator"]` records which generator ran and its acceptance statistics (`trainer.py:461-475, 690`).
- **Crash point:** none here; see stage 9.

**Illustrative output** with R = 4 (T = 77), written as decoded labels per prompt group:

```text
prompt 1 (POSITIVE): POS POS POS POS NEG POS        -> 5 of 6 correct
prompt 2 (POSITIVE): POS POS POS POS POS POS        -> 6 of 6
prompt 3 (NEGATIVE): NEG POS POS NEG POS POS        -> 2 of 6
prompt 4 (NEGATIVE): POS POS POS POS POS POS        -> 0 of 6
prompt 5 (NEGATIVE): NEG NEG POS (truncated) NEG POS -> 3 of 6   (the truncated row has 4 tokens, no EOS, reward 0)
```

### 4. Check grouping and decode the responses

```python
check_grouping(rollouts, tok(prompts, ...)["input_ids"].to(model.device), G)   # trainer.py:691
responses = decode(tok, rollouts)                                               # trainer.py:692
```

`check_grouping` re-tokenizes and asserts rows `6b..6b+5` begin with prompt b (`training.py:50-53`). `decode` (`training.py:22-28`) keeps each row's masked tokens and drops a terminating EOS: `"POSITIVE"`, `"NEGATIVE"`, `"Positive"`, or a truncated string. CPU strings.

### 5. Compute deterministic rewards

```python
rewards = group_rewards(responses, [e["label"] for e in batch], G)   # [5, 6], trainer.py:693
```

`sentiment.reward` (`sentiment.py:41-42`) is `1.0` iff `parse_label(response) == label`, where `parse_label` strips, upper-cases, and accepts only the exact strings `POSITIVE` / `NEGATIVE` (`sentiment.py:35-38`). "positive." with a period earns 0; "Positive" earns 1. Rewards are deterministic given the text; the only randomness is in the sampling of stage 3.

### 6. Reshape into groups

`group_rewards` builds `[30]` then `.reshape(5, 6)` (`training.py:67-70`); row-major reshape is correct **only because** rows are prompt-major (stage 4 checked it). Illustrative:

```text
rewards = [[1, 1, 1, 1, 0, 1],
           [1, 1, 1, 1, 1, 1],
           [1, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 1, 0]]
```

The trainer counts groups whose rewards are not all equal: `mixed = 3` (`trainer.py:694-695`) and builds the `group` log record (`trainer.py:696-700`): prompt ids, labels, responses, rewards, mean reward 0.533, invalid responses, a 16-hex-digit hash of `response_ids`, mean completion length.

### 7. Group-relative advantages

Inside `prepare` (`training.py:84-95`):

```python
adv = group_advantages(rewards).flatten().to(device)    # [30] fp32
```

`group_advantages` (`grpo.py:4-8`): per row of `[5, 6]`, subtract the mean and divide by the **population** std + 1e-4. Values (computed with the repo function) **[code]**:

| group | correct | advantage of a correct row | advantage of a wrong row |
|---|---|---|---|
| 1 | 5 / 6 | +0.447 | −2.235 |
| 2 | 6 / 6 | 0 | — |
| 3 | 2 / 6 | +1.414 | −0.707 |
| 4 | 0 / 6 | — | 0 |
| 5 | 3 / 6 | +1.000 | −1.000 |

Uniform groups get exactly 0 (numerator 0, denominator 1e-4). Each group's advantages sum to 0.

### 8. Identify zero-signal batches

```python
if (rewards == rewards[:, :1]).all(): return None        # training.py:90-91
```

Only a batch in which **all five** groups are uniform is skipped. Then the trainer still writes the rollout record (with `zero_signal: true`), appends the group to `result["zero_signal"]`, and `continue`s: the attempt counter advanced, the step counter did not, and that batch is consumed, never retried (`trainer.py:705-719`). Our batch has three mixed groups, so it proceeds. Groups 2 and 4 stay in the batch with advantage 0: they cost compute and still count in the loss's 1/N (stage 12).

### 9. Sampling-time ("old") log-probabilities

```python
with torch.no_grad():
    old_logp = torch.cat([logprobs(model, c) for _, c in micro_batches(rollouts, 6)])   # training.py:93-94
```

`logprobs` is `backend.logprobs` = `output_head.streamed_logprobs(model, chunk, None, executor)` (`backend.py:81-84`). Per chunk of 6 rows:

1. `explicit_hidden` runs embedding → 36 streamed decoder layers (no grad: the forward-only path of the executor, nothing saved) → final norm: `[6, 77, 2560]` bf16.
2. Keep columns `P-1 … T-1` = 72..76: `h [6, 5, 2560]`.
3. `linear(h, E)` with the tied embedding `E [151936, 2560]`: logits `[6, 5, 151936]` bf16 → fp32 → `log_softmax` → gather the response tokens → `[6, 4]` fp32 (`grpo.response_logprobs`, `grpo.py:22-38`; the last column predicts past the sequence and is dropped).

Five chunks = **five full 36-layer weight sweeps** (≈ 36.3 GB of host-to-device copies). Result: `old_logp [30, 4]` fp32 on the GPU, no graph.

> [!IMPORTANT]
> `old_logp` is **not** the probability under which the tokens were actually sampled in stage 3. Those came from KV-cached incremental logits (or DFlash block logits); `old_logp` is a teacher-forced recomputation in chunks of 6. In bf16 the two can differ in the last bits. The `INIT_TOL` check in stage 11 compares `old_logp` with the update's own teacher-forced log-probs, not with the sampler. With one update per batch this does not change the gradient (the ratio is 1 by construction), but it means "on-policy" holds up to that numerical gap **[unverified magnitude at 4B]**.

Then the rollout record is written to disk **before** any update uses it (`trainer.py:705-712`): `RolloutStore.write` (`streaming/rollouts.py:197-219`) `torch.save`s `{group_id: 1, policy_version: 0, occurrences, prompt_ids, labels, input_ids, attention_mask, response_ids, completion_mask, prompt_len, rewards, zero_signal: false, advantages [30], old_logp [30, 4]}` (tensors moved to CPU) to a temp file, fsyncs, renames to `group_000001.pt`, fsyncs the directory, appends an `index.jsonl` line. **Crash point** `rollouts_generated`. The batch becomes `pending = {rollouts, adv, old_logp, group, iteration: 0, policy_version: 0}` (`trainer.py:720-722`).

### 10. Split into micro-batches

`grpo_update` (`training.py:98-148`) iterates `micro_batches(rollouts, 6)` again: the **same five chunks** as stage 9. That is deliberate: bf16 results depend on batch shape, and iteration 0 needs `logp == old_logp` exactly (`training.py:84-88` docstring).

### 11. Recompute current-policy log-probabilities (with autograd)

`optimizer.zero_grad(set_to_none=True)` (`training.py:115`), then for chunk k (rows `6k .. 6k+5`):

```python
logp = logprobs(model, chunk)                                    # [6, 4] fp32, with graph
init_diff = max |logp - old_logp[rows]| over valid tokens        # must be <= 1e-6, training.py:119-123
```

Same arithmetic as stage 9, but grad mode is on. In the executor that changes only layers 34 and 35: they run as `_CheckpointedLayer` autograd nodes that save their input hidden state `[6, 77, 2560]` and a bit-pattern checksum of their output; layers 0–33 have no input requiring grad and no LoRA, so no node is created and nothing is saved ([06-checkpointed-backward.md](06-checkpointed-backward.md)). The LM head's autograd saves its input and the fp32 `log_softmax` output `[6, 5, 151936]` (≈ 18.2 MB). Another full 36-layer weight sweep. In a grad pass the prefetcher does **not** wrap to layer 0 after layer 35, so layers 34 and 35 are still in the two buffers when backward starts ([05](05-cuda-prefetch-and-buffer-safety.md)).

If `init_diff > 1e-6`: gradients cleared, `RuntimeError`, no step; the run fails rather than training on a stale ratio.

### 12. The clipped GRPO objective

```python
part = grpo_loss(logp, old_logp[rows], adv[rows], mask, clip_eps=0.2)    # training.py:124
part = part * (6 / 30)                                                    # training.py:125-126
```

`grpo_loss` (`grpo.py:41-65`): `ratio = exp(logp − old_logp)` on valid tokens (exactly 1 here), surrogate `min(ratio·A, clamp(ratio, 0.8, 1.2)·A)`, zeroed on padding, summed per sequence and divided by that sequence's length, then **mean over the 6 rows**, negated. Times 6/30, the five parts sum to the loss over all 30 rows. Chunk 2 (group 2, all advantages 0) contributes a loss of exactly 0 and a zero gradient, yet costs a full forward and backward.

A non-finite part clears gradients and raises (`training.py:127-129`).

### 13. Backpropagate

`part.backward()` (`training.py:130`). The autograd engine walks: `part` → mean → per-sequence sums → `exp`/`minimum`/`clamp` → `log_softmax` → `gather` → LM-head `linear` (frozen `E`, gradient only w.r.t. its input) → final RMSNorm → **layer 35 node** → **layer 34 node**.

Each node's `backward` re-binds its layer's weights (both are buffer **hits**: zero host-to-device copies), recomputes the layer with grad enabled from the saved input, checks the output checksum against the forward's, and calls `torch.autograd.grad` for the hidden-state gradient (layer 35 only) and that layer's six LoRA tensors (`execution/backward.py:282-300`). The recomputed graph is freed before the next layer. The executor then prefetches layer 0 for the next chunk into the buffer layer 35 vacated.

### 14. Accumulate LoRA gradients

The node returns gradients for its LoRA inputs; autograd accumulates them into each leaf's `.grad` (created on chunk 1, added to on chunks 2–5). Shapes: e.g. layer 34 `q_proj.lora_A.grad [32, 2560]`, `q_proj.lora_B.grad [4096, 32]`, fp32, GPU. `new_logp` collects `logp.detach()` per chunk; the chunk's graph is gone after `backward` (`training.py:131-133`).

On the very first step every B is 0, so every `lora_A.grad` is exactly zero ([06](06-checkpointed-backward.md), [07](07-grpo-mathematics.md)).

### 15. Apply the AdamW update

After the five chunks (`training.py:134-142`):

1. Per-tensor gradient norms; any non-finite norm clears gradients and raises (no step).
2. `clip_stats(new_logp, old_logp, adv, mask, 0.2)` (`grpo.py:69-85`): ratio min/mean/max, outside/clip fractions, k3 KL estimate. `new_logp` was computed **before** the step, so with `num_iterations = 1` these are always ratio 1, clip fraction 0, KL 0. The comment "ratios at the updated parameters" (`training.py:139`) is only true for iterations > 0.
3. A copy of every parameter (`before`), `optimizer.step()`, `zero_grad(set_to_none=True)`. AdamW creates `exp_avg` and `exp_avg_sq` (same shapes as the parameters, fp32, 8.65 MB total) on its first step and updates every A and B. On step 1 only the B matrices move.

Returns `loss`, per-tensor and total gradient norms, `update_norm` = ‖θ_after − θ_before‖, plus the clip statistics. **Crash point** `update` (`trainer.py:730`): the update exists in GPU memory but is not committed until a checkpoint containing it is published.

### 16. Record diagnostics

`pending["iteration"]` becomes 1 = `num_iterations`, so `pending = None`; `step = 1` (`trainer.py:731-734`). The training row (`trainer.py:735-747`) merges the `group` record with the update logs and adds: A/B gradient norms, learning rate, step time, ‖A‖, ‖B‖, ‖(α/r)·B·A‖ summed over modules (the adapter's actual weight delta), distance from the initial adapter, and CUDA allocator figures (allocated, peak allocated, peak reserved; these exclude the CUDA context). Wall-clock phases and host-to-device bytes copied go to a separate `timings` list so that runs can be compared bitwise on the rows alone (`trainer.py:748-756`). One console line is printed (and tee'd, timestamped, into `training/training.log`).

### 17. Evaluate when scheduled

`step % 20 == 0` or the last step triggers `save(step)` (`trainer.py:764-765`), which **first** runs `eval_subset(step)` (`trainer.py:548-551, 626`): `evaluation.evaluate(model, tok, val_sub, generator=backend.greedy, batch=16, max_new=4)` over the 128-prompt validation subset. This uses the **plain** greedy `StreamedGenerator` through the same L2L executor (DFlash is never used for evaluation, `backend.py:105-107`), with the current LoRA adapter active: 8 batches of 16 rows, ≤ 4 sweeps each. Metrics (accuracy, per-class recall, predicted-POSITIVE rate, invalid rate) and raw predictions are appended to `result["evaluations"]`.

### 18. Save a crash-consistent checkpoint

`save` (`trainer.py:621-642`) builds the training state: config, step, attempts, mixed-group counts, elapsed time, the full result log, `pending: None`, and a `streaming` section (sampler state, rollout group, policy version = step, optimizer steps, run id, dataset hash, pending reference). Then `stream_resume.publish(checkpoints/step20, write)` (`streaming/resume.py:286-310`):

1. `write(tmp)`: `model.save_pretrained(tmp)` → `adapter_model.safetensors` (≈ 4.3 MB fp32) + `adapter_config.json`; `torch.save` of the state + `optimizer.state_dict()` + RNG states (torch CPU, Python `random`, all CUDA devices) → `training_state.pt` (`checkpointing.py:456-460`).
2. fsync every file; record size and sha256 of each in `COMMIT.json`, written and fsynced **last**; fsync the directory.
3. `os.rename(tmp, step20)`; fsync `checkpoints/`.

Only now are updates 1..20 committed. A crash before the rename loses them and they are replayed on resume, bitwise, from the step-0 (or previous) checkpoint ([12-streaming-and-resume.md](12-streaming-and-resume.md)).

---

## The same step as one table

| # | stage | main function | output | shape / dtype | device | grad | weight sweeps | disk |
|---|---|---|---|---|---|---|---|---|
| 1 | read | `PromptStream.next_batch` | 5 records | — | CPU | — | 0 | read jsonl |
| 2 | tokenize | `build_prompts`, tokenizer | ids, mask | `[5,73]` int64 | GPU | — | 0 | — |
| 3 | generate | `dflash_generate` | rollouts | `[30,77]` int64 | GPU | no | 2–4 | — |
| 4 | decode | `decode` | 30 strings | — | CPU | — | 0 | — |
| 5 | reward | `reward` | 30 floats | — | CPU | — | 0 | — |
| 6 | group | `group_rewards` | rewards | `[5,6]` fp32 | CPU | — | 0 | — |
| 7 | advantages | `group_advantages` | adv | `[30]` fp32 | GPU | — | 0 | — |
| 8 | zero-signal | `prepare` | `None` or targets | — | — | — | 0 | (record written either way) |
| 9 | old log-probs | `prepare` → `streamed_logprobs` ×5 | old_logp | `[30,4]` fp32 | GPU | no | 5 | rollout record |
| 10 | chunk | `micro_batches` | 5 views | `[6,77]` | GPU | — | 0 | — |
| 11 | current log-probs | `streamed_logprobs` ×5 | logp | `[6,4]` fp32 | GPU | yes | 5 | — |
| 12 | loss | `grpo_loss` × 6/30 | part | scalar fp32 | GPU | yes | 0 | — |
| 13 | backward | `_CheckpointedLayer.backward` (35, 34) | grads | — | GPU | — | 0 (buffer hits) | — |
| 14 | accumulate | autograd leaf accumulation | `.grad` | LoRA shapes fp32 | GPU | — | 0 | — |
| 15 | update | `AdamW.step` | new A, B | LoRA shapes fp32 | GPU | — | 0 | — |
| 16 | log | `_train` | row + timings | — | CPU | — | 0 | log line |
| 17 | evaluate (every 20) | `evaluate` (greedy, plain) | metrics | — | GPU | no | ≤ 32 | — |
| 18 | checkpoint (every 20) | `publish` | step dir | — | CPU | — | 0 | adapter, state, COMMIT |

Per training step, without evaluation: 12–14 sweeps of 7.267 GB ≈ 87–102 GB host-to-device **[arithmetic from code]**. How long that takes depends on the PCIe rate and on whether copies overlap compute; neither has been measured on the target GPU ([05](05-cuda-prefetch-and-buffer-safety.md), H06).

## Where the step can stop

| condition | where | effect |
|---|---|---|
| every group uniform | `prepare` returns `None` | attempt consumed, no update, record kept |
| `init_diff > 1e-6` | `grpo_update` | `RuntimeError`, grads cleared, no step, run fails |
| non-finite advantage / loss / gradient | `grpo_update` | same |
| recomputed layer output ≠ forward checksum | `_CheckpointedLayer.backward` | `RuntimeError` (state changed between forward and backward) |
| copy into an in-use or protected buffer | `PrefetchMixin._claim` | `BufferReuseError` before the copy is enqueued |
| GPU memory sample above 8 GiB | memory guard thread | KeyboardInterrupt in the main thread, run marked failed |
| `max_attempts` reached before `max_steps` | loop condition | final checkpoint, "attempt budget exhausted" |

## Open questions / unverified

- Real R, acceptance per DFlash round, and the share of zero-signal attempts at 4B: only the first GPU run of `qwen3_4b_final` (or `qwen3_4b_smoke_dflash`) will show them, in `training.json` (`timings[].rollout_generator`, `zero_signal`).
- The size of the sampler-vs-scorer log-prob gap in bf16 at 4B: a diagnostic would compare the generator's step logits (`keep_logits=True` in `generator.generate`) with `old_logp` on the same tokens.
- Step wall time and the split between transfer and compute: `timings[]` + the prefetch timeline (`timeline=True`), H06.
