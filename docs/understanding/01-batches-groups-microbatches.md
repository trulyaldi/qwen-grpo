# 01 — Batches, groups and micro-batches

**What you will be able to explain after this file**
- How five prompts are drawn from a 1,024-prompt pool deterministically, what happens at an epoch boundary, and how the sampler's position survives a crash.
- How five text prompts become a left-padded `[5, P]` token tensor, with concrete ids, masks and position ids from the real Qwen3 tokenizer.
- How those five prompts become 30 completions, in which row order, what is shared and what is duplicated, and how sampling and EOS termination work.
- Why the 30 completions are processed as five micro-batches of six, and why that changes **memory only**, not the gradient, not the rollout batch, and not the number of optimizer steps.
- The four different "batch sizes" in this system, which are routinely confused.

Everything below is for the final run `configs/experiments/qwen3_4b_final.yaml` **as resolved by `qwen_grpo.config.load`** on the working tree of 2026-11-18. Labels: see [00-reading-map.md](00-reading-map.md).

---

## 0. The four numbers that must never be confused

| Name | Value | Meaning | Read by |
|---|---|---|---|
| Prompt batch **B** | 5 | prompts drawn per rollout attempt | `PromptStream.next_batch` (`qwen_grpo/streaming/dataset.py:105`) |
| Group size **G** | 6 | sampled completions per prompt; the unit over which rewards are normalized | `generate_rollouts(..., G=G)` (`qwen_grpo/trainer.py:686`), `group_advantages` (`qwen_grpo/grpo.py:4`) |
| Rollout batch **N = B·G** | 30 | rows produced by **one** generation call, all decoded together | the generator (`execution/generator.py:115` or `execution/dflash.py:193`) |
| Micro-batch size | 6 | rows per forward/backward **chunk** in old-log-prob scoring and in the update | `training.micro_batches` (`qwen_grpo/training.py:74`), called from `prepare` and `grpo_update` |
| Update frequency | 1 AdamW step per rollout batch | `num_iterations: 1` | `trainer._train` loop (`qwen_grpo/trainer.py:672-767`) |

Verified resolved values **[code]** (output of `config.load` on the working tree):

```text
training: batch_size 5, group_size 6, micro_batch_size 6, learning_rate 2.0e-4, max_steps 60,
          max_attempts 1000, num_iterations 1, clip_eps 0.2, max_new_tokens 4, seed 42
data:     train_samples 1024, val_subset 128, subset_seed 42
lora:     rank 32, alpha 64, targets {modules: [q_proj, v_proj, o_proj], num_layers: 2}   -> layers 34, 35
```

> [!WARNING]
> `micro_batch_size: 6`, `learning_rate: 2.0e-4`, `evaluation.every: 20` and `evaluation.batch_size: 16` are **uncommitted** edits to `configs/experiments/qwen3_4b_final.yaml`. Committed `HEAD` inherits `micro_batch_size: null` (one chunk of 30), lr `5e-4`, evaluation every 10, eval batch 32 from `configs/profiles/shared.yaml:113-131`. If you are asked "what did you run", the answer depends on which tree ran it.

The config chain is `qwen3_4b_final.yaml` → `qwen3_4b_l2l.yaml` → `profiles/qwen3_4b.yaml` → `profiles/shared.yaml`; `config.read` merges parents first (`qwen_grpo/config.py:406-425`), mappings merge key by key, scalars replace (`config.merge`, `qwen_grpo/config.py:384-389`).

---

## A. Dataset sampling: five records out of 1,024

### What a record is

`data/sentiment.json` holds Rotten Tomatoes splits built once by `sentiment.make_splits` (`qwen_grpo/sentiment.py:70-87`): deduplicated by a normalized-text key across splits (test first, then validation, then train), label-balanced. Sizes: train 1,500, validation 1,066, test 1,066 **[code]** (counted from the file). One record:

```json
{"text": "guaranteed to move anyone who ever shook , rattled , or rolled .", "label": "POSITIVE"}
```

`training.balanced_subset(splits["train"], 1024, seed=42)` (`qwen_grpo/training.py:57-63`) draws 512 POSITIVE + 512 NEGATIVE row indices with `random.Random(42)`, shuffles them, and attaches `"id"` = the row's index in the 1,500-row split. That `id` is the **stable sample id** used everywhere afterwards (logs, rollout records, coverage).

### Writing the pool to disk

With `streaming.enabled: true`, `PromptStream.build` (`qwen_grpo/streaming/dataset.py:60-87`) writes `runs/qwen3_4b_final/streaming/prompts/prompts.jsonl` (one `{"id","label","text"}` line per prompt, keys sorted) and `prompts.meta.json` = `{sha256, offsets, n}`. Only the 1,024 byte offsets stay in memory; each record is read with `seek(offset)` + `readline()` when drawn (`record`, line 92). A second build against a different file is refused by hash.

### The sampler

One `random.Random(42)` stream (`dataset.py:55`). At the start of every epoch, `_new_epoch` (`dataset.py:98-103`) stores the generator state (`epoch_rng`) and shuffles the index list `0..1023`. `next_batch` (`dataset.py:105-116`) takes the next five indices from that order, rolling into a new epoch when the order is exhausted.

Real output for the first three batches (scratch script, CPU, real data) **[code]**:

```text
batch 1  occurrences (epoch, position, id): (0,0,1373) (0,1,35) (0,2,882) (0,3,558) (0,4,1003)   labels P P N N N
batch 2  (0,5,5) (0,6,649) (0,7,1363) (0,8,1267) (0,9,954)                                       labels N P N N P
batch 3  (0,10,1357) (0,11,13) (0,12,906) (0,13,1433) (0,14,536)                                  labels N P N P N
```

The same script checked that these equal `training.shuffled_batches(train, 5, 42)` element for element (`qwen_grpo/training.py:32-46`), the non-streaming sampler — the docstring's claim of equivalence holds for these batches. Note that batch 1 has 2 POSITIVE / 3 NEGATIVE: **no per-batch class balance is enforced** (`training.py:35`).

### Epoch boundary

1,024 / 5 = 204.8, so batch 205 takes positions 1020–1023 of epoch 0 and position 0 of epoch 1. Within an epoch no prompt repeats; a straddling batch *can* contain the same prompt twice (if epoch 1's first index is one of epoch 0's last four). With `max_steps: 60` the run needs only 60 successful updates; even with zero-signal attempts it will stay far below 205 attempts unless more than ~70% of attempts are zero-signal, so **the final run never reaches an epoch boundary** in practice **[predicted]** (the trainer records the real count in `training.json["coverage"]`, `qwen_grpo/trainer.py:267-280`). 60 updates × 5 prompts ≥ 300 selections ≈ 29% of the pool.

### Saving the sampler

`state()` (`dataset.py:118-121`) returns a JSON-serializable dict: dataset sha256, seed, B, epoch, position, batches drawn, `epoch_rng` (the RNG state at the **start** of the current epoch), and `order_sha256`. After batch 3 it was `epoch 0, position 15, batches 3` **[code]**. `restore` (`dataset.py:123-132`) re-seeds from `epoch_rng`, re-shuffles to recompute the order, checks the order hash, then sets position and batch count. Storing the epoch-start state rather than the current state is what makes the order recomputable instead of storing 1,024 indices. Full resume mechanics: [12-streaming-and-resume.md](12-streaming-and-resume.md).

---

## B. Prompt tokenization: five strings → `[5, P]`

### Chat template, non-thinking mode

`sentiment.build_prompts` (`qwen_grpo/sentiment.py:23-32`) wraps each sentence in the task text (`PROMPT`, lines 11-15) and applies the tokenizer's chat template with `add_generation_prompt=True, enable_thinking=False`. It **asserts** that every string ends with the empty thinking block, so a template change that re-enables thinking fails loudly. Real string for batch 1, row 0 (id 1373):

```text
<|im_start|>user\nClassify the sentiment of the following sentence as POSITIVE or NEGATIVE.\nONE WORD ONLY.\n\n
Sentence: una película oscura , precisa , por momentos grandiosa y casi siempre conmovedora .<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

"Non-thinking mode" therefore means: the assistant turn begins **after** an already-closed `<think></think>` block, so the first generated token is the answer itself.

### Tokenization and left padding

`policy.generate_rollouts` (`qwen_grpo/policy.py:7-41`) calls `tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)` with a tokenizer built `padding_side="left"` (`qwen_grpo/trainer.py:518`). For batch 1, with the Qwen3 tokenizer (loaded offline from the cached Qwen3-0.6B snapshot; the Qwen3 family shares one tokenizer, and the trainer asserts the pad id, `trainer.py:519`) **[code]**:

| tensor | shape | dtype | content (batch 1) |
|---|---|---|---|
| `input_ids` | `[5, 73]` | int64 | real lengths per row `[51, 73, 55, 60, 65]`; shorter rows are **left**-padded with id 151643 `<|endoftext|>` |
| `attention_mask` | `[5, 73]` | int64 | 0 on padding, 1 on tokens |
| `P` (`prompt_len`) | 73 | Python int | width of the padded prompt block, **shared by all rows** |

Row 0 has 22 pad columns, then 51 tokens. Its last 12 ids and their tokens:

```text
[94818, 6215, 659, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271]
 'moved' 'ora' ' .' '<|im_end|>' '\n' '<|im_start|>' 'assistant' '\n' '<think>' '\n\n' '</think>' '\n\n'
```

Why **left** padding: generation appends new tokens at the right. With left padding, the last column is the last real prompt token for every row, so "the logits of the last column" is the next-token distribution for every row at once, and all rows append at the same column index.

### Position ids

Left padding would shift RoPE positions if positions were simply `0..72`. Two formulas are used:

| where | formula | value on row 0 |
|---|---|---|
| training / scoring: `policy.compute_logprobs` (`policy.py:54`), `output_head.streamed_logprobs` | `(mask.cumsum(1) - 1).clamp(min=0)` | pads → 0; first real token → 0; last prompt token → 50 |
| generation: `generator._positions` (`execution/generator.py:73-76`) | `cumsum - 1`, pads set to 1 | pads → 1; real tokens identical |

Real tokens get identical positions under both. Padding positions differ (0 vs 1), but padded columns are excluded as attention **keys** by the mask, and their own query outputs are never read for anything that is scored, so the difference cannot reach a used value **[code]** (H04's generation equivalence tests check the result, [11-research-hypotheses.md](11-research-hypotheses.md)).

Distribution over the pool **[code]**: prompt length min 37, median 57, max 94 tokens; padded width P over the first 60 batches min 51, median 71.5, max 93. The budget calculator assumes P = 100 (conservative; [09-memory-and-tensor-lifetimes.md](09-memory-and-tensor-lifetimes.md)).

---

## C. Rollout generation: five prompts → 30 completions

### Row order: prompt-major

The generator duplicates each prompt row G = 6 times with `repeat_interleave(6, dim=0)` (`execution/generator.py:115-116`, `execution/dflash.py:193-194`):

```text
row  0.. 5   prompt 1 (id 1373, label POSITIVE)   completions 1..6
row  6..11   prompt 2 (id 35,   label POSITIVE)
row 12..17   prompt 3 (id 882,  label NEGATIVE)
row 18..23   prompt 4 (id 558,  label NEGATIVE)
row 24..29   prompt 5 (id 1003, label NEGATIVE)
```

Every later step relies on this order: `group_rewards` reshapes `[30] → [5, 6]` (`training.py:67-70`), and `check_grouping` re-tokenizes the prompts and asserts that rows `6b..6b+5` start with prompt b's padded tokens (`training.py:50-53`, called at `trainer.py:691`).

### All 30 rows are generated simultaneously

Yes: one generator call, one `[30, ·]` batch. In the final config the generator is DFlash ([08-dflash-integration.md](08-dflash-integration.md)); the plain L2L generator (used by the baseline smoke run and by evaluation) works like this:

| step | tensor fed to the target | shape | KV cache after |
|---|---|---|---|
| prefill | the whole prompt block | `[30, 73]` | 73 positions per row, every layer |
| decode 1 | last sampled token | `[30, 1]` | 74 |
| decode 2 | | `[30, 1]` | 75 |
| decode 3 | | `[30, 1]` | 76 (only if some row is unfinished) |

Each of these is one full sweep of the 36 decoder layers through the L2L executor (weights streamed from CPU every time: [04-l2l-weight-streaming.md](04-l2l-weight-streaming.md)).

### What is shared, what is duplicated

- **Shared:** the policy (base weights + current LoRA tensors), the prompt *content*, and the label (used only later, by the reward).
- **Duplicated:** the prompt rows themselves. The prefill computes the same 73-token prompt six times, and the KV cache stores six identical copies of the prompt's keys/values: `[30, 8 kv heads, capacity, 128]` per layer per K and V (`execution/generator.py:42-48`). There is **no prefix sharing**. At P = 73 the plain-generator cache is 30 · 77 · 36 layers · 4,096 B ≈ 340.6 MB, five sixths of which are duplicate prompt entries **[code + arithmetic]**.
- **Different per row:** only the random draws. Each row's token is drawn independently by `torch.multinomial(softmax(logits))` (`generator.py:136`), from the global torch RNG (the CUDA RNG, since logits are on the GPU), which the trainer seeds once with `torch.manual_seed(42)` before the first rollout (`trainer.py:593`).

### Sampling

The target produces fp32 next-token logits `[30, 151936]` for the last column only (`generator.py:92`). Temperature 1, no top-k, no top-p: the sampler is exactly the policy's softmax, which matters because GRPO scores these tokens with untempered log-probs (`policy.py:10-14`). Any temperature ≠ 1 would make the sampler differ from the policy being optimized.

### EOS termination

`unfinished` is a `[30]` long vector, initially 1 (`generator.py:124`). After each draw:

```python
nxt = nxt * unfinished + pad_token_id * (1 - unfinished)   # a finished row gets pad (151643)
unfinished = unfinished & ~torch.isin(nxt, eos)            # either EOS id finishes the row
if unfinished.max() == 0: break                            # every row done -> stop early
```

(`generator.py:139-144`). So the number of generated columns **R** is the step at which the *last* row finished, at most `max_new_tokens = 4`.

### The response as tokens

Real token splits **[code]**: `"POSITIVE"` = `['POS', 'ITIVE']` (17456, 44350), `"NEGATIVE"` = `['NEG', 'ATIVE']` (97127, 23215), `"Positive"` = one token (35490). A canonical answer is therefore three tokens including `<|im_end|>`. Illustrative (not measured) responses for prompt 1 with R = 4:

```text
row 0: POS ITIVE <|im_end|> <|endoftext|>     completion_mask 1 1 1 0   decoded "POSITIVE"  reward 1
row 1: NEG ATIVE <|im_end|> <|endoftext|>     completion_mask 1 1 1 0   decoded "NEGATIVE"  reward 0
row 2: Positive <|im_end|> <|endoftext|> ...  completion_mask 1 1 0 0   decoded "Positive"  reward 1  (parse_label upper-cases)
row 3: The sentiment is ...                   completion_mask 1 1 1 1   truncated, no EOS    reward 0
```

`grpo.response_mask` (`qwen_grpo/grpo.py:11-19`) builds the mask from **positions** (tokens up to and including the first EOS), not from `id != pad`, because Qwen3's pad id 151643 is itself an EOS id. A truncated response is kept whole.

### The rollout dictionary

`generate_rollouts` returns (`policy.py:32-41`):

| key | shape | dtype | device | meaning |
|---|---|---|---|---|
| `input_ids` | `[30, P+R]` | int64 | GPU | prompt block + responses |
| `attention_mask` | `[30, P+R]` | int64 | GPU | prompt mask repeated ×6, then `completion_mask` (tokens after EOS are masked) |
| `response_ids` | `[30, R]` | int64 | GPU | generated columns |
| `completion_mask` | `[30, R]` | bool | GPU | tokens that count in the loss (incl. EOS) |
| `prompt_len` | `P` | int | — | 73 for batch 1 |

---

## D. Micro-batching: 30 rows → five chunks of six

### The slicing

`training.micro_batches(rollouts, 6)` (`training.py:74-81`) yields `(slice(0,6), chunk), (slice(6,12), chunk), …`. Each chunk is a set of **views** of rows of every tensor; it keeps the batch's padded width `P+R` and `prompt_len`. Because the rows are prompt-major and the micro-batch size equals G, **each micro-batch is exactly one prompt's group** here. That is a coincidence of `micro_batch_size == group_size`, not a requirement: the advantages are computed on the whole `[5, 6]` reward matrix *before* chunking (`training.py:92`), so any chunking is valid.

A consequence of keeping the batch width: chunk 1 (row 0's prompt has 51 real tokens) still carries 22 left-pad columns because another prompt in the *batch* was 73 tokens long.

### Why chunk at all, and what it does not change

Chunking bounds what one forward/backward holds at once: hidden states `[6, T, 2560]` instead of `[30, T, 2560]`, the two saved layer-boundary activations, the recomputed layer-34/35 graph, and the LM-head output `[6, R+1, 151936]` in fp32 (≈ 18.2 MB at R = 4 instead of ≈ 91 MB) **[arithmetic]**. Details: [09-memory-and-tensor-lifetimes.md](09-memory-and-tensor-lifetimes.md).

It does **not** change:
- **the gradient.** The per-chunk loss is rescaled by `n_c / N = 6/30` (`training.py:125-126`) and gradients accumulate, giving the full-batch gradient (derivation in [07-grpo-mathematics.md](07-grpo-mathematics.md)). Up to floating-point summation order, it equals the unchunked gradient. Relative to a resident run *with the same chunking* it is bitwise equal (H05 grid includes micro-batch sizes 1, 2, 4, all; [11-research-hypotheses.md](11-research-hypotheses.md));
- **the rollout batch.** Generation never reads `micro_batch_size`. It is passed only to `prepare` and `grpo_update` (`trainer.py:702, 725-727`). The generator always decodes 30 rows at once, so the KV cache and prefill activations scale with 30, whatever the micro-batch size. If generation memory were the problem, lowering `micro_batch_size` would not help;
- **the number of optimizer steps.** Exactly one `optimizer.step()` per call of `grpo_update` (`training.py:141`).

It **does** cost time: every chunk is a full forward through all 36 streamed layers, so the five chunks mean five full weight sweeps (≈ 7.27 GB of host-to-device copies each) for old log-probs and five more for the update. The L2L paper's remedy (loop micro-batches *inside* the layer loop) is not implemented ([04-l2l-weight-streaming.md](04-l2l-weight-streaming.md)).

### The complete timeline of one rollout batch

```mermaid
sequenceDiagram
    participant T as trainer._train
    participant G as generator (DFlash / plain)
    participant P as training.prepare
    participant U as training.grpo_update
    participant O as AdamW
    T->>G: 5 prompts -> [5,P] ids
    G-->>T: rollouts [30, P+R]
    T->>T: decode, rewards [5,6], mixed-group count
    T->>P: rollouts, rewards, micro_batch_size=6
    P->>P: adv = group_advantages(rewards).flatten() -> [30]
    loop chunks rows 0-5, 6-11, 12-17, 18-23, 24-29 (no_grad)
        P->>P: logprobs(model, chunk) -> [6, R] fp32
    end
    P-->>T: adv [30], old_logp [30, R]
    T->>T: write rollout record to disk
    T->>U: rollouts, adv, old_logp
    U->>U: zero_grad(set_to_none=True)
    loop the same 5 chunks (grad enabled)
        U->>U: logp = logprobs(model, chunk) [6,R]
        U->>U: check max|logp - old_logp[rows]| <= 1e-6
        U->>U: part = grpo_loss(...) * 6/30
        U->>U: part.backward()  (LoRA .grad += chunk gradient; graph freed)
    end
    U->>U: grad norms, finiteness check, clip_stats
    U->>O: optimizer.step()  (the only parameter change)
    U->>U: zero_grad(set_to_none=True)
    U-->>T: logs (loss, norms, ratios)
```

Shapes at every stage, for batch 1 with an assumed R = 4 (so T = 77):

| stage | tensor | shape | dtype | grad? |
|---|---|---|---|---|
| sampler | 5 records | list | Python | — |
| tokenizer | `input_ids`, `attention_mask` | `[5, 73]` | int64 | — |
| generator | `input_ids` | `[30, 77]` | int64 | no (`@torch.no_grad`, `policy.py:6`) |
| reward | `rewards` | `[5, 6]` | fp32 | — |
| prepare | `adv` | `[30]` | fp32 | — |
| prepare, chunk k | `logp_k` | `[6, 4]` | fp32 | no |
| prepare | `old_logp` | `[30, 4]` | fp32 | no |
| update, chunk k | hidden state | `[6, 77, 2560]` | bf16 | layers 34-35 only |
| update, chunk k | LM-head logits | `[6, 5, 151936]` | bf16 → fp32 | yes |
| update, chunk k | `logp` | `[6, 4]` | fp32 | yes |
| update, chunk k | `part` | scalar | fp32 | yes |
| accumulate | each LoRA `.grad` | e.g. `[32, 2560]`, `[4096, 32]` | fp32 | — |

### When gradients are cleared, accumulated and applied

| moment | what happens to LoRA `.grad` | code |
|---|---|---|
| start of `grpo_update` | set to `None` | `training.py:115` |
| `part.backward()` of chunk 1 | autograd **creates** `.grad` = chunk-1 gradient | `training.py:130` |
| chunks 2–5 | autograd **adds** into `.grad` (leaf accumulation) | same line |
| a non-finite loss or init mismatch | cleared again, exception raised, no step | `training.py:121-123, 127-129` |
| after the loop | norms measured, finiteness checked | `training.py:135-138` |
| `optimizer.step()` | AdamW reads `.grad`, updates A, B and its moments | `training.py:141` |
| right after | set to `None` (frees 4.3 MB) | `training.py:142` |

## Open questions / unverified

- Actual R per batch and padded P on the real run: recorded per step (`timings[].rollout_tokens`, `mean_completion_tokens`) only once `qwen3_4b_final` runs on a GPU.
- Whether the six duplicated prompt rows produce bitwise-identical KV entries on GPU (kernels are row-independent in principle; not checked). A check: compare `cache.keys[l][0]` and `cache.keys[l][5]` after prefill in a CUDA test.
- The fraction of zero-signal attempts at 4B (0.6B at lr 5e-4: 42 of 142 attempts **[measured]**, `runs/qwen/report.md`).
