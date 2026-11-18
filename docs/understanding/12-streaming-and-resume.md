# 12 — The four kinds of "streaming", checkpointing, and deterministic resume

**What you will be able to explain after this file**
- The difference between weight streaming, dataset streaming, rollout streaming and checkpoint streaming, and the code of each.
- How the prompt sampler picks five prompts deterministically, crosses an epoch boundary, and saves/restores its state in a few hundred bytes.
- How a rollout record and a checkpoint are written so that a crash at any instant leaves either the old state or the new one, never a torn file.
- What exactly a checkpoint contains, in what order it is written, and how a resumed run continues the *same* trajectory.
- Why "deterministic resume" is a harder problem than "save the weights", and which parts of it are verified only on CPU.

---

## 1. Four things called streaming

| name | what moves | from → to | granularity | code | doc |
|---|---|---|---|---|---|
| **weight streaming** (L2L) | frozen bf16 decoder weights | pinned CPU slab → GPU buffer | one layer (201.9 MB), every forward | `execution/weight_store.py`, `layer_stream.py`, `prefetch.py` | [04](04-l2l-weight-streaming.md), [05](05-cuda-prefetch-and-buffer-safety.md) |
| **dataset streaming** | training prompts | `prompts.jsonl` on disk → Python dicts | one record per prompt, read by byte offset | `streaming/dataset.py` `PromptStream` | §2 |
| **rollout streaming** | each rollout batch's tensors (ids, masks, rewards, advantages, old log-probs) | GPU → `group_<id>.pt` on disk | one file per attempt, written once | `streaming/rollouts.py` `RolloutStore` | §3 |
| **checkpoint streaming** | adapter, AdamW state, counters, RNG, sampler state | memory → `checkpoints/step<N>/` | one directory every 20 updates | `streaming/resume.py`, `checkpointing.py`, `trainer.save` | §4 |

Only the first is about GPU memory. The other three exist for **crash consistency and reproducibility**
(Ticket 09 / H08). All three are enabled by `streaming.enabled: true, directory: streaming`
(`configs/experiments/qwen3_4b_final.yaml:51-53`), resolved to `runs/qwen3_4b_final/streaming/` (`trainer.py:96-99`).

---

## 2. Dataset streaming: `PromptStream`

### 2.1 What is on disk **[code]** `dataset.py:61-87`

`PromptStream.build(dir/"prompts", train_set, seed=42, batch_size=5)` (`trainer.py:647`):
- `prompts.jsonl`: one line per prompt, `{"id", "label", "text"}` with sorted keys. `id` is the stable index of the
  sentence in `data/sentiment.json`'s train split, chosen by `balanced_subset(…, 1024, subset_seed=42)`
  (`training.py:57-63`): 512 POSITIVE + 512 NEGATIVE, shuffled.
- `prompts.meta.json`: `{"sha256": <of the jsonl bytes>, "offsets": [byte offset of each line], "n": 1024}`.
- Written via temp file + `fsync` + `os.replace` + directory `fsync` (`dataset.py:27-44, 79-86`). A second build on the
  same directory must find byte-identical content, otherwise it refuses to mix datasets (`dataset.py:71-73`). Opening a
  stream re-hashes the file (`dataset.py:52-53`).

Only the offsets stay in memory; `record(i)` seeks to `offsets[i]` and parses one line (`dataset.py:92-96`).

### 2.2 The sampler **[code]** `dataset.py:98-116`

```
rng = random.Random(42)                 # private stream, not Python's global random
epoch 0: epoch_rng = rng.getstate(); order = shuffle(range(1024)) with rng; position = 0
next_batch(): 5 times: if position == 1024: new epoch; take order[position]; position += 1
```

It reproduces `training.shuffled_batches` element for element (`dataset.py:8-10`; the non-streaming sampler,
`training.py:32-46`). Verified for the first three batches of the real pool in a scratch run:

| batch | occurrences `(epoch, position, id)` | labels |
|---|---|---|
| 1 | (0,0,1373) (0,1,35) (0,2,882) (0,3,558) (0,4,1003) | P P N N N |
| 2 | (0,5,5) (0,6,649) (0,7,1363) (0,8,1267) (0,9,954) | N P N N P |
| 3 | (0,10,1357) (0,11,13) (0,12,906) (0,13,1433) (0,14,536) | N P N P N |

No class balance per batch is enforced (batch 1 is 2:3).

**Epoch boundary.** 1024 / 5 = 204.8: batches 1–204 consume positions 0–1019 of epoch 0; batch 205 takes positions
1020–1023 of epoch 0 **and** position 0 of epoch 1 (a fresh shuffle). Within an epoch every prompt appears once;
a batch straddling two epochs may contain the same prompt twice. With `max_steps: 60` and a 0.6B-like zero-signal
rate (42 of 142 attempts), the run consumes ~85 batches ≈ 425 selections — **well under one epoch**; the trainer
records actual coverage in `training.json["coverage"]` (`trainer.py:267-280`) and never claims an epoch.

A zero-signal batch is **consumed**, not retried: its prompts count as drawn (`trainer.py:713-719`).

### 2.3 Sampler state (JSON, a few KB) `dataset.py:118-132`

| field | meaning |
|---|---|
| `dataset_sha256`, `seed`, `batch_size` | identity; restore refuses a mismatch |
| `epoch`, `position`, `batches` | where the stream is |
| `epoch_rng` | the `random.Random` state **at the start of the current epoch** (625 ints) |
| `order_sha256` | hash of the current epoch's permutation |

`restore` sets the RNG to `epoch_rng`, re-runs `_new_epoch()` (re-shuffles), checks the permutation hash, then sets
`position` and `batches`. The permutation is recomputed rather than stored (1024 ints), and the hash proves the
recomputation is the same.

---

## 3. Rollout streaming: `RolloutStore`

### 3.1 When a record is written **[code]** `trainer.py:705-712`

After rewards and `prepare` (advantages + old log-probs), **before** any update uses them:

```
rollout_store.write({group_id: attempts, policy_version: step, occurrences, prompt_ids, labels,
                     input_ids [30,T], attention_mask [30,T], response_ids [30,R], completion_mask [30,R],
                     prompt_len, rewards [5,6], zero_signal, advantages [30] | None, old_logp [30,R] | None})
```

`policy_version = step` = the number of committed optimizer steps of the policy that sampled it. Zero-signal batches
are recorded too (`zero_signal: true`, no advantages).

### 3.2 How it is written `rollouts.py:197-219`

1. tensors → CPU; `run_id` and `sha256` over the tensors' dtype, shape and bytes (`rollouts.py:173-181`).
2. refuse if `group_<id>.pt` already exists: a committed group is never regenerated.
3. `torch.save` into `.group_<id>.pt.tmp-<pid>`, `flush`, `fsync`; `os.replace` to the final name; `fsync` the directory.
4. append one line to `index.jsonl`, `flush`, `fsync`.

`read(group_id, policy_version)` refuses another run's record, another policy version, or a hash mismatch
(`rollouts.py:221-230`). `discard_after(g)` moves records with id > g to `orphaned/<stamp>/`, deletes interrupted
temp files, and atomically rewrites the index (`rollouts.py:232-244`).

**What the store is used for in the final config.** Only the *pending* group (a batch with updates left) would ever
be read back (`trainer.py:610-619`). With `num_iterations: 1`, `pending` is set to `None` immediately after its single
update (`trainer.py:731-733`), before `save()` can run (`trainer.py:764-765`). So **the pending-restore path is never
exercised by the final run**; the records serve as an audit log (which tokens, rewards and old log-probs each update
used) and as the reason uncommitted attempts must be moved aside on resume.

---

## 4. Checkpoint streaming

### 4.1 Contents of `checkpoints/step<N>/` **[code]**

| file | content | written by |
|---|---|---|
| `adapter_config.json`, `adapter_model.safetensors` | PEFT adapter: 12 fp32 tensors, 4.33 MB | `model.save_pretrained` (`checkpointing.py:459`) |
| `training_state.pt` | `torch.save({**st, "optimizer": optimizer.state_dict(), "rng": rng_state()})` (`checkpointing.py:456-460`) | `save_checkpoint` |
| `COMMIT.json` | `{"step": N, "files": {name: {bytes, sha256}}}` | `stream_resume.publish` |

`st` (`trainer.py:627-639`): resolved `config`; `step`, `attempts`, `checkpoint` path; `mixed_counts`;
`train_seconds`; the whole `result` log so far; `base_full`; `pending: None` (streaming stores a reference instead);
`streaming = {sampler: PromptStream.state(), rollout_group: attempts, policy_version: step, optimizer_steps: step,
run_id, dataset_sha256, pending: None | {group_id, policy_version, iteration, group}}`.
`rng_state()` = torch CPU RNG, Python `random` global state, **all CUDA generators** (`checkpointing.py:444-446`).

Step 0 (`trainer.py:554-555`) publishes the initial adapter only (B = 0), never resumed from.

### 4.2 Order inside `save(step)` `trainer.py:621-642`

1. `eval_subset(step, ...)` — greedy evaluation of the 128-prompt subset, appended to `result`. It runs **first** so
   the saved log contains it and the saved RNG state is the one *after* it. (Greedy decoding draws no random numbers,
   but ordering it first makes this true by construction.)
2. build `st`, with the sampler state captured now.
3. `publish(path, lambda tmp: save_checkpoint(model, optimizer, tmp, st), step)`.
4. `crash_point("after_checkpoint")`.

### 4.3 Atomic publication `streaming/resume.py:286-310`

```mermaid
sequenceDiagram
  participant T as trainer.save
  participant FS as file system
  T->>FS: mkdir checkpoints/.tmp-step20-<pid>/
  T->>FS: save_pretrained + torch.save(training_state.pt)
  T->>FS: fsync every file; sha256 + size of each
  Note over T,FS: crash point files_written
  T->>FS: write COMMIT.json, fsync it, fsync tmp dir
  Note over T,FS: crash point committed_before_rename
  T->>FS: rename .tmp-step20-<pid> -> step20 (atomic on one FS)
  T->>FS: fsync checkpoints/
```

Invariant: a directory named `step<N>` is either absent or complete. `validate(path)` (`resume.py:313-332`): valid
iff `COMMIT.json` parses, every listed file exists with its recorded size and sha256, and (for resumable ones)
`training_state.pt` is listed. `latest_valid(name)` returns the highest valid step > 0 (`resume.py:335-344`).
`prepare(name)` deletes `.tmp-step*` leftovers and renames invalid checkpoints to `.invalid-step<N>-<stamp>` (kept
for inspection) (`resume.py:347-365`).

**Commit semantics.** An optimizer update is *committed* only when a published checkpoint contains it
(`resume.py:254-256`). With `evaluation.every: 20`, a crash just before step 40's publication loses (and replays)
up to 19 updates — plus their rollouts and zero-signal attempts.

---

## 5. One GRPO attempt, annotated with the crash points

`crash_point(name)` raises `InjectedCrash` (a `KeyboardInterrupt`) at the N-th occurrence when
`QWEN_GRPO_CRASH=name[@N]` (`trainer.py:78-90`); publication shares the hook (`trainer.py:93`).

| order | action | line | crash point |
|---|---|---|---|
| 1 | `prompts_on_disk.next_batch()` → 5 records | `trainer.py:679` | `prompt_read` (`:680`) |
| 2 | rollouts, rewards, `prepare` | `:686-704` | — |
| 3 | `rollout_store.write(...)` | `:707-711` | `rollouts_generated` (`:712`) |
| 4 | `grpo_update` (AdamW step applied in memory) | `:725-727` | `update` (`:730`) |
| 5 | `step += 1`, log row | `:734-747` | — |
| 6 | if `step % 20 == 0`: `save(step)` | `:764-765` | `files_written`, `committed_before_rename`, `after_checkpoint` |

---

## 6. Resume, step by step

`scripts/run.py train --config … --resume latest` → `train()` resolves `latest` with `latest_valid`
(`trainer.py:356-359`), takes the run lock, then:

1. **`start`** (`trainer.py:283-300`): `validate(resume)`; `stream_resume.prepare(name)` (remove temp dirs,
   quarantine invalid checkpoints); `load_training_state` (CPU first, `checkpointing.py:463-469`); `check_resume`
   (`trainer.py:154-169`): the checkpoint belongs to this run, its step matches, the config is identical except
   `run.overwrite`/`verification`, and no later checkpoint exists that resuming would overwrite.
2. Model built and seeded exactly as a fresh run (`torch.manual_seed(seed)` before LoRA init, `trainer.py:517`), then
   **`load_adapter`** (asserts every tensor took effect, `checkpointing.py:407-415`) and
   **`optimizer.load_state_dict`** (AdamW moments and step counts) (`trainer.py:595-596`).
3. Counters and log from the state (`trainer.py:597-600`); a `resumed` entry is appended to the log.
4. **Sampler** (`trainer.py:649-655`): `prompts_on_disk.restore(s["sampler"])`, then a consistency check: sampler
   batches == `attempts` == `rollout_group`, and `optimizer_steps == step`; otherwise refuse.
5. **Rollout records** (`trainer.py:656-657`): `discard_after(attempts)` moves records of attempts after the checkpoint
   (generated by the uncommitted part of the old session) to `orphaned/`, so they can be regenerated.
6. **Pending group** (`trainer.py:658-661`): read back from disk if the checkpoint referenced one (never, here).
7. **RNG last** (`trainer.py:667-668`): `set_rng_state(state["rng"])` — torch CPU, Python, every CUDA generator. Nothing
   between this line and the next rollout may draw a random number.
8. The loop continues at `attempts + 1`.

### Worked example

Uninterrupted run A: step 20 published after attempt 29; steps 21–27 use attempts 30–37; checkpoint 40 later.
Run B crashes at `update@27` (the 27th update applied in memory, uncommitted).

| item | B before resume | B after resume |
|---|---|---|
| latest valid checkpoint | step20 (attempts 29) | — |
| adapter, AdamW | step 27 in memory (lost) | step 20 from disk |
| sampler | 37 batches drawn | 29 (restored state) |
| rollout records | groups 1–37 on disk | 30–37 moved to `orphaned/` |
| RNG | advanced | exactly A's state after step 20's checkpoint |
| next attempt | — | 30, regenerated; must equal A's attempt 30 bitwise |

H08's criterion (`research/l2l/hypotheses.yaml`) checks exactly this for six crash points: committed training rows,
zero-signal attempts, evaluations, counters, rollout records and every common checkpoint's adapter and AdamW state
equal A's. **SUPPORTED on a tiny random Qwen3 on CPU** (`t09_resume_tiny_cpu`) **[test/measured, CPU]**.

---

## 7. Why deterministic resume is a separate problem from checkpointing

Saving the adapter gives you a model. Resuming *the same run* also requires that every input of every future update
is the same:

1. **Which prompts come next** — the sampler's own RNG and position (§2.3), not Python's global RNG.
2. **Which tokens are sampled** — the CUDA generator state used by `torch.multinomial` in generation
   (`generator.py:136`, `dflash.py:175`); restored by `set_rng_state`.
3. **That regenerated rollouts are bitwise identical** — this needs the *same kernels producing the same bits* for
   the same inputs. On GPU, bf16 matmuls and SDPA are deterministic for fixed shapes on the same hardware/library in
   practice, but nothing in the code forces `torch.use_deterministic_algorithms` **[unverified on GPU]**. If one
   logit differs in the last bit, a `multinomial` draw can flip, and the resumed trajectory diverges from A while
   still being a valid run.
4. **No double application** — an update committed in a checkpoint must not be replayed; one after it must be
   (commit semantics, §4.3).
5. **No orphan mixing** — records from the dead session must not be read as if they belonged to the new one
   (`discard_after`, policy-version check on read).
6. **AdamW's internal state** — moments and per-parameter step count; resuming with fresh moments would change every
   later update.

`check_resume`'s config comparison (`trainer.py:161-167`) prevents resuming under different hyperparameters; a
different library version or GPU is *not* refused, only recorded (`environment`, `trainer.py:135-151`).

---

## 8. Two related mechanisms

**Continuation** (`continue_run`, `trainer.py:204-264`): `max_steps` is part of a run's identity, so extending 60 →
210 updates (`qwen3_4b_final_ext210.yaml`, opt-in) creates a *new* run that starts from a byte-for-byte copy of the
source's step-N checkpoint, step-0 adapter and streaming directory, recording tree hashes; only
`CONTINUATION_KEYS = {run.name, training.max_steps}` may differ (`trainer.py:179`). It then uses the normal resume
path.

**Run lock** (`training_lock`, `checkpointing.py:430-441`): `runs/.<name>.lock` holds the PID; a second trainer on the
same run refuses while that PID is alive; a stale lock is taken over.

---

## 9. Failure modes and their guards

| failure | guard |
|---|---|
| crash mid-write of a checkpoint | temp dir + COMMIT.json + rename; `validate` rejects incomplete dirs; `prepare` removes/quarantines |
| corrupted checkpoint file (bit rot, truncation) | sha256 + size per file in COMMIT.json (`tests/test_streaming.py`) |
| dataset file changed between sessions | sha256 on open and on build (`dataset.py:52-53, 71-73`); sampler state carries the hash |
| resumed with another config | `check_resume` diff refuses |
| uncommitted rollout record reused | `discard_after` + "already exists" refusal + policy-version check |
| RNG consumed between restore and the next rollout | structural: `set_rng_state` is the last statement before the loop |
| non-streaming run resumed | replays the in-memory sampler by `attempts` (`trainer.py:663-666`); not the final config |
| GPU nondeterminism | **none** — would show up as a divergence in H08-style comparison |

## Open questions / unverified

- **Bitwise resume on a real model on GPU**: `H08.resume_qwen3_0.6b` (stage C) — crash and resume a Qwen3-0.6B streaming run on CUDA and compare with the uninterrupted run.
- **Same at 4B with DFlash rollouts**: no entry exists; an interrupted `qwen3_4b_final` resumed by the orchestrator (stage F) would show it only if compared against an uninterrupted copy.
- **fsync semantics on the target file system** (e.g. overlay or network FS in containers): the atomic-rename argument assumes a local POSIX FS; untested elsewhere.
- **Whether `pending` restore works** at `num_iterations > 1` end-to-end on a real model: exercised by `tests/test_streaming.py` on tiny models only.
