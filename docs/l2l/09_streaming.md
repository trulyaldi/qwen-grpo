# Ticket 09: disk-backed prompt and rollout streaming with deterministic resume

## Code

- `qwen_grpo/streaming/dataset.py` (`PromptStream`), `streaming/rollouts.py` (`RolloutStore`) and
  `streaming/resume.py` (crash-consistent `publish`, `validate`, `latest_valid`, `prepare`).
- `qwen_grpo/trainer.py` uses them when `streaming.enabled`. The in-memory path is unchanged when it is off.
  - The crash-injection hook `crash_point` (environment variable `QWEN_GRPO_CRASH=point[@n]`).
  - `train(resume="latest")`, also available as `scripts/run.py train --resume latest`.
- `config.unimplemented` no longer refuses `streaming`. `streaming.directory` is resolved inside the run directory
  (`runs/<name>/<directory>`), so it is never committed.
- `configs/experiments/qwen_stream_short.yaml` and `scripts/resume_experiment.py`.
- Tests: `tests/test_streaming.py`.

## What is on disk (`runs/<name>/streaming/`)

| file | content |
|---|---|
| `prompts/prompts.jsonl` | the training prompts, one record per line (stable sample id, text, label), written once |
| `prompts/prompts.meta.json` | its sha256 (dataset identity: a different dataset is refused) and the byte offset of every line |
| `rollouts/group_<id>.pt` | one record per rollout group (details below), published once by temp file + fsync + rename |
| `rollouts/index.jsonl` | one line per published record, appended and fsynced after the record file |
| `rollouts/orphaned/<stamp>/` | records of uncommitted groups, set aside on resume |

**A rollout record** holds the group id (the attempt), run id, policy version (committed optimizer steps of the
generating policy), the `(epoch, position, sample_id)` occurrences, prompt ids and labels, input / response ids and
masks, rewards, advantages and old log-probs (none for a zero-signal group), and the sha256 of its tensors.

**Reading a record back** refuses another run's record, another policy version than the checkpoint expects, or
tensors that don't match their hash. So records of incompatible policy versions are never mixed.

**Memory:** only the pending group's tensors are kept. Every earlier group lives only on disk.

## The sampler

- `PromptStream` reproduces `training.shuffled_batches` exactly, including batches that straddle two epochs (checked
  batch for batch). That's why a streaming run equals the in-memory run bitwise.
- Only line offsets are held in memory. Records are read from disk by offset.
- **Its state:** dataset sha256, seed, batch size, epoch, position, batches drawn, the `random.Random` state at the start
  of the current epoch (the order is recomputed from it), and the sha256 of the epoch's order (checked on restore).
- **No replay on resume:** the in-memory path replays `attempts` batches; the stream restores its state directly.

## What a checkpoint records

A checkpoint records `training_state.pt["streaming"]`:
- sampler state;
- rollout group (attempt count);
- policy version and optimizer steps;
- run id and dataset sha256;
- the pending group as a reference: group id, policy version, next iteration and its log record.

The existing items stay as before: optimizer state, step, attempts, RNG states and the training log. A pending group
resumes at its next GRPO iteration from its on-disk record, which keeps the existing ability to resume a group across
multiple iterations.

**Consistency check.** On resume, the restored sampler's batch count, the rollout group and the optimizer steps must
agree with the checkpoint's step and attempts. Otherwise the resume is refused as inconsistent.

## Crash-consistent publication and commit semantics

**Publishing.**
1. A checkpoint (including the initial adapter at step 0) is written into `checkpoints/.tmp-step<N>-<pid>/`.
2. Every file is fsynced.
3. `COMMIT.json` is written and fsynced last, with the step plus the size and sha256 of every file.
4. The directory is renamed to `step<N>`, and its parent directory is fsynced.

So under its final name a checkpoint is either complete or absent.

**Validity.** A checkpoint is valid iff `COMMIT.json` exists and every listed file matches. A resumable checkpoint
must also contain `training_state.pt`.

**Commit semantics.**
- An update is **committed** when the checkpoint containing it is published.
- Resuming from the latest valid checkpoint never re-applies a committed update: its effect is in the restored
  adapter, optimizer state and step counter.
- **Uncommitted** updates and rollout groups are replayed, and their stale records are moved to `orphaned/`.

**Before resuming,** `prepare` deletes interrupted `.tmp-*` publications and quarantines invalid checkpoints as
`.invalid-step<N>-<stamp>`; they are kept, never deleted. An explicitly named invalid checkpoint is refused, with the
latest valid one named in the message.

## Evidence (CPU, tiny random Qwen3, real tokenizer and data, content-hash rewards with zero-signal batches)

`tests/test_streaming.py`:
- A streaming run equals the in-memory run bitwise (rows, zero-signal attempts, evaluations, adapter, AdamW state).
- **Six crash points** (`prompt_read`, `rollouts_generated`, `update`, `files_written`, `committed_before_rename`,
  `after_checkpoint`): each crashed run, resumed from the latest valid checkpoint, commits exactly what the
  uninterrupted run commits. That covers rows, zero-signal attempts, evaluations and counters; on-disk records
  (policy versions, occurrences, generated tokens, rewards); final adapter and AdamW state.
  - No sample occurs twice in an epoch, and completed epochs are complete.
  - No `.tmp-*` directory is left, and every checkpoint validates.
- A checkpoint with one flipped byte is rejected: `latest_valid` falls back to the previous one, explicitly resuming
  it is refused, and it is quarantined. The resumed run still equals the uninterrupted one.
- Unit tests: sampler equivalence and identity refusal, the rollout store's refusals (policy version, run, tampering,
  rewriting a committed group), and all-or-nothing publication.

`scripts/resume_experiment.py --tiny` records the same experiment as JSON: `t09_resume_tiny_cpu.json` (commit
`726e2ab`, clean tree) gives **PASS, H08 SUPPORTED**. All six crash points crashed and resumed (from step 2, or from
step 4 for `after_checkpoint@2`) with committed rows, records and checkpoints identical to run A and no epoch problems.
Run A had 3 attempts (1 zero-signal) and 6 optimizer steps. The real-model version runs in
`scripts/laptop_checks.sh` (`configs/experiments/qwen_stream_short.yaml`: checkpoints every 3 steps, 2 GRPO
iterations per group).

## Limits

- **What counts as a crash.** A crash is an exception raised at the injection point inside the process. Power loss
  is covered only by the fsync/rename protocol, not by an actual power cut.
- **The commit granularity is the checkpoint interval.** Updates between checkpoints are replayed after a crash, at
  the cost of recomputation.
- **Checkpoint evaluation commands** (`verify` / `validate` / `test`) are unaware of `COMMIT.json`. They read the
  adapter as before.
