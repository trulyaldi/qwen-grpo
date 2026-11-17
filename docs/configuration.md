# Configuration

One schema (`qwen_grpo/config.py`), small YAML files layered with `extends`. Every run saves the fully resolved config,
so what ran is always explicit.

## Layout

```
configs/
  profiles/
    shared.yaml          abstract: model-independent defaults (GRPO, LoRA rank, data, evaluation); cannot be run
    qwen3_0_6b.yaml      Qwen3-0.6B, resident, 4 GB laptop (the reference run's settings)
    qwen3_4b.yaml        Qwen3-4B, l2l_async + pinned store, 8 GiB budget, memory guard
  default.yaml, qwen.yaml   the reference run (runs/qwen); self-contained, pinned equal to profiles/qwen3_0_6b.yaml
  gpu/*.yaml             canonical resident / l2l_sync / l2l_async comparisons for Qwen3-0.6B and Qwen3-4B
  experiments/*.yaml     small overrides of a profile or of another experiment
  budget/*.yaml          inputs of the Step-0 budget calculator (scripts/budget.py), a separate schema; preflight
                         now derives the same inputs from a training config
```

`extends: <path relative to this file>` loads the parent first; mappings merge key by key, any other value (list,
scalar, `null`, a `lora.targets` preset) replaces the parent's. A file that extends another must set its own
`run.name`. `abstract: true` marks a shared base: it can be extended but `load()` refuses it, and the marker is not
inherited or hashed. The result is validated as a whole: unknown keys, missing required keys, wrong types (`True` for
an int, `"1e-4"` for a float: write `1.0e-4`) and inconsistent combinations are errors.

## Selecting a setup

**Canonical comparisons (`configs/gpu/`).** Within one model, the three files differ only in `execution`; the 4B
resident file also differs in the guard ceiling, `run.gpu_budget_gib`. Model, data, LoRA, GRPO hyperparameters,
seed, prompt, reward and evaluation are identical. DFlash and disk streaming are off.

| config | model | backend | GPU target | purpose |
|---|---|---|---|---|
| `gpu/qwen3_0.6b_resident.yaml` | Qwen3-0.6B | resident | any ≥ 4 GB (est. 2.2 GiB) | reference |
| `gpu/qwen3_0.6b_l2l_sync.yaml` | Qwen3-0.6B | l2l_sync, pinned, 2 buffers | ≥ 4 GB (est. 1.4 GiB) | L2L = resident? |
| `gpu/qwen3_0.6b_l2l_async.yaml` | Qwen3-0.6B | l2l_async, prefetch 1 | ≥ 4 GB | prefetch effect |
| `gpu/qwen3_4b_resident.yaml` | Qwen3-4B | resident, guard 16 GiB | ≥ 16 GB (est. 9.5 GiB) | 4B reference |
| `gpu/qwen3_4b_l2l_sync.yaml` | Qwen3-4B | l2l_sync, guard 8 GiB | 8 GiB research budget | 4B L2L, no overlap |
| `gpu/qwen3_4b_l2l_async.yaml` | Qwen3-4B | l2l_async, guard 8 GiB | 8 GiB research budget | 4B L2L, prefetching |

The estimates come from `qwen-grpo preflight` (the calculator, unmeasured). All six use `num_iterations: 1`: every
update is taken at the sampling policy, so the importance ratio is exactly 1 and clipping never binds. That is pinned
by `tests/test_training_invariants.py`. Use `num_iterations > 1` (e.g. `experiments/grpo_clip_k3.yaml`) to study
clipping. `training.micro_batch_size` splits the B·G completions of one update into chunks whose gradients accumulate
before a single optimizer step. It does not change B, G or the loss, but under L2L every chunk streams the layers
again.

**Pinned or pageable.** L2L defaults to pinned host memory. `weight_storage: pageable` is the documented fallback
where page-locking fails; it is slower for host-to-GPU copies, and must be recorded as a different setup.

**Other configs:**

| goal | config | backend | notes |
|---|---|---|---|
| Qwen3-0.6B reference (resident) | `configs/profiles/qwen3_0_6b.yaml` (or `configs/default.yaml`) | `resident` | 100 steps, full validation |
| Qwen3-0.6B through the 4B code path | `configs/experiments/qwen_local.yaml` | `l2l_async` + streaming | 10 steps; laptop development |
| resident vs synchronous L2L | `configs/experiments/qwen_resident.yaml` vs `qwen_l2l.yaml` | `resident` / `l2l_sync` | bitwise comparison: `scripts/compare_runs.py` |
| Qwen3-4B smoke | `configs/experiments/qwen3_4b_smoke.yaml` | `l2l_async` + streaming + guard | 4 steps, the final run's batch shape |
| Qwen3-4B smoke with DFlash | `configs/experiments/qwen3_4b_smoke_dflash.yaml` | + `dflash` | drafter `z-lab/Qwen3-4B-DFlash-b16` |
| Qwen3-4B final | `configs/experiments/qwen3_4b_final.yaml` | `l2l_async` + streaming + guard | 1024 prompts, 210 steps; DFlash off (see docs/status.md) |
| LR=0 negative control | `configs/experiments/qwen3_4b_final_lr0.yaml` | as final | 30 steps, `learning_rate: 0.0` |

Execution backends (`execution` section):

```yaml
execution: {backend: resident}                                   # whole model on the GPU (HF generate)
execution: {backend: l2l_sync, weight_storage: pinned, resident_layers: 0, weight_buffers: 2,
            prefetch_depth: 0, activation_checkpointing: true, kv_cache: gpu}
execution: {backend: l2l_async, weight_storage: pinned, resident_layers: 0, weight_buffers: 2,
            prefetch_depth: 1, activation_checkpointing: true, kv_cache: gpu}
dflash: {enabled: true, draft_model: z-lab/Qwen3-4B-DFlash-b16}  # with an l2l backend and model Qwen/Qwen3-4B only
```

## Schema

| section | keys (default) | notes |
|---|---|---|
| `run` | `name`, `gpu_budget_gib` (4.0), `overwrite` (false) | `name` names `runs/<name>/`; the budget includes the CUDA context and is enforced only with `memory.guard` |
| `model` | `name`, `dtype` (bfloat16) | `Qwen/Qwen3-<size>B`, always non-thinking |
| `lora` | `rank`, `alpha`, `targets` | preset `last2-attention` / `last-o-proj`, or `{modules: [...], num_layers: N}` (last N layers of the loaded model) |
| `data` | `train_samples`, `val_subset`, `subset_seed` (42) | balanced subsets of `data/sentiment.json` |
| `training` | `batch_size`, `group_size`, `learning_rate`, `max_steps`, `max_attempts`, `max_new_tokens`, `seed`, `clip_eps` (0.2), `num_iterations` (1), `micro_batch_size` (null) | `learning_rate: 0.0` is the negative control; `null` micro-batch = all B·G rows at once |
| `evaluation` | `every`, `full_validation`, `max_new_tokens` (4), `batch_size` (32), `score_batch_size` (8), `bootstrap_samples` (10000) | greedy; defaults match the saved baseline |
| `verification` | `split` (validation), `control_checkpoint` (null), `train_control` (false) | `verify` of resident runs |
| `execution` | `backend` (resident), `weight_storage`, `resident_layers`, `weight_buffers`, `prefetch_depth`, `activation_checkpointing`, `vocab_chunk_size`, `kv_cache` (gpu) | the null keys are required for `l2l_*` and must stay null for `resident` |
| `streaming` | `enabled` (false), `directory` (null) | disk-backed prompts and rollout records, crash-consistent checkpoints, `resume` from the latest valid one |
| `dflash` | `enabled` (false), `draft_model` (null) | the drafter must be trained for `model.name` |
| `memory` | `guard` (false), `interval_s` (0.5), `allocator_cap` (false) | guard: nvidia-smi sampling during `train`, the run stops (failed) at the first sample above the budget; raw samples in `training/memory/`. `allocator_cap` needs `guard` |

## Refused combinations

Validation errors (the config does not load): `model.name` not a dense Qwen3 id; a drafter for another target (e.g.
Qwen3-0.6B with `z-lab/Qwen3-4B-DFlash-b16`); `dflash.enabled` without `draft_model` or the reverse; L2L keys set under
`resident`, or missing under `l2l_*`; `prefetch_depth` ≠ 0 for `l2l_sync` or < 1 for `l2l_async`;
`weight_buffers < prefetch_depth + 1`; `micro_batch_size` outside 1..B·G; `group_size < 2`; odd or too small subsets;
`max_steps > max_attempts · num_iterations`; `memory.interval_s <= 0`; `memory.allocator_cap` without `guard`.

Loadable but refused by `train` (`config.unimplemented`, no silent fallback): `execution.resident_layers > 0`,
`vocab_chunk_size`, `kv_cache: cpu` with `resident`, DFlash with `resident`, and `activation_checkpointing: false` under
L2L (the executors always checkpoint; the key used to be silently ignored).

## Identity and provenance

The SHA-256 of the resolved config is in `config.yaml`'s first line, `manifest.json`, `training.json`, every checkpoint
and every evaluation output. Optional sections (`execution`, `streaming`, `dflash`, `memory`) are left out of the hash
while at their defaults, so configs older than a section keep their hash (pinned by `tests/test_profiles.py`). Each
training session also records the git commit and dirty flag, Python and package versions, the dataset file's sha256
and split sizes, the device and GPU, host CPU count and RAM, and the cached revision of the target model (and drafter);
seeds and the backend are part of the resolved config. `preflight` records the memory signature (model, LoRA,
execution, DFlash, memory, budget and batch shape): a GPU preflight vouches for every config with the same signature.
