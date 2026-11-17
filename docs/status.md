# Implementation status and known limitations

State of the code and the evidence, audited before the configuration/CLI refactor and kept current since. A feature
counts as **verified** only when saved evidence of the current code exists. Every PASS that `qwen-grpo ship` reports
lists the commit it was produced at, and the engine files modified since (`research/l2l/final_report.md`, "Evidence
freshness").

## Subsystems

| subsystem | code | in the training path | evidence | status |
|---|---|---|---|---|
| GRPO objective, LoRA (fp32 on a bf16 base) | `grpo.py`, `training.py`, `model.py` | yes | unit tests; runs/qwen (0.6B, 100 steps: val. accuracy 0.7486 → 0.8096) | verified (0.6B) |
| resident backend | `policy.py`, `execution/backend.py` | `backend: resident` | unit tests, runs/qwen | verified |
| explicit Qwen3 forward | `execution/qwen3_forward.py` | via L2L | bitwise vs HF (tests) | verified on tiny models |
| pinned weight store | `execution/weight_store.py` | L2L | t02_* PASS (CUDA) | verified; a pinned 4B store never built |
| synchronous L2L forward / checkpointed backward | `layer_stream.py`, `backward.py` | `l2l_sync` | t03, t04 PASS (tiny, CUDA); 0.6B 100-step run identical to resident | verified, **evidence predates later executor changes** |
| streamed generation | `execution/generator.py` | L2L rollouts and greedy evaluation | t06 PASS (CUDA) | verified, evidence predates later changes |
| full GRPO update through L2L | `backend.py`, `trainer.py` | yes | t07 cuda_v2 PASS; compare_* runs identical | verified, evidence predates later changes |
| async prefetch | `execution/prefetch.py` | `l2l_async` | CPU tests only (fault caught); t08 CUDA cases missing | **implemented, unverified on GPU** |
| memory-efficient output head | `execution/output_head.py` (chunked) | **no** (refused: `vocab_chunk_size`) | v2 PASS at 6657edf (20304 comparisons, CPU), raw-logit fault caught; v1 NOT_SUPPORTED kept | verified on CPU, not used for training |
| disk streaming, crash-consistent resume | `streaming/`, `trainer.py` | `streaming.enabled` | t09_resume_tiny_cpu PASS (6 crash points, tiny model) | verified on a tiny model only |
| streamed LoRA merge | `export/merge.py` | `qwen-grpo merge`, `verify` of L2L runs | t10_merge_equivalence_tiny PASS at 6657edf; the real-model variant is missing | verified on tiny models |
| profiler + memory guard | `execution/profiler.py` | `memory.guard` (new) | guard tested with injected samples on CPU | **never run on an 8 GB-capped GPU** |
| DFlash drafter and verification | `execution/dflash.py` | `dflash.enabled` (rollouts only) | greedy equality, KV crop, chi-square (tiny models) | **unverified on the real model** |
| held-out final evaluation | `scripts/final_eval.py` | `qwen-grpo verify` (L2L runs) | no run yet | not run |
| preflight | `diagnostics/preflight.py` | gates `qwen-grpo train` | offline mode exercised; GPU mode untested on Ampere | offline verified |
| Soup diagnostics | `integrations/soup.py` | optional | `research/soup/0.77.2_cpu/` | executed (CPU) |

Final verdict as derived from the evidence: **DON'T SHIP**, with 6 of 15 requirements PASS and 9 NOT TESTED. Every
remaining requirement needs a GPU (`docs/deployment.md`).

## How the L2L engine schedules work

- **The scheduling is micro-batch-major, not layer-major.** Every pass streams every decoder layer through the GPU
  buffers:
  - each micro-batch of the old-log-prob pass (`training.py`, `prepare`);
  - each micro-batch's forward and its checkpointed backward (`grpo_update`, from the first adapted layer up);
  - every prefill and decode step of generation (all B·G rows at once, not micro-batched).
- The shipped configs leave `micro_batch_size: null`, so the update is one chunk.
- **Weight traffic per GRPO step** is roughly (1 + new tokens) generation sweeps, plus one old-log-prob sweep, plus one
  forward sweep, plus the backward over the adapted layers. Preflight estimates **~44 GB H2D per step** for the
  Qwen3-4B smoke config, which is about 3.7 s at an assumed 12 GB/s.
- The Step-0 budget YAMLs assumed `schedule: layer_major` with a micro-batch of 6. Preflight now models what executes.

## Found by the audit and fixed in this refactor

- `execution.activation_checkpointing` was required but **never read**: `false` was accepted and silently ignored. It is
  now refused (`config.unimplemented`).
- `run.gpu_budget_gib` was only reported. The end-of-run line printed `budget PASS/FAIL` from PyTorch allocator peaks,
  which exclude the CUDA context. Now:
  - `memory.guard` enforces the budget (nvidia-smi sampling, the run stops at the first violation, raw samples kept);
  - the allocator line says it is not budget evidence.
- The final run (`scripts/final_run.sh`) trained without any memory guard. The Qwen3-4B profile now enables it, and
  `train` requires a passing GPU preflight.
- `training.log` had no timestamps, and the provenance lacked the model revision and host RAM. Both are now recorded.
- `README.md` said L2L, streaming and DFlash were not implemented, and that training refused L2L. Rewritten.
- `final_report.py` treated old evidence as current. It now reports the evidence commit and the engine files changed
  since.

## Open decisions (owner)

- **DFlash in the final run.** The assignment asks for DFlash-accelerated rollouts, but
  `configs/experiments/qwen3_4b_final.yaml` does not enable DFlash. Its real-model correctness and speed (H10) are
  unmeasured. Two reasons a speedup is doubtful:
  - Rollouts are only `max_new_tokens: 4` long, so at most 3 drafted tokens can be accepted per sequence.
  - The batched acceptance advances every row by the minimum accepted length across 30 rows
    (`docs/l2l/12_dflash.md`).

  `qwen3_4b_smoke_dflash.yaml` and `research run H10` measure both. Enabling it in the final run is one `dflash:` block.
- **The task.** Binary sentiment with an exact-label reward is verifiable but easy. The 0.6B base already scores 0.75,
  so the room to show learning is smaller than with, for example, exact-match math. Changing the task would change the
  scientific scope, so it is not changed here.

## Unsupported combinations

Refused before training, never silently replaced:
- `resident_layers > 0`
- `vocab_chunk_size`
- `kv_cache: cpu` under `resident`
- DFlash under `resident`
- a drafter for another model (Qwen3-0.6B has none)
- `activation_checkpointing: false`

Accepted but outside the evidence:
- `kv_cache: cpu` with L2L (implemented, never profiled);
- micro-batching with L2L (bitwise-tested on CPU, never profiled on a GPU).

## Known limitations

- **Real-model GPU evidence is missing for Qwen3-4B.** The 8 GB budget, L2L at 4B scale, DFlash, the merge peak and
  held-out learning all lack it. Under WSL2, nvidia-smi gives no per-process memory, so budget evidence from the laptop
  can only be BLOCKED.
- **The passing CUDA equivalence evidence (t03–t07) predates later executor changes.** These are bc5a05d and the
  Ticket 08 refactor of `layer_stream.py`. Rerun it on the GPU host (`docs/deployment.md`, step 6).
- **The resume evidence is tiny-model only.** It covers six crash points on CPU.
- **No forgetting check.** No general benchmark is evaluated (registry I2). Soup's ship requires one.
- **The verification check cannot detect** semantic label noise in the dataset, paraphrased leakage, or gains that
  depend on the prompt template.
- **Soup coverage is partial.** It was measured on CPU only, and `soup doctor` / `env` describe Soup's own interpreter
  unless soup-cli is installed into the training venv.
- **Local hosts differ from the brief.** This audit ran on a 4 GB Quadro P1000 (Pascal) with 31 GiB RAM under WSL2.
  Its torch is `2.14.1+cu130` (the evidence used `+cu126`), so four saved-fixture tests skip as environment-specific.
