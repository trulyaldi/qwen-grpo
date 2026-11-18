# 10 — Every hyperparameter, what it does in the code

**What you will be able to explain after this file**
- The resolved value of every setting of the final run and the file each value comes from.
- For each knob: the exact code that reads it, what it changes mathematically, what it costs in GPU memory and time.
- Which knobs change the optimization trajectory and which only change *how* the same computation is executed.
- Which validation rules reject bad values, and which "settings" are accepted but not actually read.
- Where the chosen values rest on evidence and where they do not.

---

## 1. How the values are resolved

`configs/experiments/qwen3_4b_final.yaml` `extends: qwen3_4b_l2l.yaml` → `../profiles/qwen3_4b.yaml` →
`shared.yaml`. `config.read` merges parents first; mappings merge key by key, scalars replace (`config.py:384-389,
406-425`). Every field is typed and checked (`config.py:192-220`), then `validate` (`config.py:223-265`) and
`execution_checks` (`config.py:268-307`) reject inconsistent values; `unimplemented` (`config.py:310-321`) refuses
settings that are accepted syntactically but not executable. The resolved values below were printed by
`config.load` on the working tree **[code]**.

> [!NOTE]
> Committed `HEAD` (e8440e4) has `micro_batch_size: null`, `learning_rate: 5e-4`, `evaluation.every: 10`,
> `evaluation.batch_size: 32`. The working tree sets 6 / 2e-4 / 20 / 16. Everything below is the working tree.

---

## 2. The master table

Effects marked **[code]** are read off the implementation; memory/time effects for 4B are **[predicted]**.

### 2.1 Data and batching

| Parameter | Current value | Implemented in | Mathematical/computational effect | GPU memory effect | Runtime effect | Why chosen |
|---|---|---|---|---|---|---|
| `data.train_samples` | 1024 (from final yaml) | `training.balanced_subset` (`training.py:57-63`) via `trainer.py:502`; `PromptStream.build` (`trainer.py:647`) | size of the prompt pool; exactly 512 per label; must be even (`config.py:240`) | none | none per step; changes epoch length (1024/5 = 204.8 batches) | assignment asks for 1–2k prompts |
| `data.val_subset` | 128 | `trainer.py:504` | monitoring subset for every-20-steps greedy eval (64 per label, sorted by id) | eval KV cache scales with eval batch, not subset | 128/16 = 8 generate batches per eval | cost of eval under L2L (each batch streams 4 sweeps) |
| `data.subset_seed` | 42 | `trainer.py:502, 504` | which prompts are in the pool / subset; separate from `training.seed` (`config.py:81`) | none | none | seed studies see the same data |
| `training.batch_size` B | 5 | `trainer.py:494`; `PromptStream.next_batch` (`dataset.py:105-116`) | prompts per rollout batch = number of GRPO groups per update | rollout rows N = B·G (KV cache, prefill logits ∝ N) | rollout cost ∝ N | ≥ 1 mixed group per batch is likelier with more groups |
| `training.group_size` G | 6 | `generate_rollouts(..., G)` (`trainer.py:686`) → `repeat_interleave` (`generator.py:115`, `dflash.py:193`); `group_rewards` reshape (`training.py:67-70`) | samples per prompt; the advantage baseline is the group mean and scale the group std; G ≥ 2 required (`config.py:243`) | ∝ N | ∝ N | binary reward: 6 samples give 7 possible reward counts; std is 0 only if all 6 agree |
| `training.micro_batch_size` | 6 | `training.micro_batches` (`training.py:74-81`), used by `prepare` (`training.py:94`) and `grpo_update` (`training.py:116`); chunk loss × n_c/N (`training.py:125-126`) | **none** on the gradient: sum of chunk losses × n_c/N equals the full-batch loss; one AdamW step per batch ([01](01-batches-groups-microbatches.md), [07](07-grpo-mathematics.md)); 1 ≤ m ≤ N (`config.py:250-251`) | bounds activations, boundary checkpoints and [n, R+1, V] logits to 6 rows | **more** weight sweeps: N/m sweeps in `prepare` and N/m in the update (micro-batch-major engine) → 10 sweeps per step instead of 2 | bounded per-chunk memory; matches the Step-0 budget assumption |

### 2.2 Optimization

| Parameter | Current value | Implemented in | Mathematical/computational effect | GPU memory effect | Runtime effect | Why chosen |
|---|---|---|---|---|---|---|
| `training.learning_rate` | 2.0e-4 | `AdamW(lr=...)` (`trainer.py:546`) | AdamW step size; with Adam the first update moves each LoRA element by ≈ lr·sign(g) (bias-corrected m̂/√v̂ ≈ ±1) | none | none | every Qwen3-0.6B run at 5e-4 collapsed toward one label at some point (yaml comment); 2e-4 itself is **untested at any scale** |
| AdamW other args | betas (0.9, 0.999), eps 1e-8 (PyTorch defaults), `weight_decay=0.0` | `trainer.py:546` | no decoupled decay; LoRA B starts at 0 so A has zero gradient at step 1 | moments 8.65 MB fp32 | negligible | — |
| gradient clipping | none | recorded as `grad_clipping: None` (`trainer.py:574`) | unbounded gradient norm; Adam's normalization limits step size anyway | — | — | — |
| `training.num_iterations` | 1 | `trainer.py:731-733`; `grpo_update(iteration=k)` | updates per rollout batch; with 1 the ratio is exactly 1, so the clip never binds ([07](07-grpo-mathematics.md)); `max_steps ≤ max_attempts · num_iterations` (`config.py:245-246`) | `pending` batch kept on GPU across iterations | each extra iteration = 5 more sweeps without new rollouts | on-policy, simplest |
| `training.clip_eps` | 0.2 | `grpo_loss` clamp (`grpo.py:217`); `clip_stats` (`grpo.py:240`) | PPO trust region [0.8, 1.2]; **inactive** at ratio 1 (gradient identical to REINFORCE, verified numerically, max diff 3.7e-9); must be in (0, 1) (`config.py:248`) | none | none | PPO/GRPO default |
| KL β | **0, hard-coded** (not a config key) | `grpo_loss` has no KL term (`grpo.py:199-223`); logged as `kl_beta: 0.0` (`trainer.py:574`) | no pull toward the base policy; `approx_kl` is only a diagnostic (`grpo.py:243`) | saves a reference forward (would need either a second model or adapter-disabled passes) | saves 5 sweeps per step | DAPO/Dr-GRPO-style simplification; nothing in code limits drift except the small lr and 60 steps |
| `training.max_new_tokens` | 4 | `trainer.py:686` → generator loop bound (`generator.py:128`, `dflash.py:211, 213`) | response length cap R ≤ 4; `POSITIVE`+EOS needs 3 tokens; longer answers are truncated and scored on their text | KV capacity P+4 (+16 with DFlash) | ≤ 4 rollout sweeps | the reward needs one label word |
| `training.max_steps` | 60 | loop condition (`trainer.py:672`) | number of AdamW updates | none | linear | a correct, memory-bounded end-to-end run, not maximal accuracy (yaml comment) |
| `training.max_attempts` | 1000 | loop condition (`trainer.py:672`) | cap on rollout batches incl. zero-signal ones (0.6B run needed 142 attempts for 100 steps) | none | upper bound on wasted rollouts | safety stop |
| `training.seed` | 42 | `torch.manual_seed` before LoRA init (`trainer.py:517`) and before sampling (`trainer.py:593`); `PromptStream(seed)` (`trainer.py:647`) | LoRA A init, every rollout sample, batch order | none | none | — |

### 2.3 LoRA

| Parameter | Current value | Implemented in | Mathematical/computational effect | GPU memory effect | Runtime effect | Why chosen |
|---|---|---|---|---|---|---|
| `lora.rank` r | 32 | `lora_config` (`model.py:171-173`) | W_eff = W + (α/r)·B A, A [32, in], B [out, 32] | 1,081,344 params ×4 B, ×4 with grad + 2 moments ≈ 17.3 MB | negligible | shared profile |
| `lora.alpha` α | 64 | same | scale α/r = 2 multiplies the adapter's output (asserted `trainer.py:534`) | none | none | — |
| `lora.targets.modules` | [q_proj, v_proj, o_proj] | `model.target_modules` (`model.py:140-154`) | which projections adapt; k_proj and the MLP stay frozen | — | — | same placement as the 0.6B reference |
| `lora.targets.num_layers` | 2 → layers 34, 35 | same | adapted layers; **sets the backward floor**: only layers ≥ 34 build autograd nodes and are recomputed ([06](06-checkpointed-backward.md)) | 2 boundary checkpoints per chunk; going to k layers adds k−2 more and k recomputations | backward cost ∝ k; with > 2 layers, backward needs reloads (only the last 2 stay in the buffers) | cheapest backward under L2L |
| LoRA dropout | 0.0 (fixed) | `model.py:171-173`; removed key `lora.dropout` must be 0 (`config.py:36`) | recomputation must equal the forward; dropout would differ (`backward.py:241-242, 317-318`) | — | — | correctness of checkpointing |

### 2.4 Evaluation

| Parameter | Current value | Implemented in | Mathematical/computational effect | GPU memory effect | Runtime effect | Why chosen |
|---|---|---|---|---|---|---|
| `evaluation.every` | 20 | `trainer.py:764` | subset eval **and checkpoint** every 20 steps (step0/20/40/60); also the commit granularity for resume ([12](12-streaming-and-resume.md)) | eval KV cache | ~48 sweeps per eval (budget) | the DFlash drift study needs exactly those checkpoints |
| `evaluation.full_validation` | false | `trainer.py:583, 807` | no in-process full-validation (1066) comparison; held-out test is `scripts/final_eval.py` only | — | saves ~1066-example evals | cost |
| `evaluation.max_new_tokens` | 4 | `evaluation.generate` (`evaluation.py:21-39`) | greedy answer length | small | — | label = 2 tokens + EOS (`evaluation.py:13`) |
| `evaluation.batch_size` | 16 | `evaluation.predict` chunks (`evaluation.py:42-59`) | **changes bf16 outputs** (batch shape changes kernels/rounding; `evaluation.py:43`) → affects reported accuracy slightly, not training | KV cache ∝ 16 | 8 batches per eval | smaller eval batches under the cap |
| `evaluation.score_batch_size` | 8 | `label_scores` (`evaluation.py:143-161`) | AUC scoring; only used with full validation / final eval | ∝ 8 | — | default |
| `evaluation.bootstrap_samples` | 10000 | `paired_bootstrap` (`trainer.py:808-811`); multiple of 1000 (`config.py:256-257`) | CI resolution | CPU | CPU seconds | — |

### 2.5 Execution (should not change results)

| Parameter | Current value | Implemented in | Mathematical/computational effect | GPU memory effect | Runtime effect | Why chosen |
|---|---|---|---|---|---|---|
| `model.dtype` | bfloat16 | `trainer.py:498, 522`; LoRA always fp32 (asserted `trainer.py:536-537`) | frozen weights and activations in bf16; LoRA math in fp32 inside PEFT, cast back to bf16 | halves every weight term vs fp32 | bf16 tensor cores | 4B in fp32 = 16 GB host store; the 0.6B Kaggle T4 runs had to use fp32 |
| `execution.backend` | l2l_async | `make_backend` (`backend.py:90-114`) | none if equivalence holds | weights off the GPU | adds transfers; async aims to hide them | assignment: model cannot be resident under 8 GB |
| `execution.weight_storage` | pinned | `WeightStore._adopt` (`weight_store.py:262-269`) | none | none on GPU; 7.27 GB page-locked RAM | async DMA possible only from pinned (measured on the laptop: 6.58 vs 4.44 GB/s pinned vs pageable, `research/l2l/literature.md`) | assignment requires pinned; `pageable` is the opt-out |
| `execution.resident_layers` | 0 | refused if > 0 (`config.py:316`) | — | — | — | not implemented |
| `execution.weight_buffers` | 2 | `gpu_buffers` (`backend.py:69`); ≥ depth+1 (`config.py:303-305`) | none | 2 × 201.9 MB; each extra buffer +201.9 MB | more buffers let deeper prefetch; with 2 the backward of layers 34–35 is free (both still resident) | minimum for depth 1 |
| `execution.prefetch_depth` | 1 | `PrefetchMixin._upcoming` (`prefetch.py:137-150`); ≥ 1 for async (`config.py:302`) | none | — | how many layers ahead are copied on the transfer stream | double buffering |
| `execution.activation_checkpointing` | true | **not read**; `false` is refused (`config.py:320-321`) | the L2L executors always checkpoint layer boundaries | — | — | documentation-only key |
| `execution.vocab_chunk_size` | null | refused if set (`config.py:317`) | chunked head (H03) off the training path | full [n, R+1, V] logits | — | H03 dropped from the final protocol |
| `execution.kv_cache` | gpu | `backend.py:96` → `KVCache(device)` | none | KV on GPU (0.34–0.41 GB); `cpu` moves it to RAM and copies each layer's prefix in per call | `cpu` adds a PCIe copy per layer per step | fits under the cap |
| `dflash.enabled` / `draft_model` | true / z-lab/Qwen3-4B-DFlash-b16 | `backend.py:99-110`; drafter must match the model (`config.py:279-282`) | rollouts exact in law, not bitwise vs plain ([08](08-dflash-integration.md)) | +1.075 GB resident + ~2 GB prefill transient | saves ≤ 1 of ~13 sweeps per step [predicted] | Ticket 12 / H10 |
| `streaming.enabled` / `directory` | true / `streaming` | `trainer.py:644-662` | changes nothing numerically (sampler is the same as `shuffled_batches`) | none | disk writes + fsync per attempt | crash-consistent resume (H08) |
| `memory.guard` | true | `trainer.memory_guard` (`trainer.py:400-415`) | stops the run when a sample exceeds **8 GiB** (not 8e9 B) | the sampler thread itself none | nvidia-smi subprocess every 0.5 s | enforce, do not just report |
| `memory.interval_s` | 0.5 | `Profiler(interval=...)` | sampling period; spikes shorter than it can be missed | — | — | — |
| `memory.allocator_cap` | true | `set_per_process_memory_fraction` (`trainer.py:412-414`); requires `guard` (`config.py:284-286`) | caps PyTorch reserved at 8 GiB; excludes the CUDA context | an allocation beyond it raises OOM immediately | — | catch spikes between samples |
| `run.gpu_budget_gib` | 8.0 | guard threshold = 8 × 2³⁰ (`trainer.py:408`) | — | — | — | H11's pre-registered threshold; the orchestrator's stricter 8e9 B cap is separate ([09](09-memory-and-tensor-lifetimes.md)) |

---

## 3. What changing each one would actually do

**Optimization-affecting** (the trajectory changes; equivalence tests say nothing about these):

- **B ↑**: more groups per update → lower-variance gradient and fewer zero-signal batches (a batch is skipped only if
  *all* B groups are uniform, `training.py:90`). Rollout memory ∝ B·G (prefill logits ∝ N·P·V) — with DFlash this is the
  first thing to hit the cap.
- **G ↑**: better per-group baseline; more chance a group is mixed; same memory scaling as B. G = 2 is the minimum
  (population std of 2 samples).
- **lr**: the only scale of the step size under Adam (gradient magnitude is normalized away per element). Too high:
  label collapse (0.6B at 5e-4: final predicted-POSITIVE rate 0.968, accuracy 0.528 at step 100, `runs/qwen/report.md`
  **[measured]**). 2e-4 has no evidence either way.
- **num_iterations > 1**: reuse each rollout batch; the clip starts to bind; `pending` must survive checkpoints (that
  code path is then exercised); costs 5 sweeps per extra iteration.
- **clip_eps**: irrelevant at num_iterations = 1; matters only with reuse.
- **max_new_tokens**: < 3 makes `POSITIVE`/`NEGATIVE` unreachable with EOS (they are 2 tokens + EOS); the 1-token
  `Positive`/`Negative` spellings still earn reward because `parse_label` upper-cases (`sentiment.py:35-42`). Larger
  values add rollout sweeps and KV memory.
- **LoRA r/α, targets, num_layers**: capacity and the scale of W_eff; adding earlier layers lowers the backward floor —
  more recomputation and, beyond 2 layers, backward weight reloads.
- **seed / subset_seed**: different samples and data; single-seed results have no seed-variance estimate.
- **max_steps, max_attempts, train_samples**: length and coverage; 60 steps × 5 = 300 selections, < 1/3 of an epoch.

**Execution-only** (must leave training rows bitwise identical whenever the equivalence hypotheses hold):
`backend`, `weight_storage`, `weight_buffers`, `prefetch_depth`, `kv_cache`, `streaming.*`, `memory.*`.
Evidence: sync L2L = resident bitwise for 100 real 0.6B steps **[measured]**; async = sync on tiny CPU models
**[test]** (`tests/test_prefetch.py:111`).

**In between:**
- `micro_batch_size` is mathematically neutral, but in bf16 the chunk shape changes kernel choice and rounding.
  The code keeps `prepare` and the update in **the same** chunks so iteration 0 sees ratio exactly 1
  (`training.py:86-88`); switching sizes between runs changes results at rounding level.
- `evaluation.batch_size` changes greedy outputs at rounding level (`config.py:101-102`, `evaluation.py:43`): monitoring
  numbers are only comparable at equal eval batch.
- `dflash.enabled` preserves the sampling law but not the sample sequence.
- `model.dtype` changes everything numerically.

## 4. Genuine issues worth knowing (not changes)

1. **lr 2e-4 is a guess** informed only by 0.6B collapse at 5e-4; no run at any scale used it.
2. **The guard enforces 8 GiB, the assignment is read as 8e9 B** (562.6 MiB apart); only the orchestrator's watchdog
   enforces the stricter one.
3. **`activation_checkpointing` is a decorative key**: it is never read; only `false` is refused.
4. **micro_batch_size 6 multiplies PCIe traffic**: 10 sweeps for prepare + update instead of 2 at micro-batch null. The
   memory saving is real but small for 4-token responses (logits [30, 5, V] fp32 = 91 MB vs 18 MB per chunk).
5. **No KL term and no grad clipping**: drift from the base is limited only by lr × steps.

## Open questions / unverified

- Whether lr 2e-4 avoids one-label collapse at 4B: the final run itself (`H12.train_final`), watching `predicted_positive_rate` in subset evals.
- Wall-time cost of micro-batch 6 vs null at 4B (5× more sweeps vs smaller transients): `H11.profile` with `--micro-batch-size 6` and a second profile at null — only the first is in the protocol.
- Whether execution-only knobs are bitwise neutral on GPU for async L2L: `H05.lab_async_cuda`, `qwen3_0.6b_{resident,l2l_async}_short` + `scripts/compare_runs.py`.
- Seed variance of every result: no multi-seed study exists.
