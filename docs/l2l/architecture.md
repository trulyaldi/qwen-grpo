# L2L architecture audit (Ticket 00)

Where the resident GRPO + LoRA code assumes that the whole model sits on one device, and what an L2L backend has to
replace. The GRPO step itself (tensor shapes and the objective) is described in [`docs/architecture.md`](../architecture.md).
Resident behaviour, the tests that pin it and the frozen fixtures are in [`resident_reference.md`](resident_reference.md).
Line numbers refer to commit `805a7b1`.

## The one model object

Every phase uses a single `PeftModel` returned by `model.lora_model`:

```
AutoModelForCausalLM.from_pretrained(name, dtype) .to(device) .eval()   # whole model, one device
get_peft_model(base, LoraConfig(target_modules=...))                     # lora.Linear replaces q/v/o in the last layers
```

So the base weights, the tied embedding / LM head, and the fp32 LoRA A/B factors all live on `device`. The LoRA factors
are inside the decoder-layer modules (`layers[i].self_attn.q_proj.lora_A["default"]`). Moving a layer module
with `.to()` therefore moves its adapters too, and that would replace the `Parameter` objects AdamW holds. An L2L
backend has to move base weights *only* and leave the trainable LoRA tensors in place, with the same identities.

`model.device` (the device of the first parameter) is used as "where inputs go". Once weights live on CPU, it means CPU,
not the device where computation runs.

## Execution paths requiring modification

| # | Location | What it does | Resident assumption | Ticket |
|---|---|---|---|---|
| 1 | `qwen_grpo/model.py:78` `lora_model` | `from_pretrained(...).to(device)` | whole model moved to the GPU in one step; a full CPU copy exists first | 02 |
| 2 | `qwen_grpo/policy.py:16` `generate_rollouts` | `tokenizer(...).to(model.device)` | input device = model device | 06 |
| 3 | `qwen_grpo/policy.py:19` `generate_rollouts` | `model.generate(do_sample=True, T=1, top_p=1, top_k=0)` | HF generation loop + HF KV cache, all layers resident | 06 |
| 4 | `qwen_grpo/policy.py:48` `compute_logprobs` | `model(..., logits_to_keep=R+1).logits` | full-model forward, autograd graph over every layer kept until backward | 03–05 |
| 5 | `qwen_grpo/training.py:84` `prepare` → `compute_logprobs` | old log-probs, no grad | same as 4 (no graph) | 05, 07 |
| 6 | `qwen_grpo/training.py:96` `grpo_update` → `compute_logprobs` + `backward()` | differentiable forward, plain autograd backward | the whole graph and every layer's activations stay on the GPU until `backward()` | 04, 05, 07 |
| 7 | `qwen_grpo/training.py:90` `prepare` | advantages `.to(rollouts["input_ids"].device)` | rollouts on the compute device (fine; keep it) | 07 |
| 8 | `qwen_grpo/evaluation.py:23-24` `generate` | `.to(model.device)`, `model.generate(do_sample=False)` | as 2–3, greedy | 06, 10 |
| 9 | `qwen_grpo/evaluation.py:143-150` `label_scores` | `.to(model.device)`, `compute_logprobs` | as 4, no grad | 05, 10 |
| 10 | `qwen_grpo/verification.py:186` `sequence_logits` | `model(input_ids).logits`, full `[T, V]` logits per example | full-model forward and full logits | 10 |
| 11 | `qwen_grpo/verification.py:228-244` `merge_check` | `model.merge_and_unload()` in place on the model's device | whole model on the GPU during the merge (bf16 check) | 10 |
| 12 | `qwen_grpo/verification.py:253` `fp32_merge_check` | `lora_model(torch.float32, "cpu")`, whole model in fp32 on CPU | estimated ≈ 4.0 B params × 4 bytes ≈ 16 GB for Qwen3-4B (arithmetic, not measured), more than this host's 6.7 GiB RAM | 10 |
| 13 | `qwen_grpo/verification.py:360`, `scripts/run.py:56` | `lora_model(..., "cuda")` for verify / validate / test | whole model on the GPU | 10, 11 |
| 14 | `qwen_grpo/trainer.py:207` | `lora_model(dtype, device)` | as 1 | 07 |
| 15 | `qwen_grpo/trainer.py:274, 304` | pending batch and prompt ids `.to(model.device)` | as 2 | 07 |
| 16 | `qwen_grpo/trainer.py:369-373`, `checkpointing.py:42-50` | adapter reload, `torch.equal(p, loaded.to(p.device))` | adapter tensors on the model device; must keep working with L2L | 07, 09 |
| 17 | `qwen_grpo/checkpointing.py:102` `load_training_state` | `torch.load(path, weights_only=True)` with no `map_location` | the state was saved from CUDA tensors and is restored to CUDA; fails on a host without CUDA (see baseline failures) | 09 |
| 18 | `qwen_grpo/trainer.py:337-342, 380-385` | memory record = `torch.cuda.max_memory_*` only | allocator statistics only: no CUDA context, no non-PyTorch allocations, no nvidia-smi, no CPU/pinned RAM | 11 |
| 19 | `qwen_grpo/trainer.py:402`, `verification.py:235, 367, 376`, `scripts/run.py:63` | `with model.disable_adapter()` for base-model evaluation | PEFT's adapter switch is a module flag; a streamed layer must respect it | 03, 10 |
| — | `research/train_grpo.py:81`, `research/baseline.py:41`, `research/train_lora.py:115` | `.cuda()` / `lora_model(..., "cuda")` | legacy Stage 1–2 scripts, not used by `scripts/run.py`; not to be ported | — |

Already layer-explicit and not resident-specific: `qwen_grpo/execution/qwen3_forward.py`. `explicit_forward` owns the
layer loop and calls `layer_executor.execute(index, layer, hidden, **kwargs)` for each decoder layer. Embedding, mask,
RoPE table, final norm and LM head run outside the executor. These, and the tied `embed_tokens` / `lm_head` weight,
are the pieces an L2L backend must place explicitly. `explicit_forward` is not yet called by training or evaluation
(rows 4–9 still use `model(...)`).

## State that crosses layer boundaries

From `explicit_forward`: `hidden [N, T, H]`, the mask dict (`None` or `[N, 1, T, T]` per attention type),
`position_ids [N, T]`, and `(cos, sin)` each `[N, T, head_dim]`. These are built once before the layer loop. A streamed
executor needs only these plus the layer's weights. With backward (Ticket 04), it also needs the boundary `hidden`
checkpoint of each layer that is recomputed.
