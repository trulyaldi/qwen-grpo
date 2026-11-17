**FAULT INJECTION RUN** (detach_boundary=3): a FAIL is the expected outcome.

# t04_gradient_equivalence_detach_boundary: gradient_equivalence (correctness)

**Status: FAIL** (execution PASS; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- hidden_grads.0.6b-like.eager.float32.last2.b1.mb1.2: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.k_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.k_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.v_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.v_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.o_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.self_attn.o_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.mlp.gate_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.mlp.gate_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.mlp.up_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.mlp.up_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.mlp.down_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.mb1.base_model.model.model.layers.2.mlp.down_proj.lora_B.default.weight: missing in candidate
- hidden_grads.0.6b-like.eager.float32.last2.b1.mb2.2: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.acc.base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.acc.base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.acc.base_model.model.model.layers.2.self_attn.k_proj.lora_A.default.weight: missing in candidate
- grads.0.6b-like.eager.float32.last2.b1.acc.base_model.model.model.layers.2.self_attn.k_proj.lora_B.default.weight: missing in candidate
- ... 2284 more

- commit `49e1506196b83a2e762a4c2f60e6e697e4411676` (tracked tree dirty: False), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/store_cpu.yaml` sha256 `cd9f52c56460a60b`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `f3a4ce5e3ad4a6d6`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like.eager.float32.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b1 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.eager.float32.last2.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.float32.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b2 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.eager.float32.last2.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.float32.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b1 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.eager.float32.middle.b1 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.eager.float32.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b2 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.eager.float32.middle.b2 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.eager.float32.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.float32.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.float32.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.float32.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.float32.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.float32.distributed.b1 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.eager.float32.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.float32.distributed.b2 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.eager.bfloat16.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b1 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.eager.bfloat16.last2.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.bfloat16.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b2 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.eager.bfloat16.last2.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.bfloat16.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b1 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.eager.bfloat16.middle.b1 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.eager.bfloat16.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b2 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.eager.bfloat16.middle.b2 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.eager.bfloat16.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.bfloat16.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.bfloat16.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.bfloat16.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.eager.bfloat16.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.bfloat16.distributed.b1 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.eager.bfloat16.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.eager.bfloat16.distributed.b2 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.sdpa.float32.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b1 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.sdpa.float32.last2.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.float32.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b2 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.sdpa.float32.last2.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.float32.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b1 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.sdpa.float32.middle.b1 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.sdpa.float32.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b2 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.sdpa.float32.middle.b2 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.sdpa.float32.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.float32.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.float32.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.float32.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.float32.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.float32.distributed.b1 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.sdpa.float32.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.float32.distributed.b2 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b1 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.last2.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b2 | hidden_grads | 4 | 2/4 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.last2.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b1 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.middle.b1 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b2 | hidden_grads | 4 | 0/4 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.middle.b2 | grads | 28 | 0/28 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b1 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.distributed.b1 | grads | 112 | 28/112 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | hidden_grads | 8 | 2/8 | n/a | None | None |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | grads | 112 | 28/112 | n/a | None | None |
| 4b-like.eager.float32.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b1 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b1 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b1 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b2 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.first_last.b1 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.eager.float32.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.eager.float32.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.first_last.b2 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.eager.float32.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.eager.float32.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.distributed.b1 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.eager.float32.distributed.b1 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.eager.float32.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.distributed.b2 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.eager.float32.distributed.b2 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.eager.bfloat16.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b1 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b1 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b1 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.first_last.b1 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.eager.bfloat16.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.eager.bfloat16.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.first_last.b2 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.eager.bfloat16.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.eager.bfloat16.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.distributed.b1 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.eager.bfloat16.distributed.b1 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.eager.bfloat16.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.distributed.b2 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.eager.bfloat16.distributed.b2 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.sdpa.float32.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b1 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b1 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b1 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.first_last.b1 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.sdpa.float32.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.sdpa.float32.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.first_last.b2 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.sdpa.float32.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.sdpa.float32.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.distributed.b1 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.sdpa.float32.distributed.b1 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.sdpa.float32.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.distributed.b2 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.sdpa.float32.distributed.b2 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.sdpa.bfloat16.final.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b1 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | hidden_grads | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.final.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b1 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b1 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b1 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b1 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | hidden_grads | 6 | 6/6 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.first_last.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.first_last.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.first_last.b1 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.sdpa.bfloat16.first_last.b1 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.sdpa.bfloat16.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.first_last.b2 | hidden_grads | 12 | 2/12 | n/a | None | None |
| 4b-like.sdpa.bfloat16.first_last.b2 | grads | 56 | 28/56 | n/a | None | None |
| 4b-like.sdpa.bfloat16.distributed.b1 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.distributed.b1 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.distributed.b1 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.sdpa.bfloat16.distributed.b1 | grads | 112 | 56/112 | n/a | None | None |
| 4b-like.sdpa.bfloat16.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.distributed.b2 | hidden_grads | 12 | 4/12 | n/a | None | None |
| 4b-like.sdpa.bfloat16.distributed.b2 | grads | 112 | 56/112 | n/a | None | None |


measurements: `{"saved_bytes": {"0.6b-like.eager.float32.final.b1": 92080, "0.6b-like.eager.float32.final.b2": 92080, "0.6b-like.eager.float32.last2.b1": 138672, "0.6b-like.eager.float32.last2.b2": 138672, "0.6b-like.eager.float32.middle.b1": 46592, "0.6b-like.eager.float32.middle.b2": 46592, "0.6b-like.eager.float32.first_last.b1": 154032, "0.6b-like.eager.float32.first_last.b2": 154032, "0.6b-like.eager.float32.distributed.b1": 231856, "0.6b-like.eager.float32.distributed.b2": 231856, "0.6b-like.eager.bfloat16.final.b1": 79920, "0.6b-like.eager.bfloat16.final.b2": 79920, "0.6b-like.eager.bfloat16.last2.b1": 122672, "0.6b-like.eager.bfloat16.last2.b2": 122672, "0.6b-like.eager.bfloat16.middle.b1": 42752, "0.6b-like.eager.bfloat16.middle.b2": 42752, "0.6b-like.eager.bfloat16.first_last.b1": 130352, "0.6b-like.eager.bfloat16.first_last.b2": 130352, "0.6b-like.eager.bfloat16.distributed.b1": 208176, "0.6b-like.eager.bfloat16.distributed.b2": 208176, "0.6b-like.sdpa.float32.final.b1": 92080, "0.6b-like.sdpa.float32.final.b2": 92080, "0.6b-like.sdpa.float32.last2.b1": 138672, "0.6b-like.sdpa.float32.last2.b2": 138672, "0.6b-like.sdpa.float32.middle.b1": 46592, "0.6b-like.sdpa.float32.middle.b2": 46592, "0.6b-like.sdpa.float32.first_last.b1": 154032, "0.6b-like.sdpa.float32.first_last.b2": 154032, "0.6b-like.sdpa.float32.distributed.b1": 231856, "0.6b-like.sdpa.float32.distributed.b2": 231856, "0.6b-like.sdpa.bfloat16.final.b1": 79920, "0.6b-like.sdpa.bfloat16.final.b2": 79920, "0.6b-like.sdpa.bfloat16.last2.b1": 122672, "0.6b-like.sdpa.bfloat16.last2.b2": 122672, "0.6b-like.sdpa.bfloat16.middle.b1": 42752, "0.6b-like.sdpa.bfloat16.middle.b2": 42752, "0.6b-like.sdpa.bfloat16.first_last.b1": 130352, "0.6b-like.sdpa.bfloat16.first_last.b2": 130352, "0.6b-like.sdpa.bfloat16.distributed.b1": 208176, "0.6b-like.sdpa.bfloat16.distributed.b2": 208176, "4b-like.eager.float32.final.b1": 109680, "4b-like.eager.float32.final.b2": 109680, "4b-like.eager.float32.last2.b1": 165872, "4b-like.eager.float32.last2.b2": 165872, "4b-like.eager.float32.middle.b1": 128880, "4b-like.eager.float32.middle.b2": 128880, "4b-like.eager.float32.first_last.b1": 185072, "4b-like.eager.float32.first_last.b2": 185072, "4b-like.eager.float32.distributed.b1": 287856, "4b-like.eager.float32.distributed.b2": 287856, "4b-like.eager.bfloat16.final.b1": 94480, "4b-like.eager.bfloat16.final.b2": 94480, "4b-like.eager.bfloat16.last2.b1": 145872, "4b-like.eager.bfloat16.last2.b2": 145872, "4b-like.eager.bfloat16.middle.b1": 104080, "4b-like.eager.bfloat16.middle.b2": 104080, "4b-like.eager.bfloat16.first_last.b1": 155472, "4b-like.eager.bfloat16.first_last.b2": 155472, "4b-like.eager.bfloat16.distributed.b1": 253456, "4b-like.eager.bfloat16.distributed.b2": 253456, "4b-like.sdpa.float32.final.b1": 109680, "4b-like.sdpa.float32.final.b2": 109680, "4b-like.sdpa.float32.last2.b1": 165872, "4b-like.sdpa.float32.last2.b2": 165872, "4b-like.sdpa.float32.middle.b1": 128880, "4b-like.sdpa.float32.middle.b2": 128880, "4b-like.sdpa.float32.first_last.b1": 185072, "4b-like.sdpa.float32.first_last.b2": 185072, "4b-like.sdpa.float32.distributed.b1": 287856, "4b-like.sdpa.float32.distributed.b2": 287856, "4b-like.sdpa.bfloat16.final.b1": 94480, "4b-like.sdpa.bfloat16.final.b2": 94480, "4b-like.sdpa.bfloat16.last2.b1": 145872, "4b-like.sdpa.bfloat16.last2.b2": 145872, "4b-like.sdpa.bfloat16.middle.b1": 104080, "4b-like.sdpa.bfloat16.middle.b2": 104080, "4b-like.sdpa.bfloat16.first_last.b1": 155472, "4b-like.sdpa.bfloat16.first_last.b2": 155472, "4b-like.sdpa.bfloat16.distributed.b1": 253456, "4b-like.sdpa.bfloat16.distributed.b2": 253456}, "resident_saved_bytes": {"0.6b-like.eager.float32.final": 418384, "0.6b-like.eager.float32.last2": 846288, "0.6b-like.eager.float32.middle": 724816, "0.6b-like.eager.float32.first_last": 1459152, "0.6b-like.eager.float32.distributed": 1702096, "0.6b-like.eager.bfloat16.final": 318480, "0.6b-like.eager.bfloat16.last2": 638288, "0.6b-like.eager.bfloat16.middle": 516816, ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 731.69921875 MiB, high-water mark 731.69921875 MiB (this process)
