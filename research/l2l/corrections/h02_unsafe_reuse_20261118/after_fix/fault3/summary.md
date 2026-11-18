**FAULT INJECTION RUN** (unsafe_reuse=3): a FAIL is the expected outcome.

Detected: 48 configurations refused at runtime (BufferReuseError), 0 caught by the comparisons only.

# fault3: gradient_equivalence (correctness)

**Status: FAIL** (execution FAIL; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- 0.6b-like.eager.float32.last2.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.last2.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.middle.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.middle.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.first_last.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.first_last.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.distributed.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.float32.distributed.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.last2.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.last2.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.middle.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.middle.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.first_last.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.first_last.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.distributed.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.eager.bfloat16.distributed.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.sdpa.float32.last2.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.sdpa.float32.last2.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.sdpa.float32.middle.b2: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- 0.6b-like.sdpa.float32.middle.b3: fault unsafe_reuse=3 not detected: the fault was never injected (no power)
- ... 60 more

- commit `e8440e41c84486e9a30cd1633ffc1029db5c190e` (tracked tree dirty: True), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU Quadro P1000
- config `/home/trulyaldi/qwen-grpo/research/l2l/configs/async_cpu.yaml` sha256 `f6c240cc9bd59f00`; fields used: ['model.dtype', 'execution']; backend `l2l_async`
- input sha256 `cb7569af732767ae`, tolerances sha256 `54e48e883478a48e`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like.eager.float32.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.middle.b3 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.first_last.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b2 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.distributed.b3 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.middle.b3 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.first_last.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b2 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.distributed.b3 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.middle.b3 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.first_last.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b2 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.distributed.b3 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b2 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b2 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b3 | hidden_grads | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.middle.b3 | grads | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b2 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.first_last.b3 | grads | 56 | 56/56 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b2 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b3 | loss | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b3 | logits | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b3 | hidden_grads | 8 | 8/8 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.distributed.b3 | grads | 112 | 112/112 | 0.0 | 0.0 | 0.0 |


measurements: `{"saved_bytes": {"0.6b-like.eager.float32.final.b2": 46040, "0.6b-like.eager.float32.final.b3": 46040, "0.6b-like.eager.float32.last2.b2": 138672, "0.6b-like.eager.float32.last2.b3": 138672, "0.6b-like.eager.float32.middle.b2": 99760, "0.6b-like.eager.float32.middle.b3": 99760, "0.6b-like.eager.float32.first_last.b2": 154032, "0.6b-like.eager.float32.first_last.b3": 154032, "0.6b-like.eager.float32.distributed.b2": 231856, "0.6b-like.eager.float32.distributed.b3": 231856, "0.6b-like.eager.bfloat16.final.b2": 39960, "0.6b-like.eager.bfloat16.final.b3": 39960, "0.6b-like.eager.bfloat16.last2.b2": 122672, "0.6b-like.eager.bfloat16.last2.b3": 122672, "0.6b-like.eager.bfloat16.middle.b2": 83760, "0.6b-like.eager.bfloat16.middle.b3": 83760, "0.6b-like.eager.bfloat16.first_last.b2": 130352, "0.6b-like.eager.bfloat16.first_last.b3": 130352, "0.6b-like.eager.bfloat16.distributed.b2": 208176, "0.6b-like.eager.bfloat16.distributed.b3": 208176, "0.6b-like.sdpa.float32.final.b2": 46040, "0.6b-like.sdpa.float32.final.b3": 46040, "0.6b-like.sdpa.float32.last2.b2": 138672, "0.6b-like.sdpa.float32.last2.b3": 138672, "0.6b-like.sdpa.float32.middle.b2": 99760, "0.6b-like.sdpa.float32.middle.b3": 99760, "0.6b-like.sdpa.float32.first_last.b2": 154032, "0.6b-like.sdpa.float32.first_last.b3": 154032, "0.6b-like.sdpa.float32.distributed.b2": 231856, "0.6b-like.sdpa.float32.distributed.b3": 231856, "0.6b-like.sdpa.bfloat16.final.b2": 39960, "0.6b-like.sdpa.bfloat16.final.b3": 39960, "0.6b-like.sdpa.bfloat16.last2.b2": 122672, "0.6b-like.sdpa.bfloat16.last2.b3": 122672, "0.6b-like.sdpa.bfloat16.middle.b2": 83760, "0.6b-like.sdpa.bfloat16.middle.b3": 83760, "0.6b-like.sdpa.bfloat16.first_last.b2": 130352, "0.6b-like.sdpa.bfloat16.first_last.b3": 130352, "0.6b-like.sdpa.bfloat16.distributed.b2": 208176, "0.6b-like.sdpa.bfloat16.distributed.b3": 208176, "4b-like.eager.float32.final.b2": 0, "4b-like.eager.float32.final.b3": 0, "4b-like.eager.float32.last2.b2": 0, "4b-like.eager.float32.last2.b3": 0, "4b-like.eager.float32.middle.b2": 0, "4b-like.eager.float32.middle.b3": 0, "4b-like.eager.float32.first_last.b2": 37696, "4b-like.eager.float32.first_last.b3": 37696, "4b-like.eager.float32.distributed.b2": 60992, "4b-like.eager.float32.distributed.b3": 60992, "4b-like.eager.bfloat16.final.b2": 0, "4b-like.eager.bfloat16.final.b3": 0, "4b-like.eager.bfloat16.last2.b2": 0, "4b-like.eager.bfloat16.last2.b3": 0, "4b-like.eager.bfloat16.middle.b2": 0, "4b-like.eager.bfloat16.middle.b3": 0, "4b-like.eager.bfloat16.first_last.b2": 30496, "4b-like.eager.bfloat16.first_last.b3": 30496, "4b-like.eager.bfloat16.distributed.b2": 53792, "4b-like.eager.bfloat16.distributed.b3": 53792, "4b-like.sdpa.float32.final.b2": 0, "4b-like.sdpa.float32.final.b3": 0, "4b-like.sdpa.float32.last2.b2": 0, "4b-like.sdpa.float32.last2.b3": 0, "4b-like.sdpa.float32.middle.b2": 0, "4b-like.sdpa.float32.middle.b3": 0, "4b-like.sdpa.float32.first_last.b2": 37696, "4b-like.sdpa.float32.first_last.b3": 37696, "4b-like.sdpa.float32.distributed.b2": 60992, "4b-like.sdpa.float32.distributed.b3": 60992, "4b-like.sdpa.bfloat16.final.b2": 0, "4b-like.sdpa.bfloat16.final.b3": 0, "4b-like.sdpa.bfloat16.last2.b2": 0, "4b-like.sdpa.bfloat16.last2.b3": 0, "4b-like.sdpa.bfloat16.middle.b2": 0, "4b-like.sdpa.bfloat16.middle.b3": 0, "4b-like.sdpa.bfloat16.first_last.b2": 30496, "4b-like.sdpa.bfloat16.first_last.b3": 30496, "4b-like.sdpa.bfloat16.distributed.b2": 53792, "4b-like.sdpa.bfloat16.distributed.b3": 53792}, "resident_saved_bytes": {"0.6b-like.eager.float32.final": 418384, "0.6b-like.eager.float32.last2": 846288, "0.6b-like.eager.float32.middle": 724816, "0.6b-like.eager.float32.first_last": 1459152, "0.6b-like.eager.float32.distributed": 1702096, "0.6b-like.eager.bfloat16.final": 318480, "0.6b-like.eager.bfloat16.last2": 638288, "0.6b-like.eager.bfloat16.middle": 516816, "0.6b-like.eager.bfloat16.first_last": 1034960, "0.6b-like.eager.bfloat16.distributed": 1277904, "0.6b-like.sdpa.float32.final": 41 ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 744.69140625 MiB, high-water mark 754.80859375 MiB (this process)
