**FAULT INJECTION RUN** (weights.1=1e-3): a FAIL is the expected outcome.

# t02_store_equivalence_perturbed_weight: store_equivalence (correctness)

**Status: FAIL** (execution PASS; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- weights.1.self_attn.q_proj.weight: max_abs 3.859e-07 > bound 0.000e+00

- commit `721d064a5ac5a87045dfec1ff2608c9381aecac9` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- config `research/l2l/configs/store.yaml` sha256 `e84db9dd6c03770e`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `0e4e29d6b58b6f1a`

| fixture | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | logp | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | loss | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | grads | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | final_lora | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | optimizer | 36 | 36/36 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | updates | 21 | 21/21 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | weights | 44 | 43/44 | 3.8594589568674564e-07 | 5.5834415782491366e-06 | 0.0 |
| 4b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 4b-like | logp | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 4b-like | loss | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 4b-like | grads | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 4b-like | final_lora | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 4b-like | optimizer | 36 | 36/36 | 0.0 | 0.0 | 0.0 |
| 4b-like | updates | 21 | 21/21 | 0.0 | 0.0 | 0.0 |
| 4b-like | weights | 66 | 66/66 | 0.0 | 0.0 | 0.0 |


measurements: `{"0.6b-like": {"store": {"storage": "pinned", "layers": 4, "slab_bytes": [50176, 50176, 50176, 50176], "store_bytes": 200704, "max_slab_bytes": 50176, "registered_bytes": 217088, "tensors": 44, "dtype": "torch.float32"}, "verify_problems": [], "build_delta": {"allocated": 0, "device_used": 0}, "pinned_views": 44}, "4b-like": {"store": {"storage": "pinned", "layers": 6, "slab_bytes": [72704, 72704, 72704, 72704, 72704, 72704], "store_bytes": 436224, "max_slab_bytes": 72704, "registered_bytes": 460800, "tensors": 66, "dtype": "torch.float32"}, "verify_problems": [], "build_delta": {"allocated": 0, "device_used": 0}, "pinned_views": 66}}`
