# t01_resident_vs_resident: fixture_equivalence (correctness)

**Status: PASS**


- commit `59c68974b553c9ddd84ef42767ca866888f5299e` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/resident.yaml` sha256 `bbd460751d39cdb7`; fields used: ['model.dtype', 'execution']; backend `resident`
- input sha256 `cb7569af732767ae`, tolerances sha256 `b722e3c70069a4f8`

| fixture | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | logp | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | loss | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | grads | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | final_lora | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | optimizer | 36 | 36/36 | 0.0 | 0.0 | 0.0 |
| 0.6b-like | updates | 21 | 21/21 | 0.0 | 0.0 | 0.0 |
| 4b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 4b-like | logp | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 4b-like | loss | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 4b-like | grads | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 4b-like | final_lora | 12 | 12/12 | 0.0 | 0.0 | 0.0 |
| 4b-like | optimizer | 36 | 36/36 | 0.0 | 0.0 | 0.0 |
| 4b-like | updates | 21 | 21/21 | 0.0 | 0.0 | 0.0 |

