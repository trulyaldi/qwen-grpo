# t08_forward_equivalence_async_r20261119T051124Z: forward_equivalence (correctness)

**Status: PASS** (execution PASS; hypothesis H01: SUPPORTED, all 2690 comparisons within predefined thresholds)


- commit `29852d11c2a37eeda94cd5ee53692fffd8d75d69` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3060
- config `research/l2l/configs/async.yaml` sha256 `f3d2268fb2f68f9b`; fields used: ['model.dtype', 'execution']; backend `l2l_async`
- input sha256 `cb7569af732767ae`, tolerances sha256 `54e48e883478a48e`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| fixture.0.6b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| fixture.4b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b3 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b3 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b3 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |


measurements: `{"residency": {"0.6b-like.eager.float32.plain": {"before_forward": 9186304, "after_forward": 9186304, "before_each_layer": [9209344, 9209344, 9209344, 9209344], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.disabled": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9364992, 9364992, 9364992, 9364992], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.last2-attention": {"before_forward": 9219072, "after_forward": 9219072, "before_each_layer": [9242112, 9242112, 9242112, 9242112], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.all": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9364992, 9364992, 9364992, 9364992], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.plain": {"before_forward": 8874496, "after_forward": 8874496, "before_each_layer": [8886784, 8886784, 8886784, 8886784], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.disabled": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9042432, 9042432, 9042432, 9042432], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.last2-attention": {"before_forward": 8907264, "after_forward": 8907264, "before_each_layer": [8919552, 8919552, 8919552, 8919552], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.all": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9042432, 9042432, 9042432, 9042432], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.plain": {"before_forward": 9186304, "after_forward": 9186304, "before_each_layer": [9205760, 9205760, 9205760, 9205760], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.disabled": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9361408, 9361408, 9361408, 9361408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.last2-attention": {"before_forward": 9219072, "after_forward": 9219072, "before_each_layer": [9238528, 9238528, 9238528, 9238528], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.all": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9361408, 9361408, 9361408, 9361408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.plain": {"before_forward": 8874496, "after_forward": 8874496, "before_each_layer": [8885760, 8885760, 8885760, 8885760], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.disabled": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9041408, 9041408, 9041408, 9041408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.last2-attention": {"before_forward": 8907264, "after_forward": 8907264, "before_each_layer": [8918528, 8918528, 8918528, 8918528], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.all": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9041408, 9041408, 9041408, 9041408], "buffers": [0, 1, 0, 1]}, "4b-like.eager.float32.plain": {"before_forward": 9692160, "after_forward": 9692160, "before_each_layer": [9713664, 9713664, 9713664, 9713664, 9713664, 9713664], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.disabled": {"before_forward": 9993216, "after_forward": 9993216, "before_each_layer": [10014720, 10014720, 10014720, 10014720, 10014720, 10014720], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.last2-attention": {"before_forward": 9729024, "after_forward": 9729024, "before_each_layer": [9750528, 9750528, 9750528, 9750528, 9750528, 9750528], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.all": {"before_forward": 9993216, "after_forward": 9993216, "before_each_layer": [10014720, 10014720, 10014720, 10014720, 10014720, 10014720], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.bfloat16.plain": {"before_forward": 9129472, "after_forward": 9129472, "before_each_layer": [9140736, 9140736, 9140736, 9140736, 9140736, 9140736], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.bfloat16.disabled": {"before_forward": 9430528, "after_forward": 9430528, "before_each_layer": [9441792, 9441792, 9441792, 9441792, 9441792, 9441792], "buffers": [0, 1, 0, 1, 0, 1] ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **assignment**:
- CPU RSS 1699.015625 MiB, high-water mark 1699.015625 MiB (this process)
- PyTorch allocator, this process: peak allocated 9.7 MiB, peak reserved 22.0 MiB (excludes the CUDA context and non-PyTorch allocations)
- cudaMemGetInfo total - free: 198.1 MiB, device-wide (every process and context on the GPU, not process-specific)
- nvidia-smi raw: compute apps `8690, 178`, memory.used `199` (unreliable under WSL2)
