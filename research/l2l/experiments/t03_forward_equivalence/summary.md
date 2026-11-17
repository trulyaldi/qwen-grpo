# t03_forward_equivalence: forward_equivalence (correctness)

**Status: PASS** (execution PASS; hypothesis H01: SUPPORTED, all 2690 comparisons within predefined thresholds)


- commit `c4b391be6651fe58f4904ae7ceec6758d4efadfa` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- config `research/l2l/configs/store.yaml` sha256 `e84db9dd6c03770e`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `1ac59eeabd232279`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| fixture.0.6b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| fixture.4b-like | logits | 1 | 1/1 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b1 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b2 | layers | 28 | 28/28 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b1 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b1 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b2 | logits | 7 | 7/7 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b2 | layers | 42 | 42/42 | 0.0 | 0.0 | 0.0 |


measurements: `{"residency": {"0.6b-like.eager.float32.plain": {"before_forward": 9186304, "after_forward": 9186304, "before_each_layer": [9209344, 9209344, 9209344, 9209344], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.disabled": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9364992, 9364992, 9364992, 9364992], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.last2-attention": {"before_forward": 9237504, "after_forward": 9237504, "before_each_layer": [9260544, 9260544, 9260544, 9260544], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.all": {"before_forward": 9360384, "after_forward": 9360384, "before_each_layer": [9383424, 9383424, 9383424, 9383424], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.plain": {"before_forward": 8874496, "after_forward": 8874496, "before_each_layer": [8886784, 8886784, 8886784, 8886784], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.disabled": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9042432, 9042432, 9042432, 9042432], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.last2-attention": {"before_forward": 8916480, "after_forward": 8916480, "before_each_layer": [8928768, 8928768, 8928768, 8928768], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.all": {"before_forward": 9039360, "after_forward": 9039360, "before_each_layer": [9051648, 9051648, 9051648, 9051648], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.plain": {"before_forward": 9186304, "after_forward": 9186304, "before_each_layer": [9205760, 9205760, 9205760, 9205760], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.disabled": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9361408, 9361408, 9361408, 9361408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.last2-attention": {"before_forward": 9237504, "after_forward": 9237504, "before_each_layer": [9256960, 9256960, 9256960, 9256960], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.all": {"before_forward": 9360384, "after_forward": 9360384, "before_each_layer": [9379840, 9379840, 9379840, 9379840], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.plain": {"before_forward": 8874496, "after_forward": 8874496, "before_each_layer": [8885760, 8885760, 8885760, 8885760], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.disabled": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9041408, 9041408, 9041408, 9041408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.last2-attention": {"before_forward": 8916480, "after_forward": 8916480, "before_each_layer": [8927744, 8927744, 8927744, 8927744], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.all": {"before_forward": 9039360, "after_forward": 9039360, "before_each_layer": [9050624, 9050624, 9050624, 9050624], "buffers": [0, 1, 0, 1]}, "4b-like.eager.float32.plain": {"before_forward": 9692160, "after_forward": 9692160, "before_each_layer": [9713664, 9713664, 9713664, 9713664, 9713664, 9713664], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.disabled": {"before_forward": 9993216, "after_forward": 9993216, "before_each_layer": [10014720, 10014720, 10014720, 10014720, 10014720, 10014720], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.last2-attention": {"before_forward": 9747456, "after_forward": 9747456, "before_each_layer": [9768960, 9768960, 9768960, 9768960, 9768960, 9768960], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.all": {"before_forward": 10011648, "after_forward": 10011648, "before_each_layer": [10033152, 10033152, 10033152, 10033152, 10033152, 10033152], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.bfloat16.plain": {"before_forward": 9129472, "after_forward": 9129472, "before_each_layer": [9140736, 9140736, 9140736, 9140736, 9140736, 9140736], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.bfloat16.disabled": {"before_forward": 9430528, "after_forward": 9430528, "before_each_layer": [9441792, 9441792, 9441792, 9441792, 9441792, 9441792], "buffers": [0, 1, 0, 1, 0,  ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **diagnostic**:
- CPU RSS 1641.0078125 MiB, high-water mark 1641.0078125 MiB (this process)
- PyTorch allocator, this process: peak allocated 9.7 MiB, peak reserved 22.0 MiB (excludes the CUDA context and non-PyTorch allocations)
- cudaMemGetInfo total - free: 857.3 MiB, device-wide (every process and context on the GPU, not process-specific)
- nvidia-smi raw: compute apps `322247, [N/A]`, memory.used `134` (unreliable under WSL2)
