**FAULT INJECTION RUN** (wrong_layer=0): a FAIL is the expected outcome.

# t03_forward_equivalence_wrong_layer: forward_equivalence (correctness)

**Status: FAIL** (execution PASS; hypothesis H01: NOT_EVALUATED, fault-injection run: it tests the test's power, not the hypothesis)

- logits.fixture.0.6b-like.cpu: max_abs 3.743e-01 > bound 0.000e+00
- logits.fixture.4b-like.cpu: max_abs 4.494e-01 > bound 0.000e+00
- logits.0.6b-like.eager.float32.plain.b1.unpadded1.k0: max_abs 2.550e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded1.k0.0: max_abs 7.058e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded1.k0.1: max_abs 9.055e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded1.k0.2: max_abs 1.145e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded1.k0.3: max_abs 9.336e-02 > bound 0.000e+00
- logits.0.6b-like.eager.float32.plain.b1.unpadded5.k0: max_abs 4.626e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded5.k0.0: max_abs 7.551e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded5.k0.1: max_abs 8.501e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded5.k0.2: max_abs 9.103e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded5.k0.3: max_abs 1.099e-01 > bound 0.000e+00
- logits.0.6b-like.eager.float32.plain.b1.unpadded9.k0: max_abs 3.501e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded9.k0.0: max_abs 6.856e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded9.k0.1: max_abs 1.015e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded9.k0.2: max_abs 1.281e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.unpadded9.k0.3: max_abs 1.398e-01 > bound 0.000e+00
- logits.0.6b-like.eager.float32.plain.b1.left9.k0: max_abs 3.594e-01 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.left9.k0.0: max_abs 6.856e-02 > bound 0.000e+00
- layers.0.6b-like.eager.float32.plain.b1.left9.k0.1: max_abs 8.622e-02 > bound 0.000e+00
- ... 2670 more

- commit `c4b391be6651fe58f4904ae7ceec6758d4efadfa` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU
- config `research/l2l/configs/store.yaml` sha256 `e84db9dd6c03770e`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `1ac59eeabd232279`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| fixture.0.6b-like | logits | 1 | 0/1 | 0.3742571622133255 | 0.9926477353568888 | 0.0 |
| fixture.4b-like | logits | 1 | 0/1 | 0.4494065195322037 | 1.196280455922463 | 0.0 |
| 0.6b-like.eager.float32.plain.b1 | logits | 7 | 0/7 | 0.48242632299661636 | 0.7987347210763783 | 0.0 |
| 0.6b-like.eager.float32.plain.b1 | layers | 28 | 0/28 | 0.1397574543952942 | 1.2573793226602035 | 0.0 |
| 0.6b-like.eager.float32.plain.b2 | logits | 7 | 0/7 | 0.48242632299661636 | 0.7987347210763783 | 0.0 |
| 0.6b-like.eager.float32.plain.b2 | layers | 28 | 0/28 | 0.1397574543952942 | 1.2573793226602035 | 0.0 |
| 0.6b-like.eager.float32.disabled.b1 | logits | 7 | 0/7 | 0.48242632299661636 | 0.7987347210763783 | 0.0 |
| 0.6b-like.eager.float32.disabled.b1 | layers | 28 | 0/28 | 0.1397574543952942 | 1.2573793226602035 | 0.0 |
| 0.6b-like.eager.float32.disabled.b2 | logits | 7 | 0/7 | 0.48242632299661636 | 0.7987347210763783 | 0.0 |
| 0.6b-like.eager.float32.disabled.b2 | layers | 28 | 0/28 | 0.1397574543952942 | 1.2573793226602035 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b1 | logits | 7 | 0/7 | 0.5782675743103027 | 1.5560226003488722 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b1 | layers | 28 | 0/28 | 11.178619861602783 | 1.2430234157449833 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b2 | logits | 7 | 0/7 | 0.5782675743103027 | 1.5560226003488722 | 0.0 |
| 0.6b-like.eager.float32.last2-attention.b2 | layers | 28 | 0/28 | 11.178619861602783 | 1.2430234157449833 | 0.0 |
| 0.6b-like.eager.float32.all.b1 | logits | 7 | 0/7 | 0.41190293431282043 | 1.1314745755343985 | 0.0 |
| 0.6b-like.eager.float32.all.b1 | layers | 28 | 0/28 | 15.141417026519775 | 0.7163783197116894 | 0.0 |
| 0.6b-like.eager.float32.all.b2 | logits | 7 | 0/7 | 0.41190293431282043 | 1.1314745755343985 | 0.0 |
| 0.6b-like.eager.float32.all.b2 | layers | 28 | 0/28 | 15.141417026519775 | 0.7163783197116894 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b1 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b1 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b2 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.eager.bfloat16.plain.b2 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b1 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b1 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b2 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.eager.bfloat16.disabled.b2 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b1 | logits | 7 | 0/7 | 0.576171875 | 1.5526315789473684 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b1 | layers | 28 | 0/28 | 11.21875 | 1.2465277777777777 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b2 | logits | 7 | 0/7 | 0.576171875 | 1.5526315789473684 | 0.0 |
| 0.6b-like.eager.bfloat16.last2-attention.b2 | layers | 28 | 0/28 | 11.21875 | 1.2465277777777777 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b1 | logits | 7 | 0/7 | 0.412109375 | 1.1283422459893049 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b1 | layers | 28 | 0/28 | 15.625 | 0.7485029940119761 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b2 | logits | 7 | 0/7 | 0.412109375 | 1.1283422459893049 | 0.0 |
| 0.6b-like.eager.bfloat16.all.b2 | layers | 28 | 0/28 | 15.625 | 0.7485029940119761 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b1 | logits | 7 | 0/7 | 0.48242639750242233 | 0.7987347656095419 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b1 | layers | 28 | 0/28 | 0.139757439494133 | 1.2573791043119331 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b2 | logits | 7 | 0/7 | 0.48242639750242233 | 0.7987347656095419 | 0.0 |
| 0.6b-like.sdpa.float32.plain.b2 | layers | 28 | 0/28 | 0.139757439494133 | 1.2573791043119331 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b1 | logits | 7 | 0/7 | 0.48242639750242233 | 0.7987347656095419 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b1 | layers | 28 | 0/28 | 0.139757439494133 | 1.2573791043119331 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b2 | logits | 7 | 0/7 | 0.48242639750242233 | 0.7987347656095419 | 0.0 |
| 0.6b-like.sdpa.float32.disabled.b2 | layers | 28 | 0/28 | 0.139757439494133 | 1.2573791043119331 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b1 | logits | 7 | 0/7 | 0.5782676041126251 | 1.5560228053243432 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b1 | layers | 28 | 0/28 | 11.178619861602783 | 1.2430231521115884 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b2 | logits | 7 | 0/7 | 0.5782676041126251 | 1.5560228053243432 | 0.0 |
| 0.6b-like.sdpa.float32.last2-attention.b2 | layers | 28 | 0/28 | 11.178619861602783 | 1.2430231521115884 | 0.0 |
| 0.6b-like.sdpa.float32.all.b1 | logits | 7 | 0/7 | 0.41190318763256073 | 1.1314750861325473 | 0.0 |
| 0.6b-like.sdpa.float32.all.b1 | layers | 28 | 0/28 | 15.14139986038208 | 0.7163772489511242 | 0.0 |
| 0.6b-like.sdpa.float32.all.b2 | logits | 7 | 0/7 | 0.41190318763256073 | 1.1314750861325473 | 0.0 |
| 0.6b-like.sdpa.float32.all.b2 | layers | 28 | 0/28 | 15.14139986038208 | 0.7163772489511242 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b1 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b1 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b2 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.sdpa.bfloat16.plain.b2 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b1 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b1 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b2 | logits | 7 | 0/7 | 0.48095703125 | 0.7943548387096774 | 0.0 |
| 0.6b-like.sdpa.bfloat16.disabled.b2 | layers | 28 | 0/28 | 0.1396484375 | 1.2543859649122806 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b1 | logits | 7 | 0/7 | 0.576171875 | 1.5526315789473684 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b1 | layers | 28 | 0/28 | 11.21875 | 1.2465277777777777 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b2 | logits | 7 | 0/7 | 0.576171875 | 1.5526315789473684 | 0.0 |
| 0.6b-like.sdpa.bfloat16.last2-attention.b2 | layers | 28 | 0/28 | 11.21875 | 1.2465277777777777 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b1 | logits | 7 | 0/7 | 0.4091796875 | 1.1203208556149733 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b1 | layers | 28 | 0/28 | 15.40625 | 0.7336309523809523 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b2 | logits | 7 | 0/7 | 0.4091796875 | 1.1203208556149733 | 0.0 |
| 0.6b-like.sdpa.bfloat16.all.b2 | layers | 28 | 0/28 | 15.40625 | 0.7336309523809523 | 0.0 |
| 4b-like.eager.float32.plain.b1 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.eager.float32.plain.b1 | layers | 42 | 0/42 | 0.18429649621248245 | 1.3189471673718227 | 0.0 |
| 4b-like.eager.float32.plain.b2 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.eager.float32.plain.b2 | layers | 42 | 0/42 | 0.18429649621248245 | 1.3189471673718227 | 0.0 |
| 4b-like.eager.float32.disabled.b1 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.eager.float32.disabled.b1 | layers | 42 | 0/42 | 0.18429649621248245 | 1.3189471673718227 | 0.0 |
| 4b-like.eager.float32.disabled.b2 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.eager.float32.disabled.b2 | layers | 42 | 0/42 | 0.18429649621248245 | 1.3189471673718227 | 0.0 |
| 4b-like.eager.float32.last2-attention.b1 | logits | 7 | 0/7 | 0.5455130785703659 | 1.4288321531116612 | 0.0 |
| 4b-like.eager.float32.last2-attention.b1 | layers | 42 | 0/42 | 19.808046340942383 | 1.187442572355973 | 0.0 |
| 4b-like.eager.float32.last2-attention.b2 | logits | 7 | 0/7 | 0.5455130785703659 | 1.4288321531116612 | 0.0 |
| 4b-like.eager.float32.last2-attention.b2 | layers | 42 | 0/42 | 19.808046340942383 | 1.187442572355973 | 0.0 |
| 4b-like.eager.float32.all.b1 | logits | 7 | 0/7 | 0.6204647719860077 | 1.4718406159812012 | 0.0 |
| 4b-like.eager.float32.all.b1 | layers | 42 | 0/42 | 41.30894756317139 | 1.293818035815661 | 0.0 |
| 4b-like.eager.float32.all.b2 | logits | 7 | 0/7 | 0.6204647719860077 | 1.4718406159812012 | 0.0 |
| 4b-like.eager.float32.all.b2 | layers | 42 | 0/42 | 41.30894756317139 | 1.293818035815661 | 0.0 |
| 4b-like.eager.bfloat16.plain.b1 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.eager.bfloat16.plain.b1 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.eager.bfloat16.plain.b2 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.eager.bfloat16.plain.b2 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b1 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b1 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b2 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.eager.bfloat16.disabled.b2 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b1 | logits | 7 | 0/7 | 0.546875 | 1.450777202072539 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b1 | layers | 42 | 0/42 | 20.0625 | 1.1977611940298507 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b2 | logits | 7 | 0/7 | 0.546875 | 1.450777202072539 | 0.0 |
| 4b-like.eager.bfloat16.last2-attention.b2 | layers | 42 | 0/42 | 20.0625 | 1.1977611940298507 | 0.0 |
| 4b-like.eager.bfloat16.all.b1 | logits | 7 | 0/7 | 0.61328125 | 1.4604651162790698 | 0.0 |
| 4b-like.eager.bfloat16.all.b1 | layers | 42 | 0/42 | 41.53125 | 1.3080708661417322 | 0.0 |
| 4b-like.eager.bfloat16.all.b2 | logits | 7 | 0/7 | 0.61328125 | 1.4604651162790698 | 0.0 |
| 4b-like.eager.bfloat16.all.b2 | layers | 42 | 0/42 | 41.53125 | 1.3080708661417322 | 0.0 |
| 4b-like.sdpa.float32.plain.b1 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.sdpa.float32.plain.b1 | layers | 42 | 0/42 | 0.18429649248719215 | 1.3189470000553474 | 0.0 |
| 4b-like.sdpa.float32.plain.b2 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.sdpa.float32.plain.b2 | layers | 42 | 0/42 | 0.18429649248719215 | 1.3189470000553474 | 0.0 |
| 4b-like.sdpa.float32.disabled.b1 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.sdpa.float32.disabled.b1 | layers | 42 | 0/42 | 0.18429649248719215 | 1.3189470000553474 | 0.0 |
| 4b-like.sdpa.float32.disabled.b2 | logits | 7 | 0/7 | 0.4819526970386505 | 0.8272259726356793 | 0.0 |
| 4b-like.sdpa.float32.disabled.b2 | layers | 42 | 0/42 | 0.18429649248719215 | 1.3189470000553474 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b1 | logits | 7 | 0/7 | 0.5455130934715271 | 0.7582058202514769 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b1 | layers | 42 | 0/42 | 19.808043479919434 | 1.187442400844841 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b2 | logits | 7 | 0/7 | 0.5455130934715271 | 0.7582058202514769 | 0.0 |
| 4b-like.sdpa.float32.last2-attention.b2 | layers | 42 | 0/42 | 19.808043479919434 | 1.187442400844841 | 0.0 |
| 4b-like.sdpa.float32.all.b1 | logits | 7 | 0/7 | 0.6204646527767181 | 1.4718405413038644 | 0.0 |
| 4b-like.sdpa.float32.all.b1 | layers | 42 | 0/42 | 41.308918476104736 | 1.293816970210292 | 0.0 |
| 4b-like.sdpa.float32.all.b2 | logits | 7 | 0/7 | 0.6204646527767181 | 1.4718405413038644 | 0.0 |
| 4b-like.sdpa.float32.all.b2 | layers | 42 | 0/42 | 41.308918476104736 | 1.293816970210292 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b1 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b1 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b2 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.sdpa.bfloat16.plain.b2 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b1 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b1 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b2 | logits | 7 | 0/7 | 0.4794921875 | 0.8238255033557047 | 0.0 |
| 4b-like.sdpa.bfloat16.disabled.b2 | layers | 42 | 0/42 | 0.1845703125 | 1.3125 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b1 | logits | 7 | 0/7 | 0.546875 | 1.450777202072539 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b1 | layers | 42 | 0/42 | 20.0625 | 1.1977611940298507 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b2 | logits | 7 | 0/7 | 0.546875 | 1.450777202072539 | 0.0 |
| 4b-like.sdpa.bfloat16.last2-attention.b2 | layers | 42 | 0/42 | 20.0625 | 1.1977611940298507 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b1 | logits | 7 | 0/7 | 0.615234375 | 1.4651162790697674 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b1 | layers | 42 | 0/42 | 41.59375 | 1.2998046875 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b2 | logits | 7 | 0/7 | 0.615234375 | 1.4651162790697674 | 0.0 |
| 4b-like.sdpa.bfloat16.all.b2 | layers | 42 | 0/42 | 41.59375 | 1.2998046875 | 0.0 |


measurements: `{"residency": {"0.6b-like.eager.float32.plain": {"before_forward": 9186304, "after_forward": 9186304, "before_each_layer": [9209344, 9209344, 9209344, 9209344], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.disabled": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9364992, 9364992, 9364992, 9364992], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.last2-attention": {"before_forward": 9237504, "after_forward": 9237504, "before_each_layer": [9260544, 9260544, 9260544, 9260544], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.float32.all": {"before_forward": 9360384, "after_forward": 9360384, "before_each_layer": [9383424, 9383424, 9383424, 9383424], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.plain": {"before_forward": 8874496, "after_forward": 8874496, "before_each_layer": [8886784, 8886784, 8886784, 8886784], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.disabled": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9042432, 9042432, 9042432, 9042432], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.last2-attention": {"before_forward": 8916480, "after_forward": 8916480, "before_each_layer": [8928768, 8928768, 8928768, 8928768], "buffers": [0, 1, 0, 1]}, "0.6b-like.eager.bfloat16.all": {"before_forward": 9039360, "after_forward": 9039360, "before_each_layer": [9051648, 9051648, 9051648, 9051648], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.plain": {"before_forward": 9186304, "after_forward": 9186304, "before_each_layer": [9205760, 9205760, 9205760, 9205760], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.disabled": {"before_forward": 9341952, "after_forward": 9341952, "before_each_layer": [9361408, 9361408, 9361408, 9361408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.last2-attention": {"before_forward": 9237504, "after_forward": 9237504, "before_each_layer": [9256960, 9256960, 9256960, 9256960], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.float32.all": {"before_forward": 9360384, "after_forward": 9360384, "before_each_layer": [9379840, 9379840, 9379840, 9379840], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.plain": {"before_forward": 8874496, "after_forward": 8874496, "before_each_layer": [8885760, 8885760, 8885760, 8885760], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.disabled": {"before_forward": 9030144, "after_forward": 9030144, "before_each_layer": [9041408, 9041408, 9041408, 9041408], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.last2-attention": {"before_forward": 8916480, "after_forward": 8916480, "before_each_layer": [8927744, 8927744, 8927744, 8927744], "buffers": [0, 1, 0, 1]}, "0.6b-like.sdpa.bfloat16.all": {"before_forward": 9039360, "after_forward": 9039360, "before_each_layer": [9050624, 9050624, 9050624, 9050624], "buffers": [0, 1, 0, 1]}, "4b-like.eager.float32.plain": {"before_forward": 9692160, "after_forward": 9692160, "before_each_layer": [9713664, 9713664, 9713664, 9713664, 9713664, 9713664], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.disabled": {"before_forward": 9993216, "after_forward": 9993216, "before_each_layer": [10014720, 10014720, 10014720, 10014720, 10014720, 10014720], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.last2-attention": {"before_forward": 9747456, "after_forward": 9747456, "before_each_layer": [9768960, 9768960, 9768960, 9768960, 9768960, 9768960], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.float32.all": {"before_forward": 10011648, "after_forward": 10011648, "before_each_layer": [10033152, 10033152, 10033152, 10033152, 10033152, 10033152], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.bfloat16.plain": {"before_forward": 9129472, "after_forward": 9129472, "before_each_layer": [9140736, 9140736, 9140736, 9140736, 9140736, 9140736], "buffers": [0, 1, 0, 1, 0, 1]}, "4b-like.eager.bfloat16.disabled": {"before_forward": 9430528, "after_forward": 9430528, "before_each_layer": [9441792, 9441792, 9441792, 9441792, 9441792, 9441792], "buffers": [0, 1, 0, 1, 0,  ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **diagnostic**:
- CPU RSS 1641.578125 MiB, high-water mark 1641.578125 MiB (this process)
- PyTorch allocator, this process: peak allocated 9.7 MiB, peak reserved 22.0 MiB (excludes the CUDA context and non-PyTorch allocations)
- cudaMemGetInfo total - free: 857.3 MiB, device-wide (every process and context on the GPU, not process-specific)
- nvidia-smi raw: compute apps `322654, [N/A]`, memory.used `134` (unreliable under WSL2)
