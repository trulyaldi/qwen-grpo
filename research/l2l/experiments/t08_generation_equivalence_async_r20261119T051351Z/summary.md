# t08_generation_equivalence_async_r20261119T051351Z: generation_equivalence (correctness)

**Status: PASS** (execution PASS; hypothesis H04: SUPPORTED, all 1292 comparisons within predefined thresholds)


- commit `29852d11c2a37eeda94cd5ee53692fffd8d75d69` (tracked tree dirty: False), torch 2.14.1+cu126, transformers 4.57.6, peft 0.21.3, GPU NVIDIA GeForce RTX 3060
- config `research/l2l/configs/async.yaml` sha256 `f3d2268fb2f68f9b`; fields used: ['model.dtype', 'execution']; backend `l2l_async`
- input sha256 `cb7569af732767ae`, tolerances sha256 `54e48e883478a48e`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like.eager.float32.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | prefix_logits | 8 | 8/8 | 6.258487701416016e-07 | 1.8088050043605121e-06 | 3.4600123763084415e-06 |
| 0.6b-like.eager.float32.V64.p9 | prefix_probs | 8 | 8/8 | 1.1175870895385742e-08 | 5.145979870470541e-07 | 8.687069267034532e-06 |
| 0.6b-like.eager.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | prefix_logits | 8 | 8/8 | 2.2351741790771484e-07 | 6.541259386053093e-07 | 3.417039513587952e-06 |
| 0.6b-like.eager.float32.V64.p17 | prefix_probs | 8 | 8/8 | 5.587935447692871e-09 | 2.625345593930621e-07 | 8.513828366994857e-06 |
| 0.6b-like.eager.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00579833984375 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.013999434001743794 |
| 0.6b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.0050048828125 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.013938802387565374 |
| 0.6b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 8/8 | 4.172325134277344e-07 | 1.1771278014380466e-06 | 3.544496297836304e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 8/8 | 9.313225746154785e-09 | 4.2852943268702524e-07 | 8.69319587945938e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 8/8 | 1.5925616025924683e-07 | 4.89264069929947e-07 | 3.2550144195556643e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 8/8 | 3.725290298461914e-09 | 1.7372815140477886e-07 | 8.577286452054977e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.0009765625 | 0.0029940119760479044 | 0.005096435546875 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 1.2843869626522064e-05 | 0.0005996516619973466 | 0.013922274857759476 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.005035400390625 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.013938802387565374 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p9 | prefix_logits | 8 | 8/8 | 5.21540641784668e-07 | 1.0889688143600015e-06 | 4.7893074154853824e-06 |
| 4b-like.eager.float32.V64.p9 | prefix_probs | 8 | 8/8 | 9.313225746154785e-09 | 3.903956730417499e-07 | 9.542345255613327e-06 |
| 4b-like.eager.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p17 | prefix_logits | 8 | 8/8 | 9.5367431640625e-07 | 2.880778616844561e-06 | 3.3104741573333744e-06 |
| 4b-like.eager.float32.V64.p17 | prefix_probs | 8 | 8/8 | 1.4901161193847656e-08 | 7.221979734222118e-07 | 8.253227919340135e-06 |
| 4b-like.eager.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.006103515625 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.015003155451267958 |
| 4b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00592041015625 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.015059542283415795 |
| 4b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 8/8 | 5.960464477539062e-07 | 1.6131278928928246e-06 | 3.6949732899665834e-06 |
| 4b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 8/8 | 1.4901161193847656e-08 | 6.929177397508372e-07 | 8.601979911327362e-06 |
| 4b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 8/8 | 7.897615432739258e-07 | 2.014367533442682e-06 | 3.920642733573914e-06 |
| 4b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 8/8 | 1.0244548320770264e-08 | 4.710768654408333e-07 | 8.698833733797074e-06 |
| 4b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.006072998046875 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.014979366026818753 |
| 4b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.005889892578125 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.015027657989412547 |
| 4b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_logits | 8 | 8/8 | 4.291534423828125e-06 | 1.2818695012694424e-06 | 3.347871541976929e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_probs | 8 | 8/8 | 1.1920928955078125e-07 | 2.0146396307064824e-06 | 2.3668608069419863e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_logits | 8 | 8/8 | 3.2782554626464844e-06 | 1.607919147978261e-06 | 2.0388185977935792e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_probs | 8 | 8/8 | 8.940696716308594e-08 | 5.243013684265716e-07 | 6.821036338806153e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | greedy_sequences | 4 | 4/4 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | cached_logits | 32 | 32/32 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00799560546875 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 6.871595496704686e-06 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |


measurements: `{"weight_loads": {"0.6b-like.eager.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.float32.V64.p9.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.float32.V64.p17.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p9.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p17.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p9.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p17.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p9.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p17.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "4b-like.eager.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.float32.V64.p9.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.float32.V64.p17.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p9.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p17.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p9.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p17.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p9.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p17.kv_cuda": {"loads": 48, "steps": 8, "expected": 48}, "0.6b-like.sdpa.float32.V64.eos.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.eos.p9.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.eos.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.eos.p17.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V151936.chat.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V151936.chat.kv_cuda": {"loads": 32, "steps": 8, "expected": 32}}, "near_ties": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 5, "0.6b-like.eager.bfloat16.V64.p17": 7, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 5, "0.6b-like.sdpa.bfloat16.V64.p17": 7, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 0, "4b-like.eager.bfloat16.V64.p9": 10, "4b-like.eager.bfloat16.V64.p17": 7, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 0, "4b-like.sdpa.bfloat16.V64.p9": 8, "4b-like.sdpa.bfloat16.V64.p17": 9, "0.6b-like.sdpa.float32.V64.eos.p9": 0, "0.6b-like.sdpa.float32.V64.eos.p17": 0, "0.6b-like.sdpa.bfloat16.V151936.chat": 12}, "greedy_steps": {"0.6b-like.eager.f ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **assignment**:
- CPU RSS 1961.41015625 MiB, high-water mark 1961.41015625 MiB (this process)
- PyTorch allocator, this process: peak allocated 204.8 MiB, peak reserved 242.0 MiB (excludes the CUDA context and non-PyTorch allocations)
- cudaMemGetInfo total - free: 420.1 MiB, device-wide (every process and context on the GPU, not process-specific)
- nvidia-smi raw: compute apps `9671, 400`, memory.used `421` (unreliable under WSL2)
