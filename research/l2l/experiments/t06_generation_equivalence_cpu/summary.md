# t06_generation_equivalence_cpu: generation_equivalence (correctness)

**Status: PASS** (execution PASS; hypothesis H04: SUPPORTED, all 950 comparisons within predefined thresholds)


- commit `04e56b95bb3e64721c4d894a664cd2caa0c160c6` (tracked tree dirty: False), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/store_cpu.yaml` sha256 `cd9f52c56460a60b`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `5948041fac234ce6`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like.eager.float32.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | prefix_logits | 8 | 8/8 | 4.6566128730773926e-07 | 1.4760679552401405e-06 | 3.1547415256500247e-06 |
| 0.6b-like.eager.float32.V64.p9 | prefix_probs | 8 | 8/8 | 7.450580596923828e-09 | 3.5579987146729644e-07 | 8.376147598028183e-06 |
| 0.6b-like.eager.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | prefix_logits | 8 | 8/8 | 1.4901161193847656e-07 | 4.360838830025317e-07 | 3.4170401096343995e-06 |
| 0.6b-like.eager.float32.V64.p17 | prefix_probs | 8 | 8/8 | 3.725290298461914e-09 | 1.7717776723766927e-07 | 8.410288393497467e-06 |
| 0.6b-like.eager.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00579833984375 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.013999434001743794 |
| 0.6b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.0050048828125 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.013938802387565374 |
| 0.6b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 8/8 | 3.2782554626464844e-07 | 9.445670521101328e-07 | 3.47064346075058e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 8/8 | 5.587935447692871e-09 | 2.57299037658716e-07 | 8.687067776918411e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 8/8 | 1.5366822481155396e-07 | 4.4971146512397644e-07 | 3.4170404076576235e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 8/8 | 5.587935447692871e-09 | 2.628782752661511e-07 | 8.502696454524994e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.001953125 | 0.0055248618784530384 | 0.005523681640625 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 3.997981548309326e-05 | 0.0018256859485039864 | 0.014234036300331354 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.005035400390625 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.013938802387565374 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p9 | prefix_logits | 8 | 8/8 | 2.682209014892578e-07 | 6.244169506723097e-07 | 4.295541644096375e-06 |
| 4b-like.eager.float32.V64.p9 | prefix_probs | 8 | 8/8 | 7.450580596923828e-09 | 3.1231648966259745e-07 | 9.542346745729448e-06 |
| 4b-like.eager.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.float32.V64.p17 | prefix_logits | 8 | 8/8 | 5.925539880990982e-07 | 1.7899372314236108e-06 | 3.3104735612869265e-06 |
| 4b-like.eager.float32.V64.p17 | prefix_probs | 8 | 8/8 | 9.313225746154785e-09 | 4.5137369264123665e-07 | 8.253228664398194e-06 |
| 4b-like.eager.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.006103515625 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.015003155451267958 |
| 4b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00592041015625 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.015059542283415795 |
| 4b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 8/8 | 7.152557373046875e-07 | 1.9357528469480073e-06 | 3.6949744820594792e-06 |
| 4b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 8/8 | 1.4901161193847656e-08 | 6.929176197171095e-07 | 8.601981401443482e-06 |
| 4b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 8/8 | 3.241002559661865e-07 | 9.790146962034124e-07 | 3.3104738593101505e-06 |
| 4b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 8/8 | 7.450580596923828e-09 | 3.610990193092284e-07 | 8.253227174282074e-06 |
| 4b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 8/8 | 0.001953125 | 0.005405405405405406 | 0.005645751953125 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 3.930181264877319e-05 | 0.0018795388413382387 | 0.013591726683080197 |
| 4b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00592041015625 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 0.015056720096617937 |
| 4b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_logits | 8 | 8/8 | 5.7220458984375e-06 | 4.150160771175957e-06 | 1.3787528276443483e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_probs | 8 | 8/8 | 3.203749656677246e-07 | 5.414323896061854e-06 | 2.3668695986270905e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_logits | 8 | 8/8 | 2.0265579223632812e-06 | 1.1458666701896449e-06 | 1.7685809135437015e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_probs | 8 | 8/8 | 9.685754776000977e-08 | 1.0962070981433371e-06 | 3.5342791676521304e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | greedy_sequences | 2 | 2/2 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | cached_logits | 16 | 16/16 | 0.0 | 0.0 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_logits | 8 | 8/8 | 0.0 | 0.0 | 0.00799560546875 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_probs | 8 | 8/8 | 0.0 | 0.0 | 6.871570667499328e-06 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | sampled_probs | 16 | 16/16 | 0.0 | 0.0 | 0.0 |


measurements: `{"weight_loads": {"0.6b-like.eager.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "4b-like.eager.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "0.6b-like.sdpa.float32.V64.eos.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.eos.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V151936.chat.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}}, "near_ties": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 5, "0.6b-like.eager.bfloat16.V64.p17": 7, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 5, "0.6b-like.sdpa.bfloat16.V64.p17": 7, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 0, "4b-like.eager.bfloat16.V64.p9": 10, "4b-like.eager.bfloat16.V64.p17": 7, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 0, "4b-like.sdpa.bfloat16.V64.p9": 8, "4b-like.sdpa.bfloat16.V64.p17": 9, "0.6b-like.sdpa.float32.V64.eos.p9": 0, "0.6b-like.sdpa.float32.V64.eos.p17": 0, "0.6b-like.sdpa.bfloat16.V151936.chat": 12}, "greedy_steps": {"0.6b-like.eager.float32.V64.p9": 8, "0.6b-like.eager.float32.V64.p17": 8, "0.6b-like.eager.bfloat16.V64.p9": 8, "0.6b-like.eager.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.p9": 8, "0.6b-like.sdpa.float32.V64.p17": 8, "0.6b-like.sdpa.bfloat16.V64.p9": 8, "0.6b-like.sdpa.bfloat16.V64.p17": 8, "4b-like.eager.float32.V64.p9": 8, "4b-like.eager.float32.V64.p17": 8, "4b-like.eager.bfloat16.V64.p9": 8, "4b-like.eager.bfloat16.V64.p17": 8, "4b-like.sdpa.float32.V64.p9": 8, "4b-like.sdpa.float32.V64.p17": 8, "4b-like.sdpa.bfloat16.V64.p9": 8, "4b-like.sdpa.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.eos.p9": 8, "0.6b-like.sdpa.float32.V64.eos.p17": 8, "0.6b-like.sdpa.bfloat16.V151936.chat": 8}, "eos_padded_tail_rows": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 0, "0.6b-like.eager.bfloat16.V64.p17": 0, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 0, "0.6b-like.sdpa.bfloat16.V64.p17": 0, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 0, "4b-like.eager.bfloat16.V64.p9": 0, "4b-like.eager.bfloat16.V64.p17": 0, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 0, "4b-like.sdpa.bfloat16.V64.p9": 0, "4b-like.sdpa.bfloat16.V64.p17": 0, "0.6b-like.sdpa.float32.V64.eos.p9": 2, "0.6b-like.sdpa.float32.V64.eos.p17": 2, "0.6b-like.sdpa.bfloat16.V151936.chat": 0}, "same_seed_samples_identical": {"0.6b-like.eager.float32.V64.p9": true, "0.6b-like.eager.float32.V64.p17": true, "0.6b-like.eager.bfloat16.V64.p9": true, "0.6b-like.eager.bfloat16.V64.p17": true, ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 975.50390625 MiB, high-water mark 975.50390625 MiB (this process)
