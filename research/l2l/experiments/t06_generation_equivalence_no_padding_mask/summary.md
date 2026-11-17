**FAULT INJECTION RUN** (no_padding_mask): a FAIL is the expected outcome.

# t06_generation_equivalence_no_padding_mask: generation_equivalence (correctness)

**Status: FAIL** (execution PASS; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.0: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.2: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.3: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.4: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.5: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.6: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.prefix.7: greedy token differs without a near-tie
- greedy_sequences.0.6b-like.eager.float32.V64.p9.kv_cpu.resident: max_abs 4.700e+01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.0: max_abs 3.475e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.1: max_abs 3.754e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.2: max_abs 2.904e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.3: max_abs 3.592e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.4: max_abs 3.759e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.5: max_abs 2.888e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.6: max_abs 2.905e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.resident.7: max_abs 3.247e-01 > bound 0.000e+00
- greedy_sequences.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed: max_abs 4.700e+01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.0: max_abs 3.475e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.1: max_abs 3.754e-01 > bound 0.000e+00
- cached_logits.0.6b-like.eager.float32.V64.p9.kv_cpu.streamed.2: max_abs 2.904e-01 > bound 0.000e+00
- ... 983 more

- commit `04e56b95bb3e64721c4d894a664cd2caa0c160c6` (tracked tree dirty: False), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/store_cpu.yaml` sha256 `cd9f52c56460a60b`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `5948041fac234ce6`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| None | greedy_sequences | 125 | 0/125 | n/a | None | None |
| 0.6b-like.eager.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | cached_logits | 16 | 0/16 | 0.3758818209171295 | 1.0604653990241952 | 0.0 |
| 0.6b-like.eager.float32.V64.p9 | prefix_logits | 8 | 0/8 | 0.37621408700942993 | 1.174221586891365 | 3.203944563865662e-06 |
| 0.6b-like.eager.float32.V64.p9 | prefix_probs | 8 | 0/8 | 0.005979050882160664 | 0.2906539286351995 | 8.22841227054596e-06 |
| 0.6b-like.eager.float32.V64.p9 | sampled_probs | 16 | 0/16 | 0.008325549773871899 | 0.36625667073723633 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 41.0 | 0.6612903225806451 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | cached_logits | 16 | 0/16 | 0.4940015375614166 | 1.4886235652212256 | 0.0 |
| 0.6b-like.eager.float32.V64.p17 | prefix_logits | 8 | 0/8 | 0.4930039271712303 | 1.4790486316280378 | 3.3332502841949467e-06 |
| 0.6b-like.eager.float32.V64.p17 | prefix_probs | 8 | 0/8 | 0.006197500973939896 | 0.29275095046496014 | 8.46794992685318e-06 |
| 0.6b-like.eager.float32.V64.p17 | sampled_probs | 16 | 0/16 | 0.008117081597447395 | 0.3700760520691929 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | cached_logits | 16 | 0/16 | 0.380859375 | 1.095505617977528 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 0/8 | 0.376953125 | 1.1840490797546013 | 0.004974365234375 |
| 0.6b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.005953023210167885 | 0.28922682442475084 | 0.013378652185201645 |
| 0.6b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 0/16 | 0.008241347037255764 | 0.36226565113650266 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 41.0 | 0.6612903225806451 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | cached_logits | 16 | 0/16 | 0.494140625 | 1.488235294117647 | 0.0 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 0/8 | 0.492431640625 | 1.4830882352941177 | 0.00518798828125 |
| 0.6b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.006174112670123577 | 0.29096911183489793 | 0.013792437314987183 |
| 0.6b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 0/16 | 0.008040354587137699 | 0.3660474818504025 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | cached_logits | 16 | 0/16 | 0.3758818209171295 | 1.0604655773531488 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 0/8 | 0.3762141168117523 | 1.17422222602561 | 3.2039430737495426e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 0/8 | 0.005979049019515514 | 0.29065381177036664 | 8.22841301560402e-06 |
| 0.6b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 0/16 | 0.008325546979904175 | 0.36625651781378715 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 41.0 | 0.6612903225806451 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | cached_logits | 16 | 0/16 | 0.4940015375614166 | 1.4886235652212256 | 0.0 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 0/8 | 0.49300384521484375 | 1.4790482535122387 | 3.3332505822181703e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 0/8 | 0.006197500973939896 | 0.29275095046496014 | 8.46794992685318e-06 |
| 0.6b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 0/16 | 0.00811708439141512 | 0.37007617945248117 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | cached_logits | 16 | 0/16 | 0.37890625 | 1.0777777777777777 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 0/8 | 0.376953125 | 1.1840490797546013 | 0.004974365234375 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.00593909714370966 | 0.28856022990492575 | 0.013378188479691744 |
| 0.6b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 0/16 | 0.008293932303786278 | 0.36453527139953784 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 41.0 | 0.6612903225806451 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | cached_logits | 16 | 0/16 | 0.49609375 | 1.4941176470588236 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 0/8 | 0.492919921875 | 1.4933431952662721 | 0.005157470703125 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.006149770691990852 | 0.290403515768412 | 0.013764815963804721 |
| 0.6b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 0/16 | 0.008077754639089108 | 0.3677602715145167 | 0.0 |
| 4b-like.eager.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 55.0 | 0.8870967741935484 | 0.0 |
| 4b-like.eager.float32.V64.p9 | cached_logits | 16 | 0/16 | 0.5751247704029083 | 1.200851424109591 | 0.0 |
| 4b-like.eager.float32.V64.p9 | prefix_logits | 8 | 0/8 | 0.5954950302839279 | 1.6116352879735285 | 3.6949738860130313e-06 |
| 4b-like.eager.float32.V64.p9 | prefix_probs | 8 | 0/8 | 0.008776119910180569 | 0.41626374140927247 | 8.433230221271516e-06 |
| 4b-like.eager.float32.V64.p9 | sampled_probs | 16 | 0/16 | 0.013104002922773361 | 0.5735614570139406 | 0.0 |
| 4b-like.eager.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 60.0 | 0.967741935483871 | 0.0 |
| 4b-like.eager.float32.V64.p17 | cached_logits | 16 | 0/16 | 0.5924208760261536 | 1.557487428980734 | 0.0 |
| 4b-like.eager.float32.V64.p17 | prefix_logits | 8 | 0/8 | 0.5897636115550995 | 1.8625204565707052 | 3.1664812564849855e-06 |
| 4b-like.eager.float32.V64.p17 | prefix_probs | 8 | 0/8 | 0.009430077858269215 | 0.3965017128863784 | 9.513278305530548e-06 |
| 4b-like.eager.float32.V64.p17 | sampled_probs | 16 | 0/16 | 0.01597227342426777 | 0.5935150048733706 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 55.0 | 0.8870967741935484 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | cached_logits | 16 | 0/16 | 0.5810546875 | 1.5824468085106382 | 0.0 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_logits | 8 | 0/8 | 0.603515625 | 1.6436170212765957 | 0.0057373046875 |
| 4b-like.eager.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.009407654404640198 | 0.40880137338333244 | 0.014958304353058338 |
| 4b-like.eager.bfloat16.V64.p9 | sampled_probs | 16 | 0/16 | 0.013092683628201485 | 0.572303795448328 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 38.0 | 0.6129032258064516 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | cached_logits | 16 | 0/16 | 0.5107421875 | 1.547337278106509 | 0.0 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_logits | 8 | 0/8 | 0.5048828125 | 1.2548543689320388 | 0.00628662109375 |
| 4b-like.eager.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.009393316693603992 | 0.3951068861204151 | 0.015453174989670516 |
| 4b-like.eager.bfloat16.V64.p17 | sampled_probs | 16 | 0/16 | 0.015917710028588772 | 0.5917860365661661 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | greedy_sequences | 2 | 0/2 | 55.0 | 0.8870967741935484 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | cached_logits | 16 | 0/16 | 0.5751248300075531 | 1.2008514738380285 | 0.0 |
| 4b-like.sdpa.float32.V64.p9 | prefix_logits | 8 | 0/8 | 0.5954953283071518 | 1.611635834559915 | 3.6949744820594792e-06 |
| 4b-like.sdpa.float32.V64.p9 | prefix_probs | 8 | 0/8 | 0.008776118978857994 | 0.4162637340113504 | 8.433229476213456e-06 |
| 4b-like.sdpa.float32.V64.p9 | sampled_probs | 16 | 0/16 | 0.013104001991450787 | 0.573561509772503 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | greedy_sequences | 2 | 0/2 | 60.0 | 0.967741935483871 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | cached_logits | 16 | 0/16 | 0.5924209952354431 | 1.5574878644152257 | 0.0 |
| 4b-like.sdpa.float32.V64.p17 | prefix_logits | 8 | 0/8 | 0.589764416217804 | 1.8625226471653984 | 3.1664818525314335e-06 |
| 4b-like.sdpa.float32.V64.p17 | prefix_probs | 8 | 0/8 | 0.009430075995624065 | 0.39650163456868015 | 9.513278305530548e-06 |
| 4b-like.sdpa.float32.V64.p17 | sampled_probs | 16 | 0/16 | 0.015972268767654896 | 0.5935149550768717 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | greedy_sequences | 2 | 0/2 | 55.0 | 0.8870967741935484 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | cached_logits | 16 | 0/16 | 0.677734375 | 1.8756756756756756 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_logits | 8 | 0/8 | 0.6005859375 | 1.6443850267379678 | 0.005706787109375 |
| 4b-like.sdpa.bfloat16.V64.p9 | prefix_probs | 8 | 8/8 | 0.009493310935795307 | 0.41009395252768227 | 0.015046922955662012 |
| 4b-like.sdpa.bfloat16.V64.p9 | sampled_probs | 16 | 0/16 | 0.013088107109069824 | 0.5751304917076348 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | greedy_sequences | 2 | 0/2 | 38.0 | 0.6129032258064516 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | cached_logits | 16 | 0/16 | 0.513671875 | 1.5654761904761905 | 0.0 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_logits | 8 | 0/8 | 0.5078125 | 1.2560386473429952 | 0.006317138671875 |
| 4b-like.sdpa.bfloat16.V64.p17 | prefix_probs | 8 | 8/8 | 0.009445028379559517 | 0.3965942283004397 | 0.015479974262416363 |
| 4b-like.sdpa.bfloat16.V64.p17 | sampled_probs | 16 | 0/16 | 0.015915678814053535 | 0.5916754884820605 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | greedy_sequences | 2 | 0/2 | 47.0 | 0.7580645161290323 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | cached_logits | 16 | 0/16 | 4.913219332695007 | 2.74441085694123 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_logits | 8 | 0/8 | 4.913219928741455 | 2.7444106416482175 | 1.790264129638672e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | prefix_probs | 8 | 0/8 | 0.19176842272281647 | 2.864029936941478 | 2.6783019304275515e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p9 | sampled_probs | 16 | 0/16 | 0.7241639670683071 | 9.1466182400567 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | greedy_sequences | 2 | 0/2 | 46.0 | 0.7419354838709677 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | cached_logits | 16 | 0/16 | 4.224479049444199 | 2.3278477305397014 | 0.0 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_logits | 8 | 0/8 | 4.031782507896423 | 2.6656359911237533 | 1.5125030279159547e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | prefix_probs | 8 | 0/8 | 0.2257335465401411 | 5.374890552761885 | 1.6799117624759674e-05 |
| 0.6b-like.sdpa.float32.V64.eos.p17 | sampled_probs | 16 | 0/16 | 0.21018890663981438 | 0.9807770008963981 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | greedy_sequences | 2 | 0/2 | 81497.0 | 0.5373381332911359 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | cached_logits | 16 | 0/16 | 0.796875 | 1.5223880597014925 | 0.0 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_logits | 8 | 0/8 | 0.888671875 | 1.5798611111111112 | 0.0087890625 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | prefix_probs | 8 | 8/8 | 6.037492767063668e-06 | 0.544878548674513 | 7.202284450613661e-06 |
| 0.6b-like.sdpa.bfloat16.V151936.chat | sampled_probs | 16 | 0/16 | 5.431337285699556e-06 | 0.4883725447125523 | 0.0 |


measurements: `{"weight_loads": {"0.6b-like.eager.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "4b-like.eager.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.eager.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.float32.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p9.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "4b-like.sdpa.bfloat16.V64.p17.kv_cpu": {"loads": 48, "steps": 8, "expected": 48}, "0.6b-like.sdpa.float32.V64.eos.p9.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.float32.V64.eos.p17.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}, "0.6b-like.sdpa.bfloat16.V151936.chat.kv_cpu": {"loads": 32, "steps": 8, "expected": 32}}, "near_ties": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 7, "0.6b-like.eager.bfloat16.V64.p17": 6, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 7, "0.6b-like.sdpa.bfloat16.V64.p17": 6, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 0, "4b-like.eager.bfloat16.V64.p9": 5, "4b-like.eager.bfloat16.V64.p17": 3, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 0, "4b-like.sdpa.bfloat16.V64.p9": 3, "4b-like.sdpa.bfloat16.V64.p17": 5, "0.6b-like.sdpa.float32.V64.eos.p9": 0, "0.6b-like.sdpa.float32.V64.eos.p17": 0, "0.6b-like.sdpa.bfloat16.V151936.chat": 10}, "greedy_steps": {"0.6b-like.eager.float32.V64.p9": 8, "0.6b-like.eager.float32.V64.p17": 8, "0.6b-like.eager.bfloat16.V64.p9": 8, "0.6b-like.eager.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.p9": 8, "0.6b-like.sdpa.float32.V64.p17": 8, "0.6b-like.sdpa.bfloat16.V64.p9": 8, "0.6b-like.sdpa.bfloat16.V64.p17": 8, "4b-like.eager.float32.V64.p9": 8, "4b-like.eager.float32.V64.p17": 8, "4b-like.eager.bfloat16.V64.p9": 8, "4b-like.eager.bfloat16.V64.p17": 8, "4b-like.sdpa.float32.V64.p9": 8, "4b-like.sdpa.float32.V64.p17": 8, "4b-like.sdpa.bfloat16.V64.p9": 8, "4b-like.sdpa.bfloat16.V64.p17": 8, "0.6b-like.sdpa.float32.V64.eos.p9": 8, "0.6b-like.sdpa.float32.V64.eos.p17": 8, "0.6b-like.sdpa.bfloat16.V151936.chat": 8}, "eos_padded_tail_rows": {"0.6b-like.eager.float32.V64.p9": 0, "0.6b-like.eager.float32.V64.p17": 0, "0.6b-like.eager.bfloat16.V64.p9": 0, "0.6b-like.eager.bfloat16.V64.p17": 0, "0.6b-like.sdpa.float32.V64.p9": 0, "0.6b-like.sdpa.float32.V64.p17": 0, "0.6b-like.sdpa.bfloat16.V64.p9": 0, "0.6b-like.sdpa.bfloat16.V64.p17": 0, "4b-like.eager.float32.V64.p9": 0, "4b-like.eager.float32.V64.p17": 1, "4b-like.eager.bfloat16.V64.p9": 0, "4b-like.eager.bfloat16.V64.p17": 0, "4b-like.sdpa.float32.V64.p9": 0, "4b-like.sdpa.float32.V64.p17": 1, "4b-like.sdpa.bfloat16.V64.p9": 0, "4b-like.sdpa.bfloat16.V64.p17": 0, "0.6b-like.sdpa.float32.V64.eos.p9": 2, "0.6b-like.sdpa.float32.V64.eos.p17": 1, "0.6b-like.sdpa.bfloat16.V151936.chat": 0}, "same_seed_samples_identical": {"0.6b-like.eager.float32.V64.p9": false, "0.6b-like.eager.float32.V64.p17": false, "0.6b-like.eager.bfloat16.V64.p9": false, "0.6b-like.eager.bfloat16.V64.p17": fal ...` (truncated; see result.json)``

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 975.39453125 MiB, high-water mark 975.39453125 MiB (this process)
