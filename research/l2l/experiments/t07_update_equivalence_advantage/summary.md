**FAULT INJECTION RUN** (advantage=2): a FAIL is the expected outcome.

# t07_update_equivalence_advantage: update_equivalence (correctness)

**Status: FAIL** (execution PASS; hypothesis -: NOT_EVALUATED, no hypothesis declared for this run)

- loss.0.6b-like.float32.mball.fresh.it0: max_abs 4.023e-01 > bound 0.000e+00
- stats.0.6b-like.float32.mball.fresh.it0.grad_norm: max_abs 7.437e-02 > bound 0.000e+00
- stats.0.6b-like.float32.mball.fresh.it0.update_norm: max_abs 2.513e-04 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight: max_abs 8.417e-03 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight: max_abs 2.738e-03 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.v_proj.lora_A.default.weight: max_abs 2.238e-02 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.v_proj.lora_B.default.weight: max_abs 1.021e-02 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.o_proj.lora_A.default.weight: max_abs 5.226e-02 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.o_proj.lora_B.default.weight: max_abs 1.527e-02 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.3.self_attn.q_proj.lora_A.default.weight: max_abs 7.431e-04 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.3.self_attn.q_proj.lora_B.default.weight: max_abs 2.383e-04 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.3.self_attn.v_proj.lora_A.default.weight: max_abs 4.142e-02 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.3.self_attn.v_proj.lora_B.default.weight: max_abs 8.931e-03 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.3.self_attn.o_proj.lora_A.default.weight: max_abs 3.234e-02 > bound 0.000e+00
- grads.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.3.self_attn.o_proj.lora_B.default.weight: max_abs 9.930e-03 > bound 0.000e+00
- updates.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.q_proj.lora_A.default.weight: max_abs 1.000e-01 > bound 0.000e+00
- updates.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight: max_abs 1.000e-01 > bound 0.000e+00
- updates.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.v_proj.lora_A.default.weight: max_abs 1.000e-01 > bound 0.000e+00
- updates.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.v_proj.lora_B.default.weight: max_abs 1.000e-01 > bound 0.000e+00
- updates.0.6b-like.float32.mball.fresh.it0.base_model.model.model.layers.2.self_attn.o_proj.lora_A.default.weight: max_abs 1.000e-01 > bound 0.000e+00
- ... 5196 more

- commit `3cbd70c20ff04894d6e88479c7d7142e50b1517b` (tracked tree dirty: False), torch 2.14.1, transformers 4.57.6, peft 0.21.3, GPU None
- config `research/l2l/configs/store_cpu.yaml` sha256 `cd9f52c56460a60b`; fields used: ['model.dtype', 'execution']; backend `l2l_sync`
- input sha256 `cb7569af732767ae`, tolerances sha256 `35d003c20a022f64`

| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |
|---|---|---|---|---|---|---|
| 0.6b-like.float32.mball.fresh | logp | 3 | 1/3 | 0.6046421527862549 | 0.13337754966158782 | 0.0 |
| 0.6b-like.float32.mball.fresh | loss | 3 | 0/3 | 0.4893858931027353 | 64.25036716523374 | 0.0 |
| 0.6b-like.float32.mball.fresh | stats | 18 | 4/18 | 0.23684406280517578 | 0.19833287114041462 | 0.0 |
| 0.6b-like.float32.mball.fresh | grads | 36 | 0/36 | 0.05226000025868416 | 1.1433994943377894 | 0.0 |
| 0.6b-like.float32.mball.fresh | updates | 36 | 0/36 | 0.09999996051192284 | 1.9999996274708924 | 0.0 |
| 0.6b-like.float32.mball.fresh | optimizer | 108 | 36/108 | 0.005935894325375557 | 1.1032031937690727 | 0.0 |
| 0.6b-like.float32.mball.warm | logp | 3 | 1/3 | 0.5205411911010742 | 0.11620302717543107 | 0.0 |
| 0.6b-like.float32.mball.warm | loss | 3 | 0/3 | 0.45513832569122314 | 10.784453684192686 | 0.0 |
| 0.6b-like.float32.mball.warm | stats | 18 | 4/18 | 0.3429373502731323 | 0.28734091340859075 | 0.0 |
| 0.6b-like.float32.mball.warm | grads | 36 | 0/36 | 0.09955507889389992 | 1.2663636675526058 | 0.0 |
| 0.6b-like.float32.mball.warm | updates | 36 | 0/36 | 0.1246623694896698 | 1.1565462276336018 | 0.0 |
| 0.6b-like.float32.mball.warm | optimizer | 108 | 36/108 | 0.008272136910818517 | 0.6787396711992704 | 0.0 |
| 0.6b-like.float32.mb1.fresh | logp | 3 | 1/3 | 0.6046421527862549 | 0.13337754966158782 | 0.0 |
| 0.6b-like.float32.mb1.fresh | loss | 3 | 0/3 | 0.48938601464033127 | 64.24989998288216 | 0.0 |
| 0.6b-like.float32.mb1.fresh | stats | 18 | 4/18 | 0.23684406280517578 | 0.19833287114041462 | 0.0 |
| 0.6b-like.float32.mb1.fresh | grads | 36 | 0/36 | 0.052260007709264755 | 1.1433996573494627 | 0.0 |
| 0.6b-like.float32.mb1.fresh | updates | 36 | 0/36 | 0.09999996051192284 | 1.9999996274708924 | 0.0 |
| 0.6b-like.float32.mb1.fresh | optimizer | 108 | 36/108 | 0.005935892928391695 | 1.1032030296117232 | 0.0 |
| 0.6b-like.float32.mb1.warm | logp | 3 | 1/3 | 0.5205411911010742 | 0.11620302717543107 | 0.0 |
| 0.6b-like.float32.mb1.warm | loss | 3 | 0/3 | 0.4551383778452873 | 10.784483478482544 | 0.0 |
| 0.6b-like.float32.mb1.warm | stats | 18 | 4/18 | 0.3429373502731323 | 0.28734091340859075 | 0.0 |
| 0.6b-like.float32.mb1.warm | grads | 36 | 0/36 | 0.09955506213009357 | 1.2663635743303212 | 0.0 |
| 0.6b-like.float32.mb1.warm | updates | 36 | 0/36 | 0.1246623694896698 | 1.1565462276336018 | 0.0 |
| 0.6b-like.float32.mb1.warm | optimizer | 108 | 36/108 | 0.008272135979495943 | 0.6787396466497628 | 0.0 |
| 0.6b-like.float32.mb2.fresh | logp | 3 | 1/3 | 0.6046421527862549 | 0.13337754966158782 | 0.0 |
| 0.6b-like.float32.mb2.fresh | loss | 3 | 0/3 | 0.4893858730792999 | 64.24994424468173 | 0.0 |
| 0.6b-like.float32.mb2.fresh | stats | 18 | 4/18 | 0.23684406280517578 | 0.19833287114041462 | 0.0 |
| 0.6b-like.float32.mb2.fresh | grads | 36 | 0/36 | 0.052260007709264755 | 1.1433995641557246 | 0.0 |
| 0.6b-like.float32.mb2.fresh | updates | 36 | 0/36 | 0.09999995678663254 | 1.999999552965071 | 0.0 |
| 0.6b-like.float32.mb2.fresh | optimizer | 108 | 36/108 | 0.005935893394052982 | 1.103203116156233 | 0.0 |
| 0.6b-like.float32.mb2.warm | logp | 3 | 1/3 | 0.5205411911010742 | 0.11620302717543107 | 0.0 |
| 0.6b-like.float32.mb2.warm | loss | 3 | 0/3 | 0.45513834059238434 | 10.78446546065698 | 0.0 |
| 0.6b-like.float32.mb2.warm | stats | 18 | 4/18 | 0.34293699264526367 | 0.2873406137587615 | 0.0 |
| 0.6b-like.float32.mb2.warm | grads | 36 | 0/36 | 0.09955506771802902 | 1.2663635253929388 | 0.0 |
| 0.6b-like.float32.mb2.warm | updates | 36 | 0/36 | 0.12466233968734741 | 1.1565462709169108 | 0.0 |
| 0.6b-like.float32.mb2.warm | optimizer | 108 | 36/108 | 0.008272136736195534 | 0.6787396568712258 | 0.0 |
| 0.6b-like.float32.mb4.fresh | logp | 3 | 1/3 | 0.6046421527862549 | 0.13337754966158782 | 0.0 |
| 0.6b-like.float32.mb4.fresh | loss | 3 | 0/3 | 0.4893859028816223 | 64.25019954613037 | 0.0 |
| 0.6b-like.float32.mb4.fresh | stats | 18 | 4/18 | 0.23684406280517578 | 0.19833287114041462 | 0.0 |
| 0.6b-like.float32.mb4.fresh | grads | 36 | 0/36 | 0.052260007709264755 | 1.1433995641557246 | 0.0 |
| 0.6b-like.float32.mb4.fresh | updates | 36 | 0/36 | 0.09999995678663254 | 1.999999552965071 | 0.0 |
| 0.6b-like.float32.mb4.fresh | optimizer | 108 | 36/108 | 0.005935893161222339 | 1.1032030728839781 | 0.0 |
| 0.6b-like.float32.mb4.warm | logp | 3 | 1/3 | 0.5205414295196533 | 0.11620308039880915 | 0.0 |
| 0.6b-like.float32.mb4.warm | loss | 3 | 0/3 | 0.45513831079006195 | 10.78445713890263 | 0.0 |
| 0.6b-like.float32.mb4.warm | stats | 18 | 4/18 | 0.34293699264526367 | 0.2873406137587615 | 0.0 |
| 0.6b-like.float32.mb4.warm | grads | 36 | 0/36 | 0.09955506585538387 | 1.2663637417341247 | 0.0 |
| 0.6b-like.float32.mb4.warm | updates | 36 | 0/36 | 0.1246623694896698 | 1.1565462276336018 | 0.0 |
| 0.6b-like.float32.mb4.warm | optimizer | 108 | 36/108 | 0.008272135630249977 | 0.6787395661269439 | 0.0 |
| 0.6b-like.bf16-base.mball.fresh | logp | 3 | 1/3 | 0.5937726497650146 | 0.1308692897735573 | 0.0 |
| 0.6b-like.bf16-base.mball.fresh | loss | 3 | 0/3 | 0.4858726463280618 | 66.67767709416174 | 0.0 |
| 0.6b-like.bf16-base.mball.fresh | stats | 18 | 4/18 | 0.23923301696777344 | 0.20008759924333236 | 0.0 |
| 0.6b-like.bf16-base.mball.fresh | grads | 36 | 0/36 | 0.052201028913259506 | 1.1391062240451395 | 0.0 |
| 0.6b-like.bf16-base.mball.fresh | updates | 36 | 0/36 | 0.09999995678663254 | 1.9999997019766917 | 0.0 |
| 0.6b-like.bf16-base.mball.fresh | optimizer | 108 | 36/108 | 0.0055319120874628425 | 1.0606266350636937 | 0.0 |
| 0.6b-like.bf16-base.mball.warm | logp | 3 | 1/3 | 0.5196170806884766 | 0.11597598096817077 | 0.0 |
| 0.6b-like.bf16-base.mball.warm | loss | 3 | 0/3 | 0.45603106543421745 | 11.150095898275415 | 0.0 |
| 0.6b-like.bf16-base.mball.warm | stats | 18 | 4/18 | 0.3415287733078003 | 0.2861885920179488 | 0.0 |
| 0.6b-like.bf16-base.mball.warm | grads | 36 | 0/36 | 0.09925923310220242 | 1.2586625260946922 | 0.0 |
| 0.6b-like.bf16-base.mball.warm | updates | 36 | 0/36 | 0.12395457923412323 | 1.1579849648300975 | 0.0 |
| 0.6b-like.bf16-base.mball.warm | optimizer | 108 | 36/108 | 0.008206090773455799 | 0.6720275127709455 | 0.0 |
| 0.6b-like.bf16-base.mb1.fresh | logp | 3 | 1/3 | 0.5937726497650146 | 0.1308692897735573 | 0.0 |
| 0.6b-like.bf16-base.mb1.fresh | loss | 3 | 0/3 | 0.48587271571159363 | 66.67777609638988 | 0.0 |
| 0.6b-like.bf16-base.mb1.fresh | stats | 18 | 4/18 | 0.23923301696777344 | 0.20008759924333236 | 0.0 |
| 0.6b-like.bf16-base.mb1.fresh | grads | 36 | 0/36 | 0.05220102518796921 | 1.1391061427536122 | 0.0 |
| 0.6b-like.bf16-base.mb1.fresh | updates | 36 | 0/36 | 0.09999995678663254 | 1.9999997019766917 | 0.0 |
| 0.6b-like.bf16-base.mb1.fresh | optimizer | 108 | 36/108 | 0.005531911505386233 | 1.0606265234628682 | 0.0 |
| 0.6b-like.bf16-base.mb1.warm | logp | 3 | 1/3 | 0.5196170806884766 | 0.11597598096817077 | 0.0 |
| 0.6b-like.bf16-base.mb1.warm | loss | 3 | 0/3 | 0.456031009554863 | 11.150079298023943 | 0.0 |
| 0.6b-like.bf16-base.mb1.warm | stats | 18 | 4/18 | 0.3415287733078003 | 0.2861885920179488 | 0.0 |
| 0.6b-like.bf16-base.mb1.warm | grads | 36 | 0/36 | 0.09925922378897667 | 1.2586625269131062 | 0.0 |
| 0.6b-like.bf16-base.mb1.warm | updates | 36 | 0/36 | 0.12395454943180084 | 1.1579850088152732 | 0.0 |
| 0.6b-like.bf16-base.mb1.warm | optimizer | 108 | 36/108 | 0.008206090948078781 | 0.6720275270714756 | 0.0 |
| 0.6b-like.bf16-base.mb2.fresh | logp | 3 | 1/3 | 0.5937726497650146 | 0.1308692897735573 | 0.0 |
| 0.6b-like.bf16-base.mb2.fresh | loss | 3 | 0/3 | 0.48587268590927124 | 66.67749930472623 | 0.0 |
| 0.6b-like.bf16-base.mb2.fresh | stats | 18 | 4/18 | 0.23923301696777344 | 0.20008759924333236 | 0.0 |
| 0.6b-like.bf16-base.mb2.fresh | grads | 36 | 0/36 | 0.052201028913259506 | 1.1391062240451395 | 0.0 |
| 0.6b-like.bf16-base.mb2.fresh | updates | 36 | 0/36 | 0.09999995678663254 | 1.9999997019766917 | 0.0 |
| 0.6b-like.bf16-base.mb2.fresh | optimizer | 108 | 36/108 | 0.005531911621801555 | 1.0606265457830333 | 0.0 |
| 0.6b-like.bf16-base.mb2.warm | logp | 3 | 1/3 | 0.5196170806884766 | 0.11597598096817077 | 0.0 |
| 0.6b-like.bf16-base.mb2.warm | loss | 3 | 0/3 | 0.456031009554863 | 11.150079298023943 | 0.0 |
| 0.6b-like.bf16-base.mb2.warm | stats | 18 | 4/18 | 0.3415287733078003 | 0.2861885920179488 | 0.0 |
| 0.6b-like.bf16-base.mb2.warm | grads | 36 | 0/36 | 0.09925922565162182 | 1.2586625505324895 | 0.0 |
| 0.6b-like.bf16-base.mb2.warm | updates | 36 | 0/36 | 0.12395454943180084 | 1.1579850088152732 | 0.0 |
| 0.6b-like.bf16-base.mb2.warm | optimizer | 108 | 36/108 | 0.008206090773455799 | 0.6720275127709455 | 0.0 |
| 0.6b-like.bf16-base.mb4.fresh | logp | 3 | 1/3 | 0.5937726497650146 | 0.1308692897735573 | 0.0 |
| 0.6b-like.bf16-base.mb4.fresh | loss | 3 | 0/3 | 0.48587270081043243 | 66.67763770027504 | 0.0 |
| 0.6b-like.bf16-base.mb4.fresh | stats | 18 | 4/18 | 0.23923301696777344 | 0.20008759924333236 | 0.0 |
| 0.6b-like.bf16-base.mb4.fresh | grads | 36 | 0/36 | 0.052201032638549805 | 1.1391063053366668 | 0.0 |
| 0.6b-like.bf16-base.mb4.fresh | updates | 36 | 0/36 | 0.09999995678663254 | 1.9999997019766917 | 0.0 |
| 0.6b-like.bf16-base.mb4.fresh | optimizer | 108 | 36/108 | 0.005531910923309624 | 1.0606264118620428 | 0.0 |
| 0.6b-like.bf16-base.mb4.warm | logp | 3 | 1/3 | 0.5196170806884766 | 0.11597598096817077 | 0.0 |
| 0.6b-like.bf16-base.mb4.warm | loss | 3 | 0/3 | 0.4560309946537018 | 11.150074871297878 | 0.0 |
| 0.6b-like.bf16-base.mb4.warm | stats | 18 | 4/18 | 0.3415287733078003 | 0.2861885920179488 | 0.0 |
| 0.6b-like.bf16-base.mb4.warm | grads | 36 | 0/36 | 0.09925922751426697 | 1.2586625741518729 | 0.0 |
| 0.6b-like.bf16-base.mb4.warm | updates | 36 | 0/36 | 0.12395454943180084 | 1.1579850088152732 | 0.0 |
| 0.6b-like.bf16-base.mb4.warm | optimizer | 108 | 36/108 | 0.008206091821193695 | 0.6720275985741265 | 0.0 |
| 4b-like.float32.mball.fresh | logp | 3 | 1/3 | 0.4080214500427246 | 0.09312638122433858 | 0.0 |
| 4b-like.float32.mball.fresh | loss | 3 | 0/3 | 0.4419200923293829 | 14.582296870836856 | 0.0 |
| 4b-like.float32.mball.fresh | stats | 18 | 4/18 | 0.4806455365235655 | 3.539759302126698 | 0.0 |
| 4b-like.float32.mball.fresh | grads | 36 | 0/36 | 0.060725012794137 | 1.9753347130151873 | 0.0 |
| 4b-like.float32.mball.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999992549417849 | 0.0 |
| 4b-like.float32.mball.fresh | optimizer | 108 | 36/108 | 0.006747547071427107 | 1.9220403735671825 | 0.0 |
| 4b-like.float32.mball.warm | logp | 3 | 1/3 | 0.41358423233032227 | 0.09527697637202959 | 0.0 |
| 4b-like.float32.mball.warm | loss | 3 | 0/3 | 0.4341905005276203 | 10.571250242735244 | 0.0 |
| 4b-like.float32.mball.warm | stats | 18 | 4/18 | 0.21453459929583202 | 1.4879739801529919 | 0.0 |
| 4b-like.float32.mball.warm | grads | 36 | 0/36 | 0.057231079787015915 | 1.5333969468506068 | 0.0 |
| 4b-like.float32.mball.warm | updates | 36 | 0/36 | 0.2388986051082611 | 1.0958807682327874 | 0.0 |
| 4b-like.float32.mball.warm | optimizer | 108 | 36/108 | 0.010514809750020504 | 1.5812593544299267 | 0.0 |
| 4b-like.float32.mb1.fresh | logp | 3 | 1/3 | 0.4080214500427246 | 0.09312638122433858 | 0.0 |
| 4b-like.float32.mb1.fresh | loss | 3 | 0/3 | 0.44192007184028625 | 14.582286335797088 | 0.0 |
| 4b-like.float32.mb1.fresh | stats | 18 | 4/18 | 0.48064089859086107 | 3.5397233435000857 | 0.0 |
| 4b-like.float32.mb1.fresh | grads | 36 | 0/36 | 0.06072503700852394 | 1.9753351416307432 | 0.0 |
| 4b-like.float32.mb1.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999992549417849 | 0.0 |
| 4b-like.float32.mb1.fresh | optimizer | 108 | 36/108 | 0.006747529259882867 | 1.9220351724716531 | 0.0 |
| 4b-like.float32.mb1.warm | logp | 3 | 1/3 | 0.41358423233032227 | 0.09527697637202959 | 0.0 |
| 4b-like.float32.mb1.warm | loss | 3 | 0/3 | 0.4341905564069748 | 10.571258314906153 | 0.0 |
| 4b-like.float32.mb1.warm | stats | 18 | 4/18 | 0.21453439033622793 | 1.487972652933344 | 0.0 |
| 4b-like.float32.mb1.warm | grads | 36 | 0/36 | 0.05723108537495136 | 1.533396943517191 | 0.0 |
| 4b-like.float32.mb1.warm | updates | 36 | 0/36 | 0.2388986051082611 | 1.0958810678682553 | 0.0 |
| 4b-like.float32.mb1.warm | optimizer | 108 | 36/108 | 0.010514810215681791 | 1.581259535190411 | 0.0 |
| 4b-like.float32.mb2.fresh | logp | 3 | 1/3 | 0.408022403717041 | 0.09312659888994892 | 0.0 |
| 4b-like.float32.mb2.fresh | loss | 3 | 0/3 | 0.44192005693912506 | 14.582278673958932 | 0.0 |
| 4b-like.float32.mb2.fresh | stats | 18 | 4/18 | 0.48064428869868375 | 3.5397503088516773 | 0.0 |
| 4b-like.float32.mb2.fresh | grads | 36 | 0/36 | 0.060725012794137 | 1.9753342342706977 | 0.0 |
| 4b-like.float32.mb2.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999992549417849 | 0.0 |
| 4b-like.float32.mb2.fresh | optimizer | 108 | 36/108 | 0.006747542065568268 | 1.9220384377544273 | 0.0 |
| 4b-like.float32.mb2.warm | logp | 3 | 1/3 | 0.41358375549316406 | 0.0952768665235414 | 0.0 |
| 4b-like.float32.mb2.warm | loss | 3 | 0/3 | 0.4341905117034912 | 10.57124572078707 | 0.0 |
| 4b-like.float32.mb2.warm | stats | 18 | 4/18 | 0.21453467613380153 | 1.487974842210191 | 0.0 |
| 4b-like.float32.mb2.warm | grads | 36 | 0/36 | 0.05723104998469353 | 1.53339706666318 | 0.0 |
| 4b-like.float32.mb2.warm | updates | 36 | 0/36 | 0.2388986051082611 | 1.0958801689623427 | 0.0 |
| 4b-like.float32.mb2.warm | optimizer | 108 | 36/108 | 0.010514807421714067 | 1.581259115022285 | 0.0 |
| 4b-like.float32.mb4.fresh | logp | 3 | 1/3 | 0.4080219268798828 | 0.09312649005714375 | 0.0 |
| 4b-like.float32.mb4.fresh | loss | 3 | 0/3 | 0.4419202655553818 | 14.582314238384383 | 0.0 |
| 4b-like.float32.mb4.fresh | stats | 18 | 4/18 | 0.4806467469053424 | 3.5397665410868093 | 0.0 |
| 4b-like.float32.mb4.fresh | grads | 36 | 0/36 | 0.06072504073381424 | 1.9753353824975204 | 0.0 |
| 4b-like.float32.mb4.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999992549417849 | 0.0 |
| 4b-like.float32.mb4.fresh | optimizer | 108 | 36/108 | 0.006747560342773795 | 1.9220440264398724 | 0.0 |
| 4b-like.float32.mb4.warm | logp | 3 | 1/3 | 0.41358375549316406 | 0.0952768665235414 | 0.0 |
| 4b-like.float32.mb4.warm | loss | 3 | 0/3 | 0.4341905117034912 | 10.57124572078707 | 0.0 |
| 4b-like.float32.mb4.warm | stats | 18 | 4/18 | 0.21453466226317391 | 1.4879746914915635 | 0.0 |
| 4b-like.float32.mb4.warm | grads | 36 | 0/36 | 0.05723104998469353 | 1.5333973727662928 | 0.0 |
| 4b-like.float32.mb4.warm | updates | 36 | 0/36 | 0.2388986051082611 | 1.0958804685974832 | 0.0 |
| 4b-like.float32.mb4.warm | optimizer | 108 | 36/108 | 0.01051480695605278 | 1.581259044994264 | 0.0 |
| 4b-like.bf16-base.mball.fresh | logp | 3 | 1/3 | 0.4506533145904541 | 0.10231189149069947 | 0.0 |
| 4b-like.bf16-base.mball.fresh | loss | 3 | 0/3 | 0.4390290379524231 | 13.528546895519373 | 0.0 |
| 4b-like.bf16-base.mball.fresh | stats | 18 | 4/18 | 0.3615815027013182 | 2.452132500923994 | 0.0 |
| 4b-like.bf16-base.mball.fresh | grads | 36 | 0/36 | 0.06337326020002365 | 1.8502872728095394 | 0.0 |
| 4b-like.bf16-base.mball.fresh | updates | 36 | 0/36 | 0.09999993816018105 | 1.9999988824128438 | 0.0 |
| 4b-like.bf16-base.mball.fresh | optimizer | 108 | 36/108 | 0.006853009341284633 | 1.120137373707236 | 0.0 |
| 4b-like.bf16-base.mball.warm | logp | 3 | 1/3 | 0.4158670902252197 | 0.09579747760342998 | 0.0 |
| 4b-like.bf16-base.mball.warm | loss | 3 | 0/3 | 0.4338981918990612 | 10.548066455349717 | 0.0 |
| 4b-like.bf16-base.mball.warm | stats | 18 | 4/18 | 0.2188144320026249 | 1.5150449934781443 | 0.0 |
| 4b-like.bf16-base.mball.warm | grads | 36 | 0/36 | 0.05728827975690365 | 1.5329757334918723 | 0.0 |
| 4b-like.bf16-base.mball.warm | updates | 36 | 0/36 | 0.2501996159553528 | 0.9618670011558024 | 0.0 |
| 4b-like.bf16-base.mball.warm | optimizer | 108 | 36/108 | 0.010513655375689268 | 1.5748680036198823 | 0.0 |
| 4b-like.bf16-base.mb1.fresh | logp | 3 | 1/3 | 0.45067834854125977 | 0.10231757495241048 | 0.0 |
| 4b-like.bf16-base.mb1.fresh | loss | 3 | 0/3 | 0.43906300514936447 | 13.529578053307834 | 0.0 |
| 4b-like.bf16-base.mb1.fresh | stats | 18 | 4/18 | 0.3616091167461052 | 2.4523197700234216 | 0.0 |
| 4b-like.bf16-base.mb1.fresh | grads | 36 | 0/36 | 0.06333804875612259 | 1.8492590152608497 | 0.0 |
| 4b-like.bf16-base.mb1.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999992549417849 | 0.0 |
| 4b-like.bf16-base.mb1.fresh | optimizer | 108 | 36/108 | 0.006852598395198584 | 1.1200702037901036 | 0.0 |
| 4b-like.bf16-base.mb1.warm | logp | 3 | 1/3 | 0.4158670902252197 | 0.09579747760342998 | 0.0 |
| 4b-like.bf16-base.mb1.warm | loss | 3 | 0/3 | 0.4338981732726097 | 10.548061226282549 | 0.0 |
| 4b-like.bf16-base.mb1.warm | stats | 18 | 4/18 | 0.21880927623480026 | 1.5150093217669662 | 0.0 |
| 4b-like.bf16-base.mb1.warm | grads | 36 | 0/36 | 0.05728827975690365 | 1.5329757334918723 | 0.0 |
| 4b-like.bf16-base.mb1.warm | updates | 36 | 0/36 | 0.25019972026348114 | 0.9618670715487001 | 0.0 |
| 4b-like.bf16-base.mb1.warm | optimizer | 108 | 36/108 | 0.010513655375689268 | 1.5748680036198823 | 0.0 |
| 4b-like.bf16-base.mb2.fresh | logp | 3 | 1/3 | 0.4506533145904541 | 0.10231189149069947 | 0.0 |
| 4b-like.bf16-base.mb2.fresh | loss | 3 | 0/3 | 0.4390289783477783 | 13.528532634898536 | 0.0 |
| 4b-like.bf16-base.mb2.fresh | stats | 18 | 4/18 | 0.3615814985745877 | 2.452132461002718 | 0.0 |
| 4b-like.bf16-base.mb2.fresh | grads | 36 | 0/36 | 0.0633732583373785 | 1.850287017178177 | 0.0 |
| 4b-like.bf16-base.mb2.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999989569186543 | 0.0 |
| 4b-like.bf16-base.mb2.fresh | optimizer | 108 | 36/108 | 0.00685300980694592 | 1.1201374498204566 | 0.0 |
| 4b-like.bf16-base.mb2.warm | logp | 3 | 1/3 | 0.4158670902252197 | 0.09579747760342998 | 0.0 |
| 4b-like.bf16-base.mb2.warm | loss | 3 | 0/3 | 0.4338981807231903 | 10.548063317908849 | 0.0 |
| 4b-like.bf16-base.mb2.warm | stats | 18 | 4/18 | 0.21881443020643204 | 1.5150449927813523 | 0.0 |
| 4b-like.bf16-base.mb2.warm | grads | 36 | 0/36 | 0.05728828348219395 | 1.5329756803622152 | 0.0 |
| 4b-like.bf16-base.mb2.warm | updates | 36 | 0/36 | 0.2501997649669647 | 0.9618670230006596 | 0.0 |
| 4b-like.bf16-base.mb2.warm | optimizer | 108 | 36/108 | 0.010513655841350555 | 1.5748679635213327 | 0.0 |
| 4b-like.bf16-base.mb4.fresh | logp | 3 | 1/3 | 0.4507131576538086 | 0.10232547767059912 | 0.0 |
| 4b-like.bf16-base.mb4.fresh | loss | 3 | 0/3 | 0.43906308710575104 | 13.529589897424994 | 0.0 |
| 4b-like.bf16-base.mb4.fresh | stats | 18 | 4/18 | 0.3623417398706246 | 2.4572881965493045 | 0.0 |
| 4b-like.bf16-base.mb4.fresh | grads | 36 | 0/36 | 0.06334188021719456 | 1.849370881099285 | 0.0 |
| 4b-like.bf16-base.mb4.fresh | updates | 36 | 0/36 | 0.09999994188547134 | 1.9999992549417849 | 0.0 |
| 4b-like.bf16-base.mb4.fresh | optimizer | 108 | 36/108 | 0.006852603051811457 | 1.1200709649223088 | 0.0 |
| 4b-like.bf16-base.mb4.warm | logp | 3 | 1/3 | 0.4158670902252197 | 0.09579747760342998 | 0.0 |
| 4b-like.bf16-base.mb4.warm | loss | 3 | 0/3 | 0.4338981807231903 | 10.548063317908849 | 0.0 |
| 4b-like.bf16-base.mb4.warm | stats | 18 | 4/18 | 0.21881442656230685 | 1.5150449089479894 | 0.0 |
| 4b-like.bf16-base.mb4.warm | grads | 36 | 0/36 | 0.0572882816195488 | 1.5329756305197424 | 0.0 |
| 4b-like.bf16-base.mb4.warm | updates | 36 | 0/36 | 0.2501997649669647 | 0.9618670230006596 | 0.0 |
| 4b-like.bf16-base.mb4.warm | optimizer | 108 | 36/108 | 0.010513655375689268 | 1.5748680036198823 | 0.0 |


measurements: `{"clip_frac": {"0.6b-like.float32.mball.fresh": [0.0, 0.0625, 0.25], "0.6b-like.float32.mball.warm": [0.0, 0.0625, 0.1875], "0.6b-like.float32.mb1.fresh": [0.0, 0.0625, 0.25], "0.6b-like.float32.mb1.warm": [0.0, 0.0625, 0.1875], "0.6b-like.float32.mb2.fresh": [0.0, 0.0625, 0.25], "0.6b-like.float32.mb2.warm": [0.0, 0.0625, 0.1875], "0.6b-like.float32.mb4.fresh": [0.0, 0.0625, 0.25], "0.6b-like.float32.mb4.warm": [0.0, 0.0625, 0.1875], "0.6b-like.bf16-base.mball.fresh": [0.0, 0.125, 0.25], "0.6b-like.bf16-base.mball.warm": [0.0, 0.0625, 0.1875], "0.6b-like.bf16-base.mb1.fresh": [0.0, 0.125, 0.25], "0.6b-like.bf16-base.mb1.warm": [0.0, 0.0625, 0.1875], "0.6b-like.bf16-base.mb2.fresh": [0.0, 0.125, 0.25], "0.6b-like.bf16-base.mb2.warm": [0.0, 0.0625, 0.1875], "0.6b-like.bf16-base.mb4.fresh": [0.0, 0.125, 0.25], "0.6b-like.bf16-base.mb4.warm": [0.0, 0.0625, 0.1875], "4b-like.float32.mball.fresh": [0.0, 0.125, 0.1875], "4b-like.float32.mball.warm": [0.0, 0.125, 0.0625], "4b-like.float32.mb1.fresh": [0.0, 0.125, 0.1875], "4b-like.float32.mb1.warm": [0.0, 0.125, 0.0625], "4b-like.float32.mb2.fresh": [0.0, 0.125, 0.1875], "4b-like.float32.mb2.warm": [0.0, 0.125, 0.0625], "4b-like.float32.mb4.fresh": [0.0, 0.125, 0.1875], "4b-like.float32.mb4.warm": [0.0, 0.125, 0.0625], "4b-like.bf16-base.mball.fresh": [0.0, 0.125, 0.3125], "4b-like.bf16-base.mball.warm": [0.0, 0.125, 0.0625], "4b-like.bf16-base.mb1.fresh": [0.0, 0.125, 0.3125], "4b-like.bf16-base.mb1.warm": [0.0, 0.125, 0.0625], "4b-like.bf16-base.mb2.fresh": [0.0, 0.125, 0.3125], "4b-like.bf16-base.mb2.warm": [0.0, 0.125, 0.0625], "4b-like.bf16-base.mb4.fresh": [0.0, 0.125, 0.3125], "4b-like.bf16-base.mb4.warm": [0.0, 0.125, 0.0625]}}`

memory snapshot (snapshot), evidence grade **none**:
- CPU RSS 738.2578125 MiB, high-water mark 738.2578125 MiB (this process)
