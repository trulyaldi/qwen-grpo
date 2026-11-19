# H04.lab_async_cuda (t08_generation_equivalence_async_r20261119T051351Z)

- status: **PASS**
- hypothesis criterion: **MET**
- detail: status PASS, execution PASS, hypothesis_result SUPPORTED: all 1292 comparisons within predefined thresholds
- commit 29852d11c2a37eeda94cd5ee53692fffd8d75d69 (tracked tree dirty: False)
- hardware: NVIDIA GeForce RTX 3060, bf16 True, Linux-7.0.0-38-generic-x86_64-with-glibc2.39
- precision ['float32', 'bfloat16']

| step | exit | seconds | log |
|---|---|---|---|
| `scripts/l2l_lab.py run --name t08_generation_equivalence_async_r20261119T051351Z --case generation_equivalence --config research/l2l/configs/async.yaml --device cuda --hypothesis H04` | 0 | 42.9 | raw/step01.log |

Recorded statuses in research/l2l/hypotheses.yaml are not changed by this run.
