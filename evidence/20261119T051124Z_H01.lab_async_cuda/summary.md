# H01.lab_async_cuda (t08_forward_equivalence_async_r20261119T051124Z)

- status: **PASS**
- hypothesis criterion: **MET**
- detail: status PASS, execution PASS, hypothesis_result SUPPORTED: all 2690 comparisons within predefined thresholds
- commit 29852d11c2a37eeda94cd5ee53692fffd8d75d69 (tracked tree dirty: False)
- hardware: NVIDIA GeForce RTX 3060, bf16 True, Linux-7.0.0-38-generic-x86_64-with-glibc2.39
- precision ['float32', 'bfloat16']

| step | exit | seconds | log |
|---|---|---|---|
| `scripts/l2l_lab.py run --name t08_forward_equivalence_async_r20261119T051124Z --case forward_equivalence --config research/l2l/configs/async.yaml --device cuda --hypothesis H01` | 0 | 40.2 | raw/step01.log |

Recorded statuses in research/l2l/hypotheses.yaml are not changed by this run.
