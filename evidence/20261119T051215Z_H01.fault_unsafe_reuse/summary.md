# H01.fault_unsafe_reuse (t08_forward_equivalence_async_unsafe_reuse_r20261119T051215Z)

- status: **PASS**
- hypothesis criterion: **NOT_EVALUATED**
- detail: fault injection: detected (status FAIL, execution PASS)
- commit 29852d11c2a37eeda94cd5ee53692fffd8d75d69 (tracked tree dirty: False)
- hardware: NVIDIA GeForce RTX 3060, bf16 True, Linux-7.0.0-38-generic-x86_64-with-glibc2.39
- precision ['float32', 'bfloat16']

| step | exit | seconds | log |
|---|---|---|---|
| `scripts/l2l_lab.py run --name t08_forward_equivalence_async_unsafe_reuse_r20261119T051215Z --case forward_equivalence --config research/l2l/configs/async.yaml --device cuda --perturb unsafe_reuse=2` | 0 | 14.1 | raw/step01.log |

Recorded statuses in research/l2l/hypotheses.yaml are not changed by this run.
