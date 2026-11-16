# L2L experiment protocol

How an L2L hypothesis (`hypotheses.yaml`) is tested and how its outcome is recorded. The runner is
`scripts/l2l_lab.py` (logic in `qwen_grpo/lab.py`).

```
python scripts/l2l_lab.py run --case <case> --config <yaml> [--name N] [--perturb name=rel] [--trials k] [--device d]
python scripts/l2l_lab.py report --run <name>
```

## Pairing

Each case runs the **resident reference** and the **candidate** (the backend named by the config's `execution`
section) on the same frozen inputs: the Ticket 00 fixtures, with the same weights, LoRA tensors, rollouts, advantages,
old log-probs, optimizer settings, dtype and thread count. The reference is recomputed on every run and must also
equal the saved one bitwise. A drifting reference fails the experiment. Nothing is resampled between the two sides.

## Correctness cases

The thresholds are set in `tolerances.yaml` **before** the acceptance run, each with its justification. A comparison
passes iff shapes and dtypes match, both sides are finite, and `max|got − ref| ≤ atol + rtol·max|ref|`. Every tensor is
reported separately, per parameter for gradients, trained LoRA tensors and AdamW state. The report gives max abs error,
relative error, relative L2 error, cosine and non-finite counts.

The experiment is **PASS** only if all of the following hold:
- at least one comparison exists, and every reference quantity has a candidate;
- every comparison passes;
- every reference gradient is nonzero;
- the fresh reference equals the saved one;
- every required measurement is present.

A run that produces no measurements cannot pass.

## Performance cases

- Warmup, then reference and candidate trials **interleaved**, so that drift affects both sides alike.
- The report gives the median, IQR, range and a bootstrap 95% CI of the median for each side, and the
  candidate/reference median ratio with a bootstrap 95% CI.
- **PASS** means the measurement is valid: at least `MIN_TRIALS` = 5 finite trials, plus the required memory evidence.
  What the measurement shows is the separate `conclusion`: faster, slower, or no detectable difference.
- No speedup is claimed unless the ratio's CI excludes 1.

## Memory evidence

`lab.memory` takes a **snapshot**, not a peak trace. Peak sampling is Ticket 11. It records:
- CPU RSS and its high-water mark;
- on CUDA, the allocator's current and peak allocated/reserved bytes, `cudaMemGetInfo` (total − free), `nvidia-smi`
  `memory.used`, the per-process `nvidia-smi` query, and the pinned host allocator statistics.

The two device-level readings include the CUDA context and every other process on the GPU, so they are upper bounds
for this process. On this host (WSL2) the per-process `nvidia-smi` query returns nothing. GPU memory evidence counts as
complete when the allocator statistics and at least one device-level reading are present. A case on CUDA requires it,
and the run fails without it. An allocator limit is never treated as proof of a budget.

## Statuses

| status | meaning |
|---|---|
| PASS | the acceptance rule above holds on the recorded raw data |
| FAIL | it does not, or the experiment was ill-posed: missing fixture, fixture built with other settings or package versions, config dtype ≠ fixture dtype, a quantity without a threshold, an unrepresentable perturbation, or an exception |
| BLOCKED | the candidate cannot run with this code base yet (an `unimplemented` setting); never a PASS |

The status is derived from the raw data by `lab.decide`. `report` re-derives it and refuses a result whose recorded
status disagrees with its data.

## Records

Results are written to `research/l2l/experiments/<name>/result.json` and `summary.md`, **whatever the outcome**, and
are never overwritten. Exceptions are recorded with a traceback and the measurements made up to that point. Each
result records:
- the config path, its sha256, which config fields the case actually used, and the execution section;
- the fixture file hashes (input sha256) and the `tolerances.yaml` hash;
- commit, tracked-dirty flag, Python, torch, transformers and peft versions, CUDA, GPU, thread count;
- every comparison and every trial;
- any fault injection.

Fixture cases use only `model.dtype` and `execution` from the config. The model named in the config is **not** the
model these cases run.

## Fault injection

`--perturb name=rel` adds `rel·|x₀|` to the first element of the candidate's first tensor named `name`. It is refused
if the change rounds away (Ticket 00 showed that small shifts can be invisible). Such runs are marked
`fault_injection: true`, and their summary says so on its first line. Every correctness test has at least one fault
that it must detect.

## Rules

Hypotheses are tested one ticket at a time. Thresholds are never widened to make a failing run pass. Failed runs are
kept. Theoretical predictions are never recorded as measurements. Tests write to temporary directories only.
