# L2L experiment protocol

How an L2L hypothesis (`hypotheses.yaml`) is tested and how its outcome is recorded. The runner is
`scripts/l2l_lab.py` (logic in `qwen_grpo/lab.py`).

```
python scripts/l2l_lab.py run --case <case> --config <yaml> [--name N] [--hypothesis H..] [--perturb name=rel] [--trials k] [--device d]
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

`lab.memory` takes a **snapshot**, not a peak trace. Peak sampling is Ticket 11. Each reading keeps its own meaning, and
none is combined with another into one number:

| reading | scope | meaning |
|---|---|---|
| PyTorch peak allocated | this process, PyTorch allocator | bytes in live tensors; excludes the CUDA context, the allocator's cache and non-PyTorch allocations |
| PyTorch peak reserved | this process, PyTorch allocator | allocated plus cached blocks; still excludes the CUDA context and non-PyTorch allocations |
| `cudaMemGetInfo` total − free | **device-wide** | every process and CUDA context on the GPU (and, under WSL2, whatever the Windows host holds); not process-specific |
| `nvidia-smi` raw | device / per process | `memory.used` and the per-process query, stored verbatim |
| CPU RSS and high-water mark | this process | `/proc/self/status` |
| pinned host allocator statistics | this process | recorded raw; their meaning is unverified, and they are not evidence that memory is pinned (Ticket 02 checks `is_pinned()` directly) |

**Evidence grades** (`lab.evidence_grade`, stored in every snapshot):

| grade | condition | use |
|---|---|---|
| `none` | no GPU reading | — |
| `incomplete` | the allocator statistics or `cudaMemGetInfo` are missing | a CUDA case that requires GPU memory fails |
| `diagnostic` | allocator statistics + `cudaMemGetInfo` | local development on this WSL2 host |
| `assignment` | also a numeric per-process `nvidia-smi` reading, outside WSL2 | the only grade that may support a claim that the 8 GB budget was respected |

On this host (WSL2) `nvidia-smi` lists the process with `[N/A]` memory, and its `memory.used` read 134 MiB while
`cudaMemGetInfo` read 857 MiB and the allocator held 22 MiB reserved (`t01_timing_cuda`). That reading is below what the
process itself must hold (reserved memory plus the CUDA context), so `nvidia-smi` is unreliable here. Its output is kept
raw and is not counted. Local measurements are therefore at most `diagnostic`. **No 8 GB compliance is claimed without
`assignment`-grade evidence from the target Linux environment**, and an allocator limit is never treated as proof of a
budget.

## Statuses

Each result (schema 2) records three fields, all derived from the raw data and re-checked by `report`:

| field | values | meaning |
|---|---|---|
| `execution_status` | PASS / FAIL / BLOCKED | did the experiment run validly? FAIL: an exception, an ill-posed setup (missing fixture, fixture built with other settings or package versions, config dtype ≠ fixture dtype, a quantity without a threshold, an unrepresentable perturbation), missing required measurements, no data, a drifted reference, or too few trials. BLOCKED: the candidate cannot run with this code base yet. |
| `hypothesis_result` | SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / NOT_EVALUATED | the outcome for the hypothesis the run declares (`--hypothesis H..`). Correctness: SUPPORTED iff every comparison meets its predefined threshold, NOT_SUPPORTED otherwise; no significance test. Performance: from the median-ratio bootstrap CI against the declared `prediction`; a CI containing 1 is INCONCLUSIVE. NOT_EVALUATED: no hypothesis declared, a fault-injection run, or an execution that is not PASS. |
| `status` | PASS / FAIL / BLOCKED | the Ticket 01 acceptance status, unchanged in meaning: PASS iff execution PASS and, for correctness, every comparison is within its threshold |

A benchmark can therefore execute (`execution_status` PASS) without supporting a speedup (`INCONCLUSIVE`). A completed
experiment is never by itself a supported hypothesis.

Ticket 01 results (no `schema` field) record `status` only. `report` checks them and derives the other two fields for
display, without rewriting their files.

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
