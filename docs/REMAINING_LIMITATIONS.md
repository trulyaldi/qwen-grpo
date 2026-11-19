# Remaining limitations (as of 2026-11-19)

## Mandatory assignment requirements not satisfied

| requirement | status | why | what would close it |
|---|---|---|---|
| **2(c) DFlash rollouts** (drafter `z-lab/Qwen3-4B-DFlash-b16`; acceptance length; step time with and without; greedy token identity; sampling distribution check; acceptance under policy drift) | **NOT RUN.** Deliberately out of scope for the 150-update experiment | DFlash stays disabled (`dflash.enabled: false`) in every `*_150_v2` config. Code exists (`qwen_grpo/execution/dflash.py`, `scripts/dflash_bench.py`, `configs/experiments/qwen3_4b_*dflash*.yaml`) and has CPU tests, but it has never run on Qwen3-4B. Nothing in this repository claims a DFlash result | `scripts/run_assignment.sh` full mode stage E (`H10.dflash_bench`, `H10.smoke_dflash`) on the GPU, then the drift study on the 150-update run's checkpoints. Not prepared as part of this experiment |
| **8 GB cap verified for every phase, including sub-sampling spikes** | **Partially, by construction.** nvidia-smi is sampled every 200 ms, so a peak is a lower bound | the in-process guard (0.5 s) and the allocator cap (8e9 B as a fraction of the card, PyTorch allocations only) are the second line. Neither sees a sub-200 ms non-PyTorch spike | not closable with nvidia-smi alone. A `cudaMallocAsync` pool limit or `nsys` memory tracing would narrow it |
| **H01 equivalence on the rented GPU with the new same-host methodology** | **NOT YET RUN** | the methodology is implemented and tested on CPU (`tests/test_h01_same_host.py`). The historical Vast.ai result `t08_forward_equivalence_async` stays FAIL (two cross-host fixture rows) and is not rewritten | `scripts/final150.py correctness` |
| **Copy/compute overlap demonstrated** | **NOT YET MEASURED** | instrumentation and the Qwen3-4B lab entry `H06.prefetch_timing_4b` are ready | `scripts/final150.py profile`. Prediction (not a result): the 60-update step is transfer-bound (8.5 s measured vs an 8.5 s transfer bound), so async can hide at most the compute share. Expect a small speed-up or none. A "no speed-up" outcome will be reported as measured |
| **Merged-model evaluation under nvidia-smi** | **NOT YET RUN** for the 150-update run; it was never measured for the 60-update run (allocator figure only) | the merge and eval stages now run as monitored items | `scripts/final150.py merge` and `eval` |
| **Soup checks on the Qwen3-4B artifacts** | **NOT EXECUTED** in this session | Soup could not be installed here. 0.77.2 evidence (Qwen3-0.6B, CPU) is restored | `scripts/final150.py soup` with Soup installed (CPU suffices) |

## Methodological limits that remain after the run

- **The LR = 0 control** shows that the pipeline attributes no change to an unchanged model. It does not show that the
  *reward* caused the trained model's change. A shuffled-reward control would; it is not implemented.
- **The validation subset** (n = 128) cannot resolve differences of a few points (SE ≈ 2.8 pp). The validation curve
  only monitors, and checkpoint selection never uses it: the final evaluation uses step 150 by construction.
- **Sparse signal.** The base model is about 88% accurate on 4-token answers, so most groups of 6 are uniform. In the
  60-update run 77% of rollout batches carried no signal. 150 updates may still not move held-out accuracy
  significantly. If so, that is the result.
- **Per-update PPO diagnostics** (ratio, clip fraction, approx-KL) are structurally 1 / 0 / 0 with
  `num_iterations = 1`, because they are measured before `optimizer.step()`. The update norm and the label-score shift
  are the informative quantities.
- **Async correctness in performance mode.** The H01/H02/H05 async correctness labs run with `verify=True` (a host
  synchronization per layer use). Production and H06 run with `verify=False`. H06 checks bitwise equality between sync
  and async for the first trial in that mode, which is one trial on one batch.
- **Async equivalence on tiny models only.** Equivalence of the async executor was established on tiny random models.
  Real-model bitwise evidence exists for `l2l_sync` on Qwen3-0.6B, and in H06 for one trial on Qwen3-4B (when it runs).
- **The 0.78.0 Soup behaviour is unverified.** The coverage matrix reports 0.77.2 findings plus prepared 0.78.0 calls.
- **Real-model kill/resume.** It is prepared (`configs/experiments/qwen3_4b_resume_kill.yaml`) but is not part of the
  default stage list, so it costs no credit unless run explicitly.
- **A second orchestrator.** `scripts/final150.py` sits beside `scripts/run_assignment.sh`. They share telemetry, the
  study dispatcher and the training code, but their state files are separate.

## Superseded or failed records kept on purpose

- `research/l2l/experiments/t08_forward_equivalence_async/`: the historical H01 FAIL (cross-host fixture rows).
- `research/l2l/experiments/t09_resume_kill_tiny_cpu_r20261119T040250Z.json`: the first external-kill run. It FAILed
  only on a wrong check (Python's unseeded, unused global RNG compared across processes); the check was replaced
  (`docs/FINAL_150_STEP_READINESS.md`). The run itself is unchanged.
- `assignment_results/20261118T171404Z/`: the 60-update orchestrated run, in which the gate blocked on H01. The training
  runs that followed were started by hand.
