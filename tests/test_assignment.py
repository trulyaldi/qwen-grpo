"""scripts/run_assignment.sh end to end with fakes: the real Bash entrypoint, stage runner, telemetry and report. A fake
dispatcher, fake tools and a fake nvidia-smi (tests/assignment_fakes/) replace experiment.py, pytest/budget/run.py and
the GPU. Every number these tests produce is synthetic. Checked: a successful run; a missing GPU; missing per-process
nvidia-smi readings; a memory-cap violation stopping the workload; interruption and resume; stale configuration and
code rejection; a failed DFlash benchmark; reports with incomplete evidence; logs kept on failure; no unmeasured value
judged PASS; --dry-run writes nothing. CPU only, no model, no network."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FAKES = ROOT / "tests" / "assignment_fakes"
ESSENTIAL_ENTRIES = ["H01.lab_sync", "H01.fault_wrong_layer", "H01.lab_async_cuda", "H01.fault_unsafe_reuse",
                     "H02.lab_sync_cuda", "H02.fault_detach_boundary", "H02.lab_async_cuda", "H02.fault_unsafe_reuse",
                     "H04.lab_sync_cuda", "H04.fault_stale_position", "H04.lab_async_cuda", "H05.lab_sync_cuda",
                     "H05.fault_advantage", "H05.lab_async_cuda", "H08.resume_tiny_cpu", "H09.merge_tiny_cpu"]
GIB = 2**30

PREDICTION = {"phases": [{"phase": "training", "gpu_peak": {"gpu_allocated": 3.0e9, "gpu_reserved": 3.3e9,
                                                            "gpu_device": 3.9e9, "cpu": 1.5e10},
                          "notes": ["synthetic"], "moments": [{"gpu": [{"name": "layer buffers x2", "bytes": 403723264,
                                                                       "kind": "exact"}]}]}],
              "transfers": [{"name": "rollout decode", "bytes": 21801056256, "transfer_s": 1.82}],
              "residency": {"recommendation": "stream all 36 layers (synthetic)"}}
PROFILE = {"verdict": "PASS", "pinned_bytes": 7267018752, "h2d_bandwidth_bytes_per_s": 2.2e10,
           "profile": {"evidence": "assignment", "violations": [], "phases": [
               {"phase": p, "seconds": 10.0, "samples": 40, "peak_allocated": 2.9e9, "peak_reserved": 3.2e9,
                "device_used_peak": 4.1e9, "process_used_peak": 3.9e9, "cpu_rss_peak": 1.4e10}
               for p in ("setup", "rollout", "old_logp", "forward", "backward", "optimizer", "merge", "evaluation")]},
           "transfers": {"rollout": {"bytes": 3.6e10, "predicted_transfer_s": 1.6, "phase_wall_s": 10.0,
                                     "transfer_share_of_wall": 0.16}},
           "prediction": {"file": "results/budget/predictions/qwen3-4b-final_8gb.json", "rows": [
               {"phase": "forward", "predicted_phase": "training", "predicted_allocated": 3.0e9,
                "observed_peak_allocated": 2.9e9, "predicted_reserved": 3.3e9, "observed_peak_reserved": 3.2e9,
                "predicted_device": 3.9e9, "observed_device_peak": 4.1e9, "observed_process_peak": 3.9e9,
                "predicted_cpu": 1.5e10, "observed_cpu_rss_peak": 1.4e10}]}}


def bench(equal=True):
    run = {"max_new": 4, "greedy_equal": equal, "greedy_passes_per_token": 0.5, "sample_passes_per_token": 0.6,
           "greedy_plain_s": 3.0, "greedy_dflash_s": 2.0, "sample_plain_s": 3.1, "sample_dflash_s": 2.2,
           "greedy_plain_s_trials": [3.0, 3.1, 2.9], "greedy_dflash_s_trials": [2.0, 2.1, 1.9],
           "sample_plain_s_trials": [3.1, 3.0, 3.2], "sample_dflash_s_trials": [2.2, 2.3, 2.1],
           "sample_accepted_per_pass_trials": [[2, 3], [3, 3], [2, 2]],
           "sample_plain_responses": [["POSITIVE"] * 15 + ["NEGATIVE"] * 15] * 3,
           "sample_dflash_responses": [["POSITIVE"] * 14 + ["NEGATIVE"] * 16] * 3}
    return {"drafter": "z-lab/Qwen3-4B-DFlash-b16", "drafter_bytes": 1.08e9, "all_greedy_equal": equal, "B": 5, "G": 6,
            "repeats": 3, "budget_verdict": "PASS",
            "policies": [{"policy": {"b_scale": 0.0}, "runs": [run], "grpo_step_plain_s_trials": [20.0, 21.0, 19.0],
                          "grpo_step_dflash_s_trials": [15.0, 16.0, 14.0], "grpo_step_plain_s": 20.0,
                          "grpo_step_dflash_s": 15.0}]}


def metrics(acc, pos=0.5, inv=0.0):
    return {"n": 1066, "accuracy": acc, "macro_f1": acc - 0.01, "invalid_rate": inv, "predicted_positive_rate": pos,
            "predicted_negative_rate": 1 - pos - inv, "recall_POSITIVE": acc, "recall_NEGATIVE": acc, "auc": 0.9,
            "changed_vs_base": 0.0}


EVAL = {"n_test": 1066, "split": "test", "checkpoint": "runs/qwen3_4b_final/checkpoints/step60",
        "models": {"base": metrics(0.80), "trained": {**metrics(0.86), "changed_vs_base": 0.1},
                   "merged": {**metrics(0.86), "changed_vs_base": 0.1}, "control": metrics(0.80)},
        "paired": {"trained_minus_base": {"diff": 0.06, "ci_low": 0.04, "ci_high": 0.08},
                   "merged_minus_trained": {"diff": 0.0, "ci_low": -0.002, "ci_high": 0.002},
                   "control_minus_base": {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}},
        "data_separation": {"test_texts_in_train": 0}, "merge": {"all_within_half_ulp": True},
        "merged_vs_trained_prediction_agreement": 0.999, "label_scores": {"trained": [1.0, -2.0], "merged": [1.001, -2.0]},
        "control_log": {"params_equal_init": True}}


DRAFTER = "z-lab/Qwen3-4B-DFlash-b16"


def generator(dflash):
    """A rollout_generator record as qwen_grpo.trainer.rollout_evidence writes it (synthetic numbers)."""
    if not dflash:
        return {"generator": "StreamedGenerator"}
    return {"generator": "DFlashGenerator", "draft_model": DRAFTER, "drafter_bytes": 1074854400, "steps": 4,
            "target_passes": 2, "accepted_per_pass": [2], "accepted_drafts": 2, "calls": 1}


def training_log(steps=4, dflash=False, record=True, max_steps=None):
    """record False: a log from before the trainer recorded its rollout generator (config-only evidence)."""
    gen = {"rollout_generator": generator(dflash)} if record else {}
    backend = "l2l_async+dflash" if dflash else "l2l_async"
    return {"timings": [{"step": s, "rollout_s": 12.0, "prepare_s": 3.0, "update_s": 6.0, "step_s": 21.0 + s * 0.1,
                         "cpu_rss_bytes": 1.3e10, "backend": backend, **gen} for s in range(1, steps + 1)],
            "config": {"steps": steps if max_steps is None else max_steps, "train_samples": 1024, "G": 6},
            **({"rollout_backend": {"backend": backend, "generator": generator(dflash)["generator"],
                                    "draft_model": DRAFTER if dflash else None}} if record else {}),
            "training": [{"step": s, "attempt": s, "prompt_ids": list(range(5 * s, 5 * s + 5)), "mean_reward": 0.5,
                          "grad_norm": 0.1, "update_norm": 0.01,
                          "peak_reserved_gib": 3.0, "peak_allocated_gib": 2.8, "change_from_init": 0.1}
                         for s in range(1, steps + 1)], "zero_signal": [], "successful_steps": steps,
            "groups_generated": steps, "frozen_changed": [], "final_checkpoint": {"path": "x"}}


def scenario(**over):
    s = {"*": {"status": "PASS", "sleep": 0.4},
         "H01.lab_sync": {"status": "PASS", "files": {"t03_forward_equivalence/result.json": {
             "status": "PASS", "case": "forward_equivalence", "provenance": {"gpu": "NVIDIA Fake A6000"},
             "comparisons": [{"quantity": "logits", "max_abs": 0.0, "bound": 0.0, "pass": True,
                              "dtype": "torch.bfloat16", "shape": [3, 10, 64]}]}}},
         "H11.smoke": {"status": "PASS", "files": {"runs/qwen3_4b_smoke/training/training.json": training_log()}},
         "H11.profile": {"status": "PASS", "files": {"t11_profile_qwen3_4b_async.json": PROFILE}},
         "H10.dflash_bench": {"status": "PASS", "files": {"t12_dflash_bench.json": bench()}},
         "H10.smoke_dflash": {"status": "PASS",
                              "files": {"runs/qwen3_4b_smoke_dflash/training/training.json": training_log(dflash=True)}},
         "H12.train_final": {"status": "PASS", "files": {
             "runs/qwen3_4b_final/training/training.json": training_log(8, dflash=True),
             "runs/qwen3_4b_final_lr0/training/training.json": training_log(3, dflash=True)}},
         "H12.final_eval": {"status": "PASS", "files": {"t13_final_eval_qwen3_4b_final.json": EVAL}},
         "H10.dflash_drift": {"status": "PASS", "files": {"t12_dflash_drift_final.json": bench()}}}
    for k, v in over.items():
        s[k.replace("__", ".")] = {**s.get(k.replace("__", "."), {"status": "PASS"}), **v}
    return s


@pytest.fixture
def env(tmp_path):
    for d in ("results", "evidence", "exp", "runs"):
        (tmp_path / d).mkdir()
    probe = {"cuda": True, "bf16": True, "gpu": "NVIDIA Fake A6000 (8.6)", "capability": "8.6", "vram_gib": 48.0,
             "per_process_smi": True, "linux_native": True, "fixture_packages": True, "host_ram_gib": 64,
             "packages": {"torch": "fake"}}
    (tmp_path / "probe.json").write_text(json.dumps(probe))
    fake = f"{sys.executable} {FAKES / 'fake_tool.py'}"
    e = {**os.environ, "ASSIGNMENT_RESULTS_ROOT": str(tmp_path / "results"),
         "ASSIGNMENT_EVIDENCE_ROOT": str(tmp_path / "evidence"), "ASSIGNMENT_EXPERIMENTS_ROOT": str(tmp_path / "exp"),
         "ASSIGNMENT_RUNS_ROOT": str(tmp_path / "runs"), "ASSIGNMENT_PROBE_JSON": str(tmp_path / "probe.json"),
         "ASSIGNMENT_DISPATCHER": f"{sys.executable} {FAKES / 'fake_dispatcher.py'}",
         "ASSIGNMENT_NVIDIA_SMI": str(FAKES / "fake_nvidia_smi.py"), "ASSIGNMENT_GRACE_S": "2",
         "ASSIGNMENT_TOOL_OVERRIDES": json.dumps({k: fake for k in ("cuda_check", "budget", "pytest", "compare_runs",
                                                                     "run_py")}),
         "ASSIGNMENT_STAGE_ITEMS": json.dumps({"B": ESSENTIAL_ENTRIES}), "FAKE_SMI_MARKER": "assignment_fakes",
         "FAKE_SMI_MIB": json.dumps({"*": 5000}), "FAKE_LOG": str(tmp_path / "dispatch.log"),
         "FAKE_SCENARIO": str(tmp_path / "scenario.json"), "FAKE_PREDICTION": json.dumps(PREDICTION),
         "PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"]}
    e.pop("SOUP_BIN", None)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario()))
    return tmp_path, e


def run(e, *args, timeout=600):
    return subprocess.run(["bash", str(ROOT / "scripts" / "run_assignment.sh"), *args], env=e, cwd=ROOT,
                          capture_output=True, text=True, timeout=timeout)


def the_run(tmp):
    dirs = sorted((tmp / "results").iterdir())
    assert len(dirs) == 1
    d = dirs[0]
    return d, json.loads((d / "state.json").read_text()), json.loads((d / "requirements.json").read_text())


def reqs(r):
    return {x["id"]: x for x in r["requirements"]}


def items(state, stage):
    return {i["id"]: i for i in state["stages"][stage]["attempts"][-1]["items"]}


def dispatched(tmp):
    log = tmp / "dispatch.log"
    return [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []


def no_unmeasured_pass(r):
    """A memory requirement is PASS only with a numeric per-process peak for every item it names."""
    for x in r["requirements"]:
        if x["id"] in ("M1", "M2", "M3", "M4") and x["status"] == "PASS":
            assert x["measured"] and all(v is not None for v in x["measured"]["process_peak_mib"].values()), x


def test_successful_synthetic_run(env):
    tmp, e = env
    p = run(e)
    d, state, r = the_run(tmp)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert r["verdict"] == "SHIP"
    R = reqs(r)
    for rid in ("T1", "M1", "M2", "M3", "M4", "L2", "L5", "S1", "S2", "V1", "V2", "V3", "V4", "D2", "D3", "D4"):
        assert R[rid]["status"] == "PASS", R[rid]
    assert R["D0"]["status"] == "PASS" and "8/8 rollout batches by DFlash" in R["D0"]["detail"]
    assert R["D0c"]["status"] == "PASS" and "3/3 rollout batches by DFlash" in R["D0c"]["detail"]
    # coverage is measured from the run's own records and no epoch is claimed (8 synthetic updates x 5 prompts)
    assert R["T2"]["status"] == "PASS" and "40/1024 unique prompts visited" in R["T2"]["detail"]
    assert "full pass over the pool: False" in R["T2"]["detail"]
    assert not any(x in json.dumps(a) for a in dispatched(tmp) for x in ("H12.train_extended", "drift_extended"))
    no_unmeasured_pass(r)
    for f in ("manifest.json", "requirements.json", "checksums.sha256", "measurements/memory.csv",
              "measurements/memory_summary.json", "measurements/timing.csv", "measurements/throughput.csv",
              "measurements/correctness.json", "measurements/dflash.json", "measurements/learning.json",
              "comparisons/l2l_equivalence.csv", "comparisons/backend_performance.csv", "comparisons/dflash_vs_plain.csv",
              "comparisons/model_artifacts.csv", "reports/summary.md", "reports/requirements_matrix.md",
              "reports/silent_failures.md", "reports/memory_analysis.md", "reports/experimental_results.md",
              "reports/report_tables.md", "reports/limitations.md", "reports/final_verdict.md", "logs/orchestrator.log",
              "resolved_configs/hashes.json", "environment/capabilities.json"):
        assert (d / f).exists(), f
    smoke = items(state, "D")["H11.smoke"]
    raw = ROOT / smoke["raw_dir"] if not Path(smoke["raw_dir"]).is_absolute() else Path(smoke["raw_dir"])
    apps = (raw / "telemetry" / "nvidia_smi_apps.csv").read_text()
    assert apps.startswith("timestamp, pid, process_name, used_gpu_memory") and "5000 MiB" in apps
    assert smoke["memory_status"] == "PASS" and smoke["telemetry"]["process_peak_mib"] == 5000
    assert not os.access(d / "predictions", os.W_OK)  # predictions are frozen before the GPU stages
    log = (d / "logs" / "orchestrator.log").read_text()
    assert log[:4].isdigit() and "stage F" in log  # UTC-stamped lines
    assert json.loads((d / "measurements" / "dflash.json").read_text())["distribution_checks"][0]["p_value"] > 1e-3


def test_missing_gpu_blocks_gpu_stages_and_runs_none_of_them(env):
    tmp, e = env
    probe = json.loads(Path(e["ASSIGNMENT_PROBE_JSON"]).read_text())
    probe.update(cuda=False, bf16=False, gpu=None, cuda_error="no CUDA device", per_process_smi=False)
    Path(e["ASSIGNMENT_PROBE_JSON"]).write_text(json.dumps(probe))
    p = run(e)
    d, state, r = the_run(tmp)
    assert p.returncode == 1 and r["verdict"] == "DON'T SHIP"
    gpu_entries = {i for s in "CDEFG" for i in items(state, s)}
    assert not any(a[a.index("--entry") + 1] in gpu_entries for a in dispatched(tmp) if "--entry" in a)
    for s in "CDEFG":
        assert state["stages"][s]["status"] == "BLOCKED"
    R = reqs(r)
    assert all(R[m]["status"] in ("BLOCKED", "NOT_TESTED") for m in ("M1", "M2", "M3", "M4", "V1"))
    assert "no usable CUDA GPU" in (d / "reports" / "limitations.md").read_text()


def test_missing_per_process_readings_are_inconclusive_and_block_the_final_run(env):
    tmp, e = env
    e["FAKE_SMI_NA"] = "1"  # nvidia-smi lists the processes with [N/A] memory (WSL2, some containers)
    p = run(e)
    d, state, r = the_run(tmp)
    assert p.returncode == 1
    assert items(state, "D")["H11.smoke"]["memory_status"] == "INCONCLUSIVE"
    assert state["stages"]["F"]["status"] == "BLOCKED"
    R = reqs(r)
    assert R["M1"]["status"] == "INCONCLUSIVE" and R["V1"]["status"] == "BLOCKED"
    no_unmeasured_pass(r)


def test_absent_nvidia_smi_is_never_a_pass(env):
    tmp, e = env
    e["ASSIGNMENT_NVIDIA_SMI"] = str(tmp / "does-not-exist")
    run(e)
    _, _, r = the_run(tmp)
    assert all(x["status"] != "PASS" for x in r["requirements"] if x["id"] in ("M1", "M2", "M3", "M4", "X2"))
    no_unmeasured_pass(r)


def test_memory_violation_stops_the_workload_and_fails(env):
    tmp, e = env
    e["FAKE_SMI_MIB"] = json.dumps({"*": 5000, "H11.smoke": 9000})  # 9000 MiB > 7629 MiB (8e9 B)
    (tmp / "scenario.json").write_text(json.dumps(scenario(H11__smoke={"sleep": 60})))
    t0 = time.time()
    p = run(e)
    d, state, r = the_run(tmp)
    smoke = items(state, "D")["H11.smoke"]
    assert smoke["status"] == "FAIL" and smoke["memory_status"] == "FAIL" and smoke["violation"]
    assert smoke["seconds"] < 30 and smoke["signals_sent"][0][0] == "SIGINT"  # stopped, not run to completion
    assert time.time() - t0 < 300
    assert state["stages"]["E"]["status"] == "BLOCKED" and state["stages"]["F"]["status"] == "BLOCKED"
    assert reqs(r)["M1"]["status"] == "FAIL" and r["verdict"] == "DON'T SHIP" and p.returncode == 1
    raw = ROOT / smoke["raw_dir"] if not Path(smoke["raw_dir"]).is_absolute() else Path(smoke["raw_dir"])
    for f in ("stdout.log", "stderr.log", "telemetry/nvidia_smi_apps.csv", "telemetry/nvidia_smi_gpu.csv",
              "telemetry/telemetry.json", "telemetry/pids.csv"):
        assert (raw / f).exists(), f  # every log of the failed item is kept
    assert "9000 MiB" in (raw / "telemetry" / "nvidia_smi_apps.csv").read_text()


def test_interrupt_then_resume_continues_without_rerunning_finished_stages(env):
    tmp, e = env
    (tmp / "scenario.json").write_text(json.dumps(scenario(H12__train_final={"sleep": 60})))
    p = subprocess.Popen(["bash", str(ROOT / "scripts" / "run_assignment.sh")], env=e, cwd=ROOT, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.time()
    while not any("H12.train_final" in a for a in dispatched(tmp)) and time.time() - t0 < 300:
        time.sleep(0.3)
    time.sleep(1.0)
    os.kill(p.pid, signal.SIGINT)
    out, _ = p.communicate(timeout=120)
    assert p.returncode == 130, out[-3000:]
    d, state, r = the_run(tmp)
    f = state["stages"]["F"]["attempts"][-1]
    assert f["interrupted"] and f["status"] == "INCONCLUSIVE"
    assert "G" not in state["stages"] and r["verdict"] == "DON'T SHIP"  # the report is written anyway
    (tmp / "scenario.json").write_text(json.dumps(scenario()))
    n_b = sum("H01.lab_sync" in a for a in dispatched(tmp))
    p2 = run(e, "--resume")
    assert p2.returncode == 0, p2.stdout[-3000:]
    d2, state, r = the_run(tmp)
    assert d2 == d and r["verdict"] == "SHIP"
    for s in "ABCDE":
        assert len(state["stages"][s]["attempts"]) == 1  # finished stages are not re-run
    assert len(state["stages"]["F"]["attempts"]) == 2
    assert sum("H01.lab_sync" in a for a in dispatched(tmp)) == n_b


def test_resume_refuses_changed_code_and_stale_evidence_is_never_reused(env):
    tmp, e = env
    run(e)
    d, state, _ = the_run(tmp)
    state["identity"]["git_commit"] = "0" * 40  # as if the code had changed since the run started
    (d / "state.json").write_text(json.dumps(state))
    p = run(e, "--resume")
    assert p.returncode == 2 and "refusing to resume" in p.stderr
    # a new run finds the first run's Qwen3-4B results; they name no GPU, so they are not reused, and nothing that
    # depends on them starts
    fresh = {**e, "ASSIGNMENT_RESULTS_ROOT": str(tmp / "results2"), "ASSIGNMENT_EVIDENCE_ROOT": str(tmp / "evidence2")}
    for k in ("results2", "evidence2"):
        (tmp / k).mkdir()
    run(fresh)
    d2 = sorted((tmp / "results2").iterdir())[0]
    state = json.loads((d2 / "state.json").read_text())
    prof = items(state, "D")["H11.profile"]
    assert prof["status"] == "INCONCLUSIVE" and "stale evidence" in prof["detail"]
    assert state["stages"]["F"]["status"] == "BLOCKED"


def test_a_training_run_with_another_config_is_neither_resumed_nor_reused(env):
    tmp, e = env
    m = tmp / "runs" / "qwen3_4b_final" / "manifest.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"config_sha256": "deadbeef" * 8, "status": "interrupted", "git_commit": "x"}))
    run(e)
    _, state, r = the_run(tmp)
    F = items(state, "F")
    assert F["resume:qwen3_4b_final"]["status"] == "FAIL"
    assert "stale configuration" in F["resume:qwen3_4b_final"]["detail"]
    assert F["H12.train_final"]["status"] == "BLOCKED" and r["verdict"] == "DON'T SHIP"


def test_failed_dflash_benchmark_blocks_the_dflash_final_run(env):
    """The final run samples through DFlash, so greedy DFlash != plain decoding on Qwen3-4B must stop it."""
    tmp, e = env
    (tmp / "scenario.json").write_text(json.dumps(scenario(
        H10__dflash_bench={"status": "FAIL", "files": {"t12_dflash_bench.json": bench(equal=False)}})))
    run(e)
    _, state, r = the_run(tmp)
    assert items(state, "E")["H10.dflash_bench"]["status"] == "FAIL"
    assert reqs(r)["D2"]["status"] == "FAIL"
    F = items(state, "F")["H12.train_final"]
    assert F["status"] == "BLOCKED" and "H10.dflash_bench" in F["detail"]
    assert not any("H12.train_final" in a for a in map(json.dumps, dispatched(tmp)))  # never started
    assert reqs(r)["D0"]["status"] in ("BLOCKED", "NOT_TESTED") and r["verdict"] == "DON'T SHIP"


def test_d0_is_never_passed_from_the_configuration_alone(env):
    """dflash.enabled in the resolved final config, the training items PASS, but the logs carry no per-batch rollout
    generator records (e.g. written by an older trainer): D0 / D0c are INCONCLUSIVE, not PASS. A log showing plain
    rollouts despite the config is a FAIL."""
    tmp, e = env
    (tmp / "scenario.json").write_text(json.dumps(scenario(H12__train_final={"status": "PASS", "files": {
        "runs/qwen3_4b_final/training/training.json": training_log(8, dflash=True, record=False),
        "runs/qwen3_4b_final_lr0/training/training.json": training_log(3, dflash=False)}})))
    run(e)
    _, _, r = the_run(tmp)
    R = reqs(r)
    assert R["D0"]["status"] == "INCONCLUSIVE" and "not recorded" in R["D0"]["detail"], R["D0"]
    assert R["D0c"]["status"] == "FAIL" and "without DFlash" in R["D0c"]["detail"], R["D0c"]


def test_d0_fails_when_the_final_training_fails(env):
    tmp, e = env
    (tmp / "scenario.json").write_text(json.dumps(scenario(H12__train_final={"status": "FAIL", "files": {
        "runs/qwen3_4b_final/training/training.json": training_log(8, dflash=True, max_steps=60),
        "runs/qwen3_4b_final_lr0/training/training.json": training_log(3, dflash=True)}})))
    run(e)
    _, _, r = the_run(tmp)
    R = reqs(r)
    assert R["D0"]["status"] == "FAIL" and "training item FAIL" in R["D0"]["detail"], R["D0"]


def test_a_pass_without_its_result_file_is_not_a_measured_pass(env):
    """The tools say PASS but wrote no result (a crash after judging, a lost file): every requirement that needs the
    measurement is INCONCLUSIVE, never PASS."""
    tmp, e = env
    s = scenario()
    for k in ("H11.profile", "H10.dflash_bench", "H10.dflash_drift", "H12.final_eval"):
        s[k] = {"status": "PASS"}
    (tmp / "scenario.json").write_text(json.dumps(s))
    run(e)
    _, _, r = the_run(tmp)
    R = reqs(r)
    for rid in ("P2", "P3", "D1", "D2", "D4", "D5", "D5b", "D6", "V1", "V2", "V3", "V4"):
        assert R[rid]["status"] == "INCONCLUSIVE", R[rid]
    assert r["verdict"] == "DON'T SHIP"


def test_report_with_incomplete_evidence(env):
    tmp, e = env
    d = Path(subprocess.run([sys.executable, str(ROOT / "scripts" / "assignment.py"), "init"], env=e, cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip())
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "assignment.py"), "report", "--run-dir", str(d)], env=e,
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 1, p.stderr
    r = json.loads((d / "requirements.json").read_text())
    assert r["verdict"] == "DON'T SHIP"
    assert all(x["status"] in ("NOT_TESTED", "BLOCKED", "INCONCLUSIVE") for x in r["requirements"]), \
        [(x["id"], x["status"]) for x in r["requirements"] if x["status"] in ("PASS", "FAIL")]
    summary = (d / "reports" / "summary.md").read_text()
    assert "MISSING" in summary and "DON'T SHIP" in summary
    mem = (d / "measurements" / "memory.csv").read_text().splitlines()
    assert mem[0].startswith("stage,item") and all(",0," not in l for l in mem[1:])  # nothing filled with zeros


def test_dry_run_writes_nothing(env):
    tmp, e = env

    def snapshot():
        files = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.name != "dispatch.log")
        git = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, capture_output=True,
                             text=True).stdout
        return files, git
    before = snapshot()
    p = run(e, "--dry-run")
    assert p.returncode == 0, p.stderr
    assert "nothing is written or run" in p.stdout and "H12.train_final" in p.stdout
    assert snapshot() == before
    assert all("--dry-run" in a or a[0] == "plan" for a in dispatched(tmp))  # no experiment was dispatched


def test_status_and_usage(env):
    tmp, e = env
    assert run(e, "--status").returncode == 1  # no run yet
    assert run(e, "--from-stage", "Z").returncode == 2
    assert run(e, "--bogus").returncode == 2


def test_the_default_protocol_refuses_to_run_an_opt_in_entry(env):
    """The 210-update continuation (H12.train_extended) is opt-in: even an override that slips it into a stage list
    is refused before anything runs."""
    tmp, e = env
    e = {**e, "ASSIGNMENT_STAGE_ITEMS": json.dumps({"B": ESSENTIAL_ENTRIES, "F": ["H12.train_final",
                                                                                   "H12.train_extended"]})}
    p = run(e)
    assert p.returncode != 0 and "opt-in entries ['H12.train_extended']" in p.stderr, p.stderr[-2000:]
    assert not any("H12.train_extended" in json.dumps(a) for a in dispatched(tmp))
