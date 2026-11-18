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


# --- --experiment ---------------------------------------------------------------------------------------------------

def experiment_env(env, **over):
    tmp, e = env
    e = {k: v for k, v in e.items() if k != "ASSIGNMENT_STAGE_ITEMS"}  # the mode's own B list
    (tmp / "scenario.json").write_text(json.dumps(scenario(**over)))
    return tmp, e


def stage_order(tmp):
    """The stage of each dispatched entry, in dispatch order, collapsed."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import assignment as A
    where = {i: s for s, ids in A.STAGE_ENTRIES.items() for i in ids}
    where.update({i: s for s, ids in A.EXPERIMENT_STAGE_ENTRIES.items() for i in ids})
    out = []
    for a in dispatched(tmp):
        if "--entry" in a and "--dry-run" not in a:
            s = where[a[a.index("--entry") + 1]]
            if not out or out[-1] != s:
                out.append(s)
    return out


PLAIN_TRAIN = {"H12__train_final_plain": {"status": "PASS", "files": {   # the configured 60 / 30 updates, plain rollouts
    "runs/qwen3_4b_final_plain/training/training.json": training_log(60),
    "runs/qwen3_4b_final_plain_lr0/training/training.json": training_log(30)}},
    "H12__final_eval_plain": {"status": "PASS", "files": {"t13_final_eval_qwen3_4b_final_plain.json": EVAL}}}
DFLASH_REQS = ("D0", "D0c", "D1", "D2", "D3", "D4", "D5", "D5b", "D6")


def test_experiment_mode_runs_A_D_B_F_G_H_without_dflash_waives_only_h02_and_never_ships(env):
    tmp, e = experiment_env(env, H02__fault_unsafe_reuse={"status": "FAIL"}, **PLAIN_TRAIN)
    p = run(e, "--experiment")
    d, state, r = the_run(tmp)
    assert p.returncode == 1, p.stdout[-3000:] + p.stderr[-3000:]
    assert state["mode"] == "experiment"
    assert list(state["stages"]) == ["A", "D", "B", "F", "G", "H"]  # executed in this order; no C, no E
    assert stage_order(tmp) == ["D", "B", "F", "G"]
    # no DFlash item, test, config or drafter anywhere in what was dispatched or resolved
    ran = [a[a.index("--entry") + 1] for a in dispatched(tmp) if "--entry" in a and "--dry-run" not in a]
    assert ran and not any("dflash" in x.lower() for x in ran), ran
    assert "H12.train_final_plain" in ran and "H12.final_eval_plain" in ran and "H12.train_final" not in ran
    assert not any("dflash" in f.name.lower() for f in (d / "resolved_configs").iterdir())
    models = json.loads((d / "environment" / "models_data_seeds.json").read_text())["models"]
    assert "Qwen/Qwen3-4B" in models and not any("dflash" in m.lower() for m in models), models  # no drafter
    B = items(state, "B")
    assert "dflash_tests" not in B and "unit_tests_cpu" not in B and "H01.lab_sync" not in B
    assert not any(i.get("kind") == "pytest" for i in B.values())
    assert B["H02.fault_unsafe_reuse"]["status"] == "FAIL"  # recorded as it is, never relabelled
    assert state["waivers"][0]["item"] == "H02.fault_unsafe_reuse" and state["waivers"][0]["status"] == "FAIL"
    # the essential gates still hold: every non-waived B item, D smoke + profile with PASS per-process memory
    assert items(state, "F")["H12.train_final_plain"]["status"] == "PASS"  # the waiver let the 4B training run
    # the 8 GB cap is enforced on every Qwen3-4B stage (no cap would make memory_status INCONCLUSIVE)
    for s, i in (("D", "H11.smoke"), ("D", "H11.profile"), ("F", "H12.train_final_plain"), ("G", "H12.final_eval_plain")):
        assert items(state, s)[i]["memory_status"] == "PASS", (s, i, items(state, s)[i].get("telemetry"))
    cfg = items(state, "A")["assignment_config"]
    assert cfg["status"] == "PASS", cfg
    assert cfg["checks"]["plain L2L rollouts (dflash disabled, no drafter)"]
    assert "DFlash rollouts (z-lab/Qwen3-4B-DFlash-b16)" not in cfg["checks"]
    assert r["verdict"] == "NO SHIP VERDICT (experiment mode)" and r["mode"] == "experiment"
    R = reqs(r)
    assert R["L3"]["status"] == "FAIL" and R["L2"]["status"] == "NOT_TESTED"  # H01.lab_sync omitted, not passed
    for rid in ("T1", "T2", "T3", "T3c", "M1", "M2", "M3", "M4", "V1", "V2", "V3", "V4"):
        assert R[rid]["status"] == "PASS", R[rid]
    assert "60/60 rollout batches by StreamedGenerator" in R["T3"]["detail"]
    assert "30/30 rollout batches by StreamedGenerator" in R["T3c"]["detail"]
    for rid in DFLASH_REQS:  # intentionally excluded: NOT_TESTED, never FAIL, no fabricated evidence
        assert R[rid]["status"] == "NOT_TESTED" and "intentionally excluded" in R[rid]["detail"], R[rid]
        assert R[rid]["evidence"] == [] and R[rid]["measured"] is None
    no_unmeasured_pass(r)
    con = {c["conclusion"].split(" (")[0]: c for c in r["conclusions"]}
    assert con["training completed"]["status"] == "PASS"
    assert con["memory compliance established"]["status"] == "PASS"
    assert con["performance improvement established"]["status"] == "PASS"
    assert con["DFlash validity established"]["status"] == "NOT_TESTED"
    assert con["full assignment qualification established"]["status"] == "NOT_TESTED"
    assert "known limitation H02.fault_unsafe_reuse FAIL" in con["training correctness established"]["detail"]
    assert r["reuse_banner"] is None
    summary = (d / "reports" / "summary.md").read_text()
    # the limitation is reported as the FAIL it is; the fake wrote no lab result, so the report says it cannot be
    # analysed rather than inventing an analysis. The waiver's own wording is recorded unchanged.
    assert "Known limitation: H02.fault_unsafe_reuse" in summary and "H02.fault_unsafe_reuse is FAIL" in summary
    assert "cannot be analysed" in summary
    sys.path.insert(0, str(ROOT / "scripts"))
    import assignment as A
    assert state["waivers"][0]["waiver"] == A.WAIVED["H02.fault_unsafe_reuse"]
    assert "incomplete" in A.WAIVED["H02.fault_unsafe_reuse"] and list(A.WAIVED) == ["H02.fault_unsafe_reuse"]
    assert r["waivers"] == state["waivers"]
    assert "DFlash was intentionally excluded" in summary
    # the run never claims a SHIP verdict or full qualification (rule text mentioning SHIP is allowed)
    assert summary.startswith("# GRPO + LoRA on Qwen3-4B under an 8 GB cap: NO SHIP VERDICT (experiment mode)")
    assert "**SHIP.**" not in summary and "**NO SHIP VERDICT (experiment mode).**" in summary
    assert "Final verdict: NO SHIP VERDICT (experiment mode)" in (d / "reports" / "final_verdict.md").read_text()
    assert json.loads((d / "measurements" / "dflash.json").read_text())["benchmarks"] == {}
    assert "experiment" in json.loads((d / "manifest.json").read_text())["command"]


def test_a_passing_training_item_is_not_a_completed_training(env):
    """Stage execution success is not training completion: the training item PASSes (exit 0), but the logs record 8
    and 3 updates against the configured 60 and 30, so `training completed` is INCONCLUSIVE, never PASS."""
    tmp, e = experiment_env(env, **{**PLAIN_TRAIN, "H12__train_final_plain": {"status": "PASS", "files": {
        "runs/qwen3_4b_final_plain/training/training.json": training_log(8),
        "runs/qwen3_4b_final_plain_lr0/training/training.json": training_log(3)}}})
    run(e, "--experiment")
    _, state, r = the_run(tmp)
    assert items(state, "F")["H12.train_final_plain"]["status"] == "PASS"
    con = {c["conclusion"].split(" (")[0]: c for c in r["conclusions"]}
    done = con["training completed"]
    assert done["status"] == "INCONCLUSIVE", done
    assert "'qwen3_4b_final_plain': 8" in done["detail"] and "'qwen3_4b_final_plain': 60" in done["detail"], done
    assert r["verdict"] == "NO SHIP VERDICT (experiment mode)"


def test_experiment_mode_never_reuses_training_whose_provenance_does_not_match(env):
    """A completed plain run left by an earlier experiment run is reused only when its own identity matches (GPU,
    experiment code, the config it was trained with). Here its manifest and log name another config hash: the second
    run's resume check FAILs (stale configuration), it is neither marked as prior evidence nor trained over, and the
    training item is BLOCKED."""
    from qwen_grpo.evidence import code_sha256
    ident = {"environment": {"gpu": {"name": "NVIDIA Fake A6000"}}, "code_sha256": code_sha256(ROOT),
             "config_sha256": "0" * 64}
    stale = {"runs/qwen3_4b_final_plain/training/training.json": {**training_log(60), **ident},
             "runs/qwen3_4b_final_plain/manifest.json": {"status": "completed", "config_sha256": "0" * 64},
             "runs/qwen3_4b_final_plain_lr0/training/training.json": {**training_log(30), **ident},
             "runs/qwen3_4b_final_plain_lr0/manifest.json": {"status": "completed", "config_sha256": "0" * 64}}
    tmp, e = experiment_env(env, **{**reusable_scenario(), "H12__train_final_plain": {"files": stale}})
    assert run(e, "--experiment").returncode == 1
    n_train = sum("H12.train_final_plain" in a for a in dispatched(tmp))
    run(e, "--experiment")
    second = sorted((tmp / "results").iterdir())[-1]
    Fs = items(json.loads((second / "state.json").read_text()), "F")
    for name in ("qwen3_4b_final_plain", "qwen3_4b_final_plain_lr0"):
        res = Fs[f"resume:{name}"]
        assert res["status"] == "FAIL" and "stale configuration" in res["detail"], res
        assert "refusing to resume or reuse it" in res["detail"]
    F = Fs["H12.train_final_plain"]
    assert F["status"] == "BLOCKED" and not F.get("prior_evidence"), F
    assert sum("H12.train_final_plain" in a for a in dispatched(tmp)) == n_train  # not retrained silently
    assert json.loads((second / "requirements.json").read_text())["reuse_banner"] is None


def test_experiment_mode_reports_dflash_rollouts_in_a_plain_run_as_a_failure(env):
    """T3 is judged from the training logs, not the config: a DFlash batch in the plain-L2L run is a FAIL."""
    tmp, e = experiment_env(env, **{**PLAIN_TRAIN, "H12__train_final_plain": {"status": "PASS", "files": {
        "runs/qwen3_4b_final_plain/training/training.json": training_log(60, dflash=True),
        "runs/qwen3_4b_final_plain_lr0/training/training.json": training_log(30, record=False)}}})
    run(e, "--experiment")
    _, _, r = the_run(tmp)
    R = reqs(r)
    assert R["T3"]["status"] == "FAIL" and "DFlash in a plain-L2L run" in R["T3"]["detail"], R["T3"]
    assert R["T3c"]["status"] == "INCONCLUSIVE" and "not recorded" in R["T3c"]["detail"], R["T3c"]


def test_experiment_mode_a_failed_essential_correctness_check_blocks_training(env):
    tmp, e = experiment_env(env, H02__lab_async_cuda={"status": "FAIL"}, **PLAIN_TRAIN)
    p = run(e, "--experiment")
    d, state, r = the_run(tmp)
    assert p.returncode == 1
    assert state["stages"]["F"]["status"] == "BLOCKED" and "H02.lab_async_cuda" in json.dumps(state["stages"]["F"])
    assert not any("H12.train_final" in a for a in dispatched(tmp))
    assert r["verdict"] == "NO SHIP VERDICT (experiment mode)"


def test_experiment_mode_a_failed_smoke_or_memory_blocks_training(env):
    """D's gate is unchanged: the 8 GB watchdog stops a smoke run above the cap, and F does not start."""
    tmp, e = experiment_env(env, H11__smoke={"sleep": 60}, **PLAIN_TRAIN)
    e["FAKE_SMI_MIB"] = json.dumps({"*": 5000, "H11.smoke": 9000})  # 9000 MiB > 7629 MiB (8e9 B)
    run(e, "--experiment")
    _, state, r = the_run(tmp)
    smoke = items(state, "D")["H11.smoke"]
    assert smoke["status"] == "FAIL" and smoke["memory_status"] == "FAIL" and smoke["violation"]
    assert state["stages"]["F"]["status"] == "BLOCKED" and "H11.smoke" in json.dumps(state["stages"]["F"])
    assert not any("H12.train_final" in a for a in dispatched(tmp))
    assert reqs(r)["M1"]["status"] == "FAIL"


def test_experiment_mode_resumes_only_a_plain_run_of_the_same_configuration(env):
    """F's crash-consistent resume works on the plain runs as in full mode; a plain run with another config (or a
    DFlash-trained qwen3_4b_final) is never resumed into the experiment's runs."""
    tmp, e = experiment_env(env, **PLAIN_TRAIN)
    m = tmp / "runs" / "qwen3_4b_final_plain" / "manifest.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"config_sha256": "deadbeef" * 8, "status": "interrupted", "git_commit": "x"}))
    other = tmp / "runs" / "qwen3_4b_final" / "manifest.json"  # a full-mode (DFlash) run: never touched here
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps({"config_sha256": "abc", "status": "interrupted"}))
    run(e, "--experiment")
    _, state, _ = the_run(tmp)
    F = items(state, "F")
    assert F["resume:qwen3_4b_final_plain"]["status"] == "FAIL"
    assert "stale configuration" in F["resume:qwen3_4b_final_plain"]["detail"]
    assert F["H12.train_final_plain"]["status"] == "BLOCKED"
    assert not any("qwen3_4b_final:" in k or k == "resume:qwen3_4b_final" for k in F)
    assert json.loads(other.read_text()) == {"config_sha256": "abc", "status": "interrupted"}


def test_experiment_gate_refuses_a_dflash_primary_config():
    """Experiment mode has no stage E, so F refuses to start if its primary config enabled DFlash."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("assignment_gate_probe", ROOT / "scripts" / "assignment.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    A.install_experiment()
    assert (A.STAGES, A.PRIMARY, A.CONTROL, A.TRAIN_ID, A.EVAL_ID) == (
        "ADBFGH", "qwen3_4b_final_plain", "qwen3_4b_final_plain_lr0", "H12.train_final_plain", "H12.final_eval_plain")
    assert "dflash_tests" not in A.ESSENTIAL_B and A.BUDGET_CONFIGS == ("configs/budget/qwen3-4b_final.yaml",)
    assert A.final_uses_dflash() is False
    A.final_uses_dflash = lambda: True
    passing = {"stages": {s: {"attempts": [{"items": [
        {"id": i, "status": "PASS", "memory_status": "PASS"} for i in ids]}]} for s, ids in A.STAGE_ENTRIES.items()}}
    passing["stages"]["A"] = {"attempts": [{"facts": {"cuda": True, "bf16": True, "cuda_streams_ok": True,
                                                       "ram_ok": True, "disk_ok": True},
                                             "items": [{"id": "assignment_config", "status": "PASS"}]}]}
    why = A.gate("F", passing)
    assert len(why) == 1 and "enables DFlash" in why[0], why


def load_runner(name):
    """A fresh copy of scripts/assignment.py (install_experiment rebinds module globals)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "assignment.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    return A


@pytest.mark.parametrize("mode", ["full", "experiment"])
def test_a_failed_or_missing_configuration_check_blocks_training(mode):
    """F starts only if stage A's assignment_config is PASS, in both modes; every other gate is unchanged (a state
    where everything else passes opens F only with a PASS configuration check)."""
    A = load_runner(f"assignment_config_gate_{mode}")
    if mode == "experiment":
        A.install_experiment()
    facts = {"cuda": True, "bf16": True, "cuda_streams_ok": True, "ram_ok": True, "disk_ok": True}

    def state(config_item):
        s = {"stages": {st: {"attempts": [{"items": [{"id": i, "status": "PASS", "memory_status": "PASS"}
                                                     for i in ids]}]} for st, ids in A.STAGE_ENTRIES.items()}}
        s["stages"]["B"]["attempts"][0]["items"] += [{"id": i, "status": "PASS"} for i in A.ESSENTIAL_B]
        s["stages"]["A"] = {"attempts": [{"facts": facts, "items": [config_item] if config_item else []}]}
        return s
    assert A.gate("F", state({"id": "assignment_config", "status": "PASS"})) == []
    bad = {"id": "assignment_config", "status": "FAIL", "detail": "violated: L2L backend"}
    why = A.gate("F", state(bad))
    assert why == ["stage A: assignment_config is FAIL (violated: L2L backend)"], why
    assert A.gate("F", state(None)) == ["stage A: assignment_config is None"]  # stage A never ran the check
    # the other gates still apply on top of it
    s = state(bad)
    s["stages"]["D"]["attempts"][0]["items"][0]["memory_status"] = "FAIL"
    assert len(A.gate("F", s)) == 2


def test_training_is_not_dispatched_after_a_failed_configuration_check(env):
    """End to end (experiment mode, fake tools): stages A, D and B pass, then the recorded configuration check is
    FAIL; stage F is BLOCKED with that reason and no training is dispatched."""
    tmp, e = experiment_env(env, **PLAIN_TRAIN)
    e = {**e, "ASSIGNMENT_MODE": "experiment"}
    py = [sys.executable, str(ROOT / "scripts" / "assignment.py")]
    d = Path(subprocess.run(py + ["init"], env=e, cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
    for st in "ADB":
        subprocess.run(py + ["stage", st, "--run-dir", str(d)], env=e, cwd=ROOT, capture_output=True, text=True,
                       check=True, timeout=600)
    state = json.loads((d / "state.json").read_text())
    cfg = items(state, "A")["assignment_config"]
    assert cfg["status"] == "PASS"  # the real configs pass; the failure is injected into the recorded result
    cfg.update(status="FAIL", detail="violated: plain L2L rollouts (dflash disabled, no drafter)")
    (d / "state.json").write_text(json.dumps(state))
    subprocess.run(py + ["stage", "F", "--run-dir", str(d)], env=e, cwd=ROOT, capture_output=True, text=True,
                   check=True, timeout=600)
    state = json.loads((d / "state.json").read_text())
    F = state["stages"]["F"]
    assert F["status"] == "BLOCKED" and "stage A: assignment_config is FAIL" in json.dumps(F["attempts"][-1]["blocked_by"])
    assert items(state, "F")["H12.train_final_plain"]["status"] == "BLOCKED"
    assert not any("H12.train_final" in json.dumps(a) for a in dispatched(tmp))


def test_experiment_dry_run_shows_the_plain_4b_plan_and_writes_nothing(env):
    tmp, e = experiment_env(env)
    p = run(e, "--experiment", "--dry-run")
    assert p.returncode == 0, p.stderr[-2000:]
    out = p.stdout
    assert "mode: experiment; stage order: A D B F G H" in out and "8000000000 bytes" in out
    assert "qwen3_4b_final_plain: Qwen/Qwen3-4B bfloat16, 60 updates, B=5 G=6, 4 new tokens, lr 0.0002" in out
    assert "qwen3_4b_final_plain_lr0: Qwen/Qwen3-4B bfloat16, 30 updates, B=5 G=6, 4 new tokens, lr 0.0" in out
    assert "micro-batch 6, 1 iteration(s), eval every 20 on 128, pool 1024" in out
    assert "disk streaming True, DFlash False (None), memory guard True" in out
    assert "z-lab/Qwen3-4B-DFlash-b16" not in out
    assert "unit_tests_cpu" not in out and "tests/test_grpo.py" not in out and "H05.real_0.6b_bf16" not in out
    assert "tests/test_dflash" not in out and "H10." not in out and "E dflash" not in out
    assert "H12.train_final_plain" in out and "H12.final_eval_plain" in out and "H11.smoke" in out
    assert "budget predictions configs/budget/qwen3-4b_final.yaml ->" in out
    assert not list((tmp / "results").iterdir()) and all("--dry-run" in a or a[0] == "plan" for a in dispatched(tmp))


def identity_files(name, cfg, steps, dflash=False):
    """A completed run as the fake leaves it, with the provenance foreign_check reads (same GPU, code and config)."""
    from qwen_grpo import config
    from qwen_grpo.evidence import code_sha256
    sha = config.sha256(config.load(ROOT / cfg))
    log = {**training_log(steps, dflash=dflash), "environment": {"gpu": {"name": "NVIDIA Fake A6000"}},
           "code_sha256": code_sha256(ROOT), "config_sha256": sha}
    return {f"runs/{name}/training/training.json": log,
            f"runs/{name}/manifest.json": {"status": "completed", "config_sha256": sha}}


def reusable_scenario(**over):
    from qwen_grpo.evidence import code_sha256
    c, ident = "configs/experiments/", {"gpu": "NVIDIA Fake A6000", "code_sha256": code_sha256(ROOT)}
    return {"H11__profile": {"files": {"t11_profile_qwen3_4b_async.json": {**PROFILE, **ident}}},
            "H10__dflash_bench": {"files": {"t12_dflash_bench.json": {**bench(), **ident}}},
            "H11__smoke": {"files": identity_files("qwen3_4b_smoke", c + "qwen3_4b_smoke.yaml", 4)},
            "H10__smoke_dflash": {"files": identity_files("qwen3_4b_smoke_dflash", c + "qwen3_4b_smoke_dflash.yaml", 4,
                                                          True)},
            "H12__train_final": {"files": {**identity_files("qwen3_4b_final", c + "qwen3_4b_final.yaml", 8, True),
                                           **identity_files("qwen3_4b_final_lr0", c + "qwen3_4b_final_lr0.yaml", 3,
                                                            True)}},
            "H12__train_final_plain": {"files": {
                **identity_files("qwen3_4b_final_plain", c + "qwen3_4b_final_plain.yaml", 60),
                **identity_files("qwen3_4b_final_plain_lr0", c + "qwen3_4b_final_plain_lr0.yaml", 30)}}, **over}


def test_modes_resume_separately_and_a_full_run_never_reuses_experiment_training(env):
    tmp, e = experiment_env(env, **reusable_scenario())
    assert run(e, "--experiment").returncode == 1
    [exp_dir] = sorted((tmp / "results").iterdir())
    e_full = {**e, "ASSIGNMENT_STAGE_ITEMS": json.dumps({"B": ESSENTIAL_ENTRIES})}
    p = run(e_full)
    full_dir = sorted((tmp / "results").iterdir())[-1]
    assert full_dir != exp_dir and json.loads((full_dir / "state.json").read_text())["mode"] == "full"
    smoke = items(json.loads((full_dir / "state.json").read_text()), "D")["H11.smoke"]
    assert smoke["status"] == "INCONCLUSIVE" and "never reuses training from another mode" in smoke["detail"], smoke
    assert json.loads((full_dir / "requirements.json").read_text())["verdict"] == "DON'T SHIP"
    # --experiment --resume continues the experiment run although a full run is newer; --resume the full one
    p = run(e, "--experiment", "--resume")
    assert p.returncode == 1, p.stderr[-2000:]
    assert json.loads((exp_dir / "state.json").read_text())["history"][-1]["event"] == "resume"
    p = run(e_full, "--resume")
    assert json.loads((full_dir / "state.json").read_text())["history"][-1]["event"] == "resume"
    assert run(e, "--experiment", "--full-tests").returncode == 2
    assert run(e, "--fresh", "--resume").returncode == 2


def test_reused_training_is_announced_and_fresh_moves_it_aside_and_retrains(env):
    tmp, e = experiment_env(env, **reusable_scenario())
    assert run(e, "--experiment").returncode == 1
    n_train = sum("H12.train_final_plain" in a for a in dispatched(tmp))
    p = run(e, "--experiment")  # a second experiment run: the completed runs pass the provenance check
    second = sorted((tmp / "results").iterdir())[-1]
    r = json.loads((second / "requirements.json").read_text())
    F = items(json.loads((second / "state.json").read_text()), "F")["H12.train_final_plain"]
    assert F["prior_evidence"] is True, F
    assert r["reuse_banner"] == "REUSED PRIOR EVIDENCE — NO NEW TRAINING EXECUTED"
    assert "REUSED PRIOR EVIDENCE — NO NEW TRAINING EXECUTED" in (second / "reports" / "summary.md").read_text()
    assert "REUSED PRIOR EVIDENCE" in p.stdout
    p = run(e, "--experiment", "--fresh")
    third = sorted((tmp / "results").iterdir())[-1]
    man = json.loads((third / "manifest.json").read_text())
    moved = {m["from"] for m in man["archived_by_fresh"]}
    assert any(x.endswith("runs/qwen3_4b_final_plain") for x in moved) and not any("dflash" in x for x in moved)
    assert all((tmp / "runs" / n).exists() for n in ("qwen3_4b_final_plain", "qwen3_4b_smoke"))  # trained again
    assert len(list((tmp / "runs").glob("qwen3_4b_final_plain.archived-*"))) == 1  # nothing deleted
    assert sum("H12.train_final_plain" in a for a in dispatched(tmp)) == n_train + 2
    F = items(json.loads((third / "state.json").read_text()), "F")["H12.train_final_plain"]
    assert F["status"] == "PASS" and not F["prior_evidence"]
    assert json.loads((third / "requirements.json").read_text())["reuse_banner"] is None
