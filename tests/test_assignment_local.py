"""The LOCAL laptop validation (scripts/run_assignment_local.sh -> scripts/assignment_local.py, report
scripts/assignment_local_report.py) on Qwen3-0.6B: configs, study, gates and report.

Fast checks: the local configs carry the required settings with DFlash explicitly off and pair up for bitwise
comparisons; the local study reuses the 4B study's tiny correctness entries verbatim and names only local runs and
results; no Qwen3-4B or DFlash entry is reachable from it. End to end with the fakes of tests/test_assignment.py (fake
dispatcher, tools and nvidia-smi; every number synthetic): a passing run, a correctness failure that keeps training
from starting, a streaming-vs-memory difference that fails, and --dry-run / --status writing nothing. CPU only."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from qwen_grpo import config
from tests.test_assignment import FAKES, PROFILE, dispatched

ROOT = Path(__file__).parents[1]
EXP = ROOT / "configs" / "experiments"
LOCAL = {"streaming": "qwen3_0.6b_local_streaming.yaml", "l2l_async": "qwen3_0.6b_local_l2l_async.yaml",
         "resident": "qwen3_0.6b_local_resident.yaml", "smoke": "qwen3_0.6b_local_smoke.yaml",
         "resume": "qwen3_0.6b_local_resume.yaml"}
STUDY = ROOT / "configs" / "studies" / "assignment_local.yaml"


def load(k):
    return config.load(EXP / LOCAL[k])


def diff(a, b):
    return set(config.differences(config.as_dict(load(a)), config.as_dict(load(b))))


# --------------------------------------------------------------------------------------------------- configs

@pytest.mark.parametrize("k", sorted(LOCAL))
def test_local_configs_carry_the_required_settings_with_dflash_off(k):
    c = load(k)
    t = c.training
    assert (c.model.name, c.model.dtype) == ("Qwen/Qwen3-0.6B", "bfloat16")
    assert (t.batch_size, t.group_size, t.max_new_tokens, t.learning_rate, t.clip_eps) == (5, 6, 4, 5e-4, 0.2)
    modules, layers = c.lora.spec()  # the preset last2-attention resolves to q/v/o of the last two layers
    assert (c.lora.rank, c.lora.alpha, sorted(modules), layers) == (32, 64, ["o_proj", "q_proj", "v_proj"], 2)
    assert c.dflash == config.DFlash() and c.memory.guard and config.unimplemented(c) == []
    assert c.run.name.startswith("local_")


def test_local_schedule_and_execution():
    assert (load("smoke").training.max_steps, load("streaming").training.max_steps) == (2, 10)
    for k in ("streaming", "l2l_async", "resident", "smoke"):
        assert load(k).evaluation.every == 5
    s = load("streaming").execution
    assert (s.backend, s.weight_storage, s.weight_buffers, s.prefetch_depth, s.activation_checkpointing) == \
        ("l2l_async", "pinned", 2, 1, True)
    assert load("resident").execution.backend == "resident"
    assert load("streaming").streaming.enabled and not load("l2l_async").streaming.enabled


def test_local_configs_pair_up_for_bitwise_comparisons():
    """compare_runs.py accepts run/execution/memory differences (+ streaming with --allow-differ streaming)."""
    assert {d.split(".")[0] for d in diff("resident", "l2l_async")} == {"run", "execution"}
    assert diff("l2l_async", "streaming") == {"run.name", "streaming.enabled", "streaming.directory"}


def test_the_local_config_check_passes_on_the_resolved_configs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import assignment_local as L
    import assignment as A
    configs = A.resolved_configs(list(L.LOCAL_RUNS.values()))
    r = L.local_config_check(configs)
    assert r["status"] == "PASS", r


# ------------------------------------------------------------------------------------------------------ study

def test_the_local_study_reuses_tiny_entries_and_names_only_local_runs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("experiment", ROOT / "scripts" / "experiment.py")
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    study, _ = X.load_study(STUDY)
    main = {e["id"]: e for e in yaml.safe_load((ROOT / "configs" / "studies" / "assignment.yaml").read_text())["entries"]}
    raw = yaml.safe_load(STUDY.read_text())
    for i in raw["include"]["entries"]:
        assert next(e for e in study["entries"] if e["id"] == i) == main[i]  # verbatim
        assert main[i]["model"] in ("tiny", "Qwen3-0.6B")
    for e in raw["entries"]:
        assert e["model"] == "Qwen3-0.6B" and e["name"].startswith("local_")
        argv = " ".join(" ".join(s["argv"]) for s in X.steps(e))
        assert "4b" not in argv.lower() and "dflash" not in argv.lower(), e["id"]
        for c in e.get("configs", []):
            assert config.load(ROOT / c).run.name.startswith("local_")
    assert yaml.safe_load(STUDY.read_text())["gpu_memory_cap"]["bytes"] == 4 * 2**30


def test_the_local_stages_never_reach_a_qwen3_4b_or_dflash_entry():
    sys.path.insert(0, str(ROOT / "scripts"))
    import assignment_local as L
    ids = {i for v in L.LOCAL_STAGE_ENTRIES.values() for i in v}
    assert not any("dflash" in i.lower() or i.startswith(("H10.", "H11.smoke", "H11.profile", "H12.train", "H12.final"))
                   for i in ids)
    assert not any("dflash" in p for p in L.LOCAL_PYTEST_CPU + L.LOCAL_PYTEST_CUDA)


# ------------------------------------------------------------------------------------------ end to end (fakes)

def lab(quantity="logits"):
    return {"status": "PASS", "execution_status": "PASS", "hypothesis_result": "SUPPORTED",
            "provenance": {"gpu": "NVIDIA Fake 3050 Ti"},
            "comparisons": [{"quantity": quantity, "max_abs": 0.0, "bound": 0.0, "pass": True, "dtype": "torch.bfloat16",
                             "shape": [3, 10, 64]}]}


def log(steps, change=0.05):
    return {"training": [{"step": s, "attempt": s, "prompt_ids": list(range(5 * s, 5 * s + 5)), "mean_reward": 0.5,
                          "grad_norm": 0.1, "update_norm": 0.01, "change_from_init": change * s,
                          "effective_adapter_norm": 0.02 * s, "peak_reserved_gib": 1.2} for s in range(1, steps + 1)],
            "timings": [{"step": s, "rollout_s": 1.0, "prepare_s": 0.3, "update_s": 0.6, "step_s": 2.0,
                         "backend": "l2l_async"} for s in range(1, steps + 1)],
            "zero_signal": [], "successful_steps": steps, "groups_generated": steps, "zero_signal_groups": 0,
            "frozen_changed": [], "params_equal_init": False, "final_checkpoint": {"path": "x"},
            "config": {"steps": steps, "train_samples": 64, "G": 6}}


REAL_STEP = {"rollouts_equal": True, "all_equal": True, "resident_generation": {"seconds": 1.0},
             "l2l_generation": {"seconds": 2.0},
             "comparison": {"old_logp": {"max_abs": 0.0, "equal": True},
                            "iterations": [{"loss": {"equal": True}, "all_grads_equal": True, "worst_grad_rel": 0.0,
                                            "all_updates_equal": True, "clip_frac_equal": True}] * 2}}


def scenario(**over):
    study = yaml.safe_load(STUDY.read_text())
    main = {e["id"]: e for e in yaml.safe_load((ROOT / "configs" / "studies" / "assignment.yaml").read_text())["entries"]}
    s = {"*": {"status": "PASS", "sleep": 0.3}}
    for i in study["include"]["entries"]:
        if ".lab_" in i:
            s[i] = {"status": "PASS", "files": {f"{main[i]['name']}/result.json": lab()}}
    s.update({
        "H05.local_real_step_0.6b": {"status": "PASS", "files": {"local_real_step_0.6b.json": REAL_STEP}},
        "H05.local_runs_0.6b": {"status": "PASS", "files": {
            "runs/local_qwen3_0.6b_resident/training/training.json": log(10),
            "runs/local_qwen3_0.6b_l2l_async/training/training.json": log(10)}},
        "H11.local_profile_0.6b": {"status": "PASS", "files": {"local_profile_0.6b.json": {  # as under WSL2
            **PROFILE, "verdict": "INCOMPLETE_EVIDENCE", "profile": {**PROFILE["profile"], "evidence": "diagnostic"}}}},
        "H12.local_smoke_0.6b": {"status": "PASS", "files": {"runs/local_qwen3_0.6b_smoke/training/training.json": log(2)}},
        "H12.local_train_0.6b": {"status": "PASS", "files": {
            "runs/local_qwen3_0.6b_streaming/training/training.json": log(10)}},
        "H08.local_resume_0.6b": {"status": "PASS", "files": {"local_resume_0.6b.json": {
            "pass": True, "B": {"update@5": {"equal": True}}, "A_epoch_problems": []}}}})
    for k, v in over.items():
        s[k.replace("__", ".")] = {**s.get(k.replace("__", "."), {"status": "PASS"}), **v}
    return s


@pytest.fixture
def env(tmp_path):
    for d in ("results", "evidence", "exp", "runs"):
        (tmp_path / d).mkdir()
    probe = {"cuda": True, "bf16": True, "gpu": "NVIDIA Fake 3050 Ti (8.6)", "capability": "8.6", "vram_gib": 4.0,
             "per_process_smi": False, "linux_native": False, "fixture_packages": True, "host_ram_gib": 16,
             "packages": {"torch": "fake"}}
    (tmp_path / "probe.json").write_text(json.dumps(probe))
    fake = f"{sys.executable} {FAKES / 'fake_tool.py'}"
    e = {**os.environ, "ASSIGNMENT_RESULTS_ROOT": str(tmp_path / "results"),
         "ASSIGNMENT_EVIDENCE_ROOT": str(tmp_path / "evidence"), "ASSIGNMENT_EXPERIMENTS_ROOT": str(tmp_path / "exp"),
         "ASSIGNMENT_RUNS_ROOT": str(tmp_path / "runs"), "ASSIGNMENT_PROBE_JSON": str(tmp_path / "probe.json"),
         "ASSIGNMENT_STUDY": str(STUDY),
         "ASSIGNMENT_DISPATCHER": f"{sys.executable} {FAKES / 'fake_dispatcher.py'}",
         "ASSIGNMENT_NVIDIA_SMI": str(FAKES / "fake_nvidia_smi.py"), "ASSIGNMENT_GRACE_S": "2",
         "ASSIGNMENT_TOOL_OVERRIDES": json.dumps({k: fake for k in ("cuda_check", "budget", "pytest", "compare_runs",
                                                                     "run_py")}),
         "FAKE_SMI_MARKER": "assignment_fakes", "FAKE_SMI_MIB": json.dumps({"*": 1500}),
         "FAKE_LOG": str(tmp_path / "dispatch.log"), "FAKE_SCENARIO": str(tmp_path / "scenario.json"),
         "FAKE_PREDICTION": "{}", "PYTHON": sys.executable, "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ["PATH"]}
    e.pop("SOUP_BIN", None)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario()))
    return tmp_path, e


def run(e, *args):
    return subprocess.run(["bash", str(ROOT / "scripts" / "run_assignment_local.sh"), *args], env=e, cwd=ROOT,
                          capture_output=True, text=True, timeout=900)


def result(tmp):
    [d] = sorted((tmp / "results").iterdir())
    req = json.loads((d / "requirements.json").read_text())
    return d, json.loads((d / "state.json").read_text()), {r["id"]: r for r in req["requirements"]}, req


def test_a_passing_local_run(env):
    tmp, e = env
    p = run(e)
    d, state, R, req = result(tmp)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert req["verdict"] == "LOCAL VALIDATION PASS" and "does NOT verify DFlash" in req["scope"]
    for rid in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L11", "L14", "L15", "L16", "L17"):
        assert R[rid]["status"] == "PASS", R[rid]
    assert R["L12"]["status"] == "INCONCLUSIVE" and "WSL2" in R["L12"]["detail"]  # no per-process nvidia-smi
    for rid in ("X1", "X2", "X3", "X4"):
        assert R[rid]["status"] == "NOT_TESTED"
    assert set(state["stages"]) == set("ABCDF")
    calls = [json.dumps(a) for a in dispatched(tmp)]
    assert not any("H10." in c or "H11.smoke" in c or "H12.train_final" in c for c in calls)
    order = [c for c in calls if "H12.local" in c or "H08.local" in c]
    assert "H12.local_smoke_0.6b" in order[0]  # the smoke runs first
    report = (d / "reports" / "local_report.md").read_text()
    assert "does not verify DFlash or the Qwen3-4B assignment" in report
    for f in ("measurements/memory.csv", "measurements/timing.csv", "comparisons/equivalence.csv",
              "comparisons/backend_performance.csv", "checksums.sha256", "logs/orchestrator.log"):
        assert (d / f).exists(), f
    st = subprocess.run(["bash", str(ROOT / "scripts" / "run_assignment_local.sh"), "--status"], env=e, cwd=ROOT,
                        capture_output=True, text=True, timeout=300)
    assert "F training_0.6b" in st.stdout and "LOCAL VALIDATION PASS" in st.stdout


def test_a_correctness_failure_keeps_training_from_starting(env):
    tmp, e = env
    (tmp / "scenario.json").write_text(json.dumps(scenario(**{"H05__local_real_step_0.6b": {"status": "FAIL", "files": {
        "local_real_step_0.6b.json": {**REAL_STEP, "all_equal": False}}}})))
    p = run(e)
    d, state, R, req = result(tmp)
    assert p.returncode == 1 and req["verdict"] == "LOCAL VALIDATION FAIL"
    F = {i["id"]: i for i in state["stages"]["F"]["attempts"][-1]["items"]}
    assert all(i["status"] == "BLOCKED" for i in F.values()) and "H05.local_real_step_0.6b" in json.dumps(F)
    assert not any("H12.local" in json.dumps(a) or "H08.local" in json.dumps(a) for a in dispatched(tmp))
    assert R["L7"]["status"] == "FAIL" and R["L15"]["status"] == "BLOCKED"


def test_a_streaming_run_that_differs_from_the_in_memory_run_fails(env):
    tmp, e = env
    p = run({**e, "FAKE_COMPARE_DIFFER": "1"})
    _, state, R, req = result(tmp)
    assert req["verdict"] == "LOCAL VALIDATION FAIL"
    assert R["L8"]["status"] == "FAIL"  # resident vs L2L differ -> C fails ...
    F = {i["id"]: i for i in state["stages"]["F"]["attempts"][-1]["items"]}
    assert F["H12.local_train_0.6b"]["status"] == "BLOCKED"  # ... and training does not start


def test_dry_run_writes_nothing(env):
    tmp, e = env
    p = run(e, "--dry-run")
    assert p.returncode == 0, p.stderr[-2000:]
    assert "LOCAL laptop validation" in p.stdout and "DFlash" in p.stdout and "H audit" not in p.stdout
    assert not list((tmp / "results").iterdir()) and not list((tmp / "runs").iterdir())
    assert all("--dry-run" in a for a in dispatched(tmp))
