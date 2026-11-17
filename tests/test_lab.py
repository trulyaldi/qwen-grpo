"""L2L lab (qwen_grpo/lab.py): acceptance arithmetic, statuses, fail-closed behaviour, records. Results go to tmp_path."""
import json

import pytest
import torch

from qwen_grpo import lab
from qwen_grpo.execution import fixtures as F
from scripts import l2l_lab

RESIDENT = lab.ROOT / "research" / "l2l" / "configs" / "resident.yaml"
L2L_SYNC = lab.ROOT / "research" / "l2l" / "configs" / "l2l_sync.yaml"
TOL0 = {"atol": 0.0, "rtol": 0.0}


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)  # tests/test_grpo.py sets float64 at import


# --- acceptance arithmetic -----------------------------------------------------------------------------------------

def test_bound_is_inclusive_and_one_ulp_above_fails():
    ref = torch.tensor([2.0, -4.0], dtype=torch.float64)
    tol = {"atol": 1e-3, "rtol": 1e-2}                                  # bound = 1e-3 + 1e-2 * 4
    bound = 1e-3 + 1e-2 * 4.0
    at = ref.clone(); at[0] += bound
    assert lab.compare("x", ref, at, tol)["pass"]
    above = ref.clone(); above[0] = torch.nextafter(at[0], torch.tensor(float("inf"), dtype=torch.float64))
    assert not lab.compare("x", ref, above, tol)["pass"]


def test_zero_reference_has_no_relative_error_and_require_nonzero_fails():
    zero = torch.zeros(3)
    row = lab.compare("clip_frac", zero, zero.clone(), TOL0)
    assert row["pass"] and row["rel"] is None and row["rel_l2"] is None
    assert not lab.compare("grads.w", zero, zero.clone(), {**TOL0, "require_nonzero": True})["pass"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("side", ["ref", "got"])
def test_nonfinite_values_fail_whatever_the_tolerance(bad, side):
    x, y = torch.ones(4), torch.ones(4)
    (x if side == "ref" else y)[1] = bad
    row = lab.compare("t", x, y, {"atol": 1e30, "rtol": 1e30})
    assert not row["pass"] and row["reason"] == "non-finite values"


@pytest.mark.parametrize("got", [torch.ones(3, 2), torch.ones(2, 3, dtype=torch.float64), None])
def test_shape_dtype_mismatch_and_missing_tensor_fail_before_any_metric(got):
    row = lab.compare("t", torch.ones(2, 3), got, {"atol": 1e30, "rtol": 1e30})
    assert not row["pass"] and "max_abs" not in row


def test_missing_and_extra_candidate_tensors_fail_and_unthresholded_quantities_are_refused():
    ref = {"logits": torch.ones(2), "grads.a": torch.ones(2)}
    rows = lab.compare_all(ref, {"logits": torch.ones(2), "grads.b": torch.ones(2)}, {"logits": TOL0, "grads": TOL0})
    assert [r["pass"] for r in rows] == [True, False, False]                 # grads.a missing, grads.b unexpected
    with pytest.raises(lab.SetupError, match="no tolerance for quantity 'loss'"):
        lab.compare_all({"loss": torch.ones(())}, {"loss": torch.ones(())}, {})


# --- statuses ------------------------------------------------------------------------------------------------------

def base_result(kind="correctness", **kw):
    return {"kind": kind, "errors": [], "required_measurements": [], **kw}


def test_an_experiment_without_measurements_cannot_pass():
    assert lab.decide(base_result(comparisons=[], reference_integrity=[]))[0] == "FAIL"
    assert lab.decide(base_result(comparisons=[{"name": "x", "pass": True}]))[0] == "FAIL"   # integrity unchecked
    assert lab.decide(base_result("performance", timing={}))[0] == "FAIL"
    few = {"reference": {"trials_s": [0.1] * (lab.MIN_TRIALS - 1)}}
    assert lab.decide(base_result("performance", timing=few))[0] == "FAIL"


def test_missing_gpu_memory_fails_a_case_that_requires_it():
    timing = {"reference": {"trials_s": [0.1] * lab.MIN_TRIALS}}
    cpu_only = {"kind": "snapshot", "cpu": {"rss_mib": 1.0}, "gpu": None}
    allocator_only = {"gpu": {"allocated_mib": 1, "reserved_mib": 1, "peak_allocated_mib": 1, "peak_reserved_mib": 1}}
    smi_only = {"gpu": {**allocator_only["gpu"], "device_used_mib_nvidia_smi": 134.0, "process_used_mib_nvidia_smi": 20.0}}
    complete = {"gpu": {**allocator_only["gpu"], "device_used_mib_mem_get_info": 300.0}}
    for mem, status in [(None, "FAIL"), (cpu_only, "FAIL"), (allocator_only, "FAIL"), (smi_only, "FAIL"),
                        (complete, "PASS")]:
        r = base_result("performance", timing=timing, memory=mem, required_measurements=["gpu_memory"])
        assert lab.decide(r)[0] == status, mem


# --- end-to-end cases on the frozen fixtures -------------------------------------------------------------------------

def test_resident_vs_resident_passes_on_real_tensors(tmp_path):
    r = lab.run("fixture_equivalence", RESIDENT, "ok", root=tmp_path)
    assert r["status"] == "PASS" and r["reasons"] == [] and not r["fault_injection"]
    for shape in F.SHAPES:
        rows = [c for c in r["comparisons"] if c["fixture"] == shape]
        assert {"logits", "logp", "loss"} <= {c["name"] for c in rows}
        assert sum(c["quantity"] == "grads" for c in rows) == 12               # q/v/o x 2 layers x (A, B)
    assert all(c["max_abs"] == 0 and c["nonfinite_got"] == 0 for c in r["comparisons"])
    assert all(c["ref_max_abs"] > 0 for c in r["comparisons"] if c["quantity"] == "grads")   # active adapters
    ident = r["identity"]
    assert ident["config_sha256"] and ident["input_sha256"] and ident["tolerances_sha256"]
    assert ident["fixtures_sha256"]["4b-like"] == F.file_sha256(F.path("4b-like"))
    saved = json.loads((tmp_path / "ok" / "result.json").read_text())
    assert saved["status"] == "PASS" and (tmp_path / "ok" / "summary.md").exists()


def test_a_controlled_perturbation_makes_the_same_experiment_fail_and_is_recorded(tmp_path):
    r = lab.run("fixture_equivalence", RESIDENT, "fault", root=tmp_path, perturbation="logits=1e-6")
    assert r["status"] == "FAIL" and r["fault_injection"]
    failed = [c["name"] for c in r["comparisons"] if not c["pass"]]
    assert failed == ["logits"] and r["perturbation"]["before"] != r["perturbation"]["after"]
    assert (tmp_path / "fault" / "summary.md").read_text().startswith("**FAULT INJECTION RUN**")


def test_perturbing_one_lora_gradient_fails_that_parameter_only(tmp_path):
    name = "grads.base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight"
    r = lab.run("fixture_equivalence", RESIDENT, "grad", root=tmp_path, perturbation=f"{name}=1e-4")
    assert r["status"] == "FAIL"
    assert [(c["fixture"], c["name"]) for c in r["comparisons"] if not c["pass"]] == [("0.6b-like", name)]


def test_a_perturbation_that_rounds_away_is_refused(tmp_path):
    r = lab.run("fixture_equivalence", RESIDENT, "tiny", root=tmp_path, perturbation="logits=1e-12")
    assert r["status"] == "FAIL" and "not representable" in r["reasons"][0]


def test_an_unimplemented_backend_is_blocked_never_passed(tmp_path):
    r = lab.run("fixture_equivalence", L2L_SYNC, "blocked", root=tmp_path)
    assert r["status"] == "BLOCKED" and "l2l_sync" in r["reasons"][0] and "comparisons" not in r


def test_explicitly_blocked_candidate_inside_a_case(tmp_path, monkeypatch):
    def refuse(backend, shape):
        raise lab.Blocked("candidate needs a GPU this host does not have")
    monkeypatch.setattr(lab, "execute", refuse)
    r = lab.run("fixture_equivalence", RESIDENT, "b", root=tmp_path)
    assert r["status"] == "BLOCKED" and r["reasons"] == ["candidate needs a GPU this host does not have"]


def test_missing_fixture_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "DIR", tmp_path / "nowhere")
    r = lab.run("fixture_equivalence", RESIDENT, "nofix", root=tmp_path)
    assert r["status"] == "FAIL" and "missing experiment fixture" in r["reasons"][0]


def test_fixture_built_with_other_settings_or_config_dtype_mismatch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "LR", 1e-3)                                       # code no longer matches the saved fixture
    r = lab.run("fixture_equivalence", RESIDENT, "lr", root=tmp_path)
    assert r["status"] == "FAIL" and "different settings" in r["reasons"][0] and "'lr'" in r["reasons"][0]
    monkeypatch.undo()
    bf16 = tmp_path / "bf16.yaml"
    bf16.write_text(f"extends: {RESIDENT}\nrun: {{name: bf16}}\nmodel: {{dtype: bfloat16}}\n")
    r = lab.run("fixture_equivalence", bf16, "dtype", root=tmp_path)
    assert r["status"] == "FAIL" and "does not match fixture dtype" in r["reasons"][0]


def test_results_are_never_overwritten_and_report_rechecks_the_status(tmp_path):
    lab.run("fixture_equivalence", RESIDENT, "once", root=tmp_path)
    with pytest.raises(SystemExit, match="never overwritten"):
        lab.run("fixture_equivalence", RESIDENT, "once", root=tmp_path)
    assert "Status: PASS" in lab.report("once", root=tmp_path)
    path = tmp_path / "once" / "result.json"
    r = json.loads(path.read_text())
    r["comparisons"][0]["pass"] = False                                     # tampered raw data, recorded status PASS
    path.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="disagree with the raw data"):
        lab.report("once", root=tmp_path)


def test_timing_case_records_repeated_trials_with_uncertainty(tmp_path):
    r = lab.run("fixture_timing", RESIDENT, "t", root=tmp_path, trials=6, warmup=1)
    assert r["status"] == "PASS" and r["kind"] == "performance"
    for side in ("reference", "candidate"):
        t = r["timing"][side]
        assert len(t["trials_s"]) == 6 and t["median_ci95_s"][0] <= t["median_s"] <= t["median_ci95_s"][1]
    assert r["conclusion"]["statement"] in ("candidate faster", "candidate slower", "no detectable difference")
    assert r["memory"]["gpu"] is None and r["memory"]["cpu"]["peak_rss_mib"] > 0


def test_cli_dispatches_to_the_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "RESULTS", tmp_path)
    r = l2l_lab.main(["run", "--case", "fixture_equivalence", "--config", str(RESIDENT), "--name", "cli"])
    assert r["status"] == "PASS" and (tmp_path / "cli" / "result.json").exists()


def test_nvidia_smi_not_available_readings_are_recorded_raw_not_parsed(monkeypatch):
    import os
    import subprocess

    outputs = {"--query-compute-apps=pid,used_memory": f"{os.getpid()}, [N/A]", "--query-gpu=memory.used": "[N/A]"}
    monkeypatch.setattr(lab.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, outputs[args[1]], ""))
    smi = lab._nvidia_smi()
    assert smi["process_used_mib_nvidia_smi"] is None and smi["device_used_mib_nvidia_smi"] is None
    assert smi["nvidia_smi_compute_apps_raw"].endswith("[N/A]")


# --- execution status vs hypothesis result (schema 2) -----------------------------------------------------------------

def correctness_result(passes, **kw):
    rows = [{"name": f"t{i}", "pass": p, "reason": None if p else "over"} for i, p in enumerate(passes)]
    return base_result(comparisons=rows, reference_integrity=[], hypothesis={"id": "H01", "kind": "correctness"}, **kw)


def test_threshold_violation_is_an_outcome_not_an_execution_failure():
    ok, bad = correctness_result([True, True]), correctness_result([True, False])
    assert lab.execution_check(bad)[0] == "PASS" and lab.decide(bad)[0] == "FAIL"
    assert lab.hypothesis_result(ok)[0] == "SUPPORTED" and lab.hypothesis_result(bad)[0] == "NOT_SUPPORTED"
    assert lab.hypothesis_result(correctness_result([True], fault_injection=True))[0] == "NOT_EVALUATED"
    assert lab.hypothesis_result(correctness_result([True], blocked="no GPU"))[0] == "NOT_EVALUATED"
    assert lab.hypothesis_result(correctness_result([True], errors=["setup: x"]))[0] == "NOT_EVALUATED"
    assert lab.hypothesis_result({**ok, "hypothesis": None})[0] == "NOT_EVALUATED"


@pytest.mark.parametrize("ci, expected", [([0.8, 0.95], "SUPPORTED"), ([0.9, 1.1], "INCONCLUSIVE"),
                                          ([1.0, 1.2], "NOT_SUPPORTED")])
def test_a_valid_benchmark_supports_a_speedup_only_when_the_ci_excludes_one(ci, expected):
    trials = {"trials_s": [0.1] * lab.MIN_TRIALS}
    r = base_result("performance", timing={"reference": trials, "candidate": trials}, conclusion={"ci95": ci},
                    hypothesis={"id": "H06", "kind": "performance", "prediction": "candidate_faster"})
    assert lab.execution_check(r)[0] == "PASS" and lab.hypothesis_result(r)[0] == expected


def test_declared_hypotheses_must_exist_and_match_the_case_kind(tmp_path):
    with pytest.raises(SystemExit, match="unknown hypothesis"):
        lab.run("fixture_equivalence", RESIDENT, "h", root=tmp_path, hypothesis_id="H99")
    with pytest.raises(SystemExit, match="H06 is a performance hypothesis"):
        lab.run("fixture_equivalence", RESIDENT, "h", root=tmp_path, hypothesis_id="H06")


def test_new_results_record_both_fields_and_report_rechecks_them(tmp_path):
    r = lab.run("fixture_equivalence", RESIDENT, "s2", root=tmp_path)
    assert (r["schema"], r["execution_status"], r["hypothesis_result"]) == (2, "PASS", "NOT_EVALUATED")
    path = tmp_path / "s2" / "result.json"
    saved = json.loads(path.read_text())
    saved["hypothesis_result"] = "SUPPORTED"
    path.write_text(json.dumps(saved))
    with pytest.raises(SystemExit, match="disagree with the raw data"):
        lab.report("s2", root=tmp_path)


def test_ticket01_results_are_checked_but_never_rewritten(tmp_path):
    import shutil

    src = lab.RESULTS / "t01_timing_cuda"
    shutil.copytree(src, tmp_path / "t01_timing_cuda")
    before = {f: (tmp_path / "t01_timing_cuda" / f).read_bytes() for f in ("result.json", "summary.md")}
    text = lab.report("t01_timing_cuda", root=tmp_path)
    assert "derived by report, not recorded: execution_status PASS, hypothesis_result NOT_EVALUATED" in text
    assert before == {f: (tmp_path / "t01_timing_cuda" / f).read_bytes() for f in before}


def test_evidence_grade_never_upgrades_wsl_or_incomplete_readings():
    allocator = {"allocated_mib": 1, "reserved_mib": 1, "peak_allocated_mib": 1, "peak_reserved_mib": 1}
    diag = {"gpu": {**allocator, "device_used_mib_mem_get_info": 900.0, "process_used_mib_nvidia_smi": None}}
    full = {"gpu": {**diag["gpu"], "process_used_mib_nvidia_smi": 700.0}}
    assert lab.evidence_grade({"gpu": None}) == "none"
    assert lab.evidence_grade({"gpu": allocator}) == "incomplete"
    assert lab.evidence_grade(diag, "6.8.0-generic") == "diagnostic"
    assert lab.evidence_grade(full, "6.18.58.1-microsoft-standard-WSL2") == "diagnostic"
    assert lab.evidence_grade(full, "6.8.0-generic") == "assignment"


# --- Ticket 02: weight-store cases ------------------------------------------------------------------------------------

STORE = lab.ROOT / "research" / "l2l" / "configs" / "store.yaml"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="page-locking and GPU memory need a CUDA device")


def pageable(tmp_path):
    path = tmp_path / "pageable.yaml"
    path.write_text(f"extends: {STORE}\nrun: {{name: pageable}}\nexecution: {{weight_storage: pageable}}\n")
    return path


def test_store_equivalence_passes_bitwise_and_a_perturbed_weight_fails_alone(tmp_path):
    r = lab.run("store_equivalence", pageable(tmp_path), "ok", root=tmp_path)
    assert (r["status"], r["execution_status"]) == ("PASS", "PASS")
    weights = [c for c in r["comparisons"] if c["quantity"] == "weights"]
    assert len(weights) == 44 + 66 and all(c["max_abs"] == 0 for c in r["comparisons"])  # 11 per layer: 4 + 6 layers
    bad = lab.run("store_equivalence", pageable(tmp_path), "bad", root=tmp_path,
                  perturbation="weights.3.mlp.down_proj.weight=1e-3")
    assert bad["status"] == "FAIL" and bad["execution_status"] == "PASS"
    assert [c["name"] for c in bad["comparisons"] if not c["pass"]] == ["weights.3.mlp.down_proj.weight"]


def test_store_cases_are_gated_on_their_own_requirements_not_on_l2l_execution(tmp_path, monkeypatch):
    assert lab.run("store_equivalence", RESIDENT, "r", root=tmp_path)["status"] == "BLOCKED"   # no storage settings
    assert lab.run("fixture_equivalence", STORE, "f", root=tmp_path)["status"] == "BLOCKED"   # l2l execution: not yet
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = lab.run("store_equivalence", STORE, "nocuda", root=tmp_path)
    assert r["status"] == "BLOCKED" and "page-locking needs a CUDA device" in r["reasons"][0]


def test_forward_equivalence_is_blocked_without_l2l_sync_or_cuda(tmp_path, monkeypatch):
    r = lab.run("forward_equivalence", RESIDENT, "r", root=tmp_path)
    assert r["status"] == "BLOCKED" and "comparisons" not in r
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = lab.run("forward_equivalence", STORE, "nocuda", root=tmp_path)
    assert r["status"] == "BLOCKED" and any("CUDA device" in x for x in r["reasons"]) and "comparisons" not in r


def depth_result(streamed, resident):
    rows = [{"depth": d, "streamed_peak_allocated": s, "resident_peak_allocated": q}
            for d, s, q in zip((2, 4, 8), streamed, resident)]
    return base_result("memory", measurements={"depths": rows},
                       hypothesis={"id": "H07", "kind": "memory", "prediction": "streamed_constant_resident_increasing"})


@pytest.mark.parametrize("streamed, resident, expected", [
    ([100, 100, 100], [10, 20, 40], "SUPPORTED"), ([100, 100, 101], [10, 20, 40], "NOT_SUPPORTED"),
    ([100, 100, 100], [10, 20, 20], "INCONCLUSIVE")])
def test_h07_rule_is_the_one_written_in_hypotheses_yaml(streamed, resident, expected):
    assert lab.hypothesis_result(depth_result(streamed, resident))[0] == expected


def test_a_memory_case_without_measurements_cannot_pass():
    assert lab.execution_check(base_result("memory"))[0] == "FAIL"


@needs_cuda
def test_store_depth_keeps_streamed_gpu_memory_constant_on_a_small_scan(tmp_path):
    r = lab.run("store_depth", STORE, "d", root=tmp_path, depths=(2, 4), hypothesis_id="H07")
    assert (r["execution_status"], r["hypothesis_result"]) == ("PASS", "SUPPORTED"), r["reasons"]
    assert all(d["streamed_peak_allocated"] == d["planned_buffer_bytes"] for d in r["measurements"]["depths"])


STORE_CPU = lab.ROOT / "research" / "l2l" / "configs" / "store_cpu.yaml"


def test_gradient_equivalence_is_blocked_without_l2l_sync_or_a_requested_cuda_device(tmp_path, monkeypatch):
    r = lab.run("gradient_equivalence", RESIDENT, "r", root=tmp_path)
    assert r["status"] == "BLOCKED" and "comparisons" not in r
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    r = lab.run("gradient_equivalence", STORE_CPU, "nocuda", root=tmp_path, device="cuda")
    assert r["status"] == "BLOCKED" and "no CUDA device" in r["reasons"][0] and "comparisons" not in r


def test_gradient_equivalence_passes_on_cpu_and_a_detached_boundary_fails(tmp_path):
    r = lab.run("gradient_equivalence", STORE_CPU, "ok", root=tmp_path, hypothesis_id="H02", device="cpu")
    assert r["status"] == "PASS" and r["hypothesis_result"] == "SUPPORTED", r["reasons"][:5]
    quantities = {c["quantity"] for c in r["comparisons"]}
    assert quantities == {"loss", "logits", "grads", "hidden_grads"}
    grads = [c for c in r["comparisons"] if c["quantity"] == "grads"]
    assert all(c["ref_norm"] > 0 and c["max_abs"] == 0 and c["rel"] == 0 for c in grads)
    steps = {next(s for s in ("mb1", "acc") if f".{s}.base_model." in c["name"]) for c in grads}
    assert steps == {"mb1", "acc"}  # single and accumulated micro-batches
    assert all(r["measurements"]["saved_bytes"][f"{t}.b1"] < v for t, v in r["measurements"]["resident_saved_bytes"].items())
    f = lab.run("gradient_equivalence", STORE_CPU, "fault", root=tmp_path, perturbation="detach_boundary=3", device="cpu")
    assert f["status"] == "FAIL" and f["execution_status"] == "PASS"  # the comparisons, not an exception, catch it
    failed = [c for c in f["comparisons"] if not c["pass"]]
    assert failed and all(c["reason"] == "missing in candidate" for c in failed)
    assert {p for c in failed for p in ("last2", "first_last", "distributed") if f".{p}." in c["model"]} == \
        {"last2", "first_last", "distributed"}
    assert lab.report("ok", root=tmp_path) and lab.report("fault", root=tmp_path)


@pytest.fixture
def small_head_grid(monkeypatch):
    from qwen_grpo import head_lab
    monkeypatch.setattr(head_lab, "VOCABS", {64: (7, 64)})
    monkeypatch.setattr(head_lab, "LENGTHS", (4,))
    monkeypatch.setattr(head_lab, "MICRO", (None, 2))
    monkeypatch.setattr(head_lab, "QWEN3_VOCAB", 3000)
    monkeypatch.setattr(head_lab, "REAL_HEAD", {"hidden": 64, "vocab": 3000, "rows": 2, "R": 8, "chunk": 256})


def test_logprob_equivalence_is_blocked_without_l2l_sync(tmp_path):
    r = lab.run("logprob_equivalence", RESIDENT, "r", root=tmp_path)
    assert r["status"] == "BLOCKED" and "comparisons" not in r


def test_logprob_equivalence_passes_on_a_small_grid_and_a_raw_logit_fails(tmp_path, small_head_grid):
    r = lab.run("logprob_equivalence", STORE_CPU, "ok", root=tmp_path, hypothesis_id="H03", trials=2, warmup=0)
    assert r["status"] == "PASS" and r["hypothesis_result"] == "SUPPORTED", r["reasons"][:5]
    assert {c["quantity"] for c in r["comparisons"]} == {"logp", "loss", "grads", "head_grads", "hidden_grads"}
    names = " ".join(c["name"] for c in r["comparisons"])
    assert all(s in names for s in (".chunked7.", ".chunked64.", ".streamed_full.", ".streamed7.", ".it3.", ".old"))
    assert all(c["max_abs"] == 0 for c in r["comparisons"] if ".streamed_full." in c["name"])  # L2L itself is exact
    head = r["measurements"]["head"]
    assert all(v["chunked"]["saved_bytes"] < v["full"]["saved_bytes"] for v in head.values())
    f = lab.run("logprob_equivalence", STORE_CPU, "fault", root=tmp_path, perturbation="raw_logit", trials=2, warmup=0)
    assert f["status"] == "FAIL" and f["execution_status"] == "PASS"
    failed = {c["quantity"] for c in f["comparisons"] if not c["pass"]}
    assert {"logp", "grads"} <= failed
    for side in (".chunked7.", ".chunked64.", ".streamed_full.", ".streamed7."):  # every candidate side is caught
        assert any(not c["pass"] for c in f["comparisons"] if side in c["name"]), side
    assert lab.report("ok", root=tmp_path) and lab.report("fault", root=tmp_path)
