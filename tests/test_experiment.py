"""scripts/experiment.py and configs/studies/assignment.yaml: the study manifest is well-formed and points at existing
tools, configs and hypotheses; hardware and prerequisite checks block (and record BLOCKED) instead of running or
passing; recorded evidence is judged from what the tools wrote, compressed or not; and nothing is overwritten.
CPU only, no model is loaded, nothing is run."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from qwen_grpo import config, lab
from qwen_grpo.evidence import EXPERIMENTS, read_result

ROOT = Path(__file__).parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X = script("experiment")
STUDY, HYPS = X.load_study()
ENTRIES = {e["id"]: e for e in STUDY["entries"]}


def test_every_hypothesis_has_an_entry_and_every_entry_a_hypothesis():
    assert {e.split(".")[0] for e in ENTRIES} == set(HYPS) == {f"H{i:02d}" for i in range(1, 13)}


@pytest.mark.parametrize("eid", sorted(ENTRIES))
def test_entries_name_existing_tools_cases_and_loadable_configs(eid):
    e = ENTRIES[eid]
    for s in X.steps(e):
        argv = s["argv"]
        assert (ROOT / argv[0]).exists(), argv[0]
        if "--case" in argv:
            assert argv[argv.index("--case") + 1] in lab.CASES
        if "--hypothesis" in argv:
            assert argv[argv.index("--hypothesis") + 1] == eid.split(".")[0]  # a case never declares another hypothesis
        if "--perturb" in argv:
            assert X.judge_kind(e) == "lab_fault" and "--hypothesis" not in argv  # fault runs are never evidence for H
    for p in X.config_paths(e):
        assert config.load(ROOT / p)  # loads in the current schema (abstract bases are never named)
    assert set(e.get("requires", [])) <= set(X.CAPABILITIES)


def test_precision_labels_are_explicit():
    """bf16 entries require native bf16; a float32 entry of a bf16 hypothesis is labelled diagnostic and runs configs
    that say float32."""
    for e in ENTRIES.values():
        precision = e.get("precision", [])
        if e.get("model") != "tiny" and precision == ["bfloat16"]:
            assert "bf16" in e["requires"], e["id"]
        if e.get("diagnostic"):
            assert precision == ["float32"] and all(config.load(ROOT / p).model.dtype == "float32"
                                                   for p in X.config_paths(e)), e["id"]


def test_the_manifest_cannot_carry_criteria_or_thresholds(tmp_path):
    study = yaml.safe_load(X.STUDY.read_text())
    study["entries"][0]["rtol"] = 0.5
    p = tmp_path / "study.yaml"
    p.write_text(yaml.safe_dump(study))
    with pytest.raises(SystemExit, match="belong to hypotheses.yaml / tolerances.yaml"):
        X.load_study(p)


def test_capability_gating_on_the_kaggle_t4():
    t4 = X.TARGETS["kaggle-t4"]
    assert any("bfloat16" in w for w in X.unmet(ENTRIES["H05.real_0.6b_bf16"], t4))
    assert X.unmet(ENTRIES["H05.real_0.6b_fp32_diagnostic"], t4) == []
    assert X.unmet(ENTRIES["H07.depth_residency_execution"], t4) == []
    cpu = X.TARGETS["kaggle-cpu"]
    assert any("package versions" in w for w in X.unmet(ENTRIES["H02.lab_sync_cpu"], cpu))  # torch +cpu vs fixtures


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ROOT", tmp_path)
    monkeypatch.setattr(X, "EVIDENCE", tmp_path / "evidence")
    monkeypatch.setattr(X, "STUDY", ROOT / "configs" / "studies" / "assignment.yaml")
    monkeypatch.setattr(X, "git", lambda *a: None)
    monkeypatch.setattr(X.subprocess, "run", lambda *a, **k: pytest.fail("nothing may run in these tests"))
    return tmp_path


def test_missing_prerequisites_block_and_are_recorded_never_passed(sandbox, monkeypatch):
    monkeypatch.setattr(X, "resolved_configs", lambda e: {})
    res = X.execute(ENTRIES["H12.final_eval"], X.TARGETS["ampere"])
    assert res["status"] == "BLOCKED" and res["criterion"] == "NOT_EVALUATED"
    assert "run qwen3_4b_final does not exist" in res["detail"] and "qwen3_4b_final_lr0" in res["detail"]
    [d] = list((sandbox / "evidence").iterdir())
    assert d.name.endswith("_H12.final_eval") and json.loads((d / "results.json").read_text())["status"] == "BLOCKED"
    assert {p.name for p in d.iterdir()} >= {"manifest.json", "results.json", "summary.md", "raw"}


def test_an_incomplete_or_changed_run_is_not_a_prerequisite(sandbox):
    def make(name, status, frozen):
        (sandbox / "runs" / name / "training").mkdir(parents=True)
        (sandbox / "runs" / name / "manifest.json").write_text(json.dumps({"status": status}))
        (sandbox / "runs" / name / "training" / "training.json").write_text(
            json.dumps({"frozen_changed": frozen, "final_checkpoint": {"path": "x"}}))
    make("ok", "completed", [])
    make("interrupted", "interrupted", [])
    make("changed", "completed", ["model.layers.0.mlp.up_proj.weight"])
    assert X.run_state("ok")[0] and not X.run_state("missing")[0]
    assert "resume it" in X.run_state("interrupted")[1] and "frozen weights" in X.run_state("changed")[1]


def test_hardware_blocks_before_anything_runs(sandbox, monkeypatch):
    monkeypatch.setattr(X, "resolved_configs", lambda e: {})
    res = X.execute(ENTRIES["H10.dflash_bench"], X.TARGETS["kaggle-t4"])
    assert res["status"] == "BLOCKED" and "no native bfloat16" in res["detail"]


def test_existing_evidence_is_done_and_judged_not_rerun(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(X, "ROOT", ROOT)  # the repository's recorded evidence
    res = X.execute(ENTRIES["H05.lab_sync_cpu"], X.TARGETS["ampere"])
    assert (res["status"], res["outcome"], res["criterion"]) == ("DONE", "PASS", "MET")
    assert not (sandbox / "evidence").exists()  # nothing written, nothing re-run
    fault = X.judge(ENTRIES["H05.fault_advantage"], "t07_update_equivalence_advantage")
    assert fault[0] == "PASS" and "detected" in fault[2] and fault[1] == "NOT_EVALUATED"


def test_judging_reads_compressed_results_and_keeps_negative_ones():
    """t05 v1 is stored as result.json.gz: it is read like any result and stays NOT_SUPPORTED."""
    assert (EXPERIMENTS / "t05_logprob_equivalence_cpu" / "result.json.gz").exists()
    v1 = read_result(EXPERIMENTS / "t05_logprob_equivalence_cpu")
    assert (v1["status"], v1["hypothesis_result"]) == ("FAIL", "NOT_SUPPORTED")
    assert lab.derived(v1)["hypothesis_result"] == "NOT_SUPPORTED"  # re-derived from the raw comparisons
    v2 = X.judge(ENTRIES["H03.lab_v2_cpu"], "t05_logprob_equivalence_cpu_v2")
    assert v2[:2] == ("PASS", "MET")


def test_the_evidence_index_matches_the_files_and_the_hypotheses():
    idx = json.loads((ROOT / "research" / "l2l" / "evidence_index.json").read_text())
    assert idx["cited_but_missing"] == []
    by_name = {e["name"]: e for e in idx["experiments"]}
    for hid, h in HYPS.items():
        for name in h.get("evidence") or []:
            assert hid in by_name[name]["cited_by"], (hid, name)
    import gzip
    import hashlib
    for e in idx["experiments"]:
        data = (ROOT / e["file"]).read_bytes()
        data = gzip.decompress(data) if e["file"].endswith(".gz") else data
        assert hashlib.sha256(data).hexdigest() == e["sha256"], e["file"]


def test_requirements_judge_maps_diagnostic_memory_evidence_to_inconclusive(monkeypatch):
    fake = type("FR", (), {"requirements": staticmethod(lambda: ([{"id": "R13", "status": "BLOCKED", "evidence": "WSL2"}], None))})
    monkeypatch.setattr(X, "final_report", lambda: fake)
    assert X.judge(ENTRIES["H11.profile"], "t11_profile_qwen3_4b_async")[:2] == ("BLOCKED", "INCONCLUSIVE")
    fake.requirements = staticmethod(lambda: ([{"id": "R14", "status": "PASS", "evidence": ""},
                                               {"id": "R15", "status": "PASS", "evidence": ""}], None))
    outcome, crit, detail = X.judge(ENTRIES["H12.final_eval"], "t13_final_eval_qwen3_4b_final")
    assert (outcome, crit) == ("PASS", "NOT_EVALUATED") and "collapse" in detail  # partial criterion: never MET


def test_the_final_report_records_how_old_its_evidence_is(monkeypatch, tmp_path):
    rep = script("final_report")
    monkeypatch.setattr(rep, "OUT_JSON", tmp_path / "final_results.json")
    monkeypatch.setattr(rep, "OUT_MD", tmp_path / "final_report.md")
    rep.main()
    res = json.loads((tmp_path / "final_results.json").read_text())
    assert res["verdict"] == "DON'T SHIP"  # GPU evidence pending
    fresh = {f["evidence"]: f for f in res["evidence_freshness"]}
    assert fresh["t03_forward_equivalence"]["commit"].startswith("c4b391b")
    assert "qwen_grpo/execution/layer_stream.py" in fresh["t03_forward_equivalence"]["engine_files_changed_since"]
    assert "## Evidence freshness" in (tmp_path / "final_report.md").read_text()


def test_drift_checkpoints_must_be_saved_and_complete_before_the_drift_study_runs(sandbox, monkeypatch):
    """H10.dflash_drift names step0/20/40/60 of the primary run: each must exist as a complete (COMMIT.json-verified)
    adapter checkpoint, else the entry is BLOCKED, never run against a missing or torn checkpoint."""
    from qwen_grpo.streaming import resume as R
    monkeypatch.setattr(X, "run_state", lambda name: (True, "completed"))
    drift = ENTRIES["H10.dflash_drift"]
    assert any("step20 was not saved" in w for w in X.unmet_prerequisites(drift))
    for step in (0, 20, 40, 60):
        final = sandbox / "runs" / "qwen3_4b_final" / "checkpoints" / f"step{step}"
        R.publish(final, lambda tmp: (tmp / "adapter_model.safetensors").write_bytes(b"adapter"), step)
    assert X.unmet_prerequisites(drift) == []
    (sandbox / "runs" / "qwen3_4b_final" / "checkpoints" / "step40" / "adapter_model.safetensors").write_bytes(b"torn")
    assert any("step40 is not a valid adapter checkpoint" in w for w in X.unmet_prerequisites(drift))


def test_opt_in_entries_run_only_when_named(monkeypatch):
    """The 210-update continuation and its drift study are opt_in: `run --hypothesis H12` (or H10) never selects them,
    `run --entry H12.train_extended` does."""
    ran = []
    monkeypatch.setattr(X, "probe", lambda: X.TARGETS["ampere"])
    monkeypatch.setattr(X, "execute", lambda e, cap, **k: ran.append(e["id"]) or {"status": "PASS"})
    X.cmd_run(type("A", (), {"study": X.STUDY, "entry": [], "hypothesis": ["H12", "H10"], "again": False,
                             "dry_run": True})())
    assert "H12.train_final" in ran and "H10.dflash_drift" in ran
    assert "H12.train_extended" not in ran and "H10.dflash_drift_extended" not in ran
    ran.clear()
    X.cmd_run(type("A", (), {"study": X.STUDY, "entry": ["H12.train_extended"], "hypothesis": [], "again": False,
                             "dry_run": True})())
    assert ran == ["H12.train_extended"]
    ext = ENTRIES["H12.train_extended"]
    assert ext["steps"][0]["argv"][-2:] == ["--continue-from", "runs/qwen3_4b_final/checkpoints/step60"]
    assert ext["prerequisites"]["checkpoints"] == ["runs/qwen3_4b_final/checkpoints/step60"]


def test_the_assignment_stages_never_list_an_opt_in_entry():
    A = script("assignment")
    opt_in = {i for i, e in ENTRIES.items() if e.get("opt_in")}
    assert opt_in == {"H12.train_extended", "H10.dflash_drift_extended"}
    assert not opt_in & {i for ids in A.STAGE_ENTRIES.values() for i in ids}
    with pytest.raises(SystemExit, match="opt-in"):
        A.check_no_opt_in(ENTRIES, {"F": ["H12.train_final", "H12.train_extended"]})
