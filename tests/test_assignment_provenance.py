"""Release-blocker regressions of the assignment orchestrator (scripts/assignment.py), CPU only:
  1. held-out evaluation evidence is reused only if it evaluated this checkpoint, config, data, code and models;
  2. resume identity covers tracked and untracked experiment code, while report-only code changes only force the
     report to be regenerated;
  3. the Qwen3-4B budget prediction the profile is compared with describes the configuration that actually runs;
  4. an interrupted non-streaming run (no crash-consistent checkpoint) is never "resumed".
"""
import dataclasses
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def load_assignment(monkeypatch, tmp_path):
    for k, d in (("ASSIGNMENT_EXPERIMENTS_ROOT", "exp"), ("ASSIGNMENT_RUNS_ROOT", "runs"),
                 ("ASSIGNMENT_RESULTS_ROOT", "results"), ("ASSIGNMENT_EVIDENCE_ROOT", "evidence")):
        (tmp_path / d).mkdir(exist_ok=True)
        monkeypatch.setenv(k, str(tmp_path / d))
    spec = importlib.util.spec_from_file_location("assignment_under_test", ROOT / "scripts" / "assignment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- 1. held-out evaluation identity -------------------------------------------------------------------------------

def checkpoint(runs, name, step, payload):
    d = runs / name / "checkpoints" / f"step{step}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(payload)
    return d


@pytest.fixture
def heldout(monkeypatch, tmp_path):
    a = load_assignment(monkeypatch, tmp_path)
    import qwen_grpo.trainer as trainer
    monkeypatch.setattr(trainer, "model_revision", lambda m: f"rev-of-{m}")
    checkpoint(tmp_path / "runs", "qwen3_4b_final", 0, b"init")
    checkpoint(tmp_path / "runs", "qwen3_4b_final", 60, b"trained adapter")  # the primary run's last update
    checkpoint(tmp_path / "runs", "qwen3_4b_final_lr0", 30, b"control adapter")
    _, entries = a.load_study()
    return a, entries["H12.final_eval"], tmp_path


def write_eval(tmp, identity):
    (tmp / "exp" / "t13_final_eval_qwen3_4b_final.json").write_text(json.dumps(
        {"n_test": identity.get("n_test") if identity else 1066, "split": "test",
         "provenance": {"git_commit": "whatever", "environment": {"gpu": {"name": "NVIDIA GeForce RTX 3090"}}},
         **({"identity": identity} if identity is not None else {})}))


GPU = {"gpu": "NVIDIA GeForce RTX 3090"}


def test_matching_heldout_evidence_is_reusable(heldout):
    a, entry, tmp = heldout
    write_eval(tmp, a.heldout_identity_expected())
    ok, why = a.foreign_check(entry, GPU)
    assert ok, why


@pytest.mark.parametrize("field,value", [
    ("checkpoint_adapter_sha256", "0" * 64),        # another checkpoint (e.g. an earlier run)
    ("control_adapter_sha256", "0" * 64),
    ("config_sha256", "f" * 64),                    # another configuration
    ("dataset_sha256", "e" * 64),                   # another data file / split
    ("code_sha256", "d" * 64),                      # another evaluation implementation
    ("model_revisions", {"Qwen/Qwen3-4B": "other"}),  # another base-model revision
    ("split", "validation"),
    ("n_test", 100),
])
def test_incompatible_heldout_evidence_is_not_reused(heldout, field, value):
    a, entry, tmp = heldout
    write_eval(tmp, {**a.heldout_identity_expected(), field: value})
    ok, why = a.foreign_check(entry, GPU)
    assert not ok and field.split("_")[0] in why


def test_heldout_evidence_without_an_identity_is_not_reused(heldout):
    """Evidence written before identities were recorded (e.g. by another commit) names only its GPU: not enough."""
    a, entry, tmp = heldout
    write_eval(tmp, None)
    ok, why = a.foreign_check(entry, GPU)
    assert not ok and "identity" in why


def test_other_qwen3_4b_outputs_need_the_same_experiment_code(heldout):
    a, _, tmp = heldout
    _, entries = a.load_study()
    prov = {"gpu": "NVIDIA GeForce RTX 3090", "git_commit": "0000000"}
    (tmp / "exp" / "t11_profile_qwen3_4b_async.json").write_text(json.dumps({"provenance": prov}))
    ok, why = a.foreign_check(entries["H11.profile"], GPU)
    assert not ok and "code" in why
    prov["code_sha256"] = a.code_identity()["experiment_code_sha256"]
    (tmp / "exp" / "t11_profile_qwen3_4b_async.json").write_text(json.dumps({"provenance": prov}))
    ok, why = a.foreign_check(entries["H11.profile"], GPU)
    assert ok, why


# --- 2. experiment vs report identity -------------------------------------------------------------------------------

def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    for p, text in (("qwen_grpo/engine.py", "x = 1\n"), ("scripts/assignment_report.py", "# report\n"),
                    ("configs/a.yaml", "a: 1\n"), (".gitignore", "__pycache__/\n")):
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(text)
    git(r, "init", "-q")
    git(r, "add", "qwen_grpo/engine.py", "configs/a.yaml", ".gitignore")  # the report script stays untracked
    git(r, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init")
    return r


def test_experiment_identity_covers_tracked_and_untracked_code_and_not_the_report(repo):
    from qwen_grpo.evidence import code_sha256, report_code_sha256
    e0, r0 = code_sha256(repo), report_code_sha256(repo)
    (repo / "qwen_grpo" / "engine.py").write_text("x = 2\n")                 # tracked change
    e1 = code_sha256(repo)
    assert e1 != e0
    (repo / "qwen_grpo" / "engine.py").write_text("x = 1\n")
    assert code_sha256(repo) == e0
    (repo / "qwen_grpo" / "new_kernel.py").write_text("y = 1\n")             # untracked experiment code
    assert code_sha256(repo) != e0
    (repo / "qwen_grpo" / "new_kernel.py").unlink()
    (repo / "qwen_grpo" / "__pycache__").mkdir()
    (repo / "qwen_grpo" / "__pycache__" / "engine.cpython-312.pyc").write_bytes(b"\0")  # ignored: no effect
    assert code_sha256(repo) == e0
    (repo / "scripts" / "assignment_report.py").write_text("# report v2\n")  # untracked report-only change
    assert code_sha256(repo) == e0 and report_code_sha256(repo) != r0


def test_resume_refuses_experiment_changes_and_regenerates_the_report_after_report_changes(monkeypatch, tmp_path, capsys):
    a = load_assignment(monkeypatch, tmp_path)
    ident = {"git_commit": "c" * 40, "experiment_code_sha256": "e" * 64, "report_code_sha256": "r" * 64,
             "tracked_dirty": False, "study_sha256": "s" * 64, "config_sha256": {"configs/x.yaml": "1"}}
    monkeypatch.setattr(a, "code_identity", lambda: dict(ident))
    monkeypatch.setattr(a, "load_study", lambda: ({"gpu_memory_cap": {"bytes": 8e9}}, {}))
    assert a.main(["init"]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    ident["report_code_sha256"] = "R" * 64                                    # e.g. an untracked report-script edit
    assert a.main(["init", "--resume"]) == 0
    state = json.loads((run_dir / "state.json").read_text())
    assert state["identity"]["report_code_sha256"] == "R" * 64
    assert any("report code changed" in h["event"] for h in state["history"])
    ident["experiment_code_sha256"] = "E" * 64                                # tracked or untracked engine edit
    assert a.main(["init", "--resume"]) == 2
    assert "experiment_code_sha256" in capsys.readouterr().err


def test_the_report_records_its_identity_and_status_flags_a_stale_report(monkeypatch, tmp_path, capsys):
    a = load_assignment(monkeypatch, tmp_path)
    assert a.main(["init"]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    a.main(["report", "--run-dir", str(run_dir)])
    req = json.loads((run_dir / "requirements.json").read_text())
    assert req["report_identity"]["report_code_sha256"] == a.code_identity()["report_code_sha256"]
    req["report_identity"]["report_code_sha256"] = "old"
    (run_dir / "requirements.json").write_text(json.dumps(req))
    capsys.readouterr()
    a.main(["status", "--run-dir", str(run_dir)])
    assert "stale" in capsys.readouterr().out


# --- 3. the budget prediction describes the configuration that runs --------------------------------------------------

def budget_mismatches(budget, cfg):
    """{field: (budget value, resolved config value)} where a budget file describes another configuration."""
    t, x, e = cfg.training, cfg.execution, cfg.evaluation
    N = t.batch_size * t.group_size
    g, s = budget["grpo"], budget["streaming"]
    expected = {
        "model": (budget["model"]["id"], cfg.model.name), "dtype": (budget["model"]["weight_dtype"], cfg.model.dtype),
        "B": (g["prompt_batch_size"], t.batch_size), "G": (g["group_size"], t.group_size),
        "rollout batch": (g["rollout_batch_size"], N),
        "micro-batch": (g["micro_batch_size"], t.micro_batch_size or N),       # null = all B*G rows in one chunk
        "new tokens": (g["generation_tokens"], t.max_new_tokens), "iterations": (g["num_iterations"], t.num_iterations),
        "LoRA rank": (budget["lora"]["rank"], cfg.lora.rank),
        "layer buffers": (s["gpu_layer_buffers"], x.weight_buffers),
        "resident layers": (s["resident_decoder_layers"], x.resident_layers),
        "pinned store": (s["weight_placement"], f"{x.weight_storage}_cpu"),
        "KV cache": (s["kv_cache"], x.kv_cache),
        "schedule": (s["schedule"], "microbatch_major"),                     # docs/architecture.md: micro-batch-major
        "checkpoints on GPU": (s["activation_checkpoints"], "gpu"),
        "merge device": (budget["merge"]["device"], "cpu"),                  # export/merge.py never touches a GPU
        "eval examples": (budget["evaluation"]["examples"], cfg.data.val_subset),
        "eval batch": (budget["evaluation"]["batch_size"], e.batch_size),
        "eval new tokens": (budget["evaluation"]["generation_tokens"], e.max_new_tokens),
        "DFlash rollouts": (budget["dflash"]["enabled"], cfg.dflash.enabled),
        # peaks are per phase and do not depend on the number of updates; what the schedule decides is whether the
        # AdamW state is live during the subset evaluations (an evaluation every 10 of 60 updates: yes)
        "eval during training": (budget["evaluation"]["during_training"],
                                 cfg.evaluation.every <= cfg.training.max_steps),
    }
    if cfg.dflash.enabled:
        expected["drafter"] = (budget["dflash"]["draft_model_id"], cfg.dflash.draft_model)
    return {k: v for k, v in expected.items() if v[0] != v[1]}


def committed_prediction(path):
    pred = json.loads(path.read_text())
    cfg_file = ROOT / pred["provenance"]["config"]
    assert pred["provenance"]["config_sha256"] == hashlib.sha256(cfg_file.read_bytes()).hexdigest(), \
        f"{path.name}: its budget file changed after the prediction was made"
    return pred, yaml.safe_load(cfg_file.read_text())


def test_the_final_prediction_matches_the_resolved_final_configuration_with_dflash():
    """The report's final-run prediction (assignment_report.FINAL_PREDICTION) and the budget recomputed in stage A
    describe qwen3_4b_final.yaml as it resolves now, DFlash rollouts and drafter included (and so the LR=0 control,
    which differs only in learning rate and length)."""
    from qwen_grpo import config
    sys.path.insert(0, str(ROOT / "scripts"))
    import assignment_report
    pred, budget = committed_prediction(ROOT / "results" / "budget" / "predictions" / assignment_report.FINAL_PREDICTION)
    for rel in ("qwen3_4b_final.yaml", "qwen3_4b_final_lr0.yaml", "qwen3_4b_final_ext210.yaml"):
        cfg = config.load(ROOT / "configs" / "experiments" / rel)
        assert cfg.dflash.enabled
        assert not (wrong := budget_mismatches(budget, cfg)), f"{rel}: {wrong}"
    assert pred["provenance"]["config"] == "configs/budget/qwen3-4b_final_dflash.yaml"
    assert config.load(ROOT / "configs" / "experiments" / "qwen3_4b_final.yaml").training.max_steps == 60
    assert pred["overall_verdict"] == "PASS" and pred["config"]["hardware"]["gpu_vram_gib"] * 2**30 <= 8e9 + 1e5
    assert "configs/budget/qwen3-4b_final_dflash.yaml" in (ROOT / "scripts" / "assignment.py").read_text()  # stage A


def test_the_profiled_prediction_matches_the_plain_l2l_configurations_it_profiles():
    """H11.profile profiles qwen3_4b_l2l.yaml (plain rollouts) at the final batch shape; its prediction is the plain
    one, which must keep describing the plain-L2L baseline (qwen3_4b_smoke.yaml) exactly."""
    from qwen_grpo import config
    study = yaml.safe_load((ROOT / "configs" / "studies" / "assignment.yaml").read_text())
    prof = next(e for e in study["entries"] if e["id"] == "H11.profile")
    argv = prof["steps"][0]["argv"] if isinstance(prof["steps"][0], dict) else prof["steps"][0]
    pred_path = ROOT / argv[argv.index("--prediction") + 1]
    pred, budget = committed_prediction(pred_path)
    smoke = config.load(ROOT / "configs" / "experiments" / "qwen3_4b_smoke.yaml")
    smoke = dataclasses.replace(smoke, data=dataclasses.replace(smoke.data, val_subset=128))  # smoke evaluates 16, the
    # budget sizes the final run's 128-example subset evaluation; every other field must match as it resolves
    assert not smoke.dflash.enabled and not (wrong := budget_mismatches(budget, smoke)), wrong
    assert config.load(ROOT / argv[argv.index("--config") + 1]).dflash.enabled is False


def test_the_historical_prediction_is_preserved_and_labelled():
    hist = ROOT / "results" / "budget" / "predictions" / "qwen3-4b_8gb.json"
    assert hist.exists()
    assert json.loads(hist.read_text())["provenance"]["config"].endswith("configs/budget/qwen3-4b.yaml")
    text = (ROOT / "configs" / "budget" / "qwen3-4b_final.yaml").read_text()
    assert "historical" in text and "qwen3-4b_8gb" in text


# --- 4. non-streaming runs have no crash-consistent checkpoint ------------------------------------------------------

def test_an_interrupted_non_streaming_run_is_not_resumed(monkeypatch, tmp_path):
    a = load_assignment(monkeypatch, tmp_path)
    from qwen_grpo import config
    marker = tmp_path / "run_py_was_called"
    fake = tmp_path / "fake_run_py.py"
    fake.write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('called')\n")
    monkeypatch.setenv("ASSIGNMENT_TOOL_OVERRIDES", json.dumps({"run_py": f"{sys.executable} {fake}"}))
    _, entries = a.load_study()
    entry = entries["H05.real_0.6b_bf16"]
    ident = a.code_identity()
    for name, cfg_path in a.run_configs(entry).items():
        d = tmp_path / "runs" / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({
            "config_sha256": config.sha256(config.load(ROOT / cfg_path)), "status": "interrupted",
            "git_commit": ident["git_commit"]}))
    state = {"identity": ident, "started_runs": list(a.run_configs(entry))}
    items = a.resume_runs(state, entry, None, tmp_path / "attempt")
    assert not marker.exists(), "run.py --resume latest was started for a run without crash-consistent checkpoints"
    assert items and all(i["status"] == "BLOCKED" and "not crash-consistent" in i["detail"] for i in items)
