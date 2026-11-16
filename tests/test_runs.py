"""runs/<name>/: path resolution in both layouts, run identity and status, summary and report rebuilt from saved files
(no model), and the run-based CLI. Training is the tiny CPU stand-in of tests/test_trainer.py, in a temporary ROOT."""
import json
import shutil
from pathlib import Path

import pytest

from qwen_grpo import checkpointing, runs, trainer
from scripts import run
from tests.test_trainer import float32_default, make_cfg, sandbox  # noqa: F401 (fixtures)
from tests.test_verification import synthetic_evidence

REPO = Path(__file__).parents[1]


def test_checkpoint_paths_in_both_layouts(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    new, old = runs.checkpoint_path("qwen", 30), runs.legacy_checkpoint("qwen", 30)
    assert new == tmp_path / "runs" / "qwen" / "checkpoints" / "step30" and old == tmp_path / "checkpoints" / "qwen_step30"
    assert runs.parse_checkpoint(new) == ("qwen", 30, "run") and runs.parse_checkpoint(old) == ("qwen", 30, "legacy")
    assert runs.parse_checkpoint(tmp_path / "elsewhere" / "checkpoints" / "step3") == (None, None, None)  # not under runs/
    assert runs.parse_checkpoint(old.parent / "qwen_step3_step10") == ("qwen_step3", 10, "legacy")
    assert runs.sibling(new, 0) == runs.checkpoint_path("qwen", 0) and runs.sibling(old, 0) == runs.legacy_checkpoint("qwen", 0)
    assert runs.training_log_for(new) == tmp_path / "runs" / "qwen" / "training" / "training.json"
    assert runs.training_log_for(old) == tmp_path / "results" / "qwen" / "training.json"
    assert runs.evaluation_output(new, "test") == tmp_path / "runs" / "qwen" / "evaluation" / "step30" / "test.json"
    assert runs.evaluation_output(old, "test") == tmp_path / "results" / "eval" / "qwen_step30" / "test.json"  # unchanged
    assert runs.verification_output(new) == (runs.verification_dir("qwen", 30), runs.verification_dir("qwen", 30) / "evidence")
    assert runs.verification_output(old) == (tmp_path / "results" / "verification" / "qwen_step30",) * 2


def test_latest_checkpoint_and_resolve(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no run 'qwen'"):
        runs.resolve("qwen")
    for step in (0, 10, 2):
        runs.write_json(runs.checkpoint_path("qwen", step) / "adapter_config.json", {})
    runs.checkpoint_path("qwen", 99).mkdir()  # no adapter: not a checkpoint
    assert [s for s, _ in runs.checkpoints("qwen")] == [0, 2, 10] and runs.latest_checkpoint("qwen") == runs.checkpoint_path("qwen", 10)
    assert runs.resolve("qwen") == (runs.run_dir("qwen") / "config.yaml", runs.checkpoint_path("qwen", 10))
    assert runs.resolve("qwen", 2)[1] == runs.checkpoint_path("qwen", 2)
    with pytest.raises(SystemExit, match=r"no checkpoint at step 5 \(available: \[0, 2, 10\]\)"):
        runs.resolve("qwen", 5)
    runs.write_json(runs.legacy_results("old") / "training.json", {})
    with pytest.raises(SystemExit, match="before runs/ existed: use --config and --checkpoint"):
        runs.resolve("old")


def test_legacy_outputs_match_names_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    for p in ("checkpoints/qwen_step10", "checkpoints/qwen_step3_step10", "checkpoints/qwen_lr0_step10", "results/qwen.v2"):
        (tmp_path / p).mkdir(parents=True)
    assert runs.legacy_outputs("qwen") == [tmp_path / "checkpoints" / "qwen_step10"]
    assert runs.legacy_outputs("other") == []


def test_new_run_identity_and_name_collisions(sandbox, tmp_path):
    root = sandbox("r")
    cfg = make_cfg(tmp_path, max_steps=2)
    (root / "checkpoints" / "tiny_step30").mkdir(parents=True)  # a run named tiny from before runs/ existed
    with pytest.raises(SystemExit, match="belongs to a run from before runs/ existed"):
        trainer.train(cfg, device="cpu")
    assert not runs.run_dir("tiny").exists()  # refused before creating anything
    shutil.rmtree(root / "checkpoints")
    trainer.train(cfg, device="cpu")
    m = runs.manifest("tiny")
    assert m["run"] == "tiny" and m["id"].startswith("tiny-") and m["id"].endswith(m["config_sha256"][:8])
    assert m["status"] == "completed" and len(m["git_commit"]) == 40 and m["created"]
    assert {"config.yaml", "summary.json", "report.md", "training/training.json", "training/training.log",
            "checkpoints/step2/training_state.pt"} <= set(m["artifacts"])
    assert all((runs.run_dir("tiny") / a).is_file() for a in m["artifacts"])
    with pytest.raises(SystemExit, match="already exists"):
        trainer.train(cfg, device="cpu")  # the same run.name again: refused, the run is untouched
    assert runs.manifest("tiny") == m


def test_a_run_killed_without_cleanup_reads_as_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    runs.create("x", "0" * 64, {"git_commit": None, "git_dirty": None})
    assert runs.status("x", checkpointing.lock_owner) == "interrupted"  # recorded running, no live process
    runs.lock_path("x").write_text(str(__import__("os").getpid()))
    assert runs.status("x", checkpointing.lock_owner) == "running"


def test_report_is_rebuilt_from_saved_files_without_a_model(sandbox, tmp_path, monkeypatch, capsys):
    root = sandbox("r")
    trainer.train(make_cfg(tmp_path), device="cpu")
    d = runs.run_dir("tiny")
    ev = {"metrics": {"n": 4, "accuracy": 0.75, "macro_f1": 0.7, "recall_NEGATIVE": 1.0, "recall_POSITIVE": 0.5,
                      "predicted_positive_rate": 0.25, "invalid_rate": 0.0},
          "auc": {"trained": 0.8, "base": 0.6}, "vs_baseline": {"accuracy_diff": 0.25, "ci95": [-0.25, 0.75]}}
    runs.write_json(runs.evaluation_path("tiny", 4, "validation"), ev)
    from qwen_grpo import reporting
    reporting.save(synthetic_evidence("runs/tiny/training/training.json"), runs.verification_dir("tiny", 4) / "evidence")
    for f in ("summary.json", "report.md"):
        (d / f).unlink()
    shutil.rmtree(d / "plots")

    def no_model(*a, **k):
        raise AssertionError("report must not load a model")
    monkeypatch.setattr(trainer, "lora_model", no_model)
    run.main(["report", "--run", "tiny"])
    s = runs.read_json(d / "summary.json")
    assert s["status"] == "completed" and s["progress"]["optimizer_steps"] == 4 and s["latest_checkpoint"] == "checkpoints/step4"
    assert s["evaluations"]["step4"]["validation"]["accuracy"] == 0.75 and s["evaluations"]["step4"]["validation"]["auc"] == 0.8
    v = s["verification"]["step4"]
    assert v["negative_control"] == "NOT_RUN" and v["report"] == "verification/step4/report.md"
    assert s["full_validation"] is None and s["memory"]["peak_reserved_gib"] is None  # not measured: None, not invented
    assert {"plots/reward.png", "plots/prediction_bias.png"} <= set(s["artifacts"]["plots"])
    report = (d / "report.md").read_text()
    assert "[validation](evaluation/step4/validation.json)" in report and "NOT_RUN" in report
    assert "Not available: evaluation.full_validation is false" in report and "peak reserved not recorded" in report
    assert (runs.verification_dir("tiny", 4) / "report.md").is_file()  # the per-checkpoint report is rebuilt too
    assert "runs/tiny/summary.json" in capsys.readouterr().out


def test_cli_accepts_run_names_and_the_older_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    runs.write_json(runs.checkpoint_path("qwen", 10) / "adapter_config.json", {})
    runs.write_json(runs.checkpoint_path("qwen", 20) / "adapter_config.json", {})
    shutil.copy(REPO / "configs" / "default.yaml", runs.run_dir("qwen") / "config.yaml")
    calls = []
    monkeypatch.setattr(run, "evaluate_checkpoint", lambda cfg, c, split: calls.append((split, cfg.run.name, runs.rel(c))))
    monkeypatch.setattr(run, "verify", lambda cfg, c: calls.append(("verify", cfg.run.name, runs.rel(c))))
    monkeypatch.setattr("qwen_grpo.cleanup.remove_run", lambda name, confirm: calls.append(("remove", name, confirm)))
    run.main(["validate", "--run", "qwen"])
    run.main(["test", "--run", "qwen", "--step", "10"])
    run.main(["verify", "--run", "qwen"])
    run.main(["remove", "--run", "qwen"])
    run.main(["remove", "--run", "qwen", "--confirm"])
    legacy = tmp_path / "checkpoints" / "grpo_lora_last2_b2_balanced_step30"  # a run from before runs/, historical config
    runs.write_json(legacy / "adapter_config.json", {})
    run.main(["validate", "--config", str(REPO / "configs" / "default.yaml"), "--checkpoint", str(legacy)])
    assert calls == [("validation", "default", "runs/qwen/checkpoints/step20"), ("test", "default", "runs/qwen/checkpoints/step10"),
                     ("verify", "default", "runs/qwen/checkpoints/step20"), ("remove", "qwen", False), ("remove", "qwen", True),
                     ("validation", "default", "checkpoints/grpo_lora_last2_b2_balanced_step30")]
    for argv in (["verify"], ["verify", "--run", "qwen", "--config", "c.yaml"], ["verify", "--config", "c.yaml"],
                 ["verify", "--config", "c.yaml", "--checkpoint", "x", "--step", "3"], ["train", "--run", "qwen"]):
        with pytest.raises(SystemExit):
            run.parse(argv)


def test_summary_paths_resolve_inside_the_run(sandbox, tmp_path):
    sandbox("r")
    trainer.train(make_cfg(tmp_path, max_steps=2), device="cpu")
    s = json.loads((runs.run_dir("tiny") / "summary.json").read_text())
    paths = [c["path"] for c in s["checkpoints"]] + [v for k, v in s["artifacts"].items() if k != "plots"] + s["artifacts"]["plots"]
    assert all((runs.run_dir("tiny") / p).exists() for p in paths)


def test_full_validation_sections_from_a_real_final_block(sandbox, tmp_path):
    """The end-of-training comparison as a GPU run records it (taken from the committed runs/qwen training log)."""
    sandbox("r")
    trainer.train(make_cfg(tmp_path, max_steps=2), device="cpu")
    real = json.loads((REPO / "runs" / "qwen" / "training" / "training.json").read_text())
    bench = {**real["final"], "memory": real["memory"]}
    path = runs.training_dir("tiny") / "training.json"
    log = json.loads(path.read_text())
    log["final"] = {k: bench[k] for k in ("base", "trained", "accuracy_diff", "bootstrap_ci95", "bootstrap", "paired_changes",
                                          "vs_saved", "auc", "auc_vs")}
    log["memory"] = bench["memory"]
    path.write_text(json.dumps(log))
    run.main(["report", "--run", "tiny"])
    s = runs.read_json(runs.run_dir("tiny") / "summary.json")
    fv = s["full_validation"]
    assert fv["trained"]["accuracy"] == bench["trained"]["accuracy"] and fv["trained"]["auc"] == bench["auc"]["trained"]
    assert fv["ci95"] == bench["bootstrap_ci95"] and fv["auc_diff"] == bench["auc_vs"]["base"]["auc_diff"]
    assert s["memory"]["peak_reserved_gib"] == bench["memory"]["peak_reserved_gib"]
    report = (runs.run_dir("tiny") / "report.md").read_text()
    assert f"| accuracy | {bench['base']['accuracy']:.4f} | {bench['trained']['accuracy']:.4f} |" in report
    assert (f"predicted-POSITIVE rate base {bench['base']['predicted_positive_rate']:.4f} → trained "
            f"{bench['trained']['predicted_positive_rate']:.4f}") in report
    assert f"peak reserved {bench['memory']['peak_reserved_gib']:.2f} GiB" in report


def test_report_of_a_run_that_failed_before_its_first_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    runs.create("early", "0" * 64, {"git_commit": None, "git_dirty": None})
    runs.set_status("early", "failed", error="RuntimeError: CUDA out of memory")
    shutil.copy(REPO / "configs" / "default.yaml", runs.run_dir("early") / "config.yaml")
    run.main(["report", "--run", "early"])
    s = runs.read_json(runs.run_dir("early") / "summary.json")
    assert s["status"] == "failed" and s["checkpoints"] == [] and s["training_log"] is None
    assert "No training log is available" in (runs.run_dir("early") / "report.md").read_text()
