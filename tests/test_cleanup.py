"""scripts/run.py remove on synthetic artifacts in a temporary root: dry run, deletion, ownership, references, symlinks,
locks, git-tracked files. No real results are touched; only the Qwen3-0.6B config file is read (no weights)."""
import copy
import dataclasses
import json
import os
import re
import subprocess
import sys

import pytest
import yaml

from qwen_grpo import cleanup, config
from qwen_grpo.checkpointing import lock_path
from tests.test_config import RAW

TARGET = r"model\.layers\.(26|27)\.self_attn\.(q|v|o)_proj"


def make_cfg(tmp_path, name="qwen", **training):
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = name
    raw["training"].update(training)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def fake_run(root, cfg, steps=(0, 10), r=8):
    """The files train / validate / verify write for a run, with small synthetic contents."""
    n = cfg.run.name
    for s in steps:
        c = root / "checkpoints" / f"{n}_step{s}"
        write(c / "adapter_config.json", {"r": r, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": TARGET,
                                          "base_model_name_or_path": "Qwen/Qwen3-0.6B"})
        write(c / "adapter_model.safetensors", "x" * 1000)
    last = f"checkpoints/{n}_step{steps[-1]}"
    write(root / "results" / n / "training.json", {
        "yaml_config": config.as_dict(cfg), "config": {"initial_checkpoint": f"checkpoints/{n}_step0"},
        "evaluations": [{"step": s, "checkpoint": None if s == 0 else f"checkpoints/{n}_step{s}"} for s in steps],
        "final_checkpoint": {"path": last}})
    write(root / "results" / n / "plots" / "loss.png", "png")
    write(root / "results" / "benchmarks" / f"{n}.json", "{}")
    write(root / "results" / "benchmarks" / f"{n}.md", "# md")
    write(root / "results" / "eval" / f"{n}_step{steps[-1]}" / "validation.json", {"checkpoint": last})
    write(root / "results" / "verification" / f"{n}_step{steps[-1]}" / "manifest.json", {"checkpoint": last})
    return root / last


@pytest.fixture
def repo(tmp_path):
    """A synthetic repository: run 'qwen', look-alike neighbours, shared files."""
    root = tmp_path / "repo"
    cfg = make_cfg(tmp_path)
    ckpt = fake_run(root, cfg)
    fake_run(root, make_cfg(tmp_path, "qwen_lr0"))
    fake_run(root, make_cfg(tmp_path, "qwen.v2"))
    fake_run(root, make_cfg(tmp_path, "qwen_step3"))  # writes checkpoints/qwen_step3_step10: not a 'qwen' checkpoint
    write(root / "results" / "baseline" / "validation.json", "{}")
    write(root / "configs" / "default.yaml", "run: {name: default}\n")
    write(root / "README.md", "see checkpoints/qwen_lr0_step10 and results/qwen.v2/ and checkpoints/qwen_step100\n")
    return root, cfg, ckpt


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None) for p in root.rglob("*")}


QWEN = ["checkpoints/qwen_step0", "checkpoints/qwen_step10", "results/qwen", "results/benchmarks/qwen.json",
        "results/benchmarks/qwen.md", "results/eval/qwen_step10", "results/verification/qwen_step10"]


def test_dry_run_lists_exactly_the_run_with_sizes_and_changes_nothing(repo, capsys):
    root, cfg, ckpt = repo
    before = snapshot(root)
    items = cleanup.remove(cfg, root, ckpt)
    assert sorted(p.relative_to(root).as_posix() for p, _ in items) == sorted(QWEN)
    sizes = dict((p.relative_to(root).as_posix(), n) for p, n in items)
    assert sizes["checkpoints/qwen_step10"] == len(json.dumps({"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": TARGET,
                                                               "base_model_name_or_path": "Qwen/Qwen3-0.6B"})) + 1000
    out = capsys.readouterr().out
    assert "dry run: nothing deleted" in out and "total, 7 paths" in out
    assert snapshot(root) == before


def test_confirm_deletes_only_the_run(repo):
    root, cfg, ckpt = repo
    before = snapshot(root)
    cleanup.remove(cfg, root, ckpt, confirm=True)
    after = snapshot(root)
    gone = {k for k in before if k not in after}
    assert all(any(k == q or k.startswith(q + "/") for q in QWEN) for k in gone)
    assert all(k in gone for k in QWEN)
    for kept in ("checkpoints/qwen_lr0_step10", "checkpoints/qwen_step3_step10", "results/qwen.v2", "results/benchmarks/qwen.v2.json",
                 "results/eval/qwen_lr0_step10", "results/baseline/validation.json", "configs/default.yaml", "results/benchmarks"):
        assert kept in after, kept
    assert all(after[k] == before[k] for k in after)


def test_rejects_checkpoints_of_other_runs_and_paths_outside_checkpoints(repo, tmp_path):
    root, cfg, _ = repo
    for bad in (root / "checkpoints" / "qwen_lr0_step10", root / "checkpoints" / "qwen_step3_step10",
                root / "checkpoints" / ".." / "results" / "qwen", root / "results" / "qwen_step10", tmp_path / "qwen_step10"):
        with pytest.raises(SystemExit, match="is not a checkpoint of run 'qwen'"):
            cleanup.plan(cfg, root, bad)


def test_rejects_a_config_or_adapter_that_differs_from_the_run(repo, tmp_path):
    root, cfg, ckpt = repo
    with pytest.raises(SystemExit, match="different config"):
        cleanup.plan(make_cfg(tmp_path, learning_rate=2e-5), root, ckpt)
    write(root / "checkpoints" / "qwen_step20" / "adapter_config.json", {"r": 4, "lora_alpha": 16, "lora_dropout": 0.0,
          "target_modules": TARGET, "base_model_name_or_path": "Qwen/Qwen3-0.6B"})
    with pytest.raises(SystemExit, match="qwen_step20: .*does not match the config"):
        cleanup.plan(cfg, root, ckpt)


def test_unlisted_checkpoint_and_foreign_evaluation_are_rejected(repo):
    root, cfg, ckpt = repo
    fake_run(root, cfg, steps=(0, 10))
    write(root / "checkpoints" / "qwen_step20" / "adapter_config.json",
          json.loads((ckpt / "adapter_config.json").read_text()))
    with pytest.raises(SystemExit, match="qwen_step20 is not listed in"):
        cleanup.plan(cfg, root, ckpt)
    (root / "checkpoints" / "qwen_step20" / "training_state.pt").write_text("state")  # a crashed run's checkpoint: allowed
    cleanup.plan(cfg, root, ckpt)
    write(root / "results" / "eval" / "qwen_step10" / "validation.json", {"checkpoint": "checkpoints/qwen_lr0_step10"})
    with pytest.raises(SystemExit, match="names a checkpoint of another run"):
        cleanup.plan(cfg, root, ckpt)


def test_missing_artifacts_are_fine_but_some_evidence_is_required(repo):
    root, cfg, ckpt = repo
    import shutil
    for p in ("results/eval/qwen_step10", "results/verification/qwen_step10", "results/benchmarks/qwen.md"):
        shutil.rmtree(root / p) if (root / p).is_dir() else (root / p).unlink()
    shutil.rmtree(ckpt)  # the named checkpoint itself is already gone; the training log still identifies the run
    items = cleanup.remove(cfg, root, ckpt, confirm=True)
    assert sorted(p.name for p, _ in items) == ["qwen", "qwen.json", "qwen_step0"]
    with pytest.raises(SystemExit, match="nothing confirms"):
        cleanup.plan(cfg, root, ckpt)


def test_references_from_other_experiments_block_removal(repo):
    root, cfg, ckpt = repo
    write(root / "results" / "verification" / "other_step30" / "verification" / "negative_control.json",
          {"checkpoint": "checkpoints/qwen_step10"})
    with pytest.raises(SystemExit, match="other_step30/verification/negative_control.json references checkpoints/qwen_step10"):
        cleanup.plan(cfg, root, ckpt)
    (root / "results" / "verification" / "other_step30" / "verification" / "negative_control.json").unlink()
    write(root / "configs" / "control.yaml", "verification: {negative_control: {checkpoint: checkpoints/qwen_step0}}\n")
    with pytest.raises(SystemExit, match="configs/control.yaml references checkpoints/qwen_step0"):
        cleanup.plan(cfg, root, ckpt)


def test_symlinks_are_refused_and_never_followed(repo, tmp_path):
    root, cfg, ckpt = repo
    outside = write(tmp_path / "precious.txt", "keep me")
    os.symlink(outside, ckpt / "link")
    with pytest.raises(SystemExit, match="symbolic link checkpoints/qwen_step10/link"):
        cleanup.plan(cfg, root, ckpt)
    (ckpt / "link").unlink()
    os.symlink(root / "results" / "qwen_lr0", root / "results" / "eval" / "qwen_step20")
    with pytest.raises(SystemExit, match="symbolic link results/eval/qwen_step20"):
        cleanup.plan(cfg, root, ckpt)
    assert outside.read_text() == "keep me" and (root / "results" / "qwen_lr0" / "training.json").exists()


def test_live_lock_refuses_and_stale_lock_is_removed(repo):
    root, cfg, ckpt = repo
    lock = lock_path("qwen", root / "results")
    lock.write_text(str(os.getpid()))  # this process is alive: the run counts as training
    with pytest.raises(SystemExit, match="being trained by process"):
        cleanup.plan(cfg, root, ckpt)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lock.write_text(str(dead.pid))
    items = cleanup.remove(cfg, root, ckpt, confirm=True)
    assert lock in [p for p, _ in items] and not lock.exists()


def test_git_tracked_artifacts_are_refused(repo):
    root, cfg, ckpt = repo
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=root, check=True, capture_output=True)
    git("init", "-q")
    git("add", "results/benchmarks/qwen.json")
    git("commit", "-qm", "x")
    with pytest.raises(SystemExit, match="results/benchmarks/qwen.json is committed to git"):
        cleanup.plan(cfg, root, ckpt)


def test_output_directory_symlink_is_refused(repo, tmp_path):
    root, cfg, ckpt = repo
    elsewhere = tmp_path / "elsewhere"
    (root / "results" / "eval").rename(elsewhere)
    os.symlink(elsewhere, root / "results" / "eval")
    with pytest.raises(SystemExit, match="symbolic links"):
        cleanup.plan(cfg, root, ckpt)


def test_cli_dispatches_remove_with_dry_run_by_default(monkeypatch):
    from scripts import run
    calls = []
    monkeypatch.setattr("qwen_grpo.cleanup.remove", lambda cfg, root, ckpt, confirm: calls.append((cfg.run.name, ckpt.name, confirm)))
    default = str(run.ROOT / "configs" / "default.yaml")
    run.main(["remove", "--config", default, "--checkpoint", "checkpoints/missing_step3"])  # no adapter_config.json needed
    run.main(["remove", "--config", default, "--checkpoint", "checkpoints/missing_step3", "--confirm"])
    assert calls == [("default", "missing_step3", False), ("default", "missing_step3", True)]


@pytest.mark.parametrize("name", [".", "..", "...", "-x", "_x"])
def test_run_names_that_could_escape_or_hide_are_rejected(tmp_path, name):
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = name
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="run.name"):
        config.load(path)


def test_training_holds_the_lock_and_refuses_a_second_trainer(tmp_path, monkeypatch):
    from qwen_grpo import runs, trainer
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    seen = []
    monkeypatch.setattr(trainer, "_train", lambda cfg, resume, state, device: seen.append(runs.lock_path(cfg.run.name).read_text()))
    cfg = make_cfg(tmp_path)
    trainer.train(cfg)
    assert seen == [str(os.getpid())] and not runs.lock_path("qwen").exists()
    runs.lock_path("qwen").write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="already being trained"):
        trainer.train(dataclasses.replace(cfg))


def test_ownership_follows_the_resolved_config_not_how_the_yaml_is_written(repo, tmp_path):
    """A run trained from a full YAML is owned by an equivalent `extends` config; a parent that changes a training key
    makes the same child config a different one."""
    root, cfg, ckpt = repo
    parent = tmp_path / "configs" / "default.yaml"
    parent.parent.mkdir()
    parent.write_text(yaml.safe_dump({**RAW, "run": {"name": "default"}}))
    child = tmp_path / "configs" / "experiments" / "qwen.yaml"
    child.parent.mkdir()
    child.write_text("extends: ../default.yaml\nrun: {name: qwen}\n")
    assert config.load(child) == cfg
    assert [p.name for p, _ in cleanup.plan(config.load(child), root, ckpt)] == [p.name for p, _ in cleanup.plan(cfg, root, ckpt)]
    parent.write_text(yaml.safe_dump({**RAW, "run": {"name": "default"}, "training": {**RAW["training"], "seed": 7}}))
    with pytest.raises(SystemExit, match="different config"):
        cleanup.plan(config.load(child), root, ckpt)


# --- remove --run: runs/<name>/ ----------------------------------------------------------------------------------

@pytest.fixture
def run_repo(tmp_path, monkeypatch):
    """A synthetic runs/ with run 'qwen', look-alike neighbours and shared files."""
    from qwen_grpo import runs
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    for name in ("qwen", "qwen_lr0", "qwen.v2"):
        runs.create(name, "0" * 64, {"git_commit": None, "git_dirty": None})
        write(runs.checkpoint_path(name, 10) / "adapter_model.safetensors", "x" * 1000)
        write(runs.run_dir(name) / "report.md", "# report")
    write(tmp_path / "configs" / "qwen.yaml", "run: {name: qwen}\n")
    write(tmp_path / "results" / "baseline" / "validation.json", "{}")
    return tmp_path, runs


def test_remove_run_dry_run_then_confirm_deletes_only_that_run(run_repo, capsys):
    root, runs = run_repo
    runs.lock_path("qwen").write_text("999999999")  # stale: no such process
    items = cleanup.remove_run("qwen")
    assert [p for p, _ in items] == [runs.run_dir("qwen"), runs.lock_path("qwen")] and runs.run_dir("qwen").exists()
    out = capsys.readouterr().out
    assert "dry run" in out and "runs/qwen/" in out and "total, 2 paths" in out and "KiB" in out
    cleanup.remove_run("qwen", confirm=True)
    assert not runs.run_dir("qwen").exists() and not runs.lock_path("qwen").exists()
    assert runs.run_dir("qwen_lr0").exists() and runs.run_dir("qwen.v2").exists()
    assert (root / "configs" / "qwen.yaml").exists() and (root / "results" / "baseline" / "validation.json").exists()


@pytest.mark.parametrize("break_it, message", [
    (lambda root, runs: (runs.run_dir("qwen") / "manifest.json").unlink(), "ownership cannot be confirmed"),
    (lambda root, runs: runs.write_json(runs.run_dir("qwen") / "manifest.json", {"run": "other"}), "ownership cannot be confirmed"),
    (lambda root, runs: runs.lock_path("qwen").write_text(str(os.getpid())), "being trained by process"),
    (lambda root, runs: os.symlink(root / "configs", runs.run_dir("qwen") / "link"), "symbolic link runs/qwen/link"),
    (lambda root, runs: write(runs.run_dir("qwen_lr0") / "config.yaml", "checkpoint: runs/qwen/checkpoints/step10\n"),
     "runs/qwen_lr0/config.yaml references runs/qwen"),
])
def test_remove_run_refuses_uncertain_ownership_and_unsafe_paths(run_repo, break_it, message):
    root, runs = run_repo
    break_it(root, runs)
    with pytest.raises(SystemExit, match=re.escape(message)):
        cleanup.remove_run("qwen", confirm=True)
    assert (runs.checkpoint_path("qwen", 10) / "adapter_model.safetensors").exists()


def test_remove_run_refuses_symlinked_run_and_runs_directories(run_repo, tmp_path):
    root, runs = run_repo
    elsewhere = tmp_path / "elsewhere"
    runs.run_dir("qwen").rename(elsewhere)
    os.symlink(elsewhere, runs.run_dir("qwen"))
    with pytest.raises(SystemExit, match="is a symbolic link"):
        cleanup.remove_run("qwen", confirm=True)
    os.unlink(runs.run_dir("qwen"))
    runs.runs_dir().rename(tmp_path / "real_runs")
    os.symlink(tmp_path / "real_runs", runs.runs_dir())
    with pytest.raises(SystemExit, match="passes through a symbolic link"):
        cleanup.remove_run("qwen_lr0", confirm=True)
    assert elsewhere.exists() and (tmp_path / "real_runs" / "qwen_lr0").exists()


@pytest.mark.parametrize("name", ["..", "../qwen", ".qwen", "qwen/../qwen_lr0", ""])
def test_remove_run_rejects_names_that_are_not_run_names(run_repo, name):
    with pytest.raises(SystemExit, match="not a run name"):
        cleanup.remove_run(name, confirm=True)


def test_remove_run_of_a_missing_or_legacy_run(run_repo):
    root, runs = run_repo
    with pytest.raises(SystemExit, match="no run 'gone'"):
        cleanup.remove_run("gone")
    (root / "checkpoints" / "old_step10").mkdir(parents=True)
    with pytest.raises(SystemExit, match="use --config and --checkpoint"):
        cleanup.remove_run("old")


def test_remove_run_refuses_git_tracked_runs(run_repo):
    root, runs = run_repo
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "runs/qwen/report.md"], cwd=root, check=True)
    with pytest.raises(SystemExit, match="committed to git"):
        cleanup.remove_run("qwen", confirm=True)
