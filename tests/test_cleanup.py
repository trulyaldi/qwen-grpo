"""scripts/run.py remove on synthetic artifacts in a temporary root: dry run, deletion, ownership, references, symlinks,
locks, git-tracked files. No real results are touched; only the Qwen3-0.6B config file is read (no weights)."""
import copy
import dataclasses
import json
import os
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
    lock = lock_path(root / "results", "qwen")
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
    from qwen_grpo import trainer
    monkeypatch.setattr(trainer, "RESULTS", tmp_path)
    seen = []
    monkeypatch.setattr(trainer, "_train", lambda cfg, resume, device: seen.append(lock_path(tmp_path, cfg.run.name).read_text()))
    cfg = make_cfg(tmp_path)
    trainer.train(cfg)
    assert seen == [str(os.getpid())] and not lock_path(tmp_path, "qwen").exists()
    lock_path(tmp_path, "qwen").write_text(str(os.getpid()))
    with pytest.raises(SystemExit, match="already being trained"):
        trainer.train(dataclasses.replace(cfg))
