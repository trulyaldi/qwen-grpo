"""Kaggle gpu-train (no network, no Kaggle account): the config is taken from the pinned commit and refused when a T4
cannot run it as written; evidence attribution and the strict pass rule of summary; and checkpoint recovery across
kernel versions: an interrupted tiny run is packed, restored into a fresh "kernel" with every file verified, and resumed
to the uninterrupted run's result, bitwise (CPU)."""
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from qwen_grpo import checkpointing, config, runs, trainer
from tests.test_kaggle import ROOT, runner, submit
from tests.test_trainer import ckpt, make_cfg, sandbox  # noqa: F401 (fixture)

EXP = "configs/experiments"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository holding this repository's configs/, committed; submit.ROOT points at it."""
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", root / "configs")
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    git("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "configs")
    monkeypatch.setattr(submit, "ROOT", root)
    return root, git("rev-parse", "HEAD").strip()


@pytest.mark.parametrize("name, backend, streaming, steps", [
    ("kaggle_06b_fp32_train_resident", "resident", False, 210), ("kaggle_06b_fp32_train_l2l_sync", "l2l_sync", False, 210),
    ("kaggle_06b_fp32_train_l2l_async", "l2l_async", False, 210), ("kaggle_06b_fp32_train_streaming", "l2l_async", True, 210),
    ("kaggle_06b_fp32_pilot", "l2l_async", True, 20)])
def test_the_backend_and_schedule_come_from_the_config(repo, name, backend, streaming, steps):
    root, sha = repo
    spec = submit.train_spec(sha, f"{EXP}/{name}.yaml", 11.0)
    assert (spec["backend"], spec["streaming"], spec["max_steps"], spec["run_name"]) == (backend, streaming, steps, name)
    assert (spec["dtype"], spec["train_samples"], spec["seed"]) == ("float32", 1024, 42)
    assert spec["config_sha256"] == config.sha256(config.load(ROOT / EXP / f"{name}.yaml"))


@pytest.mark.parametrize("path, hours, message", [
    (f"{EXP}/qwen3_0.6b_l2l_async.yaml", 11.0, "no native bfloat16"),          # bf16: refused, never switched
    (f"{EXP}/qwen3_4b_l2l.yaml", 11.0, "gpu-train trains"),                      # 4B: not on a T4
    (f"{EXP}/kaggle_06b_fp32_train_base.yaml", 11.0, "not a runnable config"),   # abstract base
    (f"{EXP}/not_committed.yaml", 11.0, "is not in commit"),
    (f"{EXP}/kaggle_06b_fp32_pilot.yaml", 12.0, "--max-hours"),                  # no time left to copy the output
    ("research/l2l/configs/async.yaml", 11.0, "under configs/"),
])
def test_what_a_t4_session_cannot_run_as_written_is_refused(repo, path, hours, message):
    root, sha = repo
    with pytest.raises(SystemExit, match=message):
        submit.train_spec(sha, path, hours)


def test_the_committed_config_is_used_not_the_working_copy(repo):
    root, sha = repo
    p = root / EXP / "kaggle_06b_fp32_pilot.yaml"
    p.write_text(p.read_text().replace("max_steps: 20", "max_steps: 5"))  # uncommitted edit
    assert submit.train_spec(sha, f"{EXP}/kaggle_06b_fp32_pilot.yaml", 11.0)["max_steps"] == 20


# --- evidence attribution and the pass rule ------------------------------------------------------------------------

def fake_submission(tmp_path, monkeypatch, overall, resolved="c" * 64):
    monkeypatch.setattr(submit, "LEDGER", tmp_path / "submissions.jsonl")
    monkeypatch.setattr(submit, "OUTPUTS", tmp_path / "outputs")
    sub = {"submission_id": "s1", "kernel_id": "u/qwen-grpo-gpu-train", "mode": "gpu-train", "commit": "a" * 40,
           "train": {"config": f"{EXP}/kaggle_06b_fp32_pilot.yaml", "config_sha256": "c" * 64, "run_name": "kaggle_06b_fp32_pilot"},
           "kernel_version": 3}
    (tmp_path / "submissions.jsonl").write_text(json.dumps(sub) + "\n")
    ev = tmp_path / "outputs" / "gpu-train" / "s1" / "evidence"
    ev.mkdir(parents=True)
    (ev / "git_commit.txt").write_text("a" * 40 + "\n")
    (ev / "results.json").write_text(json.dumps({"submission": sub, "utc_start": "x", "utc_end": "y", "overall": overall,
                                                "resolved_config_sha256": resolved, "steps": [{"name": "train", "status": overall}]}))
    return sub


@pytest.mark.parametrize("overall, resolved, ok", [
    ("PASS", "c" * 64, True),
    ("PASS", "d" * 64, False),            # Kaggle resolved another config than the one submitted
    ("INCOMPLETE", "c" * 64, False),      # interrupted (time budget): resumable, never a pass
    ("PASS_WITH_SKIPS", "c" * 64, False),  # a training run has no skippable part
    ("FAIL", "c" * 64, False),
])
def test_summary_passes_only_an_attributed_completed_run(tmp_path, monkeypatch, capsys, overall, resolved, ok):
    fake_submission(tmp_path, monkeypatch, overall, resolved)
    with pytest.raises(SystemExit) as e:
        submit.summary(submit.argparse.Namespace(mode="gpu-train", submission=None))
    assert (e.value.code == 0) == ok
    if overall == "INCOMPLETE":
        assert "--resume-from s1" in capsys.readouterr().out


def test_the_runner_never_reports_an_incomplete_or_skipped_run_as_passed(monkeypatch):
    for steps, expected in [([{"status": "PASS"}, {"status": "INCOMPLETE"}], "INCOMPLETE"),
                            ([{"status": "INCOMPLETE"}, {"status": "FAIL"}], "FAIL")]:
        monkeypatch.setattr(runner, "RESULTS", steps)
        assert runner.overall() == expected


def test_a_resume_submission_renders_its_dataset_and_carries_the_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "BUILD", tmp_path / "build")
    sub = {"submission_id": "s2", "kernel_id": "u/qwen-grpo-gpu-train", "mode": "gpu-train", "commit": "b" * 40,
           "repo_url": "https://github.com/x/y.git", "train": {"run_name": "r"},
           "resume": {"dataset": "u/qwen-grpo-resume-r", "marker_sha256": "e" * 64, "target": "latest"}}
    folder = submit.render("gpu-train", sub)
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["u/qwen-grpo-resume-r"] and meta["enable_gpu"] and meta["is_private"]
    assert '"marker_sha256": "' + "e" * 64 in (folder / "run.py").read_text()


# --- checkpoint recovery across kernel versions ------------------------------------------------------------------

def test_an_interrupted_run_is_packed_restored_verified_and_resumed_bitwise(sandbox, tmp_path, monkeypatch):  # noqa: F811
    cfg = make_cfg(tmp_path)
    sandbox("full")
    ref = trainer.train(cfg, device="cpu")

    part = sandbox("part")  # kernel version 1: stopped after the step-2 checkpoint (the time budget's SIGINT)
    save = checkpointing.save_checkpoint

    def stop_after_step2(model, optimizer, path, st):
        save(model, optimizer, path, st)
        if st["step"] == 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "save_checkpoint", stop_after_step2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cfg, device="cpu")
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    run_v1 = part / "runs" / "tiny"
    perf = runner.perf_summary(run_v1, tmp_path)
    assert runner.train_status(1, perf, run_v1) == ("INCOMPLETE", "interrupted at 2/4 steps; resumable from step 2")

    target, step = submit.resume_target(run_v1, streaming=False)  # local: after `download`
    assert (target, step) == ("runs/tiny/checkpoints/step2", 2)
    marker = submit.pack_run(run_v1, tmp_path / "dataset")
    marker_sha = hashlib.sha256((tmp_path / "dataset" / submit.RESUME_MARKER).read_bytes()).hexdigest()

    kernel = sandbox("kernel")  # kernel version 2: a fresh clone, the dataset attached under /kaggle/input/
    ok, detail = runner.restore_run(tmp_path / "dataset", kernel, "0" * 64, "tiny")
    assert not ok and "none with the submitted sha256" in detail  # a stale dataset version is refused
    ok, detail = runner.restore_run(tmp_path / "dataset", kernel, marker_sha, "tiny")
    assert ok, detail
    resumed_cfg = config.load(kernel / "runs" / "tiny" / "config.yaml", historical=True)
    out = trainer.train(resumed_cfg, resume=kernel / target, device="cpu")
    keys = ("attempt", "rollout_sha256", "rewards", "loss", "grad_norm", "update_norm", "change_from_init")
    assert [[r[k] for k in keys] for r in out["training"]] == [[r[k] for k in keys] for r in ref["training"]]
    assert [e["predictions"] for e in out["evaluations"]] == [e["predictions"] for e in ref["evaluations"]]
    perf = runner.perf_summary(kernel / "runs" / "tiny", tmp_path)
    assert runner.train_status(0, perf, kernel / "runs" / "tiny") == ("PASS", None)
    assert perf["steps"] == 4 and [h["status"] for h in perf["status_history"]][-2:] == ["running", "completed"]
    assert {"rollout_s", "step_s"} <= set(perf["timings_total_s"]) and perf["cpu_rss_max_bytes"]
    assert perf["evaluations"][0].keys() >= {"accuracy", "predicted_positive_rate"} and len(marker["files"]) > 5

    # a packed file changed after the marker was written is refused
    bad = tmp_path / "bad"
    shutil.copytree(tmp_path / "dataset", bad)
    with tarfile.open(bad / "run.tar", "a") as t:
        extra = tmp_path / "manifest.json"
        extra.write_text("{}")
        t.add(extra, arcname="runs/tiny/manifest.json")  # a second, different member: the later one wins on extract
    ok, detail = runner.restore_run(bad, sandbox("kernel2"), marker_sha, "tiny")
    assert not ok and "differ from the marker" in detail


def test_a_run_without_a_resumable_checkpoint_is_a_failure(tmp_path):
    run = tmp_path / "runs" / "r"
    (run / "checkpoints" / "step0").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"status": "interrupted"}))
    perf = runner.perf_summary(run, tmp_path)
    assert runner.train_status(1, perf, run)[0] == "FAIL"
    assert submit.resume_target(run, streaming=False)[0] is None
