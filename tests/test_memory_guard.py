"""memory.guard in the trainer, on CPU: the real Profiler guard thread with injected samples (no GPU needed).

A sample above the budget must stop the run, mark it failed (not interrupted), name the violation, and keep the raw
samples; samples within the budget must let the run finish and still be saved. On CPU without the injection the guard
is not enforced and says so (it never claims a budget it did not measure)."""
import dataclasses
import json
import time

import pytest

from qwen_grpo import runs, trainer
from qwen_grpo.execution import profiler
from tests.test_trainer import make_cfg, sandbox  # noqa: F401 - fixture


def guarded(cfg, budget_gib=1.0):
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, gpu_budget_gib=budget_gib),
                               memory=dataclasses.replace(cfg.memory, guard=True, interval_s=0.01))


def fake_guard(used_bytes, monkeypatch):
    """trainer.memory_guard -> a real Profiler on CPU whose samples report `used_bytes` of device memory."""
    def sample(self):
        return {"t": time.time(), "phase": self._phase, "device_used": used_bytes, "smi_apps_raw": "injected"}
    monkeypatch.setattr(profiler.Profiler, "sample", sample)
    monkeypatch.setattr(trainer, "memory_guard", lambda cfg, device: profiler.Profiler(
        budget_bytes=int(cfg.run.gpu_budget_gib * 2**30), interval=cfg.memory.interval_s, device="cpu", smi=False))


def memory_files(name):
    return sorted((runs.training_dir(name) / "memory").glob("*.json"))


def test_a_sample_above_the_budget_stops_the_run_as_failed(sandbox, tmp_path, monkeypatch):  # noqa: F811
    sandbox("over")
    cfg = guarded(make_cfg(tmp_path, max_steps=50, max_attempts=200))
    fake_guard(2 * 2**30, monkeypatch)  # 2 GiB used, budget 1 GiB
    with pytest.raises(SystemExit, match="GPU memory budget exceeded"):
        trainer.train(cfg, device="cpu")
    m = runs.manifest("tiny")
    assert m["status"] == "failed" and "GPU memory budget exceeded" in m["status_history"][-1]["error"]
    (path,) = memory_files("tiny")
    rep = json.loads(path.read_text())
    assert rep["violations"] and rep["violations"][0]["bytes"] == 2 * 2**30 and rep["budget_gib"] == 1.0
    assert rep["samples"] and rep["samples"][0]["smi_apps_raw"] == "injected"  # raw samples kept


def test_samples_within_the_budget_let_the_run_finish(sandbox, tmp_path, monkeypatch):  # noqa: F811
    sandbox("under")
    cfg = guarded(make_cfg(tmp_path))
    fake_guard(2**20, monkeypatch)
    trainer.train(cfg, device="cpu")
    assert runs.manifest("tiny")["status"] == "completed"
    (path,) = memory_files("tiny")
    assert json.loads(path.read_text())["violations"] == []


def test_the_guard_is_not_claimed_without_cuda(sandbox, tmp_path, capsys):  # noqa: F811
    sandbox("cpu")
    cfg = guarded(make_cfg(tmp_path))
    assert trainer.memory_guard(cfg, "cpu") is None
    trainer.train(cfg, device="cpu")
    assert "memory.guard: not enforced on device 'cpu'" in capsys.readouterr().out
    assert memory_files("tiny") == []


def test_training_log_lines_are_timestamped(sandbox, tmp_path):  # noqa: F811
    sandbox("log")
    trainer.train(make_cfg(tmp_path), device="cpu")
    lines = (runs.training_dir("tiny") / "training.log").read_text().splitlines()
    assert lines and all(line[:4].isdigit() and line[10] == "T" and line[26] == "Z" for line in lines), lines[:3]
