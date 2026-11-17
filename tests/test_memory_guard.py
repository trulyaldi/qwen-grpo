"""memory.guard in the trainer: a sample above run.gpu_budget_gib stops the run with status failed and keeps the raw
samples; samples within budget let it complete. Runs on CPU: the Profiler's sample() is replaced by fake readings (the
real sampler needs CUDA); everything else (sampler thread, violation, interrupt, trainer handling) is the real code."""
import dataclasses
import json
import time

import pytest

from qwen_grpo import config, runs, trainer
from qwen_grpo.execution import profiler
from tests.test_trainer import make_cfg, sandbox  # noqa: F401  (fixture)


def guarded(cfg, budget_gib=1.0):
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, gpu_budget_gib=budget_gib),
                               memory=config.Memory(guard=True, interval_s=0.01))


def fake_guard(monkeypatch, used_bytes):
    def sample(self):
        return {"t": time.time(), "phase": self._phase, "device_used": used_bytes, "smi_gpu_raw": "fake",
                "process_used": used_bytes}
    monkeypatch.setattr(profiler.Profiler, "sample", sample)
    monkeypatch.setattr(trainer, "memory_guard", lambda cfg, device: profiler.Profiler(
        budget_bytes=int(cfg.run.gpu_budget_gib * 2**30), interval=cfg.memory.interval_s, device="cpu"))


def memory_files(name):
    return sorted((runs.training_dir(name) / "memory").glob("*.json"))


def test_a_sample_above_the_budget_stops_the_run(sandbox, tmp_path, monkeypatch):  # noqa: F811
    sandbox("over")
    cfg = guarded(make_cfg(tmp_path))
    fake_guard(monkeypatch, used_bytes=2 * 2**30)  # 2 GiB > 1 GiB budget
    with pytest.raises(SystemExit, match="GPU memory budget exceeded"):
        trainer.train(cfg, device="cpu")
    assert runs.manifest("tiny")["status"] == "failed"
    [f] = memory_files("tiny")
    rec = json.loads(f.read_text())
    assert rec["violations"] and rec["violations"][0]["bytes"] == 2 * 2**30 and rec["samples"]


def test_samples_within_the_budget_let_the_run_complete(sandbox, tmp_path, monkeypatch):  # noqa: F811
    sandbox("within")
    cfg = guarded(make_cfg(tmp_path), budget_gib=4.0)
    fake_guard(monkeypatch, used_bytes=2**30)
    trainer.train(cfg, device="cpu")
    assert runs.manifest("tiny")["status"] == "completed"
    [f] = memory_files("tiny")
    rec = json.loads(f.read_text())
    assert rec["violations"] == [] and len(rec["samples"]) >= 1 and rec["budget_gib"] == 4.0


def test_the_guard_is_not_claimed_without_cuda(capsys):
    cfg = guarded(config.load("configs/profiles/qwen3_4b.yaml"))
    assert trainer.memory_guard(cfg, "cpu") is None
    assert "not enforced" in capsys.readouterr().out


def test_training_log_lines_are_timestamped(sandbox, tmp_path):  # noqa: F811
    sandbox("log")
    trainer.train(make_cfg(tmp_path), device="cpu")
    lines = (runs.training_dir("tiny") / "training.log").read_text().splitlines()
    assert lines and all(len(line) >= 27 and line[4] == "-" and line[26] == "Z" for line in lines)
