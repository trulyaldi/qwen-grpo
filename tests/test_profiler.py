"""execution/profiler.py logic on CPU (no CUDA here): phase windows, CPU memory, the guard, evidence grades."""
import time

import pytest

from qwen_grpo.execution import profiler as P


def test_phases_record_wall_time_and_cpu_memory_and_cpu_only_evidence_is_none():
    with P.Profiler(budget_bytes=None, interval=0.01, device="cpu") as prof:
        with prof.phase("work"):
            buf = bytearray(20 * 2**20)
            time.sleep(0.05)
            del buf
    rep = prof.report(raw=False)
    (ph,) = rep["phases"]
    assert ph["phase"] == "work" and ph["seconds"] >= 0.05 and ph["samples"] >= 1 and ph["cpu_rss_peak"] > 0
    assert rep["evidence"] == "none"  # no GPU reading at all: never enough for a budget claim


def test_the_guard_records_a_violation_and_interrupts_the_main_thread(monkeypatch):
    monkeypatch.setattr(P.Profiler, "sample", lambda self: {"t": time.time(), "phase": self._phase,
                                                            "device_used": 9 * 2**30, "process_used": None})
    prof = P.Profiler(budget_bytes=8 * 2**30, interval=0.01, device="cpu")
    with pytest.raises(KeyboardInterrupt):
        with prof:
            time.sleep(2)
    assert prof.violations and prof.violations[0]["source"] == "cudaMemGetInfo device"


def test_evidence_is_assignment_only_with_numeric_per_process_readings_outside_wsl(monkeypatch):
    prof = P.Profiler(device="cpu")
    prof.samples = [{"device_used": 1, "process_used": 5}, {"device_used": 1, "process_used": None}]
    monkeypatch.setattr(P, "is_wsl", lambda: False)
    assert prof.evidence() == "diagnostic"
    prof.samples[1]["process_used"] = 6
    assert prof.evidence() == "assignment"
    monkeypatch.setattr(P, "is_wsl", lambda: True)
    assert prof.evidence() == "diagnostic"
