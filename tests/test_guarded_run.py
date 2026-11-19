"""qwen_grpo.telemetry.run_guarded and scripts/guarded_run.py with a fake nvidia-smi: raw files are produced before the
workload starts and kept unmodified, attribution to the workload's process group, the exact 8e9-byte cap stops the
workload, a missing nvidia-smi refuses to start it, a sampler that dies or a hidden PID makes the record INCONCLUSIVE
(never PASS), a CPU-only workload is PASS at 0 attributed bytes only with complete sampling, sampling gaps and
uncovered workload windows are detected from the raw timestamps, and per-phase peaks come from the raw CSV."""
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from qwen_grpo import telemetry as T
from scripts import guarded_run

FAKES = Path(__file__).parent / "assignment_fakes"
SMI = str(FAKES / "fake_nvidia_smi.py")
CHILD = [sys.executable, "-c", "import time, sys; time.sleep(float(sys.argv[1]))"]


@pytest.fixture
def fake_smi(monkeypatch):
    monkeypatch.setenv("ASSIGNMENT_NVIDIA_SMI", SMI)
    monkeypatch.setenv("FAKE_SMI_MARKER", "guarded_child_marker")
    return monkeypatch


def child(seconds):
    return CHILD + [str(seconds), "guarded_child_marker"]


def test_raw_files_attribution_and_pass_under_the_cap(tmp_path, fake_smi):
    fake_smi.setenv("FAKE_SMI_MIB", '{"*": 3000}')
    rec = T.run_guarded(tmp_path / "item", child(2.0), cap_bytes=8_000_000_000)
    d = tmp_path / "item" / "telemetry"
    assert all(T.required_files_present(d).values())
    apps = (d / "nvidia_smi_apps.csv").read_text()
    assert apps.startswith("timestamp, pid, process_name, used_gpu_memory [MiB]")
    meta = json.loads((d / "telemetry.json").read_text())
    assert meta["started_utc"] < meta["workload_started_utc"] < meta["workload_ended_utc"] <= meta["stopped_utc"]
    t = rec["telemetry"]
    assert rec["exit"] == 0 and rec["memory_status"] == "PASS" and t["process_peak_mib"] == 3000
    assert t["cap_bytes"] == 8_000_000_000 and t["attributed_numeric_samples"] >= 3
    assert t["sampling"]["device_samples"] >= 5 and not t["sampling"]["problems"]
    assert rec["children_max_rss_bytes"] > 0


def test_the_exact_8e9_cap_stops_the_workload_7630_mib_is_over_7629_is_not(tmp_path, fake_smi):
    # 8e9 B = 7629.39 MiB: 7629 MiB is below the cap, 7630 MiB above (8 GiB = 8192 MiB would let 7630 through)
    fake_smi.setenv("FAKE_SMI_MIB", '{"*": 7629}')
    assert T.run_guarded(tmp_path / "ok", child(1.5), cap_bytes=8_000_000_000)["memory_status"] == "PASS"
    fake_smi.setenv("FAKE_SMI_MIB", '{"*": 7630}')
    t0 = time.time()
    rec = T.run_guarded(tmp_path / "over", child(60), cap_bytes=8_000_000_000, grace_s=2)
    assert time.time() - t0 < 30, "the workload was not stopped"
    assert rec["memory_status"] == "FAIL" and rec["violation"]["used_mib"] == 7630
    assert rec["signals_sent"][0][0] == "SIGINT" and rec["exit"] != 0


def test_without_nvidia_smi_the_workload_is_not_started(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSIGNMENT_NVIDIA_SMI", str(tmp_path / "missing"))
    marker = tmp_path / "ran"
    rec = T.run_guarded(tmp_path / "item", [sys.executable, "-c", f"open({str(marker)!r}, 'w')"], cap_bytes=8e9)
    assert rec["exit"] is None and rec["memory_status"] == "INCONCLUSIVE" and not marker.exists()
    rc = guarded_run.main(["--out", str(tmp_path / "cli"), "--", sys.executable, "-c", "pass"])
    assert rc == guarded_run.EXIT_INCONCLUSIVE


def test_a_dying_sampler_stops_a_capped_workload_and_is_inconclusive(tmp_path, fake_smi):
    fake_smi.setenv("FAKE_SMI_FAIL", "1")
    t0 = time.time()
    rec = T.run_guarded(tmp_path / "item", child(60), cap_bytes=8_000_000_000, grace_s=2)
    assert time.time() - t0 < 30
    assert rec["memory_status"] == "INCONCLUSIVE" and "monitor_failure" in rec["stop_reason"]


def test_hidden_pids_trigger_the_attribution_timeout(tmp_path, fake_smi):
    fake_smi.setenv("FAKE_SMI_MARKER", "nobody_has_this_marker")
    rec = T.run_guarded(tmp_path / "item", child(60), cap_bytes=8_000_000_000, attribution_timeout_s=1.5, grace_s=2)
    assert rec["memory_status"] == "INCONCLUSIVE" and "no numeric per-process" in rec["stop_reason"]["monitor_failure"]


def test_na_readings_are_inconclusive_never_zero(tmp_path, fake_smi):
    fake_smi.setenv("FAKE_SMI_NA", "1")
    rec = T.run_guarded(tmp_path / "item", child(1.5), cap_bytes=8_000_000_000)
    assert rec["memory_status"] == "INCONCLUSIVE" and rec["telemetry"]["process_peak_mib"] is None


def test_a_cpu_only_workload_passes_at_zero_bytes_only_when_declared(tmp_path, fake_smi):
    fake_smi.setenv("FAKE_SMI_MARKER", "nobody_has_this_marker")
    rec = T.run_guarded(tmp_path / "cpu", child(1.5), cap_bytes=8_000_000_000, expect_gpu=False)
    assert rec["memory_status"] == "PASS" and "0 bytes attributed" in rec["telemetry"]["memory_reason"]
    rec = T.run_guarded(tmp_path / "gpu", child(1.5), cap_bytes=8_000_000_000)  # a GPU workload with no rows
    assert rec["memory_status"] == "INCONCLUSIVE"


def test_the_cli_writes_its_record_and_exit_codes(tmp_path, fake_smi):
    fake_smi.setenv("FAKE_SMI_MIB", '{"*": 100}')
    assert guarded_run.main(["--out", str(tmp_path / "a"), "--"] + child(1.0)) == 0
    rec = json.loads((tmp_path / "a" / "guarded_run.json").read_text())
    assert rec["cap_bytes"] == 8_000_000_000 and rec["files_present"]["nvidia_smi_gpu.csv"]
    fake_smi.setenv("FAKE_SMI_MIB", '{"*": 9000}')
    assert guarded_run.main(["--out", str(tmp_path / "b"), "--grace-s", "1", "--"] + child(60)) == guarded_run.EXIT_CAP
    with pytest.raises(FileExistsError):  # an attempt directory is never reused
        guarded_run.main(["--out", str(tmp_path / "a"), "--"] + child(0.1))


def smi_ts(t):
    return t.astimezone().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def write(d, gpu_times, apps=(), meta=None):
    d.mkdir(parents=True)
    rows = "".join(f"{smi_ts(t)}, 0, GPU, {m} MiB, 12288 MiB, 1 %, 10.0 W\n" for t, m in gpu_times)
    (d / "nvidia_smi_gpu.csv").write_text("timestamp, index, name, memory.used [MiB], memory.total [MiB], "
                                          "utilization.gpu [%], power.draw [W]\n" + rows)
    (d / "nvidia_smi_apps.csv").write_text("timestamp, pid, process_name, used_gpu_memory [MiB]\n" +
                                           "".join(f"{smi_ts(t)}, 7, python, {m} MiB\n" for t, m in apps))
    (d / "pids.csv").write_text("pid,first,last,cmd\n7,a,b,\"x\"\n")
    (d / "telemetry.json").write_text(json.dumps({"interval_ms": 200, "utc_offset_s": datetime.now().astimezone()
                                                  .utcoffset().total_seconds(), **(meta or {})}))
    return d


def test_sampling_gaps_and_uncovered_windows_make_a_pass_inconclusive(tmp_path):
    t0 = datetime.now(timezone.utc)
    ok = [(t0 + timedelta(seconds=0.2 * i), 100) for i in range(20)]
    apps = [(t, 500) for t, _ in ok]
    s = T.summarize(write(tmp_path / "ok", ok, apps), cap_bytes=8e9)
    assert s["memory_status"] == "PASS" and s["sampling"]["max_gap_s"] == pytest.approx(0.2, abs=0.01)
    gap = ok[:5] + [(t + timedelta(seconds=30), m) for t, m in ok[5:]]
    s = T.summarize(write(tmp_path / "gap", gap, apps), cap_bytes=8e9)
    assert s["memory_status"] == "INCONCLUSIVE" and s["sampling"]["gaps_over_max"] == 1
    late = {"workload_started_utc": (t0 - timedelta(seconds=60)).isoformat(),
            "workload_ended_utc": (t0 + timedelta(seconds=2)).isoformat()}
    s = T.summarize(write(tmp_path / "late", ok, apps, late), cap_bytes=8e9)
    assert s["memory_status"] == "INCONCLUSIVE" and "not covered" in s["memory_reason"]


def test_phase_peaks_slice_the_raw_csv_and_keep_missing_phases_none(tmp_path):
    t0 = datetime.now(timezone.utc)
    times = [(t0 + timedelta(seconds=0.2 * i), 100) for i in range(30)]
    apps = [(t, 1000 + 10 * i) for i, (t, _) in enumerate(times)]
    d = write(tmp_path / "x", times, apps)
    ph = [{"name": "trained", "start_utc": t0.isoformat(), "end_utc": (t0 + timedelta(seconds=1.9)).isoformat()},
          {"name": "merged", "start_utc": (t0 + timedelta(seconds=2.1)).isoformat(),
           "end_utc": (t0 + timedelta(seconds=5.9)).isoformat()},
          {"name": "never", "start_utc": (t0 + timedelta(seconds=100)).isoformat(),
           "end_utc": (t0 + timedelta(seconds=101)).isoformat()}]
    out = {p["name"]: p for p in T.phase_peaks(d, ph)}
    assert out["trained"]["process_peak_mib"] == 1090 and out["merged"]["process_peak_mib"] == 1290
    assert out["never"]["process_peak_mib"] is None and out["never"]["attributed_samples"] == 0
