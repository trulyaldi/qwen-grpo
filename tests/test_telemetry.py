"""qwen_grpo.telemetry: parsing of raw nvidia-smi CSV, process-group attribution, the memory verdict (never PASS
without a numeric per-process reading), the watchdog, and clean shutdown of the nvidia-smi processes. CPU only: a fake
nvidia-smi (tests/assignment_fakes/fake_nvidia_smi.py) stands in for the real one."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from qwen_grpo import telemetry as T

FAKES = Path(__file__).parent / "assignment_fakes"
SMI = str(FAKES / "fake_nvidia_smi.py")

APPS = """timestamp, pid, process_name, used_gpu_memory [MiB]
2026/11/18 10:00:00.100, 111, python, 5000 MiB
2026/11/18 10:00:00.100, 112, /usr/bin/python3, 1200 MiB
2026/11/18 10:00:00.100, 999, someone-else, 7000 MiB
timestamp, pid, process_name, used_gpu_memory [MiB]
2026/11/18 10:00:00.300, 111, my, odd, name, 6000 MiB
2026/11/18 10:00:00.500, 111, python, [N/A]
garbage line
"""
GPU = """timestamp, index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%], power.draw [W]
2026/11/18 10:00:00.100, 0, NVIDIA RTX A6000, 13200 MiB, 49140 MiB, 99 %, 280.10 W
2026/11/18 10:00:00.300, 0, NVIDIA RTX A6000, 6100 MiB, 49140 MiB, 12 %, [N/A]
"""


def test_numbers_keep_unavailable_readings_as_none():
    assert T.number("5120 MiB", "MiB") == 5120.0
    for s in ("[N/A]", "[Not Supported]", "", "N/A"):
        assert T.number(s, "MiB") is None
    assert T.number("5120 GiB", "MiB") is None  # an unexpected unit is not silently read as MiB


def test_apps_parsing_skips_repeated_headers_and_keeps_commas_in_names():
    rows, problems = T.parse_apps(APPS)
    assert [r["pid"] for r in rows] == [111, 112, 999, 111, 111]
    assert rows[3]["process_name"] == "my, odd, name" and rows[3]["used_mib"] == 6000.0
    assert rows[4]["used_mib"] is None
    assert problems == ["garbage line"]


def test_gpu_parsing():
    rows, problems = T.parse_gpu(GPU)
    assert rows[0]["memory_used_mib"] == 13200 and rows[0]["power_w"] == 280.1 and rows[1]["power_w"] is None
    assert rows[0]["name"] == "NVIDIA RTX A6000" and not problems


def test_attribution_sums_the_group_per_sample_and_ignores_other_pids():
    rows, _ = T.parse_apps(APPS)
    s = {x["timestamp"]: x for x in T.attributed_samples(rows, {111, 112})}
    assert s["2026/11/18 10:00:00.100"]["used_mib"] == 6200  # 5000 + 1200; pid 999 is not ours
    assert s["2026/11/18 10:00:00.500"]["used_mib"] is None


def write(d, apps=APPS, gpu=GPU, pids=(111, 112), meta=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "nvidia_smi_apps.csv").write_text(apps)
    (d / "nvidia_smi_gpu.csv").write_text(gpu)
    (d / "pids.csv").write_text("pid,first,last,cmd\n" + "".join(f"{p},a,b,\"x\"\n" for p in pids))
    (d / "telemetry.json").write_text(json.dumps(meta or {"interval_ms": 200}))
    return d


def test_summary_pass_and_fail_against_the_cap(tmp_path):
    s = T.summarize(write(tmp_path / "a"), cap_bytes=8e9)  # 8e9 B = 7629 MiB
    assert s["process_peak_mib"] == 6200 and s["memory_status"] == "PASS"
    assert s["device_peak_used_mib"] == 13200  # device-wide is reported, never used as the process figure
    s = T.summarize(write(tmp_path / "b"), cap_bytes=6000 * 2**20)
    assert s["memory_status"] == "FAIL"
    assert "200 ms" in s["note"]


def test_summary_is_inconclusive_without_attributable_numeric_readings(tmp_path):
    s = T.summarize(write(tmp_path / "a", pids=(4242,)), cap_bytes=8e9)  # e.g. a container hiding host PIDs
    assert s["memory_status"] == "INCONCLUSIVE" and s["process_peak_mib"] is None
    na = "timestamp, pid, process_name, used_gpu_memory [MiB]\n2026/11/18 10:00:00.100, 111, python, [N/A]\n"
    s = T.summarize(write(tmp_path / "b", apps=na), cap_bytes=8e9)  # e.g. WSL2
    assert s["memory_status"] == "INCONCLUSIVE" and s["process_peak_mib"] is None
    s = T.summarize(write(tmp_path / "c", apps="", gpu="", meta={"status": "unavailable", "reason": "no smi"}), cap_bytes=8e9)
    assert s["memory_status"] == "INCONCLUSIVE" and s["device_peak_used_mib"] is None and s["gpu_rows"] == 0


def child(marker="assignment_fakes_child", seconds=30):
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})", marker],
                            start_new_session=True)


def test_live_telemetry_records_raw_output_attributes_the_group_and_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_MARKER", "assignment_fakes_child")
    monkeypatch.setenv("FAKE_SMI_MIB", '{"*": 3000}')
    tel = T.Telemetry(tmp_path / "t", cap_bytes=8e9, smi=SMI)
    tel.start()
    c = child(seconds=1.5)
    tel.attach(c.pid)
    c.wait()
    smi_pids = [p.pid for p in tel.procs.values()]
    tel.stop()
    for pid in smi_pids:  # no nvidia-smi process survives
        assert not Path(f"/proc/{pid}").exists() or "zombie" in Path(f"/proc/{pid}/status").read_text().lower()
    meta = json.loads((tmp_path / "t" / "telemetry.json").read_text())
    assert meta["commands"]["apps"][1:] == ["--query-compute-apps=timestamp,pid,process_name,used_gpu_memory",
                                            "--format=csv", "-lms", "200"]
    assert meta["version"].startswith("NVIDIA-SMI") and meta["orphans"] == [] and c.pid in meta["pids"]
    s = T.summarize(tmp_path / "t", cap_bytes=8e9)
    assert s["memory_status"] == "PASS" and s["process_peak_mib"] == 3000 and s["attributed_numeric_samples"] >= 2
    assert (tmp_path / "t" / "nvidia_smi_gpu.csv").read_text().startswith("timestamp, index, name")


def test_the_watchdog_fires_on_the_first_sample_above_the_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SMI_MARKER", "assignment_fakes_child")
    monkeypatch.setenv("FAKE_SMI_MIB", '{"*": 9000}')
    fired = []
    c = None
    with T.Telemetry(tmp_path / "t", cap_bytes=8e9, smi=SMI, on_violation=fired.append) as tel:
        c = child()
        tel.attach(c.pid)
        t0 = time.time()
        while not fired and time.time() - t0 < 15:
            time.sleep(0.1)
        os.killpg(c.pid, 9)
        c.wait()
    assert fired and fired[0]["used_mib"] == 9000 and fired[0]["cap_bytes"] == 8e9
    s = T.summarize(tmp_path / "t", cap_bytes=8e9)
    assert s["memory_status"] == "FAIL" and s["violation"]


def test_missing_or_failing_nvidia_smi_is_recorded_not_invented(tmp_path, monkeypatch):
    tel = T.Telemetry(tmp_path / "none", cap_bytes=8e9, smi="")
    tel.start()
    tel.stop()
    s = T.summarize(tmp_path / "none", cap_bytes=8e9)
    assert s["memory_status"] == "INCONCLUSIVE" and "unavailable" in s["memory_reason"] and s["process_peak_mib"] is None
    monkeypatch.setenv("FAKE_SMI_FAIL", "1")
    tel = T.Telemetry(tmp_path / "fail", cap_bytes=8e9, smi=SMI)
    tel.start()
    time.sleep(0.5)
    tel.stop()
    meta = json.loads((tmp_path / "fail" / "telemetry.json").read_text())
    assert meta["exit_codes"] == {"apps": 9, "gpu": 9} and meta["stderr_nonempty"] == {"apps": True, "gpu": True}
    assert T.summarize(tmp_path / "fail", cap_bytes=8e9)["memory_status"] == "INCONCLUSIVE"
