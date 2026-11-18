"""Official nvidia-smi telemetry for one experimental item, kept raw, with process-group attribution and a watchdog.

    with Telemetry(out_dir, cap_bytes=8e9, on_violation=stop) as tel:
        child = subprocess.Popen(argv, start_new_session=True)   # its PID is the process-group id
        tel.attach(child.pid)
        child.wait()
    summary = summarize(out_dir, cap_bytes=8e9)

Two nvidia-smi processes per item, exactly the assignment-grade queries, each started in its own session so that it can
be stopped as a group and never receives the terminal's Ctrl-C:
    nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_gpu_memory --format=csv -lms 200
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv -lms 200
Their stdout goes unmodified to nvidia_smi_apps.csv / nvidia_smi_gpu.csv, stderr to *.stderr. telemetry.json records
argv, nvidia-smi version, interval, start/stop UTC and exit status.

Attribution: a tracker thread lists the members of the attached process group from /proc every 200 ms (pids.csv:
pid, first and last seen). Per-process memory counts only rows whose PID was in that group. Under a container that
hides host PIDs no row matches, and the per-process reading is reported as unavailable, never guessed.

The watchdog reads the apps CSV as it grows. When the sum of the attributed processes' used_gpu_memory in one sample
exceeds the cap, it records the violation once and calls on_violation(). The caller stops the workload.

summarize() derives numbers from the raw files only. Unavailable readings stay None, never 0. Sampling every 200 ms
can miss transient spikes shorter than the interval; every summary says so.
"""
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

APPS_FIELDS = "timestamp,pid,process_name,used_gpu_memory"
GPU_FIELDS = "timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw"
INTERVAL_MS = 200
MIB = 2**20
UNAVAILABLE = ("[N/A]", "[Not Supported]", "N/A", "[Insufficient Permissions]", "[Unknown Error]", "")
SAMPLING_NOTE = (f"nvidia-smi was sampled every {INTERVAL_MS} ms; allocations that rise and fall between two samples "
                 "are not seen, so a measured peak is a lower bound on the true peak")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def smi_binary():
    """The nvidia-smi executable: $ASSIGNMENT_NVIDIA_SMI (tests use a fake), else the one on PATH, else None."""
    explicit = os.environ.get("ASSIGNMENT_NVIDIA_SMI")
    if explicit:
        return explicit if Path(explicit).exists() else None
    return shutil.which("nvidia-smi")


def commands(smi, interval_ms=INTERVAL_MS):
    return {"apps": [smi, f"--query-compute-apps={APPS_FIELDS}", "--format=csv", "-lms", str(interval_ms)],
            "gpu": [smi, f"--query-gpu={GPU_FIELDS}", "--format=csv", "-lms", str(interval_ms)]}


# --- parsing (pure) ----------------------------------------------------------------------------------------------

def number(field, unit=None):
    """'5120 MiB' -> 5120.0; '[N/A]', '[Not Supported]', '' -> None. A unit other than `unit` is an error (None)."""
    s = field.strip()
    if s in UNAVAILABLE:
        return None
    parts = s.split()
    if unit and len(parts) == 2 and parts[1] != unit:
        return None
    try:
        return float(parts[0])
    except (ValueError, IndexError):
        return None


def _rows(text):
    """Data lines of an nvidia-smi CSV: header lines (repeated per loop on some drivers) and blank lines dropped."""
    out, skipped = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("timestamp"):
            continue
        if line.startswith("No running") or line.startswith("NVIDIA-SMI has failed"):
            skipped.append(line)
            continue
        out.append(line)
    return out, skipped


def parse_apps(text):
    """Rows of the compute-apps CSV: timestamp, pid, process_name, used_mib (None if unavailable), raw line."""
    rows, problems = [], []
    lines, skipped = _rows(text)
    problems += skipped
    for line in lines:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            problems.append(line)
            continue
        try:
            pid = int(f[1])
        except ValueError:
            problems.append(line)
            continue
        # a process name may itself contain commas: the memory is always the last field
        rows.append({"timestamp": f[0], "pid": pid, "process_name": ", ".join(f[2:-1]), "used_mib": number(f[-1], "MiB"),
                     "raw": line})
    return rows, problems


def parse_gpu(text):
    rows, problems = [], []
    lines, skipped = _rows(text)
    problems += skipped
    for line in lines:
        f = [x.strip() for x in line.split(",")]
        if len(f) < 7:
            problems.append(line)
            continue
        rows.append({"timestamp": f[0], "index": f[1], "name": ", ".join(f[2:-4]), "memory_used_mib": number(f[-4], "MiB"),
                     "memory_total_mib": number(f[-3], "MiB"), "utilization_pct": number(f[-2], "%"),
                     "power_w": number(f[-1], "W"), "raw": line})
    return rows, problems


def attributed_samples(apps_rows, pids):
    """Per nvidia-smi sample (timestamp) of the attributed processes: (timestamp, summed MiB or None, rows). A sample
    whose attributed rows are all unavailable has None; one with some numeric rows sums those and is marked partial."""
    by_ts = {}
    for r in apps_rows:
        if r["pid"] in pids:
            by_ts.setdefault(r["timestamp"], []).append(r)
    out = []
    for ts, rs in by_ts.items():
        vals = [r["used_mib"] for r in rs if r["used_mib"] is not None]
        out.append({"timestamp": ts, "used_mib": sum(vals) if vals else None, "partial": 0 < len(vals) < len(rs),
                    "pids": sorted({r["pid"] for r in rs})})
    return out


def read_pids(path):
    pids = set()
    if Path(path).exists():
        for line in Path(path).read_text().splitlines()[1:]:
            try:
                pids.add(int(line.split(",")[0]))
            except (ValueError, IndexError):
                pass
    return pids


def summarize(out_dir, cap_bytes=None):
    """Derived numbers from one item's raw telemetry files. Every figure names its source file; unavailable stays None.
    memory_status: PASS (attributed numeric per-process samples exist and none exceeds the cap), FAIL (one does),
    INCONCLUSIVE (no attributed numeric per-process sample: nvidia-smi absent, [N/A], or PIDs not visible)."""
    d = Path(out_dir)
    meta = json.loads((d / "telemetry.json").read_text()) if (d / "telemetry.json").exists() else {}
    apps_text = (d / "nvidia_smi_apps.csv").read_text(errors="replace") if (d / "nvidia_smi_apps.csv").exists() else ""
    gpu_text = (d / "nvidia_smi_gpu.csv").read_text(errors="replace") if (d / "nvidia_smi_gpu.csv").exists() else ""
    apps, apps_problems = parse_apps(apps_text)
    gpu, gpu_problems = parse_gpu(gpu_text)
    pids = read_pids(d / "pids.csv")
    samples = attributed_samples(apps, pids)
    numeric = [s for s in samples if s["used_mib"] is not None]
    peak = max(numeric, key=lambda s: s["used_mib"]) if numeric else None
    dev = [r for r in gpu if r["memory_used_mib"] is not None]
    dpeak = max(dev, key=lambda r: r["memory_used_mib"]) if dev else None
    cap_mib = cap_bytes / MIB if cap_bytes else None
    if meta.get("status") == "unavailable":
        status, why = "INCONCLUSIVE", f"nvidia-smi unavailable: {meta.get('reason')}"
    elif not numeric:
        why = ("no nvidia-smi compute-app row belongs to this item's process group"
               + (f" ({len(apps)} rows of other PIDs: the container may hide host PIDs)" if apps else "")
               if not samples else "every attributed per-process reading is unavailable ([N/A])")
        status = "INCONCLUSIVE"
    elif cap_mib is None:
        status, why = "INCONCLUSIVE", "no cap given"
    elif peak["used_mib"] > cap_mib:
        status, why = "FAIL", f"per-process peak {peak['used_mib']:.0f} MiB > cap {cap_mib:.0f} MiB at {peak['timestamp']}"
    else:
        status, why = "PASS", f"per-process peak {peak['used_mib']:.0f} MiB <= cap {cap_mib:.0f} MiB over {len(numeric)} samples"
    if meta.get("violation") and status != "FAIL":  # the live watchdog saw it; the raw file is the same data
        status, why = "FAIL", f"watchdog: {meta['violation']}"
    return {
        "raw_files": {"apps": str(d / "nvidia_smi_apps.csv"), "gpu": str(d / "nvidia_smi_gpu.csv"),
                      "pids": str(d / "pids.csv"), "meta": str(d / "telemetry.json")},
        "interval_ms": meta.get("interval_ms"), "started_utc": meta.get("started_utc"), "stopped_utc": meta.get("stopped_utc"),
        "nvidia_smi_version": meta.get("version"), "exit_codes": meta.get("exit_codes"),
        "stderr_nonempty": meta.get("stderr_nonempty"),
        "apps_rows": len(apps), "apps_numeric_rows": sum(r["used_mib"] is not None for r in apps),
        "apps_unparsed_lines": len(apps_problems), "gpu_rows": len(gpu), "gpu_unparsed_lines": len(gpu_problems),
        "attributed_pids": sorted({p for s in samples for p in s["pids"]}), "tracked_pids": len(pids),
        "attributed_samples": len(samples), "attributed_numeric_samples": len(numeric),
        "partial_samples": sum(s["partial"] for s in samples),
        "process_peak_mib": None if peak is None else peak["used_mib"],
        "process_peak_timestamp": None if peak is None else peak["timestamp"],
        "device_peak_used_mib": None if dpeak is None else dpeak["memory_used_mib"],
        "device_peak_timestamp": None if dpeak is None else dpeak["timestamp"],
        "device_total_mib": None if dpeak is None else dpeak["memory_total_mib"],
        "gpu_names": sorted({r["name"] for r in gpu}),
        "power_peak_w": max((r["power_w"] for r in gpu if r["power_w"] is not None), default=None),
        "utilization_peak_pct": max((r["utilization_pct"] for r in gpu if r["utilization_pct"] is not None), default=None),
        "cap_bytes": cap_bytes, "cap_mib": cap_mib, "memory_status": status, "memory_reason": why,
        "violation": meta.get("violation"), "note": SAMPLING_NOTE,
        "scope": "process_peak_mib: sum of used_gpu_memory over this item's process group per sample (CUDA context "
                 "included); device_peak_used_mib: memory.used of the whole GPU (every process, may include others)",
    }


# --- live supervision --------------------------------------------------------------------------------------------

def group_members(pgid):
    """PIDs whose process group is `pgid` (Linux /proc; empty elsewhere)."""
    out = set()
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for e in entries:
        if not e.isdigit():
            continue
        try:
            stat = Path(f"/proc/{e}/stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2:].split()  # after "pid (comm) "
        if len(fields) > 2 and fields[2] == str(pgid):
            out.add(int(e))
    return out


def stop_group(proc, grace=5.0):
    """SIGTERM a process started with start_new_session, then SIGKILL after `grace` seconds; return its exit code."""
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=grace)
    return proc.returncode


class Telemetry:
    def __init__(self, out_dir, cap_bytes=None, on_violation=None, interval_ms=INTERVAL_MS, smi=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap, self.on_violation, self.interval_ms = cap_bytes, on_violation, interval_ms
        self.smi = smi if smi is not None else smi_binary()
        self.procs, self.files, self.pgid = {}, [], None
        self.seen, self.violation = {}, None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._apps_offset, self._apps_partial = 0, ""
        self.meta = {"interval_ms": interval_ms, "cap_bytes": cap_bytes, "note": SAMPLING_NOTE}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self.meta["started_utc"] = utc()
        if not self.smi:
            self.meta.update(status="unavailable", reason="nvidia-smi not found on PATH")
        else:
            try:
                v = subprocess.run([self.smi, "--version"], capture_output=True, text=True, timeout=30)
                self.meta["version"] = (v.stdout or v.stderr).strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                self.meta["version"] = f"ERROR {type(e).__name__}: {e}"
            argv = commands(self.smi, self.interval_ms)
            self.meta["commands"] = argv
            for name, cmd in argv.items():
                out = open(self.dir / f"nvidia_smi_{name}.csv", "wb")
                err = open(self.dir / f"nvidia_smi_{name}.stderr", "wb")
                self.files += [out, err]
                try:
                    self.procs[name] = subprocess.Popen(cmd, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                                        start_new_session=True)
                except OSError as e:
                    self.meta.setdefault("start_errors", {})[name] = f"{type(e).__name__}: {e}"
            self.meta["status"] = "running" if self.procs else "unavailable"
            if not self.procs:
                self.meta["reason"] = f"nvidia-smi could not start: {self.meta.get('start_errors')}"
        self._write_meta()
        (self.dir / "pids.csv").write_text("pid,first_seen_utc,last_seen_utc,cmdline\n")
        self._thread.start()

    def attach(self, pgid):
        """Attribute the process group `pgid` (a child started with start_new_session=True: pgid == its PID)."""
        self.pgid = pgid
        self.meta["pgid"] = pgid
        self._track()

    def _track(self):
        if self.pgid is None:
            return
        with self._lock:
            self._track_locked()

    def _track_locked(self):
        now = utc()
        for pid in group_members(self.pgid) | {self.pgid}:
            if pid not in self.seen:
                try:
                    cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                except OSError:
                    cmd = ""
                self.seen[pid] = [now, now, cmd[:300]]
            else:
                self.seen[pid][1] = now

    def _check(self):
        """Watchdog: parse the complete lines appended to the apps CSV since the last check."""
        if self.cap is None or self.violation is not None or "apps" not in self.procs:
            return
        p = self.dir / "nvidia_smi_apps.csv"
        try:
            with open(p, "rb") as f:
                f.seek(self._apps_offset)
                chunk = f.read().decode(errors="replace")
        except OSError:
            return
        self._apps_offset += len(chunk.encode())
        text = self._apps_partial + chunk
        complete, _, self._apps_partial = text.rpartition("\n")
        rows, _ = parse_apps(complete)
        with self._lock:
            pids = set(self.seen)
        for s in attributed_samples(rows, pids):
            if s["used_mib"] is not None and s["used_mib"] * MIB > self.cap:
                self.violation = {"timestamp": s["timestamp"], "used_mib": s["used_mib"], "pids": s["pids"],
                                  "cap_bytes": self.cap, "cap_mib": self.cap / MIB, "detected_utc": utc(),
                                  "source": "nvidia-smi --query-compute-apps (per process, CUDA context included)"}
                self.meta["violation"] = self.violation
                self._write_meta()
                if self.on_violation:
                    self.on_violation(self.violation)
                return

    def _loop(self):
        while not self._stop.is_set():
            self._track()
            self._check()
            self._stop.wait(self.interval_ms / 1000)

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        self._track()
        self._check()
        codes = {name: stop_group(p) for name, p in self.procs.items()}
        for f in self.files:
            f.close()
        self.meta.update(stopped_utc=utc(), exit_codes=codes, status=self.meta.get("status") if not self.procs else "stopped",
                         stderr_nonempty={n: (self.dir / f"nvidia_smi_{n}.stderr").stat().st_size > 0 for n in self.procs},
                         orphans=[n for n, p in self.procs.items() if p.poll() is None])
        with self._lock:
            seen = sorted(self.seen.items())
        with open(self.dir / "pids.csv", "a") as f:
            for pid, (first, last, cmd) in seen:
                f.write(f"{pid},{first},{last},{json.dumps(cmd)}\n")
        self.meta["pids"] = [pid for pid, _ in seen]
        self._write_meta()

    def _write_meta(self):
        tmp = self.dir / "telemetry.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=1, default=str) + "\n")
        tmp.replace(self.dir / "telemetry.json")
