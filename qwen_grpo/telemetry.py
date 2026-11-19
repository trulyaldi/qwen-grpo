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

Fail-closed supervision (Telemetry(on_monitor_failure=...)): the workload is stopped, and the item is INCONCLUSIVE,
when an nvidia-smi sampler exits while the workload runs, or when `attribution_timeout_s` seconds after attach() no
attributed numeric per-process sample exists (nvidia-smi hides this workload's PIDs or reports [N/A]: the cap could
not be verified). summarize() also checks the device CSV's timestamps: sample count, largest gap between consecutive
samples, and coverage of the workload's own start/end (recorded by run_guarded); a gap above `max_gap_s` or missing
coverage makes a PASS INCONCLUSIVE. nvidia-smi timestamps are local time: the host's UTC offset at start is recorded
(telemetry.json utc_offset_s) and used to place them on the UTC axis.

A workload that never creates a CUDA context (the CPU tensor-by-tensor merge) has no compute-app row. With
expect_gpu=False such an item is PASS when the samplers ran with complete coverage and no row of its process group
ever appeared (0 bytes attributed over N samples); any attributed row is judged against the cap as usual.

run_guarded(out_dir, argv, cap_bytes=...) is the one supervisor used by scripts/guarded_run.py (and, through it, by
the 150-update experiment driver scripts/final150.py): monitoring starts before the workload process exists (so
before any model is initialised), the process runs in its own session, violations and monitor failures stop it
(SIGINT, then SIGTERM, then SIGKILL), and the record keeps argv, start/end UTC, exit code, signals, the child's peak
RSS and the telemetry summary. Raw CSVs are never rewritten.
"""
import json
import os
import resource
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
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


MAX_GAP_S = 5.0  # a device-CSV gap above this (no sample for > 25 intervals) makes a PASS INCONCLUSIVE
COVERAGE_SLACK_S = 5.0  # the first/last device sample must lie within this of the workload's start/end


def smi_time(ts, utc_offset_s):
    """nvidia-smi's local 'YYYY/MM/DD HH:MM:SS.mmm' -> aware UTC datetime, or None if unparsable."""
    try:
        local = datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f")
    except ValueError:
        return None
    return (local - timedelta(seconds=utc_offset_s or 0)).replace(tzinfo=timezone.utc)


def sampling_quality(gpu_rows, meta, max_gap_s=MAX_GAP_S, slack_s=COVERAGE_SLACK_S):
    """Sample count, largest gap and coverage of the workload window, from the device CSV (one row per sample per GPU,
    present whether or not a process exists). problems: why the record cannot certify the whole window."""
    off = meta.get("utc_offset_s")
    times = sorted({t for t in (smi_time(r["timestamp"], off) for r in gpu_rows) if t is not None})
    out = {"device_samples": len(times), "first_sample_utc": times[0].isoformat() if times else None,
           "last_sample_utc": times[-1].isoformat() if times else None, "max_gap_s": None, "gaps_over_1s": 0,
           "gaps_over_max": 0, "max_gap_limit_s": max_gap_s, "workload_started_utc": meta.get("workload_started_utc"),
           "workload_ended_utc": meta.get("workload_ended_utc"), "problems": []}
    if not times:
        out["problems"].append("no parsable device sample (nvidia_smi_gpu.csv)")
        return out
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    out.update(max_gap_s=max(gaps) if gaps else 0.0, gaps_over_1s=sum(g > 1 for g in gaps),
               gaps_over_max=sum(g > max_gap_s for g in gaps))
    if out["gaps_over_max"]:
        out["problems"].append(f"{out['gaps_over_max']} sampling gaps > {max_gap_s} s (largest {out['max_gap_s']:.1f} s)")
    for key, edge, cmp in (("workload_started_utc", times[0], lambda w, e: (e - w).total_seconds() <= slack_s),
                           ("workload_ended_utc", times[-1], lambda w, e: (w - e).total_seconds() <= slack_s)):
        if meta.get(key):
            w = datetime.fromisoformat(meta[key])
            if not cmp(w, edge):
                out["problems"].append(f"{key} {meta[key]} not covered (nearest device sample {edge.isoformat()})")
    return out


def phase_peaks(out_dir, phases):
    """Per-phase maxima from the raw files of one item: phases [{name, start_utc, end_utc}] (recorded by the workload,
    e.g. scripts/final_eval.py) -> [{name, process_peak_mib, device_peak_mib, attributed_samples, device_samples}].
    A phase without an attributed numeric sample has process_peak_mib None (never 0)."""
    d = Path(out_dir)
    meta = json.loads((d / "telemetry.json").read_text()) if (d / "telemetry.json").exists() else {}
    off = meta.get("utc_offset_s")
    apps, _ = parse_apps((d / "nvidia_smi_apps.csv").read_text(errors="replace") if (d / "nvidia_smi_apps.csv").exists() else "")
    gpu, _ = parse_gpu((d / "nvidia_smi_gpu.csv").read_text(errors="replace") if (d / "nvidia_smi_gpu.csv").exists() else "")
    samples = attributed_samples(apps, read_pids(d / "pids.csv"))
    out = []
    for ph in phases:
        a, b = datetime.fromisoformat(ph["start_utc"]), datetime.fromisoformat(ph["end_utc"])
        inside = lambda ts: (t := smi_time(ts, off)) is not None and a <= t <= b
        proc = [s["used_mib"] for s in samples if s["used_mib"] is not None and inside(s["timestamp"])]
        dev = [r["memory_used_mib"] for r in gpu if r["memory_used_mib"] is not None and inside(r["timestamp"])]
        out.append({"name": ph["name"], "start_utc": ph["start_utc"], "end_utc": ph["end_utc"],
                    "process_peak_mib": max(proc) if proc else None, "device_peak_mib": max(dev) if dev else None,
                    "attributed_samples": len(proc), "device_samples": len(dev)})
    return out


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


def summarize(out_dir, cap_bytes=None, expect_gpu=True, max_gap_s=MAX_GAP_S):
    """Derived numbers from one item's raw telemetry files. Every figure names its source file; unavailable stays None.
    memory_status: PASS (attributed numeric per-process samples exist and none exceeds the cap), FAIL (one does),
    INCONCLUSIVE (no attributed numeric per-process sample: nvidia-smi absent, [N/A], or PIDs not visible; a monitor
    failure; a sampling gap or uncovered workload start/end). expect_gpu=False: a workload without any attributed row
    (no CUDA context) is PASS at 0 bytes when sampling was complete."""
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
    quality = sampling_quality(gpu, meta, max_gap_s=max_gap_s)
    if meta.get("status") == "unavailable":
        status, why = "INCONCLUSIVE", f"nvidia-smi unavailable: {meta.get('reason')}"
    elif not expect_gpu and not samples and not quality["problems"] and cap_mib is not None:
        status, why = "PASS", (f"no row of this item's process group in {len(apps)} compute-app rows over "
                               f"{quality['device_samples']} device samples: no CUDA context, 0 bytes attributed")
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
    if status == "PASS" and (meta.get("monitor_failures") or quality["problems"]):
        status, why = "INCONCLUSIVE", ("monitoring incomplete: " + "; ".join(
            [str(m) for m in meta.get("monitor_failures", [])] + quality["problems"]) + f" (would be: {why})")
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
        "sampling": quality, "monitor_failures": meta.get("monitor_failures", []), "expect_gpu": expect_gpu,
        "utc_offset_s": meta.get("utc_offset_s"),
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
    def __init__(self, out_dir, cap_bytes=None, on_violation=None, interval_ms=INTERVAL_MS, smi=None,
                 on_monitor_failure=None, attribution_timeout_s=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap, self.on_violation, self.interval_ms = cap_bytes, on_violation, interval_ms
        self.on_monitor_failure, self.attribution_timeout_s = on_monitor_failure, attribution_timeout_s
        self._attached_at, self._attributed, self._failed = None, False, set()
        self.smi = smi if smi is not None else smi_binary()
        self.procs, self.files, self.pgid = {}, [], None
        self.seen, self.violation = {}, None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._apps_offset, self._apps_partial = 0, ""
        self.meta = {"interval_ms": interval_ms, "cap_bytes": cap_bytes, "note": SAMPLING_NOTE,
                     "attribution_timeout_s": attribution_timeout_s}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    def start(self):
        self.meta["started_utc"] = utc()
        self.meta["utc_offset_s"] = time.localtime().tm_gmtoff  # nvidia-smi timestamps are local time
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
        self._attached_at = time.time()
        self._track()

    def mark(self, key, value=None):
        """Record a workload fact in telemetry.json (e.g. workload_started_utc), now."""
        self.meta[key] = value if value is not None else utc()
        self._write_meta()

    def _monitor_failure(self, what):
        if what in self._failed:
            return
        self._failed.add(what)
        self.meta.setdefault("monitor_failures", []).append({"utc": utc(), "what": what})
        self._write_meta()
        if self.on_monitor_failure:
            self.on_monitor_failure(what)

    def _supervise(self):
        """Fail closed: a sampler that died, or no per-process attribution after attribution_timeout_s."""
        for name, proc in self.procs.items():
            if proc.poll() is not None:
                self._monitor_failure(f"nvidia-smi {name} sampler exited with {proc.returncode} while the workload ran")
        if (self.attribution_timeout_s is not None and self._attached_at is not None and not self._attributed
                and time.time() - self._attached_at > self.attribution_timeout_s):
            self._monitor_failure(f"no numeric per-process nvidia-smi sample of the workload's process group within "
                                  f"{self.attribution_timeout_s} s of its start (PIDs hidden or [N/A]): the cap "
                                  "cannot be verified")

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
        if self.violation is not None or "apps" not in self.procs or (self.cap is None and self._attributed):
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
            if s["used_mib"] is not None:
                self._attributed = True
            if s["used_mib"] is not None and self.cap is not None and s["used_mib"] * MIB > self.cap:
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
            if self.pgid is not None:
                self._supervise()
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


# --- one supervised workload ---------------------------------------------------------------------------------------

def _killpg(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def run_guarded(out_dir, argv, cap_bytes=None, cwd=None, env=None, telemetry_on=True, expect_gpu=True,
                attribution_timeout_s=None, grace_s=60.0, should_stop=lambda: None, require_smi=True,
                on_started=None, rel=str):
    """Run `argv` in its own session under telemetry (started first, so it covers model initialisation). On a cap
    violation, a monitor failure or should_stop() returning a reason: SIGINT to the group, SIGTERM after grace_s,
    SIGKILL after 1.5 grace_s. Returns the record (also the caller's to save). out_dir must not exist yet: an attempt
    is never written into another's directory. require_smi: refuse to start the workload without nvidia-smi
    (fail closed); the record then says why and no workload ran."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    rec = {"argv": [rel(a) for a in argv], "started_utc": utc(), "stdout": rel(out_dir / "stdout.log"),
           "stderr": rel(out_dir / "stderr.log"), "raw_dir": rel(out_dir), "cap_bytes": cap_bytes,
           "expect_gpu": expect_gpu}
    if telemetry_on and require_smi and smi_binary() is None:
        rec.update(exit=None, error="nvidia-smi not found: refusing to run an unmonitored workload (fail closed)",
                   ended_utc=utc(), seconds=0.0, memory_status="INCONCLUSIVE")
        return rec
    run_env = {**os.environ, "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
               **(env or {})}
    stop_reason = {}
    tel = Telemetry(out_dir / "telemetry", cap_bytes=cap_bytes,
                    on_violation=lambda v: stop_reason.setdefault("violation", v),
                    # an uncapped item (cap None) only records a monitor failure; a capped one is stopped
                    on_monitor_failure=(lambda w: stop_reason.setdefault("monitor_failure", w))
                    if cap_bytes is not None else None,
                    attribution_timeout_s=attribution_timeout_s if expect_gpu else None) if telemetry_on else None
    t0 = time.time()
    with open(out_dir / "stdout.log", "wb") as so, open(out_dir / "stderr.log", "wb") as se:
        if tel:
            tel.start()
            time.sleep(min(1.0, 3 * INTERVAL_MS / 1000))  # a few device samples before the workload exists
        try:
            if tel:
                tel.mark("workload_started_utc")
            child = subprocess.Popen(argv, cwd=cwd, env=run_env, stdout=so, stderr=se, stdin=subprocess.DEVNULL,
                                     start_new_session=True)
        except OSError as e:
            if tel:
                tel.stop()
            rec.update(exit=None, error=f"{type(e).__name__}: {e}", ended_utc=utc(), seconds=0.0)
            return rec
        if tel:
            tel.attach(child.pid)
        if on_started:
            on_started(child)
        escalation, sent = None, []
        while child.poll() is None:
            reason = stop_reason or ({"requested": r} if (r := should_stop()) else {})
            if reason and escalation is None:
                escalation = time.time()
                sent.append(("SIGINT", utc()))
                _killpg(child.pid, signal.SIGINT)
            elif escalation is not None:
                waited = time.time() - escalation
                if waited > grace_s and len(sent) == 1:
                    sent.append(("SIGTERM", utc()))
                    _killpg(child.pid, signal.SIGTERM)
                elif waited > 1.5 * grace_s and len(sent) == 2:
                    sent.append(("SIGKILL", utc()))
                    _killpg(child.pid, signal.SIGKILL)
            time.sleep(0.2)
        _killpg(child.pid, signal.SIGTERM)  # a group member that outlived the leader
        if tel:
            tel.mark("workload_ended_utc")
            time.sleep(min(1.0, 3 * INTERVAL_MS / 1000))  # samples after the end: coverage of the shutdown
            tel.stop()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    rec.update(exit=child.returncode, ended_utc=utc(), seconds=round(time.time() - t0, 1), signals_sent=sent,
               stop_reason=stop_reason or None, violation=stop_reason.get("violation"),
               children_max_rss_bytes=usage.ru_maxrss * 1024,
               children_max_rss_note="ru_maxrss of the largest waited-for descendant of this supervisor (Linux, KiB)")
    if tel:
        rec["telemetry"] = summarize(out_dir / "telemetry", cap_bytes=cap_bytes, expect_gpu=expect_gpu)
        rec["memory_status"] = rec["telemetry"]["memory_status"] if cap_bytes is not None else "NOT_JUDGED"
    return rec


def required_files_present(tel_dir):
    """The raw evidence an item must have: {file: exists and non-empty} (pids.csv and telemetry.json always written;
    the CSVs only when nvidia-smi ran)."""
    d = Path(tel_dir)
    out = {f: (d / f).exists() and (d / f).stat().st_size > 0
           for f in ("nvidia_smi_apps.csv", "nvidia_smi_gpu.csv", "pids.csv", "telemetry.json")}
    out.update({f: (d / f).exists() for f in ("nvidia_smi_apps.stderr", "nvidia_smi_gpu.stderr")})  # empty is normal
    return out
