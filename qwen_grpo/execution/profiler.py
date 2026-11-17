"""Resource accounting for L2L runs (Ticket 11): phases, continuous sampling, a memory guard, evidence grading.

Profiler(budget_bytes, interval) starts a sampler thread that, every `interval` seconds, records
  nvidia-smi raw output: `--query-gpu=memory.used,memory.total` (device) and `--query-compute-apps=pid,used_memory`
      (per process), retained verbatim with a timestamp;
  cudaMemGetInfo total - free (device-wide), PyTorch allocated / reserved (this process's allocator), and CPU
      VmRSS / RssAnon of this process (/proc/self/status);
  the current phase name.
phase(name) is a context manager around one workload phase: it synchronizes, resets the allocator peaks, and on exit
records wall time, the allocator's peak allocated / reserved, and the maxima of the samples taken inside the window
(device used, this process's nvidia-smi memory, CPU RSS).
The guard: when a sample's process-level GPU memory (nvidia-smi per process if numeric, otherwise the device-wide
cudaMemGetInfo figure, which is >= it) exceeds the budget, the violation is recorded and the main thread is
interrupted (_thread.interrupt_main: a KeyboardInterrupt), so the run stops. The PyTorch allocator's limit is never
treated as the budget: the CUDA context, cuBLAS workspaces and non-PyTorch allocations are outside it.
evidence(): "assignment" only if every sample has a numeric per-process nvidia-smi reading and the host is not WSL2
(where nvidia-smi reports per-process memory as [N/A] and device figures include the Windows host); otherwise
"diagnostic" (allocator + cudaMemGetInfo) or "incomplete". A run is never reported as passing a budget on less than
"assignment" evidence.
"""
import _thread
import os
import platform
import subprocess
import threading
import time
from contextlib import contextmanager

import torch


def is_wsl():
    return "microsoft" in platform.release().lower()


def _smi(args):
    try:
        out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() if out.returncode == 0 else f"ERROR {out.returncode}: {out.stderr.strip()}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"ERROR {type(e).__name__}"


def _proc_status():
    out = {}
    try:
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "RssAnon", "VmHWM"):
                out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def _mib(x):
    try:
        return float(x) * 2**20
    except ValueError:
        return None


class Profiler:
    def __init__(self, budget_bytes=None, interval=0.25, device="cuda", smi=True):
        self.budget, self.interval, self.smi = budget_bytes, interval, smi
        self.cuda = torch.device(device).type == "cuda" and torch.cuda.is_available()
        self.device = torch.device(device)
        self.samples, self.phases, self.violations = [], [], []
        self.pid, self._phase = os.getpid(), None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        return False

    def sample(self):
        s = {"t": time.time(), "phase": self._phase, **_proc_status()}
        if self.cuda:
            free, total = torch.cuda.mem_get_info(self.device)
            s.update(device_used=total - free, device_total=total, allocated=torch.cuda.memory_allocated(self.device),
                     reserved=torch.cuda.memory_reserved(self.device))
        if self.smi and self.cuda:
            gpu, apps = _smi(["--query-gpu=memory.used,memory.total"]), _smi(["--query-compute-apps=pid,used_memory"])
            s["smi_gpu_raw"], s["smi_apps_raw"] = gpu, apps
            mine = [line.split(",")[1].strip() for line in apps.splitlines() if line.split(",")[0].strip() == str(self.pid)]
            s["process_used"] = _mib(mine[0]) if mine else None  # None: absent or [N/A] (WSL2)
        return s

    def _run(self):
        while not self._stop.is_set():
            s = self.sample()
            self.samples.append(s)
            used = s.get("process_used") if s.get("process_used") is not None else s.get("device_used")
            if self.budget is not None and used is not None and used > self.budget:
                self.violations.append({"t": s["t"], "phase": s["phase"], "bytes": used, "budget": self.budget,
                                        "source": "nvidia-smi process" if s.get("process_used") is not None
                                        else "cudaMemGetInfo device"})
                _thread.interrupt_main()
                return
            self._stop.wait(self.interval)

    @contextmanager
    def phase(self, name):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start, self._phase = time.time(), name
        rss0 = _proc_status()
        try:
            yield
        finally:
            if self.cuda:
                torch.cuda.synchronize(self.device)
            end = time.time()
            self.samples.append(self.sample())  # at least one sample inside every window
            self._phase = None
            win = [s for s in self.samples if start <= s["t"] <= end and s["phase"] == name]
            rec = {"phase": name, "seconds": end - start, "samples": len(win),
                   "cpu_rss_start": rss0.get("VmRSS"), "cpu_rss_peak": max((s.get("VmRSS", 0) for s in win), default=None),
                   "cpu_anon_peak": max((s.get("RssAnon", 0) for s in win), default=None)}
            if self.cuda:
                rec.update(peak_allocated=torch.cuda.max_memory_allocated(self.device),
                           peak_reserved=torch.cuda.max_memory_reserved(self.device),
                           device_used_peak=max((s["device_used"] for s in win), default=None),
                           process_used_peak=max((s["process_used"] for s in win if s.get("process_used") is not None),
                                                 default=None),
                           process_used_samples=sum(s.get("process_used") is not None for s in win))
            self.phases.append(rec)

    def evidence(self):
        gpu = [s for s in self.samples if "device_used" in s]
        if not gpu:
            return "none"
        if all(s.get("process_used") is not None for s in gpu) and not is_wsl():
            return "assignment"
        return "diagnostic"

    def report(self, raw=True):
        return {"budget_bytes": self.budget, "interval_s": self.interval, "wsl2": is_wsl(), "evidence": self.evidence(),
                "violations": self.violations, "phases": self.phases,
                "peak": {k: max((s.get(k) or 0 for s in self.samples), default=None)
                         for k in ("device_used", "process_used", "reserved", "allocated", "VmRSS", "RssAnon")},
                "samples": self.samples if raw else len(self.samples)}


def h2d_bandwidth(nbytes, device="cuda", repeats=10):
    """Effective pinned host-to-device bandwidth in bytes/s (median of `repeats` copies of `nbytes`)."""
    src = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
    times = []
    for _ in range(repeats + 2):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        dst.copy_(src, non_blocking=True)
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / 1000)
    times = sorted(times[2:])
    return nbytes / times[len(times) // 2]
