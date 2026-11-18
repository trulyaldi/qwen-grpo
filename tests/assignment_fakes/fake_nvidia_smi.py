#!/usr/bin/env python3
"""A stand-in for nvidia-smi in tests (never used outside them). Supports `--version` and the two looping queries of
qwen_grpo.telemetry. The compute-apps query lists every process whose command line contains FAKE_SMI_MARKER (default
"fake_dispatcher"). Each listed process is charged the MiB of its entry in FAKE_SMI_MIB (JSON {entry id or "*": MiB}),
"[N/A]" if FAKE_SMI_NA=1. The GPU query reports a fixed device. FAKE_SMI_FAIL=1 exits with an error."""
import json
import os
import sys
import time
from datetime import datetime

args = sys.argv[1:]
if "--version" in args:
    print("NVIDIA-SMI version  : 999.99 (fake for tests)")
    sys.exit(0)
if os.environ.get("FAKE_SMI_FAIL") == "1":
    print("NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.", file=sys.stderr)
    sys.exit(9)
interval = int(args[args.index("-lms") + 1]) / 1000 if "-lms" in args else None  # None: print once
query = next((a for a in args if a.startswith("--query-")), "")
marker = os.environ.get("FAKE_SMI_MARKER", "fake_dispatcher")
mib = json.loads(os.environ.get("FAKE_SMI_MIB", '{"*": 5000}'))


def ts():
    return datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


if query.startswith("--query-compute-apps"):
    print("timestamp, pid, process_name, used_gpu_memory [MiB]", flush=True)
else:
    print("timestamp, index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%], power.draw [W]", flush=True)
while True:
    t = ts()
    if query.startswith("--query-compute-apps"):
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="replace").split("\0")
            except OSError:
                continue
            if not any(marker in c for c in cmd) or any("fake_nvidia_smi" in c for c in cmd):
                continue
            entry = cmd[cmd.index("--entry") + 1] if "--entry" in cmd else "*"
            val = "[N/A]" if os.environ.get("FAKE_SMI_NA") == "1" else f"{mib.get(entry, mib.get('*', 5000))} MiB"
            print(f"{t}, {pid}, python, {val}", flush=True)
    else:
        print(f"{t}, 0, NVIDIA Fake A6000, 6000 MiB, 49140 MiB, 37 %, 120.50 W", flush=True)
    if interval is None:
        break
    time.sleep(interval)
