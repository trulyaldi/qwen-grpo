"""Run one GPU workload under raw nvidia-smi telemetry and the exact GPU memory cap (qwen_grpo.telemetry.run_guarded).

    python scripts/guarded_run.py --out <new dir> [--cap-bytes 8000000000] [--no-gpu-expected] -- <command ...>

Monitoring starts BEFORE the workload process exists (so before any model initialisation) and ends after it exited.
<out>/ (must not exist; an attempt is never written into another's directory):
  telemetry/nvidia_smi_apps.csv   raw `nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_gpu_memory
                                  --format=csv -lms 200` output, unmodified
  telemetry/nvidia_smi_gpu.csv    raw `nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,
                                  utilization.gpu,power.draw --format=csv -lms 200` output, unmodified
  telemetry/nvidia_smi_*.stderr   the samplers' stderr
  telemetry/pids.csv              the workload's process group: pid, first/last seen (UTC), command line
  telemetry/telemetry.json        argv, nvidia-smi version, interval, start/stop UTC, workload start/end UTC, UTC
                                  offset of nvidia-smi's local timestamps, exit codes, violation, monitor failures
  stdout.log, stderr.log          the workload's output
  guarded_run.json                this record: argv, start/end, exit code, signals sent, child peak RSS, telemetry
                                  summary (sample counts, largest gap, coverage, per-process and device peaks, status)
Status (memory_status): PASS / FAIL (a per-process sample above the cap: the workload was stopped) / INCONCLUSIVE
(no attributable numeric per-process sample, a sampler died, a sampling gap > 5 s, or the workload's start or end not
covered). Exit code: the workload's if it exited 0 and memory is PASS; 10 if the cap was exceeded; 11 if the
telemetry is INCONCLUSIVE or nvidia-smi is missing (the workload is not started without it, fail closed); otherwise the
workload's non-zero exit code. --no-gpu-expected: a workload that should never create a CUDA context (the CPU merge)
is PASS at 0 attributed bytes when sampling was complete. --attribution-timeout-s (default 900): stop the workload if no
numeric per-process sample of its group appeared by then (the cap would be unverifiable).
nvidia-smi sampling every 200 ms cannot see allocations that rise and fall between two samples: a measured peak is a
lower bound on the true peak (the in-process memory.guard and the allocator cap are the second line).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_grpo import telemetry  # noqa: E402

CAP_BYTES = 8_000_000_000  # the assignment's "8 GB of GPU memory as shown by nvidia-smi (CUDA context included)"
EXIT_CAP, EXIT_INCONCLUSIVE = 10, 11


def rel(p):
    s = str(p)
    return s[len(str(ROOT)) + 1:] if s.startswith(str(ROOT) + "/") else s


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: guarded_run.py --out DIR [options] -- COMMAND ...")
    cut = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cap-bytes", type=int, default=CAP_BYTES)
    ap.add_argument("--no-gpu-expected", action="store_true")
    ap.add_argument("--attribution-timeout-s", type=float, default=900.0)
    ap.add_argument("--grace-s", type=float, default=60.0, help="SIGINT -> SIGTERM delay after a violation")
    a = ap.parse_args(argv[:cut])
    cmd = argv[cut + 1:]
    if not cmd:
        raise SystemExit("no command after --")
    rec = telemetry.run_guarded(a.out, cmd, cap_bytes=a.cap_bytes, cwd=ROOT, expect_gpu=not a.no_gpu_expected,
                                attribution_timeout_s=a.attribution_timeout_s, grace_s=a.grace_s, rel=rel)
    rec["files_present"] = telemetry.required_files_present(Path(a.out) / "telemetry")
    (Path(a.out) / "guarded_run.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    status = rec.get("memory_status")
    t = rec.get("telemetry") or {}
    smp = t.get("sampling") or {}
    print(f"[guarded_run] exit {rec.get('exit')}  memory {status}: {t.get('memory_reason') or rec.get('error')}\n"
          f"[guarded_run] per-process peak {t.get('process_peak_mib')} MiB, device peak {t.get('device_peak_used_mib')} "
          f"MiB, cap {a.cap_bytes} B; device samples {smp.get('device_samples')}, attributed samples "
          f"{t.get('attributed_numeric_samples')}, largest gap {smp.get('max_gap_s')} s -> {rel(a.out)}/guarded_run.json",
          flush=True)
    if status == "FAIL":
        return EXIT_CAP
    if status != "PASS":
        return EXIT_INCONCLUSIVE
    return rec["exit"] if rec.get("exit") is not None else EXIT_INCONCLUSIVE


if __name__ == "__main__":
    sys.exit(main())
