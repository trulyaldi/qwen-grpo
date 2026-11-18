"""Stage A check: CUDA streams, events and pinned host-to-device copies work as the async L2L executor uses them.

    python scripts/assignment_cuda_check.py --out result.json

A pinned host tensor is copied to the GPU on a side stream. The compute stream waits on an event recorded after the
copy, then reads the data, and the result must equal the source bit for bit. A second copy into the same buffer,
ordered after the first reader by an event, must not corrupt the first result. Timed events must report a positive
elapsed time. The process holds its CUDA context for ~3 s so that the telemetry around it can show whether nvidia-smi
lists this process with a numeric memory figure. Writes {"ok": bool, "detail": str, ...} to --out. Never raises.
"""
import argparse
import json
import os
import platform
import time
import traceback


def check():
    import torch
    out = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "pid": os.getpid(), "platform": platform.platform()}
    if not torch.cuda.is_available():
        return {**out, "ok": False, "detail": "torch.cuda.is_available() is False"}
    p = torch.cuda.get_device_properties(0)
    out.update(gpu=p.name, capability=f"{p.major}.{p.minor}", total_bytes=p.total_memory,
               bf16_native=torch.cuda.is_bf16_supported(including_emulation=False))
    n = 64 * 2**20
    src = torch.arange(n // 4, dtype=torch.int32).pin_memory()
    src2 = (src * 3).pin_memory()
    out["pinned"] = bool(src.is_pinned())
    buf = torch.empty(n // 4, dtype=torch.int32, device="cuda")
    side, main = torch.cuda.Stream(), torch.cuda.current_stream()
    copied, freed = torch.cuda.Event(), torch.cuda.Event()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(side):
        t0.record()
        buf.copy_(src, non_blocking=True)
        t1.record()
        copied.record()
    main.wait_event(copied)
    first = buf.sum(dtype=torch.int64)             # reader of the first copy, on the main stream
    first_copy = buf.clone()
    freed.record(main)
    side.wait_event(freed)                          # the second copy may only start after the reader is done
    with torch.cuda.stream(side):
        buf.copy_(src2, non_blocking=True)
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1)
    exact = bool(torch.equal(first_copy.cpu(), src)) and first.item() == int(src.sum(dtype=torch.int64))
    second = bool(torch.equal(buf.cpu(), src2))
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    bf16_ok = bool(torch.isfinite((x @ x).float()).all())
    time.sleep(3.0)  # keep the context alive for the telemetry
    ok = exact and second and ms > 0 and out["pinned"] and bf16_ok
    out.update(copy_ms=ms, h2d_gib_per_s=(n / 2**30) / (ms / 1000) if ms > 0 else None, first_copy_exact=exact,
               second_copy_exact=second, bf16_matmul_finite=bf16_ok, ok=ok,
               detail=(f"{p.name} sm_{p.major}{p.minor}: side-stream pinned copy exact, event ordering held, "
                       f"{(n / 2**30) / (ms / 1000):.1f} GiB/s for 64 MiB, bf16 matmul finite" if ok else
                       f"failed: exact {exact}, second {second}, elapsed {ms} ms, pinned {out['pinned']}, bf16 {bf16_ok}"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        res = check()
    except Exception as e:  # noqa: BLE001 - recorded, never raised
        res = {"ok": False, "detail": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res.get(k) for k in ("ok", "detail")}))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
