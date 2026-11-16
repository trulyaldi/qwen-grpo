"""L2L research lab: paired reference/candidate experiments with explicit acceptance rules (research/l2l/protocol.md).

A case runs the resident reference and the candidate backend named by the config's execution section on the frozen
fixtures (qwen_grpo/execution/fixtures.py), then decides a status from the raw data alone:

  correctness  PASS only if there is at least one comparison, every reference quantity has a candidate, every
               comparison is finite and within  max|got - ref| <= atol + rtol * max|ref|  (research/l2l/tolerances.yaml),
               every reference gradient is nonzero (otherwise the comparison has no power), the fresh reference equals
               the saved one bitwise, and every required measurement is present.
  performance  PASS only if enough timed trials exist after warmup, all finite, with the required memory measurements.
               PASS means the measurement is valid; what it shows (faster, slower, no detectable difference) is the
               separate `conclusion`.
  BLOCKED      the candidate cannot run yet (an unimplemented backend or setting); never PASS.

Every result is written, failures included, to <root>/<name>/result.json with a summary.md, and never overwritten. It
records the config, input (fixture files) and tolerance hashes, the provenance, all raw comparisons and trials.
"""
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from qwen_grpo import config
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.qwen3_forward import explicit_logprobs
from qwen_grpo.grpo import grpo_loss

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "l2l" / "experiments"
TOLERANCES = ROOT / "research" / "l2l" / "tolerances.yaml"
MIN_TRIALS = 5
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 0


class Blocked(Exception):
    """The candidate cannot be executed by this code base yet."""


class SetupError(Exception):
    """The experiment is ill-posed (missing fixture, identity mismatch, missing threshold): FAIL, never PASS."""


# --- comparisons ---------------------------------------------------------------------------------------------------

def compare(name, ref, got, tol):
    """One tensor comparison. tol: {atol, rtol, require_nonzero?}. Shape/dtype mismatches and non-finite values fail
    before any threshold is applied; rel is None when the reference is identically zero."""
    out = {"name": name, "shape": list(ref.shape), "dtype": str(ref.dtype), "atol": tol["atol"], "rtol": tol["rtol"]}
    if got is None:
        return {**out, "pass": False, "reason": "missing in candidate"}
    if got.shape != ref.shape or got.dtype != ref.dtype:
        return {**out, "pass": False, "reason": f"candidate {got.dtype} {list(got.shape)} != reference"}
    r, g = ref.detach().double().cpu(), got.detach().double().cpu()
    out.update(nonfinite_ref=int((~torch.isfinite(r)).sum()), nonfinite_got=int((~torch.isfinite(g)).sum()))
    if out["nonfinite_ref"] or out["nonfinite_got"]:
        return {**out, "pass": False, "reason": "non-finite values"}
    diff = (g - r).abs()
    ref_max, max_abs = r.abs().max().item(), diff.max().item()
    bound = tol["atol"] + tol["rtol"] * ref_max
    r_norm = r.norm().item()
    out.update(max_abs=max_abs, ref_max_abs=ref_max, bound=bound,
               rel=max_abs / ref_max if ref_max > 0 else None,
               rel_l2=(g - r).norm().item() / r_norm if r_norm > 0 else None,
               cosine=(torch.dot(g.flatten(), r.flatten()) / (g.norm() * r_norm)).item() if r.numel() > 1 and r_norm > 0
               and g.norm() > 0 else None,
               ref_norm=r_norm, got_norm=g.norm().item())
    if tol.get("require_nonzero") and ref_max == 0:
        return {**out, "pass": False, "reason": "reference is identically zero: the comparison has no power"}
    ok = max_abs <= bound
    return {**out, "pass": ok, "reason": None if ok else f"max_abs {max_abs:.3e} > bound {bound:.3e}"}


def compare_all(ref, got, tolerances):
    """Every reference tensor against the candidate's; candidate tensors absent from the reference fail too."""
    rows = []
    for name, r in ref.items():
        quantity = name.split(".")[0]
        if quantity not in tolerances:
            raise SetupError(f"no tolerance for quantity {quantity!r}: cannot judge {name}")
        rows.append({"quantity": quantity, **compare(name, r, got.get(name), tolerances[quantity])})
    rows += [{"quantity": n.split(".")[0], "name": n, "pass": False, "reason": "not in reference"} for n in got if n not in ref]
    return rows


def flatten(out):
    """fixtures.reference() output -> {name: tensor}: forward, loss, per-parameter gradients, trained LoRA tensors,
    AdamW state and the per-iteration GRPO logs."""
    t = {"logits": out["logits"], "logp": out["logp"], "loss": out["loss"]}
    t.update({f"grads.{n}": g for n, g in out["grads"].items()})
    t.update({f"final_lora.{n}": p for n, p in out["final_lora"].items()})
    for i, s in out["optimizer"]["state"].items():
        t.update({f"optimizer.{i}.{k}": v for k, v in s.items()})
    for u in out["updates"]:
        t.update({f"updates.{u['iteration']}.{k}": torch.tensor(float(u[k]), dtype=torch.float64)
                  for k in ("loss", "grad_norm", "update_norm", "ratio_min", "ratio_max", "clip_frac", "approx_kl")})
    return t


def perturb(tensors, spec):
    """Fault injection 'name=rel': adds rel * max(|x0|, tiny) to the first element of the first tensor whose name
    starts with `name`; refuses if the change rounds away (Ticket 00: small shifts can be invisible)."""
    name, rel = spec.split("=")
    key = next((k for k in tensors if k == name or k.startswith(name + ".")), None)
    if key is None:
        raise SetupError(f"perturbation target {name!r} is not an output of this case")
    t = tensors[key].clone()
    flat = t.view(-1)
    before = flat[0].clone()
    flat[0] = before + float(rel) * max(before.abs().item(), torch.finfo(t.dtype).tiny)
    if torch.equal(flat[0], before):
        raise SetupError(f"perturbation {spec} is not representable in {t.dtype} at {key}[0] = {before.item()}")
    tensors[key] = t
    return {"spec": spec, "tensor": key, "element": 0, "before": before.item(), "after": flat[0].item()}


# --- measurements --------------------------------------------------------------------------------------------------

def memory(device):
    """Snapshot (not a peak sampler): CPU RSS / high-water mark; on CUDA the allocator statistics, cudaMemGetInfo and
    nvidia-smi. The two device-level readings count every process on the GPU and the CUDA context: upper bounds."""
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    kib = lambda k: int(status[k].split()[0]) / 1024 if k in status else None
    out = {"kind": "snapshot", "cpu": {"rss_mib": kib("VmRSS"), "peak_rss_mib": kib("VmHWM")}, "gpu": None}
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        mib = 2 ** 20
        smi = _nvidia_smi()
        out["gpu"] = {"allocated_mib": torch.cuda.memory_allocated() / mib, "reserved_mib": torch.cuda.memory_reserved() / mib,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
                      "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
                      "device_used_mib_mem_get_info": (total - free) / mib, **smi}
        out["pinned_host"] = {k: v for k, v in torch.cuda.host_memory_stats().items() if "allocated_bytes" in k}
    return out


def _nvidia_smi():
    def query(*args):
        try:
            return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                  check=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    def number(text):  # WSL2 reports per-process memory as "[N/A]"
        try:
            return float(text)
        except (TypeError, ValueError):
            return None

    apps, used = query("--query-compute-apps=pid,used_memory"), query("--query-gpu=memory.used")
    mine = [line.split(",")[1] for line in (apps or "").splitlines() if line.split(",")[0].strip() == str(os.getpid())]
    return {"nvidia_smi_compute_apps_raw": apps, "nvidia_smi_memory_used_raw": used,
            "process_used_mib_nvidia_smi": number(mine[0]) if mine else None,
            "device_used_mib_nvidia_smi": number(used.splitlines()[0]) if used else None}


def gpu_memory_complete(mem):
    """GPU memory evidence: allocator statistics plus at least one device-level reading (process-level nvidia-smi is
    empty under WSL2, so the device-wide readings stand in as upper bounds)."""
    g = (mem or {}).get("gpu")
    if not g:
        return False
    allocator = all(g.get(k) is not None for k in ("allocated_mib", "reserved_mib", "peak_allocated_mib", "peak_reserved_mib"))
    device = any(g.get(k) is not None for k in ("process_used_mib_nvidia_smi", "device_used_mib_nvidia_smi",
                                                "device_used_mib_mem_get_info"))
    return allocator and device


def summarize(seconds):
    """Median, IQR, range and a percentile-bootstrap 95% CI of the median of repeated trials."""
    x = sorted(seconds)
    q = lambda p: float(torch.quantile(torch.tensor(x, dtype=torch.float64), p))
    rng = random.Random(BOOTSTRAP_SEED)
    meds = sorted(_median([rng.choice(x) for _ in x]) for _ in range(BOOTSTRAP))
    return {"n": len(x), "median_s": _median(x), "iqr_s": [q(0.25), q(0.75)], "min_s": x[0], "max_s": x[-1],
            "median_ci95_s": [meds[int(0.025 * BOOTSTRAP)], meds[int(0.975 * BOOTSTRAP) - 1]]}


def _median(x):
    s = sorted(x)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def ratio_ci(a, b):
    """Median(a) / median(b) with a percentile-bootstrap 95% CI (trials resampled independently)."""
    rng = random.Random(BOOTSTRAP_SEED + 1)
    rs = sorted(_median([rng.choice(a) for _ in a]) / _median([rng.choice(b) for _ in b]) for _ in range(BOOTSTRAP))
    return {"ratio": _median(a) / _median(b), "ci95": [rs[int(0.025 * BOOTSTRAP)], rs[int(0.975 * BOOTSTRAP) - 1]]}


# --- status ----------------------------------------------------------------------------------------------------------

def decide(result):
    """Status and reasons from the raw data in a result (also used by `report` to re-check a saved result)."""
    if result.get("blocked"):
        return "BLOCKED", [result["blocked"]]
    reasons = list(result.get("errors", []))
    for req in result.get("required_measurements", []):
        if req == "gpu_memory" and not gpu_memory_complete(result.get("memory")):
            reasons.append("required measurement missing: gpu_memory")
    if result["kind"] == "correctness":
        rows = result.get("comparisons", [])
        if not rows:
            reasons.append("no comparisons were produced")
        reasons += [f"{r['name']}: {r['reason']}" for r in rows if not r["pass"]]
        if result.get("reference_integrity") is None:
            reasons.append("reference integrity was not checked")
        elif result["reference_integrity"]:
            reasons.append(f"fresh reference differs from the saved one: {result['reference_integrity'][:5]}")
    else:
        for side, t in result.get("timing", {}).items():
            trials = t.get("trials_s", [])
            if len(trials) < MIN_TRIALS:
                reasons.append(f"{side}: {len(trials)} timed trials < {MIN_TRIALS}")
            if not all(math.isfinite(s) and s > 0 for s in trials):
                reasons.append(f"{side}: non-finite or non-positive trial times")
        if not result.get("timing"):
            reasons.append("no timings were produced")
    return ("FAIL" if reasons else "PASS"), reasons


# --- cases -----------------------------------------------------------------------------------------------------------

def check_fixture(shape, cfg):
    """The saved fixture exists and matches what the code and config now say it is."""
    if not F.path(shape).exists():
        raise SetupError(f"missing experiment fixture {F.path(shape)} (python -m qwen_grpo.execution.fixtures)")
    meta = torch.load(F.path(shape), weights_only=True)["meta"]
    expected = {**F.metadata(), "shape": F.SHAPES[shape]}
    if diff := sorted(k for k in expected.keys() | meta.keys() if expected.get(k) != meta.get(k)):
        raise SetupError(f"fixture {shape} was built with different settings or package versions {diff}; regenerate it "
                         "deliberately and record why (bitwise tolerances assume the same versions)")
    if cfg.model.dtype != meta["dtype"]:
        raise SetupError(f"model.dtype {cfg.model.dtype} does not match fixture dtype {meta['dtype']}")


def execute(backend, shape):
    """Outputs of `backend` on fixture `shape`, as fixtures.reference() returns them."""
    if backend != "resident":
        raise Blocked(f"execution.backend {backend!r} is not implemented yet (l2l_sync: Ticket 03; l2l_async: Ticket 08)")
    model, fixture = F.load(shape)
    with F.isolated():
        return F.reference(model, fixture), fixture


def fixture_equivalence(cfg, result, perturbation=None, **_):
    for shape in F.SHAPES:
        check_fixture(shape, cfg)
    tol = tolerances_for("fixture_equivalence", cfg)
    result.update(comparisons=[], reference_integrity=[])
    for shape in F.SHAPES:
        ref_out, fixture = execute("resident", shape)
        result["reference_integrity"] += [f"{shape}:{m}" for m in F.mismatches(ref_out, fixture["reference"])]
        got_out, _ = execute(cfg.execution.backend, shape)
        ref, got = flatten(ref_out), flatten(got_out)
        if perturbation and shape == next(iter(F.SHAPES)):
            result["perturbation"] = perturb(got, perturbation)
        result["comparisons"] += [{"fixture": shape, **row} for row in compare_all(ref, got, tol)]


def fixture_timing(cfg, result, trials=20, warmup=3, device="cpu", **_):
    """Forward + GRPO loss + backward of the fixture model (no optimizer step), reference and candidate interleaved
    trial by trial so that drift affects both alike."""
    shape = "4b-like"
    check_fixture(shape, cfg)
    if cfg.execution.backend != "resident":
        raise Blocked(f"execution.backend {cfg.execution.backend!r} is not implemented yet")
    model, fx = F.load(shape)
    model.to(device)
    to = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x
    rollouts = {k: to(v) for k, v in fx["rollouts"].items()}
    adv, old_logp = to(fx["advantages"]), to(fx["old_logp"])
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    def step():
        model.zero_grad(set_to_none=True)
        grpo_loss(explicit_logprobs(model, rollouts), old_logp, adv, rollouts["completion_mask"]).backward()

    def timed():
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    sides = {"reference": step, "candidate": step}  # both resident until a candidate backend exists
    result["timing"] = {s: {"warmup": warmup, "trials_s": []} for s in sides}
    for _ in range(warmup):
        for s in sides:
            timed()
    for _ in range(trials):
        for s in sides:
            result["timing"][s]["trials_s"].append(timed())
    for s, t in result["timing"].items():
        t.update(summarize(t["trials_s"]))
    r = ratio_ci(result["timing"]["candidate"]["trials_s"], result["timing"]["reference"]["trials_s"])
    lo, hi = r["ci95"]
    result["conclusion"] = {**r, "statement": "candidate faster" if hi < 1 else "candidate slower" if lo > 1
                            else "no detectable difference"}
    result["memory"] = memory(device)
    if cuda:
        result["required_measurements"] = ["gpu_memory"]


@dataclass(frozen=True)
class Case:
    kind: str
    run: object
    uses: tuple  # config fields the case reads; every other section is recorded but unused


CASES = {
    "fixture_equivalence": Case("correctness", fixture_equivalence, ("model.dtype", "execution")),
    "fixture_timing": Case("performance", fixture_timing, ("model.dtype", "execution")),
}


def tolerances_for(case, cfg):
    table = yaml.safe_load(TOLERANCES.read_text())
    try:
        entry = table[case][cfg.execution.backend][cfg.model.dtype]
    except (KeyError, TypeError):
        raise SetupError(f"no tolerances for {case} / {cfg.execution.backend} / {cfg.model.dtype} in {TOLERANCES.name}")
    return entry["quantities"]


# --- running and reporting -------------------------------------------------------------------------------------------

def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def provenance():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    tracked = git("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": git("rev-parse", "HEAD"), "git_tracked_dirty": None if tracked is None else bool(tracked),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")},
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads()}


def run(case, config_path, name=None, root=RESULTS, perturbation=None, **options):
    """Run `case`, write <root>/<name>/{result.json, summary.md} whatever happens, and return the result."""
    if case not in CASES:
        raise SystemExit(f"unknown case {case!r}: one of {list(CASES)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = name or f"{case}_{stamp}"
    if not re.fullmatch(config.RUN_NAME, name):
        raise SystemExit(f"bad experiment name {name!r}")
    out = Path(root) / name
    if out.exists():
        raise SystemExit(f"{out} exists: experiment results are never overwritten")
    cfg = config.load(config_path)
    spec = CASES[case]
    result = {"name": name, "case": case, "kind": spec.kind, "created_utc": stamp,
              "fault_injection": perturbation is not None, "options": {k: str(v) for k, v in options.items()},
              "identity": {"config_path": str(config_path), "config_sha256": config.sha256(cfg),
                           "config_fields_used": list(spec.uses), "execution": config.as_dict(cfg)["execution"],
                           "fixtures_sha256": {s: sha256_file(F.path(s)) if F.path(s).exists() else None for s in F.SHAPES},
                           "tolerances_sha256": sha256_file(TOLERANCES)},
              "provenance": provenance(), "errors": [], "required_measurements": []}
    result["identity"]["input_sha256"] = hashlib.sha256(json.dumps(result["identity"]["fixtures_sha256"],
                                                                   sort_keys=True).encode()).hexdigest()
    try:
        if missing := config.unimplemented(cfg):
            raise Blocked(f"unimplemented settings {missing}")
        spec.run(cfg, result, perturbation=perturbation, **options)
    except Blocked as e:
        result["blocked"] = str(e)
    except SetupError as e:
        result["errors"].append(f"setup: {e}")
    except Exception as e:  # recorded as a FAIL with everything measured so far, then re-raised
        result["errors"].append(f"{type(e).__name__}: {e}")
        result["traceback"] = traceback.format_exc()
        _write(out, result)
        raise
    _write(out, result)
    return result


def _write(out, result):
    result["status"], result["reasons"] = decide(result)
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(markdown(result))


def report(name, root=RESULTS):
    """Re-derive the status from the saved raw data, refuse a result whose recorded status disagrees, rewrite
    summary.md."""
    path = Path(root) / name / "result.json"
    if not path.exists():
        raise SystemExit(f"no experiment {name!r} under {root}")
    result = json.loads(path.read_text())
    for key in ("config_sha256", "input_sha256", "tolerances_sha256"):
        if not result.get("identity", {}).get(key):
            raise SystemExit(f"{path}: identity.{key} is missing")
    status, reasons = decide(result)
    if (status, reasons) != (result["status"], result["reasons"]):
        raise SystemExit(f"{path}: recorded status {result['status']} disagrees with its raw data ({status}: {reasons})")
    text = markdown(result)
    (path.parent / "summary.md").write_text(text)
    return text


def markdown(r):
    lines = []
    if r.get("fault_injection"):
        lines.append(f"**FAULT INJECTION RUN** ({r.get('perturbation', {}).get('spec')}): a FAIL is the expected outcome.\n")
    ident = r["identity"]
    lines += [f"# {r['name']}: {r['case']} ({r['kind']})", "",
              f"**Status: {r['status']}**", ""]
    lines += [f"- {x}" for x in r["reasons"][:20]] + ([f"- ... {len(r['reasons']) - 20} more"] if len(r["reasons"]) > 20 else [])
    p = r["provenance"]
    lines += ["", f"- commit `{p['git_commit']}` (tracked tree dirty: {p['git_tracked_dirty']}), torch {p['packages']['torch']}, "
              f"transformers {p['packages']['transformers']}, peft {p['packages']['peft']}, GPU {p['gpu']}",
              f"- config `{ident['config_path']}` sha256 `{ident['config_sha256'][:16]}`; fields used: {ident['config_fields_used']}; "
              f"backend `{ident['execution']['backend']}`",
              f"- input sha256 `{ident['input_sha256'][:16]}`, tolerances sha256 `{ident['tolerances_sha256'][:16]}`", ""]
    if r.get("comparisons"):
        lines += ["| fixture | quantity | tensors | pass | worst max_abs | worst rel | bound |", "|---|---|---|---|---|---|---|"]
        groups = {}
        for c in r["comparisons"]:
            groups.setdefault((c.get("fixture"), c["quantity"]), []).append(c)
        for (fx, q), rows in groups.items():
            worst = max(rows, key=lambda c: c.get("max_abs", math.inf))
            lines.append(f"| {fx} | {q} | {len(rows)} | {sum(c['pass'] for c in rows)}/{len(rows)} | "
                         f"{worst.get('max_abs', 'n/a')} | {worst.get('rel')} | {worst.get('bound')} |")
        lines.append("")
    if r.get("timing"):
        lines += ["| side | n | median s | IQR s | median 95% CI s |", "|---|---|---|---|---|"]
        for side, t in r["timing"].items():
            lines.append(f"| {side} | {t.get('n')} | {t.get('median_s', 0):.3e} | {t.get('iqr_s')} | {t.get('median_ci95_s')} |")
        if c := r.get("conclusion"):
            lines.append(f"\ncandidate / reference median ratio {c['ratio']:.3f}, 95% CI {c['ci95']}: {c['statement']}")
    if r.get("memory"):
        lines += ["", "memory snapshot: `" + json.dumps(r["memory"], default=str) + "`"]
    return "\n".join(lines) + "\n"
