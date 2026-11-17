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
from qwen_grpo import store_lab as store_cases  # Ticket 02 cases; store_lab reads lab at call time
from qwen_grpo import forward_lab as forward_cases  # Ticket 03
from qwen_grpo import backward_lab as backward_cases  # Ticket 04
from qwen_grpo import head_lab as head_cases  # Ticket 05

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research" / "l2l" / "experiments"
TOLERANCES = ROOT / "research" / "l2l" / "tolerances.yaml"
HYPOTHESES = ROOT / "research" / "l2l" / "hypotheses.yaml"
SCHEMA = 2  # 2: execution_status and hypothesis_result next to status (Ticket 01 results have no schema field)
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
    out["evidence_grade"] = evidence_grade(out)
    return out


def evidence_grade(mem, release=None):
    """none (no GPU reading) | incomplete | diagnostic (allocator + cudaMemGetInfo: enough for local development) |
    assignment (also a numeric per-process nvidia-smi reading outside WSL2). Only `assignment` may support a claim
    that a GPU-memory budget (the 8 GB cap) was respected; nothing here combines readings into one number."""
    g = (mem or {}).get("gpu")
    if not g:
        return "none"
    if not gpu_memory_complete(mem):
        return "incomplete"
    wsl = "microsoft" in (release or platform.release()).lower()
    return "assignment" if g.get("process_used_mib_nvidia_smi") is not None and not wsl else "diagnostic"


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
    """GPU memory evidence: allocator statistics plus cudaMemGetInfo's device-wide reading (CUDA context and every
    process included: an upper bound for this process). nvidia-smi is recorded raw but does not count: under WSL2 its
    per-process memory is [N/A] and memory.used read below this process's own reserved memory + context
    (research/l2l/protocol.md)."""
    g = (mem or {}).get("gpu")
    if not g:
        return False
    allocator = all(g.get(k) is not None for k in ("allocated_mib", "reserved_mib", "peak_allocated_mib", "peak_reserved_mib"))
    return allocator and g.get("device_used_mib_mem_get_info") is not None


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

def execution_check(result):
    """Did the experiment run validly? BLOCKED, FAIL (errors, missing required measurements, no data, a drifted
    reference, too few trials) or PASS. Threshold violations are not execution failures: they are the outcome."""
    if result.get("blocked"):
        return "BLOCKED", [result["blocked"]]
    reasons = list(result.get("errors", []))
    for req in result.get("required_measurements", []):
        if req == "gpu_memory" and not gpu_memory_complete(result.get("memory")):
            reasons.append("required measurement missing: gpu_memory")
    if result["kind"] == "correctness":
        if not result.get("comparisons"):
            reasons.append("no comparisons were produced")
        if result.get("reference_integrity") is None:
            reasons.append("reference integrity was not checked")
        elif result["reference_integrity"]:
            reasons.append(f"fresh reference differs from the saved one: {result['reference_integrity'][:5]}")
    elif result["kind"] == "memory":
        if not result.get("measurements"):
            reasons.append("no measurements were produced")
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


def decide(result):
    """Overall acceptance `status` (the Ticket 01 field, unchanged in meaning): BLOCKED, or PASS only if the experiment
    executed validly and, for correctness, every comparison is within its threshold."""
    execution, reasons = execution_check(result)
    if execution == "BLOCKED":
        return execution, reasons
    failed = [f"{r['name']}: {r['reason']}" for r in result.get("comparisons", []) if not r["pass"]]
    if result["kind"] == "correctness":  # the Ticket 01 reason order: errors, measurements, data, comparisons, integrity
        integrity = [x for x in reasons if x.startswith(("reference integrity", "fresh reference"))]
        reasons = [x for x in reasons if x not in integrity] + failed + integrity
    return ("FAIL" if reasons else "PASS"), reasons


HYPOTHESIS_RESULTS = ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE", "NOT_EVALUATED")


def hypothesis_result(result):
    """(result, reason) for the hypothesis the run declares. Correctness: SUPPORTED iff every comparison meets its
    predefined threshold, NOT_SUPPORTED iff any exceeds it (no significance test). Performance: from the bootstrap CI of
    the candidate/reference median ratio against the declared prediction; a CI containing 1 is INCONCLUSIVE.
    NOT_EVALUATED: no hypothesis declared, a fault-injection run, or the experiment did not execute validly."""
    h = result.get("hypothesis")
    if not h:
        return "NOT_EVALUATED", "no hypothesis declared for this run"
    if result.get("fault_injection"):
        return "NOT_EVALUATED", "fault-injection run: it tests the test's power, not the hypothesis"
    execution, _ = execution_check(result)
    if execution != "PASS":
        return "NOT_EVALUATED", f"execution {execution}"
    if result["kind"] == "correctness":
        bad = [r["name"] for r in result["comparisons"] if not r["pass"]]
        return ("NOT_SUPPORTED", f"{len(bad)} comparisons exceed their thresholds") if bad else \
               ("SUPPORTED", f"all {len(result['comparisons'])} comparisons within predefined thresholds")
    prediction = h.get("prediction")
    if prediction == "streamed_constant_resident_increasing":
        rows = result["measurements"]["depths"]
        streamed, resident = [r["streamed_peak_allocated"] for r in rows], [r["resident_peak_allocated"] for r in rows]
        if len(set(streamed)) > 1:
            return "NOT_SUPPORTED", f"streamed peak allocated differs across depths: {streamed}"
        if len(rows) >= 2 and all(a < b for a, b in zip(resident, resident[1:])):
            return "SUPPORTED", f"streamed peak {streamed[0]} B at every depth; resident peak {resident}"
        return "INCONCLUSIVE", f"streamed constant but resident peaks not strictly increasing: {resident}"
    lo, hi = result["conclusion"]["ci95"]
    if prediction == "candidate_faster":
        return ("SUPPORTED", f"ratio CI {[lo, hi]} < 1") if hi < 1 else ("NOT_SUPPORTED", f"ratio CI {[lo, hi]} >= 1") \
            if lo >= 1 else ("INCONCLUSIVE", f"ratio CI {[lo, hi]} contains 1")
    return "NOT_EVALUATED", f"no rule for prediction {prediction!r}"


def hypothesis(hid, kind):
    """The declared hypothesis from research/l2l/hypotheses.yaml; its kind must match the case's."""
    table = yaml.safe_load(HYPOTHESES.read_text())
    if hid not in table:
        raise SystemExit(f"unknown hypothesis {hid!r}: one of {list(table)}")
    if table[hid]["kind"] != kind:
        raise SystemExit(f"{hid} is a {table[hid]['kind']} hypothesis; this case is {kind}")
    return {"id": hid, **table[hid]}


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
        raise Blocked(f"execution.backend {backend!r} cannot run this case yet (l2l_sync is forward-only, see the "
                      "forward_equivalence case; its backward is Ticket 04; l2l_async: Ticket 08)")
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


def needs_store(cfg):
    """Blocked reasons of the weight-store cases: they need l2l storage settings, and page-locking needs CUDA. They
    do not execute layers, so the unimplemented l2l execution itself does not block them."""
    x = cfg.execution
    if x.weight_storage is None:
        return ["execution.weight_storage is not set (the store cases need an l2l execution section)"]
    return ["page-locking needs a CUDA device; none is available"] * (x.weight_storage == "pinned" and not torch.cuda.is_available())


def needs_forward(cfg):
    """Blocked reasons of forward_equivalence: an l2l_sync config with store settings and a CUDA device (the case runs
    on the GPU; its CPU part only ties the result to the fixtures)."""
    reasons = needs_store(cfg)
    if cfg.execution.backend != "l2l_sync":
        reasons.append(f"forward_equivalence tests the l2l_sync backend, not {cfg.execution.backend!r}")
    if not torch.cuda.is_available():
        reasons.append("forward_equivalence runs on a CUDA device; none is available")
    return reasons


def needs_l2l(cfg, case):
    """Blocked reasons of a case that runs l2l_sync on its --device (default CPU; a CUDA device is checked in the case)."""
    reasons = needs_store(cfg)
    if cfg.execution.backend != "l2l_sync":
        reasons.append(f"{case} tests the l2l_sync backend, not {cfg.execution.backend!r}")
    return reasons


@dataclass(frozen=True)
class Case:
    kind: str
    run: object
    uses: tuple  # config fields the case reads; every other section is recorded but unused
    blocked_by: object = config.unimplemented  # cfg -> reasons the case cannot run with this code base


CASES = {
    "fixture_equivalence": Case("correctness", fixture_equivalence, ("model.dtype", "execution")),
    "fixture_timing": Case("performance", fixture_timing, ("model.dtype", "execution")),
    "store_equivalence": Case("correctness", lambda *a, **k: store_cases.store_equivalence(*a, **k),
                              ("model.dtype", "execution"), needs_store),
    "store_depth": Case("memory", lambda *a, **k: store_cases.store_depth(*a, **k), ("execution",), needs_store),
    "store_build": Case("memory", lambda *a, **k: store_cases.store_build(*a, **k),
                        ("model.name", "model.dtype", "lora", "execution"), needs_store),
    "forward_equivalence": Case("correctness", lambda *a, **k: forward_cases.forward_equivalence(*a, **k),
                                ("model.dtype", "execution"), needs_forward),
    "gradient_equivalence": Case("correctness", lambda *a, **k: backward_cases.gradient_equivalence(*a, **k),
                                 ("model.dtype", "execution"), lambda cfg: needs_l2l(cfg, "gradient_equivalence")),
    "logprob_equivalence": Case("correctness", lambda *a, **k: head_cases.logprob_equivalence(*a, **k),
                                ("model.dtype", "execution"), lambda cfg: needs_l2l(cfg, "logprob_equivalence")),
    "pinned_transfer": Case("performance", lambda *a, **k: store_cases.pinned_transfer(*a, **k), ("execution",),
                            needs_store),
}


def tolerances_for(case, cfg, dtype=None):
    """Thresholds of `case` for the config's backend and `dtype` (default: the config's model.dtype; a case that sweeps
    dtypes itself passes each one)."""
    table = yaml.safe_load(TOLERANCES.read_text())
    dtype = dtype or cfg.model.dtype
    try:
        entry = table[case][cfg.execution.backend][dtype]
    except (KeyError, TypeError):
        raise SetupError(f"no tolerances for {case} / {cfg.execution.backend} / {dtype} in {TOLERANCES.name}")
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


def run(case, config_path, name=None, root=RESULTS, perturbation=None, hypothesis_id=None, **options):
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
    declared = hypothesis(hypothesis_id, spec.kind) if hypothesis_id else None
    result = {"schema": SCHEMA, "name": name, "case": case, "kind": spec.kind, "created_utc": stamp, "hypothesis": declared,
              "fault_injection": perturbation is not None, "options": {k: str(v) for k, v in options.items()},
              "identity": {"config_path": str(config_path), "config_sha256": config.sha256(cfg),
                           "config_fields_used": list(spec.uses), "execution": config.as_dict(cfg)["execution"],
                           "fixtures_sha256": {s: sha256_file(F.path(s)) if F.path(s).exists() else None for s in F.SHAPES},
                           "tolerances_sha256": sha256_file(TOLERANCES)},
              "provenance": provenance(), "errors": [], "required_measurements": []}
    result["identity"]["input_sha256"] = hashlib.sha256(json.dumps(result["identity"]["fixtures_sha256"],
                                                                   sort_keys=True).encode()).hexdigest()
    try:
        if missing := spec.blocked_by(cfg):
            raise Blocked(f"unimplemented settings {missing}" if spec.blocked_by is config.unimplemented else "; ".join(missing))
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


def derived(result):
    """Every field `report` re-derives from the raw data."""
    status, reasons = decide(result)
    out = {"status": status, "reasons": reasons}
    if result.get("schema", 1) >= 2:
        execution, execution_reasons = execution_check(result)
        h, why = hypothesis_result(result)
        out.update(execution_status=execution, execution_reasons=execution_reasons, hypothesis_result=h,
                   hypothesis_reason=why)
    return out


def _write(out, result):
    result.update(derived(result))
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(markdown(result))


def report(name, root=RESULTS):
    """Re-derive every status field from the saved raw data and refuse a result whose recorded fields disagree.
    Rewrites summary.md for current-schema results only; a Ticket 01 (schema 1) result is checked and printed with its
    hypothesis outcome derived on the fly, and its files are left exactly as recorded."""
    path = Path(root) / name / "result.json"
    if not path.exists():
        raise SystemExit(f"no experiment {name!r} under {root}")
    result = json.loads(path.read_text())
    for key in ("config_sha256", "input_sha256", "tolerances_sha256"):
        if not result.get("identity", {}).get(key):
            raise SystemExit(f"{path}: identity.{key} is missing")
    expected = derived(result)
    if bad := {k: (result.get(k), v) for k, v in expected.items() if result.get(k) != v}:
        raise SystemExit(f"{path}: recorded fields disagree with the raw data: {bad}")
    if result.get("schema", 1) >= 2:
        text = markdown(result)
        (path.parent / "summary.md").write_text(text)
        return text
    execution, _ = execution_check(result)
    h, why = hypothesis_result(result)
    return ((path.parent / "summary.md").read_text() + f"\n(derived by report, not recorded: execution_status {execution}, "
            f"hypothesis_result {h}: {why})\n")


def markdown(r):
    lines = []
    if r.get("fault_injection"):
        lines.append(f"**FAULT INJECTION RUN** ({r.get('perturbation', {}).get('spec')}): a FAIL is the expected outcome.\n")
    ident = r["identity"]
    lines += [f"# {r['name']}: {r['case']} ({r['kind']})", "",
              f"**Status: {r['status']}**" + (f" (execution {r['execution_status']}; hypothesis "
              f"{(r.get('hypothesis') or {}).get('id', '-')}: {r['hypothesis_result']}, {r['hypothesis_reason']})"
              if "execution_status" in r else ""), ""]
    lines += [f"- {x}" for x in r["reasons"][:20]] + ([f"- ... {len(r['reasons']) - 20} more"] if len(r["reasons"]) > 20 else [])
    p = r["provenance"]
    lines += ["", f"- commit `{p['git_commit']}` (tracked tree dirty: {p['git_tracked_dirty']}), torch {p['packages']['torch']}, "
              f"transformers {p['packages']['transformers']}, peft {p['packages']['peft']}, GPU {p['gpu']}",
              f"- config `{ident['config_path']}` sha256 `{ident['config_sha256'][:16]}`; fields used: {ident['config_fields_used']}; "
              f"backend `{ident['execution']['backend']}`",
              f"- input sha256 `{ident['input_sha256'][:16]}`, tolerances sha256 `{ident['tolerances_sha256'][:16]}`", ""]
    if r.get("comparisons"):
        lines += ["| fixture / model | quantity | tensors | pass | worst max_abs | worst rel | bound |", "|---|---|---|---|---|---|---|"]
        groups = {}
        for c in r["comparisons"]:
            groups.setdefault((c.get("fixture") or c.get("model"), c["quantity"]), []).append(c)
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
    m = r.get("measurements") or {}
    if m.get("depths"):
        lines += ["", "| depth | max slab B | planned buffers B | streamed peak allocated B | resident peak allocated B | "
                      "build alloc delta B | view problems |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {d['depth']} | {d['max_slab_bytes']} | {d['planned_buffer_bytes']} | {d['streamed_peak_allocated']} | "
                  f"{d['resident_peak_allocated']} | {d['build_allocated_delta']} | {d['view_problems']} |" for d in m["depths"]]
    elif m:
        text = json.dumps(m, default=str)
        lines += ["", "measurements: `" + (text if len(text) < 4000 else text[:4000] + " ...` (truncated; see result.json)`") + "`"]
    if r.get("memory"):
        lines += ["", *memory_lines(r["memory"])]
    return "\n".join(lines) + "\n"


def memory_lines(m):
    """The memory snapshot with each reading's meaning; never combined into one number."""
    g, cpu = m.get("gpu"), m.get("cpu", {})
    out = [f"memory snapshot ({m.get('kind')}), evidence grade **{m.get('evidence_grade', 'not recorded')}**:",
           f"- CPU RSS {cpu.get('rss_mib')} MiB, high-water mark {cpu.get('peak_rss_mib')} MiB (this process)"]
    if g:
        out += [f"- PyTorch allocator, this process: peak allocated {g['peak_allocated_mib']:.1f} MiB, peak reserved "
                f"{g['peak_reserved_mib']:.1f} MiB (excludes the CUDA context and non-PyTorch allocations)",
                f"- cudaMemGetInfo total - free: {g['device_used_mib_mem_get_info']:.1f} MiB, device-wide (every process "
                "and context on the GPU, not process-specific)",
                f"- nvidia-smi raw: compute apps `{g['nvidia_smi_compute_apps_raw']}`, memory.used "
                f"`{g['nvidia_smi_memory_used_raw']}` (unreliable under WSL2)"]
    return out
