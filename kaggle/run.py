"""Kaggle kernel runner for qwen-grpo (submitted by scripts/kaggle_submit.py; see kaggle/README.md).

Standard library only. On the Kaggle machine it:
  1. clones the public repository at the pinned commit (SUBMISSION["commit"]) into /tmp and verifies the checkout;
  2. creates an isolated virtualenv with the repository's pinned requirements (torch from the CPU or cu126 index);
  3. records the environment (GPU model, compute capability, native bf16 support, memory, RAM, disk, versions);
  4. runs the steps of SUBMISSION["mode"] (cpu | gpu-tests | gpu-short | gpu-train), each as a subprocess with its raw
     output kept;
  5. writes /kaggle/working/evidence/ (raw files, plus the derived results.json) and exits non-zero on any failure.

Nothing here changes training code, configs or tolerances. gpu-short trains the repository's committed float32
diagnostic configs (configs/experiments/kaggle_06b_fp32_*.yaml: the T4 has no native bfloat16), whose resolved form is
copied into the evidence. A step that cannot run on this hardware is SKIPPED with its reason, never counted as passed.
Every mode also records `scripts/experiment.py plan --target local` (which study entries this machine can run).
"""
import datetime as _dt
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Replaced by scripts/kaggle_submit.py when the kernel is rendered; a placeholder refuses to run.
SUBMISSION = {"placeholder": True}

WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd() / "kaggle_working"
EVIDENCE = WORK / "evidence"
RAW = EVIDENCE / "raw"
TMP = Path(os.environ.get("KAGGLE_RUNNER_TMP", "/tmp"))
REPO = TMP / "qwen-grpo"
VENV = TMP / "qwen-grpo-venv"
PY = VENV / "bin" / "python"
INPUT = Path(os.environ.get("KAGGLE_RUNNER_INPUT", "/kaggle/input"))  # attached datasets (gpu-train resume)
RESUME_MARKER = "qwen_grpo_resume.json"
T0 = time.time()  # kernel start: gpu-train's time budget counts setup too
RESULT_EXTRA = {}  # mode-specific top-level fields of results.json (gpu-train: the resolved config sha256)
# Local testing of this runner only (never set on Kaggle): reuse an existing interpreter instead of building the venv.
LOCAL_PYTHON = os.environ.get("KAGGLE_RUNNER_PYTHON")
TORCH_INDEX = {"cpu": "https://download.pytorch.org/whl/cpu", "gpu": "https://download.pytorch.org/whl/cu126"}

# Mode A: a quick subset (default for the first submission) or the whole CPU suite.
CPU_QUICK_TESTS = ["tests/test_config.py", "tests/test_profiles.py", "tests/test_experiment.py",
                   "tests/test_sentiment.py", "tests/test_grpo.py", "tests/test_policy.py",
                   "tests/test_qwen3_forward.py", "tests/test_layer_stream.py", "tests/test_backward.py",
                   "tests/test_trainer.py", "tests/test_streaming.py", "tests/test_audit.py", "tests/test_runs.py"]
# Mode B: the files with CUDA-parametrized items (resident vs L2L forward, checkpointed backward / LoRA gradients,
# sync vs async prefetch, buffer reuse and synchronization, pinned store, streamed generation, DFlash exactness).
GPU_TEST_FILES = ["tests/test_layer_stream.py", "tests/test_backward.py", "tests/test_prefetch.py",
                  "tests/test_generator.py", "tests/test_weight_store.py", "tests/test_output_head.py",
                  "tests/test_backend.py", "tests/test_dflash.py", "tests/test_profiler.py"]
# Mode B lab cases (each grid mixes float32 and bfloat16; they run only where bf16 is native).
LAB_CASES = [("forward_equivalence", "H01"), ("gradient_equivalence", "H02"), ("generation_equivalence", "H04"),
             ("update_equivalence", "H05")]

# pytest plugin for GPU mode: skip CUDA x bfloat16 items when the GPU has no native bf16, with the reason.
BF16_PLUGIN = r'''
import os, pytest, torch
REASON = os.environ.get("KAGGLE_NO_NATIVE_BF16")
def _is_cuda(v):
    return (isinstance(v, str) and v.startswith("cuda")) or (isinstance(v, torch.device) and v.type == "cuda")
def pytest_collection_modifyitems(config, items):
    if not REASON:
        return
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        vals = list(params.values())
        if any(_is_cuda(v) for v in vals) and any(v is torch.bfloat16 or v == "bfloat16" for v in vals):
            item.add_marker(pytest.mark.skip(reason=f"hardware: {REASON}"))
'''


def now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


LOG = None


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sh(cmd, name, cwd=None, env=None, timeout=None, check=False):
    """Run cmd, keep its combined raw output in raw/<name>.log; returns (returncode, seconds)."""
    out = RAW / f"{name}.log"
    log(f"$ {' '.join(map(str, cmd))}  (cwd={cwd or Path.cwd()}, output raw/{out.name})")
    t0 = time.time()
    with open(out, "w") as f:
        f.write(f"# {now()} $ {' '.join(map(str, cmd))}\n")
        f.flush()
        try:
            rc = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        f.write(f"\n# {now()} exit {rc}\n")
    log(f"  -> exit {rc} after {time.time() - t0:.0f} s")
    if check and rc != 0:
        raise RuntimeError(f"{name} failed (exit {rc}); see raw/{out.name}")
    return rc, time.time() - t0


def capture(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else f"ERROR exit {r.returncode}: {r.stderr.strip()[:500]}"
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"ERROR {type(e).__name__}"


# ---------------------------------------------------------------------------------------------------------- setup

def clone():
    commit, url = SUBMISSION["commit"], SUBMISSION["repo_url"]
    if REPO.exists():
        shutil.rmtree(REPO)
    sh(["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout", url, REPO], "git_clone", check=True)
    sh(["git", "-C", REPO, "checkout", "--quiet", "--detach", commit], "git_checkout", check=True)
    head = capture(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    if head != commit:
        raise RuntimeError(f"checked out {head}, expected the pinned commit {commit}")
    (EVIDENCE / "git_commit.txt").write_text(head + "\n")
    log(f"repository {url} at {head}")


def venv(kind):
    global PY
    if LOCAL_PYTHON:
        PY = Path(LOCAL_PYTHON)
        log(f"LOCAL TEST: KAGGLE_RUNNER_PYTHON={PY}; the pinned venv was NOT built")
        sh([PY, "-m", "pip", "freeze"], "pip_freeze")
        return
    rc, _ = sh([sys.executable, "-m", "venv", VENV], "venv_create")
    if rc == 0:
        pip = [PY, "-m", "pip"]
    else:  # e.g. Kaggle's Python 3.13 image has no ensurepip: a pip-less venv, filled by the system pip (>= 22.3)
        log("venv with pip failed; using a venv without pip, populated with the system pip's --python option")
        if VENV.exists():
            shutil.rmtree(VENV)
        sh([sys.executable, "-m", "venv", "--without-pip", VENV], "venv_create_without_pip", check=True)
        pip = [sys.executable, "-m", "pip", "--python", PY]
    install = [*pip, "install", "--disable-pip-version-check", "--no-input"]
    torch_pin = next(l.split("#")[0].strip() for l in (REPO / "requirements.txt").read_text().splitlines()
                     if l.startswith("torch=="))
    sh([*install, "--index-url", TORCH_INDEX[kind], torch_pin], "pip_torch", check=True)
    sh([*install, "-r", REPO / "requirements.txt"], "pip_requirements", check=True)
    sh([*pip, "freeze"], "pip_freeze")


def environment():
    probe = r'''
import json, torch, sys, platform
d = {"python": sys.version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda,
     "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
     "cuda_available": torch.cuda.is_available(), "devices": []}
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    d["devices"].append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
        "total_bytes": total, "free_bytes_at_start": free,
        "bf16_native": torch.cuda.is_bf16_supported(including_emulation=False) if i == 0 else None,
        "bf16_including_emulation": torch.cuda.is_bf16_supported(including_emulation=True) if i == 0 else None})
for lib in ("transformers", "peft"):
    d[lib] = __import__(lib).__version__
print(json.dumps(d))
'''
    env = {"utc_start": STARTED, "submission": SUBMISSION, "local_test_python": LOCAL_PYTHON, "kaggle": {k: os.environ.get(k) for k in
           ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_CONTAINER_NAME")},
           "host": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "runner_python": sys.version.split()[0]},
           "nvidia_smi": capture(["nvidia-smi", "--query-gpu=index,name,compute_cap,driver_version,memory.total,memory.used",
                                  "--format=csv"]) if shutil.which("nvidia-smi") else "not available"}
    try:  # CPU model: bitwise CPU references (research/l2l/fixtures) depend on it
        env["host"]["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                                        if l.startswith("model name"))
    except (OSError, StopIteration):
        env["host"]["cpu_model"] = None
    try:
        mem = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        env["host"]["mem_total"], env["host"]["mem_available"] = mem["MemTotal"].strip(), mem["MemAvailable"].strip()
    except (OSError, KeyError):
        env["host"]["mem_total"] = None
    for p in ("/kaggle/working", "/tmp"):
        if Path(p).exists():
            u = shutil.disk_usage(p)
            env["host"][f"disk_free_bytes{p.replace('/', '_')}"] = u.free
    try:
        env["torch"] = json.loads(subprocess.run([str(PY), "-c", probe], capture_output=True, text=True,
                                                 timeout=300).stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - recorded, not hidden
        env["torch"] = {"error": f"{type(e).__name__}: {e}"}
    (EVIDENCE / "environment.json").write_text(json.dumps(env, indent=1))
    return env


class SmiSampler:
    """nvidia-smi sampled every second for the whole run, raw CSV (gpu_memory.csv)."""

    def __init__(self):
        self.proc = None

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self.f = open(EVIDENCE / "gpu_memory.csv", "w")
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,name,memory.used,memory.total,"
                                          "utilization.gpu", "--format=csv", "-l", "1"], stdout=self.f,
                                         stderr=subprocess.STDOUT)
        else:
            (EVIDENCE / "gpu_memory.csv").write_text("# nvidia-smi not available on this machine: no GPU telemetry\n")
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)
            self.f.close()
        return False


# ---------------------------------------------------------------------------------------------------------- steps

RESULTS = []


def record(name, status, **kw):
    RESULTS.append({"name": name, "status": status, **kw})
    log(f"STEP {name}: {status}" + (f" ({kw['reason']})" if kw.get("reason") else ""))


def junit_counts(path):
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    suites = [root] if root.tag == "testsuite" else list(root)
    c = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
    c["passed"] = c["tests"] - c["failures"] - c["errors"] - c["skipped"]
    skipped = []
    for s in suites:
        for tc in s.iter("testcase"):
            sk = tc.find("skipped")
            if sk is not None:
                skipped.append({"test": f"{tc.get('classname')}::{tc.get('name')}", "reason": sk.get("message")})
    c["skipped_tests"] = skipped
    return c


def pytest_step(name, files, env, extra=()):
    xml = RAW / f"{name}.junit.xml"
    cmd = [PY, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", f"--junitxml={xml}", *extra, *files]
    rc, secs = sh(cmd, name, cwd=REPO, env=env, timeout=SUBMISSION["step_timeout_s"])
    counts = junit_counts(xml)
    status = "PASS" if rc == 0 else "FAIL"
    if counts is None:
        status = "ERROR"
    record(name, status, command=" ".join(map(str, cmd)), returncode=rc, seconds=round(secs), counts=counts)


def base_env(cuda):
    env = {k: v for k, v in os.environ.items() if not k.startswith("KAGGLE_")}  # no tokens reach the tests
    env.update(PYTHONHASHSEED="0", TOKENIZERS_PARALLELISM="false")
    env["CUDA_VISIBLE_DEVICES"] = "0" if cuda else ""  # one device: two T4s are two separate 16 GB GPUs, not a pool
    return env


def mode_cpu(env):
    log("Mode A: CPU smoke tests (GPU hidden)")
    e = base_env(cuda=False)
    rc, _ = sh([PY, "-c", "import qwen_grpo.trainer, qwen_grpo.lab, qwen_grpo.execution.backend, qwen_grpo.export.merge; "
                "print('imports ok')"], "imports", cwd=REPO, env=e)
    record("imports", "PASS" if rc == 0 else "FAIL", returncode=rc)
    rc, _ = sh([PY, "-c", CONFIG_CHECK], "config_validation", cwd=REPO, env=e)
    record("config_validation", "PASS" if rc == 0 else "FAIL", returncode=rc)
    files = CPU_QUICK_TESTS if SUBMISSION["cpu_scope"] == "quick" else []
    pytest_step("pytest_cpu_" + SUBMISSION["cpu_scope"], files, e)


CONFIG_CHECK = r'''
import glob, json, sys
from qwen_grpo import config
bad, out = [], {}
paths = sorted(glob.glob("configs/*.yaml") + glob.glob("configs/experiments/*.yaml") + glob.glob("configs/historical/*.yaml")
               + glob.glob("configs/profiles/*.yaml") + glob.glob("research/l2l/configs/*.yaml"))
for p in paths:
    try:
        if config.is_abstract(p):
            out[p] = {"abstract": "a shared base: extended, never loaded as a run"}; continue
        c = config.load(p, historical=True); out[p] = {"sha256": config.sha256(c), "unimplemented": config.unimplemented(c)}
    except Exception as e:
        bad.append(p); out[p] = {"error": f"{type(e).__name__}: {e}"}
print(json.dumps(out, indent=1))
sys.exit(1 if bad else 0)
'''


def mode_gpu_tests(env):
    log("Mode B: GPU correctness tests (device 0 only)")
    dev = (env.get("torch") or {}).get("devices") or []
    if not dev:
        record("gpu_tests", "FAIL", reason="no CUDA device visible to torch (was the kernel submitted with a GPU?)")
        return
    d0 = dev[0]
    e = base_env(cuda=True)
    plugin = REPO / "kaggle_bf16_guard.py"
    plugin.write_text(BF16_PLUGIN)
    e["PYTHONPATH"] = str(REPO)
    no_bf16 = None if d0["bf16_native"] else (f"{d0['name']} (compute capability {d0['capability']}) has no native "
                                              "bfloat16; CUDA x bfloat16 items need it (torch.cuda.is_bf16_supported("
                                              "including_emulation=False) is False)")
    if no_bf16:
        e["KAGGLE_NO_NATIVE_BF16"] = no_bf16
    pytest_step("pytest_gpu", GPU_TEST_FILES, e, extra=("-p", "kaggle_bf16_guard"))
    for case, hyp in LAB_CASES:
        name = f"lab_{case}_async_cuda"
        if no_bf16:
            record(name, "SKIPPED", reason=f"hardware: {no_bf16}; the lab grid mixes float32 and bfloat16 and is not "
                                           "split here (fp32 CUDA coverage comes from pytest_gpu)")
            continue
        rc, secs = sh([PY, "scripts/l2l_lab.py", "run", "--case", case, "--config", "research/l2l/configs/async.yaml",
                       "--device", "cuda", "--name", f"kaggle_{case}_async", "--hypothesis", hyp], name, cwd=REPO, env=e,
                      timeout=SUBMISSION["step_timeout_s"])
        res = REPO / "research/l2l/experiments" / f"kaggle_{case}_async" / "result.json"
        status = json.loads(res.read_text())["status"] if res.exists() else "ERROR"
        if res.exists():
            shutil.copytree(res.parent, EVIDENCE / "artifacts" / res.parent.name, dirs_exist_ok=True)
        record(name, "PASS" if status == "PASS" and rc == 0 else "FAIL", lab_status=status, returncode=rc,
               seconds=round(secs))


# Mode C: the committed float32 diagnostic configs, one per backend; they differ in `execution` and run.name only
# (tests/test_profiles.py). The bf16 comparison is study entry H05.real_0.6b_bf16 (configs/experiments/qwen3_0.6b_*.yaml).
SHORT_RUNS = ["kaggle_06b_fp32_resident", "kaggle_06b_fp32_l2l_sync", "kaggle_06b_fp32_l2l_async"]
RESOLVE = ("import yaml; from qwen_grpo import config; c = config.load({path!r}); "
           "print('# {name} sha256 ' + config.sha256(c)); print(yaml.safe_dump(config.as_dict(c), sort_keys=False))")


def train_short(name, e):
    path = f"configs/experiments/{name}.yaml"
    sh([PY, "-c", RESOLVE.format(path=path, name=name)], f"resolve_{name}", cwd=REPO, env=e)
    with open(EVIDENCE / "resolved_config.yaml", "a") as f:
        f.write(f"--- # {path}\n" + (REPO / path).read_text() + "\n" + (RAW / f"resolve_{name}.log").read_text() + "\n")
    step = "train_" + name.removeprefix("kaggle_06b_fp32_") + "_fp32"
    rc, secs = sh([PY, "scripts/run.py", "train", "--config", path], step, cwd=REPO, env=e,
                  timeout=SUBMISSION["step_timeout_s"])
    run = REPO / "runs" / name
    if run.exists():
        shutil.copytree(run, EVIDENCE / "artifacts" / "runs" / name, dirs_exist_ok=True)
    tj = run / "training" / "training.json"
    summary = None
    if tj.exists():
        t = json.loads(tj.read_text())
        rows = t.get("training", [])
        summary = {"steps": len(rows), "max_steps": t["yaml_config"]["training"]["max_steps"], "status": t.get("status"),
                   "frozen_changed": t.get("frozen_changed"), "params_equal_init": t.get("params_equal_init"),
                   "memory": t.get("memory"), "final_checkpoint": t.get("final_checkpoint"),
                   "per_step": [{k: r.get(k) for k in ("step", "mean_reward", "mixed_groups", "loss", "grad_norm",
                                                       "grad_norm_A", "grad_norm_B", "update_norm", "change_from_init",
                                                       "peak_reserved_gib", "step_seconds")} for r in rows],
                   "evaluations": [{"step": v["step"], "metrics": v["metrics"]} for v in t.get("evaluations", [])]}
    ok = (rc == 0 and summary is not None and summary["steps"] == summary["max_steps"] and not summary["frozen_changed"]
          and summary["params_equal_init"] is False and summary["final_checkpoint"] is not None)
    record(step, "PASS" if ok else "FAIL", returncode=rc, seconds=round(secs), training=summary, precision="float32",
           criteria="exit 0; every configured optimizer step; frozen weights unchanged; LoRA parameters changed from "
                    "init; final checkpoint saved and reloaded bitwise")
    return rc == 0


def mode_gpu_short(env):
    log("Mode C: short GRPO + LoRA on Qwen3-0.6B, resident vs L2L sync vs L2L async, float32 diagnostic (device 0)")
    dev = (env.get("torch") or {}).get("devices") or []
    if not dev:
        record("gpu_short", "FAIL", reason="no CUDA device visible to torch")
        return
    d0, e = dev[0], base_env(cuda=True)
    trained = {name: train_short(name, e) for name in SHORT_RUNS}
    ref = SHORT_RUNS[0]
    for cand in SHORT_RUNS[1:]:
        tag = cand.removeprefix("kaggle_06b_fp32_")
        step = f"compare_resident_vs_{tag}_fp32"
        if not (trained[ref] and trained[cand]):
            record(step, "FAIL", reason="a training run failed; nothing to compare")
            continue
        rc, _ = sh([PY, "scripts/compare_runs.py", ref, cand, "--out", EVIDENCE / "artifacts" / f"{step}.json"], step,
                   cwd=REPO, env=e)
        record(step, "PASS" if rc == 0 else "FAIL", returncode=rc, precision="float32",
               criteria="scripts/compare_runs.py exit 0: configs differ in execution/run only and every compared "
                        "quantity is bitwise identical (tolerances.yaml update_equivalence: 0)")
    why = (f"hardware: {d0['name']} (compute capability {d0['capability']}) has no native bfloat16" if not d0["bf16_native"]
           else "not part of this short mode")
    record("compare_backends_bf16", "SKIPPED", reason=f"{why}; the bf16 comparison is study entry H05.real_0.6b_bf16 "
                                                      "(configs/experiments/qwen3_0.6b_*.yaml) on an Ampere-or-newer GPU")


def study_plan(cuda):
    """Which study entries this machine can run (recorded, never a pass/fail step of its own)."""
    sh([PY, "scripts/experiment.py", "plan", "--target", "local"], "study_plan", cwd=REPO, env=base_env(cuda=cuda))


# Mode D: one full training run (scripts/kaggle_submit.py --mode gpu-train), optionally resumed from a packed run.

def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore_run(input_root, repo, expected_marker_sha, run_name):
    """Put the packed run (an attached dataset: run.tar + the marker, or a directory Kaggle extracted) at
    repo/runs/<run_name> and verify every file against the marker. -> (ok, detail). Nothing unverified is used."""
    markers = sorted(Path(input_root).rglob(RESUME_MARKER)) if Path(input_root).exists() else []
    if not markers:
        return False, f"no {RESUME_MARKER} under {input_root}: the resume dataset is not attached"
    found = [m for m in markers if sha256_file(m) == expected_marker_sha]
    if not found:
        return False, f"{len(markers)} resume marker(s) found, none with the submitted sha256 (a stale dataset version?)"
    m = found[0]
    meta = json.loads(m.read_text())
    if meta["run_name"] != run_name:
        return False, f"the dataset holds run {meta['run_name']!r}, not {run_name!r}"
    dest = Path(repo) / "runs" / run_name
    if dest.exists():
        return False, f"{dest} exists before the restore"
    tars = list(m.parent.rglob("run.tar"))
    if tars:
        with tarfile.open(tars[0]) as t:
            bad = [x.name for x in t.getmembers() if not x.name.startswith(f"runs/{run_name}/")]
            if bad:
                return False, f"run.tar holds paths outside runs/{run_name}/: {bad[:3]}"
            t.extractall(repo, filter="data")
    else:
        src = next((d for d in m.parent.rglob(run_name) if (d / "manifest.json").exists()), None)
        if src is None:
            return False, "neither run.tar nor an extracted run directory next to the marker"
        shutil.copytree(src, dest)
    bad = [rel for rel, h in meta["files"].items() if not (dest / rel).is_file() or sha256_file(dest / rel) != h]
    extra = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()
                   and str(p.relative_to(dest)) not in meta["files"])
    if bad or extra:
        return False, f"restored files differ from the marker: changed/missing {bad[:5]}, unexpected {extra[:5]}"
    return True, f"{len(meta['files'])} files restored and verified into runs/{run_name}"


def run_until(cmd, name, cwd, env, budget_s, on_start=None):
    """Like sh(), but at budget_s it sends SIGINT (the trainer then ends the run `interrupted`, keeping its saved
    checkpoints) and kills only if that does not end it within 10 minutes. -> (returncode, seconds, reason or None)."""
    out = RAW / f"{name}.log"
    log(f"$ {' '.join(map(str, cmd))}  (budget {budget_s / 3600:.2f} h, output raw/{out.name})")
    t0, reason = time.time(), None
    with open(out, "w") as f:
        f.write(f"# {now()} $ {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen(list(map(str, cmd)), cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if on_start:
            on_start(p.pid)
        try:
            rc = p.wait(timeout=max(budget_s, 1))
        except subprocess.TimeoutExpired:
            reason = f"time budget reached after {budget_s / 3600:.2f} h: SIGINT sent"
            p.send_signal(signal.SIGINT)
            try:
                rc = p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait()
                reason += "; killed 10 min later"
        f.write(f"\n# {now()} exit {rc}\n")
    log(f"  -> exit {rc} after {time.time() - t0:.0f} s" + (f" ({reason})" if reason else ""))
    return rc, time.time() - t0, reason


class HostSampler:
    """Every 5 s: host RAM, free disk, and the training process's RSS and peak RSS (VmHWM), raw CSV."""
    HEADER = "utc,mem_total_kib,mem_available_kib,disk_free_working_bytes,disk_free_tmp_bytes,train_pid,train_rss_kib,train_hwm_kib\n"

    def __init__(self, path, interval=5.0):
        self.path, self.interval, self.pid, self.stop = path, interval, None, threading.Event()

    def track(self, pid):
        self.pid = pid

    def status(self, key):
        try:
            return next(int(l.split()[1]) for l in open(f"/proc/{self.pid}/status") if l.startswith(key))
        except (OSError, StopIteration, TypeError, ValueError):
            return ""

    def loop(self):
        with open(self.path, "w") as f:
            f.write(self.HEADER)
            while True:
                try:
                    mem = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
                    total, avail = (int(mem[k].split()[0]) for k in ("MemTotal", "MemAvailable"))
                except (OSError, KeyError, ValueError):
                    total = avail = ""
                disk = [shutil.disk_usage(d).free if Path(d).exists() else "" for d in (WORK, TMP)]
                f.write(f"{now()},{total},{avail},{disk[0]},{disk[1]},{self.pid or ''},{self.status('VmRSS:')},"
                        f"{self.status('VmHWM:')}\n")
                f.flush()
                if self.stop.wait(self.interval):
                    return

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=30)
        return False


class AppsSampler:
    """nvidia-smi per-process GPU memory every second (may list nothing in a container), raw CSV."""

    def __enter__(self):
        self.proc = None
        if shutil.which("nvidia-smi"):
            self.f = open(EVIDENCE / "gpu_apps.csv", "w")
            self.proc = subprocess.Popen(["nvidia-smi", "--query-compute-apps=timestamp,pid,process_name,used_memory",
                                          "--format=csv", "-l", "1"], stdout=self.f, stderr=subprocess.STDOUT)
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=30)
            self.f.close()
        return False


def csv_column(path, column):
    """Numeric values of one column of a raw CSV (nvidia-smi or HostSampler); units such as 'MiB' are stripped."""
    try:
        lines = [l for l in Path(path).read_text().splitlines() if l.strip()]
    except OSError:
        return []
    if not lines:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    idx = next((i for i, h in enumerate(header) if h.startswith(column)), None)
    vals = []
    for l in lines[1:]:
        parts = l.split(",")
        if idx is not None and idx < len(parts):
            m = re.match(r"\s*([0-9.]+)", parts[idx])
            if m:
                vals.append(float(m.group(1)))
    return vals


def perf_summary(run, evidence):
    """Performance and outcome of one training run from its own files plus the raw samplers (no claim is derived from
    the allocator alone: it excludes the CUDA context)."""
    read = lambda p: json.loads(Path(p).read_text()) if Path(p).exists() else {}
    man, t = read(run / "manifest.json"), read(run / "training" / "training.json")
    rows, tim, cfg = t.get("training", []), t.get("timings", []), t.get("yaml_config", {})
    progress = read(run / "summary.json").get("progress") or {}  # training.json is written when a run ends; an
    # interrupted run's progress is rebuilt from its last checkpoint into summary.json (its per-step rows are there too)
    num = sorted({k for r in tim for k, v in r.items() if isinstance(v, (int, float)) and k not in
                  ("step", "cpu_rss_bytes", "weight_bytes_copied")})
    rewards = [r.get("mean_reward") for r in rows]
    smi, apps, host = evidence / "gpu_memory.csv", evidence / "gpu_apps.csv", evidence / "host_resources.csv"
    mx = lambda v: max(v) if v else None
    mn = lambda v: min(v) if v else None
    final = t.get("final") or {}
    return {
        "run": run.name, "status": man.get("status"), "status_history": man.get("status_history"),
        "steps": len(rows) if t else progress.get("optimizer_steps", 0),
        "max_steps": (cfg.get("training") or {}).get("max_steps") or progress.get("max_steps"),
        "per_step_source": "training/training.json" if t else "summary.json progress (interrupted: per-step rows are in "
                                                                "the last checkpoint's training_state.pt)",
        "train_seconds": t.get("train_seconds"),
        "seconds_per_step": round(sum(r.get("step_s") or 0 for r in tim) / len(tim), 3) if tim else None,
        "timings_total_s": {k: round(sum(r.get(k) or 0 for r in tim), 2) for k in num},
        "weight_bytes_copied_total": sum(r["weight_bytes_copied"] for r in tim if r.get("weight_bytes_copied") is not None)
        if any(r.get("weight_bytes_copied") is not None for r in tim) else None,
        "cpu_rss_max_bytes": mx([r["cpu_rss_bytes"] for r in tim if r.get("cpu_rss_bytes")]),
        "allocator_gib": {"peak_allocated_max": mx([r["peak_allocated_gib"] for r in rows if r.get("peak_allocated_gib") is not None]),
                          "peak_reserved_max": mx([r["peak_reserved_gib"] for r in rows if r.get("peak_reserved_gib") is not None]),
                          "note": "PyTorch allocator only: excludes the CUDA context"},
        "nvidia_smi_mib": {"device_used_max": mx(csv_column(smi, "memory.used")),
                           "process_used_max": mx(csv_column(apps, "used_gpu_memory") or csv_column(apps, "used_memory")),
                           "note": "device-wide every 1 s for the whole kernel; per-process may be empty in a container"},
        "host": {"mem_total_kib": mx(csv_column(host, "mem_total_kib")), "mem_available_min_kib": mn(csv_column(host, "mem_available_kib")),
                 "train_rss_max_kib": mx(csv_column(host, "train_rss_kib")), "train_peak_rss_kib": mx(csv_column(host, "train_hwm_kib")),
                 "disk_free_working_min_bytes": mn(csv_column(host, "disk_free_working_bytes"))},
        "rewards": {"per_step": rewards, "mean": round(sum(rewards) / len(rewards), 4) if rewards else None,
                    "first_10": rewards[:10], "last_10": rewards[-10:]},
        "evaluations": [{"step": e["step"], **{k: e["metrics"].get(k) for k in ("accuracy", "predicted_positive_rate",
                                                                               "invalid_rate", "macro_f1")}}
                        for e in t.get("evaluations", [])],
        "final_validation": {k: final.get(k) for k in ("base", "trained", "accuracy_diff", "bootstrap_ci95", "auc")} if final else None,
        "frozen_changed": t.get("frozen_changed"), "final_checkpoint": t.get("final_checkpoint"),
        "environment": {k: (t.get("environment") or {}).get(k) for k in ("git_commit", "git_dirty", "packages", "gpu", "models",
                                                                       "host")},
        "resume": SUBMISSION.get("resume"),
    }


def train_status(rc, perf, run):
    """PASS only for a completed run with every step, unchanged frozen weights and a final checkpoint; INCOMPLETE for an
    interrupted run that left a resumable checkpoint; FAIL otherwise."""
    resumable = sorted(int(p.name[4:]) for p in (run / "checkpoints").glob("step*")
                       if p.name[4:].isdigit() and int(p.name[4:]) > 0 and (p / "training_state.pt").exists())
    complete = (rc == 0 and perf["status"] in ("completed", "complete") and perf["steps"] == perf["max_steps"]
                and perf["frozen_changed"] is not None and not perf["frozen_changed"] and perf["final_checkpoint"])
    if complete:
        return "PASS", None
    if perf["status"] == "interrupted" and resumable:
        return "INCOMPLETE", f"interrupted at {perf['steps']}/{perf['max_steps']} steps; resumable from step {resumable[-1]}"
    return "FAIL", f"exit {rc}, run status {perf['status']}, {perf['steps']}/{perf['max_steps']} steps"


def mode_gpu_train(env):
    t = SUBMISSION["train"]
    name = t["run_name"]
    log(f"Mode D: training run {name} ({t['config']}, {t['backend']}, {t['dtype']}, {t['train_samples']} prompts, "
        f"{t['max_steps']} steps)" + (f", resumed from submission {SUBMISSION['resume']['from_submission']}"
                                       if SUBMISSION.get("resume") else ""))
    dev = (env.get("torch") or {}).get("devices") or []
    if not dev:
        record("train", "FAIL", reason="no CUDA device visible to torch")
        return
    if t["dtype"] == "bfloat16" and not dev[0]["bf16_native"]:
        record("train", "FAIL", reason=f"{dev[0]['name']} has no native bfloat16; precision is never switched")
        return
    e = base_env(cuda=True)
    sh([PY, "-c", RESOLVE.format(path=t["config"], name=name)], "resolve_config", cwd=REPO, env=e)
    text = (RAW / "resolve_config.log").read_text()
    m = re.search(r"^# \S+ sha256 ([0-9a-f]{64})$", text, re.MULTILINE)
    RESULT_EXTRA["resolved_config_sha256"] = m.group(1) if m else None
    (EVIDENCE / "resolved_config.yaml").write_text(f"--- # {t['config']}\n" + (REPO / t["config"]).read_text() + "\n" + text)
    if RESULT_EXTRA["resolved_config_sha256"] != t["config_sha256"]:
        record("config_identity", "FAIL", reason=f"resolved sha256 {RESULT_EXTRA['resolved_config_sha256']} != submitted "
                                                 f"{t['config_sha256']}")
        return
    record("config_identity", "PASS", config=t["config"], sha256=t["config_sha256"])
    argv = ["scripts/run.py", "train", "--config", t["config"]]
    if SUBMISSION.get("resume"):
        r = SUBMISSION["resume"]
        ok, detail = restore_run(INPUT, REPO, r["marker_sha256"], name)
        record("restore_checkpoint", "PASS" if ok else "FAIL", reason=None if ok else detail, detail=detail,
               from_submission=r["from_submission"], target=r["target"])
        if not ok:
            return
        argv = ["scripts/run.py", "train", "--config", f"runs/{name}/config.yaml", "--resume", r["target"]]
    budget = t["max_hours"] * 3600 - (time.time() - T0)
    if budget < 600:
        record("train", "FAIL", reason=f"only {budget:.0f} s of the {t['max_hours']} h budget left after setup")
        return
    with HostSampler(EVIDENCE / "host_resources.csv") as hs, AppsSampler():
        rc, secs, reason = run_until([PY, *argv], "train", REPO, e, budget, on_start=hs.track)
    run = REPO / "runs" / name
    if run.exists():  # everything, checkpoints included: the next version resumes from this output
        shutil.copytree(run, EVIDENCE / "artifacts" / "runs" / name, dirs_exist_ok=True)
    perf = perf_summary(run, EVIDENCE)
    (EVIDENCE / "performance_summary.json").write_text(json.dumps(perf, indent=1) + "\n")
    status, why = train_status(rc, perf, run)
    record("train", status, reason="; ".join(x for x in (why, reason) if x) or None, returncode=rc, seconds=round(secs),
           argv=argv, precision=t["dtype"], steps=perf["steps"], max_steps=perf["max_steps"],
           criteria="exit 0; run completed; every configured step; frozen weights unchanged; final checkpoint saved")


# ----------------------------------------------------------------------------------------------------------- main

def overall():
    statuses = [r["status"] for r in RESULTS]
    if not statuses or any(s in ("FAIL", "ERROR") for s in statuses):
        return "FAIL"
    if "INCOMPLETE" in statuses:  # gpu-train: interrupted with a resumable checkpoint; never a pass
        return "INCOMPLETE"
    return "PASS_WITH_SKIPS" if "SKIPPED" in statuses or any((r.get("counts") or {}).get("skipped") for r in RESULTS) \
        else "PASS"


def main():
    global LOG, STARTED
    if SUBMISSION.get("placeholder"):
        sys.exit("kaggle/run.py is a template: submit it with scripts/kaggle_submit.py, which pins the commit and mode")
    for d in (EVIDENCE, RAW):
        d.mkdir(parents=True, exist_ok=True)
    STARTED = now()
    LOG = open(EVIDENCE / "execution.log", "a")
    log(f"submission {SUBMISSION['submission_id']} kernel {SUBMISSION['kernel_id']} mode {SUBMISSION['mode']} "
        f"commit {SUBMISSION['commit']}")
    error = None
    try:
        with SmiSampler():
            clone()
            venv("cpu" if SUBMISSION["mode"] == "cpu" else "gpu")
            env = environment()
            {"cpu": mode_cpu, "gpu-tests": mode_gpu_tests, "gpu-short": mode_gpu_short,
             "gpu-train": mode_gpu_train}[SUBMISSION["mode"]](env)
            study_plan(cuda=SUBMISSION["mode"] != "cpu")
    except Exception as e:  # noqa: BLE001 - every failure is recorded and makes the kernel fail
        error = f"{type(e).__name__}: {e}"
        log(f"ERROR {error}")
        record("runner", "ERROR", reason=error)
    if not (EVIDENCE / "resolved_config.yaml").exists():
        (EVIDENCE / "resolved_config.yaml").write_text(
            "# this mode trains nothing; the repository configs it validated are listed in raw/config_validation.log\n")
    status = overall()
    res = {"submission": SUBMISSION, "utc_start": STARTED, "utc_end": now(), "overall": status, "error": error,
           **RESULT_EXTRA,
           "steps": RESULTS, "rule": "PASS only if no step FAILED or ERRORED; skipped steps or tests are listed and "
                                     "make the overall PASS_WITH_SKIPS, never PASS"}
    (EVIDENCE / "results.json").write_text(json.dumps(res, indent=1))
    lines = [f"{r['name']}: {r['status']}" + (f"  {r['counts']['passed']} passed, {r['counts']['failures']} failed, "
             f"{r['counts']['errors']} errors, {r['counts']['skipped']} skipped" if r.get("counts") else "")
             + (f"  [{r['reason']}]" if r.get("reason") else "") for r in RESULTS]
    (EVIDENCE / "test_results.txt").write_text("\n".join(lines + [f"OVERALL: {status}"]) + "\n")
    log(f"OVERALL {status}")
    LOG.close()
    sys.exit(0 if status in ("PASS", "PASS_WITH_SKIPS") else 1)


STARTED = None
if __name__ == "__main__":
    main()
