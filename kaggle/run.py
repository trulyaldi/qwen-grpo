"""Kaggle kernel runner for qwen-grpo (submitted by scripts/kaggle_submit.py; see kaggle/README.md).

Standard library only. On the Kaggle machine it:
  1. clones the public repository at the pinned commit (SUBMISSION["commit"]) into /tmp and verifies the checkout;
  2. creates an isolated virtualenv with the repository's pinned requirements (torch from the CPU or cu126 index);
  3. records the environment (GPU model, compute capability, native bf16 support, memory, RAM, disk, versions);
  4. runs the steps of SUBMISSION["mode"] (cpu | gpu-tests | gpu-short), each as a subprocess with its raw output kept;
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
import shutil
import subprocess
import sys
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


# ----------------------------------------------------------------------------------------------------------- main

def overall():
    statuses = [r["status"] for r in RESULTS]
    if not statuses or any(s in ("FAIL", "ERROR") for s in statuses):
        return "FAIL"
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
            {"cpu": mode_cpu, "gpu-tests": mode_gpu_tests, "gpu-short": mode_gpu_short}[SUBMISSION["mode"]](env)
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
