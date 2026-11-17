"""Submit qwen-grpo test runs to Kaggle and fetch their evidence, with the official Kaggle CLI (kaggle >= 2.2).

    python scripts/kaggle_submit.py submit   --mode cpu [--scope quick|full] [--commit SHA] [--force]
    python scripts/kaggle_submit.py submit   --mode gpu-tests
    python scripts/kaggle_submit.py submit   --mode gpu-short
    python scripts/kaggle_submit.py submit   --mode gpu-train --config configs/experiments/<exp>.yaml [--max-hours H]
    python scripts/kaggle_submit.py submit   --mode gpu-train --config <same yaml> --resume-from <submission id>
    python scripts/kaggle_submit.py status   --mode cpu          # kaggle kernels status
    python scripts/kaggle_submit.py logs     --mode cpu          # kaggle kernels logs
    python scripts/kaggle_submit.py download --mode cpu          # kaggle kernels output -> kaggle/outputs/...
    python scripts/kaggle_submit.py summary  --mode cpu          # pass/fail/skip from the downloaded evidence

One kernel per mode (<username>/qwen-grpo-<mode>); every submission is a new version of it, so Kaggle keeps the version
history. The kernel source is kaggle/run.py with the submission (commit, mode, ids) written into it; the repository
itself is cloned on Kaggle at the pinned commit, which must already be on GitHub. Nothing else is uploaded (no
credentials, checkpoints, caches or data). Local ledger: kaggle/submissions.jsonl. Before pushing, submit refuses if
the kernel's latest version is still queued/running, or if the same mode and commit already ran (use --force).
`summary` checks that the evidence names the submission recorded in the ledger and exits non-zero unless every step
passed (skipped steps are listed, never counted as passed).

gpu-train: one training run (scripts/run.py train) of a YAML committed at the pinned commit. submit resolves the config
from that commit's tree (what Kaggle will run), refuses what a T4 session cannot run as written (bf16: the T4 has no
native bfloat16, and precision is never switched; models other than Qwen3-0.6B; unimplemented settings), and records the
resolved config's sha256, which the kernel re-checks. Training stops at --max-hours (Ctrl-C, so the run ends
`interrupted` with its checkpoints) to leave time to copy everything to the output before Kaggle's 12 h session limit.
Kaggle's working storage is gone after a version ends, so a run continues in a NEW version: download the interrupted
version's output, then submit with --resume-from <its submission id>. The run directory is packed with a per-file sha256
manifest and uploaded as a PRIVATE Kaggle dataset (kaggle datasets create/version), attached to the new version as
input; the kernel restores it, verifies every file, and resumes from the exact checkpoint (--resume latest for streaming
runs, else the highest step with optimizer state). A resume must use the same commit and config (--force overrides the
commit). Each gpu-train version's output is downloaded by its own version number; --submission <id> selects one for
status / logs / download / summary. gpu-train summary passes only on a completed run: INCOMPLETE (time budget, with a
resumable checkpoint), FAIL and any skip exit non-zero.
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
LEDGER = KAGGLE / "submissions.jsonl"
BUILD = KAGGLE / "build"
OUTPUTS = KAGGLE / "outputs"
MODES = {  # mode: (GPU, accelerator for `kaggle kernels push --accelerator`, kernel run timeout in seconds)
    "cpu": (False, None, 4 * 3600),
    "gpu-tests": (True, "NvidiaTeslaT4", 3 * 3600),
    "gpu-short": (True, "NvidiaTeslaT4", 3 * 3600),
    "gpu-train": (True, "NvidiaTeslaT4", 12 * 3600),  # Kaggle's GPU session limit; training stops at --max-hours
}
TRAIN_MODELS = ("Qwen/Qwen3-0.6B",)  # what a T4 session trains in float32 within hours; Qwen3-4B belongs on Vast.ai
RESUME_MARKER = "qwen_grpo_resume.json"
PLACEHOLDER = 'SUBMISSION = {"placeholder": True}'


def kaggle(*args, capture=True):
    if not shutil.which("kaggle"):
        sys.exit("the kaggle CLI is not on PATH (pipx install kaggle; credentials in ~/.kaggle)")
    r = subprocess.run(["kaggle", *args], capture_output=capture, text=True)
    if capture and r.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed (exit {r.returncode}):\n{r.stdout}{r.stderr}")
    return r.stdout if capture else r.returncode


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def username():
    m = re.search(r"username:\s*(\S+)", kaggle("config", "view"))
    if not m or m.group(1) == "None":
        sys.exit("kaggle config view shows no username: authenticate first (kaggle auth login)")
    return m.group(1)


def kernel_id(mode):
    return f"{username()}/qwen-grpo-{mode}"


def repo_url():
    url = git("remote", "get-url", "origin")
    m = re.match(r"(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?$", url)
    if not m:
        sys.exit(f"origin {url!r} is not a GitHub repository URL")
    return f"https://github.com/{m.group(1)}.git", m.group(1)


def commit_on_github(slug, sha):
    """True if GitHub serves the commit (public repo, anonymous API through gh if present, else a remote branch has it)."""
    if shutil.which("gh"):
        r = subprocess.run(["gh", "api", f"repos/{slug}/commits/{sha}", "--jq", ".sha"], capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() == sha
    return bool(git("branch", "-r", "--contains", sha))


def ledger():
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []


def latest(mode, submission=None):
    rows = [r for r in ledger() if r["mode"] == mode and (submission is None or r["submission_id"] == submission)]
    if not rows:
        sys.exit(f"no {mode} submission{' ' + submission if submission else ''} in {LEDGER.relative_to(ROOT)}")
    return rows[-1]


# --- gpu-train: the config as the pinned commit holds it, and checkpoint transfer between kernel versions ----------

def resolve_at_commit(sha, path):
    """(resolved config dict, sha256) of `path` exactly as the pinned commit holds it (its `extends` chain included)."""
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    if path.parts[0] != "configs":
        sys.exit(f"--config must be a YAML under configs/ (got {path})")
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["git", "-C", str(ROOT), "archive", sha, "configs"], capture_output=True)
        if r.returncode != 0:
            sys.exit(f"git archive {sha[:12]} configs failed: {r.stderr.decode()[:300]}")
        tarfile.open(fileobj=io.BytesIO(r.stdout)).extractall(tmp, filter="data")
        if not (Path(tmp) / path).is_file():
            sys.exit(f"{path} is not in commit {sha[:12]}: commit and push the config first (Kaggle runs the commit)")
        sys.path.insert(0, str(ROOT))
        from qwen_grpo import config
        try:
            cfg = config.load(Path(tmp) / path)
        except (ValueError, SystemExit) as e:
            sys.exit(f"{path} at {sha[:12]} is not a runnable config: {e}")
        return cfg, config.sha256(cfg)


def train_spec(sha, path, max_hours):
    """What gpu-train will run, after the checks that would otherwise fail (or silently differ) on the T4."""
    from qwen_grpo import config
    cfg, digest = resolve_at_commit(sha, path)
    why = [f"model {cfg.model.name}: gpu-train trains {list(TRAIN_MODELS)} (Qwen3-4B: a rented Ampere GPU)"] * (
        cfg.model.name not in TRAIN_MODELS)
    why += ["model.dtype bfloat16: the T4 has no native bfloat16 and precision is never switched silently; use a float32 "
            "diagnostic config (configs/experiments/kaggle_06b_fp32_*)"] * (cfg.model.dtype != "float32")
    why += [f"not implemented: {config.unimplemented(cfg)}"] * bool(config.unimplemented(cfg))
    why += [f"--max-hours {max_hours}: between 0.25 and 11 (Kaggle stops a GPU session at 12 h; the rest is for copying "
            "the output)"] * (not 0.25 <= max_hours <= 11)
    if why:
        sys.exit("refusing gpu-train:\n  " + "\n  ".join(why))
    return {"config": str(Path(path).relative_to(ROOT) if Path(path).is_absolute() else Path(path)),
            "config_sha256": digest, "run_name": cfg.run.name, "backend": cfg.execution.backend,
            "dtype": cfg.model.dtype, "streaming": cfg.streaming.enabled, "train_samples": cfg.data.train_samples,
            "max_steps": cfg.training.max_steps, "seed": cfg.training.seed, "subset_seed": cfg.data.subset_seed,
            "evaluation_every": cfg.evaluation.every, "max_hours": max_hours}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resume_target(run_dir, streaming):
    """(argument for `run.py train --resume`, step) of a downloaded run, or (None, why). Streaming runs: the
    highest-step checkpoint whose COMMIT.json validates (the kernel passes `latest`, which picks the same one); other
    runs: the highest step that saved its optimizer state."""
    sys.path.insert(0, str(ROOT))
    from qwen_grpo.checkpointing import STATE
    steps = sorted((int(p.name[4:]), p) for p in (Path(run_dir) / "checkpoints").glob("step*") if p.name[4:].isdigit())
    if streaming:
        from qwen_grpo.streaming.resume import validate
        ok = [(s, p) for s, p in steps if s > 0 and validate(p, resumable=True)[0]]
        return ("latest", ok[-1][0]) if ok else (None, "no valid resumable checkpoint")
    ok = [(s, p) for s, p in steps if s > 0 and (p / STATE).is_file()]
    return (f"runs/{Path(run_dir).name}/checkpoints/step{ok[-1][0]}", ok[-1][0]) if ok else \
        (None, "no checkpoint with optimizer state")


def pack_run(run_dir, out):
    """out/run.tar (the run directory, as runs/<name>/...) and out/qwen_grpo_resume.json: every file's sha256, which
    the kernel checks after extraction. Returns the marker's content."""
    run_dir, out = Path(run_dir), Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files = {str(p.relative_to(run_dir)): sha256_file(p) for p in sorted(run_dir.rglob("*")) if p.is_file()}
    with tarfile.open(out / "run.tar", "w") as t:
        for rel in files:
            t.add(run_dir / rel, arcname=f"runs/{run_dir.name}/{rel}")
    marker = {"run_name": run_dir.name, "files": files, "tar_sha256": sha256_file(out / "run.tar")}
    (out / RESUME_MARKER).write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    return marker


def downloaded_run(sub):
    """The run directory a gpu-train submission's downloaded output holds, or None."""
    base = OUTPUTS / "gpu-train" / sub["submission_id"]
    hits = sorted(base.rglob(f"evidence/artifacts/runs/{sub['train']['run_name']}/manifest.json"))
    return hits[0].parent if hits else None


def resume_spec(a, spec, sha):
    """Checks and packs the run to resume (nothing is uploaded here)."""
    prev = latest("gpu-train", a.resume_from)
    if prev["train"]["config_sha256"] != spec["config_sha256"]:
        sys.exit(f"--resume-from {a.resume_from}: it trained config sha256 {prev['train']['config_sha256'][:12]}, this "
                 f"submission {spec['config_sha256'][:12]}; a resume needs the same config")
    if prev["commit"] != sha and not a.force:
        sys.exit(f"--resume-from {a.resume_from} ran commit {prev['commit'][:12]}, this one {sha[:12]}: exact resume needs "
                 "the same code (--force to resume under another commit; it is recorded)")
    run_dir = downloaded_run(prev)
    if run_dir is None:
        sys.exit(f"no downloaded run for submission {a.resume_from}: python scripts/kaggle_submit.py download --mode "
                 f"gpu-train --submission {a.resume_from}")
    status = json.loads((run_dir / "manifest.json").read_text()).get("status")
    if status == "completed":
        sys.exit(f"run {run_dir.name} of submission {a.resume_from} is completed: nothing to resume")
    target, step = resume_target(run_dir, spec["streaming"])
    if target is None:
        sys.exit(f"run {run_dir.name} of submission {a.resume_from} cannot be resumed: {step}")
    marker = pack_run(run_dir, BUILD / "gpu-train-resume")
    slug = re.sub(r"[^a-z0-9-]+", "-", f"qwen-grpo-resume-{spec['run_name']}".lower())[:50].strip("-")
    return {"from_submission": a.resume_from, "from_status": status, "from_step": step, "target": target,
            "dataset": f"{username()}/{slug}", "marker_sha256": hashlib.sha256(
                (json.dumps(marker, indent=1, sort_keys=True) + "\n").encode()).hexdigest(),
            "n_files": len(marker["files"]), "commit_changed": prev["commit"] != sha}


def upload_resume(resume):
    """Create or version the private dataset holding the packed run, and wait until Kaggle has processed it."""
    folder = BUILD / "gpu-train-resume"
    slug = resume["dataset"].split("/")[1]
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug[:50], "id": resume["dataset"], "licenses": [{"name": "CC0-1.0"}]}, indent=1) + "\n")
    exists = subprocess.run(["kaggle", "datasets", "status", resume["dataset"]], capture_output=True, text=True)
    if exists.returncode == 0:
        print(kaggle("datasets", "version", "-p", str(folder), "-m",
                     f"resume of submission {resume['from_submission']} at step {resume['from_step']}").strip())
    else:
        print(kaggle("datasets", "create", "-p", str(folder)).strip())  # private: no --public
    for _ in range(60):
        st = subprocess.run(["kaggle", "datasets", "status", resume["dataset"]], capture_output=True, text=True).stdout
        if "ready" in st.lower():
            return
        time.sleep(10)
    sys.exit(f"dataset {resume['dataset']} not ready after 10 min; check `kaggle datasets status {resume['dataset']}` "
             "and submit again (the kernel verifies the files, so a stale version is refused, never used)")


def kernel_status(kid):
    r = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True)
    if r.returncode != 0:
        return None  # the kernel does not exist yet
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?([A-Za-z_]+)', r.stdout)
    return (m.group(1).lower() if m else r.stdout.strip())


def render(mode, sub):
    gpu, _, _ = MODES[mode]
    out = BUILD / mode
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    src = (KAGGLE / "run.py").read_text()
    assert src.count(PLACEHOLDER) == 1, "kaggle/run.py: the SUBMISSION placeholder line is missing"
    (out / "run.py").write_text(src.replace(PLACEHOLDER, "SUBMISSION = " + json.dumps(sub, indent=1)))
    meta = json.loads((KAGGLE / "kernel-metadata.json").read_text()
                      .replace("{username}", sub["kernel_id"].split("/")[0]).replace("{mode}", mode))
    meta["enable_gpu"] = gpu
    if (sub.get("resume") or {}).get("dataset"):
        meta["dataset_sources"] = [sub["resume"]["dataset"]]  # the packed run, mounted under /kaggle/input/
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    validate(out, sub)
    return out


def validate(folder, sub):
    """The rendered kernel: exactly run.py + metadata, the pinned commit inside, ids consistent, internet on."""
    files = sorted(p.name for p in folder.iterdir())
    if files != ["kernel-metadata.json", "run.py"]:
        sys.exit(f"refusing to upload unexpected files {files}")
    meta = json.loads((folder / "kernel-metadata.json").read_text())
    required = {"id", "title", "code_file", "language", "kernel_type", "is_private", "enable_gpu", "enable_internet"}
    if missing := required - set(meta):
        sys.exit(f"kernel-metadata.json misses {sorted(missing)}")
    if meta["id"] != sub["kernel_id"] or meta["id"].split("/")[1] != meta["title"] or meta["code_file"] != "run.py":
        sys.exit(f"kernel metadata inconsistent: {meta}")
    if not (meta["enable_internet"] and meta["is_private"] and meta["kernel_type"] == "script"):
        sys.exit("the runner needs a private script kernel with internet (it clones the repository)")
    code = (folder / "run.py").read_text()
    if PLACEHOLDER in code or f'"commit": "{sub["commit"]}"' not in code:
        sys.exit("rendered run.py does not carry the pinned commit")
    compile(code, "run.py", "exec")


def submit(a):
    gpu, accel, timeout = MODES[a.mode]
    sha = git("rev-parse", a.commit)
    url, slug = repo_url()
    if not commit_on_github(slug, sha):
        msg = (f"commit {sha[:12]} is not on GitHub ({slug}); Kaggle clones it from there. Push it first or pass "
               "--commit with a pushed commit")
        if not (a.dry_run and a.mode == "gpu-train"):
            sys.exit(msg)
        print(f"WARNING (dry run only): {msg}")
    kid = kernel_id(a.mode)
    st = kernel_status(kid)
    if st in ("queued", "running"):
        sys.exit(f"{kid} is {st}: wait for it (status/logs) instead of submitting again")
    extra = {}
    if a.mode == "gpu-train":
        if not a.config:
            sys.exit("gpu-train needs --config <yaml under configs/>")
        extra["train"] = train_spec(sha, a.config, a.max_hours)
        prior = [r for r in ledger() if r["mode"] == "gpu-train"]
        if prior and not (OUTPUTS / "gpu-train" / prior[-1]["submission_id"]).exists() and not a.force:
            sys.exit(f"the previous gpu-train submission {prior[-1]['submission_id']} has not been downloaded: download it "
                     "first (its checkpoints are needed to resume), or --force")
        if a.resume_from:
            extra["resume"] = resume_spec(a, extra["train"], sha)
        key = lambda r: (r["commit"], (r.get("train") or {}).get("config_sha256"), (r.get("resume") or {}).get("from_submission"))
        done = [r for r in prior if key(r) == (sha, extra["train"]["config_sha256"], a.resume_from)]
    else:
        done = [r for r in ledger() if r["mode"] == a.mode and r["commit"] == sha and r.get("scope") == a.scope]
    if done and not a.force:
        sys.exit(f"{a.mode} (scope {a.scope}) already submitted for {sha[:12]} at {done[-1]['submitted_utc']} "
                 f"(submission {done[-1]['submission_id']}); --force to run it again")
    sub = {"submission_id": str(uuid.uuid4()), "kernel_id": kid, "mode": a.mode, "commit": sha, "repo_url": url,
           "cpu_scope": a.scope, "step_timeout_s": timeout,
           "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **extra}
    folder = render(a.mode, sub)
    print(f"rendered {folder.relative_to(ROOT)}: kernel {kid}, mode {a.mode}, commit {sha[:12]}, "
          f"{'GPU ' + accel if gpu else 'CPU'}, timeout {timeout} s")
    if "train" in sub:
        t = sub["train"]
        print(f"  train {t['config']} (sha256 {t['config_sha256'][:12]}): run {t['run_name']}, {t['backend']}, {t['dtype']}, "
              f"{t['train_samples']} prompts, {t['max_steps']} steps, eval every {t['evaluation_every']}, seed {t['seed']}, "
              f"streaming {t['streaming']}, budget {t['max_hours']} h")
    if "resume" in sub:
        r = sub["resume"]
        print(f"  resume from submission {r['from_submission']} ({r['from_status']}, step {r['from_step']}, --resume "
              f"{r['target']}): {r['n_files']} files packed in {(BUILD / 'gpu-train-resume').relative_to(ROOT)}, "
              f"private dataset {r['dataset']}")
    if a.dry_run:
        print("dry run: nothing uploaded or pushed")
        return
    if "resume" in sub:
        upload_resume(sub["resume"])
    args = ["kernels", "push", "-p", str(folder), "-t", str(timeout)] + (["--accelerator", accel] if gpu else [])
    out = kaggle(*args)
    print(out.strip())
    version = re.search(r"[Kk]ernel version (\d+)", out)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({**sub, "scope": a.scope, "kernel_version": int(version.group(1)) if version else None,
                            "push_output": out.strip()[-500:]}) + "\n")
    print(f"recorded in {LEDGER.relative_to(ROOT)}; next: python scripts/kaggle_submit.py status --mode {a.mode}")


def download(a):
    sub = latest(a.mode, a.submission)
    out = OUTPUTS / a.mode / sub["submission_id"]
    out.mkdir(parents=True, exist_ok=True)
    kid = sub["kernel_id"]
    if a.mode == "gpu-train":  # this submission's own version, not whatever version ran last
        if not sub.get("kernel_version"):
            sys.exit(f"submission {sub['submission_id']} has no recorded kernel version")
        kid = f"{kid}/{sub['kernel_version']}"
    print(kaggle("kernels", "output", kid, "-p", str(out), "-o").strip())
    print(f"downloaded to {out.relative_to(ROOT)}")
    return out


def summary(a):
    sub = latest(a.mode, a.submission)
    out = OUTPUTS / a.mode / sub["submission_id"]
    res_path = next(iter(sorted(out.rglob("evidence/results.json"))), None)
    if res_path is None:
        sys.exit(f"no evidence/results.json under {out.relative_to(ROOT)}: run download first (or the run produced none)")
    res = json.loads(res_path.read_text())
    ev = res["submission"]
    commit_file = res_path.parent / "git_commit.txt"
    checks = {"submission id": ev["submission_id"] == sub["submission_id"], "kernel": ev["kernel_id"] == sub["kernel_id"],
              "mode": ev["mode"] == sub["mode"], "commit (submitted)": ev["commit"] == sub["commit"],
              "commit (checked out)": commit_file.exists() and commit_file.read_text().strip() == sub["commit"]}
    for k, ok in checks.items():
        print(f"attribution {k}: {'ok' if ok else 'MISMATCH'}")
    print(f"\n{sub['kernel_id']}  mode {sub['mode']}  commit {sub['commit'][:12]}  {res['utc_start']} -> {res['utc_end']}")
    for r in res["steps"]:
        c = r.get("counts")
        print(f"  {r['status']:8} {r['name']}" + (f"  passed {c['passed']}, failed {c['failures']}, errors {c['errors']}, "
                                                  f"skipped {c['skipped']} of {c['tests']}" if c else "")
              + (f"  [{r['reason']}]" if r.get("reason") else ""))
        for s in (c or {}).get("skipped_tests", [])[:50]:
            print(f"           skipped {s['test']}: {s['reason']}")
    if a.mode == "gpu-train":
        train_checks(sub, res, res_path.parent, checks)
    print(f"OVERALL {res['overall']}")
    passing = ("PASS",) if a.mode == "gpu-train" else ("PASS", "PASS_WITH_SKIPS")  # a training run has no skippable part
    ok = all(checks.values()) and res["overall"] in passing
    sys.exit(0 if ok else 1)


def train_checks(sub, res, evidence, checks):
    """gpu-train attribution (config, run, resume chain) and the performance summary."""
    t, ev = sub["train"], res["submission"]
    more = {"config sha256 (submitted)": (ev.get("train") or {}).get("config_sha256") == t["config_sha256"],
            "config sha256 (resolved on Kaggle)": res.get("resolved_config_sha256") == t["config_sha256"],
            "resume (submitted)": ev.get("resume") == sub.get("resume")}
    for k, ok in more.items():
        print(f"attribution {k}: {'ok' if ok else 'MISMATCH'}")
    checks.update(more)
    perf = evidence / "performance_summary.json"
    if perf.exists():
        p = json.loads(perf.read_text())
        print(f"\nrun {p.get('run')}: status {p.get('status')}, steps {p.get('steps')}/{p.get('max_steps')}, "
              f"{p.get('seconds_per_step')} s/step")
        for k in ("timings_total_s", "weight_bytes_copied_total", "allocator_gib", "nvidia_smi_mib", "host", "rewards",
                  "evaluations", "final_validation", "resume"):
            if p.get(k) is not None:
                print(f"  {k}: {json.dumps(p[k])[:400]}")
    if res["overall"] == "INCOMPLETE":
        print(f"\nINCOMPLETE: continue it with\n  python scripts/kaggle_submit.py submit --mode gpu-train --config "
              f"{t['config']} --commit {sub['commit']} --resume-from {sub['submission_id']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["submit", "status", "logs", "download", "summary"])
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--scope", choices=["quick", "full"], default="quick", help="cpu mode: test subset or whole suite")
    ap.add_argument("--commit", default="HEAD", help="commit to test (must be on GitHub); default HEAD")
    ap.add_argument("--force", action="store_true", help="submit although this mode and commit already ran")
    ap.add_argument("--dry-run", action="store_true", help="render and validate the kernel without pushing")
    ap.add_argument("--config", help="gpu-train: the experiment YAML (under configs/, committed at --commit)")
    ap.add_argument("--resume-from", help="gpu-train: continue the run of this (downloaded) submission")
    ap.add_argument("--max-hours", type=float, default=11.0, help="gpu-train: training time budget (default 11)")
    ap.add_argument("--submission", help="status / logs / download / summary of this submission (default: the latest)")
    a = ap.parse_args(argv)
    if a.command == "submit":
        return submit(a)
    if a.command == "status":
        kid = latest(a.mode, a.submission)["kernel_id"]
        print(f"{kid}: {kernel_status(kid)}")
        return
    if a.command == "logs":
        return sys.exit(kaggle("kernels", "logs", latest(a.mode, a.submission)["kernel_id"], capture=False))
    if a.command == "download":
        return download(a)
    return summary(a)


if __name__ == "__main__":
    main()
