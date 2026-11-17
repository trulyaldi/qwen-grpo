"""Submit qwen-grpo test runs to Kaggle and fetch their evidence, with the official Kaggle CLI (kaggle >= 2.2).

    python scripts/kaggle_submit.py submit   --mode cpu [--scope quick|full] [--commit SHA] [--force]
    python scripts/kaggle_submit.py submit   --mode gpu-tests
    python scripts/kaggle_submit.py submit   --mode gpu-short
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
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
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
}
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


def latest(mode):
    rows = [r for r in ledger() if r["mode"] == mode]
    if not rows:
        sys.exit(f"no {mode} submission in {LEDGER.relative_to(ROOT)}")
    return rows[-1]


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
        sys.exit(f"commit {sha[:12]} is not on GitHub ({slug}); Kaggle clones it from there. Push it first or pass "
                 "--commit with a pushed commit")
    kid = kernel_id(a.mode)
    st = kernel_status(kid)
    if st in ("queued", "running"):
        sys.exit(f"{kid} is {st}: wait for it (status/logs) instead of submitting again")
    done = [r for r in ledger() if r["mode"] == a.mode and r["commit"] == sha and r.get("scope") == a.scope]
    if done and not a.force:
        sys.exit(f"{a.mode} (scope {a.scope}) already submitted for {sha[:12]} at {done[-1]['submitted_utc']} "
                 f"(submission {done[-1]['submission_id']}); --force to run it again")
    sub = {"submission_id": str(uuid.uuid4()), "kernel_id": kid, "mode": a.mode, "commit": sha, "repo_url": url,
           "cpu_scope": a.scope, "step_timeout_s": timeout,
           "submitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    folder = render(a.mode, sub)
    print(f"rendered {folder.relative_to(ROOT)}: kernel {kid}, mode {a.mode}, commit {sha[:12]}, "
          f"{'GPU ' + accel if gpu else 'CPU'}, timeout {timeout} s")
    if a.dry_run:
        print("dry run: nothing pushed")
        return
    args = ["kernels", "push", "-p", str(folder), "-t", str(timeout)] + (["--accelerator", accel] if gpu else [])
    out = kaggle(*args)
    print(out.strip())
    with open(LEDGER, "a") as f:
        f.write(json.dumps({**sub, "scope": a.scope, "push_output": out.strip()[-500:]}) + "\n")
    print(f"recorded in {LEDGER.relative_to(ROOT)}; next: python scripts/kaggle_submit.py status --mode {a.mode}")


def download(a):
    sub = latest(a.mode)
    out = OUTPUTS / a.mode / sub["submission_id"]
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", sub["kernel_id"], "-p", str(out), "-o").strip())
    print(f"downloaded to {out.relative_to(ROOT)}")
    return out


def summary(a):
    sub = latest(a.mode)
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
    print(f"OVERALL {res['overall']}")
    ok = all(checks.values()) and res["overall"] in ("PASS", "PASS_WITH_SKIPS")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["submit", "status", "logs", "download", "summary"])
    ap.add_argument("--mode", required=True, choices=list(MODES))
    ap.add_argument("--scope", choices=["quick", "full"], default="quick", help="cpu mode: test subset or whole suite")
    ap.add_argument("--commit", default="HEAD", help="commit to test (must be on GitHub); default HEAD")
    ap.add_argument("--force", action="store_true", help="submit although this mode and commit already ran")
    ap.add_argument("--dry-run", action="store_true", help="render and validate the kernel without pushing")
    a = ap.parse_args(argv)
    if a.command == "submit":
        return submit(a)
    if a.command == "status":
        kid = latest(a.mode)["kernel_id"]
        print(f"{kid}: {kernel_status(kid)}")
        return
    if a.command == "logs":
        return sys.exit(kaggle("kernels", "logs", latest(a.mode)["kernel_id"], capture=False))
    if a.command == "download":
        return download(a)
    return summary(a)


if __name__ == "__main__":
    main()
