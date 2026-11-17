"""The Kaggle workflow's bookkeeping (no network, no Kaggle account): overall status never treats a skip as a pass, JUnit
counts and skip reasons are read, a rendered kernel carries the pinned commit and nothing else."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = load("kaggle_runner", "kaggle/run.py")
submit = load("kaggle_submit", "scripts/kaggle_submit.py")


@pytest.mark.parametrize("steps, expected", [
    ([], "FAIL"),
    ([{"status": "PASS"}], "PASS"),
    ([{"status": "PASS"}, {"status": "SKIPPED"}], "PASS_WITH_SKIPS"),
    ([{"status": "PASS", "counts": {"skipped": 3}}], "PASS_WITH_SKIPS"),
    ([{"status": "PASS"}, {"status": "FAIL"}], "FAIL"),
    ([{"status": "SKIPPED"}, {"status": "ERROR"}], "FAIL"),
])
def test_overall_status(monkeypatch, steps, expected):
    monkeypatch.setattr(runner, "RESULTS", steps)
    assert runner.overall() == expected


def test_junit_counts_keep_skip_reasons(tmp_path):
    xml = tmp_path / "j.xml"
    xml.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="1">'
                   '<testcase classname="t" name="a"/><testcase classname="t" name="b"><failure/></testcase>'
                   '<testcase classname="t" name="c"><skipped message="hardware: T4 has no native bfloat16"/></testcase>'
                   '<testcase classname="t" name="d"/></testsuite></testsuites>')
    c = runner.junit_counts(xml)
    assert (c["passed"], c["failures"], c["skipped"]) == (2, 1, 1)
    assert c["skipped_tests"] == [{"test": "t::c", "reason": "hardware: T4 has no native bfloat16"}]
    assert runner.junit_counts(tmp_path / "missing.xml") is None


def sub(commit="a" * 40):
    return {"submission_id": "s", "kernel_id": "someone/qwen-grpo-cpu", "mode": "cpu", "commit": commit,
            "repo_url": "https://github.com/x/y.git", "cpu_scope": "quick", "step_timeout_s": 10}


def test_rendered_kernel_carries_the_pinned_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "BUILD", tmp_path)
    out = submit.render("cpu", sub())
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["id"] == "someone/qwen-grpo-cpu" and meta["title"] == "qwen-grpo-cpu" and meta["enable_gpu"] is False
    assert meta["enable_internet"] and meta["is_private"] and meta["code_file"] == "run.py"
    code = (out / "run.py").read_text()
    assert f'"commit": "{"a" * 40}"' in code and submit.PLACEHOLDER not in code
    assert json.loads(submit.render("gpu-tests", {**sub(), "kernel_id": "someone/qwen-grpo-gpu-tests",
                                                  "mode": "gpu-tests"}).joinpath("kernel-metadata.json")
                      .read_text())["enable_gpu"] is True


def test_validation_refuses_extra_files_and_a_wrong_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(submit, "BUILD", tmp_path)
    out = submit.render("cpu", sub())
    with pytest.raises(SystemExit, match="does not carry the pinned commit"):
        submit.validate(out, sub(commit="b" * 40))
    (out / ".env").write_text("SECRET=1")
    with pytest.raises(SystemExit, match="unexpected files"):
        submit.validate(out, sub())


def test_the_template_refuses_to_run():
    assert runner.SUBMISSION == {"placeholder": True}


def test_bf16_guard_skips_only_cuda_bfloat16_items(tmp_path):
    """The gpu-tests plugin, on synthetic items (the tests only record that they ran; no GPU is touched)."""
    import os
    import subprocess
    import sys
    (tmp_path / "kaggle_bf16_guard.py").write_text(runner.BF16_PLUGIN)
    (tmp_path / "test_x.py").write_text(
        "import pytest, torch\n"
        "@pytest.mark.parametrize('device', ['cpu', 'cuda'])\n"
        "@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])\n"
        "def test_grid(device, dtype):\n    pass\n"
        "@pytest.mark.parametrize('dt', ['bfloat16'])\n"
        "def test_cpu_only_bf16(dt):\n    pass\n")

    def run(reason):
        env = {**os.environ, "PYTHONPATH": str(tmp_path)}
        env.pop("KAGGLE_NO_NATIVE_BF16", None)
        if reason:
            env["KAGGLE_NO_NATIVE_BF16"] = reason
        return subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "-p",
                               "kaggle_bf16_guard", str(tmp_path / "test_x.py")], capture_output=True, text=True,
                              cwd=tmp_path, env=env).stdout

    out = run("Tesla T4 (compute capability 7.5) has no native bfloat16")
    assert "4 passed, 1 skipped" in out and "hardware: Tesla T4 (compute capability 7.5)" in out
    assert "5 passed" in run(None)
