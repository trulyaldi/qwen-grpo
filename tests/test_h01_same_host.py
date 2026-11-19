"""H01 fixture tie-back methodology (forward_lab.fixture_tie_back): L2L must equal the resident reference recomputed on
the same host bitwise (gating); the saved cross-host fixture logits are compared with the same zero bound, and when
the saved fixture comes from another CPU / stack that comparison is kept as a non-gating diagnostic (the two Vast.ai
rows of 1.34e-7 and 2.35e-7). CPU only: the prefetching executor on CPU buffers (research/l2l/configs/async_cpu.yaml)."""
import shutil

import pytest
import torch

from qwen_grpo import config, forward_lab, lab
from qwen_grpo.execution import fixtures as F



@pytest.fixture
def cfg():
    from pathlib import Path
    return config.load(Path(__file__).parents[1] / "research" / "l2l" / "configs" / "async_cpu.yaml")


def run(cfg, perturbation=None):
    result = {"kind": "correctness", "comparisons": [], "errors": [], "reference_integrity": []}
    forward_lab.fixture_tie_back(cfg, result, perturbation, lab.tolerances_for("forward_equivalence", cfg, "float32"))
    return result


def foreign_copy(tmp_path, delta):
    """The session's fixtures, saved as if computed on another CPU whose logits differ by `delta` (fp32 round-off)."""
    d = tmp_path / "foreign"
    d.mkdir()
    for shape in F.SHAPES:
        fx = torch.load(F.path(shape), weights_only=True)
        fx["meta"]["host"] = {"cpu": "Some Other CPU @ 3.0GHz", "cpu_capability": "AVX512"}
        logits = fx["reference"]["logits"]
        fx["reference"]["logits"] = logits + delta * logits.abs().max()
        torch.save(fx, d / f"{shape}.pt")
    return d


def test_same_host_rows_are_bitwise_and_gating_and_saved_rows_also_gate_on_the_fixture_host(cfg):
    if any(F.saved_environment_differences(s, F.DIR) for s in F.SHAPES):
        pytest.skip("the loaded fixtures were not built on this host")
    r = run(cfg)
    names = [c["name"] for c in r["comparisons"]]
    assert all(c["pass"] and c["max_abs"] == 0 for c in r["comparisons"]), r["comparisons"]
    for shape in F.SHAPES:
        assert f"logits.fixture.{shape}.cpu.same_host" in names and f"logits.fixture.{shape}.cpu" in names
    assert "diagnostics" not in r


def test_cross_host_drift_is_recorded_with_its_failure_but_does_not_gate(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "DIR", foreign_copy(tmp_path, 2e-7))
    r = run(cfg)
    assert [c["name"] for c in r["comparisons"]] == [f"logits.fixture.{s}.cpu.same_host" for s in F.SHAPES]
    assert all(c["pass"] and c["max_abs"] == 0 for c in r["comparisons"])
    assert len(r["diagnostics"]) == len(F.SHAPES)
    for d in r["diagnostics"]:
        assert d["gating"] is False and d["environment_differences"] and d["l2l_drift_equals_resident_drift"]
        saved = [x for x in d["rows"] if x["name"].endswith(".cpu")]
        drift = [x for x in d["rows"] if x["name"].endswith(".resident_drift")]
        assert saved and not saved[0]["pass"] and saved[0]["max_abs"] > 0  # the historical failure is kept, not hidden
        assert drift and drift[0]["max_abs"] == saved[0]["max_abs"]  # L2L drift == resident drift, bit for bit
    status, reasons = lab.decide({**r, "kind": "correctness"})
    assert status == "PASS", reasons


def test_a_wrong_layer_still_fails_the_gating_same_host_rows(tmp_path, monkeypatch):
    import dataclasses
    from pathlib import Path
    sync = config.load(Path(__file__).parents[1] / "research" / "l2l" / "configs" / "l2l_sync.yaml")
    sync = dataclasses.replace(sync, execution=dataclasses.replace(sync.execution, weight_storage="pageable"))
    monkeypatch.setattr(F, "DIR", foreign_copy(tmp_path, 0.0))
    r = run(sync, perturbation="wrong_layer=0")
    failing = [c["name"] for c in r["comparisons"] if not c["pass"]]
    assert failing and all(n.endswith(".same_host") for n in failing)
    assert lab.decide({**r, "kind": "correctness"})[0] == "FAIL"


def test_historical_results_without_the_same_host_rows_keep_their_recorded_status():
    """The Vast.ai FAIL (research/l2l/experiments/t08_forward_equivalence_async/) is re-derived unchanged."""
    import json
    from pathlib import Path
    res = Path(__file__).parents[1] / "research" / "l2l" / "experiments" / "t08_forward_equivalence_async" / "result.json"
    if not res.exists():
        pytest.skip("historical result not present")
    r = json.loads(res.read_text())
    assert r["status"] == "FAIL" and lab.decide(r)[0] == "FAIL"
    bad = [c for c in r["comparisons"] if not c["pass"]]
    assert sorted(c["name"] for c in bad) == ["logits.fixture.0.6b-like.cpu", "logits.fixture.4b-like.cpu"]
