"""The `qwen-grpo` command line, preflight logic, the Soup wrapper and the final report's evidence freshness.

Tier 1 (fast, CPU, no weights, no network): outputs are redirected to tmp_path (the session guard forbids writes to
runs/), external processes are fakes."""
import json
import stat
from pathlib import Path

import pytest

from qwen_grpo import cli, config
from qwen_grpo.diagnostics import preflight as P
from qwen_grpo.integrations import soup as S

ROOT = Path(__file__).parents[1]
EXP = ROOT / "configs" / "experiments"


@pytest.fixture(autouse=True)
def quiet_history(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "history", lambda record: None)
    monkeypatch.setattr(P, "OUT", tmp_path / "preflight")


def test_help_and_usage_errors_have_distinct_exit_codes(capsys):
    assert cli.main(["--help"]) == cli.OK
    assert cli.main(["train"]) == cli.USAGE  # --config missing
    assert cli.main(["no-such-command"]) == cli.USAGE


def test_research_run_is_a_dry_run_unless_executed(monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("executed without --execute"))
    assert cli.main(["research", "run", "H06"]) == cli.OK
    out = capsys.readouterr().out
    assert "[dry run] t08_prefetch_timing" in out and "scripts/l2l_lab.py run --name t08_prefetch_timing" in out
    assert cli.main(["research", "run", "H08"]) == cli.OK
    assert "skip t09_resume_tiny_cpu" in capsys.readouterr().out  # existing evidence is never overwritten


def test_every_runbook_entry_names_a_known_case_or_script():
    book, hyp = cli.runbook()
    assert set(book) == {h for h in hyp if h.startswith("H")}
    from qwen_grpo import lab
    for h, entries in book.items():
        for e in entries:
            assert (ROOT / e["argv"][0]).exists() and e["device"] in ("cpu", "cuda"), e["name"]
            if "--case" in e["argv"]:
                assert e["argv"][e["argv"].index("--case") + 1] in lab.CASES, e["name"]
            if "--config" in e["argv"]:
                assert (ROOT / e["argv"][e["argv"].index("--config") + 1]).exists(), e["name"]


def fake_report(statuses, sig="sig", mode="offline", overall=None):
    checks = [P.check(cid, "x", s, "") for cid, s in statuses.items()]
    return {"checks": checks, "memory_signature": sig, "mode": mode, "created_utc": "20260101T000000Z", "run": "r",
            "overall": overall or P.overall(checks)}


def test_the_train_gate(monkeypatch):
    smoke = config.load(EXP / "qwen3_4b_smoke.yaml")
    local = config.load(EXP / "qwen_local.yaml")
    sig = P.memory_signature(smoke)
    monkeypatch.setattr(P, "run", lambda path, **k: fake_report({"config.valid": P.PASS, "gpu.measured": P.NOT_TESTED}, sig))
    assert P.gate("x", local)["ok"]  # unguarded: BLOCKED / NOT TESTED do not stop a run
    assert not P.gate("x", smoke)["ok"]  # guarded: needs a passing GPU preflight
    assert P.gate("x", smoke, skip=True)["ok"] and "WARNING" in P.gate("x", smoke, skip=True)["message"]
    gpu = P.OUT / "r"
    gpu.mkdir(parents=True, exist_ok=True)  # gate() already saved its offline record here
    (gpu / "a_gpu.json").write_text(json.dumps(fake_report({"gpu.measured": P.FAIL}, sig, "gpu")))
    assert not P.gate("x", smoke)["ok"]  # a failed GPU preflight does not count
    (gpu / "b_gpu.json").write_text(json.dumps(fake_report({"gpu.measured": P.PASS}, "other", "gpu")))
    assert not P.gate("x", smoke)["ok"]  # nor one of a config with other memory settings
    (gpu / "c_gpu.json").write_text(json.dumps(fake_report({"gpu.measured": P.PASS}, sig, "gpu")))
    assert P.gate("x", smoke)["ok"]
    monkeypatch.setattr(P, "run", lambda path, **k: fake_report({"config.implemented": P.FAIL}, sig))
    assert not P.gate("x", local)["ok"]  # any FAIL refuses


def test_the_smoke_run_vouches_for_the_final_runs_memory_but_dflash_does_not():
    sig = {n: P.memory_signature(config.load(EXP / f"{n}.yaml"))
           for n in ("qwen3_4b_smoke", "qwen3_4b_final", "qwen3_4b_final_lr0", "qwen3_4b_smoke_dflash")}
    assert sig["qwen3_4b_smoke"] == sig["qwen3_4b_final"] == sig["qwen3_4b_final_lr0"] != sig["qwen3_4b_smoke_dflash"]


def test_an_estimate_is_never_reported_as_a_measured_pass():
    cfg = config.load(ROOT / "configs" / "profiles" / "qwen3_4b.yaml")
    est = {"gpu_peak_device_bytes": 2 * P.GiB, "gpu_verdict": "PASS", "cpu_peak_bytes_range": [10 * P.GiB, 20 * P.GiB],
           "pinned_bytes": 8 * P.GiB}
    checks = {c["id"]: c for c in P.check_estimates(cfg, est, host_ram_gib=16.0)}
    assert checks["gpu.estimate"]["status"] == P.PASS and "never evidence" in checks["gpu.estimate"]["detail"]
    assert checks["cpu.estimate"]["status"] == P.NOT_TESTED  # 10..20 GiB on a 16 GiB host: depends on the loader
    assert P.check_estimates(cfg, {**est, "gpu_verdict": "UNKNOWN"}, 64.0)[0]["status"] == P.NOT_TESTED
    assert P.check_estimates(cfg, {**est, "cpu_peak_bytes_range": [17 * P.GiB, 20 * P.GiB]}, 16.0)[1]["status"] == P.FAIL
    assert P.overall([P.check("a", "x", P.PASS, ""), P.check("gpu.measured", "x", P.NOT_TESTED, "")]) == P.NOT_TESTED


class Arch:
    num_hidden_layers = 36


@pytest.mark.parametrize("rel", ["profiles/qwen3_0_6b.yaml", "profiles/qwen3_4b.yaml",
                                 "experiments/qwen3_4b_smoke_dflash.yaml", "experiments/qwen_local.yaml"])
def test_training_configs_map_onto_the_budget_calculator(rel):
    from qwen_grpo.budget import calculator as C
    cfg = config.load(ROOT / "configs" / rel)
    b = C.validate(P.budget_config(cfg, Arch(), 100, 16.0, 12.0, "model"))
    s, g = b["streaming"], b["grpo"]
    assert b["model"]["id"] == cfg.model.name and b["hardware"]["gpu_vram_gib"] == cfg.run.gpu_budget_gib
    assert s["schedule"] == "microbatch_major"  # what training.py executes
    assert g["micro_batch_size"] == cfg.training.batch_size * cfg.training.group_size  # null: one chunk
    assert s["resident_decoder_layers"] == (36 if cfg.execution.backend == "resident" else 0)
    assert b["dflash"]["estimate"] == cfg.dflash.enabled and not b["dflash"]["enabled"]


def test_offline_preflight_makes_no_network_request(monkeypatch):
    import huggingface_hub as hh
    calls = []
    monkeypatch.setattr(hh, "get_safetensors_metadata", lambda *a, **k: calls.append(a) or pytest.fail("network"))
    with P.hub(allow_network=False):
        with pytest.raises(P.Offline):
            hh.get_safetensors_metadata("Qwen/Qwen3-4B")
        with pytest.raises(P.Offline):
            hh.snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    assert calls == []


def test_offline_preflight_of_the_reference_profile():
    rep = P.run(ROOT / "configs" / "profiles" / "qwen3_0_6b.yaml")
    st = {c["id"]: c["status"] for c in rep["checks"]}
    assert st["config.valid"] == st["config.implemented"] == st["data.splits"] == st["reward.contract"] == P.PASS
    assert st["gpu.measured"] == P.NOT_TESTED and rep["overall"] != P.PASS  # never PASS without measuring
    assert rep["memory_signature"] == P.memory_signature(config.load(ROOT / "configs" / "profiles" / "qwen3_0_6b.yaml"))


def test_an_invalid_config_fails_preflight(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x}\n")
    rep = P.run(bad)
    assert rep["overall"] == P.FAIL and rep["checks"][0]["id"] == "config.valid"


# --- Soup wrapper (a fake executable: the wrapper's contract, not Soup's behaviour) -------------------------------

FAKE = """#!/bin/sh
# fake soup: echo the arguments; write {"fake": true} to the path after -o / --output-report; exit $FAKE_EXIT
echo "args: $@ <REPO>"
prev=""
for a in "$@"; do
  if [ "$prev" = "-o" ] || [ "$prev" = "--output-report" ]; then echo '{"fake": true}' > "$a"; fi
  prev="$a"
done
exit ${FAKE_EXIT:-0}
"""


@pytest.fixture
def fake_soup(tmp_path):
    exe = tmp_path / "soup"
    exe.write_text(FAKE.replace("<REPO>", str(ROOT)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return str(exe)


def test_soup_calls_are_recorded_raw_with_paths_scrubbed(fake_soup, monkeypatch):
    out = ROOT / "runs" / "_soup_test_unused"
    monkeypatch.setattr(S, "EVIDENCE", out)
    import tempfile
    with tempfile.TemporaryDirectory(dir=ROOT / "tests") as tmp:  # Soup requires outputs under the cwd (repo root)
        tmp = Path(tmp)
        monkeypatch.setenv("FAKE_EXIT", "2")
        rec = S.ship(fake_soup, tmp, S.ship_evidence(0.5, 0.6))
        assert rec["exit"] == 2 and rec["verdict"] == "DON'T SHIP" and rec["report"] == {"fake": True}
        assert str(ROOT) not in rec["stdout"] and "<repo>" in rec["stdout"]
        assert rec["argv"][:3] == ["soup", "--no-telemetry", "ship"]
        assert (tmp / "ship.log").read_text().startswith("$ soup --no-telemetry ship --evidence ")
        monkeypatch.setenv("FAKE_EXIT", "0")
        assert S.reward_stress(fake_soup, tmp, n=4)["verdict"] == "robust"
        monkeypatch.setenv("FAKE_EXIT", "7")
        assert S.reward_stress(fake_soup, tmp, n=4)["verdict"] == "error"  # never mistaken for robust
    assert not out.exists()


def test_missing_soup_is_reported_unavailable_not_passed(monkeypatch, capsys):
    monkeypatch.delenv("SOUP_BIN", raising=False)
    monkeypatch.setattr(S.shutil, "which", lambda name: None)
    assert S.find() is None
    assert S.ship_from_final_results({"held_out": {"models": {"base": {"accuracy": 0.5}, "trained": {"accuracy": 0.6}}}}
                                     )["status"] == S.UNAVAILABLE
    assert cli.main(["soup", "all"]) == cli.CHECK_FAILED
    assert "UNAVAILABLE" in capsys.readouterr().out


def test_soup_ship_needs_held_out_scores(fake_soup):
    assert S.ship_from_final_results({"held_out": None}, fake_soup)["status"] == S.NOT_RUN


# --- final report -------------------------------------------------------------------------------------------------

def test_the_final_report_records_how_old_its_evidence_is(monkeypatch, tmp_path):
    rep = cli.script("final_report")
    monkeypatch.setattr(rep, "OUT_JSON", tmp_path / "final_results.json")
    monkeypatch.setattr(rep, "OUT_MD", tmp_path / "final_report.md")
    rep.main()
    res = json.loads((tmp_path / "final_results.json").read_text())
    assert res["verdict"] == "DON'T SHIP"  # GPU evidence pending
    fresh = {f["evidence"]: f for f in res["evidence_freshness"]}
    assert fresh["t03_forward_equivalence"]["commit"].startswith("c4b391b")
    assert "qwen_grpo/execution/layer_stream.py" in fresh["t03_forward_equivalence"]["engine_files_changed_since"]
    assert "## Evidence freshness" in (tmp_path / "final_report.md").read_text()
