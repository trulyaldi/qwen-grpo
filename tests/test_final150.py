"""The 150-update experiment: its configs, the exact byte budget, hash stability of every earlier config, the
driver's dry run and gates (scripts/final150.py), and what the trainer records per evaluation, update and zero-signal
attempt (tiny CPU model)."""
import json

import pytest
import torch

from qwen_grpo import config, runs, trainer
from qwen_grpo.config import budget_bytes
from scripts import final150
from tests.test_streaming import content_rewards
from tests.test_trainer import make_cfg, tiny_lora_model

ROOT = final150.ROOT


def load(name):
    return config.load(ROOT / "configs" / "experiments" / f"{name}.yaml")


def test_the_150_update_config_is_exactly_the_protocol():
    ok, rows, _ = final150.config_checks(final150.CFG, final150.EXPECTED)
    assert ok, [r for r in rows if not r["ok"]]
    c = load("qwen3_4b_final_plain_150_v2")
    assert c.training.max_steps == 150 and c.evaluation.every == 10 and c.memory.budget_bytes == 8_000_000_000
    assert budget_bytes(c) == 8_000_000_000 != int(c.run.gpu_budget_gib * 2**30)  # bytes, not 8 GiB
    assert c.run.overwrite is False and c.evaluation.subset_label_scores and not c.dflash.enabled
    assert c.training.max_attempts == 1200 and c.data.val_subset == 128


def test_only_the_intended_keys_differ_from_the_60_update_run():
    old, new = config.as_dict(load("qwen3_4b_final_plain")), config.as_dict(load("qwen3_4b_final_plain_150_v2"))
    assert sorted(config.differences(old, new)) == sorted([
        "run.name", "training.max_steps", "training.max_attempts", "evaluation.every", "evaluation.subset_label_scores",
        "memory.budget_bytes"])
    lr0 = config.as_dict(load("qwen3_4b_final_plain_150_v2_lr0"))
    assert sorted(config.differences(new, lr0)) == ["run.name", "training.learning_rate", "training.max_steps"]


def test_run_names_are_new_and_distinct():
    names = {load(n).run.name for n in ("qwen3_4b_final_plain_150_v2", "qwen3_4b_final_plain_150_v2_lr0",
                                        "qwen3_4b_final_plain_150_v2_smoke", "qwen3_4b_resume_kill")}
    assert len(names) == 4 and not names & {"qwen3_4b_final_plain", "qwen3_4b_final_plain_lr0", "qwen3_4b_final"}


def test_historical_config_hashes_are_unchanged_by_the_new_fields():
    """The later fields (memory.budget_bytes, evaluation.subset_label_scores) are left out of the hash at default."""
    for name in ("qwen3_4b_final_plain", "qwen3_4b_final_plain_lr0"):
        m = runs.read_json(ROOT / "runs" / name / "manifest.json") if (ROOT / "runs" / name / "manifest.json").exists() \
            else None
        if m is None:
            pytest.skip("historical run not on this machine")
        saved = config.load(ROOT / "runs" / name / "config.yaml", historical=True)
        assert config.sha256(saved) == m["config_sha256"]


def test_an_exact_budget_requires_the_guard(tmp_path):
    raw = config.read(ROOT / "configs" / "experiments" / "qwen3_4b_final_plain_150_v2.yaml")
    raw["memory"]["guard"], raw["memory"]["allocator_cap"] = False, False
    p = tmp_path / "x.yaml"
    import yaml
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="budget_bytes needs memory.guard"):
        config.load(p)


def test_dry_run_checks_everything_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(final150, "RESULTS", tmp_path / "results")
    assert final150.main(["dry-run"]) == 0
    out = capsys.readouterr().out
    assert "8000000000 bytes" in out and "[0, 10, 20" in out and "150]" in out and "dflash       NOT_RUN" in out
    assert not (tmp_path / "results").exists()


def test_gates_block_training_until_every_prerequisite_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(final150, "RESULTS", tmp_path)
    state = {"stages": {"preflight": [{"status": "PASS", "dir": "x"}], "correctness": [{"status": "FAIL", "dir": "y"}]},
             "waivers": {}}
    final150.save_state(state)
    assert final150.gate_problems(final150.load_state(), "train") == ["correctness: FAIL", "smoke: not run",
                                                                       "profile: not run"]
    called = []
    monkeypatch.setitem(final150.RUNNERS, "train", lambda *a: called.append(1) or [])
    assert final150.main(["train"]) == 1 and not called  # nothing ran
    assert final150.load_state()["stages"]["train"][-1]["status"] == "BLOCKED"
    for s in ("correctness", "smoke", "profile"):
        state["stages"][s] = [{"status": "PASS", "dir": s}]
    final150.save_state(state)
    assert final150.gate_problems(final150.load_state(), "train") == []


def test_a_passed_stage_is_not_rerun_and_a_failed_expensive_stage_needs_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(final150, "RESULTS", tmp_path)
    final150.save_state({"stages": {"preflight": [{"status": "PASS", "dir": "a"}],
                                    "merge": [{"status": "FAIL", "dir": "b"}], "train": [{"status": "PASS", "dir": "c"}]},
                         "waivers": {}})
    called = []
    monkeypatch.setitem(final150.RUNNERS, "preflight", lambda *a: called.append("p") or [])
    monkeypatch.setitem(final150.RUNNERS, "merge", lambda *a: called.append("m") or [])
    assert final150.main(["preflight"]) == 0 and final150.main(["merge"]) == 1 and not called


def test_waivers_are_explicit_and_reported_never_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(final150, "RESULTS", tmp_path)
    with pytest.raises(SystemExit):
        final150.main(["status", "--waive", "H02.fault_unsafe_reuse"])  # no reason
    final150.main(["status", "--waive", "H02.fault_unsafe_reuse", "--reason", "known lab-judge limitation"])
    assert "known lab-judge limitation" in final150.load_state()["waivers"]["H02.fault_unsafe_reuse"]


def test_item_status_never_passes_on_missing_or_failing_memory_evidence():
    assert final150.item_status({"memory_status": "PASS"}, True)[0] == "PASS"
    assert final150.item_status({"memory_status": "INCONCLUSIVE"}, True)[0] == "INCONCLUSIVE"
    assert final150.item_status({"memory_status": "FAIL"}, True)[0] == "FAIL"
    assert final150.item_status({"memory_status": "PASS"}, False)[0] == "FAIL"
    assert final150.item_status({"memory_status": None}, True)[0] == "PASS"  # uncapped CPU item


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    torch.set_default_dtype(torch.float32)
    monkeypatch.setattr(trainer, "lora_model", tiny_lora_model)
    monkeypatch.setattr(trainer, "group_rewards", content_rewards)
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    monkeypatch.delenv("QWEN_GRPO_CRASH", raising=False)
    return tmp_path


def test_trainer_records_evaluations_every_n_zero_signal_timing_advantages_and_adapter_identity(tiny, tmp_path):
    import dataclasses
    cfg = make_cfg(tmp_path, every=2, max_steps=6, max_attempts=40)
    cfg = dataclasses.replace(cfg, evaluation=dataclasses.replace(cfg.evaluation, subset_label_scores=True))
    out = trainer.train(cfg, device="cpu")
    assert [e["step"] for e in out["evaluations"]] == [0, 2, 4, 6]
    assert [t["step"] for t in out["evaluation_times"]] == [0, 2, 4, 6]
    assert all("auc" in e["metrics"] and len(e["label_scores"]["scores"]) == 2 for e in out["evaluations"])
    assert len({e["adapter_sha256"] for e in out["evaluations"]}) == 4  # step 0 differs from every trained adapter
    assert out["config"]["num_iterations"] == 1  # one update per accepted batch: attempts = updates + zero-signal
    assert len(out["zero_signal"]) > 0  # content_rewards makes some batches zero-signal
    assert len(out["zero_signal_timings"]) == len(out["zero_signal"]) == out["groups_generated"] - out["successful_steps"]
    assert all(z["rollout_s"] is not None and z["attempt_s"] >= z["rollout_s"] for z in out["zero_signal_timings"])
    assert all({"adv_mean", "adv_std", "adv_abs_max", "adv_nonzero"} <= r.keys() for r in out["training"])
    assert all(abs(r["adv_mean"]) < 1e-5 for r in out["training"])  # group-normalised advantages
    ck = sorted(p.name for p in (tiny / "runs" / "tiny" / "checkpoints").glob("step*"))
    assert ck == ["step0", "step2", "step4", "step6"]
    saved = json.loads((tiny / "runs" / "tiny" / "training" / "training.json").read_text())
    assert saved["evaluations"][-1]["adapter_sha256"] == out["evaluations"][-1]["adapter_sha256"]
