"""Continuation of a completed streaming run to more updates as a NEW run (qwen_grpo/trainer.py continue_run), on CPU
with the tiny trainer model: the mechanism behind the opt-in qwen3_4b_final_ext210 (60 -> 210 updates).

Checked: a 2-update run continued to 4 updates equals an uninterrupted 4-update run bitwise (training rows, rollout
hashes, zero-signal batches, evaluations, adapter, AdamW state); the source run is not modified; the continuation run
records its source (run, checkpoint, config hash, file hashes); a crash inside the continuation resumes to the same
result; and every unsafe use is refused: a plain --resume with a longer max_steps (the run identity check is not
bypassed), a config differing in anything but run.name / training.max_steps, a step budget not above the checkpoint,
an existing target run, an incomplete source run, a tampered copied checkpoint."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml
from peft import load_peft_weights

from qwen_grpo import checkpointing, config, runs, trainer
from tests.test_config import RAW
from tests.test_streaming import committed, content_rewards
from tests.test_trainer import SETTINGS, same_tensors, tiny_lora_model


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "lora_model", tiny_lora_model)
    monkeypatch.setattr(trainer, "group_rewards", content_rewards)
    monkeypatch.delenv("QWEN_GRPO_CRASH", raising=False)
    trainer._CRASH_SEEN.clear()

    def use(label):
        root = tmp_path / label
        root.mkdir()
        monkeypatch.setattr(runs, "ROOT", root)
        return root
    return use


def cfg(tmp_path, name, steps, **training):
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = name
    raw["model"]["dtype"] = "float32"
    raw["data"].update(train_samples=4, val_subset=2)
    raw["training"].update({**SETTINGS, "max_steps": steps, **training})
    raw["evaluation"].update(every=2, full_validation=False)
    raw["streaming"] = {"enabled": True, "directory": "streaming"}
    path = tmp_path / f"{name}_{steps}_{hash(json.dumps(training, sort_keys=True))}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


def tree(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(path).rglob("*")) if p.is_file()}


def adapter(root, name, step):
    return load_peft_weights(str(root / "runs" / name / "checkpoints" / f"step{step}"), device="cpu")


def source_run(sandbox, tmp_path, label="B"):
    root = sandbox(label)
    trainer.train(cfg(tmp_path, "tiny", 2), device="cpu")
    return root, root / "runs" / "tiny" / "checkpoints" / "step2"


def test_a_continuation_equals_the_uninterrupted_longer_run_and_leaves_the_source_untouched(sandbox, tmp_path):
    ref_root = sandbox("A")
    ref = trainer.train(cfg(tmp_path, "tiny", 4), device="cpu")
    assert ref["successful_steps"] == 4

    root, src = source_run(sandbox, tmp_path)
    before = tree(root / "runs" / "tiny")
    out = trainer.continue_run(cfg(tmp_path, "tiny_ext", 4), src, device="cpu")
    assert tree(root / "runs" / "tiny") == before  # the 2-update artifact is read, never written

    assert committed(out) == committed(ref)
    assert same_tensors(adapter(root, "tiny_ext", 4), adapter(ref_root, "tiny", 4))
    st = checkpointing.load_training_state(root / "runs" / "tiny_ext" / "checkpoints" / "step4")
    st_ref = checkpointing.load_training_state(ref_root / "runs" / "tiny" / "checkpoints" / "step4")
    assert same_tensors(st["optimizer"], st_ref["optimizer"])
    assert out["coverage"] == ref["coverage"]

    m = runs.manifest("tiny_ext")
    rec = m["continued_from"]
    assert rec["run"] == "tiny" and rec["step"] == 2 and rec["changed"] == ["run.name", "training.max_steps"]
    assert rec["config_sha256"] == runs.manifest("tiny")["config_sha256"] != m["config_sha256"]
    assert rec["checkpoint_tree_sha256"] == rec["source_checkpoint_tree_sha256"]
    assert m["status"] == "completed" and m["id"] != runs.manifest("tiny")["id"]
    assert out["continued_from"] == rec and out["config"]["steps"] == 4
    assert out["config_sha256"] == m["config_sha256"]


def test_a_crashed_continuation_resumes_to_the_same_result(sandbox, tmp_path, monkeypatch):
    ref_root = sandbox("A")
    ref = trainer.train(cfg(tmp_path, "tiny", 4), device="cpu")
    root, src = source_run(sandbox, tmp_path)
    ext = cfg(tmp_path, "tiny_ext", 4)
    monkeypatch.setenv("QWEN_GRPO_CRASH", "update@1")  # the continuation's first update (step 3), uncommitted
    trainer._CRASH_SEEN.clear()
    with pytest.raises(trainer.InjectedCrash):
        trainer.continue_run(ext, src, device="cpu")
    monkeypatch.delenv("QWEN_GRPO_CRASH")
    with pytest.raises(SystemExit, match="already exists"):  # set up once; afterwards it is resumed
        trainer.continue_run(ext, src, device="cpu")
    out = trainer.train(ext, resume="latest", device="cpu")  # latest valid: the copied step2
    assert committed(out) == committed(ref)
    assert same_tensors(adapter(root, "tiny_ext", 4), adapter(ref_root, "tiny", 4))


def test_a_longer_max_steps_is_never_resumed_in_place(sandbox, tmp_path):
    root, src = source_run(sandbox, tmp_path)
    with pytest.raises(SystemExit, match="different config"):
        trainer.train(cfg(tmp_path, "tiny", 4), resume=src, device="cpu")


@pytest.mark.parametrize("change, message", [
    ({"learning_rate": 1e-3}, "may change only"),  # another experiment, not a continuation
    ({"seed": 7}, "may change only"),
])
def test_a_continuation_may_change_only_name_and_step_budget(sandbox, tmp_path, change, message):
    root, src = source_run(sandbox, tmp_path)
    with pytest.raises(SystemExit, match=message):
        trainer.continue_run(cfg(tmp_path, "tiny_ext", 4, **change), src, device="cpu")
    assert not (root / "runs" / "tiny_ext").exists()


def test_other_unsafe_continuations_are_refused(sandbox, tmp_path):
    root, src = source_run(sandbox, tmp_path)
    with pytest.raises(SystemExit, match="max_steps > 2"):
        trainer.continue_run(cfg(tmp_path, "tiny_ext", 2), src, device="cpu")
    with pytest.raises(SystemExit, match="itself"):
        trainer.continue_run(cfg(tmp_path, "tiny", 4), src, device="cpu")
    (root / "runs" / "tiny_ext").mkdir()
    with pytest.raises(SystemExit, match="already exists"):
        trainer.continue_run(cfg(tmp_path, "tiny_ext", 4), src, device="cpu")
    (root / "runs" / "tiny_ext").rmdir()
    runs.set_status("tiny", "interrupted")
    with pytest.raises(SystemExit, match="only a completed run"):
        trainer.continue_run(cfg(tmp_path, "tiny_ext", 4), src, device="cpu")
    assert not (root / "runs" / "tiny_ext").exists()


def test_a_tampered_copied_checkpoint_is_refused(sandbox, tmp_path, monkeypatch):
    root, src = source_run(sandbox, tmp_path)
    ext = cfg(tmp_path, "tiny_ext", 4)
    monkeypatch.setattr(trainer, "train", lambda *a, **k: None)  # set up the continuation only
    trainer.continue_run(ext, src, device="cpu")
    monkeypatch.undo()
    monkeypatch.setattr(trainer, "lora_model", tiny_lora_model)
    monkeypatch.setattr(trainer, "group_rewards", content_rewards)
    monkeypatch.setattr(runs, "ROOT", root)
    copied = root / "runs" / "tiny_ext" / "checkpoints" / "step2"
    rec = runs.manifest("tiny_ext")["continued_from"]
    m = runs.manifest("tiny_ext")
    m["continued_from"] = {**rec, "checkpoint_tree_sha256": "0" * 64}
    runs.write_json(root / "runs" / "tiny_ext" / "manifest.json", m)
    with pytest.raises(SystemExit, match="changed since the continuation"):
        trainer.train(ext, resume=copied, device="cpu")


def test_coverage_counts_only_what_the_run_recorded():
    result = {"training": [{"attempt": 1, "prompt_ids": [0, 1]}, {"attempt": 1, "prompt_ids": [0, 1]},
                           {"attempt": 3, "prompt_ids": [1, 2]}],
              "zero_signal": [{"attempt": 2, "prompt_ids": [3, 0]}]}
    c = trainer.coverage(result, [0, 1, 2, 3, 4, 5], attempts=3, steps=3, G=6)
    assert c == {"train_pool_prompts": 6, "rollout_attempts": 3, "successful_updates": 3, "attempts_with_records": 3,
                 "prompt_selections": 6, "unique_prompts_visited": 4, "coverage_fraction": 4 / 6,
                 "zero_signal_batches": 1, "skipped_batches": 1, "generated_completions": 36,
                 "full_pass_over_pool": False}
    full = trainer.coverage(result, [0, 1, 2, 3], attempts=3, steps=3, G=6)
    assert full["full_pass_over_pool"] and full["coverage_fraction"] == 1.0
