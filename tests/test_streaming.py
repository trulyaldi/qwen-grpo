"""Disk-backed prompt and rollout streaming with deterministic, crash-consistent resume (qwen_grpo/streaming/).

The real trainer on CPU with a tiny random Qwen3 (real tokenizer and data; tests/test_trainer.py's sandbox), rewards
from a content hash that also yields zero-signal batches. Experiment A runs uninterrupted; B is crashed at an injected
point and resumed from the latest valid checkpoint. Committed sample occurrences, order, rollout groups, attempts
(zero-signal included), optimizer-step counters, policy versions, generated tokens, rewards, optimizer state and
final LoRA weights must be identical, and every epoch's committed occurrences must cover the dataset exactly once.
"""
import copy
import hashlib
import json

import pytest
import torch
import yaml

from qwen_grpo import config, runs, trainer
from qwen_grpo.streaming import resume as R
from qwen_grpo.streaming.dataset import PromptStream
from qwen_grpo.streaming.rollouts import RolloutStore
from qwen_grpo.training import shuffled_batches
from tests.test_config import RAW
from tests.test_trainer import SETTINGS, adapter, same_tensors, state, tiny_lora_model

STREAM = {"enabled": True, "directory": "streaming"}


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def content_rewards(responses, labels, G):
    """Deterministic rewards from the sampled text; every batch whose text hashes to 0 mod 4 is zero-signal."""
    h = int(hashlib.md5("|".join(responses).encode()).hexdigest(), 16)
    if h % 4 == 0:
        return torch.zeros(len(labels), G)
    return torch.tensor([float(int(hashlib.md5(x.encode()).hexdigest(), 16) % 2) for x in responses]).reshape(len(labels), G)


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


def make_cfg(tmp_path, streaming=True, **training):
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = "tiny"
    raw["model"]["dtype"] = "float32"
    raw["data"].update(train_samples=4, val_subset=2)
    raw["training"].update({**SETTINGS, "max_steps": 6, "num_iterations": 3, **training})
    raw["evaluation"].update(every=2, full_validation=False)
    if streaming:
        raw["streaming"] = STREAM
    path = tmp_path / f"cfg_{streaming}_{hash(json.dumps(training, sort_keys=True))}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


KEYS = ("step", "attempt", "prompt_ids", "rollout_sha256", "rewards", "loss", "grad_norm", "update_norm", "iteration",
        "clip_frac", "init_diff")


def committed(out):
    return {"rows": [[r.get(k) for k in KEYS] for r in out["training"]],
            "zero_signal": [(z["attempt"], z["rollout_sha256"]) for z in out["zero_signal"]],
            "evaluations": [e["predictions"] for e in out["evaluations"]],
            "attempts": out["groups_generated"], "steps": out["successful_steps"]}


def test_a_streaming_run_equals_the_in_memory_run_bitwise(sandbox, tmp_path):
    mem_root = sandbox("memory")
    ref = trainer.train(make_cfg(tmp_path, streaming=False), device="cpu")
    root = sandbox("stream")
    out = trainer.train(make_cfg(tmp_path), device="cpu")
    assert committed(out) == committed(ref) and out["zero_signal"]  # zero-signal batches occurred and match
    assert same_tensors(adapter(root, 6), adapter(mem_root, 6))
    assert same_tensors(state(root, 6)["optimizer"], state(mem_root, 6)["optimizer"])


def records(root):
    """{group_id: (policy_version, occurrences, response_ids, rewards)} of the committed on-disk rollout records."""
    d = root / "runs" / "tiny" / "streaming" / "rollouts"
    out = {}
    for line in (d / "index.jsonl").read_text().splitlines():
        e = json.loads(line)
        rec = torch.load(d / e["file"], weights_only=True)
        out[e["group_id"]] = (rec["policy_version"], [tuple(o) for o in rec["occurrences"]], rec["response_ids"],
                              rec["rewards"])
    return out


def assert_epochs_exact(recs, n_samples):
    """Committed occurrences: no duplicate (epoch, sample) and every completed epoch covers every sample once."""
    occ = [o for g in sorted(recs) for o in recs[g][1]]
    assert len(set((e, s) for e, _, s in occ)) == len(occ), "a sample occurs twice in one epoch"
    by_epoch = {}
    for e, pos, s in occ:
        by_epoch.setdefault(e, []).append((pos, s))
    for e, rows in by_epoch.items():
        assert [p for p, _ in sorted(rows)] == list(range(len(rows))), f"epoch {e}: positions not contiguous"
        if e < max(by_epoch):
            assert len(rows) == n_samples, f"epoch {e} incomplete"


@pytest.fixture
def reference(sandbox, tmp_path):
    root = sandbox("A")
    out = trainer.train(make_cfg(tmp_path), device="cpu")
    return root, out


CRASHES = ["prompt_read@3", "rollouts_generated@3", "update@4", "files_written@3", "committed_before_rename@3",
           "after_checkpoint@2"]


@pytest.mark.parametrize("crash", CRASHES)
def test_a_crashed_and_resumed_run_commits_the_same_training_sequence(reference, sandbox, tmp_path, monkeypatch,
                                                                     crash):
    ref_root, ref = reference
    root = sandbox("B")
    cfg = make_cfg(tmp_path)
    monkeypatch.setenv("QWEN_GRPO_CRASH", crash)
    trainer._CRASH_SEEN.clear()
    with pytest.raises(trainer.InjectedCrash):
        trainer.train(cfg, device="cpu")
    monkeypatch.delenv("QWEN_GRPO_CRASH")
    assert runs.status("tiny", lambda p: None) == "interrupted"
    latest, _ = R.latest_valid("tiny")
    assert latest is not None and latest.name in ("step2", "step4")
    out = trainer.train(cfg, resume="latest", device="cpu")
    assert committed(out) == committed(ref)
    a, b = records(ref_root), records(root)
    assert a.keys() == b.keys()
    for g in a:
        assert a[g][:2] == b[g][:2] and torch.equal(a[g][2], b[g][2]) and torch.equal(a[g][3], b[g][3]), g
    assert_epochs_exact(b, 4)
    assert same_tensors(adapter(root, 6), adapter(ref_root, 6))
    assert same_tensors(state(root, 6)["optimizer"], state(ref_root, 6)["optimizer"])
    assert not list((root / "runs" / "tiny" / "checkpoints").glob(".tmp-*"))
    for step, p in runs.checkpoints("tiny"):
        assert R.validate(p, resumable=step > 0)[0], p


def test_a_corrupted_latest_checkpoint_is_rejected_and_the_previous_valid_one_resumes(reference, sandbox, tmp_path,
                                                                                    monkeypatch):
    ref_root, ref = reference
    root = sandbox("B")
    cfg = make_cfg(tmp_path)
    monkeypatch.setenv("QWEN_GRPO_CRASH", "after_checkpoint@2")  # step4 is published, then the process dies
    trainer._CRASH_SEEN.clear()
    with pytest.raises(trainer.InjectedCrash):
        trainer.train(cfg, device="cpu")
    monkeypatch.delenv("QWEN_GRPO_CRASH")
    bad = root / "runs" / "tiny" / "checkpoints" / "step4" / "training_state.pt"
    data = bytearray(bad.read_bytes())
    data[len(data) // 2] ^= 0xFF  # a flipped byte: the file no longer matches COMMIT.json
    bad.write_bytes(bytes(data))
    latest, rejected = R.latest_valid("tiny")
    assert latest.name == "step2" and [p.name for p, _ in rejected] == ["step4"]
    with pytest.raises(SystemExit, match="not a valid checkpoint"):
        trainer.train(cfg, resume=bad.parent, device="cpu")
    out = trainer.train(cfg, resume="latest", device="cpu")
    assert committed(out) == committed(ref)
    assert same_tensors(adapter(root, 6), adapter(ref_root, 6))
    assert list((root / "runs" / "tiny" / "checkpoints").glob(".invalid-step4-*"))  # kept aside, not deleted


# --- units ------------------------------------------------------------------------------------------------------------

def test_prompt_stream_reproduces_shuffled_batches_restores_without_replay_and_refuses_another_dataset(tmp_path):
    ex = [{"id": i * 3, "text": f"text {i}", "label": "POSITIVE"} for i in range(5)]
    s = PromptStream.build(tmp_path, ex, 7, 2)
    ref = shuffled_batches(ex, 2, seed=7)
    states = []
    for _ in range(7):
        states.append(s.state())
        assert [r["id"] for r in s.next_batch()[0]] == [e["id"] for e in next(ref)]
    t = PromptStream(tmp_path, 7, 2)
    t.restore(states[4])
    ref = shuffled_batches(ex, 2, seed=7)
    for _ in range(4):
        next(ref)
    assert [r["id"] for r in t.next_batch()[0]] == [e["id"] for e in next(ref)]
    with pytest.raises(RuntimeError, match="different dataset"):
        PromptStream.build(tmp_path, ex[:4], 7, 2)
    with pytest.raises(RuntimeError, match="another dataset"):
        PromptStream(tmp_path, 8, 2).restore(states[0])


def test_rollout_store_refuses_other_policy_versions_runs_tampering_and_rewrites(tmp_path):
    store = RolloutStore(tmp_path, "run-a")
    rec = {"group_id": 1, "policy_version": 3, "occurrences": [(0, 0, 5)], "zero_signal": False,
           "input_ids": torch.arange(6).view(2, 3), "attention_mask": torch.ones(2, 3), "response_ids": torch.ones(2, 1),
           "completion_mask": torch.ones(2, 1, dtype=torch.bool), "prompt_len": 2, "rewards": torch.ones(1, 2),
           "advantages": torch.zeros(2), "old_logp": torch.zeros(2, 1)}
    store.write(rec)
    assert torch.equal(store.read(1, 3)["input_ids"], rec["input_ids"])
    with pytest.raises(RuntimeError, match="policy version"):
        store.read(1, 4)
    with pytest.raises(RuntimeError, match="belongs to run"):
        RolloutStore(tmp_path, "run-b").read(1, 3)
    with pytest.raises(RuntimeError, match="already on disk"):
        store.write(rec)
    raw = torch.load(store.path(1), weights_only=True)
    raw["rewards"] = torch.zeros(1, 2)
    torch.save(raw, store.path(1))
    with pytest.raises(RuntimeError, match="sha256"):
        store.read(1, 3)
    store.write({**rec, "group_id": 2})
    assert store.discard_after(1) == [2] and [e["group_id"] for e in store.index()] == [1]


def test_publication_is_all_or_nothing_and_validation_catches_missing_or_changed_files(tmp_path, monkeypatch):
    final = tmp_path / "step3"
    write = lambda d: ((d / "adapter_config.json").write_text("{}"), (d / "training_state.pt").write_bytes(b"x" * 10))
    monkeypatch.setattr(R, "_crash_point", lambda name: (_ for _ in ()).throw(trainer.InjectedCrash(name))
                        if name == "committed_before_rename" else None)
    with pytest.raises(trainer.InjectedCrash):
        R.publish(final, write, 3)
    assert not final.exists() and list(tmp_path.glob(".tmp-step3-*"))  # nothing under the final name
    monkeypatch.setattr(R, "_crash_point", lambda name: None)
    R.publish(final, write, 3)
    assert R.validate(final) == (True, None)
    (final / "training_state.pt").write_bytes(b"y" * 10)
    assert not R.validate(final)[0]
    (final / "training_state.pt").unlink()
    assert "missing" in R.validate(final)[1]
    with pytest.raises(RuntimeError, match="never overwritten"):
        R.publish(final, write, 3)
