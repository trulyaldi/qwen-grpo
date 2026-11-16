"""qwen_grpo.trainer on CPU with a tiny random Qwen3 (real tokenizer, no weights): resumable checkpoints, the final
checkpoint, provenance, and the direction of one GRPO update."""
import copy
import hashlib

import pytest
import torch
import yaml
from peft import get_peft_model
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import checkpointing, config, runs, trainer
from qwen_grpo.model import EOS_IDS, lora_config, target_modules
from tests.test_config import RAW

SETTINGS = {"batch_size": 2, "group_size": 3, "max_steps": 4, "max_attempts": 40, "max_new_tokens": 3}


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)  # tests/test_grpo.py sets float64 at import


def tiny_lora_model(dtype, device, target, r, alpha, name=None):
    """Stand-in for model.lora_model: a fixed random 4-layer Qwen3 base (its own RNG, so the global stream is consumed
    exactly as by from_pretrained), LoRA resolved by the real target_modules()."""
    with torch.random.fork_rng():
        torch.manual_seed(123)
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=151936, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                                            tie_word_embeddings=True)).to(dtype).eval()
    base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
    return get_peft_model(base, lora_config(target_modules(base, *target), r, alpha)).eval()


def hashed_rewards(responses, labels, G):
    """Deterministic mixed rewards from the sampled text, so steps happen and depend on the restored RNG state."""
    r = [float(int(hashlib.md5(x.encode()).hexdigest(), 16) % 2) for x in responses]
    return torch.tensor(r).reshape(len(labels), G)


def make_cfg(tmp_path, every=2, **training):
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = "tiny"
    raw["model"]["dtype"] = "float32"
    raw["data"].update(train_samples=4, val_subset=2)
    raw["training"].update(SETTINGS, **training)
    raw["evaluation"].update(every=every, full_validation=False)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run the trainer with outputs under tmp_path/<label>/."""
    monkeypatch.setattr(trainer, "lora_model", tiny_lora_model)
    monkeypatch.setattr(trainer, "group_rewards", hashed_rewards)

    def use(label):
        root = tmp_path / label
        root.mkdir()
        monkeypatch.setattr(runs, "ROOT", root)
        return root
    return use


def ckpt(root, step, name="tiny"):
    return root / "runs" / name / "checkpoints" / f"step{step}"


def state(root, step):
    return checkpointing.load_training_state(ckpt(root, step))


def adapter(root, step):
    from peft import load_peft_weights
    return load_peft_weights(str(ckpt(root, step)), device="cpu")


def same_tensors(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_tensors(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_tensors(x, y) for x, y in zip(a, b))
    return a == b


def test_resumed_run_equals_uninterrupted_run_bitwise(sandbox, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    full = sandbox("full")
    ref = trainer.train(cfg, device="cpu")
    assert ref["successful_steps"] == 4 and ref["training"][-1]["change_from_init"] > 0  # real updates happened
    assert not ref["params_equal_init"] and ref["frozen_changed"] == []

    part = sandbox("part")
    save = checkpointing.save_checkpoint

    def crash_after_step2(model, optimizer, path, st):
        save(model, optimizer, path, st)
        if st["step"] == 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "save_checkpoint", crash_after_step2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cfg, device="cpu")
    identity = runs.manifest("tiny")["id"]
    assert runs.status("tiny", checkpointing.lock_owner) == "interrupted"
    summary = runs.read_json(part / "runs" / "tiny" / "summary.json")  # rebuilt from the step-2 training state
    assert summary["status"] == "interrupted" and summary["progress"]["optimizer_steps"] == 2
    assert [c["step"] for c in summary["checkpoints"]] == [0, 2] and summary["full_validation"] is None
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    out = trainer.train(cfg, resume=ckpt(part, 2), device="cpu")
    m = runs.manifest("tiny")
    assert m["id"] == identity and [h["status"] for h in m["status_history"]] == ["running", "interrupted", "running", "completed"]

    assert out["resumed"][0]["step"] == 2
    keys = ("attempt", "rollout_sha256", "rewards", "loss", "grad_norm", "update_norm", "change_from_init")
    assert [[r[k] for k in keys] for r in out["training"]] == [[r[k] for k in keys] for r in ref["training"]]
    assert [z["rollout_sha256"] for z in out["zero_signal"]] == [z["rollout_sha256"] for z in ref["zero_signal"]]
    assert [e["predictions"] for e in out["evaluations"]] == [e["predictions"] for e in ref["evaluations"]]
    assert same_tensors(adapter(part, 4), adapter(full, 4))
    assert same_tensors(state(part, 4)["optimizer"], state(full, 4)["optimizer"])
    assert same_tensors(state(part, 4)["rng"]["torch"], state(full, 4)["rng"]["torch"])
    assert out["batches_by_mixed_groups"] == ref["batches_by_mixed_groups"]
    plots = sorted(p.name for p in (full / "runs" / "tiny" / "plots").iterdir())  # train writes plots and the report
    assert {"reward.png", "validation_accuracy.png", "gradient_norm.png", "prediction_bias.png"} <= set(plots)
    assert (full / "runs" / "tiny" / "report.md").read_text().startswith("# Run `tiny` — completed")
    log = (full / "runs" / "tiny" / "training" / "training.log").read_text()
    assert "step  4" in log and "train time" in log


def test_resume_inside_a_reused_rollout_batch_equals_uninterrupted_run_bitwise(sandbox, tmp_path, monkeypatch):
    """num_iterations=3, checkpoints every 2 updates: step 2 lands after the 2nd update on the first batch, and the run
    stops after the first update on the second. The checkpoint carries the in-flight batch and its old log-probs."""
    from qwen_grpo.verification import training_diagnostics
    cfg = make_cfg(tmp_path, num_iterations=3)
    full = sandbox("full")
    ref = trainer.train(cfg, device="cpu")
    rows = ref["training"]
    assert [r["step"] for r in rows] == [1, 2, 3, 4] and [r["iteration"] for r in rows] == [0, 1, 2, 0]
    assert len({r["attempt"] for r in rows[:3]}) == 1 and rows[3]["attempt"] > rows[0]["attempt"]
    assert rows[0]["ratio_min"] == rows[0]["ratio_max"] == 1 and rows[1]["ratio_max"] != 1  # same rollouts, moved policy
    pend = state(full, 2)["pending"]
    assert pend["iteration"] == 2 and pend["group"]["rollout_sha256"] == rows[0]["rollout_sha256"]
    assert state(full, 4)["pending"]["iteration"] == 1  # max_steps cut the second batch after one update
    attempts = {g["attempt"]: g for g in rows + ref["zero_signal"]}
    td = training_diagnostics(ref)
    assert td["num_iterations"] == 3 and td["max_init_diff"] == 0.0
    assert td["mixed_reward_group_fraction"] == sum(g["mixed_groups"] for g in attempts.values()) / (2 * len(attempts))  # per batch, B=2

    part = sandbox("part")
    save = checkpointing.save_checkpoint

    def crash_after_step2(model, optimizer, path, st):
        save(model, optimizer, path, st)
        if st["step"] == 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "save_checkpoint", crash_after_step2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cfg, device="cpu")
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    out = trainer.train(cfg, resume=ckpt(part, 2), device="cpu")
    strip = lambda rs: [{k: v for k, v in r.items() if k != "step_seconds"} for r in rs]
    assert strip(out["training"]) == strip(rows) and out["zero_signal"] == ref["zero_signal"]
    assert [e["predictions"] for e in out["evaluations"]] == [e["predictions"] for e in ref["evaluations"]]
    assert same_tensors(adapter(part, 4), adapter(full, 4))
    assert same_tensors(state(part, 4)["optimizer"], state(full, 4)["optimizer"])
    assert same_tensors(state(part, 4)["pending"], state(full, 4)["pending"])


def test_the_evaluation_interval_does_not_change_the_trajectory(sandbox, tmp_path):
    """evaluation.every only adds subset evaluations and checkpoints (greedy, no random draws): the updates are bitwise
    those of a run that evaluates less often. This is why provenance checks may ignore the evaluation section."""
    sandbox("every2")
    ref = trainer.train(make_cfg(tmp_path), device="cpu")
    sandbox("every1")
    out = trainer.train(make_cfg(tmp_path, every=1), device="cpu")
    strip = lambda rs: [{k: v for k, v in r.items() if k != "step_seconds" and not k.endswith("_gib")} for r in rs]
    assert len(out["evaluations"]) == 5 > len(ref["evaluations"]) == 3
    assert strip(out["training"]) == strip(ref["training"]) and out["zero_signal"] == ref["zero_signal"]
    assert same_tensors(adapter(tmp_path / "every1", 4), adapter(tmp_path / "every2", 4))


def test_num_iterations_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="num_iterations"):
        make_cfg(tmp_path, num_iterations=0)


def test_resume_refuses_other_configs_later_checkpoints_and_adapter_only_checkpoints(sandbox, tmp_path):
    cfg = make_cfg(tmp_path)
    root = sandbox("run")
    trainer.train(cfg, device="cpu")
    with pytest.raises(SystemExit, match=r"different config \(training.learning_rate\)"):
        trainer.train(make_cfg(tmp_path, learning_rate=1e-3), resume=ckpt(root, 4), device="cpu")
    with pytest.raises(SystemExit, match="overwrite later checkpoints"):
        trainer.train(cfg, resume=ckpt(root, 2), device="cpu")
    with pytest.raises(SystemExit, match="cannot resume"):
        trainer.train(cfg, resume=ckpt(root, 0), device="cpu")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        trainer.train(cfg, device="cpu")
    legacy = root / "checkpoints" / "tiny_step4"  # a checkpoint in the layout before runs/: not resumable into runs/
    legacy.parent.mkdir()
    import shutil
    shutil.copytree(ckpt(root, 4), legacy)
    with pytest.raises(SystemExit, match="cannot be resumed"):
        trainer.train(cfg, resume=legacy, device="cpu")
    assert runs.manifest("tiny")["status"] == "completed"  # refused resumes never touch the run


def test_resume_from_the_final_checkpoint_only_reruns_the_closing_steps(sandbox, tmp_path):
    cfg = make_cfg(tmp_path)
    root = sandbox("run")
    ref = trainer.train(cfg, device="cpu")
    saved = config.load(root / "runs" / "tiny" / "config.yaml")  # the resolved config train wrote: a valid --config
    assert saved == cfg and ref["config_sha256"] == config.sha256(cfg) == runs.manifest("tiny")["config_sha256"]
    out = trainer.train(saved, resume=ckpt(root, 4), device="cpu")
    assert out["config_sha256"] == ref["config_sha256"]
    assert out["successful_steps"] == ref["successful_steps"] and out["training"] == ref["training"]
    assert out["final_checkpoint"] == ref["final_checkpoint"]


def test_exhausted_attempt_budget_saves_and_reports_the_last_step(sandbox, tmp_path, monkeypatch):
    calls = []

    def first_batch_zero_signal(responses, labels, G):
        calls.append(1)
        return torch.zeros(len(labels), G) if len(calls) == 1 else hashed_rewards(responses, labels, G)
    monkeypatch.setattr(trainer, "group_rewards", first_batch_zero_signal)
    cfg = make_cfg(tmp_path, max_steps=5, max_attempts=5, every=10)
    root = sandbox("run")
    out = trainer.train(cfg, device="cpu")
    step = out["successful_steps"]
    assert out["zero_signal_groups"] >= 1 and 0 < step < 5  # budget ran out between evaluation.every checkpoints
    assert out["final_checkpoint"]["step"] == step and out["final_checkpoint"]["path"].endswith(f"tiny/checkpoints/step{step}")
    assert (ckpt(root, step) / checkpointing.STATE).exists() and runs.latest_checkpoint("tiny") == ckpt(root, step)


def test_training_log_records_provenance(sandbox, tmp_path):
    cfg = make_cfg(tmp_path)
    sandbox("run")
    env = trainer.train(cfg, device="cpu")["environment"]
    assert len(env["git_commit"]) == 40 and isinstance(env["git_dirty"], bool)
    assert set(env["packages"]) == set(trainer.PACKAGES) and all(env["packages"].values())
    assert env["dataset"]["name"] == "cornell-movie-review-data/rotten_tomatoes" and len(env["dataset"]["sha256"]) == 64
    assert env["dataset"]["sizes"] == {"train": 1500, "validation": 1066, "test": 1066} and env["device"] == "cpu"


def test_one_grpo_step_decreases_the_grpo_loss_through_both_lora_factors(tok):
    """Gradient flow through LoRA: a small step lowers the clipped surrogate loss (first order: -lr ||grad||^2), and
    every A and B tensor receives a gradient. Per-sequence log-prob changes have no guaranteed sign (shared gradients)."""
    from qwen_grpo.grpo import group_advantages, grpo_loss
    from qwen_grpo.policy import compute_logprobs, generate_rollouts
    from qwen_grpo.sentiment import build_prompts
    from qwen_grpo.training import grpo_step

    torch.manual_seed(0)
    model = tiny_lora_model(torch.float32, "cpu", (("q_proj", "v_proj", "o_proj"), 2), 8, 16)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.1)  # with B = 0 the gradient of A is exactly zero
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    rollouts = generate_rollouts(model, tok, build_prompts(tok, ["A gorgeous, witty film."]), G=4, max_new_tokens=4)
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    adv, mask = group_advantages(rewards).flatten(), rollouts["completion_mask"]
    with torch.no_grad():
        old = compute_logprobs(model, rollouts)
    out, = grpo_step(model, params, torch.optim.SGD(params.values(), lr=1e-3), rollouts, rewards)
    with torch.no_grad():
        after = grpo_loss(compute_logprobs(model, rollouts), old, adv, mask).item()
    assert out["loss"] == pytest.approx(0.0, abs=1e-6)  # ratio 1: -mean(A) = 0
    assert all(g > 0 for g in out["grad_norms"].values()) and len(out["grad_norms"]) == 12
    assert after < out["loss"] and after == pytest.approx(-1e-3 * out["grad_norm"] ** 2, rel=0.05)


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")


def test_overwrite_deletes_earlier_outputs_only_when_enabled(sandbox, tmp_path):
    root = sandbox("run")
    trainer.train(make_cfg(tmp_path, max_steps=2), device="cpu")
    stale = ckpt(root, 99)
    stale.mkdir()
    other = root / "runs" / "tiny_lr0"  # a different run whose name starts the same way
    other.mkdir()
    with pytest.raises(SystemExit, match="run.overwrite: true"):
        trainer.train(make_cfg(tmp_path), device="cpu")
    raw = yaml.safe_load((tmp_path / "cfg.yaml").read_text())
    raw["run"]["overwrite"] = True
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(raw))
    out = trainer.train(config.load(tmp_path / "cfg.yaml"), device="cpu")
    assert out["successful_steps"] == 4 and not stale.exists() and other.exists()
    assert [s for s, _ in runs.checkpoints("tiny")] == [0, 2, 4]
