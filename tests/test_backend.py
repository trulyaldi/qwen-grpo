"""Policy-execution backends (qwen_grpo/execution/backend.py) inside the real trainer: an l2l_sync run on CPU (pageable
store, CPU buffers) reproduces the resident run bitwise (rollouts, losses, evaluations, adapters, optimizer state)
without ever calling model.generate(). The single-update grid with per-tensor comparisons is the lab case
update_equivalence."""
import copy

import pytest
import torch
import transformers
import yaml

from qwen_grpo import config, trainer
from qwen_grpo.execution.backend import L2LBackend, ResidentBackend, make_backend
from tests.test_config import RAW
from tests.test_trainer import SETTINGS, adapter, ckpt, sandbox, same_tensors, state, tiny_lora_model  # noqa: F401

L2L_CPU = {"backend": "l2l_sync", "weight_storage": "pageable", "resident_layers": 0, "weight_buffers": 2,
           "prefetch_depth": 0, "activation_checkpointing": True}


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def make_cfg(tmp_path, execution=None, name="tiny", **training):
    raw = copy.deepcopy(RAW)
    raw["run"]["name"] = name
    raw["model"]["dtype"] = "float32"
    raw["data"].update(train_samples=4, val_subset=2)
    raw["training"].update(SETTINGS, **training)
    raw["evaluation"].update(every=2, full_validation=False)
    if execution:
        raw["execution"] = execution
    path = tmp_path / f"{name}_{'l2l' if execution else 'resident'}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return config.load(path)


KEYS = ("attempt", "rollout_sha256", "rewards", "loss", "grad_norm", "update_norm", "change_from_init", "clip_frac",
        "ratio_min", "ratio_max", "approx_kl", "init_diff")


def rows(out):
    return [[r.get(k) for k in KEYS] for r in out["training"]]


@pytest.mark.parametrize("training", [{}, {"micro_batch_size": 4, "num_iterations": 2}])
def test_an_l2l_training_run_equals_the_resident_run_bitwise_without_model_generate(sandbox, tmp_path, monkeypatch,
                                                                                     training):
    res_root = sandbox("resident")
    ref = trainer.train(make_cfg(tmp_path, **training), device="cpu")
    l2l_root = sandbox("l2l")
    monkeypatch.setattr(transformers.GenerationMixin, "generate",
                        lambda *a, **k: pytest.fail("model.generate() called under the l2l_sync backend"))
    out = trainer.train(make_cfg(tmp_path, L2L_CPU, **training), device="cpu")
    assert ref["successful_steps"] == out["successful_steps"] == 4
    assert rows(out) == rows(ref)
    assert [z["rollout_sha256"] for z in out["zero_signal"]] == [z["rollout_sha256"] for z in ref["zero_signal"]]
    assert [e["predictions"] for e in out["evaluations"]] == [e["predictions"] for e in ref["evaluations"]]
    assert same_tensors(adapter(l2l_root, 4), adapter(res_root, 4))
    assert same_tensors(state(l2l_root, 4)["optimizer"], state(res_root, 4)["optimizer"])
    assert out["frozen_changed"] == [] and not out["params_equal_init"]


def test_an_l2l_run_resumes_bitwise(sandbox, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, L2L_CPU)
    full = sandbox("full")
    ref = trainer.train(cfg, device="cpu")
    part = sandbox("part")
    save = trainer.save_checkpoint

    def crash_after_step2(model, optimizer, path, st):
        save(model, optimizer, path, st)
        if st["step"] == 2:
            raise KeyboardInterrupt
    monkeypatch.setattr(trainer, "save_checkpoint", crash_after_step2)
    with pytest.raises(KeyboardInterrupt):
        trainer.train(cfg, device="cpu")
    monkeypatch.setattr(trainer, "save_checkpoint", save)
    out = trainer.train(cfg, resume=ckpt(part, 2), device="cpu")
    assert rows(out) == rows(ref)
    assert same_tensors(adapter(part, 4), adapter(full, 4))
    assert same_tensors(state(part, 4)["optimizer"], state(full, 4)["optimizer"])


def test_backends_expose_the_models_own_parameters_and_refuse_foreign_models(tmp_path):
    m = tiny_lora_model(torch.float32, "cpu", (("q_proj", "v_proj", "o_proj"), 2), 8, 16)
    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params.values())
    resident = ResidentBackend(m)
    before = resident.frozen_digests()
    b = make_backend(make_cfg(tmp_path, L2L_CPU), m, "cpu")
    assert isinstance(b, L2LBackend) and b.trainable() == params
    assert [p for g in optimizer.param_groups for p in g["params"]] == list(params.values())
    assert b.frozen_digests() == before  # moving into the store changed no frozen byte
    other = tiny_lora_model(torch.float32, "cpu", (("q_proj", "v_proj", "o_proj"), 2), 8, 16)
    with pytest.raises(RuntimeError, match="not this backend's model"):
        b.logprobs(other, {})
    b.close()


def test_config_gates(tmp_path):
    assert config.unimplemented(make_cfg(tmp_path, L2L_CPU)) == []
    assert config.unimplemented(make_cfg(tmp_path, {**L2L_CPU, "kv_cache": "cpu"})) == []
    assert config.unimplemented(make_cfg(tmp_path, {**L2L_CPU, "resident_layers": 1}))
    assert config.unimplemented(make_cfg(tmp_path, {**L2L_CPU, "vocab_chunk_size": 4096}))


def test_checkpoint_evaluation_commands_refuse_an_l2l_config_instead_of_running_resident(tmp_path):
    from scripts import run
    cfg_path = tmp_path / "tiny_l2l.yaml"
    make_cfg(tmp_path, L2L_CPU)
    for command in ("verify", "validate"):
        with pytest.raises(SystemExit, match="resident backend only"):
            run.main([command, "--config", str(cfg_path), "--checkpoint", str(tmp_path / "missing")])
