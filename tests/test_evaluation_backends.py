"""Standalone evaluation (scripts/run.py validate / test) through the run's execution backend.

The same checkpoint evaluated through the resident backend and through synchronous L2L (CPU, pageable store) gives
identical greedy predictions and label scores; the L2L evaluation never calls model.generate(), records its backend,
and evaluates the base model through L2L instead of reusing the resident engine's saved baseline. verify stays
resident-only and says where the L2L verification is."""
import pytest
import torch
import transformers

from qwen_grpo import runs, trainer
from qwen_grpo.sentiment import load_splits
from tests.test_backend import L2L_CPU, make_cfg
from tests.test_trainer import sandbox, tiny_lora_model  # noqa: F401

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("run_script", ROOT / "scripts" / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def test_validate_runs_through_the_configured_backend(sandbox, tmp_path, monkeypatch):  # noqa: F811
    sandbox("eval")
    resident, l2l = make_cfg(tmp_path), make_cfg(tmp_path, L2L_CPU)
    trainer.train(resident, device="cpu")
    ckpt = runs.checkpoint_path("tiny", 2)
    monkeypatch.setattr("qwen_grpo.model.lora_model", tiny_lora_model)
    splits = load_splits()
    small = {**splits, "validation": splits["validation"][:12]}
    monkeypatch.setattr("qwen_grpo.sentiment.load_splits", lambda: small)
    monkeypatch.setattr(run, "updated", lambda checkpoint: None)
    # the tiny base has no Hub name, so the adapter's base_model_name_or_path cannot match model.name here
    monkeypatch.setattr("qwen_grpo.checkpointing.check_adapter_matches", lambda cfg, c: None)

    a = run.evaluate_checkpoint(resident, ckpt, "validation", device="cpu")
    monkeypatch.setattr(transformers.GenerationMixin, "generate", lambda *x, **k: pytest.fail("model.generate() used"))
    b = run.evaluate_checkpoint(l2l, ckpt, "validation", device="cpu")

    assert a["execution"]["backend"] == "resident" and b["execution"]["backend"] == "l2l_sync"
    assert [p["id"] for p in a["predictions"]] == [p["id"] for p in b["predictions"]] == list(range(12))
    assert a["predictions"] == b["predictions"]  # greedy decoding: identical through both engines
    assert a["label_scores"] == b["label_scores"]  # trained and base scores, bitwise on the same CPU
    assert b["baseline_source"] == "in-process (adapter disabled, l2l_sync)"  # never the resident engine's file
    assert a["decoding"] == b["decoding"] == {"do_sample": False, "max_new_tokens": resident.evaluation.max_new_tokens}


def test_verify_of_an_l2l_run_points_to_the_l2l_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    cfg = make_cfg(tmp_path, L2L_CPU)
    import shutil
    runs.write_json(runs.checkpoint_path("tiny", 2) / "adapter_config.json", {})
    shutil.copy(tmp_path / "tiny_l2l.yaml", runs.run_dir("tiny") / "config.yaml")
    with pytest.raises(SystemExit, match="resident backend only.*scripts/final_eval.py --run tiny"):
        run.main(["verify", "--run", "tiny"])
    assert cfg.execution.backend == "l2l_sync"


def test_cuda_is_never_silently_replaced(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from qwen_grpo import config
    c = config.load(ROOT / "configs" / "profiles" / "qwen3_0_6b.yaml")
    with pytest.raises(SystemExit, match="refusing to fall back to the CPU"):
        trainer.check_device(c, "cuda")
    trainer.check_device(c, "cpu")  # an explicit CPU run is allowed


def test_a_gpu_without_kernels_or_bf16_is_refused(monkeypatch):
    from qwen_grpo import config

    class Props:
        name, major, minor = "Old GPU", 6, 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: Props())
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"])
    c = config.load(ROOT / "configs" / "profiles" / "qwen3_0_6b.yaml")
    with pytest.raises(SystemExit, match="sm_61.*not supported by this torch build"):
        trainer.check_device(c, "cuda")
    Props.major, Props.minor = 7, 5
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(SystemExit, match="does not support bf16"):
        trainer.check_device(c, "cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    trainer.check_device(c, "cuda")
