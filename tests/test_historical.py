"""Saved artifacts (results/baseline, runs/qwen) stay valid evidence under the current code: the baseline numbers, the
reference run's config, log and checkpoints load, resume and pass the checks verify relies on. Needs the Qwen3-0.6B
tokenizer/config files only, no weights. (Older runs were deleted in commit b43b0fa; git history keeps them.)"""
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from peft import get_peft_model, get_peft_model_state_dict
from safetensors import safe_open

from qwen_grpo import checkpointing, config, runs, trainer, verification
from qwen_grpo.evaluation import metrics
from qwen_grpo.model import EXPECTED_MODULES, PRESETS, lora_config, skeleton, target_modules

ROOT = Path(__file__).parents[1]
RUN = ROOT / "runs" / "qwen"
QWEN = config.load(ROOT / "configs" / "qwen.yaml")
CHECKPOINTS = [p for _, p in sorted((int(p.name[4:]), p) for p in (RUN / "checkpoints").glob("step*"))]


def test_saved_validation_baseline_numbers():
    preds = json.loads((ROOT / "results" / "baseline" / "validation.json").read_text())["predictions"]
    m = metrics(preds)
    assert m["n"] == 1066 and round(m["accuracy"], 4) == 0.7486 and sum(p["prediction"] is None for p in preds) == 3


def test_reference_run_results_are_the_reported_numbers():
    final = json.loads((RUN / "evaluation" / "step90" / "validation.json").read_text())
    assert round(final["metrics"]["accuracy"], 4) == 0.8096 and round(final["auc"]["trained"], 4) == 0.8929
    assert round(final["auc"]["base"], 4) == 0.8277


def test_saved_log_and_checkpoint_configs_are_the_reference_yaml():
    """The training log's yaml_config and every checkpoint's training state normalize to configs/qwen.yaml."""
    log = json.loads((RUN / "training" / "training.json").read_text())
    assert config.normalized(log["yaml_config"]) == config.as_dict(QWEN)
    for ckpt in CHECKPOINTS[1:]:
        state = checkpointing.load_training_state(ckpt)
        assert config.normalized(state["config"]) == config.as_dict(QWEN), ckpt


def test_saved_log_passes_the_provenance_check():
    log, source = verification.training_log(QWEN, CHECKPOINTS[-1])
    assert log is not None, source
    other = dataclasses.replace(QWEN, training=dataclasses.replace(QWEN.training, seed=7))
    assert verification.training_log(other, CHECKPOINTS[-1])[0] is None


def test_reference_run_resumes_under_the_current_code():
    """check_resume accepts the final checkpoint with the run's saved (historical) config and with configs/qwen.yaml, and
    refuses an earlier one only because later checkpoints exist (the config comparison comes first)."""
    saved = config.load(RUN / "config.yaml", historical=True)
    last = CHECKPOINTS[-1]
    state = checkpointing.load_training_state(last)
    assert runs.parse_checkpoint(last) == ("qwen", 100, "run")
    for cfg in (saved, QWEN):
        trainer.check_resume(cfg, last, state)
    with pytest.raises(SystemExit, match="overwrite later checkpoints"):
        trainer.check_resume(QWEN, RUN / "checkpoints" / "step90", checkpointing.load_training_state(RUN / "checkpoints" / "step90"))
    micro = dataclasses.replace(QWEN, training=dataclasses.replace(QWEN.training, micro_batch_size=6))
    with pytest.raises(SystemExit, match=r"different config \(training.micro_batch_size\)"):
        trainer.check_resume(micro, last, state)


@pytest.fixture(scope="module")
def qwen06():
    return skeleton(QWEN.model.name)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_resolve_to_the_historical_0_6b_modules(qwen06, preset):
    assert target_modules(qwen06, *PRESETS[preset]) == EXPECTED_MODULES[preset]


@pytest.mark.parametrize("ckpt", CHECKPOINTS, ids=lambda p: p.name)
def test_committed_checkpoints_match_the_reference_config_and_its_tensor_names(qwen06, ckpt):
    verification.check_adapter_matches(QWEN, ckpt)  # resolved module names, not the saved regex string
    with torch.device("meta"):
        live = get_peft_model(skeleton(QWEN.model.name), lora_config(target_modules(qwen06, *QWEN.lora.spec()), 32, 64))
    with safe_open(str(ckpt / "adapter_model.safetensors"), "pt") as f:
        saved = set(f.keys())
    assert saved == set(get_peft_model_state_dict(live))  # what load_adapter requires


def test_other_targets_are_rejected_for_committed_checkpoints():
    other = dataclasses.replace(QWEN, lora=dataclasses.replace(QWEN.lora, targets="last-o-proj"))
    with pytest.raises(SystemExit, match="does not match the config"):
        verification.check_adapter_matches(other, CHECKPOINTS[0])
    rank = dataclasses.replace(QWEN, lora=dataclasses.replace(QWEN.lora, rank=8))
    with pytest.raises(SystemExit, match="does not match the config"):
        verification.check_adapter_matches(rank, CHECKPOINTS[0])
