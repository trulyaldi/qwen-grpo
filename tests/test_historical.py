"""Saved artifacts (results/, checkpoints/) stay valid evidence under the current code: historical baseline numbers,
default.yaml = the historical run, and every committed checkpoint still matches its config and loads by tensor name.
Needs the Qwen3-0.6B tokenizer/config files only, no weights."""
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from peft import get_peft_model, get_peft_model_state_dict
from safetensors import safe_open

from qwen_grpo import checkpointing, config
from qwen_grpo.evaluation import metrics
from qwen_grpo.model import EXPECTED_MODULES, PRESETS, lora_config, skeleton, target_modules

ROOT = Path(__file__).parents[1]
DEFAULT = config.load(ROOT / "configs" / "default.yaml")
CHECKPOINTS = sorted(p for p in (ROOT / "checkpoints").glob("*_step*") if (p / "adapter_config.json").exists())


def named(cfg, name):
    return dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, name=name))


def test_saved_validation_baseline_numbers():
    preds = json.loads((ROOT / "results" / "baseline" / "validation.json").read_text())["predictions"]
    m = metrics(preds)
    assert m["n"] == 1066 and round(m["accuracy"], 4) == 0.7486 and sum(p["prediction"] is None for p in preds) == 3


def test_default_yaml_is_the_historical_default_rerun_config():
    saved = config.normalized(json.loads((ROOT / "results" / "default_rerun" / "training.json").read_text())["yaml_config"])
    now = config.as_dict(named(DEFAULT, "default_rerun"))
    assert saved == now  # normalized() drops the verification section that logs of that time carried
    assert saved["evaluation"] == now["evaluation"] == dataclasses.asdict(config.Evaluation())


@pytest.fixture(scope="module")
def qwen06():
    return skeleton(DEFAULT.model.name)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_resolve_to_the_historical_0_6b_modules(qwen06, preset):
    assert target_modules(qwen06, *PRESETS[preset]) == EXPECTED_MODULES[preset]


@pytest.mark.parametrize("ckpt", CHECKPOINTS, ids=lambda p: p.name)
def test_committed_checkpoints_match_the_default_config_and_its_tensor_names(qwen06, ckpt):
    checkpointing.check_adapter_matches(DEFAULT, ckpt)  # resolved module names, not the saved regex string
    with torch.device("meta"):
        live = get_peft_model(skeleton(DEFAULT.model.name), lora_config(target_modules(qwen06, *DEFAULT.lora.spec())))
    with safe_open(str(ckpt / "adapter_model.safetensors"), "pt") as f:
        saved = set(f.keys())
    assert saved == set(get_peft_model_state_dict(live))  # what load_adapter requires


def test_other_targets_are_rejected_for_committed_checkpoints():
    other = dataclasses.replace(DEFAULT, lora=dataclasses.replace(DEFAULT.lora, targets="last-o-proj"))
    with pytest.raises(SystemExit, match="does not match the config"):
        checkpointing.check_adapter_matches(other, CHECKPOINTS[0])
