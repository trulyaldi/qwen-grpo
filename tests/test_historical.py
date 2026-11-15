"""Saved artifacts (results/, checkpoints/) stay valid evidence under the current code: historical baseline numbers,
default.yaml = the historical run, and every committed checkpoint and log still passes the checks verify relies on.
Needs the Qwen3-0.6B tokenizer/config files only, no weights."""
import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml
from peft import get_peft_model, get_peft_model_state_dict
from safetensors import safe_open

from qwen_grpo import config, verification
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
    """The minimal default.yaml (defaults omitted) equals the full YAML default_rerun was trained with, key for key,
    except the run name and the LR=0 control that run's verify used."""
    saved = json.loads((ROOT / "results" / "default_rerun" / "training.json").read_text())["yaml_config"]
    now = config.as_dict(DEFAULT)
    assert config.differences(config.normalized(saved), now) == ["run.name", "verification.negative_control.checkpoint"]
    assert config.normalized(saved)["evaluation"] == now["evaluation"] == dataclasses.asdict(config.Evaluation())
    saved["run"]["name"], saved["verification"]["negative_control"]["checkpoint"] = "default", None
    assert config.sha256(config._build(config.Config, saved, "saved")) == config.sha256(DEFAULT)


def test_a_committed_full_yaml_loads_and_equals_the_minimal_default():
    """verify wrote every key of the config it ran with; that full file still loads and equals default.yaml except for
    the LR=0 control it was pointed at."""
    path = ROOT / "results" / "verification" / "default_step30" / "config.yaml"
    full = config.load(path)
    assert "verification" in yaml.safe_load(path.read_text())
    assert config.differences(config.as_dict(full), config.as_dict(DEFAULT)) == ["verification.negative_control.checkpoint"]
    v = dataclasses.replace(full.verification, negative_control=config.NegativeControl())
    assert config.sha256(dataclasses.replace(full, verification=v)) == config.sha256(DEFAULT)


def test_saved_logs_still_pass_provenance_checks():
    log, source = verification.training_log(named(DEFAULT, "default_rerun"), ROOT / "checkpoints" / "default_rerun_step30")
    assert log is not None, source
    ok, why, _ = verification.control_provenance(named(DEFAULT, "default_rerun"), ROOT / "checkpoints" / "default_lr0_step30")
    assert ok, why


@pytest.fixture(scope="module")
def qwen06():
    return skeleton(DEFAULT.model.name)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_presets_resolve_to_the_historical_0_6b_modules(qwen06, preset):
    assert target_modules(qwen06, *PRESETS[preset]) == EXPECTED_MODULES[preset]


@pytest.mark.parametrize("ckpt", CHECKPOINTS, ids=lambda p: p.name)
def test_committed_checkpoints_match_the_default_config_and_its_tensor_names(qwen06, ckpt):
    verification.check_adapter_matches(DEFAULT, ckpt)  # resolved module names, not the saved regex string
    with torch.device("meta"):
        live = get_peft_model(skeleton(DEFAULT.model.name), lora_config(target_modules(qwen06, *DEFAULT.lora.spec())))
    with safe_open(str(ckpt / "adapter_model.safetensors"), "pt") as f:
        saved = set(f.keys())
    assert saved == set(get_peft_model_state_dict(live))  # what load_adapter requires


def test_other_targets_are_rejected_for_committed_checkpoints():
    other = dataclasses.replace(DEFAULT, lora=dataclasses.replace(DEFAULT.lora, targets="last-o-proj"))
    with pytest.raises(SystemExit, match="does not match the config"):
        verification.check_adapter_matches(other, CHECKPOINTS[0])
