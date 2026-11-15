"""YAML config: strict loading, defaults, inheritance, hashing, validation, wiring of values into the implementation,
and CLI dispatch (no GPU)."""
import copy
import dataclasses
from pathlib import Path

import pytest
import torch
import yaml
from peft import get_peft_model
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import config, verification
from qwen_grpo.model import EOS_IDS, expected_lora_params, lora_config, lora_modules, target_modules
from qwen_grpo.policy import generate_rollouts
from scripts import run

ROOT = Path(__file__).parents[1]
DEFAULT = ROOT / "configs" / "default.yaml"
EXPERIMENTS = ROOT / "configs" / "experiments"
RAW = config.as_dict(config.load(DEFAULT))  # every key written out


def write(tmp_path, edit):
    raw = copy.deepcopy(RAW)
    edit(raw)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_loads_with_the_historical_run_values():
    cfg = config.load(DEFAULT)
    assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.targets) == (8, 16, "last2-attention")
    t = cfg.training
    assert (t.batch_size, t.batch_mode, t.group_size, t.learning_rate, t.max_steps, t.clip_eps) == (2, "balanced", 6, 1e-4, 30, 0.2)
    assert (cfg.data.train_samples, cfg.data.subset_seed, t.seed, cfg.model.dtype) == (32, 42, 42, "bfloat16")


@pytest.mark.parametrize("edit, message", [
    (lambda r: r.update(extra=1), "unknown keys ['extra']"),
    (lambda r: r["training"].update(lr=1e-4), "unknown keys ['lr']"),
    (lambda r: r["lora"].pop("rank"), "missing keys ['rank']"),
    (lambda r: r.pop("data"), "missing keys ['data']"),
    (lambda r: r["training"].update(learning_rate="1e-4"), "expected float, got str"),
    (lambda r: r["training"].update(group_size=True), "expected int, got bool"),
    (lambda r: r["training"].update(group_size=6.0), "expected int, got float"),
    (lambda r: r["model"].update(thinking=0), "expected bool, got int"),
    (lambda r: r.update(lora=[8, 16]), "expected a mapping"),
    (lambda r: r["verification"].update(models=["base"]), "unknown keys ['models']"),
    (lambda r: r["verification"]["negative_control"].update(checkpoint=3), "got int"),
    (lambda r: r["training"].pop("seed"), "missing keys ['seed']"),
    (lambda r: r["lora"].update(targets={"modules": ["q_proj"]}), "missing keys ['num_layers']"),
    (lambda r: r["lora"].update(targets={"modules": "q_proj", "num_layers": 1}), "expected list, got str"),
    (lambda r: r["lora"].update(targets=3), "lora.targets: expected"),
    (lambda r: r.update(evaluation={"max_new": 4}), "unknown keys ['max_new']"),
])
def test_unknown_missing_and_ill_typed_keys_are_rejected(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        config.load(write(tmp_path, edit))


@pytest.mark.parametrize("edit, message", [
    (lambda r: r["model"].update(name="meta-llama/Llama-3.1-8B"), "model.name"),
    (lambda r: r["model"].update(name="Qwen/Qwen3-30B-A3B"), "model.name"),
    (lambda r: r["model"].update(dtype="float16"), "model.dtype"),
    (lambda r: r["model"].update(thinking=True), "model.thinking"),
    (lambda r: r["lora"].update(targets="all"), "lora.targets"),
    (lambda r: r["lora"].update(targets={"modules": [], "num_layers": 2}), "lora.targets.modules"),
    (lambda r: r["lora"].update(targets={"modules": ["q_proj", "q_proj"], "num_layers": 2}), "lora.targets.modules"),
    (lambda r: r["lora"].update(targets={"modules": ["q_proj"], "num_layers": 0}), "lora.targets.num_layers"),
    (lambda r: r.update(evaluation={"bootstrap_samples": 1500}), "evaluation.bootstrap_samples"),
    (lambda r: r.update(evaluation={"batch_size": 0}), "evaluation: token and batch sizes"),
    (lambda r: r["run"].update(gpu_budget_gib=0.0), "run.gpu_budget_gib"),
    (lambda r: r["lora"].update(dropout=0.1), "lora.dropout"),
    (lambda r: r["training"].update(temperature=0.7), "training.temperature"),
    (lambda r: r["training"].update(batch_mode="cycle"), "cycle <=> batch_size 1"),
    (lambda r: r["training"].update(batch_size=4), "balanced needs batch_size 2"),
    (lambda r: r["training"].update(group_size=1), "training.group_size"),
    (lambda r: r["training"].update(max_steps=301), "max_steps <= max_attempts"),
    (lambda r: r["training"].update(learning_rate=-1.0), "training.learning_rate"),
    (lambda r: r["data"].update(train_samples=31), "data.train_samples"),
    (lambda r: r["run"].update(name="../escape"), "run.name"),
    (lambda r: r["verification"].update(split="train"), "verification.split"),
    (lambda r: r["verification"]["negative_control"].update(checkpoint="c", train=True), "checkpoint or train, not both"),
    (lambda r: r["verification"]["statistics"].update(bootstrap_samples=150), "bootstrap_samples"),
    (lambda r: r["verification"]["statistics"].update(confidence_level=1.5), "confidence_level"),
])
def test_unsupported_values_are_rejected(tmp_path, edit, message):
    with pytest.raises(ValueError, match="invalid config") as e:
        config.load(write(tmp_path, edit))
    assert message in str(e.value)


def test_lr_zero_and_float32_are_accepted(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: (r["training"].update(learning_rate=0), r["model"].update(dtype="float32"))))
    assert cfg.training.learning_rate == 0.0 and isinstance(cfg.training.learning_rate, float)
    assert config.DTYPES[cfg.model.dtype] is torch.float32


def test_lora_settings_reach_the_adapter(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: r["lora"].update(rank=4, alpha=12, targets="last-o-proj")))
    torch.manual_seed(0)
    tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=28,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    lo = cfg.lora
    model = get_peft_model(tiny, lora_config(target_modules(tiny, *lo.spec()), lo.rank, lo.alpha, lo.dropout))
    modules = lora_modules(model)
    assert [n for n, _ in modules] == ["model.layers.27.self_attn.o_proj"]
    _, m = modules[0]
    assert m.r["default"] == 4 and m.scaling["default"] == 3.0 and m.lora_A["default"].weight.shape == (4, 16)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_lora_params(modules, 4)


def test_sampling_settings_reach_generate():
    seen = {}

    class Capture:
        device = torch.device("cpu")
        generation_config = GenerationConfig(eos_token_id=EOS_IDS)

        def generate(self, input_ids, attention_mask, **kwargs):
            seen.update(kwargs)
            n = kwargs["num_return_sequences"]
            return torch.cat([input_ids.repeat_interleave(n, 0), torch.full((len(input_ids) * n, 1), EOS_IDS[0])], 1)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
    generate_rollouts(Capture(), tok, ["a"], G=3, max_new_tokens=5, temperature=0.5)
    assert (seen["temperature"], seen["num_return_sequences"], seen["max_new_tokens"], seen["do_sample"]) == (0.5, 3, 5, True)


def test_cli_parsing_and_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("qwen_grpo.trainer.train", lambda cfg, resume=None: calls.append(("train", cfg.run.name)))
    monkeypatch.setattr(run, "evaluate_checkpoint", lambda cfg, ckpt, split: calls.append((split, Path(ckpt).name)))
    ckpt = ROOT / "checkpoints" / "grpo_lora_last2_b2_balanced_step30"
    run.main(["train", "--config", str(DEFAULT)])
    assert run.parse(["train", "--config", str(DEFAULT), "--resume", "checkpoints/x_step10"]).resume == Path("checkpoints/x_step10")
    run.main(["validate", "--config", str(DEFAULT), "--checkpoint", str(ckpt)])
    run.main(["test", "--config", str(DEFAULT), "--checkpoint", str(ckpt)])
    assert calls == [("train", "default"), ("validation", ckpt.name), ("test", ckpt.name)]
    with pytest.raises(SystemExit):
        run.parse(["validate", "--config", str(DEFAULT)])  # --checkpoint is required
    with pytest.raises(SystemExit, match="not a LoRA checkpoint"):
        run.main(["test", "--config", str(DEFAULT), "--checkpoint", str(ROOT / "checkpoints")])


def test_verification_settings_load(tmp_path):
    v = config.load(DEFAULT).verification
    assert (v.split, v.merge_subset, v.negative_control.checkpoint, v.negative_control.train) == ("validation", 8, None, False)
    assert (v.statistics.confidence_level, v.statistics.bootstrap_samples, v.statistics.seed) == (0.95, 2000, 42)
    c = config.load(write(tmp_path, lambda r: r["verification"]["negative_control"].update(checkpoint="checkpoints/x_lr0_step30")))
    assert c.verification.negative_control.checkpoint == "checkpoints/x_lr0_step30"


def test_verify_and_report_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "verify", lambda cfg, ckpt: calls.append(("verify", Path(ckpt).name)))
    monkeypatch.setattr(run, "report", lambda ckpt: calls.append(("report", Path(ckpt).name)))
    ckpt = ROOT / "checkpoints" / "grpo_lora_last2_b2_balanced_step30"
    run.main(["verify", "--config", str(DEFAULT), "--checkpoint", str(ckpt)])
    run.main(["report", "--config", str(DEFAULT), "--checkpoint", str(ckpt)])
    assert calls == [("verify", ckpt.name), ("report", ckpt.name)]
    with pytest.raises(SystemExit):
        run.parse(["verify", "--config", str(DEFAULT)])


def test_control_checkpoint_is_none_unless_configured_and_rejects_bad_provenance(tmp_path):
    assert run.control_checkpoint(config.load(DEFAULT)) is None
    cfg = config.load(write(tmp_path, lambda r: r["verification"]["negative_control"].update(
        checkpoint="checkpoints/grpo_lora_lr0_step10")))  # a pre-YAML run: its settings cannot be compared
    with pytest.raises(SystemExit, match="LR=0 control rejected"):
        run.control_checkpoint(cfg)


def test_checkpoint_must_match_config(tmp_path):
    ckpt = ROOT / "checkpoints" / "grpo_lora_last2_b2_balanced_step30"
    verification.check_adapter_matches(config.load(DEFAULT), ckpt)
    other = config.load(write(tmp_path, lambda r: r["lora"].update(targets="last-o-proj")))
    with pytest.raises(SystemExit, match="does not match the config"):
        verification.check_adapter_matches(other, ckpt)




# --- LoRA targeting and evaluation settings ----------------------------------------------------------------------

def tiny(layers):
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=layers,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=8))


@pytest.mark.parametrize("layers", [2, 5, 36])
def test_targets_follow_the_model_depth(layers):
    from qwen_grpo.model import PRESETS
    last = [f"model.layers.{layers - 1}.self_attn.o_proj"]
    assert target_modules(tiny(layers), *PRESETS["last-o-proj"]) == last
    names = target_modules(tiny(layers), ["down_proj", "q_proj"], 2)
    assert names == [f"model.layers.{i}.{m}" for i in (layers - 2, layers - 1) for m in ("self_attn.q_proj", "mlp.down_proj")]


@pytest.mark.parametrize("modules, layers, message", [(["qq_proj"], 1, "no Linear module"), (["q_proj"], 6, "decoder layers"),
                                                       (["q_norm"], 1, "no Linear module")])
def test_bad_targets_are_rejected(modules, layers, message):
    with pytest.raises(ValueError, match=message):
        target_modules(tiny(5), modules, layers)


@pytest.mark.parametrize("preset", ["last-o-proj", "last2-attention"])
def test_preset_lists_give_the_same_adapter_as_the_historical_regexes(preset):
    from qwen_grpo.model import PRESETS, TARGETS, matched_modules
    names = target_modules(tiny(28), *PRESETS[preset])
    assert matched_modules(tiny(28), TARGETS[preset]) == matched_modules(tiny(28), names) == names
    states = []
    for target in (TARGETS[preset], names):
        base = tiny(28)
        torch.manual_seed(42)  # as train() before lora_model: LoRA A init must not depend on how targets are written
        states.append(dict(get_peft_model(base, lora_config(target)).named_parameters()))
    assert states[0].keys() == states[1].keys() and all(torch.equal(states[0][k], states[1][k]) for k in states[0])


def test_custom_targets_load_and_old_yaml_without_new_keys_gets_historical_defaults(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: r["lora"].update(targets={"modules": ["q_proj", "o_proj"], "num_layers": 3})))
    assert cfg.lora.spec() == (("q_proj", "o_proj"), 3)
    assert config.as_dict(cfg)["lora"]["targets"] == {"modules": ["q_proj", "o_proj"], "num_layers": 3}
    old = config.load(write(tmp_path, lambda r: (r.pop("evaluation", None), r["run"].pop("gpu_budget_gib", None))))
    assert (old.evaluation.max_new_tokens, old.evaluation.batch_size, old.evaluation.score_batch_size) == (4, 32, 8)
    assert (old.evaluation.bootstrap_samples, old.evaluation.bootstrap_seed, old.run.gpu_budget_gib) == (10_000, 42, 4.0)


def test_evaluation_settings_reach_generation(tmp_path, monkeypatch):
    from qwen_grpo import evaluation
    seen = []
    monkeypatch.setattr(evaluation, "generate", lambda model, tok, s, max_new: seen.append((len(s), max_new)) or [("X", 1, False)] * len(s))
    cfg = config.load(write(tmp_path, lambda r: r.update(evaluation={"batch_size": 3, "max_new_tokens": 7})))
    gen, score = evaluation.settings(cfg.evaluation)
    evaluation.evaluate(None, None, [{"id": i, "text": "t", "label": "POSITIVE"} for i in range(7)], **gen)
    assert seen == [(3, 7), (3, 7), (1, 7)] and score == {"batch": 8}


def test_example_configs_load_with_distinct_run_names():
    paths = sorted((ROOT / "configs").rglob("*.yaml"))
    names = [config.load(p).run.name for p in paths]
    assert len(paths) >= 5 and len(set(names)) == len(names), names


# --- defaults, inheritance, hashing ------------------------------------------------------------------------------

REQUIRED = {"run": {"name": "x"}, "model": {"name": "Qwen/Qwen3-0.6B"},
            "lora": {"rank": 8, "alpha": 16, "targets": "last2-attention"},
            "data": {"train_samples": 32, "val_subset": 160},
            "training": {k: RAW["training"][k] for k in ("batch_size", "batch_mode", "group_size", "learning_rate", "max_steps",
                                                         "max_attempts", "max_new_tokens", "seed", "eval_every", "full_val")}}


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw if isinstance(raw, str) else yaml.safe_dump(raw))
    return path


def test_only_the_required_keys_give_the_default_config(tmp_path):
    cfg = config.load(put(tmp_path / "c.yaml", REQUIRED))
    assert config.as_dict(cfg) == {**RAW, "run": {**RAW["run"], "name": "x"}}
    assert (cfg.model.dtype, cfg.lora.dropout, cfg.data.subset_seed, cfg.training.clip_eps) == ("bfloat16", 0.0, 42, 0.2)
    assert cfg.verification == config.Verification() and cfg.evaluation == config.Evaluation()


def test_the_full_and_the_minimal_default_yaml_resolve_identically(tmp_path):
    full = config.load(put(tmp_path / "full.yaml", RAW))
    assert full == config.load(DEFAULT) and config.sha256(full) == config.sha256(config.load(DEFAULT))
    assert "verification" not in yaml.safe_load(DEFAULT.read_text())  # the file really relies on defaults


@pytest.mark.parametrize("name, changed", [
    ("grpo_32", {"run.name"}),
    ("grpo_low_lr", {"run.name", "training.learning_rate", "training.max_steps"}),
    ("grpo_1000", {"run.name", "data.train_samples", "data.val_subset", "training.learning_rate", "training.max_steps",
                   "training.max_attempts", "training.eval_every"}),
    ("qwen3_4b", {"run.name", "run.gpu_budget_gib", "model.name", "lora.targets"}),
])
def test_experiments_change_only_the_keys_they_write(name, changed):
    cfg = config.load(EXPERIMENTS / f"{name}.yaml")
    assert set(config.differences(RAW, config.as_dict(cfg))) == changed and cfg.run.name == name


def test_extends_merges_recursively_and_overrides_replace(tmp_path):
    put(tmp_path / "base.yaml", {**RAW, "lora": {**RAW["lora"], "targets": {"modules": ["q_proj", "o_proj"], "num_layers": 2}},
                                 "verification": {**RAW["verification"], "negative_control": {"checkpoint": "c", "train": False}}})
    put(tmp_path / "mid" / "mid.yaml", {"extends": "../base.yaml", "run": {"name": "mid"},
                                        "lora": {"targets": {"num_layers": 3}}, "training": {"learning_rate": 0}})
    put(tmp_path / "exp" / "leaf.yaml", {"extends": "../mid/mid.yaml", "run": {"name": "leaf"},
                                         "verification": {"negative_control": {"checkpoint": None}}})
    cfg = config.load(tmp_path / "exp" / "leaf.yaml")
    assert cfg.lora.spec() == (("q_proj", "o_proj"), 3)  # nested mapping merged key by key
    assert cfg.training.learning_rate == 0.0 and cfg.training.batch_size == RAW["training"]["batch_size"]
    assert cfg.verification.negative_control.checkpoint is None  # an explicit null overrides
    put(tmp_path / "exp" / "preset.yaml", {"extends": "../mid/mid.yaml", "run": {"name": "p"}, "lora": {"targets": "last-o-proj"}})
    assert config.load(tmp_path / "exp" / "preset.yaml").lora.targets == "last-o-proj"  # a mapping replaced by a string
    assert config.merge({"a": {"b": [1, 2], "c": 1}}, {"a": {"b": [3]}}) == {"a": {"b": [3], "c": 1}}  # lists replace


@pytest.mark.parametrize("files, message", [
    ({"c.yaml": {"extends": "missing.yaml", "run": {"name": "c"}}}, "is not a YAML file"),
    ({"c.yaml": {"extends": ["d.yaml"], "run": {"name": "c"}}}, "is not a YAML file"),
    ({"c.yaml": {"extends": ".", "run": {"name": "c"}}}, "is not a YAML file"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "c"}}, "d.yaml": {"extends": "c.yaml", "run": {"name": "d"}}},
     "circular extends: c.yaml -> d.yaml -> c.yaml"),
    ({"c.yaml": {"extends": "c.yaml", "run": {"name": "c"}}}, "circular extends"),
    ({"c.yaml": {"extends": "d.yaml", "training": {"max_steps": 3}}, "d.yaml": RAW}, "must set its own run.name"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "default"}}, "d.yaml": RAW}, "must set its own run.name"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "c"}, "training": {"lr": 1.0}}, "d.yaml": RAW}, "unknown keys ['lr']"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "c"}, "training": {"batch_size": 3}}, "d.yaml": RAW}, "balanced needs"),
    ({"c.yaml": "[1, 2]"}, "expected a mapping"),
])
def test_bad_inheritance_is_rejected(tmp_path, files, message):
    for name, raw in files.items():
        put(tmp_path / name, raw)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        config.load(tmp_path / "c.yaml")


def test_config_hash_is_order_independent_and_matches_saved_configs(tmp_path):
    cfg = config.load(DEFAULT)
    reordered = put(tmp_path / "r.yaml", {k: dict(reversed(list(v.items()))) for k, v in reversed(list(RAW.items()))})
    assert config.sha256(config.load(reordered)) == config.sha256(cfg)
    assert config.sha256(dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, seed=43))) != config.sha256(cfg)
    old = copy.deepcopy(RAW)  # a training log's yaml_config from before evaluation/gpu_budget_gib existed
    old.pop("evaluation"), old["run"].pop("gpu_budget_gib")
    assert config._build(config.Config, old, "saved") == cfg and config.normalized(old) == RAW


def test_differences_lists_dotted_keys():
    other = copy.deepcopy(RAW)
    other["training"]["seed"], other["lora"]["targets"] = 1, {"modules": ["q_proj"], "num_layers": 1}
    assert config.differences(RAW, other) == ["lora.targets", "training.seed"]
    assert config.differences(RAW, copy.deepcopy(RAW)) == []


def test_cli_loads_experiment_configs(monkeypatch):
    calls = []
    monkeypatch.setattr("qwen_grpo.trainer.train", lambda cfg, resume=None: calls.append(cfg))
    run.main(["train", "--config", str(EXPERIMENTS / "grpo_low_lr.yaml")])
    assert (calls[0].run.name, calls[0].training.learning_rate, calls[0].training.max_steps) == ("grpo_low_lr", 1e-5, 200)
