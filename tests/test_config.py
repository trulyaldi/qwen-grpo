"""YAML config: strict loading, validation, wiring of values into the implementation, and CLI dispatch (no GPU)."""
import copy
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
RAW = yaml.safe_load(DEFAULT.read_text())


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
    (lambda r: r["verification"]["statistics"].pop("seed"), "missing keys ['seed']"),
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


def test_train_refuses_to_overwrite_existing_outputs(tmp_path):
    from qwen_grpo.trainer import train

    cfg = config.load(write(tmp_path, lambda r: r["run"].update(name="grpo_lora_last2_b2_balanced")))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        train(cfg)  # raises before loading data or the model


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


def test_example_configs_load():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config.load(path)
