"""YAML config: strict loading, defaults, inheritance, hashing, validation, historical configs, wiring of values into
the implementation, the shuffled sampler, and CLI dispatch (no GPU)."""
import copy
import json
import dataclasses
import itertools
import random
import re
from pathlib import Path

import pytest
import torch
import yaml
from peft import get_peft_model
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from qwen_grpo import config, verification
from qwen_grpo.model import EOS_IDS, expected_lora_params, lora_config, lora_modules, target_modules
from qwen_grpo.policy import generate_rollouts
from qwen_grpo.training import shuffled_batches
from scripts import run

ROOT = Path(__file__).parents[1]
QWEN = ROOT / "configs" / "qwen.yaml"
DEFAULT = ROOT / "configs" / "default.yaml"  # historical schema (balanced B=2), kept as a record
EXPERIMENTS = ROOT / "configs" / "experiments"
STEP90 = ROOT / "runs" / "qwen" / "checkpoints" / "step90"
RAW = config.as_dict(config.load(QWEN))  # every key written out, current schema


def write(tmp_path, edit, raw=RAW):
    raw = copy.deepcopy(raw)
    edit(raw)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_reference_config_loads_with_the_successful_run_values():
    cfg = config.load(QWEN)
    assert (cfg.model.name, cfg.model.dtype) == ("Qwen/Qwen3-0.6B", "bfloat16")
    assert (cfg.lora.rank, cfg.lora.alpha, cfg.lora.targets) == (32, 64, "last2-attention")
    t = cfg.training
    assert (t.batch_size, t.group_size, t.learning_rate, t.max_steps, t.max_new_tokens) == (5, 6, 5e-4, 100, 4)
    assert (t.clip_eps, t.num_iterations, t.micro_batch_size, t.legacy_batch_mode, t.seed) == (0.2, 1, None, None, 42)
    assert (cfg.data.train_samples, cfg.data.val_subset, cfg.data.subset_seed) == (512, 512, 42)
    assert (cfg.evaluation.every, cfg.evaluation.full_validation) == (10, True)


@pytest.mark.parametrize("edit, message", [
    (lambda r: r.update(extra=1), "unknown keys ['extra']"),
    (lambda r: r["training"].update(lr=1e-4), "unknown keys ['lr']"),
    (lambda r: r["lora"].pop("rank"), "missing keys ['rank']"),
    (lambda r: r.pop("data"), "missing keys ['data']"),
    (lambda r: r.pop("evaluation"), "missing keys ['evaluation']"),
    (lambda r: r["evaluation"].pop("every"), "missing keys ['every']"),
    (lambda r: r["training"].update(learning_rate="1e-4"), "expected float, got str"),
    (lambda r: r["training"].update(group_size=True), "expected int, got bool"),
    (lambda r: r["training"].update(group_size=6.0), "expected int, got float"),
    (lambda r: r["training"].update(micro_batch_size=6.0), "got float"),
    (lambda r: r["evaluation"].update(full_validation=1), "expected bool, got int"),
    (lambda r: r.update(lora=[8, 16]), "expected a mapping"),
    (lambda r: r["verification"].update(models=["base"]), "unknown keys ['models']"),
    (lambda r: r["verification"].update(control_checkpoint=3), "got int"),
    (lambda r: r["training"].pop("seed"), "missing keys ['seed']"),
    (lambda r: r["lora"].update(targets={"modules": ["q_proj"]}), "missing keys ['num_layers']"),
    (lambda r: r["lora"].update(targets={"modules": "q_proj", "num_layers": 1}), "expected list, got str"),
    (lambda r: r["lora"].update(targets=3), "lora.targets: expected"),
    (lambda r: r["evaluation"].update(max_new=4), "unknown keys ['max_new']"),
])
def test_unknown_missing_and_ill_typed_keys_are_rejected(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        config.load(write(tmp_path, edit))


@pytest.mark.parametrize("edit, message", [
    (lambda r: r["model"].update(name="meta-llama/Llama-3.1-8B"), "model.name"),
    (lambda r: r["model"].update(name="Qwen/Qwen3-30B-A3B"), "model.name"),
    (lambda r: r["model"].update(dtype="float16"), "model.dtype"),
    (lambda r: r["lora"].update(targets="all"), "lora.targets"),
    (lambda r: r["lora"].update(targets={"modules": [], "num_layers": 2}), "lora.targets.modules"),
    (lambda r: r["lora"].update(targets={"modules": ["q_proj", "q_proj"], "num_layers": 2}), "lora.targets.modules"),
    (lambda r: r["lora"].update(targets={"modules": ["q_proj"], "num_layers": 0}), "lora.targets.num_layers"),
    (lambda r: r["evaluation"].update(bootstrap_samples=1500), "evaluation.bootstrap_samples"),
    (lambda r: r["evaluation"].update(batch_size=0), "evaluation: token and batch sizes"),
    (lambda r: r["evaluation"].update(every=0), "evaluation.every"),
    (lambda r: r["run"].update(gpu_budget_gib=0.0), "run.gpu_budget_gib"),
    (lambda r: r["training"].update(batch_size=0), "training.batch_size"),
    (lambda r: r["training"].update(batch_size=513), "training.batch_size"),
    (lambda r: r["training"].update(group_size=1), "training.group_size"),
    (lambda r: r["training"].update(max_steps=1001), "max_steps <= max_attempts * num_iterations"),
    (lambda r: r["training"].update(learning_rate=-1.0), "training.learning_rate"),
    (lambda r: r["training"].update(clip_eps=1.0), "training.clip_eps"),
    (lambda r: r["training"].update(num_iterations=0), "training.num_iterations"),
    (lambda r: r["training"].update(micro_batch_size=0), "training.micro_batch_size"),
    (lambda r: r["training"].update(micro_batch_size=31), "batch_size * group_size = 30"),
    (lambda r: r["training"].update(max_new_tokens=0), "training.max_new_tokens"),
    (lambda r: r["data"].update(train_samples=31), "data.train_samples"),
    (lambda r: r["run"].update(name="../escape"), "run.name"),
    (lambda r: r["verification"].update(split="train"), "verification.split"),
    (lambda r: r["verification"].update(control_checkpoint="c", train_control=True), "not both"),
])
def test_unsupported_values_are_rejected(tmp_path, edit, message):
    with pytest.raises(ValueError, match="invalid config") as e:
        config.load(write(tmp_path, edit))
    assert message in str(e.value)


def test_lr_zero_float32_any_batch_size_and_micro_batches_are_accepted(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: (r["training"].update(learning_rate=0, batch_size=3, micro_batch_size=7),
                                                  r["model"].update(dtype="float32"))))
    assert cfg.training.learning_rate == 0.0 and isinstance(cfg.training.learning_rate, float)
    assert config.DTYPES[cfg.model.dtype] is torch.float32
    assert (cfg.training.batch_size, cfg.training.micro_batch_size) == (3, 7)  # 7 does not divide B*G = 18: allowed


def test_max_steps_may_use_every_update_of_reused_batches(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: r["training"].update(max_steps=30, max_attempts=10, num_iterations=3)))
    assert cfg.training.max_steps == 30


def test_lora_settings_reach_the_adapter(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: r["lora"].update(rank=4, alpha=12, targets="last-o-proj")))
    torch.manual_seed(0)
    tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=28,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    lo = cfg.lora
    model = get_peft_model(tiny, lora_config(target_modules(tiny, *lo.spec()), lo.rank, lo.alpha))
    modules = lora_modules(model)
    assert [n for n, _ in modules] == ["model.layers.27.self_attn.o_proj"]
    _, m = modules[0]
    assert m.r["default"] == 4 and m.scaling["default"] == 3.0 and m.lora_A["default"].weight.shape == (4, 16)
    assert m.lora_dropout["default"].__class__.__name__ == "Identity"  # no dropout layer at all
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
    generate_rollouts(Capture(), tok, ["a"], G=3, max_new_tokens=5)
    assert (seen["temperature"], seen["top_p"], seen["top_k"]) == (1.0, 1.0, 0)  # pure sampling: what compute_logprobs scores
    assert (seen["num_return_sequences"], seen["max_new_tokens"], seen["do_sample"]) == (3, 5, True)


# --- historical configs ------------------------------------------------------------------------------------------

def test_the_saved_config_of_the_reference_run_upgrades_to_the_reference_yaml():
    """runs/qwen/config.yaml (historical schema, batch_mode unconstrained) is the same experiment as configs/qwen.yaml."""
    saved = config.load(ROOT / "runs" / "qwen" / "config.yaml", historical=True)
    assert saved == config.load(QWEN) and config.sha256(saved) == config.sha256(config.load(QWEN))


def test_new_runs_reject_the_historical_schema_with_the_offending_keys(tmp_path):
    for path in (ROOT / "runs" / "qwen" / "config.yaml", DEFAULT, EXPERIMENTS / "grpo_clip_k3.yaml"):
        with pytest.raises(ValueError, match="historical config schema"):
            config.load(path)
    path = write(tmp_path, lambda r: r["training"].update(batch_mode="unconstrained"))
    with pytest.raises(ValueError, match=r"\['training.batch_mode'\] belong to the historical config schema"):
        config.load(path)
    with pytest.raises(ValueError, match="training.legacy_batch_mode"):  # set by upgrade() only
        config.load(write(tmp_path, lambda r: r["training"].update(legacy_batch_mode="balanced", batch_size=2)))


HISTORICAL = yaml.safe_load((ROOT / "runs" / "qwen" / "config.yaml").read_text())  # every historical key, written out


@pytest.mark.parametrize("key, value", [("model.thinking", True), ("lora.dropout", 0.1), ("training.temperature", 0.7),
                                        ("evaluation.bootstrap_seed", 7), ("verification.merge_subset", 16),
                                        ("verification.statistics.seed", 0), ("model.thinking", 0)])
def test_historical_settings_the_code_no_longer_implements_are_refused(key, value):
    raw = copy.deepcopy(HISTORICAL)
    *parents, last = key.split(".")
    d = raw
    for p in parents:
        d = d[p]
    d[last] = value
    with pytest.raises(ValueError, match=f"{key}: .* the current code does not implement"):
        config.upgrade(raw)


def test_historical_keys_move_to_their_current_names():
    raw = copy.deepcopy(HISTORICAL)
    raw["training"].update(eval_every=7, full_val=False)
    raw["verification"]["negative_control"] = {"checkpoint": "runs/x/checkpoints/step3", "train": False}
    raw["lora"]["dropout"] = 0  # YAML int for the float 0.0
    cfg = config.normalized(raw)
    assert (cfg["evaluation"]["every"], cfg["evaluation"]["full_validation"]) == (7, False)
    assert cfg["verification"] == {"split": "validation", "control_checkpoint": "runs/x/checkpoints/step3", "train_control": False}
    assert "legacy_batch_mode" not in cfg["training"] and config.upgrade(config.upgrade(raw)) == config.upgrade(raw)


@pytest.mark.parametrize("mode, B", [("balanced", 2), ("cycle", 1)])
def test_historical_batch_modes_stay_readable_and_distinct_but_cannot_be_trained(tmp_path, monkeypatch, mode, B):
    raw = copy.deepcopy(HISTORICAL)
    raw["training"].update(batch_mode=mode, batch_size=B)
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = config.load(path, historical=True)
    assert (cfg.training.legacy_batch_mode, cfg.training.batch_size) == (mode, B)
    assert config.as_dict(cfg)["training"]["legacy_batch_mode"] == mode
    shuffled = config.normalized({**HISTORICAL, "training": {**HISTORICAL["training"], "batch_size": B}})
    assert config.differences(config.normalized(raw), shuffled) == ["training.legacy_batch_mode"]
    assert not verification.comparable(raw, shuffled, set())  # a balanced control never vouches for a shuffled run
    from qwen_grpo import runs, trainer
    monkeypatch.setattr(runs, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match=f"historical batch_mode '{mode}'.*cannot be trained or resumed"):
        trainer.train(cfg, device="cpu")
    assert not (tmp_path / "runs").exists()  # refused before creating anything


def test_historical_configs_in_the_repository_still_load_as_records():
    default = config.load(DEFAULT, historical=True)
    assert (default.training.legacy_batch_mode, default.training.batch_size, default.lora.rank) == ("balanced", 2, 8)
    names = [config.load(p, historical=True).run.name for p in sorted(EXPERIMENTS.glob("*.yaml"))]
    assert names == ["grpo_1000", "grpo_32", "grpo_clip_k3", "grpo_low_lr", "qwen3_4b"]
    k3 = config.load(EXPERIMENTS / "grpo_clip_k3.yaml", historical=True)
    assert config.differences(config.as_dict(default), config.as_dict(k3)) == ["run.name", "training.num_iterations"]


# --- the shuffled sampler ----------------------------------------------------------------------------------------

def historical_unconstrained(train, B, seed=42):
    """training.batches(train, B, "unconstrained", seed) as of commit 9bd4500, verbatim."""
    def stream(pool, rng):
        while True:
            order = list(pool)
            rng.shuffle(order)
            yield from order
    s = stream(train, random.Random(seed))
    return ([next(s) for _ in range(B)] for _ in itertools.count())


@pytest.mark.parametrize("n, B, seed", [(512, 5, 42), (4, 3, 0), (6, 1, 7), (10, 10, 1)])
def test_shuffled_sampler_is_the_historical_unconstrained_stream(n, B, seed):
    train = [{"id": i} for i in range(n)]
    epochs = 3
    k = epochs * n // B + 2  # past the third epoch boundary
    new = list(itertools.islice(shuffled_batches(train, B, seed), k))
    assert new == list(itertools.islice(historical_unconstrained(train, B, seed), k))
    flat = [e["id"] for b in new for e in b]
    for e in range(epochs):  # every epoch is a permutation of the subset: sampling without replacement
        assert sorted(flat[e * n:(e + 1) * n]) == list(range(n))
    assert flat[:n] != flat[n:2 * n] or n == 1  # reshuffled after each pass


def test_shuffled_sampler_replay_lands_where_an_uninterrupted_stream_is():
    """Resume replays `attempts` batches from a fresh stream: it must continue exactly where the run was."""
    train = [{"id": i} for i in range(512)]
    run_ = shuffled_batches(train, 5, 42)
    seen = [next(run_) for _ in range(142)]
    replay = shuffled_batches(train, 5, 42)
    assert [next(replay) for _ in range(142)] == seen
    assert [next(replay) for _ in range(30)] == [next(run_) for _ in range(30)]
    assert shuffled_batches(train, 5, 43).__next__() != seen[0]  # the seed matters


def test_reference_run_rollout_prompts_follow_the_shuffled_sampler():
    """The prompt ids logged by runs/qwen (zero-signal batches included) are the shuffled stream's, batch for batch."""
    import json

    from qwen_grpo.sentiment import load_splits
    from qwen_grpo.training import balanced_subset
    log = json.loads((ROOT / "runs" / "qwen" / "training" / "training.json").read_text())
    by_attempt = {g["attempt"]: g["prompt_ids"] for g in log["training"] + log["zero_signal"]}
    train = balanced_subset(load_splits()["train"], 512, 42)
    stream = shuffled_batches(train, 5, 42)
    assert [[e["id"] for e in next(stream)] for _ in range(len(by_attempt))] == [by_attempt[a] for a in sorted(by_attempt)]


# --- CLI ---------------------------------------------------------------------------------------------------------

def test_cli_parsing_and_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr("qwen_grpo.trainer.train", lambda cfg, resume=None: calls.append(("train", cfg.run.name)))
    monkeypatch.setattr(run, "evaluate_checkpoint", lambda cfg, ckpt, split: calls.append((split, Path(ckpt).name)))
    run.main(["train", "--config", str(QWEN)])
    assert run.parse(["train", "--config", str(QWEN), "--resume", "runs/x/checkpoints/step10"]).resume == Path("runs/x/checkpoints/step10")
    saved = str(ROOT / "runs" / "qwen" / "config.yaml")
    run.main(["validate", "--config", saved, "--checkpoint", str(STEP90)])  # a historical config evaluates
    run.main(["test", "--config", str(QWEN), "--checkpoint", str(STEP90)])
    run.main(["train", "--config", saved, "--resume", str(STEP90)])  # resuming accepts the run's saved config
    assert calls == [("train", "qwen"), ("validation", "step90"), ("test", "step90"), ("train", "qwen")]
    with pytest.raises(ValueError, match="historical config schema"):
        run.main(["train", "--config", saved])  # a new run does not
    with pytest.raises(SystemExit):
        run.parse(["validate", "--config", str(QWEN)])  # --checkpoint is required
    with pytest.raises(SystemExit, match="not a LoRA checkpoint"):
        run.main(["test", "--config", str(QWEN), "--checkpoint", str(ROOT / "runs" / "qwen")])


def test_verification_settings_load(tmp_path):
    v = config.load(QWEN).verification
    assert (v.split, v.control_checkpoint, v.train_control) == ("validation", None, False)
    c = config.load(write(tmp_path, lambda r: r["verification"].update(control_checkpoint="runs/x_lr0/checkpoints/step30")))
    assert c.verification.control_checkpoint == "runs/x_lr0/checkpoints/step30"


def test_verify_and_report_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(run, "verify", lambda cfg, ckpt: calls.append(("verify", Path(ckpt).name)))
    monkeypatch.setattr(run, "report", lambda ckpt: calls.append(("report", Path(ckpt).name)))
    run.main(["verify", "--config", str(QWEN), "--checkpoint", str(STEP90)])
    run.main(["report", "--config", str(QWEN), "--checkpoint", str(STEP90)])
    assert calls == [("verify", "step90"), ("report", "step90")]
    with pytest.raises(SystemExit):
        run.parse(["verify", "--config", str(QWEN)])


def test_control_checkpoint_is_none_unless_configured_and_rejects_bad_provenance(tmp_path):
    assert run.control_checkpoint(config.load(QWEN)) is None
    cfg = config.load(write(tmp_path, lambda r: r["verification"].update(control_checkpoint="checkpoints/grpo_lora_lr0_step10")))
    with pytest.raises(SystemExit, match="LR=0 control rejected"):
        run.control_checkpoint(cfg)


def test_checkpoint_must_match_config(tmp_path):
    verification.check_adapter_matches(config.load(QWEN), STEP90)
    other = config.load(write(tmp_path, lambda r: r["lora"].update(targets="last-o-proj")))
    with pytest.raises(SystemExit, match="does not match the config"):
        verification.check_adapter_matches(other, STEP90)


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


def test_custom_targets_load_and_omitted_keys_get_the_documented_defaults(tmp_path):
    cfg = config.load(write(tmp_path, lambda r: r["lora"].update(targets={"modules": ["q_proj", "o_proj"], "num_layers": 3})))
    assert cfg.lora.spec() == (("q_proj", "o_proj"), 3)
    assert config.as_dict(cfg)["lora"]["targets"] == {"modules": ["q_proj", "o_proj"], "num_layers": 3}
    short = config.load(write(tmp_path, lambda r: (r.update(evaluation={"every": 10, "full_validation": True}),
                                                   r["run"].pop("gpu_budget_gib"), r.pop("verification"))))
    ev = short.evaluation
    assert (ev.max_new_tokens, ev.batch_size, ev.score_batch_size, ev.bootstrap_samples) == (4, 32, 8, 10_000)
    assert short.run.gpu_budget_gib == 4.0 and short.verification == config.Verification()


def test_evaluation_settings_reach_generation(tmp_path, monkeypatch):
    from qwen_grpo import evaluation
    seen = []
    monkeypatch.setattr(evaluation, "generate", lambda model, tok, s, max_new: seen.append((len(s), max_new)) or [("X", 1, False)] * len(s))
    cfg = config.load(write(tmp_path, lambda r: r["evaluation"].update(batch_size=3, max_new_tokens=7)))
    gen, score = evaluation.settings(cfg.evaluation)
    evaluation.evaluate(None, None, [{"id": i, "text": "t", "label": "POSITIVE"} for i in range(7)], **gen)
    assert seen == [(3, 7), (3, 7), (1, 7)] and score == {"batch": 8}


def test_current_configs_load_with_distinct_run_names():
    paths = sorted((ROOT / "configs").glob("*.yaml"))
    current = [p for p in paths if p != DEFAULT]
    names = [config.load(p).run.name for p in current]
    assert len(current) >= 2 and len(set(names)) == len(names), names


# --- defaults, inheritance, hashing ------------------------------------------------------------------------------

REQUIRED = {"run": {"name": "x"}, "model": {"name": "Qwen/Qwen3-0.6B"},
            "lora": {"rank": 32, "alpha": 64, "targets": "last2-attention"},
            "data": {"train_samples": 512, "val_subset": 512},
            "training": {k: RAW["training"][k] for k in ("batch_size", "group_size", "learning_rate", "max_steps",
                                                         "max_attempts", "max_new_tokens", "seed")},
            "evaluation": {"every": 10, "full_validation": True}}


def put(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw if isinstance(raw, str) else yaml.safe_dump(raw))
    return path


def test_only_the_required_keys_give_the_reference_config(tmp_path):
    cfg = config.load(put(tmp_path / "c.yaml", REQUIRED))
    assert config.as_dict(cfg) == {**RAW, "run": {**RAW["run"], "name": "x"}}
    assert (cfg.model.dtype, cfg.data.subset_seed, cfg.training.clip_eps, cfg.training.num_iterations) == ("bfloat16", 42, 0.2, 1)
    assert cfg.verification == config.Verification()


def test_the_full_and_the_minimal_reference_yaml_resolve_identically(tmp_path):
    full = config.load(put(tmp_path / "full.yaml", RAW))
    assert full == config.load(QWEN) and config.sha256(full) == config.sha256(config.load(QWEN))
    assert "verification" not in yaml.safe_load(QWEN.read_text())  # the file really relies on defaults


def test_extends_merges_recursively_and_overrides_replace(tmp_path):
    put(tmp_path / "base.yaml", {**RAW, "lora": {**RAW["lora"], "targets": {"modules": ["q_proj", "o_proj"], "num_layers": 2}},
                                 "verification": {**RAW["verification"], "control_checkpoint": "c"}})
    put(tmp_path / "mid" / "mid.yaml", {"extends": "../base.yaml", "run": {"name": "mid"},
                                        "lora": {"targets": {"num_layers": 3}}, "training": {"learning_rate": 0}})
    put(tmp_path / "exp" / "leaf.yaml", {"extends": "../mid/mid.yaml", "run": {"name": "leaf"},
                                         "verification": {"control_checkpoint": None}})
    cfg = config.load(tmp_path / "exp" / "leaf.yaml")
    assert cfg.lora.spec() == (("q_proj", "o_proj"), 3)  # nested mapping merged key by key
    assert cfg.training.learning_rate == 0.0 and cfg.training.batch_size == RAW["training"]["batch_size"]
    assert cfg.verification.control_checkpoint is None  # an explicit null overrides
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
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "qwen"}}, "d.yaml": RAW}, "must set its own run.name"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "c"}, "training": {"lr": 1.0}}, "d.yaml": RAW}, "unknown keys ['lr']"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "c"}, "training": {"micro_batch_size": 31}}, "d.yaml": RAW}, "micro_batch_size"),
    ({"c.yaml": {"extends": "d.yaml", "run": {"name": "c"}, "training": {"eval_every": 5}}, "d.yaml": RAW}, "historical config schema"),
    ({"c.yaml": "[1, 2]"}, "expected a mapping"),
])
def test_bad_inheritance_is_rejected(tmp_path, files, message):
    for name, raw in files.items():
        put(tmp_path / name, raw)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        config.load(tmp_path / "c.yaml")


def test_config_hash_is_order_independent(tmp_path):
    cfg = config.load(QWEN)
    reordered = put(tmp_path / "r.yaml", {k: dict(reversed(list(v.items()))) for k, v in reversed(list(RAW.items()))})
    assert config.sha256(config.load(reordered)) == config.sha256(cfg)
    assert config.sha256(dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, seed=43))) != config.sha256(cfg)
    assert config.sha256(dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, micro_batch_size=6))) != config.sha256(cfg)
    assert config.normalized(RAW) == RAW


def test_differences_lists_dotted_keys():
    other = copy.deepcopy(RAW)
    other["training"]["seed"], other["lora"]["targets"] = 1, {"modules": ["q_proj"], "num_layers": 1}
    assert config.differences(RAW, other) == ["lora.targets", "training.seed"]
    assert config.differences(RAW, copy.deepcopy(RAW)) == []


def test_cli_loads_configs_that_extend_the_reference(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("qwen_grpo.trainer.train", lambda cfg, resume=None: calls.append(cfg))
    put(tmp_path / "low_lr.yaml", {"extends": str(QWEN.relative_to(tmp_path, walk_up=True)), "run": {"name": "low_lr"},
                                   "training": {"learning_rate": 1.0e-5, "max_steps": 200}})
    run.main(["train", "--config", str(tmp_path / "low_lr.yaml")])
    assert (calls[0].run.name, calls[0].training.learning_rate, calls[0].training.max_steps) == ("low_lr", 1e-5, 200)
    assert config.differences(RAW, config.as_dict(calls[0])) == ["run.name", "training.learning_rate", "training.max_steps"]


# --- execution, streaming and dflash sections (Ticket 01) ------------------------------------------------------------

L2L = {"backend": "l2l_async", "weight_storage": "pinned", "resident_layers": 0, "weight_buffers": 2,
       "prefetch_depth": 1, "activation_checkpointing": True}


def with_section(name, values):
    return lambda r: r.update({name: dict(values)})


def test_valid_execution_settings_load(tmp_path):
    cfg = config.load(write(tmp_path, with_section("execution", L2L)))
    assert cfg.execution.backend == "l2l_async" and cfg.execution.weight_buffers == 2
    sync = config.load(write(tmp_path, with_section("execution", {**L2L, "backend": "l2l_sync", "prefetch_depth": 0,
                                                                  "weight_buffers": 1})))
    assert sync.execution.prefetch_depth == 0


@pytest.mark.parametrize("execution, message", [
    # backend
    ({"backend": "l2l"}, "execution.backend: one of"),
    ({"backend": "resident", "weight_buffers": 2}, "only apply to l2l backends"),
    ({"backend": "resident", "kv_cache": "cpu"}, "resident generation"),
    ({"backend": "l2l_sync"}, f"needs {list(config.L2L_ONLY)}"),
    # memory
    ({**L2L, "weight_storage": "disk"}, "execution.weight_storage: one of"),
    ({**L2L, "weight_buffers": 0}, "execution.weight_buffers: >= 1"),
    ({**L2L, "resident_layers": -1}, "execution.resident_layers: >= 0"),
    ({**L2L, "kv_cache": "nvme"}, "execution.kv_cache: one of"),
    ({**L2L, "vocab_chunk_size": 0}, "execution.vocab_chunk_size"),
    ({**L2L, "activation_checkpointing": 1}, "got int 1"),
    # prefetch
    ({**L2L, "backend": "l2l_sync"}, "prefetch_depth: 0 for l2l_sync"),
    ({**L2L, "prefetch_depth": 0}, "prefetch_depth: >= 1 for l2l_async"),
    ({**L2L, "prefetch_depth": 2, "weight_buffers": 2}, "weight_buffers: >= prefetch_depth + 1"),
    ({**L2L, "prefetch_depth": 1.0}, "got float 1.0"),
    ({**L2L, "prefetch": 1}, "unknown keys ['prefetch']"),
])
def test_invalid_execution_settings_are_rejected(tmp_path, execution, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        config.load(write(tmp_path, with_section("execution", execution)))


@pytest.mark.parametrize("section, values, message", [
    ("streaming", {"enabled": True}, "enabled requires directory"),
    ("streaming", {"directory": "data/stream"}, "directory requires enabled"),
    ("dflash", {"enabled": True}, "enabled requires draft_model"),
    ("dflash", {"enabled": True, "draft_model": "z-lab/other"}, "dflash.draft_model: one of"),
    # incompatible identities: the drafter was trained for Qwen3-4B, the config's model is Qwen3-0.6B
    ("dflash", {"enabled": True, "draft_model": "z-lab/Qwen3-4B-DFlash-b16"}, "drafts for Qwen/Qwen3-4B"),
    ("dflash", {"enabled": True, "draft_model": "x", "block_size": 16}, "unknown keys ['block_size']"),
])
def test_invalid_streaming_and_dflash_settings_are_rejected(tmp_path, section, values, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        config.load(write(tmp_path, with_section(section, values)))


def test_dflash_loads_only_for_its_target_model(tmp_path):
    def edit(r):
        r["model"]["name"] = "Qwen/Qwen3-4B"
        r["dflash"] = {"enabled": True, "draft_model": "z-lab/Qwen3-4B-DFlash-b16"}
    assert config.load(write(tmp_path, edit)).dflash.enabled


def test_existing_config_hashes_are_unchanged_and_execution_settings_change_the_hash(tmp_path):
    recorded = json.loads((ROOT / "research" / "l2l" / "baseline.json").read_text())["configs"]  # Ticket 00, before Ticket 01
    for path, entry in recorded.items():
        assert config.sha256(config.load(ROOT / path)) == entry["sha256"], path
    explicit = config.load(write(tmp_path, with_section("execution", {"backend": "resident"})))
    assert config.sha256(explicit) == config.sha256(config.load(QWEN))         # writing the defaults out changes nothing
    assert config.sha256(config.load(write(tmp_path, with_section("execution", L2L)))) != config.sha256(explicit)
    old_log = {k: v for k, v in RAW.items() if k not in config.OPTIONAL_SECTIONS}  # a log saved before the sections
    assert verification.comparable(old_log, RAW, set())


def test_unimplemented_settings_are_refused_before_anything_is_written(tmp_path):
    from qwen_grpo import trainer

    still = {**L2L, "resident_layers": 1}  # l2l_async itself runs since Ticket 08; partial residency does not yet
    for section, values in [("execution", still), ("execution", {"vocab_chunk_size": 4096})]:  # streaming: Ticket 09
        cfg = config.load(write(tmp_path, with_section(section, values)))
        assert config.unimplemented(cfg)
        with pytest.raises(SystemExit, match="not implemented yet"):
            trainer.train(cfg)                                       # conftest: runs/ must stay untouched
    path = write(tmp_path, with_section("execution", still))
    with pytest.raises(SystemExit, match="not implemented yet"):
        run.main(["verify", "--config", str(path), "--checkpoint", str(tmp_path / "missing")])
    assert config.unimplemented(config.load(QWEN)) == []
