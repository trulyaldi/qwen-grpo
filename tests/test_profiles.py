"""The configuration layout: configs/profiles/shared.yaml (abstract) -> per-model profiles -> experiments.

Pins (1) that introducing the profiles and the memory section changed no existing resolved config or hash, (2) that
the standalone reference configs equal their profile, (3) that the smoke runs exercise the final run's batch shape,
and (4) the new validation rules (abstract bases, memory, unsupported combinations)."""
from pathlib import Path

import pytest
import yaml

from qwen_grpo import config

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"
PROFILES = CONFIGS / "profiles"
EXP = CONFIGS / "experiments"

# sha256 of the resolved configs at commit 3bc4cbe, before configs/profiles/ and the memory section existed
HASHES_BEFORE = {
    "default.yaml": "80b9c9f03414b999c4c67a969567c92d9579d29d1bd299979da7c8bc670db813",
    "qwen.yaml": "3c9cd5190cb8835d0b03e2a4ab70a79ab48658571a2ccc9ee7f06f47743da2a4",
    "experiments/qwen_l2l.yaml": "a13007a61c0599df048fe975d8dbe04dd97bb838bb0de45a391667039b197084",
    "experiments/qwen3_4b.yaml": "5b8156810da75bab7333c1a98afa9f37055edfa80d2dc38d810122d2abf8f13f",
}


def load(rel):
    return config.load(CONFIGS / rel)


def diff(a, b):
    return config.differences(config.as_dict(a), config.as_dict(b))


@pytest.mark.parametrize("rel", sorted(HASHES_BEFORE))
def test_existing_configs_keep_their_hash(rel):
    assert config.sha256(load(rel)) == HASHES_BEFORE[rel]


def test_reference_configs_equal_the_0_6b_profile():
    profile = load("profiles/qwen3_0_6b.yaml")
    for rel in ("default.yaml", "qwen.yaml"):
        assert "extends" not in yaml.safe_load((CONFIGS / rel).read_text()), rel  # copied as a run's config.yaml
        assert diff(profile, load(rel)) == ["run.name"], rel


def test_the_4b_l2l_experiment_is_the_4b_profile():
    assert diff(load("profiles/qwen3_4b.yaml"), load("experiments/qwen3_4b_l2l.yaml")) == ["run.name"]


def test_profiles_are_runnable_and_select_their_backend():
    small, big = load("profiles/qwen3_0_6b.yaml"), load("profiles/qwen3_4b.yaml")
    assert (small.model.name, small.execution.backend, small.run.gpu_budget_gib) == ("Qwen/Qwen3-0.6B", "resident", 4.0)
    assert (big.model.name, big.execution.backend, big.run.gpu_budget_gib) == ("Qwen/Qwen3-4B", "l2l_async", 8.0)
    assert big.memory == config.Memory(guard=True, interval_s=0.5, allocator_cap=True)
    assert big.execution.weight_storage == "pinned" and big.execution.activation_checkpointing is True
    assert config.unimplemented(small) == config.unimplemented(big) == []


def test_shared_settings_live_in_one_place():
    small, big = (config.as_dict(load(f"profiles/{p}.yaml")) for p in ("qwen3_0_6b", "qwen3_4b"))
    for section in ("data", "training", "evaluation", "verification"):
        assert small[section] == big[section], section
    assert (small["lora"]["rank"], small["lora"]["alpha"]) == (big["lora"]["rank"], big["lora"]["alpha"])


def test_smoke_runs_have_the_final_runs_batch_shape():
    final, smoke = load("experiments/qwen3_4b_final.yaml"), load("experiments/qwen3_4b_smoke.yaml")
    assert set(diff(final, smoke)) == {"run.name", "data.train_samples", "data.val_subset", "training.max_steps",
                                       "evaluation.every"}
    assert smoke.streaming.enabled and smoke.memory.guard and smoke.execution == final.execution
    dflash = load("experiments/qwen3_4b_smoke_dflash.yaml")
    assert diff(smoke, dflash) == ["dflash.draft_model", "dflash.enabled", "run.name"]
    assert not final.dflash.enabled  # the final run's DFlash decision is open (docs/status.md)


def test_local_experiment_runs_the_l2l_path_on_the_small_model():
    local = load("experiments/qwen_local.yaml")
    assert local.model.name == "Qwen/Qwen3-0.6B" and local.execution.backend == "l2l_async" and local.streaming.enabled
    assert config.unimplemented(local) == []


def test_the_shared_base_is_abstract(tmp_path):
    with pytest.raises(ValueError, match="abstract shared base"):
        config.load(PROFILES / "shared.yaml")
    assert config.is_abstract(PROFILES / "shared.yaml") and not config.is_abstract(PROFILES / "qwen3_4b.yaml")
    # the marker is not inherited: a child of an abstract base is a normal config
    child = tmp_path / "child.yaml"
    child.write_text(f"extends: {PROFILES / 'shared.yaml'}\nrun: {{name: c}}\nmodel: {{name: Qwen/Qwen3-0.6B}}\n"
                     "lora: {targets: last2-attention}\n")
    assert config.load(child).run.name == "c"
    bad = tmp_path / "bad.yaml"
    bad.write_text("abstract: 1\n")
    with pytest.raises(ValueError, match="abstract must be true or false"):
        config.read(bad)


def write_variant(tmp_path, rel, **sections):
    raw = config.read(CONFIGS / rel)
    for name, values in sections.items():
        raw[name] = {**raw.get(name, {}), **values}
    p = tmp_path / "variant.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def test_the_4b_drafter_is_refused_for_the_small_model(tmp_path):
    p = write_variant(tmp_path, "experiments/qwen_local.yaml",
                      dflash={"enabled": True, "draft_model": "z-lab/Qwen3-4B-DFlash-b16"})
    with pytest.raises(ValueError, match="drafts for Qwen/Qwen3-4B, not model.name 'Qwen/Qwen3-0.6B'"):
        config.load(p)


def test_disabled_activation_checkpointing_is_refused_not_ignored(tmp_path):
    cfg = config.load(write_variant(tmp_path, "profiles/qwen3_4b.yaml", execution={"activation_checkpointing": False}))
    assert any("activation_checkpointing false" in r for r in config.unimplemented(cfg))
    with pytest.raises(SystemExit, match="activation_checkpointing false"):
        config.require_implemented(cfg)


@pytest.mark.parametrize("memory, message", [
    ({"interval_s": 0.0}, "memory.interval_s: > 0"),
    ({"guard": False, "allocator_cap": True}, "allocator_cap needs memory.guard"),
    ({"guard": "yes"}, "memory.guard: expected bool"),
])
def test_invalid_memory_settings_are_errors(tmp_path, memory, message):
    with pytest.raises(ValueError, match=message):
        config.load(write_variant(tmp_path, "profiles/qwen3_4b.yaml", memory=memory))


def test_unknown_memory_keys_are_errors(tmp_path):
    with pytest.raises(ValueError, match=r"memory: unknown keys \['limit_gib'\]"):
        config.load(write_variant(tmp_path, "profiles/qwen3_4b.yaml", memory={"limit_gib": 8.0}))


def test_memory_at_its_defaults_leaves_the_hash_unchanged():
    cfg = load("default.yaml")
    assert cfg.memory == config.Memory() and "memory" in config.as_dict(cfg)
    assert config.sha256(cfg) == HASHES_BEFORE["default.yaml"]


# --- canonical GPU comparison configs (configs/gpu/) ------------------------------------------------------------

GPU = CONFIGS / "gpu"
PAIRS = {"qwen3_0.6b": "Qwen/Qwen3-0.6B", "qwen3_4b": "Qwen/Qwen3-4B"}


@pytest.mark.parametrize("model", sorted(PAIRS))
def test_gpu_configs_differ_only_in_execution(model):
    cfgs = {b: config.load(GPU / f"{model}_{b}.yaml") for b in ("resident", "l2l_sync", "l2l_async")}
    allowed = {"run.name", "run.gpu_budget_gib"} if model == "qwen3_4b" else {"run.name"}
    for b in ("l2l_sync", "l2l_async"):
        changed = set(diff(cfgs["resident"], cfgs[b]))
        assert {k for k in changed if not k.startswith("execution.")} <= allowed, (b, changed)
    assert diff(cfgs["l2l_sync"], cfgs["l2l_async"]) == ["execution.backend", "execution.prefetch_depth", "run.name"]
    for b, c in cfgs.items():
        assert c.model.name == PAIRS[model] and c.run.name == f"{model}_{b}" and c.execution.backend == b
        assert not c.dflash.enabled and not c.streaming.enabled and c.execution.kv_cache == "gpu"
        assert c.execution.vocab_chunk_size is None and config.unimplemented(c) == []
        assert c.training.num_iterations == 1 and c.training.micro_batch_size is None
    x = cfgs["l2l_sync"].execution
    assert (x.weight_storage, x.resident_layers, x.weight_buffers, x.activation_checkpointing) == ("pinned", 0, 2, True)
    assert (x.prefetch_depth, cfgs["l2l_async"].execution.prefetch_depth) == (0, 1)
    r = cfgs["resident"].execution
    assert all(getattr(r, k) is None for k in config.L2L_ONLY)


def test_the_4b_l2l_configs_carry_the_8_gib_guard():
    for b in ("l2l_sync", "l2l_async"):
        c = config.load(GPU / f"qwen3_4b_{b}.yaml")
        assert c.run.gpu_budget_gib == 8.0 and c.memory.guard and c.memory.allocator_cap


def test_every_run_name_in_the_repository_is_unique():
    paths = [p for p in sorted(CONFIGS.rglob("*.yaml")) if "budget" not in p.parts and not config.is_abstract(p)]
    names = [config.load(p, historical=True).run.name for p in paths]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)
