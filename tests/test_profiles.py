"""The configuration layout: configs/profiles/shared.yaml (abstract) -> per-model profiles -> experiments.

Pins (1) that introducing the profiles and the memory section changed no existing resolved config or hash, (2) that
the standalone reference configs equal their profile, (3) that the smoke runs exercise the final run's batch shape,
(4) the new validation rules (abstract bases, memory, unsupported combinations), and (5) that the backend comparisons
differ in execution only and that float32 appears only in configs labelled as diagnostics.
configs/default.yaml and configs/experiments/qwen3_4b.yaml were deleted in the cleanup (duplicates); the Ticket 00 hash
of default.yaml is still checked in tests/test_config.py."""
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
    "qwen.yaml": "3c9cd5190cb8835d0b03e2a4ab70a79ab48658571a2ccc9ee7f06f47743da2a4",
    "historical/qwen_l2l.yaml": "a13007a61c0599df048fe975d8dbe04dd97bb838bb0de45a391667039b197084",  # was experiments/
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
    for rel in ("qwen.yaml",):
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
    assert not final.dflash.enabled  # the final run's DFlash decision is open (H10 untested)


def test_local_experiment_runs_the_l2l_path_on_the_small_model():
    local = load("experiments/qwen3_0.6b_streaming.yaml")
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
    p = write_variant(tmp_path, "experiments/qwen3_0.6b_streaming.yaml",
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
    cfg = load("qwen.yaml")
    assert cfg.memory == config.Memory() and "memory" in config.as_dict(cfg)
    assert config.sha256(cfg) == HASHES_BEFORE["qwen.yaml"]


TRIOS = {  # backend comparison: reference first; what else may differ besides execution.* and run.name
    "qwen3_0.6b": (["qwen3_0.6b_resident", "qwen3_0.6b_l2l_sync", "qwen3_0.6b_l2l_async"], set()),
    "kaggle_06b_fp32": (["kaggle_06b_fp32_resident", "kaggle_06b_fp32_l2l_sync", "kaggle_06b_fp32_l2l_async"], set()),
    "qwen3_4b": (["qwen3_4b_resident", "qwen3_4b_l2l_sync", "qwen3_4b_l2l"], {"run.gpu_budget_gib"}),  # resident: 16 GiB
}


@pytest.mark.parametrize("trio", sorted(TRIOS))
def test_backend_comparisons_differ_in_execution_only(trio):
    names, extra = TRIOS[trio]
    cfgs = [load(f"experiments/{n}.yaml") for n in names]
    assert [c.execution.backend for c in cfgs] == ["resident", "l2l_sync", "l2l_async"]
    for c in cfgs[1:]:
        d = diff(cfgs[0], c)
        assert d and all(k.startswith("execution.") or k == "run.name" or k in extra for k in d), (c.run.name, d)
        assert c.training == cfgs[0].training and c.data == cfgs[0].data and c.model == cfgs[0].model
        assert c.lora == cfgs[0].lora and c.evaluation == cfgs[0].evaluation and config.unimplemented(c) == []


def test_float32_appears_only_in_configs_labelled_as_diagnostics():
    """No silent precision substitution: every experiment runs bf16 except the kaggle_06b_fp32_* diagnostics, which
    say so in their name (and the T4 has no native bf16)."""
    for p in sorted(EXP.glob("*.yaml")):
        if config.is_abstract(p):
            continue
        assert (config.load(p).model.dtype == "float32") == p.name.startswith("kaggle_06b_fp32_"), p.name
    assert config.is_abstract(EXP / "kaggle_06b_fp32_base.yaml")
