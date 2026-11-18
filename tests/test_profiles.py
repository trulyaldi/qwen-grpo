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


SCALE = {"run.name", "data.train_samples", "data.val_subset", "training.max_steps", "evaluation.every"}
DRAFTER = config.DFlash(enabled=True, draft_model="z-lab/Qwen3-4B-DFlash-b16")


def test_smoke_runs_have_the_final_runs_batch_shape():
    """qwen3_4b_smoke is the plain-L2L baseline, qwen3_4b_smoke_dflash the final run's exact configuration at smoke
    scale (DFlash rollouts); they differ from each other only in the dflash section."""
    final, smoke = load("experiments/qwen3_4b_final.yaml"), load("experiments/qwen3_4b_smoke.yaml")
    dflash = load("experiments/qwen3_4b_smoke_dflash.yaml")
    assert set(diff(final, dflash)) == SCALE
    assert set(diff(final, smoke)) == SCALE | {"dflash.enabled", "dflash.draft_model"}
    assert smoke.streaming.enabled and smoke.memory.guard and smoke.execution == final.execution == dflash.execution
    assert set(diff(smoke, dflash)) == {"dflash.draft_model", "dflash.enabled", "run.name", "training.max_steps"}
    # the final run's micro-batching and evaluation batch are inherited, not re-declared
    assert smoke.training.micro_batch_size == dflash.training.micro_batch_size == final.training.micro_batch_size == 6
    assert smoke.evaluation.batch_size == dflash.evaluation.batch_size == final.evaluation.batch_size == 16


def test_the_final_run_and_its_lr0_control_sample_through_dflash():
    final, lr0 = load("experiments/qwen3_4b_final.yaml"), load("experiments/qwen3_4b_final_lr0.yaml")
    assert final.dflash == lr0.dflash == DRAFTER
    assert final.execution.backend == "l2l_async" and config.unimplemented(final) == config.unimplemented(lr0) == []
    assert lr0.training.learning_rate == 0.0 and lr0.training.max_steps == 30
    assert set(diff(final, lr0)) == {"run.name", "training.learning_rate", "training.max_steps"}
    # untouched by enabling DFlash: objective, reward inputs, adapter, schedule, batch, sampling, data
    t = final.training
    assert (t.learning_rate, t.max_steps, t.batch_size, t.group_size, t.max_new_tokens, t.num_iterations, t.clip_eps,
            final.lora.rank, final.lora.alpha, final.data.train_samples) == (2e-4, 60, 5, 6, 4, 1, 0.2, 32, 64, 1024)
    assert (t.micro_batch_size, t.max_attempts, t.seed, final.data.val_subset) == (6, 1000, 42, 128)
    assert (t.batch_size * t.group_size) // t.micro_batch_size == 5 and (t.batch_size * t.group_size) % t.micro_batch_size == 0
    assert (final.lora.targets.modules, final.lora.targets.num_layers) == (["q_proj", "v_proj", "o_proj"], 2)
    assert (final.model.name, final.model.dtype, final.execution.weight_storage) == ("Qwen/Qwen3-4B", "bfloat16", "pinned")
    assert final.streaming.enabled and final.memory.guard
    e = final.evaluation
    assert (e.every, e.full_validation, e.batch_size, e.score_batch_size) == (20, False, 16, 8)


def test_the_lr0_control_keeps_a_valid_checkpoint_and_evaluation_schedule():
    """30 updates with evaluation.every 20 (inherited): the trainer saves (and evaluates the subset) at step0, at every
    multiple of `every` and at max_steps, so the control has step0/20/30, ending at its last update."""
    lr0 = load("experiments/qwen3_4b_final_lr0.yaml")
    t, every = lr0.training.max_steps, lr0.evaluation.every
    assert (t, every) == (30, 20)
    assert sorted({0, t} | set(range(every, t + 1, every))) == [0, 20, 30]
    assert lr0.training.micro_batch_size == 6 and lr0.evaluation.every <= t


def test_the_staged_schedule_smoke_10_primary_60_extension_210_opt_in():
    final, ext = load("experiments/qwen3_4b_final.yaml"), load("experiments/qwen3_4b_final_ext210.yaml")
    smoke, dsmoke = load("experiments/qwen3_4b_smoke.yaml"), load("experiments/qwen3_4b_smoke_dflash.yaml")
    assert (dsmoke.training.max_steps, smoke.training.max_steps, final.training.max_steps) == (10, 4, 60)
    assert set(diff(final, ext)) == {"run.name", "training.max_steps"} and ext.training.max_steps == 210
    assert ext.dflash == DRAFTER
    # the two smoke runs: same data, seed, batch, optimizer and sampling; they differ in rollout generator and length
    assert set(diff(smoke, dsmoke)) == {"run.name", "dflash.enabled", "dflash.draft_model", "training.max_steps"}


def test_the_drift_checkpoints_are_ones_the_primary_run_saves():
    """The trainer saves a checkpoint at every multiple of evaluation.every and at max_steps (plus step0, the initial
    adapter); H10.dflash_drift must name only those, ending at the final update."""
    import yaml as _yaml
    final = load("experiments/qwen3_4b_final.yaml")
    t, every = final.training.max_steps, final.evaluation.every
    saved = {0, t} | set(range(every, t + 1, every))
    study = _yaml.safe_load((CONFIGS / "studies" / "assignment.yaml").read_text())
    e = {x["id"]: x for x in study["entries"]}
    drift = e["H10.dflash_drift"]
    argv = drift["steps"][0]["argv"]
    adapters = argv[argv.index("--adapters") + 1:argv.index("--max-new")]
    steps = [int(a.rsplit("step", 1)[1]) for a in adapters]
    assert steps == [0, 20, 40, 60] and set(steps) <= saved and steps[-1] == t
    assert drift["prerequisites"]["checkpoints"] == adapters  # each one verified saved before the entry runs
    assert all(a.startswith("runs/qwen3_4b_final/") for a in adapters)
    ext = e["H10.dflash_drift_extended"]
    assert ext.get("opt_in") is True and e["H12.train_extended"].get("opt_in") is True
    eargv = ext["steps"][0]["argv"]
    ext_steps = [int(a.rsplit("step", 1)[1]) for a in eargv[eargv.index("--adapters") + 1:eargv.index("--max-new")]]
    assert ext_steps[-1] == load("experiments/qwen3_4b_final_ext210.yaml").training.max_steps


# sha256 of the resolved plain-L2L / DFlash reference configs before the final run enabled DFlash (commit 8aa63ee):
# enabling DFlash in qwen3_4b_final.yaml must not change any of them. qwen3_4b_smoke.yaml extends the final run, so its
# hash changed on purpose on 2026-11-18 when the final run's hyperparameters changed (micro_batch_size 6, learning rate
# 2e-4, evaluation every 20 with batch 16); it was 483f6ca97333a663ec8ed379246b2f5f13307c89a1d53710ae1f91d46f1c0fa0.
# DFlash stays disabled there (asserted below).
HASHES_BEFORE_DFLASH_FINAL = {
    "experiments/qwen3_4b_smoke.yaml": "9bc68ebcc4f1c07788f2183e77afcf18429a2d28e366bfa022ba67a5c1807b3a",
    "experiments/qwen3_4b_l2l.yaml": "24d6ba78c6adb0246713fdc50f7d0b9e7b3dd6d776ff7a951ece3a39a195dc9e",
    "experiments/qwen3_4b_l2l_dflash.yaml": "f1ea3ed565aec1e9757e451eb7acca62411950cd5b7cdffef2618e1ff79583f4",
}


@pytest.mark.parametrize("rel", sorted(HASHES_BEFORE_DFLASH_FINAL))
def test_plain_and_dflash_reference_configs_are_unchanged(rel):
    """(qwen3_4b_smoke_dflash.yaml changed on purpose: 4 -> 10 updates for the integration smoke.)"""
    cfg = load(rel)
    assert config.sha256(cfg) == HASHES_BEFORE_DFLASH_FINAL[rel]
    assert cfg.dflash.enabled == ("dflash" in rel)


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
    "qwen3_0.6b_short": (["qwen3_0.6b_resident_short", "qwen3_0.6b_l2l_sync_short", "qwen3_0.6b_l2l_async_short"],
                         set()),  # the 10-step H05.real_0.6b_bf16 trio of the assignment protocol
    "kaggle_06b_fp32": (["kaggle_06b_fp32_resident", "kaggle_06b_fp32_l2l_sync", "kaggle_06b_fp32_l2l_async"], set()),
    "kaggle_06b_fp32_train": (["kaggle_06b_fp32_train_resident", "kaggle_06b_fp32_train_l2l_sync",
                               "kaggle_06b_fp32_train_l2l_async"], set()),
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


def test_the_experiment_mode_plain_runs_are_the_final_runs_without_dflash():
    """scripts/run_assignment.sh --experiment trains qwen3_4b_final_plain(_lr0).yaml: the full mode's final run and its
    LR=0 control with plain L2L rollouts. Nothing else differs (hyperparameters, data, LoRA, async L2L, pinned store,
    checkpointing, streaming, memory guard), the run names differ (no checkpoint is shared with a DFlash run), and the
    control inherits the plain rollout path. The plain-L2L smoke has the same training tensor shapes."""
    final, lr0 = load("experiments/qwen3_4b_final.yaml"), load("experiments/qwen3_4b_final_lr0.yaml")
    plain, plain0 = load("experiments/qwen3_4b_final_plain.yaml"), load("experiments/qwen3_4b_final_plain_lr0.yaml")
    smoke = load("experiments/qwen3_4b_smoke.yaml")
    assert set(diff(final, plain)) == {"run.name", "dflash.enabled", "dflash.draft_model"}
    assert set(diff(lr0, plain0)) == {"run.name", "dflash.enabled", "dflash.draft_model"}
    assert set(diff(plain, plain0)) == {"run.name", "training.learning_rate", "training.max_steps"}
    assert (plain.run.name, plain0.run.name) == ("qwen3_4b_final_plain", "qwen3_4b_final_plain_lr0")
    assert len({plain.run.name, plain0.run.name, final.run.name, lr0.run.name}) == 4
    for c in (plain, plain0):
        assert not c.dflash.enabled and c.dflash.draft_model is None
        assert (c.model.name, c.model.dtype, c.execution.backend, c.execution.weight_storage) == \
            ("Qwen/Qwen3-4B", "bfloat16", "l2l_async", "pinned")
        assert c.execution.prefetch_depth >= 1 and c.execution.activation_checkpointing
        assert c.streaming.enabled and c.memory.guard
    t = plain.training
    assert (t.batch_size, t.group_size, t.micro_batch_size, t.learning_rate, t.max_steps) == (5, 6, 6, 2e-4, 60)
    assert (plain0.training.learning_rate, plain0.training.max_steps) == (0.0, 30)
    assert plain.dflash == plain0.dflash == smoke.dflash
    assert smoke.execution == plain.execution and smoke.lora == plain.lora and smoke.model == plain.model
    for k in ("batch_size", "group_size", "micro_batch_size", "max_new_tokens", "num_iterations"):
        assert getattr(smoke.training, k) == getattr(plain.training, k), k


def test_plain_rollouts_use_the_streamed_generator_and_load_no_drafter(monkeypatch):
    """backend.make_backend builds DFlashGenerator only under dflash.enabled; the plain configs never import or load
    the drafter."""
    import inspect
    from qwen_grpo.execution import backend
    src = inspect.getsource(backend.make_backend)
    assert "if cfg.dflash.enabled:" in src and "DFlashDrafter.from_pretrained" in src
    assert "StreamedGenerator(" in inspect.getsource(backend)
    for rel in ("experiments/qwen3_4b_final_plain.yaml", "experiments/qwen3_4b_final_plain_lr0.yaml"):
        assert load(rel).dflash.enabled is False
