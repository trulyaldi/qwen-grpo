"""Step 0 budget: unit conversions, exact metadata accounting against a tiny random Qwen3 (meta skeleton, PEFT, real
KV cache and logits), peak-of-moments accounting, residency, PCIe arithmetic, merge, and error paths. Offline."""
import copy
import math
from pathlib import Path

import pytest
import torch
import yaml
from peft import get_peft_model
from transformers import AutoModelForCausalLM, Qwen3Config

from qwen_grpo.budget import architecture as A
from qwen_grpo.budget import calculator as C
from qwen_grpo.model import lora_config

ROOT = Path(__file__).parents[1]
TINY = dict(model_type="qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16, vocab_size=1000, tie_word_embeddings=True)


def tiny_config(**over):
    return Qwen3Config(**{k: v for k, v in {**TINY, **over}.items() if k != "model_type"})


def skeleton(config):
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def checkpoint(config, store_tied_head=True, dtype="BF16"):
    """Synthetic checkpoint metadata named exactly as the HF model's state_dict."""
    sd = skeleton(config).state_dict()
    tensors = {k: A.Tensor(k, tuple(v.shape), dtype) for k, v in sd.items()
               if store_tied_head or k != "lm_head.weight"}
    return tensors, {"model.safetensors": sum(t.stored_bytes for t in tensors.values())}


def tiny_arch(targets="last2-attention", **over):
    config = tiny_config(**over)
    tensors, shards = checkpoint(config)
    arch = A.from_metadata("tiny", config, tensors, shards, "synthetic")
    arch.lora_targets = A.resolve_lora_targets(arch, config, targets)
    return arch, config


def tiny_cfg(**over):
    cfg = C.load_config(ROOT / "configs/budget/qwen3-0.6b.yaml")
    cfg["grpo"].update(prompt_tokens=10, generation_tokens=3, prompt_batch_size=2, group_size=3,
                       rollout_batch_size=6, micro_batch_size=2)
    for path, value in over.items():
        sec, key = path.split(".")
        cfg[sec][key] = value
    return C.validate(cfg)


# ------------------------------------------------------------------------------------------------ units and sizes

def test_bytes_and_unit_conversions():
    assert C.tensor_bytes((3, 5), "bfloat16") == 30 and C.tensor_bytes((3, 5), "float32") == 60
    assert C.to_gib(C.GiB) == 1 and C.to_gb(C.GB) == 1
    assert C.to_gb(C.GiB) == pytest.approx(1.073741824)  # GiB (2^30) vs GB (10^9): ~7.4 % apart
    assert C.pow2_ceil(1) == 1 and C.pow2_ceil(5) == 8 and C.pow2_ceil(8) == 8
    assert C.fmt(3 * C.GiB) == "3.000 GiB" and C.fmt(5 * C.MiB) == "5.0 MiB" and C.fmt(None) == "unknown"


def test_bf16_and_fp32_parameter_sizes():
    bf, f32 = A.Tensor("w", (128, 64), "BF16"), A.Tensor("w", (128, 64), "F32")
    assert bf.numel == 128 * 64 and bf.stored_bytes == 2 * bf.numel and f32.stored_bytes == 2 * bf.stored_bytes
    ref = torch.empty(128, 64, dtype=torch.bfloat16)
    assert bf.stored_bytes == ref.numel() * ref.element_size()


def test_read_header_matches_saved_safetensors(tmp_path):
    from safetensors.torch import save_file
    save_file({"a": torch.zeros(3, 4, dtype=torch.bfloat16), "b": torch.zeros(5, dtype=torch.float32)},
              tmp_path / "x.safetensors")
    h = A.read_header(tmp_path / "x.safetensors")
    assert h["a"] == A.Tensor("a", (3, 4), "BF16") and h["b"].stored_bytes == 20


# ------------------------------------------------------------------------------------------ architecture metadata

def test_tied_head_stored_in_checkpoint_is_counted_once():
    """Qwen3-0.6B stores lm_head.weight although tie_word_embeddings=True; live memory holds one matrix."""
    arch, config = tiny_arch()
    assert arch.lm_head is None and [t.name for t in arch.tied_duplicates] == ["lm_head.weight"]
    assert arch.live_numel == sum(p.numel() for p in skeleton(config).parameters())  # parameters() dedups ties
    assert arch.checkpoint_bytes == 2 * (arch.live_numel + arch.embed.numel)          # the disk copy is counted


def test_tied_head_absent_from_checkpoint_and_untied_head():
    config = tiny_config()
    tensors, shards = checkpoint(config, store_tied_head=False)  # Qwen3-4B layout
    arch = A.from_metadata("tiny", config, tensors, shards, "synthetic")
    assert arch.lm_head is None and arch.tied_duplicates == []
    untied = tiny_config(tie_word_embeddings=False)
    arch = A.from_metadata("tiny", untied, *checkpoint(untied), "synthetic")
    assert arch.lm_head.shape == (1000, 64)
    assert arch.live_numel == sum(p.numel() for p in skeleton(untied).parameters())
    with pytest.raises(A.MetadataError, match="untied"):
        A.from_metadata("tiny", untied, *checkpoint(untied, store_tied_head=False), "synthetic")


def test_decoder_layer_sizes_individually():
    arch, config = tiny_arch()
    layers = skeleton(config).model.layers
    assert [arch.layer_numel(i) for i in range(4)] == [sum(p.numel() for p in l.parameters()) for l in layers]
    b = C.Budget(tiny_cfg(), arch)
    assert b.layer_bytes == [2 * arch.layer_numel(i) for i in range(4)]


@pytest.mark.parametrize("over, error, match", [
    ({"model_type": "llama"}, A.UnsupportedArchitecture, "model_type"),
    ({"num_experts": 8}, A.UnsupportedArchitecture, "mixture-of-experts"),
    ({"head_dim": None}, A.MetadataError, "head_dim"),
])
def test_unsupported_or_incomplete_config(over, error, match):
    cfg = {**TINY, **over}
    with pytest.raises(error, match=match):
        A.from_metadata("x", cfg, *checkpoint(tiny_config()), "synthetic")


def test_inconsistent_or_incomplete_checkpoint():
    config = tiny_config()
    tensors, shards = checkpoint(config)
    for broken, match in [({**tensors, "model.extra.weight": A.Tensor("model.extra.weight", (1,), "BF16")},
                           "unclassified"),
                          ({k: v for k, v in tensors.items() if not k.startswith("model.layers.3.")}, "lacks"),
                          ({k: v for k, v in tensors.items() if k != "model.norm.weight"}, "lacks"),
                          ({**tensors, "model.layers.0.self_attn.k_proj.weight":
                            A.Tensor("model.layers.0.self_attn.k_proj.weight", (64, 64), "BF16")}, "k_proj")]:
        with pytest.raises(A.MetadataError, match=match):
            A.from_metadata("tiny", config, broken, shards, "synthetic")


def test_missing_remote_metadata_is_reported(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(huggingface_hub, "get_safetensors_metadata",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(A.MetadataError, match="remote safetensors metadata failed"):
        A.checkpoint_tensors("nobody/nothing", None)


# --------------------------------------------------------------------------------------------- LoRA and optimizer

def test_lora_parameters_match_peft():
    arch, config = tiny_arch(targets={"modules": ["q_proj", "v_proj", "o_proj", "down_proj"], "num_layers": 2})
    torch.manual_seed(0)
    model = get_peft_model(AutoModelForCausalLM.from_config(config),
                           lora_config([t.name.removesuffix(".weight") for t in arch.lora_targets], r=4, alpha=8))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert C.lora_numel(arch.lora_targets, 4) == trainable
    assert arch.first_adapted_layer == 2


def test_adamw_state_bytes_match_torch():
    p = torch.nn.Parameter(torch.zeros(37, 11))
    opt = torch.optim.AdamW([p])
    p.grad = torch.ones_like(p)
    opt.step()
    state = [v for v in opt.state[p].values() if torch.is_tensor(v) and v.numel() > 1]
    assert sum(t.numel() * t.element_size() for t in state) == C.adamw_state_bytes(p.numel())


# ---------------------------------------------------------------------------------------- activations and logits

def test_activation_checkpoints_only_for_the_adapted_span():
    arch, _ = tiny_arch()  # last2-attention: layers 2, 3 -> boundaries: inputs of 2, 3 + final hidden
    cfg = tiny_cfg()
    b = C.Budget(cfg, arch)
    expect = C.checkpoint_bytes(3, b.N, b.S, 64, "bfloat16")
    assert expect == 3 * 6 * 13 * 64 * 2
    head = next(m for m in b.training().moments if m.name.startswith("forward: LM head"))
    assert [i.bytes for i in head.gpu if i.name.startswith("boundary checkpoints")] == [expect]
    mb = C.Budget(tiny_cfg(**{"streaming.schedule": "microbatch_major"}), arch)  # one micro-batch's checkpoints
    head = next(m for m in mb.training().moments if m.name.startswith("forward: LM head"))
    assert [i.bytes for i in head.gpu if i.name.startswith("boundary checkpoints")] == [expect * 2 // 6]


def test_response_only_vs_full_sequence_logits_match_model_output():
    config = tiny_config()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    n, T, R = 2, 9, 3
    ids = torch.randint(0, 1000, (n, T))
    with torch.no_grad():
        kept = model(input_ids=ids, logits_to_keep=R + 1).logits
        full = model(input_ids=ids).logits
    resp, whole = C.logits_bytes(n, R + 1, R, 1000, "bfloat16"), C.logits_bytes(n, T, R, 1000, "bfloat16")
    assert resp["logits"] == kept.numel() * kept.element_size()
    assert whole["logits"] == full.numel() * full.element_size()
    assert resp["fp32_copy"] == whole["fp32_copy"] == n * R * 1000 * 4  # only R columns are upcast either way
    arch, _ = tiny_arch()
    full_cfg = C.Budget(tiny_cfg(**{"streaming.logits": "full_sequence"}), arch).logits(2)
    assert full_cfg["logits"] == 2 * 13 * 1000 * 2


def test_gqa_kv_cache_matches_dynamic_cache():
    config = tiny_config()
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    n, T = 3, 7
    with torch.no_grad():
        cache = model(input_ids=torch.randint(0, 1000, (n, T)), use_cache=True).past_key_values
    layers = getattr(cache, "layers", None)
    kv = [(l.keys, l.values) for l in layers] if layers is not None else list(zip(cache.key_cache, cache.value_cache))
    actual = sum(t.numel() * t.element_size() for pair in kv for t in pair)
    assert actual == C.kv_bytes(n, T, 4, num_kv := 2, 16, "bfloat16")
    assert actual * 2 == C.kv_bytes(n, T, 4, 4, 16, "bfloat16")  # with all 4 query heads: twice the size
    del num_kv


# -------------------------------------------------------------------------------------------- residency and peaks

def test_layer_buffer_residency():
    arch, _ = tiny_arch()
    layer = 2 * arch.layer_numel(0)
    b = C.Budget(tiny_cfg(**{"streaming.resident_decoder_layers": 1}), arch)
    assert b.resident == [3] and b.streamed == [0, 1, 2]
    assert b.buffer_bytes() == 2 * layer and b.sweep_bytes() == 3 * layer
    assert C.known(b.cpu_weights()[:1]) == 3 * layer
    rounding = b.cpu_weights()[1]
    assert rounding.kind == C.ESTIMATE and rounding.bytes == 3 * (C.pow2_ceil(layer) - layer)
    full = C.Budget(tiny_cfg(), arch, resident=4)
    assert full.streamed == [] and full.buffer_bytes() == 0 and full.sweep_bytes() == 0
    with pytest.raises(ValueError, match="resident_decoder_layers"):
        C.Budget(tiny_cfg(), arch, resident=5)
    with pytest.raises(ValueError, match="buffers"):
        C.Budget(tiny_cfg(**{"streaming.gpu_layer_buffers": 0}), arch)


def test_phase_peak_is_max_over_moments_not_sum():
    cfg = tiny_cfg()
    a = C.Moment("a", [C.Item("x", 100 * C.MiB, C.EXACT)], [])
    b = C.Moment("b", [C.Item("y", 300 * C.MiB, C.ESTIMATE)], [C.Item("z", C.GiB, C.EXACT)])
    s = C.phase_summary(C.Phase("p", [a, b]), cfg)
    assert s["gpu_peak"]["gpu_allocated"] == 300 * C.MiB and s["gpu_peak_moment"] == "b"
    assert s["gpu_peak"]["gpu_reserved"] == round(300 * C.MiB * 1.1)
    assert s["gpu_peak"]["gpu_device"] == s["gpu_peak"]["gpu_reserved"] + C.GiB // 2
    assert s["cpu_peak_bytes"] == C.GiB + 2 * C.GiB
    arch, _ = tiny_arch()
    train = C.Budget(cfg, arch).training()
    totals = [C.moment_totals(m, cfg)["gpu_allocated"] for m in train.moments]
    assert C.phase_summary(train, cfg)["gpu_peak"]["gpu_allocated"] == max(totals) < sum(totals)


def test_verdicts():
    assert C.verdict(91, 100, 0.1, False) == C.UNSURE and C.verdict(89, 100, 0.1, False) == C.PASS
    assert C.verdict(101, 100, 0.1, False) == C.FAIL and C.verdict(10, 100, 0.1, True) == C.UNSURE
    assert C.combine([C.PASS, C.UNSURE]) == C.UNSURE and C.combine([C.UNSURE, C.FAIL]) == C.FAIL
    unknown = C.Moment("m", [C.Item("draft", None, C.UNKNOWN, critical=True)], [])
    assert C.phase_summary(C.Phase("p", [unknown]), tiny_cfg())["verdict"] == C.UNSURE


def test_residency_recommendation():
    rows = [{"resident": r, "gpu_verdict": C.PASS if r <= 2 else C.FAIL, "cpu_verdict": C.PASS if r >= 1 else C.FAIL}
            for r in range(5)]
    assert C.recommend(rows) == {"max_resident_gpu_pass": 2, "min_resident_cpu_pass": 1, "feasible": True,
                                 "recommended_resident": 2}
    rows[1]["cpu_verdict"] = rows[2]["cpu_verdict"] = C.FAIL
    assert C.recommend(rows)["feasible"] is False and C.recommend(rows)["recommended_resident"] is None


# --------------------------------------------------------------------------------------------------------- PCIe

def test_pcie_transfer_volumes_and_times():
    arch, _ = tiny_arch()
    cfg = tiny_cfg()
    b = C.Budget(cfg, arch)
    layer, sweep = 2 * arch.layer_numel(0), 4 * 2 * arch.layer_numel(0)
    t = {r["name"].split(" (")[0]: r for r in b.transfer_rows()}
    assert t["rollout prefill"]["weight_bytes"] == sweep
    assert t["rollout decode"]["sweeps"] == 2 and t["rollout decode"]["weight_bytes"] == 2 * sweep  # R - 1 = 2
    assert t["old log-probs forward"]["sweeps"] == 1                     # layer-major: one sweep for 3 micro-batches
    assert t["backward recomputation"]["weight_bytes"] == 2 * layer       # only the adapted layers 2, 3
    assert t["rollout decode"]["transfer_s"] == pytest.approx(2 * sweep / (12.0 * C.GB))
    assert all(r["compute_s"] is None for r in t.values())

    mb = {r["name"].split(" (")[0]: r for r in
          C.Budget(tiny_cfg(**{"streaming.schedule": "microbatch_major"}), arch).transfer_rows()}
    assert mb["training forward"]["sweeps"] == 3 and mb["backward recomputation"]["weight_bytes"] == 3 * 2 * layer

    off = {r["name"].split(" (")[0]: r for r in C.Budget(
        tiny_cfg(**{"streaming.kv_cache": "cpu", "streaming.activation_checkpoints": "cpu"}), arch).transfer_rows()}
    kv_tok = C.kv_bytes(1, 1, 4, 2, 16, "bfloat16")
    assert off["rollout prefill"]["activation_bytes"] == 6 * 10 * kv_tok
    assert off["rollout decode"]["activation_bytes"] == 6 * ((11 + 1) + (12 + 1)) * kv_tok
    assert off["training forward"]["activation_bytes"] == C.checkpoint_bytes(3, 6, 13, 64, "bfloat16")

    resident = {r["name"].split(" (")[0]: r for r in C.Budget(cfg, arch, resident=4).transfer_rows()}
    assert resident["rollout decode"]["bytes"] == 0


# --------------------------------------------------------------------------------------------------------- merge

def test_merge_peak_memory():
    arch, _ = tiny_arch(targets={"modules": ["q_proj", "down_proj"], "num_layers": 1})
    biggest = max(t.numel for t in arch.lora_targets)
    assert biggest == 64 * 128  # down_proj [hidden, intermediate]
    gpu = C.Budget(tiny_cfg(), arch).merge().moments[0]
    work = next(i for i in gpu.gpu if i.name.startswith("W (native)"))
    assert work.bytes == biggest * (2 + 4 + 4 + 2)
    cpu = C.Budget(tiny_cfg(**{"merge.device": "cpu"}), arch).merge()
    assert cpu.moments[0].gpu == [] and C.phase_summary(cpu, tiny_cfg())["gpu_peak"]["gpu_device"] == 0
    shard = C.Budget(tiny_cfg(**{"merge.strategy": "shardwise"}), arch).merge().moments[0]
    assert shard.cpu[0].bytes == 2 * arch.checkpoint_bytes


# ------------------------------------------------------------------------------------------- config and reporting

@pytest.mark.parametrize("name", ["qwen3-0.6b.yaml", "qwen3-4b.yaml"])
def test_shipped_budget_configs_validate(name):
    cfg = C.load_config(ROOT / "configs/budget" / name)
    assert cfg["streaming"]["resident_decoder_layers"] == 0 and cfg["streaming"]["gpu_layer_buffers"] == 2
    assert cfg["dflash"]["enabled"] is False


def test_config_errors():
    cfg = tiny_cfg()
    for mutate, match in [(lambda c: c["grpo"].update(extra=1), "unknown keys"),
                          (lambda c: c["lora"].pop("rank"), "missing keys"),
                          (lambda c: c["streaming"].update(kv_cache="disk"), "kv_cache"),
                          (lambda c: c["grpo"].update(reference_logprobs=True), "beta"),
                          (lambda c: c["grpo"].update(group_size=True), "expected")]:
        broken = copy.deepcopy(cfg)
        mutate(broken)
        with pytest.raises(ValueError, match=match):
            C.validate(broken)


def test_report_round_trip_and_no_overwrite(tmp_path):
    arch, _ = tiny_arch()
    cfg = tiny_cfg(**{"dflash.estimate": True})
    rep = C.build_report(cfg, arch, draft_bytes=None)
    assert rep["overall_verdict"] in (C.PASS, C.UNSURE, C.FAIL)
    assert rep["phases"][-1]["verdict"] == C.UNSURE  # unknown draft weights are critical for the DFlash phase only
    md = C.render_markdown(rep)
    paths = C.write_prediction(rep, md, tmp_path)
    assert yaml.safe_load(paths[0].read_text())["run_name"] == cfg["run_name"]
    with pytest.raises(FileExistsError):
        C.write_prediction(rep, md, tmp_path)
    assert math.isfinite(rep["transfers_per_grpo_step"]["transfer_s"])


# ------------------------------------------------------------------------------------------------------- DFlash

DRAFT = {"num_hidden_layers": 2, "block_size": 4, "num_target_features": 3, "hidden_size": 64, "intermediate_size": 128,
         "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16}


def gpu_item(moment, prefix):
    return next(i for i in moment.gpu if i.name.startswith(prefix))


def test_dflash_enabled_keeps_the_drafter_resident_in_every_phase_and_models_its_rollout():
    arch, _ = tiny_arch()
    on = C.Budget(tiny_cfg(**{"dflash.enabled": True}), arch, draft_bytes=12345, draft=DRAFT)
    off = C.Budget(tiny_cfg(), arch)
    for p in on.phases():
        for m in p.moments:
            if m.gpu and p.name != "merge":  # the merge is standalone: training state and drafter released
                assert gpu_item(m, "DFlash drafter weights").bytes == 12345, (p.name, m.name)
    assert not any(i.name.startswith("DFlash") for p in off.phases() for m in p.moments for i in m.gpu)
    assert "dflash (optional estimate)" not in [p.name for p in on.phases()]  # folded into the real rollout phase
    n, P, R, V, H = 6, 10, 3, arch.vocab_size, arch.hidden_size
    names = [m.name for m in on.rollout().moments]
    assert all("(DFlash)" in x for x in names) and len(names) == 4
    head = on.rollout().moments[1]
    # dflash._target_step: the LM head over all P prompt positions, bf16 then an fp32 copy
    assert gpu_item(head, "prefill logits [").bytes == n * P * V * 2
    assert gpu_item(head, "prefill logits fp32").bytes == n * P * V * 4
    assert gpu_item(head, "captured target features").bytes == 3 * n * P * H * 2
    kv = gpu_item(head, "KV cache")  # dflash_generate allocates P + max_new + block_size positions
    assert kv.bytes == C.kv_bytes(n, P + R + 4, arch.num_hidden_layers, arch.num_key_value_heads, arch.head_dim,
                                  "bfloat16")
    verify = on.rollout().moments[3]
    q = min(4, R)
    assert gpu_item(verify, f"block logits [{n}, {q}, V]").bytes == n * q * V * 6
    assert gpu_item(verify, "prefill logits fp32").bytes == n * P * V * 4  # still referenced at the first verify


def test_dflash_raises_the_rollout_peak_by_at_least_the_drafter_and_prefill_logits():
    arch, _ = tiny_arch()
    cfg_on, cfg_off = tiny_cfg(**{"dflash.enabled": True}), tiny_cfg()
    on = C.phase_summary(C.Budget(cfg_on, arch, draft_bytes=10**6, draft=DRAFT).rollout(), cfg_on)
    off = C.phase_summary(C.Budget(cfg_off, arch).rollout(), cfg_off)
    assert on["gpu_peak"]["gpu_allocated"] >= off["gpu_peak"]["gpu_allocated"] + 10**6 + 6 * 10 * arch.vocab_size * 4


def test_dflash_without_drafter_metadata_is_unknown_not_pass():
    arch, _ = tiny_arch()
    cfg = tiny_cfg(**{"dflash.enabled": True})
    for kwargs in ({"draft_bytes": None, "draft": DRAFT}, {"draft_bytes": 10**6, "draft": None}):
        rep = C.build_report(cfg, arch, **kwargs)
        roll = next(p for p in rep["phases"] if p["phase"] == "rollout")
        assert roll["gpu_verdict"] == C.UNSURE and rep["overall_verdict"] != C.PASS, kwargs
    with pytest.raises(ValueError, match="lacks"):
        C.Budget(cfg, arch, draft={"block_size": 4})


def test_dflash_transfers_bound_the_verify_passes():
    arch, _ = tiny_arch()
    b = C.Budget(tiny_cfg(**{"dflash.enabled": True}), arch, draft_bytes=1, draft=DRAFT)
    t = next(x for x in b.transfers() if x.name.startswith("rollout verify"))
    assert t.sweeps == 2 and "between 1" in t.note  # R - 1 = 2 rejections at most, ceil(2 / 4) = 1 at best


def test_draft_config_reads_the_z_lab_layout():
    raw = {"num_hidden_layers": 5, "block_size": 16, "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
           "hidden_size": 2560, "intermediate_size": 9728, "num_attention_heads": 32, "num_key_value_heads": 8,
           "head_dim": 128}
    assert C.draft_config(raw) == {"num_hidden_layers": 5, "block_size": 16, "num_target_features": 5,
                                   "hidden_size": 2560, "intermediate_size": 9728, "num_attention_heads": 32,
                                   "num_key_value_heads": 8, "head_dim": 128}


def test_the_committed_dflash_final_prediction_matches_its_budget_file():
    import hashlib
    import json
    pred = json.loads((ROOT / "results/budget/predictions/qwen3-4b-final-dflash_8gb.json").read_text())
    path = ROOT / pred["provenance"]["config"].split("qwen-grpo/")[-1]
    assert pred["provenance"]["config_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pred["config"] == C.load_config(path) and pred["config"]["dflash"]["enabled"] is True
    assert pred["dflash"]["draft_bytes"] and pred["dflash"]["draft_architecture"]["block_size"] == 16
    roll = next(p for p in pred["phases"] if p["phase"] == "rollout")
    assert roll["gpu_peak_moment"].startswith("rollout (DFlash)")
    assert all(any(i["name"].startswith("DFlash drafter weights") for i in m["gpu"])
               for p in pred["phases"] if p["phase"] != "merge" for m in p["moments"] if m["gpu"])
