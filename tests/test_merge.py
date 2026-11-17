"""Streaming LoRA merge (qwen_grpo/export/merge.py) on tiny random Qwen3 models (fp32 and bf16), CPU only.

The merged checkpoint must hold one correct rounding of W0 + (alpha/r) B A per adapted tensor, every other tensor
unchanged, load on its own as a plain Hugging Face model with its tied embedding / head, and behave like the
adapter-attached model (fp32: 1e-4 relative logits; bf16: 2^-4) when run by the L2L engine, which never calls
model.generate(). The full grid with peak memory is the lab case merge_equivalence."""
import json

import pytest
import torch
import transformers
from peft import PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from qwen_grpo.execution import generator as gen
from qwen_grpo.execution.qwen3_forward import explicit_forward
from qwen_grpo.export.merge import base_files, load_merged_l2l, read_adapter, stream_merge
from qwen_grpo.model import EOS_IDS, lora_config, lora_modules, target_modules

TOL = {torch.float32: 1e-4, torch.bfloat16: 2 ** -4}


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def make(tmp_path, dtype, alpha=16, r=8, b_std=0.5):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                            tie_word_embeddings=True)).to(dtype).eval()
        base.save_pretrained(tmp_path / "base")
        model = get_peft_model(base, lora_config(target_modules(base, ("q_proj", "v_proj", "o_proj"), 2), r, alpha)).eval()
        for _, m in lora_modules(model):
            torch.nn.init.normal_(m.lora_B["default"].weight, std=b_std)
    model.save_pretrained(tmp_path / "adapter")
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merge_is_one_rounding_of_w0_plus_scaled_ba_and_leaves_everything_else_untouched(tmp_path, dtype):
    make(tmp_path, dtype, alpha=24, r=8)  # alpha / r = 3: a dropped or wrong scale would show
    rep = stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged", shard_bytes=30_000)
    assert rep["scale"] == 3.0 and len(rep["merged"]) == 6 and len(rep["shards"]) > 1
    scale, pairs, _ = read_adapter(tmp_path / "adapter")
    base, merged = base_files(tmp_path / "base"), base_files(tmp_path / "merged")
    assert set(base) == set(merged)
    for name in base:
        with safe_open(base[name], framework="pt") as h0, safe_open(merged[name], framework="pt") as hm:
            W0, Wm = h0.get_tensor(name), hm.get_tensor(name)
        assert Wm.dtype == W0.dtype and Wm.shape == W0.shape
        if name in pairs:
            A, B = pairs[name]
            assert torch.equal(Wm, (W0.float() + scale * (B @ A)).to(dtype)), name  # exactly one rounding
            assert rep["merged"][name]["within_half_ulp"] and rep["merged"][name]["changed_elements"] > 0
        else:
            assert torch.equal(Wm, W0), name
    index = json.loads((tmp_path / "merged" / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == rep["total_bytes"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_merged_checkpoint_loads_on_its_own_and_matches_the_adapter_through_l2l(tmp_path, dtype, monkeypatch):
    model = make(tmp_path, dtype)
    stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    plain = AutoModelForCausalLM.from_pretrained(tmp_path / "merged", dtype=dtype)
    assert not any("lora" in n for n, _ in plain.named_parameters())
    assert plain.lm_head.weight is plain.model.embed_tokens.weight  # tied, as in the base
    monkeypatch.setattr(transformers.GenerationMixin, "generate", lambda *a, **k: pytest.fail("model.generate() used"))
    merged, backend = load_merged_l2l(tmp_path / "merged", "cpu", dtype, "pageable", 2)
    ids = torch.randint(3, 300, (2, 9), generator=torch.Generator().manual_seed(1))
    mask = torch.ones_like(ids)
    with torch.no_grad():
        za = model(input_ids=ids).logits.float()
        with model.disable_adapter():
            zb = model(input_ids=ids).logits.float()
        zm = explicit_forward(merged, ids, mask, None, 0, layer_executor=backend.executor).float()
        out = gen.generate(merged, ids, mask, EOS_IDS, EOS_IDS[1], 4, do_sample=False, layer_executor=backend.executor)
    scale = za.abs().max()
    assert (za - zm).abs().max() <= TOL[dtype] * scale
    assert (za - zb).abs().max() > 10 * (za - zm).abs().max()  # the adapter's effect survived the merge
    assert out["sequences"].shape[1] > ids.shape[1]
    assert backend.store.verify() == [] and all(e.param.device.type == "cpu" for s in backend.store.slabs
                                                for e in s.entries)  # decoder weights stay in the CPU store
    backend.close()


def test_refuses_what_it_cannot_merge_correctly(tmp_path):
    make(tmp_path, torch.float32)
    stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    with pytest.raises(FileExistsError, match="never overwritten"):
        stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    cfg_path = tmp_path / "adapter" / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg_path.write_text(json.dumps({**cfg, "use_rslora": True}))
    with pytest.raises(ValueError, match="use_rslora"):
        stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "m2")
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="without a base tensor"):
        _merge_into_wrong_base(tmp_path)


def _merge_into_wrong_base(tmp_path):
    other = Qwen3ForCausalLM(Qwen3Config(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2, head_dim=8))
    other.save_pretrained(tmp_path / "small_base")  # 2 layers: the adapter's layers 2-3 do not exist
    stream_merge(tmp_path / "small_base", tmp_path / "adapter", tmp_path / "m4")
