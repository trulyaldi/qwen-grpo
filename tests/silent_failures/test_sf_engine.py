"""Fault injection in the engine: a dropped LoRA scale in the streamed merge, and a stale DFlash verifier.

Both faults leave every shape, dtype and file valid; only a behavioural comparison with the reference catches them.
Control and injected fault go through the same comparison. CPU, tiny random models."""
import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM

from qwen_grpo.backward_lab import build
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.dflash import dflash_generate
from qwen_grpo.execution.generator import generate
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.export import merge as M
from qwen_grpo.gen_lab import prompts
from qwen_grpo.model import lora_modules
from tests.test_dflash import V, drafter
from tests.test_merge import make


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


# --- merge ----------------------------------------------------------------------------------------------------------

def merged_vs_adapter(model, merged_dir):
    """max |logits(adapter-attached) - logits(merged)| relative to the logits' scale, and the adapter's own effect."""
    plain = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    ids = torch.randint(3, 300, (2, 9), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        za = model(input_ids=ids).logits
        with model.disable_adapter():
            zb = model(input_ids=ids).logits
        zm = plain(input_ids=ids).logits
    scale = za.abs().max()
    return ((za - zm).abs().max() / scale).item(), ((za - zb).abs().max() / scale).item()


def test_a_dropped_merge_scale_is_detected(tmp_path, monkeypatch):
    model = make(tmp_path, torch.float32, alpha=24, r=8)  # scale 3
    M.stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    diff, effect = merged_vs_adapter(model, tmp_path / "merged")
    assert diff < 1e-5 and effect > 1e-2  # control: the merge reproduces the adapter, whose effect is visible

    real = M.read_adapter
    monkeypatch.setattr(M, "read_adapter", lambda d: (0.0, *real(d)[1:]))  # injected: alpha / r lost
    M.stream_merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged_bad")
    bad, _ = merged_vs_adapter(model, tmp_path / "merged_bad")
    assert bad > 100 * max(diff, 1e-7)  # caught by the same comparison


# --- DFlash -------------------------------------------------------------------------------------------------------

def target():
    with F.isolated():
        return build("4b-like", "sdpa", torch.float32, [0, 5], vocab=V)


def streamed(state):
    m = target()
    m.load_state_dict(state)
    store = WeightStore.build(m, "pageable")
    place_resident(m, store, "cpu")
    return m, StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, "cpu"))


def shift_lora(m, seed=3, std=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, mod in lora_modules(m):
            w = mod.lora_B["default"].weight
            w.add_(torch.randn(w.shape, generator=g) * std)


def test_a_stale_dflash_verifier_is_detected(monkeypatch):
    monkeypatch.setattr(transformers.GenerationMixin, "generate", lambda *a, **k: pytest.fail("model.generate() used"))
    ids, mask = prompts([9, 5, 7], V, "cpu", 0)
    old_state = {k: v.clone() for k, v in target().state_dict().items()}

    current, ex = streamed(old_state)  # the policy being trained, updated in place below
    stale, stale_ex = streamed(old_state)  # injected: a verifier that kept the pre-update policy
    reference = target()
    reference.load_state_dict(old_state)
    with F.isolated():
        before = generate(reference, ids, mask, F.EOS_IDS, F.PAD, 10, 3, do_sample=False)["sequences"]
    shift_lora(current)  # one "optimizer step" on the LoRA tensors
    shift_lora(reference)
    with F.isolated():
        plain = generate(reference, ids, mask, F.EOS_IDS, F.PAD, 10, 3, do_sample=False)["sequences"]
        fresh = dflash_generate(current, drafter(), ids, mask, F.EOS_IDS, F.PAD, 10, 3, do_sample=False,
                                layer_executor=ex)["sequences"]
        old = dflash_generate(stale, drafter(), ids, mask, F.EOS_IDS, F.PAD, 10, 3, do_sample=False,
                              layer_executor=stale_ex)["sequences"]
    assert not torch.equal(before, plain)  # the update changes the greedy policy, so a stale verifier is observable
    assert torch.equal(fresh, plain)  # control: DFlash verified by the current policy == plain greedy decoding
    assert not torch.equal(old, plain)  # the stale verifier is caught by the same equality check
