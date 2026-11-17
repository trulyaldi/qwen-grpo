"""Streamed generator (qwen_grpo/execution/generator.py) against transformers generate(), on tiny random Qwen3 models.

Cached decoding is compared bitwise (research/l2l/tolerances.yaml, generation_equivalence); CPU only here (CUDA cases
are skipped without a GPU). The full grid with memory and time is the lab case generation_equivalence.
"""
import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from qwen_grpo.backward_lab import build
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.generator import KVCache, StreamedGenerator, generate
from qwen_grpo.execution.layer_stream import StreamingLayerExecutor, place_resident
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.gen_lab import full_prefix, prompts
from qwen_grpo.model import EOS_IDS, MODEL
from qwen_grpo.policy import generate_rollouts
from qwen_grpo.sentiment import build_prompts

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)  # tests/test_grpo.py sets float64 at import


def pair(attn="eager", dtype=torch.float32, shape="4b-like", vocab=64, device="cpu", eos=F.EOS_IDS, pad=F.PAD):
    """(resident model, L2L model with the same weights, its store, an L2L executor)."""
    with F.isolated():
        ref = build(shape, attn, dtype, [0, F.SHAPES[shape]["num_hidden_layers"] - 1], vocab=vocab)
        l2l = build(shape, attn, dtype, [0, F.SHAPES[shape]["num_hidden_layers"] - 1], vocab=vocab)
    l2l.load_state_dict(ref.state_dict())
    for m in (ref, l2l):
        m.generation_config = GenerationConfig(eos_token_id=list(eos), pad_token_id=pad)
    store = WeightStore.build(l2l, "pinned" if device == "cuda" else "pageable")
    place_resident(l2l, store, device)
    return ref.to(device), l2l, store, StreamingLayerExecutor(store, gpu_buffers(2, store.max_slab_bytes, device))


def hf(model, ids, mask, n=8, **kw):
    return model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=n, eos_token_id=F.EOS_IDS,
                          pad_token_id=F.PAD, output_logits=True, return_dict_in_generate=True, **kw)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lengths", [[9, 5, 9, 2], [17, 3, 11, 1], [4]])
def test_greedy_logits_and_tokens_equal_hf_generate_resident_and_streamed(device, attn, dtype, lengths):
    ref, l2l, store, ex = pair(attn, dtype, device=device)
    ids, mask = prompts(lengths, 64, device, seed=len(lengths))
    with F.isolated():
        want = hf(ref, ids, mask, do_sample=False)
        for model, e, kv in ((ref, None, None), (l2l, ex, None), (l2l, ex, "cpu")):
            got = generate(model, ids, mask, F.EOS_IDS, F.PAD, 8, do_sample=False, layer_executor=e, kv_device=kv,
                           keep_logits=True)
            assert torch.equal(got["sequences"], want.sequences)
            assert len(got["step_logits"]) == len(want.logits)
            assert all(torch.equal(a.float(), b) for a, b in zip(want.logits, got["step_logits"]))
    assert store.verify() == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_seed_sampling_draws_the_same_tokens_as_hf_and_the_distribution_matches(dtype):
    ref, l2l, store, ex = pair(dtype=dtype)
    ids, mask = prompts([9, 5, 9, 2], 64, "cpu", seed=0)
    with F.isolated():
        torch.manual_seed(7)
        want = hf(ref, ids, mask, do_sample=True, temperature=1.0, top_k=0, top_p=1.0, num_return_sequences=3)
        torch.manual_seed(7)
        got = generate(l2l, ids, mask, F.EOS_IDS, F.PAD, 8, 3, do_sample=True, layer_executor=ex)
        forced = generate(l2l, ids, mask, F.EOS_IDS, F.PAD, want.sequences.shape[1] - ids.shape[1], 3,
                          layer_executor=ex, forced=want.sequences[:, ids.shape[1]:], keep_logits=True)
    assert torch.equal(got["sequences"], want.sequences)
    assert torch.equal(forced["sequences"], want.sequences)  # teacher forcing follows the given trajectory
    assert all(torch.equal(a.float().softmax(-1), b.softmax(-1)) for a, b in zip(want.logits, forced["step_logits"]))


def test_rows_that_emit_eos_get_padding_and_the_loop_stops_when_all_have_finished():
    ref, l2l, store, ex = pair()
    with torch.no_grad():
        ref.get_input_embeddings().weight[F.EOS_IDS[0]] *= 24  # EOS likelier (seeded case found by search)
    l2l.load_state_dict(ref.state_dict())
    ids, mask = prompts([6, 3], 64, "cpu", seed=0)
    with F.isolated():
        want = hf(ref, ids, mask, do_sample=False)
        got = generate(l2l, ids, mask, F.EOS_IDS, F.PAD, 8, do_sample=False, layer_executor=ex)
    assert torch.equal(got["sequences"], want.sequences)
    assert got["sequences"][:, 6:].tolist() == [[1, 2, 2], [46, 35, 1]]  # row 0 padded after EOS; stop at step 3
    ids, mask = prompts([9, 5, 9, 2], 64, "cpu", seed=0)
    forced = torch.tensor([[5, F.EOS_IDS[0], 9, 9], [5, 6, 7, 8], [F.EOS_IDS[1], 9, 9, 9], [5, 6, F.EOS_IDS[0], 9]])
    out = generate(l2l, ids, mask, F.EOS_IDS, F.PAD, 4, layer_executor=ex, forced=forced)
    assert out["sequences"][:, ids.shape[1]:].tolist() == [[5, 1, 2, 2], [5, 6, 7, 8], [2, 2, 2, 2], [5, 6, 1, 2]]


def test_cached_decoding_agrees_with_the_full_prefix_forward_within_the_rounding_bound():
    ref, _, _, _ = pair(attn="sdpa")
    ids, mask = prompts([17, 3, 11, 1], 64, "cpu", seed=1)
    with F.isolated():
        out = generate(ref, ids, mask, F.EOS_IDS, F.PAD, 8, do_sample=False, keep_logits=True)
        full = full_prefix(ref, out["sequences"], mask, len(out["step_logits"]))
    for a, b in zip(full, out["step_logits"]):
        assert (a - b).abs().max() <= 1e-5 * a.abs().max()


def test_streaming_loads_every_layer_once_per_step_and_the_kv_cache_has_its_analytic_size():
    ref, l2l, store, ex = pair()
    ids, mask = prompts([9, 5, 9, 2], 64, "cpu", seed=0)
    with F.isolated():
        out = generate(l2l, ids, mask, F.EOS_IDS, F.PAD, 6, 2, do_sample=False, layer_executor=ex)
    c = l2l.config
    assert len(ex.trace) == c.num_hidden_layers * out["steps"]
    assert out["kv_cache_bytes"] == 2 * c.num_hidden_layers * 8 * c.num_key_value_heads * (9 + 6) * c.head_dim * 4
    assert all(e.param.device.type == "cpu" for s in store.slabs for e in s.entries)  # weights back in the store


def test_kv_cache_hands_out_contiguous_copies_and_refuses_overflow():
    cache = KVCache(1, 2, 2, 4, 3, torch.float32, "cpu", "cpu")
    k, v = cache.update(torch.ones(2, 2, 3, 3), torch.zeros(2, 2, 3, 3), 0)
    assert k.is_contiguous() and k.shape == (2, 2, 3, 3) and k.data_ptr() != cache.keys[0].data_ptr()
    assert cache.get_seq_length(0) == 3 and cache.get_mask_sizes(torch.arange(1), 0) == (4, 0)
    with pytest.raises(RuntimeError, match="overflow"):
        cache.update(torch.ones(2, 2, 2, 3), torch.ones(2, 2, 2, 3), 0)


def test_refuses_training_mode():
    ref, _, _, _ = pair()
    ids, mask = prompts([4], 64, "cpu", seed=0)
    with pytest.raises(RuntimeError, match="eval mode"):
        generate(ref.train(), ids, mask, F.EOS_IDS, F.PAD, 2)


# --- rollouts: the L2L backend never falls back to model.generate() -----------------------------------------------

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL, padding_side="left")


def test_l2l_rollouts_never_call_model_generate_and_match_hf_rollouts_with_the_same_seed(tok, monkeypatch):
    ref, l2l, store, ex = pair(attn="sdpa", shape="0.6b-like", vocab=151936, eos=EOS_IDS, pad=tok.pad_token_id)
    texts = build_prompts(tok, ["A gorgeous, witty film.", "Two hours of my life I will never get back."])
    assert all("<think>\n\n</think>" in t for t in texts)  # non-thinking Qwen3 formatting, unchanged
    with F.isolated():
        torch.manual_seed(3)
        want = generate_rollouts(ref, tok, texts, G=2, max_new_tokens=5)
    for m in (ref, l2l):
        monkeypatch.setattr(m, "generate", lambda *a, **k: pytest.fail("model.generate() called under the L2L backend"))
    with F.isolated():
        torch.manual_seed(3)
        got = generate_rollouts(l2l, tok, texts, G=2, max_new_tokens=5, generator=StreamedGenerator(ex))
    assert all(torch.equal(want[k], got[k]) for k in ("input_ids", "attention_mask", "response_ids", "completion_mask"))
    assert got["prompt_len"] == want["prompt_len"]
    with pytest.raises(pytest.fail.Exception, match="model.generate"):  # the resident path really is guarded
        generate_rollouts(l2l, tok, texts, G=2, max_new_tokens=5)
