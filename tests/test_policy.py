import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig, Qwen3Config, Qwen3ForCausalLM

from qwen_grpo.grpo import response_mask
from qwen_grpo.model import EOS_IDS, MODEL
from qwen_grpo.policy import compute_logprobs, generate_rollouts
from qwen_grpo.sentiment import build_prompts

B, G = 2, 4
SENTENCES = ["A gorgeous, witty film.", "Two hours of my life I will never get back, and the jokes are stale."]


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL, padding_side="left")


@pytest.fixture(scope="module")
def tiny(tok):
    """Randomly initialised 2-layer Qwen3 with the real vocabulary: real generate(), no download."""
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg).float().eval()
    model.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=tok.pad_token_id)
    return model


@pytest.fixture(scope="module")
def rollouts(tiny, tok):
    torch.manual_seed(0)
    return generate_rollouts(tiny, tok, build_prompts(tok, SENTENCES), G=G, max_new_tokens=6)


def test_shapes(rollouts):
    P, R = rollouts["prompt_len"], rollouts["response_ids"].shape[1]
    assert rollouts["input_ids"].shape == rollouts["attention_mask"].shape == (B * G, P + R)
    assert rollouts["response_ids"].shape == rollouts["completion_mask"].shape == (B * G, R)
    assert torch.equal(rollouts["input_ids"][:, P:], rollouts["response_ids"])


def test_prompt_grouping(rollouts, tok):
    batch = tok(build_prompts(tok, SENTENCES), return_tensors="pt", padding=True, add_special_tokens=False)
    P = rollouts["prompt_len"]
    assert torch.equal(rollouts["input_ids"][:, :P], batch["input_ids"].repeat_interleave(G, dim=0))
    assert torch.equal(rollouts["attention_mask"][:, :P], batch["attention_mask"].repeat_interleave(G, dim=0))
    assert batch["attention_mask"][1].all() and not batch["attention_mask"][0].all()  # row 0 is left-padded


class ScriptedModel:
    """Stands in for generate(): appends fixed responses, so EOS placement is known exactly."""

    device = torch.device("cpu")
    generation_config = GenerationConfig(eos_token_id=EOS_IDS)

    def __init__(self, responses):
        self.responses = torch.tensor(responses)

    def generate(self, input_ids, attention_mask, num_return_sequences, **kwargs):
        prompts = input_ids.repeat_interleave(num_return_sequences, dim=0)
        return torch.cat([prompts, self.responses], dim=1)


def test_eos_included_and_padding_excluded(tok):
    im_end, pad = EOS_IDS[0], tok.pad_token_id
    assert pad in EOS_IDS
    responses = [
        [7, 8, im_end, pad, pad],
        [7, pad, pad, pad, pad],   # sequence ended on <|endoftext|>, which is also the pad id
        [im_end, pad, pad, pad, pad],
        [7, 8, 9, 10, 11],         # truncated: no EOS
    ]
    out = generate_rollouts(ScriptedModel(responses), tok, ["a", "b"], G=2)
    expected = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    assert torch.equal(out["completion_mask"], expected)
    assert torch.equal(out["attention_mask"][:, out["prompt_len"]:], expected.long())


def test_logprobs_shape(tiny, rollouts):
    with torch.no_grad():
        logp = compute_logprobs(tiny, rollouts)
    assert logp.shape == rollouts["response_ids"].shape
    assert torch.isfinite(logp[rollouts["completion_mask"]]).all()


def test_logprobs_repeatable(tiny, rollouts):
    with torch.no_grad():
        a = compute_logprobs(tiny, rollouts)
        b = compute_logprobs(tiny, rollouts)
    assert torch.equal(a, b)


def test_logprobs_keep_autograd(tiny, rollouts):
    with torch.no_grad():
        assert not compute_logprobs(tiny, rollouts).requires_grad
    tiny.zero_grad()
    logp = compute_logprobs(tiny, rollouts)
    assert logp.requires_grad
    logp[rollouts["completion_mask"]].sum().backward()
    grad = tiny.model.layers[0].self_attn.q_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0
    tiny.zero_grad()


def test_left_padded_teacher_forcing_matches_generate_and_unpadded(tiny, tok):
    # Independent of response_logprobs' indexing: generate()'s own raw logits score each token it emitted.
    batch = tok(build_prompts(tok, SENTENCES), return_tensors="pt", padding=True, add_special_tokens=False)
    with torch.no_grad():
        out = tiny.generate(**batch, do_sample=False, max_new_tokens=6, output_logits=True, return_dict_in_generate=True)
    P = batch["input_ids"].shape[1]
    resp = out.sequences[:, P:]
    cmask = response_mask(resp, EOS_IDS)
    r = {"input_ids": out.sequences, "prompt_len": P, "attention_mask": torch.cat([batch["attention_mask"], cmask.long()], 1)}
    with torch.no_grad():
        got = compute_logprobs(tiny, r)
        gen = torch.stack([torch.log_softmax(l.float(), -1).gather(-1, resp[:, t : t + 1]).squeeze(-1) for t, l in enumerate(out.logits)], 1)
        torch.testing.assert_close(got[cmask], gen[cmask], atol=1e-5, rtol=0)
        n_pad = int((batch["attention_mask"][0] == 0).sum())
        assert n_pad > 0
        alone = {"input_ids": out.sequences[:1, n_pad:], "prompt_len": P - n_pad, "attention_mask": r["attention_mask"][:1, n_pad:]}
        torch.testing.assert_close(compute_logprobs(tiny, alone)[cmask[:1]], got[:1][cmask[:1]], atol=1e-5, rtol=0)
