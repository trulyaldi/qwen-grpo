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


# --- selective vocabulary logits ---------------------------------------------------------------------------------

def full_logit_logprobs(model, rollouts):
    """compute_logprobs as of commit 9bd4500 (the reference): logits for all T columns, then the response slice."""
    from qwen_grpo.grpo import response_logprobs
    mask = rollouts["attention_mask"]
    position_ids = (mask.cumsum(dim=1) - 1).clamp(min=0)
    logits = model(input_ids=rollouts["input_ids"], attention_mask=mask, position_ids=position_ids, use_cache=False).logits
    return response_logprobs(logits, rollouts["input_ids"], rollouts["prompt_len"])


def uneven(rollouts, lengths):
    """The rollouts with responses cut to `lengths` valid tokens (EOS-truncation pattern), masks consistent."""
    R = rollouts["response_ids"].shape[1]
    cmask = torch.arange(R)[None, :] < torch.tensor(lengths)[:, None]
    P = rollouts["prompt_len"]
    return {**rollouts, "completion_mask": cmask,
            "attention_mask": torch.cat([rollouts["attention_mask"][:, :P], cmask.long()], 1)}


@pytest.mark.parametrize("dtype, tol", [(torch.float64, 1e-12), (torch.float32, 2e-6)])
def test_selective_logits_match_the_full_logit_reference(tiny, rollouts, dtype, tol):
    """Same log-probs, GRPO loss and LoRA gradients as the full-logit implementation, with left padding, unequal
    response lengths and clipping active; lm_head produces only R + 1 columns."""
    import copy

    from peft import get_peft_model

    from qwen_grpo.grpo import grpo_loss
    from qwen_grpo.model import lora_config
    torch.manual_seed(1)
    model = get_peft_model(copy.deepcopy(tiny).to(dtype), lora_config(["model.layers.1.self_attn.q_proj",
                                                                       "model.layers.1.self_attn.o_proj"], 4, 8))
    for n, p in model.named_parameters():
        if "lora_" in n:
            torch.nn.init.normal_(p, std=0.2)
    model = model.to(dtype)
    r = uneven(rollouts, [6, 1, 3, 6, 2, 6, 4, 5])
    assert (rollouts["attention_mask"][:, :r["prompt_len"]] == 0).any()  # some prompts are left-padded
    shapes = []
    hook = model.get_output_embeddings().register_forward_hook(lambda m, i, o: shapes.append(tuple(o.shape)))
    results = []
    for fn in (full_logit_logprobs, compute_logprobs):
        model.zero_grad()
        logp = fn(model, r)
        old = (logp.detach() + 0.3 * torch.randn(logp.shape, generator=torch.Generator().manual_seed(0), dtype=dtype))
        adv = torch.linspace(-1.5, 1.5, logp.shape[0], dtype=dtype)
        loss = grpo_loss(logp, old, adv, r["completion_mask"])
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        results.append((logp.detach(), loss.detach(), grads))
    hook.remove()
    T, R, V = r["input_ids"].shape[1], r["response_ids"].shape[1], model.get_output_embeddings().weight.shape[0]
    assert shapes == [(B * G, T, V), (B * G, R + 1, V)]  # the [N, T, V] logits are never materialized
    (a, la, ga), (b, lb, gb) = results
    m = r["completion_mask"]
    torch.testing.assert_close(b[m], a[m], atol=tol, rtol=0)
    torch.testing.assert_close(lb, la, atol=tol, rtol=0)
    assert ga.keys() == gb.keys() and len(ga) == 4 and all(ga[k].abs().sum() > 0 for k in ga)
    for k in ga:
        torch.testing.assert_close(gb[k], ga[k], atol=tol * 10, rtol=tol * 10)
