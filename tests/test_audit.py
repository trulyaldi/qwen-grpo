"""Correctness-audit tests: value-level grouping for several (B, G), an end-to-end hand-computed loss through
compute_logprobs on a bigram model (left padding, EOS, truncation, clipping), LR=0 at the grpo_step level,
and tokenization consistency between evaluation and training."""
import math
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, GenerationConfig

from qwen_grpo.grpo import group_advantages, grpo_loss, response_mask
from qwen_grpo.model import EOS_IDS, MODEL
from qwen_grpo.policy import compute_logprobs, generate_rollouts
from qwen_grpo.sentiment import build_prompts
from qwen_grpo.training import decode, group_rewards, grpo_step

POS, NEG = [17456, 44350, EOS_IDS[0]], [97127, 23215, EOS_IDS[0]]
JUNK = 9707


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(MODEL, padding_side="left")


class Scripted:
    """generate() stand-in: row b*G+g gets responses[b][g] (padded with <|endoftext|>), prompts expanded as HF does."""

    device = torch.device("cpu")
    generation_config = GenerationConfig(eos_token_id=EOS_IDS)

    def __init__(self, responses, R):
        self.rows = torch.tensor([r + [EOS_IDS[1]] * (R - len(r)) for group in responses for r in group])

    def generate(self, input_ids, attention_mask, num_return_sequences, **kwargs):
        return torch.cat([input_ids.repeat_interleave(num_return_sequences, dim=0), self.rows], 1)


@pytest.mark.parametrize("B,G,correct", [(1, 4, [1]), (2, 4, [1, 2]), (2, 8, [1, 5])])
def test_grouping_rewards_advantages_and_loss_weights(tok, B, G, correct):
    """Prompt b has gold label POSITIVE for even b; its first correct[b] rows answer correctly, the rest alternate
    between the opposite label and a truncated junk response (|o| = R = 5). Rewards, advantages and the per-token
    loss weight dL/dlogp = -A_i / (N |o_i|) are compared with values computed here by hand."""
    R, N = 5, B * G
    gold = ["POSITIVE" if b % 2 == 0 else "NEGATIVE" for b in range(B)]
    right = {"POSITIVE": POS, "NEGATIVE": NEG}
    wrong = {"POSITIVE": NEG, "NEGATIVE": POS}
    responses = [[right[gold[b]] if g < correct[b] else (wrong[gold[b]] if (g - correct[b]) % 2 == 0 else [JUNK] * R)
                  for g in range(G)] for b in range(B)]
    sentences = [f"Sentence number {b} " + "word " * (3 * b) for b in range(B)]  # different lengths -> left padding
    prompts = build_prompts(tok, sentences)
    r = generate_rollouts(Scripted(responses, R), tok, prompts, G=G, max_new_tokens=R)

    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    for i in range(N):  # each row carries its own prompt and its own scripted response
        assert torch.equal(r["input_ids"][i, : r["prompt_len"]], enc["input_ids"][i // G])
        assert r["response_ids"][i].tolist()[: len(responses[i // G][i % G])] == responses[i // G][i % G]

    rewards = group_rewards(decode(tok, r), gold, G)
    expected_r = [[1.0 if g < correct[b] else 0.0 for g in range(G)] for b in range(B)]
    assert rewards.tolist() == expected_r

    adv = group_advantages(rewards).flatten().double()
    hand = []
    for b in range(B):
        p = correct[b] / G
        s = math.sqrt(p * (1 - p))  # population std of a Bernoulli sample with k/G ones
        hand += [((1 - p) if g < correct[b] else -p) / (s + 1e-4) for g in range(G)]
    torch.testing.assert_close(adv, torch.tensor(hand, dtype=torch.float64), rtol=1e-6, atol=1e-6)

    lengths = [len(responses[i // G][i % G]) for i in range(N)]
    assert r["completion_mask"].sum(1).tolist() == lengths
    logp = torch.randn(N, R, dtype=torch.float64, generator=torch.Generator().manual_seed(0)).requires_grad_()
    grpo_loss(logp, logp.detach(), adv, r["completion_mask"]).backward()
    w = torch.tensor([[-hand[i] / (N * lengths[i]) if t < lengths[i] else 0.0 for t in range(R)] for i in range(N)], dtype=torch.float64)
    torch.testing.assert_close(logp.grad, w, rtol=1e-6, atol=1e-12)


def test_partially_constant_batch_scales_signal_by_mixed_fraction():
    """Convention (shared with TRL): a constant group has A = 0 but stays in the mean over N = B*G sequences,
    so a B=2 batch with one mixed group has exactly half the gradient of that group alone at B=1."""
    torch.manual_seed(0)
    mask = torch.ones(8, 3, dtype=torch.bool)
    logp = torch.randn(8, 3, dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([[1.0, 0, 1, 0], [1.0, 1, 1, 1]])
    grpo_loss(logp, logp.detach(), group_advantages(rewards).flatten().double(), mask).backward()
    alone = logp[:4].detach().clone().requires_grad_()
    grpo_loss(alone, alone.detach(), group_advantages(rewards[:1]).flatten().double(), mask[:4]).backward()
    torch.testing.assert_close(logp.grad[:4], alone.grad / 2)
    assert logp.grad[4:].abs().max() == 0


class Bigram(torch.nn.Module):
    """logits[n, t] = W[input_ids[n, t]]: the next-token distribution depends only on the current token,
    so the reference log-prob of y_t is log_softmax(W[y_{t-1}])[y_t], independent of padding and positions."""

    def __init__(self, W):
        super().__init__()
        self.W = torch.nn.Parameter(W)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(logits=self.W[input_ids])


def test_end_to_end_loss_by_hand_with_padding_eos_truncation_and_clipping():
    V, PAD, EOS = 7, 1, 0  # PAD is also an EOS id, as in Qwen3
    eos_ids = [EOS, PAD]
    prompts = [[2, 3, 4], [5, 6], [3]]                       # unequal lengths -> left padding to P = 3
    responses = [[2, EOS, 5, 5], [4, PAD], [6, 2, 3, 4], [EOS]]  # post-EOS junk; ends on PAD-as-EOS; truncated; EOS only
    owner = [0, 1, 2, 0]                                     # response i continues prompt owner[i]
    P, R = 3, 4
    ids = torch.tensor([[PAD] * (P - len(prompts[o])) + prompts[o] + resp + [PAD] * (R - len(resp)) for o, resp in zip(owner, responses)])
    pmask = torch.tensor([[0] * (P - len(prompts[o])) + [1] * len(prompts[o]) for o in owner])
    cmask = response_mask(ids[:, P:], eos_ids)
    assert cmask.tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]]
    r = {"input_ids": ids, "attention_mask": torch.cat([pmask, cmask.long()], 1), "prompt_len": P}

    g = torch.Generator().manual_seed(0)
    W = torch.randn(V, V, dtype=torch.float64, generator=g)
    W_old = W + 0.3 * torch.randn(V, V, dtype=torch.float64, generator=g)  # nearby old policy: ratios on both sides of the clip
    model = Bigram(W.clone())
    logp = compute_logprobs(model, r)
    with torch.no_grad():
        old = compute_logprobs(Bigram(W_old), r)
    adv = torch.tensor([1.3, -0.7, 0.4, -1.1], dtype=torch.float64)
    loss = grpo_loss(logp, old, adv, cmask, clip_eps=0.2)

    lsm = lambda M, a, b: M[a, b].item() - math.log(sum(math.exp(v) for v in M[a].tolist()))  # log softmax by hand
    total, regions = 0.0, set()
    for i, (o, resp) in enumerate(zip(owner, responses)):
        n = next((k + 1 for k, y in enumerate(resp) if y in eos_ids), len(resp))  # through the first EOS
        prev = prompts[o][-1]
        s = 0.0
        for y in resp[:n]:
            ratio = math.exp(lsm(W, prev, y) - lsm(W_old, prev, y))
            regions.add("low" if ratio < 0.8 else "high" if ratio > 1.2 else "in")
            s += min(ratio * adv[i].item(), min(max(ratio, 0.8), 1.2) * adv[i].item())
            prev = y
        total += s / n
    assert regions == {"low", "in", "high"}, regions  # the example exercises every clip region
    assert loss.item() == pytest.approx(-total / len(responses), abs=1e-12)
    loss.backward()
    assert model.W.grad is not None and torch.isfinite(model.W.grad).all()


class Tiny(torch.nn.Module):
    def __init__(self, V=5, d=4):
        super().__init__()
        torch.manual_seed(0)
        self.emb, self.out = torch.nn.Embedding(V, d), torch.nn.Linear(d, V)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(logits=self.out(self.emb(input_ids)))


def test_lr_zero_step_leaves_parameters_and_policy_bitwise_unchanged():
    model = Tiny()
    params = dict(model.named_parameters())
    opt = torch.optim.AdamW(params.values(), lr=0.0, weight_decay=0.01)
    ids = torch.randint(0, 5, (6, 5), generator=torch.Generator().manual_seed(1))
    r = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "completion_mask": torch.ones(6, 3, dtype=torch.bool), "prompt_len": 2}
    before = {n: p.detach().clone() for n, p in params.items()}
    with torch.no_grad():
        logp0 = compute_logprobs(model, r)
    out = grpo_step(model, params, opt, r, torch.tensor([[1.0, 0, 0], [0.0, 1, 1]]))
    assert out is not None and out["grad_norm"] > 0 and out["update_norm"] == 0.0  # a real gradient, no movement
    assert all(torch.equal(p, before[n]) for n, p in params.items())
    with torch.no_grad():
        assert torch.equal(compute_logprobs(model, r), logp0)


def test_evaluation_and_training_tokenize_prompts_identically(tok):
    prompts = build_prompts(tok, ["A gorgeous, witty film.", "Stale.", "x" * 300])
    default = tok(prompts, return_tensors="pt", padding=True)  # evaluation.generate
    explicit = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)  # rollouts, label_scores
    assert torch.equal(default["input_ids"], explicit["input_ids"]) and torch.equal(default["attention_mask"], explicit["attention_mask"])
