import math

import pytest
import torch

from grpo import group_advantages, grpo_loss, response_logprobs, response_mask

torch.set_default_dtype(torch.float64)
EOS = [151645, 151643]


# --- advantages ---------------------------------------------------------------

def test_advantages_hand_computed():
    adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    std = math.sqrt(0.25 * 0.75)
    expected = torch.tensor([[0.75, -0.25, -0.25, -0.25]]) / (std + 1e-4)
    torch.testing.assert_close(adv, expected)


def test_advantages_equal_rewards_are_zero_not_nan():
    adv = group_advantages(torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(adv, torch.zeros(2, 4))


def test_advantages_normalized_per_group():
    adv = group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]]))
    torch.testing.assert_close(adv[1], -adv[0])  # rewards 1 - r flip every advantage
    torch.testing.assert_close(adv.mean(dim=1), torch.zeros(2))


# --- response mask ------------------------------------------------------------

@pytest.mark.parametrize(
    "ids, expected",
    [
        ([7, 8, 151645, 151643, 151643], [1, 1, 1, 0, 0]),  # pad == an EOS id after the first EOS
        ([7, 8, 151643, 151643, 151643], [1, 1, 1, 0, 0]),  # first EOS is the pad id itself
        ([151645, 151643, 151643], [1, 0, 0]),
        ([7, 8, 9], [1, 1, 1]),                              # truncated: no EOS
    ],
)
def test_response_mask(ids, expected):
    mask = response_mask(torch.tensor([ids]), EOS)
    assert mask.tolist() == [[bool(x) for x in expected]]


# --- token alignment ----------------------------------------------------------

def test_response_logprobs_matches_explicit_loop():
    torch.manual_seed(0)
    N, P, R, V = 3, 5, 4, 11
    ids = torch.randint(0, V, (N, P + R))
    ids[0, :2] = 0  # left padding: prompts of different true lengths
    ids[1, :1] = 0
    logits = torch.randn(N, P + R, V)

    got = response_logprobs(logits, ids, P)

    expected = torch.empty(N, R)
    for n in range(N):
        for j in range(R):
            t = P + j  # column of the j-th response token
            expected[n, j] = torch.log_softmax(logits[n, t - 1], dim=-1)[ids[n, t]]
    torch.testing.assert_close(got, expected)


def test_response_logprobs_picks_previous_position():
    # logits at column t put all mass on the token at column t+1, so a correct gather gives log p = 0
    N, P, R, V = 2, 3, 4, 9
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 0, 8, 7, 6, 5, 4]])
    logits = torch.full((N, P + R, V), -1e4)
    for n in range(N):
        for t in range(P + R - 1):
            logits[n, t, ids[n, t + 1]] = 0.0
    torch.testing.assert_close(response_logprobs(logits, ids, P), torch.zeros(N, R))


# --- loss ---------------------------------------------------------------------

def test_reinforce_form_at_ratio_one_and_per_sequence_weights():
    # G = 2 sequences of length 2 and 8, A = +1, -1. At ratio 1 the gradient is
    # dL/dlogp_{i,t} = -A_i / (G |o_i|): -0.25 per token of o_1, +0.0625 per token of o_2.
    logp = torch.full((2, 8), -1.0, requires_grad=True)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[0, :2] = True
    mask[1, :] = True
    loss = grpo_loss(logp, logp.detach().clone(), torch.tensor([1.0, -1.0]), mask)
    loss.backward()

    expected = torch.zeros(2, 8)
    expected[0, :2] = -0.25
    expected[1, :] = 0.0625
    torch.testing.assert_close(logp.grad, expected)
    torch.testing.assert_close(loss, torch.tensor(-(1.0 - 1.0) / 2))


def test_masked_positions_ignored_even_if_nonfinite():
    logp = torch.tensor([[-1.0, -2.0, float("nan"), float("-inf")]], requires_grad=True)
    old = torch.tensor([[-1.1, -1.9, float("-inf"), float("nan")]])
    mask = torch.tensor([[True, True, False, False]])
    loss = grpo_loss(logp, old, torch.tensor([0.7]), mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(logp.grad[0, 2:], torch.zeros(2))


def test_zero_advantages_give_zero_loss_and_gradient():
    logp = torch.randn(4, 6, requires_grad=True)
    loss = grpo_loss(logp, torch.randn(4, 6), torch.zeros(4), torch.ones(4, 6, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logp.grad, torch.zeros(4, 6))


def test_loss_gradient_matches_finite_differences():
    # Ratios spread over [0.6, 1.4] on both sides of the clip boundaries, kept >= 1e-2
    # away from the kinks at 1 +/- eps so central differences do not straddle one.
    torch.manual_seed(0)
    eps = 0.2
    N, R = 6, 5
    ratio = torch.empty(N, R).uniform_(0.6, 1.4)
    near_kink = ((ratio - (1 - eps)).abs() < 1e-2) | ((ratio - (1 + eps)).abs() < 1e-2)
    ratio[near_kink] += 0.03
    old = torch.randn(N, R) - 2.0
    logp = (old + ratio.log()).requires_grad_(True)
    adv = torch.tensor([1.3, -0.7, 0.4, -1.1, 0.9, -0.2])
    mask = torch.ones(N, R, dtype=torch.bool)
    mask[1, 3:] = False
    mask[4, 1:] = False

    assert torch.autograd.gradcheck(lambda lp: grpo_loss(lp, old, adv, mask, eps), (logp,))


def test_empty_response_raises():
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(ValueError):
        grpo_loss(torch.randn(2, 4), torch.randn(2, 4), torch.tensor([1.0, -1.0]), mask)


def test_no_gradient_through_old_logp():
    logp = torch.randn(2, 4, requires_grad=True)
    old = torch.randn(2, 4, requires_grad=True)
    loss = grpo_loss(logp, old, torch.tensor([1.0, -1.0]), torch.ones(2, 4, dtype=torch.bool))
    loss.backward()
    assert logp.grad is not None and logp.grad.abs().sum() > 0
    assert old.grad is None


CLIP_CASES = [(sign, region) for sign in ("A>0", "A<0") for region in ("rho<1-eps", "inside", "rho>1+eps")]


@pytest.mark.parametrize("sign, region", CLIP_CASES)
def test_clipping_cases(sign, region):
    # dL/dlogp = -A rho where the unclipped term is the active min, else 0 (clip_eps = 0.2)
    A = 1.0 if sign == "A>0" else -1.0
    rho = {"rho<1-eps": 0.7, "inside": 1.0, "rho>1+eps": 1.3}[region]
    clipped = (A > 0 and region == "rho>1+eps") or (A < 0 and region == "rho<1-eps")

    old = torch.tensor([[-2.0]])
    logp = (old + math.log(rho)).requires_grad_(True)
    loss = grpo_loss(logp, old, torch.tensor([A]), torch.ones(1, 1, dtype=torch.bool), clip_eps=0.2)
    loss.backward()

    expected = torch.tensor([[0.0 if clipped else -A * rho]])
    torch.testing.assert_close(logp.grad, expected, rtol=1e-12, atol=1e-12)
