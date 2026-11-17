"""The frozen resident reference (qwen_grpo/execution/fixtures.py, research/l2l/fixtures/*.pt).

Tolerance 0 throughout. The inputs come from seeded RNG draws; the outputs come from the same code with the same
kernels, the same CPU thread count (fixtures.THREADS) and the same reduction order. So any difference means the code,
the libraries or the platform changed, and the reference must then be regenerated deliberately (with a record of why),
not compared within a widened tolerance.
"""
import pytest
import torch

from qwen_grpo.execution import fixtures as F
from qwen_grpo.model import lora_modules

SHAPES = list(F.SHAPES)


def saved(shape):
    """The repository's saved fixture; skips (never passes) where its bitwise reference does not apply."""
    if diff := F.saved_environment_differences(shape):
        pytest.skip(f"the saved fixture was computed in another environment: {diff}")
    return torch.load(F.SAVED_DIR / f"{shape}.pt", weights_only=True)


def inputs(fixture):
    return {k: fixture[k] for k in ("state", "rollouts", "prompts", "rewards", "advantages", "old_logp")}


@pytest.mark.parametrize("shape", SHAPES)
def test_rebuilt_fixture_equals_the_saved_one_bitwise(shape):
    saved_fx = saved(shape)
    with F.isolated():
        _, rebuilt = F.build(shape)
    assert not F.mismatches(inputs(rebuilt), inputs(saved_fx)), F.mismatches(inputs(rebuilt), inputs(saved_fx))[:10]


@pytest.mark.parametrize("shape", SHAPES)
def test_resident_code_reproduces_the_saved_reference_bitwise(shape):
    fx = saved(shape)
    with F.isolated():
        model = F.tiny_model(shape)
    model.load_state_dict(fx["state"], strict=True)
    with F.isolated():
        got = F.reference(model, fx)
    assert not F.mismatches(got, fx["reference"]), F.mismatches(got, fx["reference"])[:10]


@pytest.mark.parametrize("shape", SHAPES)
def test_fixture_exercises_masks_grouping_gradients_and_clipping(shape):
    fx = torch.load(F.path(shape), weights_only=True)
    r, ref = fx["rollouts"], fx["reference"]
    lengths = r["completion_mask"].sum(1)
    assert (lengths == F.R).any() and (lengths < F.R).any() and (lengths == 1).any()      # full, tailed, EOS-only
    assert not torch.isin(r["response_ids"][0], torch.tensor(F.EOS_IDS)).any()              # row 0 truncated
    assert (r["attention_mask"][:, : F.P] == 0).any()                                       # left padding present
    assert (fx["rewards"] != fx["rewards"][:, :1]).any(dim=1).all()                         # every group mixed
    assert all(g.abs().max() > 0 for g in ref["grads"].values())                            # A and B, every module
    assert len(ref["grads"]) == 2 * 3 * 2                                                   # q/v/o of 2 layers, A + B
    assert ref["updates"][0]["init_diff"] == 0 and ref["updates"][0]["clip_frac"] == 0     # ratio exactly 1 first
    assert max(u["clip_frac"] for u in ref["updates"][1:]) > 0                              # clipping binds later
    assert len(F.mismatches(ref["final_lora"], ref["initial_lora"])) == len(ref["initial_lora"])  # every tensor moved


def test_small_base_weight_change_is_detected_but_one_ulp_is_invisible():
    """Controlled fault: one frozen layer-0 weight (0.0046) shifted by 1e-6 must break the bitwise reference.

    Limit of output comparisons: a one-ulp shift of the same weight (~5e-10) changes no element of q_proj's output (its
    contribution is below the output's own rounding), so no downstream comparison can see it. Weight placement must
    therefore be verified on the weights themselves (digests), not only through logits and gradients."""
    name = "base_model.model.model.layers.0.self_attn.q_proj.weight"
    model, fx = F.load("0.6b-like")
    with F.isolated():
        clean = F.reference(model, fx)  # the unperturbed reference in this environment
    for shift, detectable in [("ulp", False), (1e-6, True)]:
        model, saved = F.load("0.6b-like")
        w = dict(model.named_parameters())[name]
        with torch.no_grad():
            w[0, 0] = torch.nextafter(w[0, 0], torch.tensor(float("inf"))) if shift == "ulp" else w[0, 0] + shift
        with F.isolated():
            bad = F.mismatches(F.reference(model, saved), clean)
        if detectable:
            assert "logits" in bad and "logp" in bad and any(k.startswith("grads.") for k in bad), bad
        else:
            assert bad == [], bad


def test_loading_overwrites_every_tensor_and_keeps_adapters_active():
    model, saved = F.load("4b-like")
    assert not F.mismatches({k: v for k, v in model.state_dict().items()}, saved["state"])
    assert len(lora_modules(model)) == 6 and all(m.lora_B["default"].weight.abs().max() > 0 for _, m in lora_modules(model))


def test_isolated_restores_global_state():
    dtype, threads, rng = torch.get_default_dtype(), torch.get_num_threads(), torch.get_rng_state()
    with F.isolated():
        F.build("0.6b-like")
    assert (torch.get_default_dtype(), torch.get_num_threads()) == (dtype, threads)
    assert torch.equal(torch.get_rng_state(), rng)
