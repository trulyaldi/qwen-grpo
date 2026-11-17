"""Checkpointed L2L backward (qwen_grpo/execution/backward.py) against resident autograd, on tiny random Qwen3 models.

Tolerance 0 throughout (research/l2l/tolerances.yaml, gradient_equivalence): same device, same weight bytes, same
kernels, and the recomputation reproduces the forward bitwise. CPU tests use a pageable store and CPU buffers; CUDA
tests are skipped without a GPU. The full grid (both buffer counts, per-parameter errors and norms) is the lab case
gradient_equivalence.
"""
import pytest
import torch

from qwen_grpo.backward_lab import (DetachBoundary, SavedTensors, build, microbatches, placements, run_resident,
                                    step)
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backward import CheckpointedLayerExecutor, checksum
from qwen_grpo.execution.layer_stream import place_resident
from qwen_grpo.execution.qwen3_forward import explicit_forward, unwrap
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEVICES = ["cpu", pytest.param("cuda", marks=CUDA)]
DTYPES = [torch.float32, torch.bfloat16]


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)  # tests/test_grpo.py sets float64 at import


def pair(shape="4b-like", attn="eager", dtype=torch.float32, layers=(5,), device="cpu", n_buf=1, executor=None):
    """(resident reference on device, L2L candidate with the same weights, its store, its executor)."""
    with F.isolated():
        ref, cand = build(shape, attn, dtype, list(layers)), build(shape, attn, dtype, list(layers))
    cand.load_state_dict(ref.state_dict())
    store = WeightStore.build(cand, "pinned" if device == "cuda" else "pageable")
    place_resident(cand, store, device)
    ex = (executor or CheckpointedLayerExecutor)(store, gpu_buffers(n_buf, store.max_slab_bytes, device))
    return ref.to(device), cand, store, ex


def l2l_steps(cand, ex, mbs):
    """Like run_resident, for the candidate: per micro-batch loss, hidden grads and .grad snapshots."""
    cand.zero_grad(set_to_none=True)
    params = {n: p for n, p in cand.named_parameters() if p.requires_grad}
    out = []
    for mb in mbs:
        with F.isolated():
            loss, logits, hidden = step(cand, ex, mb)
        out.append({"loss": loss, "logits": logits, "hidden": hidden,
                    "grads": {n: None if p.grad is None else p.grad.clone() for n, p in params.items()}})
    return out


def mismatches(ref, got):
    """Names of every quantity where the candidate is missing a gradient or differs from the reference at all."""
    bad = []
    for k, (r, g) in enumerate(zip(ref, got)):
        for q in ("loss", "logits"):
            if not torch.equal(r[q], g[q]):
                bad.append(f"mb{k + 1}.{q}")
        for n, rg in r["grads"].items():
            gg = g["grads"][n]
            if gg is None or not torch.equal(rg, gg):
                bad.append(f"mb{k + 1}.grads.{n}")
        for i, rh in r["hidden"].items():
            gh = g["hidden"].get(i)
            if gh is None or not torch.equal(rh, gh):
                bad.append(f"mb{k + 1}.hidden.{i}")
    return bad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", list(F.SHAPES))
@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("placement", ["final", "last2", "middle", "first_last", "distributed"])
def test_lora_and_hidden_gradients_equal_resident(device, shape, attn, dtype, placement):
    layers = placements(F.SHAPES[shape]["num_hidden_layers"])[placement]
    ref, cand, store, ex = pair(shape, attn, dtype, layers, device)
    mbs = microbatches(device)
    with F.isolated():
        want = run_resident(ref, mbs)
    got = l2l_steps(cand, ex, mbs)
    assert mismatches(want, got) == []
    for r in want:  # nontrivial: every adapter (A and B) has a nonzero gradient, and hidden grads exist
        assert all(g is not None and g.abs().max() > 0 for g in r["grads"].values())
        assert r["hidden"] and all(h.abs().max() > 0 for h in r["hidden"].values())
    assert not torch.equal(want[0]["grads"][next(iter(want[0]["grads"]))],
                           want[1]["grads"][next(iter(want[1]["grads"]))])  # accumulation added something
    assert store.verify() == []
    store.close()


def test_accumulated_microbatches_are_the_sum_of_single_ones():
    ref, cand, store, ex = pair(layers=(0, 2, 5))
    mbs = microbatches("cpu")
    acc = l2l_steps(cand, ex, mbs)[-1]["grads"]
    single = [l2l_steps(cand, ex, [mb])[0]["grads"] for mb in mbs]
    for n, g in acc.items():
        assert torch.equal(g, single[0][n] + single[1][n]), n


@pytest.mark.parametrize("device", DEVICES)
def test_backward_reloads_exactly_the_needed_layers_in_reverse_order_and_restores_the_store(device):
    _, cand, store, ex = pair(layers=(2, 4), device=device, n_buf=2)
    with F.isolated():
        step(cand, ex, microbatches(device)[0])
    assert [t["layer"] for t in ex.trace if t["phase"] == "forward"] == list(range(6))
    assert [t["layer"] for t in ex.trace if t["phase"] == "backward"] == [5, 4, 3, 2]  # nothing below layer 2 needs grad
    assert all(t["buffer"] == t["layer"] % 2 for t in ex.trace)
    assert all(e.param.device.type == "cpu" and e.param.data_ptr() == s.data.data_ptr() + e.offset
               for s in store.slabs for e in s.entries)
    assert store.verify() == []


def test_the_forward_graph_saves_no_weight_buffer_view_and_less_than_resident():
    ref, cand, store, ex = pair(layers=range(6))
    mb = microbatches("cpu")[0]
    l2l, resident = SavedTensors(ex.buffers), SavedTensors([])
    with F.isolated():
        step(cand, ex, mb, l2l)
        step(ref, None, mb, resident)
    assert l2l.in_buffer == 0
    assert l2l.bytes < resident.bytes / 4  # boundaries [N, T, H] per layer instead of every intermediate


def test_no_grad_uses_the_forward_only_path_and_records_nothing():
    ref, cand, store, ex = pair(layers=(3,))
    ids, mask = microbatches("cpu")[0][0]["input_ids"], microbatches("cpu")[0][0]["attention_mask"]
    with torch.no_grad():
        got = explicit_forward(cand, ids, mask, layer_executor=ex)
        want = explicit_forward(ref, ids, mask)
    assert torch.equal(got, want) and got.grad_fn is None
    assert {t["phase"] for t in ex.trace} == {"forward"}


@pytest.mark.parametrize("device", DEVICES)
def test_optimizer_step_after_l2l_backward_equals_resident(device):
    ref, cand, store, ex = pair(layers=(0, 5), device=device)
    params = {n: p for n, p in cand.named_parameters() if p.requires_grad}
    ids = {n: id(p) for n, p in params.items()}
    opt_ref = torch.optim.AdamW([p for p in ref.parameters() if p.requires_grad], lr=F.LR, weight_decay=0.0)
    opt = torch.optim.AdamW(params.values(), lr=F.LR, weight_decay=0.0)
    mbs = microbatches(device)
    with F.isolated():
        run_resident(ref, mbs)
    l2l_steps(cand, ex, mbs)
    opt_ref.step()
    opt.step()
    assert {n: id(p) for n, p in cand.named_parameters() if p.requires_grad} == ids
    assert [p for g in opt.param_groups for p in g["params"]] == list(params.values())
    trained = {n: p for n, p in ref.named_parameters() if p.requires_grad}
    assert all(torch.equal(trained[n], p) for n, p in params.items())


# --- fault injection: must be caught ------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_a_detached_boundary_fails_the_gradient_comparison(dtype):
    ref, cand, store, ex = pair(dtype=dtype, layers=(0, 2, 5),
                                executor=lambda s, b: DetachBoundary(s, b, at=3))
    mbs = microbatches("cpu")
    with F.isolated():
        want = run_resident(ref, mbs)
    got = l2l_steps(cand, ex, mbs)
    bad = mismatches(want, got)
    assert bad  # the gradient test fails
    lost = {n for n in want[0]["grads"] if f"mb1.grads.{n}" in bad}
    assert lost == {n for n in want[0]["grads"] if ".layers.0." in n or ".layers.2." in n}  # all below the cut
    assert all(got[0]["grads"][n] is None for n in lost)  # missing, i.e. no silent zero
    # layers 3 and 4 have no adapter, so the cut leaves nothing below layer 5 that receives a gradient: dL/dhidden_i is
    # missing for every i < 5 (outputs 0..2 still require grad but get none; 3 and 4 no longer require grad)
    assert {b for b in bad if b.startswith("mb1.hidden.")} == {f"mb1.hidden.{i}" for i in range(5)}


def test_a_detached_boundary_below_every_adapter_changes_nothing():
    ref, cand, store, ex = pair(layers=(5,), executor=lambda s, b: DetachBoundary(s, b, at=3))
    mbs = microbatches("cpu")
    with F.isolated():
        want = run_resident(ref, mbs)
    assert mismatches(want, l2l_steps(cand, ex, mbs)) == []  # the graph below layer 5 carries no gradient anyway


# --- fail closed --------------------------------------------------------------------------------------------------

def forward_loss(cand, ex):
    rollouts = microbatches("cpu")[0][0]
    return explicit_forward(cand, rollouts["input_ids"], rollouts["attention_mask"], layer_executor=ex).float().square().mean()


def test_refuses_a_model_state_change_between_forward_and_backward():
    _, cand, store, ex = pair(layers=(4, 5))
    loss = forward_loss(cand, ex)
    with cand.disable_adapter(), pytest.raises(RuntimeError, match="recomputed output differs"):
        loss.backward()
    assert store.verify() == []
    assert all(e.param.device.type == "cpu" for s in store.slabs for e in s.entries)


def test_refuses_training_mode():
    _, cand, store, ex = pair()
    cand.train()
    with pytest.raises(RuntimeError, match="training mode"):
        forward_loss(cand, ex)


def test_refuses_double_backward():
    _, cand, store, ex = pair()
    params = [p for p in cand.parameters() if p.requires_grad]
    grads = torch.autograd.grad(forward_loss(cand, ex), params, create_graph=True)
    with pytest.raises(RuntimeError, match="differentiate"):
        torch.autograd.grad(sum(g.sum() for g in grads), params)


def test_refuses_keyword_tensors_that_require_grad():
    _, cand, store, ex = pair()
    layer = unwrap(cand)[0].layers[5]
    h = torch.zeros(1, 3, cand.config.hidden_size, requires_grad=True)
    cos = torch.ones(1, 3, 8, requires_grad=True)
    with pytest.raises(RuntimeError, match="keyword tensors require grad"):
        ex.execute(5, layer, h, attention_mask=None, position_embeddings=(cos, torch.zeros(1, 3, 8)))


def test_weights_return_to_the_store_when_the_backward_recomputation_raises(monkeypatch):
    _, cand, store, ex = pair()
    loss = forward_loss(cand, ex)
    layer = unwrap(cand)[0].layers[5]
    monkeypatch.setattr(layer, "forward", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        loss.backward()
    assert store.verify() == []
    assert all(e.param.device.type == "cpu" for s in store.slabs for e in s.entries)


def test_checksum_is_exact_on_bit_patterns():
    x = torch.randn(5, 7)
    assert torch.equal(checksum(x), checksum(x.clone()))
    y = x.clone()
    y.view(-1)[3] = torch.nextafter(y.view(-1)[3], torch.tensor(float("inf")))  # one ulp
    assert not torch.equal(checksum(x), checksum(y))
    assert checksum(x.bfloat16()).dtype == torch.int64
