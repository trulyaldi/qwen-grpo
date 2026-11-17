"""Chunked output head (qwen_grpo/execution/output_head.py) against the full-logits reference, and its integration with
GRPO training (training.prepare / grpo_update with a log-prob function) and with L2L. The full grid with per-tensor
errors, memory and time is the lab case logprob_equivalence; thresholds as in research/l2l/tolerances.yaml."""
import pytest
import torch

from qwen_grpo import training
from qwen_grpo.backward_lab import build
from qwen_grpo.execution import fixtures as F
from qwen_grpo.execution.backward import CheckpointedLayerExecutor
from qwen_grpo.execution.layer_stream import place_resident
from qwen_grpo.execution.output_head import output_head, selected_logprobs, streamed_logprobs
from qwen_grpo.execution.qwen3_forward import unwrap
from qwen_grpo.execution.weight_store import WeightStore, gpu_buffers
from qwen_grpo.head_lab import Allocations, head_inputs, head_memory, rollouts
from qwen_grpo.policy import compute_logprobs

TOL = {torch.float32: (1e-4, 1e-4), torch.bfloat16: (1e-4, 2 ** -6)}  # (logp atol, gradient rtol)


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)  # tests/test_grpo.py sets float64 at import


def both(h, w, t, g, chunk):
    x = h.detach().clone().requires_grad_()
    lp = selected_logprobs(x, w, t, chunk)
    (lp * g).sum().backward()
    return lp.detach(), x.grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("V, chunks", [(1, [1, 5]), (64, [1, 7, 64, 100]), (1000, [7, 256, 999]),
                                       (151936, [8192, 32768])])
def test_chunked_equals_full_logits_within_tolerance(dtype, V, chunks):
    h, w, t, g = head_inputs(3, 5, 32, V, dtype, "cpu")
    t[0, 0], t[0, 1] = 0, V - 1  # first and last vocabulary rows (the last lies in the remainder chunk)
    ref_lp, ref_gh = both(h, w, t, g, None)
    atol, rtol = TOL[dtype]
    for c in chunks:
        lp, gh = both(h, w, t, g, c)
        assert lp.dtype == ref_lp.dtype == torch.float32 and gh.dtype == dtype
        assert (lp - ref_lp).abs().max() <= atol, c
        assert (gh.float() - ref_gh.float()).abs().max() <= rtol * ref_gh.float().abs().max(), c


def test_gradcheck_in_float64():
    h, w, t, _ = head_inputs(2, 3, 6, 23, torch.float64, "cpu")
    x = h.detach().requires_grad_()
    assert torch.autograd.gradcheck(lambda y: selected_logprobs(y, w, t, 5), (x,))
    ref, got = both(h, w, t, torch.ones(2, 3, dtype=torch.float64), None), both(h, w, t, torch.ones(2, 3, dtype=torch.float64), 4)
    assert torch.allclose(ref[0], got[0], atol=1e-12) and torch.allclose(ref[1], got[1], atol=1e-12)


def test_normalized_over_the_whole_vocabulary_and_stable_for_large_logits():
    h, w, t, _ = head_inputs(1, 1, 16, 50, torch.float32, "cpu")
    every = torch.arange(50).view(1, 50)
    lp = selected_logprobs(h.expand(1, 50, 16), w, every, 7)  # every token of one position: probabilities sum to 1
    assert torch.allclose(lp.exp().sum(), torch.tensor(1.0), atol=1e-5) and (lp <= 0).all()
    big = h * 1e3  # logits ~ 1e3: exp overflows without the running maximum
    lp, gh = both(big, w, t, torch.ones(1, 1), 7)
    ref = torch.log_softmax(big.double() @ w.double().T, -1).gather(-1, t.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(lp).all() and torch.isfinite(gh).all() and (lp.double() - ref).abs().max() < 1e-3


def test_refuses_a_trainable_head_bad_targets_and_bad_chunks():
    h, w, t, _ = head_inputs(1, 2, 8, 10, torch.float32, "cpu")
    with pytest.raises(RuntimeError, match="requires grad"):
        selected_logprobs(h, torch.nn.Parameter(w.detach().clone()), t, 3)
    with pytest.raises(ValueError, match="outside"):
        selected_logprobs(h, w, t + 10, 3)
    with pytest.raises(ValueError, match="chunk_size"):
        selected_logprobs(h, w, t, 0)


def test_chunked_head_memory_is_bounded_by_the_chunk_not_the_vocabulary():
    N, R, H, V, C = 2, 8, 16, 20000, 512
    h, w, t, g = head_inputs(N, R, H, V, torch.float32, "cpu")
    full, chunked = head_memory(h, w, t, g, None), head_memory(h, w, t, g, C)
    assert full["largest_allocation_bytes"] >= N * R * V * 4 and full["saved_bytes"] >= N * R * V * 4
    assert chunked["largest_allocation_bytes"] <= max(N * R * C, N * R * H, C * H) * 4
    assert chunked["saved_bytes"] <= N * R * H * 4 + N * R * 12  # hidden, targets, logsumexp: independent of V
    with Allocations() as a:
        torch.ones(3, 5)
    assert a.largest == 60


# --- model level ------------------------------------------------------------------------------------------------------

def model(V=1000, dtype=torch.float32, layers=(0, 3)):
    with F.isolated():
        return build("0.6b-like", "eager", dtype, list(layers), vocab=V)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_head_path_is_compute_logprobs_bitwise_and_chunked_is_close(dtype):
    m = model(dtype=dtype)
    r, _ = rollouts(1000, 4)
    with F.isolated(), torch.no_grad():
        ref = compute_logprobs(m, r)
        assert torch.equal(streamed_logprobs(m, r), ref)
        assert (streamed_logprobs(m, r, 7) - ref).abs().max() <= 1e-4


def test_l2l_with_the_chunked_head_equals_resident_chunked_bitwise():
    ref, cand = model(), model()
    cand.load_state_dict(ref.state_dict())
    store = WeightStore.build(cand, "pageable")
    place_resident(cand, store, "cpu")
    ex = CheckpointedLayerExecutor(store, gpu_buffers(1, store.max_slab_bytes, "cpu"))
    r, _ = rollouts(1000, 4)
    out = {}
    for name, m, e in (("resident", ref, None), ("l2l", cand, ex)):
        with F.isolated():
            lp = streamed_logprobs(m, r, 7, e)
            lp.sum().backward()
        out[name] = (lp.detach(), [p.grad for p in m.parameters() if p.requires_grad])
    assert torch.equal(out["resident"][0], out["l2l"][0])
    assert all(torch.equal(a, b) for a, b in zip(out["resident"][1], out["l2l"][1]))
    assert store.verify() == []


def test_output_head_checks_the_tie_and_the_module():
    m = model()
    decoder, head = unwrap(m)
    assert output_head(m) is decoder.embed_tokens.weight
    head.weight = torch.nn.Parameter(head.weight.detach().clone(), requires_grad=False)  # untie
    with pytest.raises(RuntimeError, match="not the embedding"):
        output_head(m)


@pytest.mark.parametrize("mb", [None, 2])
def test_grpo_training_with_the_chunked_head_keeps_the_old_current_check_and_tracks_the_reference(mb):
    ref, cand = model(), model()
    cand.load_state_dict(ref.state_dict())
    r, adv = rollouts(1000, 4)
    rewards = torch.tensor(F.REWARDS, dtype=torch.float32)
    chunked = lambda m, c: streamed_logprobs(m, c, 7)
    logs = {}
    for name, m, fn in (("full", ref, None), ("chunked", cand, chunked)):
        params = {n: p for n, p in m.named_parameters() if p.requires_grad}
        opt = torch.optim.AdamW(params.values(), lr=5e-2, weight_decay=0.0)
        with F.isolated():
            _, old = training.prepare(m, r, rewards, mb, logprobs=fn)
            logs[name] = [training.grpo_update(m, params, opt, r, adv, old, 0.2, k, mb, logprobs=fn) for k in range(3)]
    assert logs["chunked"][0]["init_diff"] == 0.0 <= training.INIT_TOL  # unchanged check, exactly satisfied
    assert abs(logs["chunked"][0]["loss"] - logs["full"][0]["loss"]) <= 4e-4
    assert any(log["clip_frac"] > 0 for log in logs["full"][1:])  # ratios moved and clipped


def test_mismatched_old_and_current_log_prob_functions_trip_the_unchanged_init_check():
    m = model()
    r, adv = rollouts(1000, 4)
    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    with F.isolated():
        _, old = training.prepare(m, r, torch.tensor(F.REWARDS, dtype=torch.float32))
    raw = lambda mm, c: streamed_logprobs(mm, c, 7) + 1e-3  # a path that disagrees with the one prepare() used
    with pytest.raises(RuntimeError, match="old/current log-prob mismatch"):
        training.grpo_update(m, params, torch.optim.SGD(params.values(), lr=0.0), r, adv, old, 0.2, 0, logprobs=raw)


def test_the_streamed_head_runs_the_same_matmul_shapes_as_compute_logprobs():
    """Bitwise equality across machines needs identical shapes, not only identical math: some BLAS / CPU combinations
    round a row differently depending on the number of rows (a 1e-7 gradient difference was seen on a laptop CPU when
    the streamed head used R rows and the reference R+1). Forward and backward head matmuls are recorded and compared."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class HeadShapes(TorchDispatchMode):
        def __init__(self, V):
            super().__init__()
            self.V, self.shapes = V, []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.__name__.split(".")[0] in ("mm", "addmm", "bmm", "linear", "matmul"):
                shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
                if any(self.V in s for s in shapes):
                    self.shapes.append((func.__name__, shapes))
            return func(*args, **(kwargs or {}))

    m = model()
    r, _ = rollouts(1000, 4)
    seen = {}
    for name, fn in (("reference", lambda: compute_logprobs(m, r)), ("streamed", lambda: streamed_logprobs(m, r))):
        rec = HeadShapes(1000)
        with F.isolated(), rec:
            fn().sum().backward()
        seen[name] = rec.shapes
        m.zero_grad(set_to_none=True)
    assert seen["reference"] and seen["streamed"] == seen["reference"]
