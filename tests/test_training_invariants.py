"""Run-level GRPO invariants of the real trainer (CPU, tiny model) that the canonical GPU configs rely on.

num_iterations = 1 (every backend-comparison config, tests/test_profiles.py TRIOS): every update is taken at the sampling policy, so the importance ratio
is exactly 1 on every token and clipping never binds; the objective reduces to the group-baseline policy gradient.
num_iterations > 1: later iterations see moved ratios, so clipping statistics carry information. Also: per-step
phase timings are recorded outside the bitwise-compared rows, and the L2L weight store is released after training."""
import pytest
import torch

from qwen_grpo import trainer
from qwen_grpo.execution import backend as B
from tests.test_backend import L2L_CPU, make_cfg
from tests.test_trainer import sandbox  # noqa: F401


@pytest.fixture(autouse=True)
def float32_default():
    torch.set_default_dtype(torch.float32)


def test_with_one_iteration_clipping_never_binds(sandbox, tmp_path):  # noqa: F811
    sandbox("k1")
    rows = trainer.train(make_cfg(tmp_path, max_steps=4), device="cpu")["training"]
    assert rows and all(r["iteration"] == 0 for r in rows)
    assert all(r["ratio_min"] == r["ratio_max"] == 1.0 and r["clip_frac"] == 0.0 for r in rows)
    assert all(r["init_diff"] <= 1e-6 for r in rows)  # old log-probs = the current policy's, before each update


def test_with_several_iterations_the_ratio_moves(sandbox, tmp_path):  # noqa: F811
    sandbox("k3")
    rows = trainer.train(make_cfg(tmp_path, max_steps=6, num_iterations=3), device="cpu")["training"]
    first = [r for r in rows if r["iteration"] == 0]
    later = [r for r in rows if r["iteration"] > 0]
    assert all(r["ratio_min"] == r["ratio_max"] == 1.0 for r in first)
    assert later and any(r["ratio_min"] != 1.0 or r["ratio_max"] != 1.0 for r in later)


@pytest.mark.parametrize("execution", [None, L2L_CPU], ids=["resident", "l2l_sync"])
def test_step_timings_and_backend_release(sandbox, tmp_path, monkeypatch, execution):  # noqa: F811
    sandbox("t")
    made = []
    real = B.make_backend
    monkeypatch.setattr(trainer, "make_backend", lambda *a, **k: made.append(real(*a, **k)) or made[-1])
    out = trainer.train(make_cfg(tmp_path, execution, max_steps=3), device="cpu")
    t = out["timings"]
    assert [x["step"] for x in t] == [r["step"] for r in out["training"]] == [1, 2, 3]
    for x in t:
        assert x["step_s"] >= x["update_s"] > 0 and x["rollout_s"] > 0 and x["prepare_s"] > 0
        assert x["rollout_tokens"] > 0 and x["cpu_rss_bytes"] > 0
    if execution is None:
        assert all(x["weight_bytes_copied"] is None and x["backend"] == "resident" for x in t)  # not applicable
    else:
        assert all(x["weight_bytes_copied"] > 0 and x["backend"] == "l2l_sync" for x in t)
        assert made[0].store.closed  # pinned / pageable slabs released when train() returns
