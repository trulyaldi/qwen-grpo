import pytest
import torch

from diagnostics.two_example_grpo import MIN_UPDATES, classify, decompose, select_pair, separated
from train_multistep import group_rewards


def test_select_pair_one_per_class_least_confident():
    ex = [{"id": i, "label": l} for i, l in enumerate(["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"])]
    pos, neg = select_pair(ex, [2.0, -0.3, -0.5, 1.5])
    assert (pos["id"], pos["label"], neg["id"], neg["label"]) == (2, "POSITIVE", 1, "NEGATIVE")


def test_b2_rewards_row0_positive_row1_negative():
    r = group_rewards(["POSITIVE"] * 3 + ["NEGATIVE"] * 3 + ["NEGATIVE"] * 6, ["POSITIVE", "NEGATIVE"], 6)
    assert r.tolist() == [[1, 1, 1, 0, 0, 0], [1] * 6]
    assert [bool((x != x[0]).any()) for x in r] == [True, False]  # only the POSITIVE group carries signal


def test_decompose_deltas_margin_common_and_differential():
    d = decompose((0.5, 1.0), (1.5, 0.2))
    assert d["d_pos"] == pytest.approx(1.0) and d["d_neg"] == pytest.approx(-0.8)
    assert d["margin"] == pytest.approx(1.3)
    assert d["C"] == pytest.approx(0.1) and d["D"] == pytest.approx(0.9)
    assert d["margin"] - 2 * d["D"] == pytest.approx(0.5 - 1.0)  # M_t - 2 D_t = M_0


def rows(*deltas, confirmed=True):
    return [{**decompose((0.0, 0.0), (dp, dn)), "confirmed": confirmed} for dp, dn in deltas]


def test_separation_needs_both_directions_not_just_positive_D():
    both_up = decompose((0.0, 0.0), (1.0, 0.2))
    assert both_up["D"] > 0 and not separated(both_up, 0.01)
    assert separated(decompose((0.0, 0.0), (0.1, -0.1)), 0.01) and not separated(decompose((0.0, 0.0), (0.1, -0.1)), 0.2)


@pytest.mark.parametrize("deltas, n, expected", [
    ([(0.1, 0.1), (0.3, -0.2), (0.4, -0.3)], MIN_UPDATES, "PASS"),       # two consecutive separated checkpoints
    ([(0.1, 0.1), (0.3, -0.2), (0.5, 0.1)], MIN_UPDATES, "PARTIAL"),     # separated once, not persistent; margin up
    ([(0.5, 0.3), (1.0, 0.6)], MIN_UPDATES, "PARTIAL"),                  # both up, positive more: shared shift
    ([(0.5, 0.5), (1.0, 1.02)], MIN_UPDATES, "FAIL"),                    # common shift only
    ([(-0.2, 0.3)], MIN_UPDATES, "FAIL"),                                # separation reversed
    ([(0.3, -0.2), (0.4, -0.3)], MIN_UPDATES - 1, "INCONCLUSIVE"),       # too few updates
])
def test_classify(deltas, n, expected):
    assert classify(rows(*deltas), 0.05, n) == expected


def test_classify_unstable_and_unconfirmed_final():
    assert classify(rows((0.3, -0.2), (0.4, -0.3)), 0.05, 100, unstable=True) == "INCONCLUSIVE"
    assert classify(rows((0.1, 0.1), (0.4, -0.3)), 0.05, 100) == "PASS"           # final checkpoint, confirmed
    assert classify(rows((0.1, 0.1), (0.4, -0.3), confirmed=False), 0.05, 100) == "PARTIAL"
