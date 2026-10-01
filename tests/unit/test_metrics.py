import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from brier.errors import BrierError
from brier.metrics import (
    CI,
    accuracy,
    aurc,
    bootstrap_ci,
    brier,
    coverage_at_risk,
    ece,
    flip_rate,
    mae_expected,
    nll,
    paired_bootstrap_ci,
    rps,
)

P = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.5, 0.25, 0.25], [0.2, 0.2, 0.6]])
Y = np.array([0, 2, 0, 2])  # argmax: 0, 1, 0, 2 -> correct, wrong, correct, correct


def test_accuracy_hand() -> None:
    assert accuracy(P, Y) == pytest.approx(0.75)


def test_nll_hand() -> None:
    # -(log .7 + log .3 + log .5 + log .6) / 4
    expected = -(math.log(0.7) + math.log(0.3) + math.log(0.5) + math.log(0.6)) / 4
    assert nll(P, Y) == pytest.approx(expected)


def test_nll_clips_zero_probability() -> None:
    assert nll(np.array([[1.0, 0.0]]), np.array([1])) == pytest.approx(-math.log(1e-12))


def test_brier_hand() -> None:
    # row0: .09+.04+.01=.14  row1: .01+.36+.49=.86  row2: .25+.0625+.0625=.375
    # row3: .04+.04+.16=.24  mean = 1.615/4
    assert brier(P, Y) == pytest.approx(1.615 / 4)


def test_ece_equal_mass_hand() -> None:
    # confidences .7 .6 .5 .6 ; correct 1 0 1 1. Two equal-mass bins on sorted conf:
    # sorted (stable, ascending): .5(c=1) .6(c=0) | .6(c=1) .7(c=1)
    # bin1: acc .5 conf .55 -> .05 ; bin2: acc 1 conf .65 -> .35 ; ECE = .5*.05 + .5*.35 = .2
    assert ece(P, Y, n_bins=2) == pytest.approx(0.2)


def test_ece_equal_width_hand() -> None:
    # width bins on [0,1] with 2 bins: (0,.5] -> {.5 c=1}; (.5,1] -> {.7 c1, .6 c0, .6 c1}
    # bin1: |1-.5| *1/4 = .125 ; bin2: acc 2/3 conf 1.9/3 -> |.0333| * 3/4 = .025
    assert ece(P, Y, n_bins=2, scheme="width") == pytest.approx(0.125 + 0.025)


def test_ece_default_15_bins_and_more_bins_than_items() -> None:
    assert 0.0 <= ece(P, Y) <= 1.0  # 15 bins, 4 items: empty bins are skipped


def test_ece_perfect_confident_correct_is_zero() -> None:
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(p, np.array([0, 1])) == pytest.approx(0.0)


def test_flip_rate_hand() -> None:
    reversed_mapped = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.5, 0.3], [0.2, 0.2, 0.6]])
    assert flip_rate(P, reversed_mapped) == pytest.approx(0.5)


def test_aurc_hand() -> None:
    # sorted by confidence desc: .7(c) .6(w) .6(c) .5(c) -- stable for ties (index 1 before 3)
    # risks at k=1..4: 0, 1/2, 1/3, 1/4 -> mean = (0 + .5 + .3333 + .25)/4
    assert aurc(P, Y) == pytest.approx((0 + 0.5 + 1 / 3 + 0.25) / 4)


def test_coverage_at_risk_hand() -> None:
    assert coverage_at_risk(P, Y, alpha=0.0) == pytest.approx(0.25)
    assert coverage_at_risk(P, Y, alpha=0.3) == pytest.approx(1.0)  # risk(4) = .25
    assert coverage_at_risk(np.array([[0.4, 0.6]]), np.array([0]), alpha=0.1) == 0.0


def test_score_metrics_hand() -> None:
    p = np.array([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
    y = np.array([2, 0])  # level indices; expected index: 1.6, 0.5 -> errors .4, .5
    assert mae_expected(p, y) == pytest.approx(0.45)
    # RPS: cdf_p row0 (.1,.3) vs cdf_y (0,0): .01+.09=.10 ; row1 (.5,1) vs (1,1): .25
    # normalised by K-1=2: (.05 + .125)/2
    assert rps(p, y) == pytest.approx((0.05 + 0.125) / 2)


@pytest.mark.parametrize(
    ("p", "y"),
    [
        (np.array([[0.5, 0.6]]), np.array([0])),  # rows do not sum to 1
        (np.array([[np.nan, 1.0]]), np.array([0])),
        (np.array([[1.5, -0.5]]), np.array([0])),
        (np.array([[0.5, 0.5]]), np.array([2])),  # label out of range
        (np.array([[0.5, 0.5]]), np.array([0, 1])),  # length mismatch
        (np.empty((0, 2)), np.empty(0, dtype=int)),  # no items
        (np.array([0.5, 0.5]), np.array([0])),  # not 2-D
        (np.array([[0.5, 0.5]]), np.array([0.0])),  # float labels
    ],
)
def test_inputs_validated(p: np.ndarray, y: np.ndarray) -> None:
    with pytest.raises(BrierError):
        accuracy(p, y)


def test_ece_rejects_bad_options() -> None:
    with pytest.raises(BrierError):
        ece(P, Y, n_bins=0)
    with pytest.raises(BrierError):
        ece(P, Y, scheme="quantile")  # type: ignore[arg-type]
    with pytest.raises(BrierError):
        coverage_at_risk(P, Y, alpha=1.5)


def test_flip_rate_shape_mismatch() -> None:
    with pytest.raises(BrierError):
        flip_rate(P, P[:2])


# ---------- bootstrap ----------


def test_bootstrap_ci_deterministic_and_brackets_value() -> None:
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=200)
    y = rng.integers(0, 3, size=200)
    a = bootstrap_ci(accuracy, p, y, seed=0)
    assert isinstance(a, CI)
    assert a == bootstrap_ci(accuracy, p, y, seed=0)
    assert a != bootstrap_ci(accuracy, p, y, seed=1)
    assert a.lo <= a.value <= a.hi
    assert a.value == pytest.approx(accuracy(p, y))
    assert list(a) == [a.value, a.lo, a.hi]


def test_bootstrap_ci_width_shrinks_with_more_items() -> None:
    rng = np.random.default_rng(2)
    small = rng.dirichlet(np.ones(2), size=50), rng.integers(0, 2, size=50)
    big = rng.dirichlet(np.ones(2), size=2000), rng.integers(0, 2, size=2000)
    w_small = (lambda c: c.hi - c.lo)(bootstrap_ci(brier, *small, seed=0))
    w_big = (lambda c: c.hi - c.lo)(bootstrap_ci(brier, *big, seed=0))
    assert w_big < w_small


def test_paired_bootstrap_identical_is_zero_and_sign() -> None:
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.ones(3), size=100)
    y = rng.integers(0, 3, size=100)
    assert paired_bootstrap_ci(nll, (p, y), (p, y), seed=0) == CI(0.0, 0.0, 0.0)
    sharp = np.eye(3)[y] * 0.9 + 0.1 / 3  # near-perfect predictions
    d = paired_bootstrap_ci(nll, (sharp, y), (p, y), seed=0)
    assert d.hi < 0  # sharp has lower NLL on every resample


def test_bootstrap_validation() -> None:
    with pytest.raises(BrierError):
        bootstrap_ci(accuracy, P, Y, n_resamples=0)
    with pytest.raises(BrierError):
        bootstrap_ci(accuracy, P, Y, level=1.0)
    with pytest.raises(BrierError):
        bootstrap_ci(accuracy, P, Y[:2])
    with pytest.raises(BrierError):
        paired_bootstrap_ci(accuracy, (P, Y), (P[:2], Y[:2]))


# ---------- invariants ----------

dists = st.integers(1, 30).flatmap(
    lambda n: st.tuples(
        arrays(np.float64, (n, 3), elements=st.floats(0.01, 1.0)).map(
            lambda a: a / a.sum(axis=1, keepdims=True)
        ),
        arrays(np.int64, n, elements=st.integers(0, 2)),
    )
)


@given(dists)
def test_metrics_within_bounds(py: tuple[np.ndarray, np.ndarray]) -> None:
    p, y = py
    assert 0.0 <= accuracy(p, y) <= 1.0
    assert nll(p, y) >= 0.0
    assert 0.0 <= brier(p, y) <= 2.0
    assert 0.0 <= ece(p, y) <= 1.0
    assert 0.0 <= ece(p, y, scheme="width") <= 1.0
    assert 0.0 <= aurc(p, y) <= 1.0
    assert 0.0 <= coverage_at_risk(p, y, alpha=0.1) <= 1.0
    assert 0.0 <= rps(p, y) <= 1.0
    assert 0.0 <= mae_expected(p, y) <= 2.0
    assert flip_rate(p, p) == 0.0


def test_equal_width_bin_edges_are_exact() -> None:
    # 4/9 + 1 ulp belongs to bin (4/9, 5/9], not the bin ending at 4/9.
    c = np.nextafter(4 / 9, 1.0)
    p = np.array([[c, 1 - c], [0.95, 0.05]])
    y = np.array([0, 0])
    # with the right bin: bin (4/9,5/9] holds c (acc 1) -> |1 - c| / 2 ; .95 bin -> .05 / 2
    assert ece(p, y, n_bins=9, scheme="width") == pytest.approx((1 - c) / 2 + 0.05 / 2)


def test_numpy_int_options_accepted() -> None:
    assert ece(P, Y, n_bins=np.int64(2)) == pytest.approx(0.2)
    bootstrap_ci(accuracy, P, Y, n_resamples=np.int64(10), seed=np.int64(0))


def test_seed_must_be_int() -> None:
    with pytest.raises(BrierError):
        bootstrap_ci(accuracy, P, Y, seed=None)  # type: ignore[arg-type]


def test_flip_rate_validates_inputs() -> None:
    with pytest.raises(BrierError):
        flip_rate(np.array([[np.nan, 1.0]]), np.array([[0.0, 1.0]]))
