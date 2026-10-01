import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from brier._math import norm
from brier.calibrate.temperature import (
    LOG_T_BOUNDS,
    MIN_ITEMS,
    apply_temperature,
    fit_temperature,
)
from brier.errors import BrierError, InsufficientDataError


def _synthetic(t_true: float, n: int = 20000, k: int = 5, seed: int = 0):  # type: ignore[no-untyped-def]
    """Calibrated logits c with labels y ~ softmax(c); observed log-probs are norm(c * T)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(0.0, 1.5, size=(n, k))
    p = np.exp(norm(c))
    y = np.array([rng.choice(k, p=row) for row in p])
    return norm(c * t_true), y


def _nll(logp: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(logp[np.arange(len(y)), y]))


# ---------- apply ----------


def test_apply_hand_computed() -> None:
    # T = 2 takes square roots: sqrt(.8) = 2 sqrt(.2) -> (2/3, 1/3)
    out = np.exp(apply_temperature(np.log(np.array([[0.8, 0.2]])), 2.0))
    np.testing.assert_allclose(out, [[2 / 3, 1 / 3]])


def test_apply_identity_at_one() -> None:
    lp = norm(np.array([[0.3, -1.0, 2.0]]))
    np.testing.assert_allclose(apply_temperature(lp, 1.0), lp)


@pytest.mark.parametrize("t", [0.0, -1.0, math.nan, math.inf, True])
def test_apply_rejects_bad_temperature(t: object) -> None:
    with pytest.raises(BrierError):
        apply_temperature(np.log(np.array([[0.5, 0.5]])), t)  # type: ignore[arg-type]


@given(
    arrays(np.float64, (4, 6), elements=st.floats(-30, 30)),
    st.floats(math.exp(-3), math.exp(3)),
)
def test_apply_preserves_argmax_and_normalises(z: np.ndarray, t: float) -> None:
    z = z.copy()
    z[np.arange(len(z)), np.argmax(z, axis=1)] += 1.0  # unique top-1 (rounding merges near-ties)
    lp = norm(z)
    out = apply_temperature(lp, t)
    np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0, atol=1e-9)
    np.testing.assert_array_equal(np.argmax(out, axis=1), np.argmax(lp, axis=1))


# ---------- fit ----------


@pytest.mark.parametrize("t_true", [0.5, 1.0, 2.5])
def test_fit_recovers_known_temperature(t_true: float) -> None:
    logp, y = _synthetic(t_true)
    assert fit_temperature(logp, y) == pytest.approx(t_true, rel=0.05)


def test_fit_never_worse_than_identity() -> None:
    logp, y = _synthetic(1.7, n=2000, seed=3)
    t = fit_temperature(logp, y)
    assert _nll(apply_temperature(logp, t), y) <= _nll(logp, y) + 1e-12


def test_fit_returns_bound_when_optimum_is_outside() -> None:
    # Always-correct, already confident predictions: NLL keeps falling as T -> 0.
    logp = np.log(np.tile([[0.9, 0.05, 0.05]], (60, 1)))
    y = np.zeros(60, dtype=np.int64)
    assert fit_temperature(logp, y) == pytest.approx(math.exp(LOG_T_BOUNDS[0]))
    # Always-wrong, confident predictions: NLL keeps falling as T -> infinity.
    y_wrong = np.ones(60, dtype=np.int64)
    assert fit_temperature(logp, y_wrong) == pytest.approx(math.exp(LOG_T_BOUNDS[1]))


def test_fit_is_deterministic() -> None:
    logp, y = _synthetic(2.0, n=500, seed=5)
    assert fit_temperature(logp, y) == fit_temperature(logp, y)


def test_fit_minimum_items() -> None:
    assert MIN_ITEMS == 50
    logp, y = _synthetic(2.0, n=50, seed=1)
    fit_temperature(logp, y)
    with pytest.raises(InsufficientDataError):
        fit_temperature(logp[:49], y[:49])


@pytest.mark.parametrize(
    "case",
    ["labels_out_of_range", "float_labels", "length_mismatch", "non_finite", "one_class", "1d"],
)
def test_fit_validates_inputs(case: str) -> None:
    logp, y = _synthetic(2.0, n=60, k=3, seed=2)
    if case == "labels_out_of_range":
        y = y.copy()
        y[0] = 3
    elif case == "float_labels":
        y = y.astype(float)
    elif case == "length_mismatch":
        y = y[:-1]
    elif case == "non_finite":
        logp = logp.copy()
        logp[0, 0] = -np.inf
    elif case == "one_class":
        logp = np.zeros((60, 1))
    elif case == "1d":
        logp = logp[:, 0]
    with pytest.raises(BrierError):
        fit_temperature(logp, y)


@pytest.mark.parametrize("tol", [0.0, -1e-6, math.nan, math.inf])
def test_fit_rejects_bad_tolerance(tol: float) -> None:
    logp, y = _synthetic(2.0, n=60, seed=4)
    with pytest.raises(BrierError):
        fit_temperature(logp, y, tol=tol)
