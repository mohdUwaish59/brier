import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from declib._math import log_softmax, logsumexp, norm

finite = st.floats(min_value=-1e4, max_value=1e4)
vectors = arrays(np.float64, st.integers(1, 30), elements=finite)
matrices = arrays(np.float64, st.tuples(st.integers(1, 5), st.integers(1, 30)), elements=finite)


def test_logsumexp_hand_computed() -> None:
    # log(e^0 + e^0) = log 2
    assert logsumexp(np.array([0.0, 0.0])) == pytest.approx(math.log(2))
    # log(e^1 + e^2 + e^3) = 3 + log(1 + e^-1 + e^-2)
    expected = 3 + math.log(1 + math.exp(-1) + math.exp(-2))
    assert logsumexp(np.array([1.0, 2.0, 3.0])) == pytest.approx(expected)


@pytest.mark.parametrize("big", [1e4, -1e4])
def test_logsumexp_stable_at_extremes(big: float) -> None:
    assert logsumexp(np.array([big, big])) == pytest.approx(big + math.log(2))


def test_logsumexp_handles_neg_inf_entries() -> None:
    assert logsumexp(np.array([-np.inf, 0.0])) == pytest.approx(0.0)
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf


def test_logsumexp_reduces_last_axis_by_default() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(logsumexp(x), [math.log(2), 1 + math.log(2)])
    np.testing.assert_allclose(logsumexp(x, axis=0), [math.log(math.e + 1)] * 2)


def test_norm_returns_float64_and_accepts_lists() -> None:
    out = norm([0.0, 0.0])
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, [math.log(0.5)] * 2)


def test_norm_keeps_neg_inf_as_zero_probability() -> None:
    np.testing.assert_allclose(np.exp(norm(np.array([-np.inf, 0.0]))), [0.0, 1.0])


def test_log_softmax_is_norm() -> None:
    assert log_softmax is norm


@given(vectors)
def test_norm_sums_to_one(v: np.ndarray) -> None:
    out = norm(v)
    assert np.all(np.isfinite(out))
    assert math.fsum(np.exp(out)) == pytest.approx(1.0, abs=1e-9)


@given(matrices)
def test_norm_rows_sum_to_one(x: np.ndarray) -> None:
    np.testing.assert_allclose(np.exp(norm(x)).sum(axis=-1), 1.0, atol=1e-9)


@given(vectors, finite)
def test_norm_shift_invariant(v: np.ndarray, c: float) -> None:
    np.testing.assert_allclose(norm(v + c), norm(v), atol=1e-6)


@given(vectors)
def test_norm_preserves_argmax(v: np.ndarray) -> None:
    out = norm(v)
    assert out[np.argmax(v)] == out.max()


def test_norm_along_axis_zero() -> None:
    x = np.array([[0.0, 1.0], [0.0, 3.0]])
    out = norm(x, axis=0)
    assert out.shape == x.shape
    np.testing.assert_allclose(np.exp(out).sum(axis=0), [1.0, 1.0])


def test_norm_of_massless_row_is_nan_without_warning() -> None:
    # pytest runs with filterwarnings=error, so a RuntimeWarning here would fail.
    assert np.all(np.isnan(norm(np.array([-np.inf, -np.inf]))))
