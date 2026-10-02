import math

import numpy as np
import pytest

from brier.errors import BrierError, InsufficientDataError
from brier.heads.lda import LdaHead, fit_lda
from brier.heads.ridge import RidgeHead, fit_ridge


def _data(n: int, d: int, c: int, seed: int = 0, sep: float = 1.0):  # type: ignore[no-untyped-def]
    """Gaussian classes with distinct means; every class appears at least twice."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.arange(c), np.arange(c), rng.integers(0, c, size=n - 2 * c)])
    centres = rng.normal(0, sep, size=(c, d))
    h = centres[y] + rng.normal(0, 1, size=(n, d)) + rng.normal(5, 3, size=d)  # offset, scale
    return h, y


def _standardise(h: np.ndarray, h_new: np.ndarray):  # type: ignore[no-untyped-def]
    mu = h.mean(axis=0)
    sd = np.maximum(h.std(axis=0), 1e-6)
    return (h - mu) / sd, (h_new - mu) / sd


# ---------- naive references (explicit d x d matrices and inverses) ----------


def _ridge_reference(h, y, c, alpha, h_new):  # type: ignore[no-untyped-def]
    x, x_new = _standardise(h, h_new)
    onehot = np.eye(c)[y]
    ybar = onehot.mean(axis=0)
    w = np.linalg.inv(x.T @ x + alpha * np.eye(x.shape[1])) @ x.T @ (onehot - ybar)
    return x_new @ w + ybar


def _lda_reference(h, y, c, h_new):  # type: ignore[no-untyped-def]
    x, x_new = _standardise(h, h_new)
    n, d = x.shape
    m = np.stack([x[y == k].mean(axis=0) for k in range(c)])
    z = x - m[y]
    s = z.T @ z / n
    mu = np.trace(s) / d
    delta2 = np.sum((s - mu * np.eye(d)) ** 2)
    beta_bar2 = sum(np.sum((np.outer(zk, zk) - s) ** 2) for zk in z) / n**2
    gamma = min(beta_bar2, delta2) / delta2
    sg = (1 - gamma) * s + gamma * mu * np.eye(d)
    u = np.linalg.inv(sg) @ m.T
    prior = np.bincount(y, minlength=c) / n
    return x_new @ u - 0.5 * np.sum(m.T * u, axis=0) + np.log(prior), gamma


# ---------- ridge ----------


@pytest.mark.parametrize(("n", "d"), [(30, 80), (120, 10)], ids=["n<d", "n>=d"])
@pytest.mark.parametrize("solver", ["dual", "primal", "auto"])
def test_ridge_matches_reference(n: int, d: int, solver: str) -> None:
    h, y = _data(n, d, 4)
    h_new, _ = _data(15, d, 4, seed=1)
    head = fit_ridge(h, y, 4, alpha=3.0, solver=solver)  # type: ignore[arg-type]
    np.testing.assert_allclose(head.scores(h_new), _ridge_reference(h, y, 4, 3.0, h_new), atol=1e-8)


def test_ridge_auto_picks_dual_when_n_below_d() -> None:
    assert fit_ridge(*_data(30, 80, 3), 3).solver == "dual"
    assert fit_ridge(*_data(120, 10, 3), 3).solver == "primal"


def test_ridge_large_alpha_shrinks_to_class_frequencies() -> None:
    h, y = _data(60, 20, 3)
    s = fit_ridge(h, y, 3, alpha=1e12).scores(h)
    np.testing.assert_allclose(s, np.tile(np.bincount(y, minlength=3) / 60, (60, 1)), atol=1e-6)


def test_ridge_separates_well_separated_classes() -> None:
    h, y = _data(200, 30, 4, sep=4.0)
    h_test, y_test = _data(200, 30, 4, seed=0, sep=4.0)  # same centres (same seed)
    head = fit_ridge(h, y, 4, alpha=1.0)
    assert np.mean(head.scores(h_test).argmax(1) == y_test) > 0.95


# ---------- LDA ----------


@pytest.mark.parametrize(("n", "d"), [(30, 80), (150, 12)], ids=["n<d-woodbury", "n>=d-direct"])
def test_lda_matches_reference(n: int, d: int) -> None:
    h, y = _data(n, d, 4)
    h_new, _ = _data(15, d, 4, seed=1)
    ref_scores, ref_gamma = _lda_reference(h, y, 4, h_new)
    head = fit_lda(h, y, 4)
    assert head.shrinkage == pytest.approx(ref_gamma, rel=1e-9)
    np.testing.assert_allclose(head.scores(h_new), ref_scores, rtol=1e-7, atol=1e-7)


def test_lda_shrinkage_in_unit_interval() -> None:
    for n, d in [(20, 100), (500, 5)]:
        assert 0.0 <= fit_lda(*_data(n, d, 3), 3).shrinkage <= 1.0


def test_lda_separates_well_separated_classes() -> None:
    h, y = _data(200, 30, 4, sep=4.0)
    head = fit_lda(h, y, 4)
    assert np.mean(head.scores(h).argmax(1) == y) > 0.95


def test_lda_requires_every_class() -> None:
    h, y = _data(40, 10, 3)
    y = np.where(y == 2, 0, y)
    with pytest.raises(InsufficientDataError):
        fit_lda(h, y, 3)


# ---------- shared behaviour ----------


@pytest.mark.parametrize("fit", [fit_ridge, fit_lda])
def test_constant_feature_is_finite(fit) -> None:  # type: ignore[no-untyped-def]
    h, y = _data(40, 10, 3)
    h[:, 0] = 7.0  # zero variance: sigma floored at 1e-6
    s = fit(h, y, 3).scores(h)
    assert np.all(np.isfinite(s))


@pytest.mark.parametrize("fit", [fit_ridge, fit_lda])
def test_float32_input_and_float64_output(fit) -> None:  # type: ignore[no-untyped-def]
    h, y = _data(40, 10, 3)
    s = fit(h.astype(np.float32), y, 3).scores(h.astype(np.float32))
    assert s.dtype == np.float64
    assert s.shape == (40, 3)


@pytest.mark.parametrize("fit", [fit_ridge, fit_lda])
def test_heads_are_immutable(fit) -> None:  # type: ignore[no-untyped-def]
    head = fit(*_data(40, 10, 3), 3)
    with pytest.raises(AttributeError):
        head.mean = None  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        head.mean[0] = 1.0


@pytest.mark.parametrize("fit", [fit_ridge, fit_lda])
def test_deterministic(fit) -> None:  # type: ignore[no-untyped-def]
    h, y = _data(40, 10, 3)
    np.testing.assert_array_equal(fit(h, y, 3).scores(h), fit(h, y, 3).scores(h))


@pytest.mark.parametrize("fit", [fit_ridge, fit_lda])
@pytest.mark.parametrize(
    "case",
    ["h-1d", "nan", "labels-float", "labels-range", "labels-length", "one-class", "too-few"],
)
def test_fit_validates_inputs(fit, case: str) -> None:  # type: ignore[no-untyped-def]
    h, y = _data(40, 10, 3)
    c = 3
    if case == "h-1d":
        h = h[:, 0]
    elif case == "nan":
        h[0, 0] = math.nan
    elif case == "labels-float":
        y = y.astype(float)
    elif case == "labels-range":
        y = y.copy()
        y[0] = 3
    elif case == "labels-length":
        y = y[:-1]
    elif case == "one-class":
        c = 1
    elif case == "too-few":
        h, y = h[:1], y[:1]
    with pytest.raises(BrierError):
        fit(h, y, c)


@pytest.mark.parametrize("fit", [fit_ridge, fit_lda])
def test_scores_validate_dimensions(fit) -> None:  # type: ignore[no-untyped-def]
    head = fit(*_data(40, 10, 3), 3)
    with pytest.raises(BrierError):
        head.scores(np.zeros((5, 9)))
    with pytest.raises(BrierError):
        head.scores(np.full((2, 10), np.nan))


@pytest.mark.parametrize("alpha", [0.0, -1.0, math.nan, math.inf, True])
def test_ridge_alpha_validated(alpha: object) -> None:
    with pytest.raises(BrierError):
        fit_ridge(*_data(40, 10, 3), 3, alpha=alpha)  # type: ignore[arg-type]


def test_ridge_solver_validated() -> None:
    with pytest.raises(BrierError):
        fit_ridge(*_data(40, 10, 3), 3, solver="cholesky")  # type: ignore[arg-type]


def test_head_types() -> None:
    assert isinstance(fit_ridge(*_data(40, 10, 3), 3), RidgeHead)
    assert isinstance(fit_lda(*_data(40, 10, 3), 3), LdaHead)


def test_lda_rejects_zero_within_class_variance() -> None:
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(3, 8))
    y = np.repeat(np.arange(3), 4)
    with pytest.raises(BrierError, match="within-class"):
        fit_lda(centres[y], y, 3)  # every sample equals its class mean


def test_lda_rejects_degenerate_shrinkage_when_n_below_d() -> None:
    # Residuals are +v / -v: every z_k z_k^T equals S, so Ledoit-Wolf gamma is 0 and S is rank 1.
    rng = np.random.default_rng(1)
    centres, v = rng.normal(size=(2, 10)), rng.normal(size=10)
    h = np.stack([centres[0] + v, centres[0] - v, centres[1] + v, centres[1] - v])
    with pytest.raises(BrierError, match="degenerate"):
        fit_lda(h, np.array([0, 0, 1, 1]), 2)


def test_lda_rejects_degenerate_shrinkage_when_n_at_least_d() -> None:
    # d <= n < d + C: residuals +v / -v give rank 1 < d and gamma = 0 on the direct path too.
    rng = np.random.default_rng(2)
    centres, v = rng.normal(size=(2, 3)), rng.normal(size=3)
    h = np.stack([centres[0] + v, centres[0] - v, centres[1] + v, centres[1] - v])
    with pytest.raises(BrierError, match="degenerate"):
        fit_lda(h, np.array([0, 0, 1, 1]), 2)


def test_lda_singular_solve_becomes_brier_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def singular(*a, **k):  # type: ignore[no-untyped-def]
        raise np.linalg.LinAlgError("Singular matrix")

    monkeypatch.setattr(np.linalg, "solve", singular)
    with pytest.raises(BrierError, match="singular"):
        fit_lda(*_data(150, 12, 4), 4)  # n >= d: direct path
