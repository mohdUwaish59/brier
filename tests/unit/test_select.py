import numpy as np
import pytest

from brier.calibrate.temperature import fit_temperature
from brier.errors import BrierError, InsufficientDataError
from brier.heads.lda import LdaHead
from brier.heads.ridge import RidgeHead
from brier.heads.select import (
    DEFAULT_ALPHAS,
    L2Selection,
    default_layers,
    select_head,
    stratified_folds,
)

C = 4


def _hidden(n: int = 80, signal_layer: int = 2, n_layers: int = 4, d: int = 30, seed: int = 0):  # type: ignore[no-untyped-def]
    """(n, n_layers, d) features; only `signal_layer` carries the label."""
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.repeat(np.arange(C), 5), rng.integers(0, C, size=n - 5 * C)])
    rng.shuffle(y)
    h = rng.normal(size=(n, n_layers, d))
    centres = rng.normal(0, 2.5, size=(C, d))
    h[:, signal_layer] += centres[y]
    return h, y


# ---------- folds ----------


def test_stratified_folds_cover_every_item_once_and_every_class_per_fold() -> None:
    _, y = _hidden(n=83)
    folds = stratified_folds(y, C, k=5, seed=0)
    assert len(folds) == 5
    all_idx = np.sort(np.concatenate(folds))
    np.testing.assert_array_equal(all_idx, np.arange(83))
    for f in folds:
        assert set(y[f]) == set(range(C))
    sizes = [len(f) for f in folds]
    assert max(sizes) - min(sizes) <= C  # balanced within one round per class


def test_stratified_folds_seeded() -> None:
    _, y = _hidden()
    a = stratified_folds(y, C, k=5, seed=0)
    b = stratified_folds(y, C, k=5, seed=0)
    c = stratified_folds(y, C, k=5, seed=1)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c, strict=True))


# ---------- layer grid ----------


@pytest.mark.parametrize(
    ("num_layers", "expected"),
    [(28, [12, 14, 16, 18, 20, 22, 24]), (16, [7, 9, 11, 13]), (4, [2]), (1, [0])],
)
def test_default_layers(num_layers: int, expected: list[int]) -> None:
    assert default_layers(num_layers) == expected


# ---------- selection ----------


def test_selects_the_informative_layer_and_refits_on_all_labels() -> None:
    h, y = _hidden(signal_layer=2)
    sel = select_head(h, [5, 6, 7, 8], y, C)  # layer ids 5..8 -> position 2 is block 7
    assert isinstance(sel, L2Selection)
    assert sel.layer == 7
    assert sel.solver in ("ridge", "lda")
    assert isinstance(sel.head, RidgeHead if sel.solver == "ridge" else LdaHead)
    np.testing.assert_allclose(sel.head.mean, h[:, 2].mean(axis=0))  # refit on all 80 items
    assert sel.oof_accuracy > 0.8


def test_grid_records_every_candidate() -> None:
    h, y = _hidden()
    sel = select_head(h, [0, 1, 2, 3], y, C)
    expected = 4 * (len(DEFAULT_ALPHAS) + 1)  # ridge per alpha + one LDA per layer
    assert len(sel.grid) == expected
    best = min(sel.grid, key=lambda g: g.oof_nll)
    assert (best.layer, best.solver, best.alpha) == (sel.layer, sel.solver, sel.alpha)
    assert sel.oof_nll == best.oof_nll
    assert all((g.alpha is None) == (g.solver == "lda") for g in sel.grid)


def test_temperature_fitted_on_out_of_fold_scores() -> None:
    h, y = _hidden()
    sel = select_head(h, [0, 1, 2, 3], y, C, solvers=("ridge",), alphas=(1.0,))
    assert sel.temperature > 0
    logp = sel.log_probs(h[:, 2])
    np.testing.assert_allclose(np.exp(logp).sum(axis=1), 1.0)
    assert not sel.temperature_at_bound


def test_restricting_solvers_and_alphas() -> None:
    h, y = _hidden()
    sel = select_head(h, [0, 1, 2, 3], y, C, solvers=("lda",))
    assert sel.solver == "lda"
    assert sel.alpha is None
    assert len(sel.grid) == 4


def test_deterministic_for_a_seed() -> None:
    h, y = _hidden()
    a = select_head(h, [0, 1, 2, 3], y, C, seed=3)
    b = select_head(h, [0, 1, 2, 3], y, C, seed=3)
    assert (a.layer, a.solver, a.alpha, a.temperature, a.oof_nll) == (
        b.layer,
        b.solver,
        b.alpha,
        b.temperature,
        b.oof_nll,
    )


def test_degenerate_candidates_are_skipped_not_fatal() -> None:
    h, y = _hidden(n_layers=2, signal_layer=1)
    h[:, 0] = 0.0  # LDA fails on layer 0 (no within-class variance); ridge still fits
    sel = select_head(h, [0, 1], y, C)
    lda0 = next(g for g in sel.grid if g.layer == 0 and g.solver == "lda")
    assert lda0.oof_nll == np.inf
    assert sel.layer == 1


# ---------- minimum data and validation ----------


def test_minimum_labels() -> None:
    h, y = _hidden(n=59)
    with pytest.raises(InsufficientDataError, match="60"):
        select_head(h, [0, 1, 2, 3], y, C)
    h, y = _hidden(n=80)
    y = y.copy()
    y[y == 3] = 0
    y[:4] = 3  # only 4 labels of class 3
    with pytest.raises(InsufficientDataError, match="5"):
        select_head(h, [0, 1, 2, 3], y, C)


@pytest.mark.parametrize(
    "case",
    [
        "hidden-2d",
        "layers-mismatch",
        "dup-layers",
        "bad-solver",
        "no-solvers",
        "bad-alpha",
        "bad-k",
    ],
)
def test_validation(case: str) -> None:
    h, y = _hidden()
    kw: dict = {}  # type: ignore[type-arg]
    layers = [0, 1, 2, 3]
    if case == "hidden-2d":
        h = h[:, 0]
    elif case == "layers-mismatch":
        layers = [0, 1, 2]
    elif case == "dup-layers":
        layers = [0, 1, 1, 3]
    elif case == "bad-solver":
        kw["solvers"] = ("svm",)
    elif case == "no-solvers":
        kw["solvers"] = ()
    elif case == "bad-alpha":
        kw["alphas"] = (1.0, -1.0)
    elif case == "bad-k":
        kw["k"] = 1
    with pytest.raises(BrierError):
        select_head(h, layers, y, C, **kw)


def test_fit_temperature_custom_bounds() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    logp = np.log(np.full((200, 3), 1 / 3))
    logp = logp + 0.001 * np.eye(3)[y]  # nearly uninformative but slightly right
    t_default = fit_temperature(logp, y)
    t_wide = fit_temperature(logp, y, log_t_bounds=(-10.0, 10.0))
    assert t_default == pytest.approx(np.exp(-3))  # clipped at L1's bound
    assert t_wide < np.exp(-3)
    with pytest.raises(BrierError):
        fit_temperature(logp, y, log_t_bounds=(1.0, -1.0))


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_default_layers_validated(bad: object) -> None:
    with pytest.raises(BrierError):
        default_layers(bad)  # type: ignore[arg-type]


def test_ridge_candidate_requires_alpha() -> None:
    from brier.heads.select import _fit

    h, y = _hidden()
    with pytest.raises(BrierError):
        _fit(h[:, 0], y, C, "ridge", None)


def test_all_candidates_failing_raises() -> None:
    h, y = _hidden()
    h[:] = 0.0  # LDA is undefined on every layer
    with pytest.raises(BrierError, match="no L2 candidate"):
        select_head(h, [0, 1, 2, 3], y, C, solvers=("lda",))


def test_separable_data_keeps_a_finite_temperature() -> None:
    # Plain NLL would drive T to its lower bound (near-0/1 probabilities); Platt-smoothed
    # targets keep T interior and confidence bounded.
    from brier.heads.select import L2_LOG_T_BOUNDS

    rng = np.random.default_rng(5)
    y = np.repeat(np.arange(2), 40)
    h = rng.normal(size=(80, 1, 6))
    h[:, 0, 0] += 10.0 * y  # perfectly separable on one feature
    sel = select_head(h, [0], y, 2)
    assert sel.oof_accuracy == 1.0
    assert not sel.temperature_at_bound
    assert L2_LOG_T_BOUNDS[0] < np.log(sel.temperature) < L2_LOG_T_BOUNDS[1]
    assert np.exp(sel.log_probs(h[:, 0])).max() < 0.999


def test_normal_data_does_not_flag_the_bound() -> None:
    h, y = _hidden(seed=4)
    h[:, 2] += np.random.default_rng(9).normal(0, 3, size=h[:, 2].shape)  # not separable
    assert not select_head(h, [0, 1, 2, 3], y, C).temperature_at_bound


def test_k_cannot_exceed_smallest_class() -> None:
    h, _ = _hidden()
    y = np.concatenate([np.repeat(np.arange(3), 25), np.full(5, 3)])  # class 3 has exactly 5
    with pytest.raises(BrierError, match="smallest class"):
        select_head(h, [0, 1, 2, 3], y, C, k=6)


@pytest.mark.parametrize("bounds", [("a", 1.0), (None, 2.0), (True, 2.0)])
def test_log_t_bounds_must_be_numbers(bounds: tuple) -> None:  # type: ignore[type-arg]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=60)
    with pytest.raises(BrierError):
        fit_temperature(np.log(np.full((60, 3), 1 / 3)), y, log_t_bounds=bounds)


def test_platt_smoothing_keeps_t_finite_on_separable_scores() -> None:
    y = np.repeat(np.arange(3), 30)
    logp = np.log(np.full((90, 3), 0.2))
    logp[np.arange(90), y] = np.log(0.6)  # always right: separable
    plain = fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0))
    platt = fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0), smoothing="platt")
    assert np.log(plain) == pytest.approx(-7.0)  # runs to the bound
    assert -7.0 < np.log(platt) < 7.0
    # fitted confidence on the true class matches the smoothed target (30 + 1) / (30 + 3)
    p_true = np.exp((logp / platt) - np.log(np.exp(logp / platt).sum(1, keepdims=True)))[0, y[0]]
    assert p_true == pytest.approx(31 / 33, abs=1e-4)


def test_platt_smoothing_barely_matters_on_noisy_data() -> None:
    rng = np.random.default_rng(1)
    c = rng.normal(0, 1.5, size=(5000, 4))
    p = np.exp(c - np.log(np.exp(c).sum(1, keepdims=True)))
    y = np.array([rng.choice(4, p=row) for row in p])
    logp = 2.0 * c
    plain = fit_temperature(logp, y)
    platt = fit_temperature(logp, y, smoothing="platt")
    assert platt == pytest.approx(plain, rel=0.02)


def test_smoothing_validated() -> None:
    y = np.repeat(np.arange(2), 30)
    with pytest.raises(BrierError):
        fit_temperature(np.zeros((60, 2)), y, smoothing="laplace")  # type: ignore[arg-type]
