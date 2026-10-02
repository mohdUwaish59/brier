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
    assert sel.oof_accuracy >= 0.95  # ranking by plain NLL may prefer a near-separable candidate
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
    # fitted confidence on the true class matches the one-vs-rest target (30 + 1) / (30 + 2)
    p_true = np.exp((logp / platt) - np.log(np.exp(logp / platt).sum(1, keepdims=True)))[0, y[0]]
    assert p_true == pytest.approx(31 / 32, abs=1e-4)


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


@pytest.mark.parametrize(("k", "n_c"), [(2, 30), (5, 15), (20, 15)])
def test_platt_cap_does_not_depend_on_class_count(k: int, n_c: int) -> None:
    # Separable scores: the fitted true-class confidence must be the one-vs-rest Platt target
    # (n_c+1)/(n_c+2) for any K (the old (n_c+1)/(n_c+K) gave 16/35 at K = 20).
    y = np.repeat(np.arange(k), n_c)
    logp = np.zeros((len(y), k))
    logp[np.arange(len(y)), y] = 1.0  # always right by the same margin
    t = fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0), smoothing="platt")
    z = logp / t
    p = np.exp(z - np.log(np.exp(z).sum(1, keepdims=True)))
    assert p[0, y[0]] == pytest.approx((n_c + 1) / (n_c + 2), abs=1e-4)


def test_platt_two_class_targets_match_platt_1999() -> None:
    # K = 2: positives (N+ + 1) / (N+ + 2), negatives 1 / (N- + 2) for the positive class.
    from brier.calibrate.temperature import platt_targets

    y = np.array([0] * 3 + [1] * 5)
    t = platt_targets(y, 2)
    np.testing.assert_allclose(t[0], [4 / 5, 1 / 5])
    np.testing.assert_allclose(t[3], [1 / 7, 6 / 7])
    np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_platt_targets_spread_rest_evenly() -> None:
    from brier.calibrate.temperature import platt_targets

    y = np.array([0] * 4 + [1, 2, 3])
    t = platt_targets(y, 4)
    np.testing.assert_allclose(t[0], [5 / 6, 1 / 18, 1 / 18, 1 / 18])
    np.testing.assert_allclose(t.sum(axis=1), 1.0)


def test_platt_targets_need_two_classes() -> None:
    from brier.calibrate.temperature import platt_targets

    with pytest.raises(BrierError):
        platt_targets(np.zeros(5, dtype=int), 1)


def test_l2_temperature_is_plain_nll_when_not_separable() -> None:
    from brier.heads.select import fit_l2_temperature

    rng = np.random.default_rng(3)
    y = np.repeat(np.arange(20), 5)  # K = 20, 5 labels per class: the case smoothing broke
    logp = rng.normal(0, 1, size=(100, 20))
    logp[np.arange(100), y] += 2.0  # informative but overlapping
    plain = fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0))
    assert -7.0 < np.log(plain) < 7.0
    assert fit_l2_temperature(logp, y) == plain


def test_l2_temperature_falls_back_to_platt_when_separable_many_class() -> None:
    from brier.heads.select import fit_l2_temperature

    y = np.repeat(np.arange(4), 20)
    logp = np.zeros((80, 4))
    logp[np.arange(80), y] = 1.0  # always right: plain NLL runs to the lower bound
    assert np.log(fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0))) == pytest.approx(-7.0)
    t = fit_l2_temperature(logp, y)
    assert t == fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0), smoothing="platt")
    assert -7.0 < np.log(t) < 7.0


def test_l2_temperature_two_classes_always_platt() -> None:
    from brier.heads.select import fit_l2_temperature

    rng = np.random.default_rng(4)
    y = np.repeat(np.arange(2), 30)
    logp = rng.normal(0, 1, size=(60, 2))
    logp[np.arange(60), y] += 3.0  # one or two errors: plain NLL would not hit the bound
    platt = fit_temperature(logp, y, log_t_bounds=(-7.0, 7.0), smoothing="platt")
    assert fit_l2_temperature(logp, y) == platt


def test_l2_heads_are_calibrated_on_non_separable_many_class_data() -> None:
    # End to end through select_head: K = 20, 5 labels per class, overlapping classes.
    # The held-out ECE must be close to what plain temperature scaling achieves (the earlier
    # smoothing-on-every-fit gave ~0.17 here; plain ~0.02 in the math review's simulation).
    from brier.metrics import ece

    rng = np.random.default_rng(11)
    centres = rng.normal(0, 1.0, size=(20, 64))

    def draw(n_per: int) -> tuple:  # type: ignore[type-arg]
        y = np.repeat(np.arange(20), n_per)
        return (centres[y] + rng.normal(0, 3.0, size=(len(y), 64)))[:, None, :], y

    h, y = draw(5)
    sel = select_head(h, [0], y, 20)
    h_test, y_test = draw(150)
    p = np.exp(sel.log_probs(h_test[:, 0]))
    assert 0.2 < float(np.mean(p.argmax(1) == y_test)) < 0.9  # informative, not separable
    assert ece(p, y_test) < 0.08
