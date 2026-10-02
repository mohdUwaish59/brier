"""L2 model selection (METHODS.md, L2 "Selection").

Stratified k-fold (k = 5, seeded) over ``layers x alpha x solver``: ridge for each alpha on a
log grid, shrinkage LDA once per layer (it has no alpha). Each candidate's out-of-fold (OOF)
scores are temperature-scaled before their NLL is compared, so ridge scores (not
log-probabilities) and LDA scores compete fairly; the temperature is fitted against
Platt-smoothed targets so separable OOF scores do not drive it to its bound, while the
candidates are compared by plain (hard-label) OOF NLL. The best candidate is refitted on
all labels and its temperature is the one fitted on its OOF scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray, norm
from brier.calibrate.temperature import fit_temperature
from brier.errors import BrierError, InsufficientDataError
from brier.heads._features import IntArray, check_training
from brier.heads.lda import LdaHead, fit_lda
from brier.heads.ridge import RidgeHead, fit_ridge

HeadSolver = Literal["ridge", "lda"]
DEFAULT_ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
# Head scores are not log-probabilities: ridge gaps ~0.1-1 need T ~ e^-2..e^-5, LDA gaps ~100
# need T > 1. [-7, 7] covers both without allowing unrecoverable sharpness.
L2_LOG_T_BOUNDS = (-7.0, 7.0)
MIN_PER_CLASS = 5
MIN_TOTAL = 60


@dataclass(frozen=True)
class Candidate:
    """One grid point and its out-of-fold NLL (``inf`` if the head could not be fitted)."""

    layer: int
    solver: HeadSolver
    alpha: float | None
    oof_nll: float


@dataclass(frozen=True)
class L2Selection:
    """The selected head, refitted on all labels, with its OOF temperature and metrics."""

    layer: int
    solver: HeadSolver
    alpha: float | None
    head: RidgeHead | LdaHead
    temperature: float
    temperature_at_bound: bool  # T hit a search bound (lower: tiny score gaps; upper: no signal)
    oof_nll: float  # min over the grid: optimistic, a diagnostic not a held-out estimate
    oof_accuracy: float
    grid: tuple[Candidate, ...]

    def log_probs(self, h: npt.ArrayLike) -> FloatArray:
        """``(n, C)`` log-probabilities ``norm(scores / T)`` for hidden states of ``layer``."""
        return norm(self.head.scores(h) / self.temperature)


def default_layers(num_layers: int) -> list[int]:
    """Every 2nd block from 40 % to 90 % depth (0-based block indices)."""
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
        raise BrierError("num_layers must be a positive int")
    first, last = -(-4 * num_layers // 10), 9 * num_layers // 10
    return list(range(first, last + 1, 2)) or [num_layers // 2]


def stratified_folds(y: IntArray, n_classes: int, k: int, seed: int) -> list[IntArray]:
    """Split item indices into ``k`` folds; each class is shuffled and dealt round-robin."""
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    offset = 0
    for c in range(n_classes):
        idx = rng.permutation(np.flatnonzero(y == c))
        for j, i in enumerate(idx):
            folds[(offset + j) % k].append(int(i))
        offset += len(idx)  # continue the deal so fold sizes stay balanced across classes
    return [np.sort(np.asarray(f, dtype=np.int64)) for f in folds]


def _fit(
    h: FloatArray, y: IntArray, n_classes: int, solver: HeadSolver, alpha: float | None
) -> RidgeHead | LdaHead:
    if solver == "ridge":
        if alpha is None:
            raise BrierError("ridge needs an alpha")
        return fit_ridge(h, y, n_classes, alpha=alpha)
    return fit_lda(h, y, n_classes)


def select_head(
    hidden: npt.ArrayLike,
    layers: Sequence[int],
    labels: npt.ArrayLike,
    n_classes: int,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    solvers: Sequence[HeadSolver] = ("ridge", "lda"),
    k: int = 5,
    seed: int = 0,
) -> L2Selection:
    """Choose layer, solver and alpha by OOF NLL; refit on all labels; fit T on OOF scores.

    Parameters
    ----------
    hidden : array_like
        ``(n, n_layers, d)`` hidden states, e.g. from ``Backend.hidden_states``.
    layers : Sequence[int]
        Block index of each slice along axis 1 (distinct).
    labels : array_like
        ``(n,)`` int labels in ``[0, n_classes)``.
    n_classes : int
        Number of answers ``C``.
    alphas : Sequence[float]
        Ridge penalties to try (finite, > 0).
    solvers : Sequence of {"ridge", "lda"}
        Head families to try.
    k : int
        Number of folds (>= 2).
    seed : int
        Seed for the fold assignment.

    Returns
    -------
    L2Selection

    Raises
    ------
    InsufficientDataError
        With fewer than 60 labels or fewer than 5 labels for some class.
    BrierError
        On malformed inputs or if no candidate can be fitted.
    """
    h_all = np.asarray(hidden, dtype=np.float64)
    if h_all.ndim != 3:
        raise BrierError("hidden must be an (n, n_layers, d) array")
    layer_ids = list(layers)
    if len(layer_ids) != h_all.shape[1] or len(set(layer_ids)) != len(layer_ids):
        raise BrierError("layers must be distinct and match hidden.shape[1]")
    if not solvers or not set(solvers) <= {"ridge", "lda"}:
        raise BrierError("solvers must be a non-empty subset of {'ridge', 'lda'}")
    if "ridge" in solvers and (
        not alphas
        or not all(isinstance(a, (int, float)) and math.isfinite(a) and a > 0 for a in alphas)
    ):
        raise BrierError("alphas must be finite numbers > 0")
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise BrierError("k must be an int >= 2")
    _, y = check_training(h_all[:, 0], labels, n_classes)
    counts = np.bincount(y, minlength=n_classes)
    if len(y) < MIN_TOTAL:
        raise InsufficientDataError(f"L2 needs at least {MIN_TOTAL} labels, got {len(y)}")
    if counts.min() < MIN_PER_CLASS:
        raise InsufficientDataError(f"L2 needs at least {MIN_PER_CLASS} labels per class")
    if k > counts.min():
        raise BrierError(f"k={k} exceeds the smallest class count {counts.min()}")

    folds = stratified_folds(y, n_classes, k, seed)
    grid: list[Candidate] = []
    best: tuple[float, int, HeadSolver, float | None, float, FloatArray] | None = None
    for pos, layer in enumerate(layer_ids):
        h = h_all[:, pos]
        configs: list[tuple[HeadSolver, float | None]] = []
        if "ridge" in solvers:
            configs += [("ridge", float(a)) for a in alphas]
        if "lda" in solvers:
            configs.append(("lda", None))
        for solver, alpha in configs:
            oof = np.empty((len(y), n_classes))
            try:
                for f in folds:
                    train = np.setdiff1d(np.arange(len(y)), f)
                    oof[f] = _fit(h[train], y[train], n_classes, solver, alpha).scores(h[f])
                lp = norm(oof)
                t = fit_temperature(lp, y, log_t_bounds=L2_LOG_T_BOUNDS, smoothing="platt")
                oof_logp = norm(oof / t)
                score = float(-np.mean(oof_logp[np.arange(len(y)), y]))
            except BrierError:  # e.g. degenerate LDA on this layer: skip, keep searching
                score, t, oof_logp = math.inf, math.nan, oof
            grid.append(Candidate(layer, solver, alpha, score))
            if math.isfinite(score) and (best is None or score < best[0]):
                best = (score, layer, solver, alpha, t, oof_logp)
    if best is None:
        raise BrierError("no L2 candidate could be fitted")
    score, layer, solver, alpha, t, oof_logp = best
    head = _fit(h_all[:, layer_ids.index(layer)], y, n_classes, solver, alpha)
    accuracy = float(np.mean(oof_logp.argmax(axis=1) == y))
    log_t = math.log(t)
    at_bound = min(abs(log_t - b) for b in L2_LOG_T_BOUNDS) < 1e-3
    return L2Selection(layer, solver, alpha, head, t, at_bound, score, accuracy, tuple(grid))
