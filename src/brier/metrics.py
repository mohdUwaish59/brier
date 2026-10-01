"""Evaluation metrics and bootstrap confidence intervals (docs/EVALUATION.md).

Every metric takes ``probs`` of shape ``(N, K)`` (rows are distributions, in option or
level order) and integer ``labels`` of shape ``(N,)`` with values in ``[0, K)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray
from brier.errors import BrierError

NLL_FLOOR = 1e-12
_SUM_TOL = 1e-6

Metric = Callable[..., float]
IntArray = npt.NDArray[np.integer[Any]]


class CI(NamedTuple):
    """Point estimate with a percentile bootstrap interval; ``list(ci)`` is ``[v, lo, hi]``."""

    value: float
    lo: float
    hi: float


def _is_int(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _check_probs(probs: npt.ArrayLike) -> FloatArray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise BrierError("probs must be a non-empty (N, K) array with K >= 2")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise BrierError("probs must be finite and non-negative")
    if np.any(np.abs(p.sum(axis=1) - 1.0) > _SUM_TOL):
        raise BrierError("each row of probs must sum to 1")
    return p


def _check(probs: npt.ArrayLike, labels: npt.ArrayLike) -> tuple[FloatArray, IntArray]:
    p = _check_probs(probs)
    y = np.asarray(labels)
    if y.shape != (p.shape[0],) or not np.issubdtype(y.dtype, np.integer):
        raise BrierError("labels must be an int array of shape (N,)")
    if np.any((y < 0) | (y >= p.shape[1])):
        raise BrierError("labels must be in [0, K)")
    return p, y


def _correct_and_conf(p: FloatArray, y: IntArray) -> tuple[FloatArray, FloatArray]:
    return (np.argmax(p, axis=1) == y).astype(np.float64), p.max(axis=1)


def accuracy(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Mean of ``argmax p == y``."""
    p, y = _check(probs, labels)
    return float(np.mean(np.argmax(p, axis=1) == y))


def nll(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Mean ``-log p_y`` with ``p_y`` clipped at 1e-12."""
    p, y = _check(probs, labels)
    return float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], NLL_FLOOR))))


def brier(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Mean ``sum_k (p_k - 1[k = y])^2``."""
    p, y = _check(probs, labels)
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(
    probs: npt.ArrayLike,
    labels: npt.ArrayLike,
    n_bins: int = 15,
    scheme: Literal["mass", "width"] = "mass",
) -> float:
    """Top-label expected calibration error ``sum_m |B_m|/N * |acc(B_m) - conf(B_m)|``.

    Parameters
    ----------
    probs, labels : array_like
        See module docstring.
    n_bins : int
        Number of bins (default 15). Empty bins are skipped.
    scheme : {"mass", "width"}
        ``"mass"``: equal-count bins over confidence sorted ascending (stable for ties);
        ``"width"``: bins ``(m/n, (m+1)/n]`` on ``[0, 1]``.
    """
    p, y = _check(probs, labels)
    if not _is_int(n_bins) or n_bins < 1:
        raise BrierError("n_bins must be a positive int")
    correct, conf = _correct_and_conf(p, y)
    if scheme == "mass":
        bins = np.array_split(np.argsort(conf, kind="stable"), n_bins)
    elif scheme == "width":
        inner_edges = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        index = np.searchsorted(inner_edges, conf, side="left")  # bin m is (m/n, (m+1)/n]
        bins = [np.flatnonzero(index == m) for m in range(n_bins)]
    else:
        raise BrierError("scheme must be 'mass' or 'width'")
    n = len(y)
    return float(sum(len(b) / n * abs(correct[b].mean() - conf[b].mean()) for b in bins if len(b)))


def flip_rate(probs: npt.ArrayLike, probs_reversed: npt.ArrayLike) -> float:
    """Share of items whose argmax differs between two runs (both in original option order).

    ``probs_reversed`` comes from the reversed option list, mapped back to option order.
    """
    a, b = _check_probs(probs), _check_probs(probs_reversed)
    if a.shape != b.shape:
        raise BrierError("both inputs must have the same shape")
    return float(np.mean(np.argmax(a, axis=1) != np.argmax(b, axis=1)))


def _selective_risk(p: FloatArray, y: IntArray) -> FloatArray:
    """Risk of the ``k`` most confident items, for ``k = 1..N`` (stable for ties)."""
    correct, conf = _correct_and_conf(p, y)
    order = np.argsort(-conf, kind="stable")
    errors = np.cumsum(1.0 - correct[order])
    risk: FloatArray = errors / np.arange(1, len(y) + 1)
    return risk


def aurc(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Area under the risk-coverage curve: mean selective risk over coverages ``k/N``."""
    p, y = _check(probs, labels)
    return float(np.mean(_selective_risk(p, y)))


def coverage_at_risk(probs: npt.ArrayLike, labels: npt.ArrayLike, alpha: float) -> float:
    """Largest coverage ``k/N`` whose selective risk is at most ``alpha`` (0 if none).

    The threshold is chosen on the same items, so report the result as "in-sample".
    With tied confidences, ``k`` may split a tie group (order is stable, as for AURC).
    """
    p, y = _check(probs, labels)
    if not 0.0 <= float(alpha) <= 1.0:
        raise BrierError("alpha must be in [0, 1]")
    ok = np.flatnonzero(_selective_risk(p, y) <= alpha)
    return float((ok[-1] + 1) / len(y)) if len(ok) else 0.0


def mae_expected(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Score questions: mean ``|E[level] - y|`` with levels as indices ``0..K-1``."""
    p, y = _check(probs, labels)
    expected = p @ np.arange(p.shape[1], dtype=np.float64)
    return float(np.mean(np.abs(expected - y)))


def rps(probs: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Score questions: ranked probability score, normalised by ``K - 1`` to ``[0, 1]``.

    ``RPS = 1/(K-1) * sum_{k<K} (CDF_p(k) - 1[y <= k])^2``, averaged over items.
    """
    p, y = _check(probs, labels)
    k = p.shape[1]
    cdf = np.cumsum(p, axis=1)[:, :-1]
    cdf_y = (np.arange(k - 1)[None, :] >= y[:, None]).astype(np.float64)
    return float(np.mean(np.sum((cdf - cdf_y) ** 2, axis=1) / (k - 1)))


def _resample_indices(n: int, n_resamples: int, seed: int, level: float) -> IntArray:
    if not _is_int(n_resamples) or n_resamples < 1:
        raise BrierError("n_resamples must be a positive int")
    if not _is_int(seed):
        raise BrierError("seed must be an int (bootstrap CIs are always seeded)")
    if not 0.0 < level < 1.0:
        raise BrierError("level must be in (0, 1)")
    idx: IntArray = np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))
    return idx


def _interval(value: float, samples: list[float], level: float) -> CI:
    tail = (1.0 - level) / 2 * 100
    lo, hi = np.percentile(samples, [tail, 100 - tail])
    return CI(float(value), float(lo), float(hi))


def bootstrap_ci(
    metric: Metric,
    *arrays: npt.ArrayLike,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> CI:
    """Percentile bootstrap CI of ``metric(*arrays)``, resampling items with replacement.

    Parameters
    ----------
    metric : callable
        E.g. :func:`accuracy`; called as ``metric(*resampled_arrays)``.
    *arrays : array_like
        Per-item arrays sharing the first dimension (e.g. ``probs, labels``).
    n_resamples : int
        Number of bootstrap resamples (default 1,000).
    seed : int
        Seed for :func:`numpy.random.default_rng`.
    level : float
        Interval coverage (default 0.95).
    """
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    if n == 0 or any(len(a) != n for a in arrs):
        raise BrierError("arrays must be non-empty and share their first dimension")
    samples = [
        metric(*(a[i] for a in arrs)) for i in _resample_indices(n, n_resamples, seed, level)
    ]
    return _interval(metric(*arrs), samples, level)


def paired_bootstrap_ci(
    metric: Metric,
    a: tuple[npt.ArrayLike, ...],
    b: tuple[npt.ArrayLike, ...],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> CI:
    """Paired bootstrap CI of ``metric(*a) - metric(*b)`` on the same resampled items.

    Use it to compare two levels on the same test items.
    """
    xa = [np.asarray(x) for x in a]
    xb = [np.asarray(x) for x in b]
    n = len(xa[0])
    if n == 0 or any(len(x) != n for x in (*xa, *xb)):
        raise BrierError("a and b must be non-empty and cover the same items")
    samples = [
        metric(*(x[i] for x in xa)) - metric(*(x[i] for x in xb))
        for i in _resample_indices(n, n_resamples, seed, level)
    ]
    return _interval(metric(*xa) - metric(*xb), samples, level)
