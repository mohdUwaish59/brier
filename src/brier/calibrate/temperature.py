"""Level ``L1``: temperature scaling (Guo et al. 2017), METHODS.md.

``log q^T = norm(log q / T)``; ``T`` is fitted per question on labelled items by minimising
mean NLL over ``t = log T`` in ``[-3, 3]`` with a bounded golden-section search (no scipy).
``T > 0`` never changes the argmax.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray, norm
from brier.errors import BrierError, InsufficientDataError

LOG_T_BOUNDS = (-3.0, 3.0)
MIN_ITEMS = 50
_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def apply_temperature(logp: npt.ArrayLike, temperature: float) -> FloatArray:
    """Rescale log-probabilities: ``norm(log q / T)`` along the last axis.

    Parameters
    ----------
    logp : array_like
        Log-probabilities, e.g. ``(N, K)`` from L0.
    temperature : float
        ``T > 0``; ``T > 1`` softens (less confident), ``T < 1`` sharpens.

    Returns
    -------
    numpy.ndarray
        float64 log-probabilities with the same argmax as ``logp``.
    """
    if isinstance(temperature, bool) or not (
        math.isfinite(float(temperature)) and float(temperature) > 0
    ):
        raise BrierError("temperature must be a finite number > 0")
    return norm(np.asarray(logp, dtype=np.float64) / float(temperature))


def _check(logp: npt.ArrayLike, labels: npt.ArrayLike) -> tuple[FloatArray, npt.NDArray[Any]]:
    lp = np.asarray(logp, dtype=np.float64)
    y = np.asarray(labels)
    if lp.ndim != 2 or lp.shape[1] < 2:
        raise BrierError("logp must be an (N, K) array with K >= 2")
    if not np.all(np.isfinite(lp)):
        raise BrierError("logp must be finite")
    if y.shape != (lp.shape[0],) or not np.issubdtype(y.dtype, np.integer):
        raise BrierError("labels must be an int array of shape (N,)")
    if np.any((y < 0) | (y >= lp.shape[1])):
        raise BrierError("labels must be in [0, K)")
    if lp.shape[0] < MIN_ITEMS:
        raise InsufficientDataError(f"need at least {MIN_ITEMS} labelled items, got {lp.shape[0]}")
    return lp, y


def fit_temperature(logp: npt.ArrayLike, labels: npt.ArrayLike, *, tol: float = 1e-6) -> float:
    """Fit ``T`` minimising mean NLL of ``norm(logp / T)`` on labelled items.

    Golden-section search over ``t = log T`` in ``[-3, 3]``; the result is compared with
    both endpoints and the best of the three is returned.

    Parameters
    ----------
    logp : array_like
        ``(N, K)`` finite log-probabilities (option order).
    labels : array_like
        ``(N,)`` int labels in ``[0, K)``.
    tol : float
        Width of the final bracket in ``t``.

    Returns
    -------
    float
        The fitted temperature ``T`` in ``[e^-3, e^3]``.

    Raises
    ------
    InsufficientDataError
        If there are fewer than 50 items.
    BrierError
        If inputs are malformed.
    """
    if isinstance(tol, bool) or not (math.isfinite(float(tol)) and float(tol) > 0):
        raise BrierError("tol must be a finite number > 0")
    lp, y = _check(logp, labels)
    rows = np.arange(len(y))

    def nll(t: float) -> float:
        return float(-np.mean(norm(lp / math.exp(t))[rows, y]))

    a, b = LOG_T_BOUNDS
    c, d = b - _INV_PHI * (b - a), a + _INV_PHI * (b - a)
    fc, fd = nll(c), nll(d)
    while b - a > tol:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = nll(d)
    mid = (a + b) / 2
    # The NLL is unimodal in t for practical data; checking the endpoints guards the rest.
    candidates = [(nll(mid), mid), (nll(LOG_T_BOUNDS[0]), LOG_T_BOUNDS[0])]
    candidates.append((nll(LOG_T_BOUNDS[1]), LOG_T_BOUNDS[1]))
    best_t = min(candidates, key=lambda c: c[0])[1]
    return math.exp(best_t)
