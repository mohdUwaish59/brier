"""Numerically stable log-space helpers (float64)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def logsumexp(x: npt.ArrayLike, axis: int = -1) -> FloatArray:
    """Stable ``log(sum(exp(x)))`` along ``axis``.

    Parameters
    ----------
    x : array_like
        Input values; ``-inf`` entries count as zero mass.
    axis : int
        Axis to reduce.

    Returns
    -------
    numpy.ndarray
        float64 array with ``axis`` removed (``-inf`` where every entry is ``-inf``).
    """
    a = np.asarray(x, dtype=np.float64)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # all -inf rows: avoid -inf - -inf = nan
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    squeezed: FloatArray = np.squeeze(out, axis=axis)
    return squeezed


def norm(x: npt.ArrayLike, axis: int = -1) -> FloatArray:
    """Log-space renormalisation ``x - logsumexp(x)`` (METHODS.md ``norm``).

    Parameters
    ----------
    x : array_like
        Unnormalised log-probabilities.
    axis : int
        Axis along which the result sums to 1 in probability space.

    Returns
    -------
    numpy.ndarray
        float64 log-probabilities with the same shape as ``x``. A slice with no
        mass (all ``-inf``) has no distribution and comes back as ``nan``; callers
        reject it (``Decision`` refuses non-finite probabilities).
    """
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return a - np.expand_dims(logsumexp(a, axis=axis), axis)


log_softmax = norm
