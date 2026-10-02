"""Shared input checks and feature standardisation for L2 heads (METHODS.md, L2)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray
from brier.errors import BrierError

SCALE_FLOOR = 1e-6
IntArray = npt.NDArray[np.integer[Any]]


def check_training(
    h: npt.ArrayLike, y: npt.ArrayLike, n_classes: int
) -> tuple[FloatArray, IntArray]:
    """Validate ``(n, d)`` finite features, ``(n,)`` int labels in ``[0, n_classes)``."""
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) or n_classes < 2:
        raise BrierError("n_classes must be an int >= 2")
    x = np.asarray(h, dtype=np.float64)
    labels = np.asarray(y)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise BrierError("features must be an (n, d) array with n >= 2")
    if not np.all(np.isfinite(x)):
        raise BrierError("features must be finite")
    if labels.shape != (x.shape[0],) or not np.issubdtype(labels.dtype, np.integer):
        raise BrierError("labels must be an int array of shape (n,)")
    if np.any((labels < 0) | (labels >= n_classes)):
        raise BrierError(f"labels must be in [0, {n_classes})")
    return x, labels


def fit_standardiser(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Training mean and standard deviation (floored at 1e-6), read-only."""
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), SCALE_FLOOR)
    mean.flags.writeable = False
    scale.flags.writeable = False
    return mean, scale


def standardise(h: npt.ArrayLike, mean: FloatArray, scale: FloatArray) -> FloatArray:
    """``(h - mean) / scale`` after checking shape and finiteness."""
    x = np.asarray(h, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != mean.shape[0]:
        raise BrierError(f"features must be an (n, {mean.shape[0]}) array")
    if not np.all(np.isfinite(x)):
        raise BrierError("features must be finite")
    out: FloatArray = (x - mean) / scale
    return out


def read_only(a: FloatArray) -> FloatArray:
    """Return ``a`` marked read-only (heads are immutable value objects)."""
    a.flags.writeable = False
    return a
