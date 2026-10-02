"""L2 ridge head on standardised hidden states (METHODS.md, "Dual ridge").

One-hot targets ``Y`` are centred per column; with standardised (zero-mean) features the
intercept is the class-frequency vector ``Ybar``:

* dual (``n < d``):   ``A = (X X^T + alpha I_n)^{-1} (Y - Ybar)``,  ``W = X^T A``
* primal (``n >= d``): ``W = (X^TX + alpha I_d)^{-1} X^T (Y - Ybar)``

Both give the same ``W``; scores are ``x W + Ybar``. Solved with ``np.linalg.solve``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray
from brier.errors import BrierError
from brier.heads._features import check_training, fit_standardiser, read_only, standardise

Solver = Literal["auto", "dual", "primal"]


@dataclass(frozen=True)
class RidgeHead:
    """Fitted ridge head: standardiser plus linear map to class scores."""

    mean: FloatArray
    scale: FloatArray
    weights: FloatArray  # (d, C)
    bias: FloatArray  # (C,) class frequencies
    solver: Literal["dual", "primal"]

    def scores(self, h: npt.ArrayLike) -> FloatArray:
        """``(n, C)`` float64 class scores for ``(n, d)`` hidden states."""
        out: FloatArray = standardise(h, self.mean, self.scale) @ self.weights + self.bias
        return out


def fit_ridge(
    h: npt.ArrayLike,
    y: npt.ArrayLike,
    n_classes: int,
    alpha: float = 1.0,
    solver: Solver = "auto",
) -> RidgeHead:
    """Fit a ridge head on hidden states.

    Parameters
    ----------
    h : array_like
        ``(n, d)`` hidden states (any float dtype; computed in float64).
    y : array_like
        ``(n,)`` int labels in ``[0, n_classes)``.
    n_classes : int
        Number of answers ``C`` (>= 2).
    alpha : float
        Ridge penalty, finite and > 0.
    solver : {"auto", "dual", "primal"}
        ``"auto"`` uses the dual form when ``n < d``.

    Returns
    -------
    RidgeHead
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise BrierError("alpha must be a number")
    if not (math.isfinite(alpha) and alpha > 0):
        raise BrierError("alpha must be finite and > 0")
    if solver not in ("auto", "dual", "primal"):
        raise BrierError("solver must be 'auto', 'dual' or 'primal'")
    h64, labels = check_training(h, y, n_classes)
    mean, scale = fit_standardiser(h64)
    x = (h64 - mean) / scale
    onehot = np.eye(n_classes)[labels]
    ybar = onehot.mean(axis=0)
    yc = onehot - ybar
    n, d = x.shape
    use: Literal["dual", "primal"] = ("dual" if n < d else "primal") if solver == "auto" else solver
    if use == "dual":
        a = np.linalg.solve(x @ x.T + alpha * np.eye(n), yc)
        weights = x.T @ a
    else:
        weights = np.linalg.solve(x.T @ x + alpha * np.eye(d), x.T @ yc)
    return RidgeHead(mean, scale, read_only(weights), read_only(ybar), use)
