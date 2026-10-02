"""L2 shrinkage-LDA head on standardised hidden states (METHODS.md, "Shrinkage LDA").

``S`` is the pooled within-class covariance ``Z^T Z / n`` of the residuals
``Z = X - M[y]``, shrunk to ``S_g = (1 - gamma) S + gamma mu I`` with ``mu = tr(S) / d``
and the Ledoit-Wolf (2004) ``gamma``. The Ledoit-Wolf terms come from the ``n x n``
Gram matrix ``G = Z Z^T`` (no ``d x d`` matrix is formed):

* ``tr(S) = tr(G) / n`` and ``||S||_F^2 = ||G||_F^2 / n^2``
* ``delta^2 = ||S - mu I||_F^2 = ||S||_F^2 - mu^2 d``
* ``beta_bar^2 = (1/n^2) sum_k ||z_k z_k^T - S||_F^2``
  ``= (1/n^2) sum_k (||z_k||^4 - 2 sum_j G_kj^2 / n + ||S||_F^2)``
* ``gamma = min(beta_bar^2, delta^2) / delta^2``

``U = S_g^{-1} M^T`` uses the Woodbury identity when ``n < d``. Scores:
``s_c = x . u_c - (1/2) m_c . u_c + log pi_c``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray
from brier.errors import BrierError, InsufficientDataError
from brier.heads._features import check_training, fit_standardiser, read_only, standardise

# Woodbury's relative error grows ~2e-15 / gamma; 1e-8 keeps it near 1e-7 (real data: >= 1e-2).
MIN_SHRINKAGE = 1e-8


@dataclass(frozen=True)
class LdaHead:
    """Fitted shrinkage-LDA head: standardiser plus linear discriminants."""

    mean: FloatArray
    scale: FloatArray
    coef: FloatArray  # (d, C) = S_g^{-1} M^T
    intercept: FloatArray  # (C,)
    shrinkage: float  # Ledoit-Wolf gamma in [0, 1]

    def scores(self, h: npt.ArrayLike) -> FloatArray:
        """``(n, C)`` float64 class scores for ``(n, d)`` hidden states."""
        out: FloatArray = standardise(h, self.mean, self.scale) @ self.coef + self.intercept
        return out


def fit_lda(h: npt.ArrayLike, y: npt.ArrayLike, n_classes: int) -> LdaHead:
    """Fit a shrinkage-LDA head on hidden states.

    Parameters
    ----------
    h : array_like
        ``(n, d)`` hidden states (any float dtype; computed in float64).
    y : array_like
        ``(n,)`` int labels in ``[0, n_classes)``; every class must appear.
    n_classes : int
        Number of answers ``C`` (>= 2).

    Returns
    -------
    LdaHead

    Raises
    ------
    InsufficientDataError
        If a class has no training examples.
    BrierError
        If inputs are malformed or there is no within-class variance.
    """
    h64, labels = check_training(h, y, n_classes)
    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        raise InsufficientDataError("every class needs at least one training example")
    mean, scale = fit_standardiser(h64)
    x = (h64 - mean) / scale
    n, d = x.shape
    means = (np.eye(n_classes)[labels].T @ x) / counts[:, None]  # (C, d)
    z = x - means[labels]
    gram = z @ z.T
    mu = float(np.trace(gram)) / (n * d)
    if mu <= 0:
        raise BrierError("no within-class variance: LDA is undefined")
    fro2 = float(np.sum(gram**2)) / n**2
    delta2 = fro2 - mu**2 * d
    beta_bar2 = float(np.sum(np.diag(gram) ** 2 - 2 * np.sum(gram**2, axis=1) / n + fro2)) / n**2
    gamma = min(max(beta_bar2, 0.0), delta2) / delta2 if delta2 > 0 else 1.0
    a, b = (1 - gamma) / n, gamma * mu  # S_g = a Z^TZ + b I
    rhs = means.T  # (d, C)
    if gamma <= MIN_SHRINKAGE and n - n_classes < d:
        # Z has rank <= n - C < d, so S is singular and there is ~nothing to shrink it with.
        raise BrierError("degenerate covariance: Ledoit-Wolf shrinkage is ~0 and S is singular")
    if n < d:
        # (bI + a Z^TZ)^{-1} = I/b - (a/b^2) Z^T (I + (a/b) Z Z^T)^{-1} Z
        inner = np.linalg.solve(np.eye(n) + (a / b) * gram, z @ rhs)
        coef = rhs / b - (a / b**2) * (z.T @ inner)
    else:
        try:
            solved = np.linalg.solve(a * (z.T @ z) + b * np.eye(d), rhs)
        except np.linalg.LinAlgError as e:
            raise BrierError("singular shrunk covariance") from e
        coef = np.asarray(solved, dtype=np.float64)
    intercept = -0.5 * np.sum(rhs * coef, axis=0) + np.log(counts / n)
    return LdaHead(mean, scale, read_only(coef), read_only(intercept), float(gamma))
