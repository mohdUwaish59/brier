"""A fitted L2 head in the single affine form stored in artifacts (ADR-0006).

``scores = ((h - mean) / scale) @ weights + bias`` covers both head families: ridge maps to
``(W, Ybar)``, shrinkage LDA to ``(S_g^{-1} M^T, intercept)``. Probabilities are
``norm(scores / temperature)`` (METHODS.md, L2 "Probabilities").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray, norm
from brier.errors import BrierError
from brier.heads._features import SCALE_FLOOR, standardise
from brier.heads.lda import LdaHead
from brier.heads.select import L2_LOG_T_BOUNDS, L2Selection

MAX_HIDDEN = 65_536
_T_RANGE = (math.exp(L2_LOG_T_BOUNDS[0]) * (1 - 1e-9), math.exp(L2_LOG_T_BOUNDS[1]) * (1 + 1e-9))


def _number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _as_float(x: object) -> float:
    """``float(x)`` for real numbers, NaN for anything else (incl. ints too big for a float)."""
    if not _number(x):
        return math.nan
    try:
        return float(x)  # type: ignore[arg-type]
    except OverflowError:
        return math.nan


def _array(a: Any, ndim: int, name: str) -> FloatArray:
    arr = np.array(a, dtype=np.float64)  # copy, so the caller's array cannot change the head
    if arr.ndim != ndim or not np.all(np.isfinite(arr)):
        raise BrierError(f"{name} must be a finite {ndim}-D array")
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, eq=False)
class L2Head:
    """Selected, refitted L2 head with its OOF temperature and diagnostics (immutable)."""

    layer: int
    solver: Literal["ridge", "lda"]
    alpha: float | None
    temperature: float
    temperature_at_bound: bool
    oof_nll: float
    oof_accuracy: float
    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    bias: FloatArray

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise BrierError("layer must be an int >= 0")
        if self.solver not in ("ridge", "lda"):
            raise BrierError("solver must be 'ridge' or 'lda'")
        alpha = self.alpha
        if self.solver == "ridge":
            a = _as_float(alpha)
            if not (math.isfinite(a) and a > 0):
                raise BrierError("a ridge head needs a finite alpha > 0")
        elif alpha is not None:
            raise BrierError("an LDA head has no alpha")
        if not _T_RANGE[0] <= _as_float(self.temperature) <= _T_RANGE[1]:  # NaN fails
            raise BrierError("temperature must be a number in L2's range [e^-7, e^7]")
        if not isinstance(self.temperature_at_bound, bool):
            raise BrierError("temperature_at_bound must be a bool")
        oof_nll = _as_float(self.oof_nll)
        if not (math.isfinite(oof_nll) and oof_nll >= 0):
            raise BrierError("oof_nll must be a finite number >= 0")
        if not 0 <= _as_float(self.oof_accuracy) <= 1:  # NaN fails
            raise BrierError("oof_accuracy must be in [0, 1]")
        mean = _array(self.mean, 1, "mean")
        scale = _array(self.scale, 1, "scale")
        weights = _array(self.weights, 2, "weights")
        bias = _array(self.bias, 1, "bias")
        d, c = weights.shape
        if not 1 <= d <= MAX_HIDDEN or mean.shape != (d,) or scale.shape != (d,):
            raise BrierError(f"mean, scale and weights must agree on a hidden size <= {MAX_HIDDEN}")
        if c < 2 or bias.shape != (c,):
            raise BrierError("weights and bias must agree on >= 2 classes")
        if np.any(scale < SCALE_FLOOR):  # fitting floors the scale at 1e-6
            raise BrierError(f"scale must be >= {SCALE_FLOOR}")
        for name, arr in (("mean", mean), ("scale", scale), ("weights", weights), ("bias", bias)):
            object.__setattr__(self, name, arr)
        object.__setattr__(self, "temperature", float(self.temperature))

    @property
    def n_classes(self) -> int:
        """Number of classes ``C``."""
        return int(self.weights.shape[1])

    def log_probs(self, h: npt.ArrayLike) -> FloatArray:
        """``(n, C)`` log-probabilities ``norm(scores / T)`` for hidden states of ``layer``."""
        scores = standardise(h, self.mean, self.scale) @ self.weights + self.bias
        return norm(scores / self.temperature)

    @classmethod
    def from_selection(cls, sel: L2Selection) -> L2Head:
        """Convert a :class:`~brier.heads.select.L2Selection` to the stored affine form."""
        head = sel.head
        weights, bias = (
            (head.coef, head.intercept) if isinstance(head, LdaHead) else (head.weights, head.bias)
        )
        return cls(
            layer=sel.layer,
            solver=sel.solver,
            alpha=sel.alpha,
            temperature=sel.temperature,
            temperature_at_bound=sel.temperature_at_bound,
            oof_nll=sel.oof_nll,
            oof_accuracy=sel.oof_accuracy,
            mean=head.mean,
            scale=head.scale,
            weights=weights,
            bias=bias,
        )
