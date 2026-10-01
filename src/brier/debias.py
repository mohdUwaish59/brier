"""Level ``L0``: option-rotation debiasing and batch prior correction (METHODS.md)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import SupportsFloat

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray, norm
from brier.backends.base import Backend
from brier.errors import BrierError, QuestionError
from brier.questions import Choice, Question
from brier.readout import raw_logprobs

PRIOR_FLOOR = 1e-6


def unrotate(position_logp: npt.ArrayLike, shift: int) -> FloatArray:
    """Map display-position log-probs of rotation ``shift`` back to option order.

    ``l_k = norm(z)_{(k - shift) mod K}``.
    """
    z = norm(position_logp)
    k = z.shape[-1]
    return z[..., (np.arange(k) - shift) % k]


def combine(option_logps: Sequence[npt.ArrayLike]) -> FloatArray:
    """Geometric mean over rotations, renormalised: ``norm(mean_s l^(s))``."""
    if not option_logps:
        raise BrierError("combine needs at least one rotation")
    return norm(np.mean(np.stack([np.asarray(a, dtype=np.float64) for a in option_logps]), 0))


def l0_logprobs(
    backend: Backend,
    states: Sequence[str],
    question: Question,
    shifts: Sequence[int] | None = None,
) -> FloatArray:
    """``(n_states, K)`` rotation-debiased log-probs in option order (no prior).

    Parameters
    ----------
    backend : Backend
        Model access.
    states : Sequence[str]
        State texts.
    question : Choice, Noul or Score
        Only Choice is rotated; Noul and Score return the raw readout.
    shifts : Sequence[int] or None
        Distinct rotations in ``[0, K)``; default all K (Choice only).

    Returns
    -------
    numpy.ndarray
        float64 log-probabilities, rows sum to 1 in probability space.
    """
    if not isinstance(question, Choice):
        if shifts is not None:
            raise QuestionError("only Choice questions can be rotated")
        return raw_logprobs(backend, states, question)
    k = len(question.options)
    s_list = list(range(k)) if shifts is None else list(shifts)
    if not s_list or len(set(s_list)) != len(s_list) or not all(0 <= s < k for s in s_list):
        raise QuestionError(f"shifts must be distinct values in [0, {k})")
    return combine([unrotate(raw_logprobs(backend, states, question, s), s) for s in s_list])


def fit_prior(logp: npt.ArrayLike) -> FloatArray:
    """Label-free prior from an unlabelled pool: mean probability, floored, renormalised.

    ``π_k = (1/N) Σ_i p_{i,k}``, clipped at 1e-6 and renormalised (Batch Calibration).
    """
    lp = np.asarray(logp, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[0] == 0:
        raise BrierError("fit_prior needs a non-empty (N, K) array")
    if np.any(np.isnan(lp) | (lp == np.inf)):
        raise BrierError("fit_prior needs log-probabilities without NaN or +inf")
    p = np.exp(lp)
    pi: FloatArray = np.maximum(p.mean(axis=0), PRIOR_FLOOR)
    pi /= pi.sum()
    return pi


def apply_prior(logp: npt.ArrayLike, prior: npt.ArrayLike, lam: SupportsFloat = 1.0) -> FloatArray:
    """Prior correction ``norm(log p - lam * log pi)``.

    Parameters
    ----------
    logp : array_like
        ``(N, K)`` log-probabilities in option order.
    prior : array_like
        ``(K,)`` prior from :func:`fit_prior`.
    lam : float
        Correction strength in ``[0, 1]`` (default 1.0, ADR-0005).

    Returns
    -------
    numpy.ndarray
        float64 corrected log-probabilities.
    """
    if isinstance(lam, bool) or not 0.0 <= float(lam) <= 1.0:  # NaN fails the range too
        raise BrierError("lam must be in [0, 1]")
    lp = np.asarray(logp, dtype=np.float64)
    pi = np.asarray(prior, dtype=np.float64)
    if pi.shape != lp.shape[-1:]:
        raise BrierError(f"prior shape {pi.shape} does not match {lp.shape[-1:]}")
    if not np.all(np.isfinite(pi) & (pi > 0)):
        raise BrierError("prior must be finite and positive")
    return norm(lp - float(lam) * np.log(pi))
