"""Level ``L0``: option-rotation debiasing and batch prior correction (METHODS.md)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import SupportsFloat

import numpy as np
import numpy.typing as npt

from brier._math import FloatArray, norm
from brier.backends.base import Backend
from brier.errors import BrierError, QuestionError
from brier.prompts import render
from brier.questions import Choice, Question
from brier.readout import raw_logprobs, resolve_labels

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


MAX_ROTATIONS = 26  # Choice allows at most 26 options


def evenly_spaced(k: int, m: int) -> list[int]:
    """``m`` rotations spread evenly over ``K`` options: ``floor(j K / m)`` for ``j < m``.

    Raises
    ------
    BrierError
        If ``m`` is not in ``[1, K]``.
    """
    if not 1 <= m <= k:
        raise BrierError(f"rotation count must be in [1, {k}], got {m}")
    return [(j * k) // m for j in range(m)]


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

    Notes
    -----
    All rotations go to the backend in one ``label_logprobs`` call (ADR-0010, phase A): the
    label tokens are positional, so they are the same for every rotation, and a backend that
    shares prompt prefixes within a call can read each state once.
    """
    if not isinstance(question, Choice):
        if shifts is not None:
            raise QuestionError("only Choice questions can be rotated")
        return raw_logprobs(backend, states, question)
    k = len(question.options)
    s_list = list(range(k)) if shifts is None else list(shifts)
    if not s_list or len(set(s_list)) != len(s_list) or not all(0 <= s < k for s in s_list):
        raise QuestionError(f"shifts must be distinct values in [0, {k})")
    _, token_ids = resolve_labels(backend, question)
    position_logp = backend.label_logprobs(rotated_prompts(states, question, s_list), token_ids)
    return combine_rotated(position_logp, s_list, len(states))


def rotated_prompts(states: Sequence[str], question: Choice, shifts: Sequence[int]) -> list[str]:
    """Prompts for every state under every rotation, shift-major.

    All states for ``shifts[0]`` come first, then all states for ``shifts[1]``, and so on.
    """
    return [render(state, question, shift=s) for s in shifts for state in states]


def combine_rotated(
    position_logp: npt.ArrayLike, shifts: Sequence[int], n_states: int
) -> FloatArray:
    """Combine shift-major display-position log-probs into option order.

    ``position_logp`` has ``len(shifts) * n_states`` rows (see :func:`rotated_prompts`); the
    result is ``(n_states, K)``: each block normalised, unrotated, then geometric mean.
    """
    lp = np.asarray(position_logp, dtype=np.float64)
    if lp.ndim != 2 or lp.shape[0] != len(shifts) * n_states:
        raise BrierError(f"backend returned {lp.shape}, expected {len(shifts) * n_states} rows")
    blocks = lp.reshape(len(shifts), n_states, lp.shape[1])
    return combine([unrotate(norm(block), s) for block, s in zip(blocks, shifts, strict=True)])


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
