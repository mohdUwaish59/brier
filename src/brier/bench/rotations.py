"""How many L0 rotations are needed? (ROADMAP M7.2, ``python -m brier.bench rotations``).

L0 averages a Choice question over all K cyclic rotations of its options (K forward passes).
This module runs every rotation once, for the test split and the unlabelled pool in both
option orders, stores the per-rotation log-probabilities, and scores evenly spaced subsets of
``m`` rotations offline: each subset combines its rotations by geometric mean and fits its own
batch prior on the pool, exactly as L0 does with all K (METHODS.md, L0).
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from brier import __version__
from brier._math import FloatArray
from brier.backends.base import Backend
from brier.bench.run import SCHEMA_VERSION, _metrics, _pkg_version, _slug, _write
from brier.bench.tasks import Task
from brier.debias import apply_prior, combine, evenly_spaced, fit_prior, unrotate
from brier.questions import Choice
from brier.readout import raw_logprobs

_logger = logging.getLogger(__name__)

DEFAULT_COUNTS = (1, 2, 4, 5, 10, 20)
__all__ = ["evenly_spaced", "per_rotation", "run_rotations", "subset_probs"]


def per_rotation(backend: Backend, states: Sequence[str], question: Choice) -> FloatArray:
    """``(K, N, K)`` log-probs per rotation, mapped back to option order."""
    k = len(question.options)
    out = []
    for s in range(k):
        _logger.info("rotation %d/%d (%d states)", s + 1, k, len(states))
        out.append(unrotate(raw_logprobs(backend, states, question, s), s))
    return np.stack(out)


def subset_probs(
    test_rot: FloatArray,
    pool_rot: FloatArray | None,
    shifts: Sequence[int],
    lam: float = 1.0,
) -> FloatArray:
    """``(N, K)`` L0 probabilities from the rotations in ``shifts``.

    Geometric-mean combine of those rotations; if the pool is non-empty, a batch prior fitted
    on the pool with the *same* rotations is applied with strength ``lam``.
    """
    logp = combine([test_rot[s] for s in shifts])
    if pool_rot is not None and pool_rot.shape[1] > 0:
        prior = fit_prior(combine([pool_rot[s] for s in shifts]))
        logp = apply_prior(logp, prior, lam=lam)
    p: FloatArray = np.exp(logp)
    return p


def run_rotations(
    backend: Backend,
    task: Task,
    out_dir: str | Path,
    *,
    ms: Sequence[int] = DEFAULT_COUNTS,
    limit: int | None = None,
    n_resamples: int = 1000,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Score L0 with ``m`` evenly spaced rotations for each ``m`` in ``ms``.

    Writes ``<task>_<model>_seed<seed>_rotations.{json,npz}``: the JSON holds, per ``m``, the
    shifts, the forward passes per decision and the usual metrics with bootstrap CIs (flip
    rate under the reversed option order); the npz holds the per-rotation log-probabilities
    (no text), so other subsets can be scored later without a model.

    Raises
    ------
    BrierError
        If a count is outside ``[1, K]``.
    """
    q = task.question
    k = len(q.options)
    counts = sorted({int(m) for m in ms})
    plans = {m: evenly_spaced(k, m) for m in counts}  # validates before any forward pass
    q_rev = Choice(q.text, list(reversed(q.options)), name=q.name)
    states = list(task.test_states[:limit])
    pool = list(task.pool[:limit])
    labels = np.asarray(task.test_labels[:limit], dtype=np.int64)

    start = time.perf_counter()
    arrays: dict[str, Any] = {
        "labels": labels,
        "options": np.array(q.options),
        "test_rot": per_rotation(backend, states, q),
        "test_rot_reversed": per_rotation(backend, states, q_rev),
        "pool_rot": per_rotation(backend, pool, q),
        "pool_rot_reversed": per_rotation(backend, pool, q_rev),
    }
    curve: dict[str, Any] = {}
    for m, shifts in plans.items():
        p = subset_probs(arrays["test_rot"], arrays["pool_rot"], shifts)
        p_rev = subset_probs(arrays["test_rot_reversed"], arrays["pool_rot_reversed"], shifts)
        curve[str(m)] = {
            "shifts": shifts,
            "forward_passes": m,
            "metrics": _metrics(p, p_rev[:, ::-1], labels, n_resamples),
        }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "brier_version": __version__,
        "git_commit": git_commit,
        "model": {
            "id": backend.model_id,
            "revision": backend.revision,
            "dtype": getattr(backend, "dtype", None),
        },
        "env": {
            "python": platform.python_version(),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "device": getattr(backend, "device", None),
        },
        "task": task.name,
        "dataset": dict(task.source),
        "split_seed": task.seed,
        "n_test": len(states),
        "n_pool": len(pool),
        "rotation_curve": curve,
        "timing": {"wall_seconds": round(time.perf_counter() - start, 3)},
    }
    _write(
        Path(out_dir),
        f"{task.name}_{_slug(backend.model_id)}_seed{task.seed}_rotations",
        summary,
        arrays,
    )
    return summary
