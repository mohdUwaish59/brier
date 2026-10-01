"""Benchmark runner: predictions per level, metrics with CIs, result JSON + per-item npz.

The npz holds labels, option names and probabilities only (no state text, THREAT_MODEL T6).
"""

from __future__ import annotations

import json
import platform
import re
import time
from collections.abc import Sequence
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from brier import Decider, __version__
from brier._math import FloatArray
from brier.backends.base import Backend
from brier.bench.tasks import Task
from brier.decision import Level
from brier.errors import BrierError
from brier.metrics import (
    Metric,
    accuracy,
    aurc,
    bootstrap_ci,
    brier,
    coverage_at_risk,
    ece,
    flip_rate,
    nll,
    paired_bootstrap_ci,
)
from brier.questions import Choice

SCHEMA_VERSION = 1
SUPPORTED_LEVELS: tuple[Level, ...] = ("raw", "L0")
COVERAGE_ALPHA = 0.05
_METRICS: dict[str, Metric] = {
    "accuracy": accuracy,
    "nll": nll,
    "brier": brier,
    "ece": ece,
    "ece_width": partial(ece, scheme="width"),
    "aurc": aurc,
    f"coverage_at_risk_{COVERAGE_ALPHA}_in_sample": partial(coverage_at_risk, alpha=COVERAGE_ALPHA),
}


def check_levels(levels: Sequence[str]) -> list[Level]:
    """Validate requested levels (only ``raw`` and ``L0`` exist until M4/M5)."""
    out = list(levels)
    if not out or len(set(out)) != len(out) or not set(out) <= set(SUPPORTED_LEVELS):
        raise BrierError(f"levels must be distinct values from {SUPPORTED_LEVELS}")
    return [lv for lv in SUPPORTED_LEVELS if lv in out]


def run_task(
    backend: Backend,
    task: Task,
    levels: Sequence[str],
    out_dir: str | Path,
    *,
    limit: int | None = None,
    n_resamples: int = 1000,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Evaluate ``levels`` on ``task`` and write ``<task>_<model>_seed<seed>.{json,npz}``.

    Results are rewritten after each level, so a crash keeps the finished levels.
    Flip rate re-runs each level with the option list reversed (doubling its cost).

    Parameters
    ----------
    backend : Backend
        Model access.
    task : Task
        See :mod:`brier.bench.tasks`.
    levels : Sequence[str]
        Subset of ``("raw", "L0")``.
    out_dir : str or Path
        Output directory (created if needed).
    limit : int or None
        Use only the first ``limit`` test items and pool states (smoke runs).
    n_resamples : int
        Bootstrap resamples per CI.
    git_commit : str or None
        Recorded in the result file.

    Returns
    -------
    dict
        The result summary written to JSON.
    """
    lvls = check_levels(levels)
    q = task.question
    q_rev = Choice(q.text, list(reversed(q.options)), name=q.name)
    states = list(task.test_states[:limit])
    labels = np.asarray(task.test_labels[:limit], dtype=np.int64)
    pool = list(task.pool[:limit]) if "L0" in lvls else []
    decider = Decider(backend)
    if pool:
        decider.fit_prior(pool, [q])
        decider.fit_prior(pool, [q_rev])  # same name, so a separate call

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
        "levels": {},
        "comparisons": {},
        "timing": {"forward_passes_per_decision": {}, "wall_seconds": {}},
    }
    arrays: dict[str, Any] = {"labels": labels, "options": np.array(q.options)}
    stem = f"{task.name}_{_slug(backend.model_id)}_seed{task.seed}"
    for level in lvls:
        start = time.perf_counter()
        p = _predict(decider, states, q, level)
        p_rev = _predict(decider, states, q_rev, level, order=q.options)
        summary["timing"]["wall_seconds"][level] = round(time.perf_counter() - start, 3)
        summary["timing"]["forward_passes_per_decision"][level] = (
            len(q.options) if level == "L0" else 1
        )
        arrays[f"{level}_probs"], arrays[f"{level}_probs_reversed"] = p, p_rev
        summary["levels"][level] = _metrics(p, p_rev, labels, n_resamples)
        if level != "raw" and "raw" in summary["levels"]:
            base = (arrays["raw_probs"], arrays["raw_probs_reversed"])
            summary["comparisons"][f"{level}_minus_raw"] = _paired(
                (p, p_rev), base, labels, n_resamples
            )
        _write(Path(out_dir), stem, summary, arrays)
    return summary


def _predict(
    decider: Decider,
    states: list[str],
    q: Choice,
    level: Level,
    order: Sequence[str] | None = None,
) -> FloatArray:
    """``(n, K)`` probabilities with columns in ``order`` (default: ``q.options``)."""
    cols = list(order if order is not None else q.options)
    rows: list[list[float]] = []
    for start in range(0, len(states), decider.max_batch):
        batch = decider.decide_batch(states[start : start + decider.max_batch], [q], level)
        rows += [[res[q.name].probs[c] for c in cols] for res in batch]
    return np.asarray(rows, dtype=np.float64).reshape(len(states), len(cols))


def _metrics(
    p: FloatArray, p_rev: FloatArray, y: np.ndarray[Any, Any], n: int
) -> dict[str, list[float]]:
    out = {name: list(bootstrap_ci(fn, p, y, n_resamples=n)) for name, fn in _METRICS.items()}
    out["flip_rate"] = list(bootstrap_ci(flip_rate, p, p_rev, n_resamples=n))
    return out


def _paired(
    a: tuple[FloatArray, FloatArray],
    b: tuple[FloatArray, FloatArray],
    y: np.ndarray[Any, Any],
    n: int,
) -> dict[str, list[float]]:
    out = {
        name: list(paired_bootstrap_ci(fn, (a[0], y), (b[0], y), n_resamples=n))
        for name, fn in _METRICS.items()
    }
    out["flip_rate"] = list(paired_bootstrap_ci(flip_rate, a, b, n_resamples=n))
    return out


def _write(out: Path, stem: str, summary: dict[str, Any], arrays: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{stem}.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tmp.replace(out / f"{stem}.json")
    tmp_npz = out / f"{stem}.npz.tmp"
    with tmp_npz.open("wb") as f:
        np.savez(f, **arrays)
    tmp_npz.replace(out / f"{stem}.npz")


def _slug(model_id: str) -> str:
    """File-name-safe model id (no path separators or leading dots)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id.replace("/", "__")).lstrip(".") or "model"


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
