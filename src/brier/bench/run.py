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
from brier.calibrate.temperature import apply_temperature, fit_temperature
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
SUPPORTED_LEVELS: tuple[Level, ...] = ("raw", "L0", "L1")
_LOG_FLOOR = 1e-300  # probabilities that underflowed to 0, so log stays finite
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
    """Validate requested levels (``raw``, ``L0``, ``L1``; L1 is a transform of L0)."""
    out = list(levels)
    if not out or len(set(out)) != len(out) or not set(out) <= set(SUPPORTED_LEVELS):
        raise BrierError(f"levels must be distinct values from {SUPPORTED_LEVELS}")
    if "L1" in out and "L0" not in out:
        raise BrierError("L1 is fitted on top of L0; request L0 too")
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

    Levels run in the order raw, L0, L1 and results are rewritten after each one, so a crash
    keeps the finished levels; the L0 prior is fitted only when L0 starts. L1 fits a
    temperature on L0 predictions for the calibration split and applies it to the stored
    test-set L0 probabilities (what ``Decider`` does), so it only adds the calibration items.
    Flip rate re-runs each level with the option list reversed (doubling its cost).

    Parameters
    ----------
    backend : Backend
        Model access.
    task : Task
        See :mod:`brier.bench.tasks`.
    levels : Sequence[str]
        Subset of ``("raw", "L0", "L1")``; L1 requires L0.
    out_dir : str or Path
        Output directory (created if needed).
    limit : int or None
        Use only the first ``limit`` test, pool and calibration items (smoke runs).
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
        "calibration": {},
        "comparisons": {},
        "timing": {"forward_passes_per_decision": {}, "wall_seconds": {}},
    }
    arrays: dict[str, Any] = {"labels": labels, "options": np.array(q.options)}
    stem = f"{task.name}_{_slug(backend.model_id)}_seed{task.seed}"
    for level in lvls:  # raw first, so its results are written before the slow prior fit
        start = time.perf_counter()
        if level == "L0" and pool:  # L0 wall time includes fitting its prior
            decider.fit_prior(pool, [q])
            decider.fit_prior(pool, [q_rev])  # same name, so a separate call
        if level == "L1":
            p, p_rev, summary["calibration"]["L1"] = _l1(decider, task, q, q_rev, arrays, limit)
        else:
            p = _predict(decider, states, q, level)
            p_rev = _predict(decider, states, q_rev, level, order=q.options)
        summary["timing"]["wall_seconds"][level] = round(time.perf_counter() - start, 3)
        summary["timing"]["forward_passes_per_decision"][level] = (
            1 if level == "raw" else len(q.options)
        )
        arrays[f"{level}_probs"], arrays[f"{level}_probs_reversed"] = p, p_rev
        summary["levels"][level] = _metrics(p, p_rev, labels, n_resamples)
        for ref in ("raw", "L0"):
            if ref != level and ref in summary["levels"] and level in ("L0", "L1"):
                base = (arrays[f"{ref}_probs"], arrays[f"{ref}_probs_reversed"])
                summary["comparisons"][f"{level}_minus_{ref}"] = _paired(
                    (p, p_rev), base, labels, n_resamples
                )
        _write(Path(out_dir), stem, summary, arrays)
    return summary


def _l1(
    decider: Decider,
    task: Task,
    q: Choice,
    q_rev: Choice,
    arrays: dict[str, Any],
    limit: int | None,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Fit T (per option order) on the calibration split; apply to the test L0 probs."""
    cal_states = list(task.calib_states[:limit])
    cal_y = np.asarray(task.calib_labels[:limit], dtype=np.int64)
    temps = {}
    for key, question in (("temperature", q), ("temperature_reversed", q_rev)):
        cal_p = _predict(decider, cal_states, question, "L0", order=q.options)
        temps[key] = fit_temperature(_log(cal_p), cal_y)
    p = np.exp(apply_temperature(_log(arrays["L0_probs"]), temps["temperature"]))
    p_rev = np.exp(
        apply_temperature(_log(arrays["L0_probs_reversed"]), temps["temperature_reversed"])
    )
    return p, p_rev, {**temps, "n_calib": len(cal_states)}


def _log(p: FloatArray) -> FloatArray:
    out: FloatArray = np.log(np.maximum(p, _LOG_FLOOR))
    return out


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
