"""Benchmark feature cache and label budgets for L2 (ROADMAP M5.3).

Hidden states are the expensive part of an L2 benchmark: they are computed once per
(model, revision, dtype, prompt templates, layers, prompts) and reused for every label
budget and candidate head. Cache files hold the features and their key only; prompts are
represented by a SHA-256 digest, so no dataset text is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from brier import prompts as templates
from brier.backends.base import Backend
from brier.errors import BrierError


def _prompts_digest(prompts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for p in prompts:  # length-prefixed, so different splits never collide
        data = p.encode("utf-8")
        h.update(str(len(data)).encode("ascii") + b":" + data)
    return h.hexdigest()


def _cache_key(backend: Backend, prompts: Sequence[str], layers: Sequence[int]) -> str:
    key = {
        "model_id": backend.model_id,
        "revision": backend.revision,
        "dtype": getattr(backend, "dtype", None),
        "template_hash": templates.template_hash(),
        "layers": [int(b) for b in layers],
        "n_prompts": len(prompts),
        "prompts_sha256": _prompts_digest(prompts),
    }
    return json.dumps(key, sort_keys=True)


def cached_hidden_states(
    backend: Backend,
    prompts: Sequence[str],
    layers: Sequence[int],
    cache_dir: str | Path | None,
) -> npt.NDArray[np.float32]:
    """``backend.hidden_states(prompts, layers)``, cached on disk when ``cache_dir`` is set.

    A cache file is used only if its stored key matches exactly; unreadable or mismatched
    files are recomputed and overwritten.
    """
    if cache_dir is None:
        return np.asarray(backend.hidden_states(prompts, layers), dtype=np.float32)
    key = _cache_key(backend, prompts, layers)
    directory = Path(cache_dir)
    path = directory / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.npz"
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["key"]) == key and z["hidden"].shape[:2] == (len(prompts), len(layers)):
                    cached: npt.NDArray[np.float32] = z["hidden"].astype(np.float32)
                    return cached
        except Exception:  # noqa: S110 - unreadable cache: fall through and recompute
            pass
    hidden = np.asarray(backend.hidden_states(prompts, layers), dtype=np.float32)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as f:
        np.savez(f, hidden=hidden, key=np.array(key))
    os.replace(f.name, path)
    return hidden


def stratified_budget(
    labels: npt.ArrayLike, n_classes: int, budget: int, seed: int
) -> npt.NDArray[np.int64]:
    """Seeded, class-balanced subsample of ``budget`` item indices.

    Each class is shuffled, then classes are taken round-robin (one item per class per
    round) until ``budget`` items are chosen; exhausted classes are skipped.
    """
    y = np.asarray(labels)
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= len(y):
        raise BrierError(f"budget must be an int in [1, {len(y)}]")
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(np.flatnonzero(y == c))) for c in range(n_classes)]
    chosen: list[int] = []
    while len(chosen) < budget:
        for pool in pools:
            if pool and len(chosen) < budget:
                chosen.append(int(pool.pop()))
    return np.sort(np.asarray(chosen, dtype=np.int64))
