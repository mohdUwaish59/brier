"""Benchmark tasks: dataset download (pinned + checksummed), parsing and seeded splits.

BANKING77 (Casanueva et al. 2020, arXiv:2003.04807) is licensed CC-BY-4.0 by PolyAI.
It is downloaded as the original CSV from a pinned commit and parsed with ``csv``; no
dataset loading script is executed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from brier.errors import BrierError, InsufficientDataError
from brier.questions import Choice

BANKING77_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{BANKING77_COMMIT}/banking_data/train.csv"
)
BANKING77_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
BANKING77_FILE = "banking77_train.csv"
BANKING77_SOURCE: Mapping[str, str] = {
    "name": "BANKING77 train split (PolyAI)",
    "url": BANKING77_URL,
    "sha256": BANKING77_SHA256,
    "licence": "CC-BY-4.0",
    "citation": "Casanueva et al. 2020, arXiv:2003.04807",
}
_MAX_DOWNLOAD_BYTES = 10_000_000
BANKING20_QUESTION = "Which intent best describes this customer message?"


@dataclass(frozen=True)
class Task:
    """A benchmark task: one Choice question and fixed, seeded splits.

    ``pool`` is the unlabelled prior pool, ``calib_*`` the labelled calibration/training
    split (for L1/L2) and ``test_*`` the evaluation split. Labels index ``question.options``.
    """

    name: str
    question: Choice
    seed: int
    pool: tuple[str, ...]
    calib_states: tuple[str, ...]
    calib_labels: tuple[int, ...]
    test_states: tuple[str, ...]
    test_labels: tuple[int, ...]
    source: Mapping[str, str]


def default_cache_dir() -> Path:
    """``$BRIER_CACHE_DIR`` or ``~/.cache/brier``."""
    return Path(os.environ.get("BRIER_CACHE_DIR", Path.home() / ".cache" / "brier"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - pinned https URL
        data: bytes = resp.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(data) > _MAX_DOWNLOAD_BYTES:
        raise BrierError("download exceeds the size limit")
    return data


def fetch_banking77(cache_dir: str | Path | None = None) -> bytes:
    """BANKING77 train CSV bytes, from the cache or downloaded, verified by SHA-256.

    Raises
    ------
    BrierError
        If the downloaded file does not match the pinned checksum (nothing is cached).
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    path = cache / BANKING77_FILE
    if path.is_file():
        data = path.read_bytes()
        if _sha256(data) == BANKING77_SHA256:
            return data
    data = _download(BANKING77_URL)
    if _sha256(data) != BANKING77_SHA256:
        raise BrierError("BANKING77 download does not match the pinned SHA-256")
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as tmp:
        tmp.write(data)
    os.replace(tmp.name, path)
    return data


def parse_csv(data: bytes) -> list[tuple[str, str]]:
    """``(text, intent)`` rows of a BANKING77 CSV (header skipped)."""
    reader = csv.reader(
        io.StringIO(data.decode("utf-8")),
        quotechar='"',
        delimiter=",",
        quoting=csv.QUOTE_ALL,
        skipinitialspace=True,
    )
    next(reader, None)
    return [(row[0], row[1]) for row in reader if len(row) >= 2]


def make_banking20(
    rows: Sequence[tuple[str, str]],
    *,
    seed: int = 0,
    top_k: int = 20,
    n_pool: int = 500,
    n_cal: int = 300,
    source: Mapping[str, str] = BANKING77_SOURCE,
) -> Task:
    """Build banking20: the ``top_k`` most frequent intents split into pool / calib / test.

    Ties in frequency are broken by intent name; options are sorted by name and shown
    with ``_`` replaced by a space.

    Raises
    ------
    InsufficientDataError
        If there are not more than ``n_pool + n_cal`` items.
    """
    counts = Counter(intent for _, intent in rows)
    names = sorted(sorted(counts, key=lambda k: (-counts[k], k))[:top_k])
    index = {name: i for i, name in enumerate(names)}
    items = [(text, index[intent]) for text, intent in rows if intent in index]
    if len(items) <= n_pool + n_cal:
        raise InsufficientDataError(f"{len(items)} items, need more than {n_pool + n_cal}")
    shuffled = [items[i] for i in np.random.default_rng(seed).permutation(len(items))]
    cal = shuffled[n_pool : n_pool + n_cal]
    test = shuffled[n_pool + n_cal :]
    question = Choice(BANKING20_QUESTION, [n.replace("_", " ") for n in names], name="intent")
    return Task(
        name="banking20",
        question=question,
        seed=seed,
        pool=tuple(text for text, _ in shuffled[:n_pool]),
        calib_states=tuple(text for text, _ in cal),
        calib_labels=tuple(label for _, label in cal),
        test_states=tuple(text for text, _ in test),
        test_labels=tuple(label for _, label in test),
        source=dict(source),
    )


TASKS = ("banking20",)


def load_task(name: str, seed: int = 0, cache_dir: str | Path | None = None) -> Task:
    """Load a benchmark task by id (downloads the dataset on first use)."""
    if name != "banking20":
        raise BrierError(f"unknown task {name!r}; available: {TASKS}")
    return make_banking20(parse_csv(fetch_banking77(cache_dir)), seed=seed)
