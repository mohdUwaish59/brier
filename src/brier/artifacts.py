"""Calibration artifacts: a directory with ``artifact.json`` + ``arrays.npz`` (ADR-0003).

Loading treats the files as untrusted (THREAT_MODEL T2): fixed file names (no paths from
the JSON); each file opened once, must be a regular non-symlink file, read with a size cap;
strict JSON schema (unknown keys rejected); SHA-256 of the npz verified; npz members
checked (count, names, compression, encryption, size) and each ``.npy`` header parsed by
hand so its shape, dtype and order are validated *before* any allocation (no pickle, no
``np.load``); model id, revision and prompt-template hash must match. Every failure,
including unexpected ones, raises :class:`ArtifactError`.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from brier import __version__, prompts
from brier._math import FloatArray
from brier.errors import ArtifactError, QuestionError
from brier.questions import Choice, Noul, Question, Score

SCHEMA_VERSION = 1
JSON_FILE = "artifact.json"
ARRAYS_FILE = "arrays.npz"
DEFAULT_MAX_BYTES = 100_000_000
TEMPERATURE_RANGE = (1e-6, 1e6)
_SUM_TOL = 1e-6
_MAX_MEMBER_BYTES = 4096  # a prior has at most 26 float64s plus a .npy header
_F8 = np.dtype("<f8")
_NPY: Any = np.lib.format  # untyped in older numpy stubs (Python 3.10 resolves numpy 2.2)
_TOP_KEYS = {
    "schema_version",
    "brier_version",
    "model",
    "template_hash",
    "prior_strength",
    "arrays_sha256",
    "questions",
}
_QUESTION_KEYS = {
    "choice": {"type", "name", "text", "options", "prior", "temperature"},
    "noul": {"type", "name", "text", "prior", "temperature"},
    "score": {"type", "name", "text", "levels", "labels", "prior", "temperature"},
}


def _n_answers(q: Question) -> int:
    if isinstance(q, Choice):
        return len(q.options)
    return 2 if isinstance(q, Noul) else q.levels


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


@dataclass(frozen=True, eq=False)
class Calibration:
    """Fitted calibration of one question.

    Parameters
    ----------
    question : Choice, Noul or Score
        The question (calibration is keyed by the whole question).
    prior : array_like or None
        L0 prior: a strictly positive distribution over the question's answers.
    temperature : float or None
        L1 temperature in ``[1e-6, 1e6]``.
    """

    question: Question
    prior: FloatArray | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question, (Choice, Noul, Score)):
            raise ArtifactError("question must be a Choice, Noul or Score")
        name = self.question.name
        if self.prior is not None:
            p = np.array(self.prior, dtype=np.float64)
            if p.shape != (_n_answers(self.question),):
                raise ArtifactError(f"prior for {name!r} has the wrong shape")
            if not np.all(np.isfinite(p)) or np.any(p <= 0) or abs(p.sum() - 1.0) > _SUM_TOL:
                raise ArtifactError(f"prior for {name!r} is not a positive distribution")
            p.flags.writeable = False
            object.__setattr__(self, "prior", p)
        if self.temperature is not None:
            try:
                t = float(self.temperature) if _is_number(self.temperature) else math.nan
            except OverflowError:
                t = math.nan
            lo, hi = TEMPERATURE_RANGE
            if not lo <= t <= hi:  # NaN fails too
                raise ArtifactError(f"temperature for {name!r} must be a number in [{lo}, {hi}]")
            object.__setattr__(self, "temperature", t)


@dataclass(frozen=True, eq=False)
class Artifact:
    """Everything needed to reproduce calibrated decisions with the same model.

    Parameters
    ----------
    model_id, revision : str, str or None
        The model the calibration was fitted on.
    prior_strength : float
        λ used when applying L0 priors.
    calibrations : tuple of Calibration
        One entry per distinct question.
    """

    model_id: str
    revision: str | None
    prior_strength: float
    calibrations: tuple[Calibration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ArtifactError("model_id must be a non-empty string")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ArtifactError("revision must be a string or None")
        lam = self.prior_strength
        if not _is_number(lam) or not 0 <= lam <= 1:
            raise ArtifactError("prior_strength must be a number in [0, 1]")
        object.__setattr__(self, "calibrations", tuple(self.calibrations))
        if not all(isinstance(c, Calibration) for c in self.calibrations):
            raise ArtifactError("calibrations must be Calibration instances")
        questions = [c.question for c in self.calibrations]
        if len(set(questions)) != len(questions):
            raise ArtifactError("each question may appear only once")


def save_artifact(path: str | Path, artifact: Artifact) -> None:
    """Write ``artifact`` to the directory ``path`` (created; must be empty if it exists).

    Files are created exclusively, so nothing that appears concurrently is overwritten.

    Raises
    ------
    ArtifactError
        If ``path`` is a file or a non-empty directory.
    """
    d = Path(path)
    if d.exists() and (not d.is_dir() or any(d.iterdir())):
        raise ArtifactError(f"{d} exists and is not an empty directory")
    arrays: dict[str, Any] = {
        f"prior_{i}": c.prior for i, c in enumerate(artifact.calibrations) if c.prior is not None
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    doc = {
        "schema_version": SCHEMA_VERSION,
        "brier_version": __version__,
        "model": {"id": artifact.model_id, "revision": artifact.revision},
        "template_hash": prompts.template_hash(),
        "prior_strength": float(artifact.prior_strength),
        "arrays_sha256": hashlib.sha256(data).hexdigest(),
        "questions": [_question_to_json(c) for c in artifact.calibrations],
    }
    d.mkdir(parents=True, exist_ok=True)
    try:
        with (d / ARRAYS_FILE).open("xb") as f:
            f.write(data)
        with (d / JSON_FILE).open("x", encoding="utf-8") as f:
            f.write(json.dumps(doc, indent=2))
    except FileExistsError as e:
        raise ArtifactError(f"{d} is no longer empty") from e


def load_artifact(
    path: str | Path,
    *,
    model_id: str,
    revision: str | None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Artifact:
    """Load and fully validate an artifact for the given model.

    Parameters
    ----------
    path : str or Path
        Artifact directory.
    model_id, revision : str, str or None
        The backend's model; the artifact must have been fitted on exactly this.
    max_bytes : int
        Cap on each file and on the uncompressed npz contents (default 100 MB).

    Raises
    ------
    ArtifactError
        On any problem with the files, including unexpected parser errors.
    """
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ArtifactError("max_bytes must be a positive int")
    try:
        return _load(Path(path), model_id, revision, max_bytes)
    except ArtifactError:
        raise
    except Exception as e:  # fail closed: hostile input must never escape as another type
        raise ArtifactError(f"artifact could not be loaded safely ({type(e).__name__})") from e


def _load(d: Path, model_id: str, revision: str | None, max_bytes: int) -> Artifact:
    if not d.is_dir():
        raise ArtifactError(f"{d} is not a directory")
    raw_json = _read_capped(d / JSON_FILE, max_bytes)
    raw_arrays = _read_capped(d / ARRAYS_FILE, max_bytes)
    try:
        doc = json.loads(raw_json.decode("utf-8"))
    except (ValueError, RecursionError) as e:  # covers UnicodeDecodeError, JSONDecodeError
        raise ArtifactError(f"{JSON_FILE} is not valid JSON") from e
    _check_header(doc, model_id, revision)
    if hashlib.sha256(raw_arrays).hexdigest() != doc["arrays_sha256"]:
        raise ArtifactError(f"{ARRAYS_FILE} does not match its recorded SHA-256")
    entries = doc["questions"]
    if not isinstance(entries, list):
        raise ArtifactError("questions must be a list")
    for entry in entries:
        _check_question_json(entry)
    try:
        questions = [_question_from_json(entry) for entry in entries]
    except QuestionError as err:
        raise ArtifactError(f"invalid question in artifact: {err}") from err
    sizes = {
        i: _n_answers(q)
        for i, (q, e) in enumerate(zip(questions, entries, strict=True))
        if e["prior"]
    }
    priors = _read_priors(raw_arrays, sizes, max_bytes)
    calibrations = tuple(
        Calibration(q, priors.get(i), e["temperature"])
        for i, (q, e) in enumerate(zip(questions, entries, strict=True))
    )
    model = doc["model"]
    return Artifact(model["id"], model["revision"], doc["prior_strength"], calibrations)


def _read_capped(p: Path, max_bytes: int) -> bytes:
    """Open once, require a regular non-symlink file under the cap, read only its size."""
    if p.is_symlink():
        raise ArtifactError(f"{p.name} is a symlink")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(p, flags)
    except OSError as e:
        raise ArtifactError(f"{p.name} is missing or unreadable") from e
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ArtifactError(f"{p.name} is not a regular file")
        if st.st_size > max_bytes:
            raise ArtifactError(f"{p.name} exceeds the size cap of {max_bytes} bytes")
        # Read only what fstat reported (+1): read(n) pre-allocates n bytes, so never ask
        # for max_bytes up front; an extra byte means the file grew after fstat.
        data = f.read(st.st_size + 1)
    if len(data) > st.st_size:
        raise ArtifactError(f"{p.name} changed size while being read")
    return data


def _check_header(doc: Any, model_id: str, revision: str | None) -> None:
    if not isinstance(doc, dict) or set(doc) != _TOP_KEYS:
        raise ArtifactError(f"{JSON_FILE} must have exactly the keys {sorted(_TOP_KEYS)}")
    version = doc["schema_version"]
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ArtifactError(f"unsupported schema_version {version!r} (expected {SCHEMA_VERSION})")
    model = doc["model"]
    if (
        not isinstance(model, dict)
        or set(model) != {"id", "revision"}
        or not isinstance(model["id"], str)
        or not (model["revision"] is None or isinstance(model["revision"], str))
    ):
        raise ArtifactError("model must be {id: str, revision: str or null}")
    for key in ("brier_version", "template_hash", "arrays_sha256"):
        if not isinstance(doc[key], str):
            raise ArtifactError(f"{key} must be a string")
    lam = doc["prior_strength"]
    if not _is_number(lam) or not 0 <= lam <= 1:
        raise ArtifactError("prior_strength must be a number in [0, 1]")
    if model["id"] != model_id:
        raise ArtifactError(f"artifact was fitted on model {model['id']!r}, not {model_id!r}")
    if model["revision"] != revision:
        raise ArtifactError(
            f"artifact was fitted on revision {model['revision']!r}, not {revision!r}"
        )
    if doc["template_hash"] != prompts.template_hash():
        raise ArtifactError("prompt template mismatch: the artifact was fitted with other prompts")


def _check_question_json(e: Any) -> None:
    if not isinstance(e, dict):
        raise ArtifactError("each question entry must be an object")
    kind = e.get("type")
    if not isinstance(kind, str) or kind not in _QUESTION_KEYS:
        raise ArtifactError("each question needs a type of choice, noul or score")
    if set(e) != _QUESTION_KEYS[kind]:
        raise ArtifactError(f"{kind} entry must have exactly {sorted(_QUESTION_KEYS[kind])}")
    if not isinstance(e["prior"], bool):
        raise ArtifactError("prior must be true or false")
    if e["temperature"] is not None and not _is_number(e["temperature"]):
        raise ArtifactError("temperature must be a number or null")
    if kind == "choice" and not isinstance(e["options"], list):
        raise ArtifactError("options must be a list")
    if kind == "score" and e["labels"] is not None and not isinstance(e["labels"], list):
        raise ArtifactError("labels must be a list or null")


def _question_from_json(e: dict[str, Any]) -> Question:
    if e["type"] == "choice":
        return Choice(e["text"], e["options"], name=e["name"])
    if e["type"] == "noul":
        return Noul(e["text"], name=e["name"])
    return Score(e["text"], e["levels"], name=e["name"], labels=e["labels"])


def _question_to_json(c: Calibration) -> dict[str, Any]:
    q = c.question
    entry: dict[str, Any] = {"name": q.name, "text": q.text}
    if isinstance(q, Choice):
        entry.update(type="choice", options=list(q.options))
    elif isinstance(q, Noul):
        entry.update(type="noul")
    else:
        entry.update(type="score", levels=q.levels, labels=list(q.labels) if q.labels else None)
    entry.update(prior=c.prior is not None, temperature=c.temperature)
    return entry


def _read_priors(data: bytes, sizes: dict[int, int], max_bytes: int) -> dict[int, FloatArray]:
    """Read ``prior_<i>`` (length ``sizes[i]``) from the npz without trusting its headers."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArtifactError(f"{ARRAYS_FILE} is not a valid npz file") from e
    with z:
        infos = z.infolist()
        expected = {f"prior_{i}.npy": i for i in sizes}
        names = [info.filename for info in infos]
        if len(names) != len(expected) or set(names) != set(expected):
            raise ArtifactError(f"{ARRAYS_FILE} must contain exactly the referenced priors")
        if sum(info.file_size for info in infos) > max_bytes:
            raise ArtifactError(f"{ARRAYS_FILE} contents exceed the size cap of {max_bytes} bytes")
        out: dict[int, FloatArray] = {}
        for info in infos:
            if info.flag_bits & 0x1:
                raise ArtifactError(f"{info.filename} is encrypted")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ArtifactError(f"{info.filename} uses an unsupported compression method")
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ArtifactError(f"{info.filename} is too large for a prior")
            i = expected[info.filename]
            with z.open(info) as member:  # bounded: never trust the declared file_size
                raw = member.read(_MAX_MEMBER_BYTES + 1)
            if len(raw) > _MAX_MEMBER_BYTES:  # pragma: no cover - zipfile stops at file_size
                raise ArtifactError(f"{info.filename} is too large for a prior")
            out[i] = _parse_npy(raw, sizes[i], info.filename)
    return out


def _parse_npy(raw: bytes, n: int, name: str) -> FloatArray:
    """Validate a ``.npy`` header (``<f8``, C order, shape ``(n,)``), then read ``n`` floats."""
    f = io.BytesIO(raw)
    version = _NPY.read_magic(f)
    if version == (1, 0):
        shape, fortran_order, dtype = _NPY.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran_order, dtype = _NPY.read_array_header_2_0(f)
    else:
        raise ArtifactError(f"{name} has unsupported .npy version {version}")
    if dtype != _F8 or fortran_order or shape != (n,):
        raise ArtifactError(f"{name} must be a little-endian float64 vector of length {n}")
    body = f.read()
    if len(body) != n * _F8.itemsize:
        raise ArtifactError(f"{name} has {len(body)} data bytes, expected {n * _F8.itemsize}")
    return np.frombuffer(body, dtype=_F8).astype(np.float64)
