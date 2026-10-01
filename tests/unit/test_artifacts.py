import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from brier import Choice, Noul, Score, prompts
from brier.artifacts import (
    ARRAYS_FILE,
    JSON_FILE,
    SCHEMA_VERSION,
    Artifact,
    Calibration,
    load_artifact,
    save_artifact,
)
from brier.errors import ArtifactError

ROUTE = Choice("Which team?", ["billing", "technical", "sales"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
URGENCY = Score("How urgent?", levels=4, name="urgency", labels=["low", "mid", "high", "max"])
MODEL, REV = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"


def _artifact() -> Artifact:
    return Artifact(
        model_id=MODEL,
        revision=REV,
        prior_strength=0.8,
        calibrations=(
            Calibration(ROUTE, prior=np.array([0.5, 0.3, 0.2]), temperature=4.5),
            Calibration(REFUND, prior=np.array([0.7, 0.3]), temperature=None),
            Calibration(URGENCY, prior=None, temperature=1.7),
        ),
    )


def _save(tmp_path: Path) -> Path:
    d = tmp_path / "calib"
    save_artifact(d, _artifact())
    return d


def _load(d: Path, **kw):  # type: ignore[no-untyped-def]
    return load_artifact(d, model_id=MODEL, revision=REV, **kw)


def _edit_json(d: Path, fn) -> None:  # type: ignore[no-untyped-def]
    p = d / JSON_FILE
    data = json.loads(p.read_text(encoding="utf-8"))
    fn(data)
    p.write_text(json.dumps(data), encoding="utf-8")


def _rewrite_npz(d: Path, arrays: dict, *, compress: bool = False) -> None:  # type: ignore[type-arg]
    """Replace arrays.npz and update the recorded checksum (an attacker controls both)."""
    buf = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buf, **arrays)
    (d / ARRAYS_FILE).write_bytes(buf.getvalue())
    _edit_json(d, lambda j: j.update(arrays_sha256=hashlib.sha256(buf.getvalue()).hexdigest()))


# ---------- round trip ----------


def test_round_trip(tmp_path: Path) -> None:
    loaded = _load(_save(tmp_path))
    original = _artifact()
    assert loaded.model_id == MODEL
    assert loaded.revision == REV
    assert loaded.prior_strength == 0.8
    assert [c.question for c in loaded.calibrations] == [c.question for c in original.calibrations]
    for got, want in zip(loaded.calibrations, original.calibrations, strict=True):
        assert got.temperature == want.temperature
        if want.prior is None:
            assert got.prior is None
        else:
            np.testing.assert_array_equal(got.prior, want.prior)


def test_json_records_provenance(tmp_path: Path) -> None:
    d = _save(tmp_path)
    j = json.loads((d / JSON_FILE).read_text(encoding="utf-8"))
    assert j["schema_version"] == SCHEMA_VERSION == 1
    assert j["model"] == {"id": MODEL, "revision": REV}
    assert j["template_hash"] == prompts.template_hash()
    assert j["arrays_sha256"] == hashlib.sha256((d / ARRAYS_FILE).read_bytes()).hexdigest()
    assert isinstance(j["brier_version"], str)
    assert sorted(os.listdir(d)) == sorted([JSON_FILE, ARRAYS_FILE])


def test_unpinned_revision_round_trips(tmp_path: Path) -> None:
    a = Artifact(MODEL, None, 1.0, (Calibration(REFUND, None, 2.0),))
    save_artifact(tmp_path / "a", a)
    assert load_artifact(tmp_path / "a", model_id=MODEL, revision=None).revision is None


# ---------- refuses to load ----------


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"model_id": "other/model", "revision": REV}, "model"),
        ({"model_id": MODEL, "revision": "deadbeef"}, "revision"),
    ],
)
def test_model_mismatch(tmp_path: Path, kw: dict, match: str) -> None:  # type: ignore[type-arg]
    with pytest.raises(ArtifactError, match=match):
        load_artifact(_save(tmp_path), **kw)


def test_template_mismatch(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    d = _save(tmp_path)
    monkeypatch.setattr(prompts, "SYSTEM", "A different system prompt.")
    with pytest.raises(ArtifactError, match="template"):
        _load(d)


def test_tampered_npz_fails_checksum(tmp_path: Path) -> None:
    d = _save(tmp_path)
    data = bytearray((d / ARRAYS_FILE).read_bytes())
    data[-10] ^= 0xFF
    (d / ARRAYS_FILE).write_bytes(bytes(data))
    with pytest.raises(ArtifactError, match="SHA-256"):
        _load(d)


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j.update(schema_version=2),
        lambda j: j.pop("template_hash"),
        lambda j: j.update(unexpected="x"),
        lambda j: j.update(prior_strength=1.5),
        lambda j: j.update(prior_strength="1"),
        lambda j: j["model"].update(id=3),
        lambda j: j.update(questions="nope"),
        lambda j: j["questions"][0].update(type="multi"),
        lambda j: j["questions"][0].update(options=["only one"]),
        lambda j: j["questions"][0].update(temperature=-1.0),
        lambda j: j["questions"][0].update(temperature=float("nan")),
        lambda j: j["questions"][0].update(prior="../../etc/passwd"),
        lambda j: j["questions"][0].update(extra=1),
        lambda j: j["questions"].append(j["questions"][0]),  # duplicate question
    ],
    ids=[
        "schema-version",
        "missing-key",
        "unknown-key",
        "prior-strength-range",
        "prior-strength-type",
        "model-id-type",
        "questions-type",
        "question-type",
        "invalid-question",
        "negative-temperature",
        "nan-temperature",
        "prior-ref-path",
        "unknown-question-key",
        "duplicate-question",
    ],
)
def test_malformed_json_rejected(tmp_path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    d = _save(tmp_path)
    _edit_json(d, edit)
    with pytest.raises(ArtifactError):
        _load(d)


def test_invalid_json_text(tmp_path: Path) -> None:
    d = _save(tmp_path)
    (d / JSON_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactError):
        _load(d)


@pytest.mark.parametrize(
    "arrays",
    [
        {"prior_0": np.array([0.5, 0.5]), "prior_1": np.array([0.7, 0.3])},  # wrong shape
        {"prior_0": np.array([0.5, 0.3, 0.3]), "prior_1": np.array([0.7, 0.3])},  # sum != 1
        {"prior_0": np.array([1.2, -0.1, -0.1]), "prior_1": np.array([0.7, 0.3])},  # negative
        {"prior_0": np.array([np.nan, 0.5, 0.5]), "prior_1": np.array([0.7, 0.3])},
        {"prior_0": np.array([1, 0, 0]), "prior_1": np.array([0.7, 0.3])},  # int dtype
        {"prior_0": np.array([0.5, 0.3, 0.2])},  # missing array
        {  # extra array
            "prior_0": np.array([0.5, 0.3, 0.2]),
            "prior_1": np.array([0.7, 0.3]),
            "payload": np.zeros(3),
        },
    ],
    ids=["shape", "sum", "negative", "nan", "dtype", "missing", "extra"],
)
def test_bad_arrays_rejected(tmp_path: Path, arrays: dict) -> None:  # type: ignore[type-arg]
    d = _save(tmp_path)
    _rewrite_npz(d, arrays)
    with pytest.raises(ArtifactError):
        _load(d)


def test_pickled_object_array_rejected(tmp_path: Path) -> None:
    d = _save(tmp_path)
    obj = np.empty(1, dtype=object)
    obj[0] = {"evil": True}
    _rewrite_npz(d, {"prior_0": obj, "prior_1": np.array([0.7, 0.3])})
    with pytest.raises(ArtifactError):
        _load(d)


def test_size_cap(tmp_path: Path) -> None:
    d = _save(tmp_path)
    with pytest.raises(ArtifactError, match="size"):
        _load(d, max_bytes=100)


def test_zip_bomb_rejected_before_decompression(tmp_path: Path) -> None:
    d = _save(tmp_path)
    # 8 MB of zeros compresses to a few KB: small on disk, large when decompressed.
    _rewrite_npz(
        d,
        {"prior_0": np.zeros(1_000_000), "prior_1": np.array([0.7, 0.3])},
        compress=True,
    )
    assert (d / ARRAYS_FILE).stat().st_size < 100_000
    with pytest.raises(ArtifactError, match="size"):
        _load(d, max_bytes=1_000_000)


def test_not_a_zip(tmp_path: Path) -> None:
    d = _save(tmp_path)
    (d / ARRAYS_FILE).write_bytes(b"not a zip at all")
    _edit_json(d, lambda j: j.update(arrays_sha256=hashlib.sha256(b"not a zip at all").hexdigest()))
    with pytest.raises(ArtifactError):
        _load(d)


def test_missing_files(tmp_path: Path) -> None:
    d = _save(tmp_path)
    (d / ARRAYS_FILE).unlink()
    with pytest.raises(ArtifactError):
        _load(d)
    with pytest.raises(ArtifactError):
        _load(tmp_path / "does-not-exist")


def test_symlinked_file_rejected(tmp_path: Path) -> None:
    d = _save(tmp_path)
    target = tmp_path / "elsewhere.json"
    (d / JSON_FILE).replace(target)
    try:
        (d / JSON_FILE).symlink_to(target)
    except OSError:
        pytest.skip("symlinks not permitted on this system")
    with pytest.raises(ArtifactError, match="symlink"):
        _load(d)


# ---------- saving ----------


def test_save_refuses_non_empty_directory(tmp_path: Path) -> None:
    d = _save(tmp_path)
    with pytest.raises(ArtifactError):
        save_artifact(d, _artifact())
    (tmp_path / "file").write_text("x")
    with pytest.raises(ArtifactError):
        save_artifact(tmp_path / "file", _artifact())


def test_save_into_existing_empty_directory(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    save_artifact(tmp_path / "empty", _artifact())
    assert _load(tmp_path / "empty").model_id == MODEL


def test_calibration_validation() -> None:
    with pytest.raises(ArtifactError):
        Calibration(ROUTE, prior=np.array([0.5, 0.5]), temperature=None)  # wrong length
    with pytest.raises(ArtifactError):
        Calibration(ROUTE, prior=None, temperature=0.0)
    with pytest.raises(ArtifactError):
        Calibration("route", prior=None, temperature=1.0)  # type: ignore[arg-type]


# ---------- template hash ----------


def test_template_hash_is_stable_hex(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    h = prompts.template_hash()
    assert h == prompts.template_hash()
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)
    monkeypatch.setattr(prompts, "ANSWER_PREFIX", "Reply:")
    assert prompts.template_hash() != h


def test_zip_entries_are_plain_npy(tmp_path: Path) -> None:
    d = _save(tmp_path)
    with zipfile.ZipFile(d / ARRAYS_FILE) as z:
        assert all(name.endswith(".npy") for name in z.namelist())


@pytest.mark.parametrize(
    "kw",
    [
        {"model_id": ""},
        {"model_id": 3},
        {"revision": 7},
        {"prior_strength": 1.5},
        {"prior_strength": True},
        {"calibrations": (Calibration(REFUND), Calibration(REFUND, temperature=2.0))},
    ],
    ids=["empty-model", "model-type", "revision-type", "lambda-range", "lambda-bool", "dup"],
)
def test_artifact_validation(kw: dict) -> None:  # type: ignore[type-arg]
    args = {"model_id": MODEL, "revision": REV, "prior_strength": 1.0, "calibrations": ()}
    with pytest.raises(ArtifactError):
        Artifact(**{**args, **kw})


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j.update(template_hash=123),
        lambda j: j["questions"][0].update(temperature="4.5"),
        lambda j: j["questions"][0].update(options="billing"),
        lambda j: j["questions"][2].update(labels="low"),
    ],
    ids=["hash-type", "temperature-type", "options-type", "labels-type"],
)
def test_wrong_json_types_rejected(tmp_path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    d = _save(tmp_path)
    _edit_json(d, edit)
    with pytest.raises(ArtifactError):
        _load(d)


def test_file_growing_after_size_check_is_rejected(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import stat
    from types import SimpleNamespace

    from brier import artifacts

    d = _save(tmp_path)
    (d / JSON_FILE).write_text((d / JSON_FILE).read_text(encoding="utf-8") + " " * 10_000)
    # fstat reports a tiny regular file (as if checked before the file grew)
    lie = SimpleNamespace(st_mode=stat.S_IFREG, st_size=0)
    monkeypatch.setattr(artifacts.os, "fstat", lambda fd: lie)
    with pytest.raises(ArtifactError, match="size"):
        _load(d, max_bytes=5_000)
