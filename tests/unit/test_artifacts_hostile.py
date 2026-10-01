"""Crafted, hostile artifacts (security review of M4.2): every one must raise ArtifactError."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from brier import Choice, Noul
from brier.artifacts import (
    ARRAYS_FILE,
    JSON_FILE,
    Artifact,
    Calibration,
    load_artifact,
    save_artifact,
)
from brier.errors import ArtifactError

ROUTE = Choice("Which team?", ["billing", "technical", "sales"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
MODEL, REV = "m", "r"


def _save(tmp_path: Path) -> Path:
    d = tmp_path / "a"
    save_artifact(
        d,
        Artifact(MODEL, REV, 1.0, (Calibration(ROUTE, np.array([0.5, 0.3, 0.2]), 2.0),)),
    )
    return d


def _load(d: Path, **kw):  # type: ignore[no-untyped-def]
    return load_artifact(d, model_id=MODEL, revision=REV, **kw)


def _set_npz(d: Path, data: bytes) -> None:
    (d / ARRAYS_FILE).write_bytes(data)
    j = json.loads((d / JSON_FILE).read_text(encoding="utf-8"))
    j["arrays_sha256"] = hashlib.sha256(data).hexdigest()
    (d / JSON_FILE).write_text(json.dumps(j), encoding="utf-8")


def _set_json_text(d: Path, text: str) -> None:
    (d / JSON_FILE).write_text(text, encoding="utf-8")


def _npy(header: dict, body: bytes) -> bytes:  # type: ignore[type-arg]
    f = io.BytesIO()
    np.lib.format.write_array_header_1_0(f, header)
    return f.getvalue() + body


def _zip(members: list[tuple[str, bytes]], method: int = zipfile.ZIP_STORED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=method) as z:
        for name, data in members:
            z.writestr(name, data)
    return buf.getvalue()


def _patch_zip(data: bytes, local_off: int, central_off: int, value: int) -> bytes:
    """Set a 2-byte little-endian field in every local and central zip header."""
    b = bytearray(data)
    for sig, off in ((b"PK\x03\x04", local_off), (b"PK\x01\x02", central_off)):
        start = 0
        while (i := b.find(sig, start)) != -1:
            b[i + off : i + off + 2] = value.to_bytes(2, "little")
            start = i + 4
    return bytes(b)


GOOD = _npy(
    {"descr": "<f8", "fortran_order": False, "shape": (3,)}, np.array([0.5, 0.3, 0.2]).tobytes()
)


# ---------- npz ----------


def test_huge_declared_shape_is_rejected_without_allocating(tmp_path: Path) -> None:
    d = _save(tmp_path)
    bomb = _npy({"descr": "<f8", "fortran_order": False, "shape": (10**12,)}, b"")
    _set_npz(d, _zip([("prior_0.npy", bomb)]))
    with pytest.raises(ArtifactError, match="length 3"):
        _load(d)


@pytest.mark.parametrize(
    "header",
    [
        {"descr": ">f8", "fortran_order": False, "shape": (3,)},  # big-endian
        {"descr": "<f4", "fortran_order": False, "shape": (3,)},
        {"descr": "|O", "fortran_order": False, "shape": (3,)},  # object (pickle)
        {"descr": "<f8", "fortran_order": True, "shape": (3,)},
        {"descr": "<f8", "fortran_order": False, "shape": (3, 1)},
    ],
)
def test_bad_npy_headers(tmp_path: Path, header: dict) -> None:  # type: ignore[type-arg]
    d = _save(tmp_path)
    _set_npz(d, _zip([("prior_0.npy", _npy(header, b"\x00" * 24))]))
    with pytest.raises(ArtifactError):
        _load(d)


def test_truncated_and_padded_bodies(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_npz(d, _zip([("prior_0.npy", GOOD[:-8])]))
    with pytest.raises(ArtifactError):
        _load(d)
    _set_npz(d, _zip([("prior_0.npy", GOOD + b"\x00" * 8)]))
    with pytest.raises(ArtifactError):
        _load(d)


def test_bad_npy_magic(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_npz(d, _zip([("prior_0.npy", b"not an npy file at all")]))
    with pytest.raises(ArtifactError):
        _load(d)


def test_duplicate_zip_members(tmp_path: Path) -> None:
    d = _save(tmp_path)
    with pytest.warns(UserWarning, match="Duplicate name"):
        data = _zip([("prior_0.npy", GOOD), ("prior_0.npy", GOOD)])
    _set_npz(d, data)
    with pytest.raises(ArtifactError, match="exactly"):
        _load(d)


def test_oversized_member(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_npz(d, _zip([("prior_0.npy", GOOD + b" " * 5000)]))
    with pytest.raises(ArtifactError, match="too large"):
        _load(d)


def test_encrypted_member(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_npz(d, _patch_zip(_zip([("prior_0.npy", GOOD)]), 6, 8, 0x1))
    with pytest.raises(ArtifactError, match="encrypted"):
        _load(d)


def test_unknown_compression_method(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_npz(d, _patch_zip(_zip([("prior_0.npy", GOOD)]), 8, 10, 99))
    with pytest.raises(ArtifactError, match="compression"):
        _load(d)


def test_corrupt_deflate_stream(tmp_path: Path) -> None:
    d = _save(tmp_path)
    data = bytearray(_zip([("prior_0.npy", GOOD)], method=zipfile.ZIP_DEFLATED))
    start = data.find(b"PK\x03\x04") + 30 + len("prior_0.npy")
    for i in range(start, start + 20):
        data[i] ^= 0xFF
    _set_npz(d, bytes(data))
    with pytest.raises(ArtifactError):
        _load(d)


def test_crc_mismatch(tmp_path: Path) -> None:
    d = _save(tmp_path)
    data = bytearray(_zip([("prior_0.npy", GOOD)]))
    data[data.find(GOOD) + len(GOOD) - 1] ^= 0xFF  # flip a stored data byte, CRC unchanged
    _set_npz(d, bytes(data))
    with pytest.raises(ArtifactError):
        _load(d)


# ---------- JSON ----------


def test_deeply_nested_json(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_json_text(d, "[" * 200_000 + "]" * 200_000)
    with pytest.raises(ArtifactError):
        _load(d)


def test_huge_integer_literal(tmp_path: Path) -> None:
    d = _save(tmp_path)
    _set_json_text(d, '{"schema_version": ' + "9" * 5000 + "}")
    with pytest.raises(ArtifactError):
        _load(d)


def _edit(d: Path, fn) -> None:  # type: ignore[no-untyped-def]
    j = json.loads((d / JSON_FILE).read_text(encoding="utf-8"))
    fn(j)
    (d / JSON_FILE).write_text(json.dumps(j), encoding="utf-8")


@pytest.mark.parametrize(
    "edit",
    [
        lambda j: j["questions"][0].update(temperature=10**400),  # OverflowError in float()
        lambda j: j["questions"][0].update(temperature=1e-320),  # below the allowed range
        lambda j: j["questions"][0].update(temperature=1e300),
        lambda j: j["questions"][0].update(type=["choice"]),  # unhashable type
        lambda j: j["questions"].__setitem__(0, "not an object"),
        lambda j: j.update(schema_version=1.0),
        lambda j: j.update(model=["m", "r"]),
    ],
    ids=[
        "temp-overflow",
        "temp-tiny",
        "temp-huge",
        "type-list",
        "entry-str",
        "schema-float",
        "model-list",
    ],
)
def test_hostile_json_values(tmp_path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    d = _save(tmp_path)
    _edit(d, edit)
    with pytest.raises(ArtifactError):
        _load(d)


# ---------- values and arguments ----------


def test_prior_with_zero_rejected() -> None:
    with pytest.raises(ArtifactError):
        Calibration(REFUND, prior=np.array([1.0, 0.0]))


@pytest.mark.parametrize("max_bytes", [None, "100", 0, -1, True])
def test_max_bytes_validated(tmp_path: Path, max_bytes: object) -> None:
    d = _save(tmp_path)
    with pytest.raises(ArtifactError):
        _load(d, max_bytes=max_bytes)


def test_unexpected_internal_error_is_wrapped(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from brier import artifacts

    d = _save(tmp_path)

    def boom(*a, **k):  # type: ignore[no-untyped-def]
        raise MemoryError("simulated")

    monkeypatch.setattr(artifacts, "_read_priors", boom)
    with pytest.raises(ArtifactError, match="MemoryError"):
        _load(d)


def test_save_does_not_overwrite_a_file_created_concurrently(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    d = tmp_path / "race"
    d.mkdir()
    real_mkdir = Path.mkdir

    def mkdir_then_race(self, *a, **k):  # type: ignore[no-untyped-def]
        real_mkdir(self, *a, **k)
        (self / ARRAYS_FILE).write_bytes(b"someone else's file")

    monkeypatch.setattr(Path, "mkdir", mkdir_then_race)
    with pytest.raises(ArtifactError):
        save_artifact(d, Artifact(MODEL, REV, 1.0, (Calibration(REFUND, temperature=1.0),)))
    assert (d / ARRAYS_FILE).read_bytes() == b"someone else's file"


def test_artifact_requires_calibration_instances() -> None:
    with pytest.raises(ArtifactError):
        Artifact(MODEL, REV, 1.0, ("not a calibration",))  # type: ignore[arg-type]


def test_non_regular_file_rejected(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import stat
    from types import SimpleNamespace

    from brier import artifacts

    d = _save(tmp_path)
    monkeypatch.setattr(
        artifacts.os, "fstat", lambda fd: SimpleNamespace(st_mode=stat.S_IFIFO, st_size=0)
    )
    with pytest.raises(ArtifactError, match="regular"):
        _load(d)


def test_npy_format_2_loads(tmp_path: Path) -> None:
    d = _save(tmp_path)
    f = io.BytesIO()
    np.lib.format.write_array_header_2_0(f, {"descr": "<f8", "fortran_order": False, "shape": (3,)})
    _set_npz(d, _zip([("prior_0.npy", f.getvalue() + np.array([0.5, 0.3, 0.2]).tobytes())]))
    np.testing.assert_array_equal(_load(d).calibrations[0].prior, [0.5, 0.3, 0.2])


def test_npy_unknown_format_version(tmp_path: Path) -> None:
    d = _save(tmp_path)
    raw = bytearray(GOOD)
    raw[6] = 3  # major version byte after the 6-byte magic string
    _set_npz(d, _zip([("prior_0.npy", bytes(raw))]))
    with pytest.raises(ArtifactError):
        _load(d)


def _set_u32(data: bytes, local_off: int, central_off: int, value: int) -> bytes:
    b = bytearray(data)
    for sig, off in ((b"PK\x03\x04", local_off), (b"PK\x01\x02", central_off)):
        i = b.find(sig)
        b[i + off : i + off + 4] = value.to_bytes(4, "little")
    return bytes(b)


def test_deflate_bomb_with_lying_size_is_read_bounded(tmp_path: Path) -> None:
    import tracemalloc

    d = _save(tmp_path)
    buf = io.BytesIO()
    with (
        zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z,
        z.open("prior_0.npy", "w", force_zip64=False) as m,
    ):
        chunk = bytes(1 << 20)
        for _ in range(200):  # 200 MB of zeros, ~200 KB compressed
            m.write(chunk)
    # Claim the member is only 100 bytes (uncompressed size: local +22, central +24).
    bomb = _set_u32(buf.getvalue(), 22, 24, 100)
    assert len(bomb) < 1_000_000
    _set_npz(d, bomb)
    tracemalloc.start()
    try:
        with pytest.raises(ArtifactError):
            _load(d)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 20_000_000  # the unbounded read allocated hundreds of MB
