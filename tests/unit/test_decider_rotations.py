"""ADR-0009: Decider(rotations=m) and artifact schema version 4."""

import json
from pathlib import Path

import numpy as np
import pytest

from brier import Choice, Decider, Noul
from brier.artifacts import JSON_FILE, SCHEMA_VERSION
from brier.backends.fake import FakeBackend
from brier.debias import evenly_spaced, l0_logprobs
from brier.errors import ArtifactError, BrierError

ROUTE = Choice("Which team?", ["a", "b", "c", "d", "e", "f"], name="route")
SMALL = Choice("Pick one", ["x", "y", "z"], name="small")
REFUND = Noul("Is this a refund?", name="refund")
STATES = [f"message {i}" for i in range(60)]
BIASED = FakeBackend(position_bias=(2.0, 1.0, 0.5, 0.0, -0.5, -1.0))


def _probs(d: Decider, q: Choice, level: str = "L0") -> np.ndarray:
    res = d.decide_batch(STATES[:10], [q], level=level)
    return np.array([[r[q.name].probs[o] for o in q.options] for r in res])


def test_evenly_spaced_lives_in_the_core() -> None:
    assert evenly_spaced(6, 2) == [0, 3]
    assert evenly_spaced(20, 4) == [0, 5, 10, 15]


def test_default_uses_all_rotations() -> None:
    d = Decider(BIASED)
    assert d.rotations is None
    assert d.decide("s", [ROUTE], level="L0")["route"].meta["n_forward"] == 6


def test_subset_matches_l0_logprobs_with_those_shifts() -> None:
    d = Decider(BIASED, rotations=2)
    want = np.exp(l0_logprobs(BIASED, STATES[:10], ROUTE, shifts=[0, 3]))
    np.testing.assert_allclose(_probs(d, ROUTE), want, atol=1e-12)
    assert d.decide("s", [ROUTE], level="L0")["route"].meta["n_forward"] == 2


def test_rotations_at_or_above_k_use_all_k() -> None:
    full = _probs(Decider(BIASED), SMALL)
    np.testing.assert_allclose(_probs(Decider(BIASED, rotations=3), SMALL), full)
    np.testing.assert_allclose(_probs(Decider(BIASED, rotations=10), SMALL), full)
    assert Decider(BIASED, rotations=10).decide("s", [SMALL], "L0")["small"].meta["n_forward"] == 3


def test_noul_and_raw_are_unaffected() -> None:
    a, b = Decider(BIASED), Decider(BIASED, rotations=1)
    assert (
        a.decide("s", [REFUND], "L0")["refund"].probs
        == b.decide("s", [REFUND], "L0")["refund"].probs
    )
    assert (
        a.decide("s", [ROUTE], "raw")["route"].probs == b.decide("s", [ROUTE], "raw")["route"].probs
    )


def test_prior_and_temperature_are_fitted_with_the_same_rotations() -> None:
    d = Decider(BIASED, rotations=2)
    d.fit_prior(STATES, [ROUTE])
    from brier.debias import fit_prior

    want = fit_prior(l0_logprobs(BIASED, STATES, ROUTE, shifts=[0, 3]))
    np.testing.assert_allclose(d._priors[ROUTE], want)
    d.fit_temperature(STATES, ROUTE, [ROUTE.options[i % 6] for i in range(60)])
    assert d.decide("s", [ROUTE], "L1")["route"].meta["n_forward"] == 2


@pytest.mark.parametrize("bad", [0, -1, 27, 2.0, True, "2"])
def test_invalid_rotations_rejected(bad: object) -> None:
    with pytest.raises(BrierError, match="rotations"):
        Decider(BIASED, rotations=bad)  # type: ignore[arg-type]


def test_rotations_round_trip_through_artifacts(tmp_path: Path) -> None:
    d = Decider(BIASED, rotations=2)
    d.fit_prior(STATES, [ROUTE])
    d.save(tmp_path / "c")
    doc = json.loads((tmp_path / "c" / JSON_FILE).read_text(encoding="utf-8"))
    assert doc["schema_version"] == SCHEMA_VERSION == 4
    assert doc["rotations"] == 2
    loaded = Decider.load(tmp_path / "c", BIASED)
    assert loaded.rotations == 2
    np.testing.assert_array_equal(_probs(loaded, ROUTE), _probs(d, ROUTE))


def test_default_rotations_saved_as_null(tmp_path: Path) -> None:
    Decider(BIASED).save(tmp_path / "c")
    doc = json.loads((tmp_path / "c" / JSON_FILE).read_text(encoding="utf-8"))
    assert doc["rotations"] is None
    assert Decider.load(tmp_path / "c", BIASED).rotations is None


def test_version_3_artifact_loads_with_all_rotations(tmp_path: Path) -> None:
    d = Decider(BIASED)
    d.fit_prior(STATES, [ROUTE])
    d.save(tmp_path / "c")
    p = tmp_path / "c" / JSON_FILE
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["schema_version"] = 3
    del doc["rotations"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert Decider.load(tmp_path / "c", BIASED).rotations is None


@pytest.mark.parametrize("bad", [0, 27, 2.5, 2.0, True, "2", [2]])
def test_hostile_rotations_in_artifact_rejected(tmp_path: Path, bad: object) -> None:
    Decider(BIASED).save(tmp_path / "c")
    p = tmp_path / "c" / JSON_FILE
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["rotations"] = bad
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ArtifactError):
        Decider.load(tmp_path / "c", BIASED)


def test_version_4_requires_rotations_key(tmp_path: Path) -> None:
    Decider(BIASED).save(tmp_path / "c")
    p = tmp_path / "c" / JSON_FILE
    doc = json.loads(p.read_text(encoding="utf-8"))
    del doc["rotations"]
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ArtifactError):
        Decider.load(tmp_path / "c", BIASED)


def test_artifact_rejects_invalid_rotations() -> None:
    from brier.artifacts import Artifact

    with pytest.raises(ArtifactError, match="rotations"):
        Artifact("m", None, 1.0, (), rotations=0)


@pytest.mark.parametrize("doc", ["[]", '{"brier_version": "0.1.1"}'])
def test_artifact_json_without_schema_version_rejected(tmp_path: Path, doc: str) -> None:
    Decider(BIASED).save(tmp_path / "c")
    (tmp_path / "c" / JSON_FILE).write_text(doc, encoding="utf-8")
    with pytest.raises(ArtifactError):
        Decider.load(tmp_path / "c", BIASED)


def test_version_3_artifact_with_rotations_key_rejected(tmp_path: Path) -> None:
    Decider(BIASED, rotations=2).save(tmp_path / "c")
    p = tmp_path / "c" / JSON_FILE
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["schema_version"] = 3  # keeps "rotations": v3 files never had it
    p.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ArtifactError):
        Decider.load(tmp_path / "c", BIASED)


def test_rotations_is_read_only() -> None:
    d = Decider(BIASED, rotations=2)
    with pytest.raises(AttributeError):
        d.rotations = 0  # type: ignore[misc]
