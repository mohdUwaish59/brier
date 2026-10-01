import os
from pathlib import Path

import pytest

from brier import Choice, Decider, Noul, Score
from brier.artifacts import ARRAYS_FILE, JSON_FILE
from brier.backends.fake import FakeBackend
from brier.errors import ArtifactError, NotFittedError

ROUTE = Choice("Which team?", ["billing", "technical", "sales", "other"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
URGENCY = Score("How urgent?", levels=5, name="urgency")
STATES = [f"customer message {i}" for i in range(60)]
PROBE = [f"new message {i}" for i in range(7)]


def _backend(**kw) -> FakeBackend:  # type: ignore[no-untyped-def]
    return FakeBackend(position_bias=(1.0, 0.5), label_prior={"Yes": 1.0}, **kw)


def _fitted() -> Decider:
    d = Decider(_backend(), prior_strength=0.7, score_prior=True)
    d.fit_prior(STATES, [ROUTE, REFUND, URGENCY])
    d.fit_temperature(STATES, ROUTE, [ROUTE.options[i % 4] for i in range(60)])
    d.fit_temperature(STATES, REFUND, [i % 3 == 0 for i in range(60)])
    d.fit_temperature(STATES, URGENCY, [i % 5 + 1 for i in range(60)])
    return d


def _probs(d: Decider, qs: list, level: str) -> list[dict]:  # type: ignore[type-arg]
    return [
        {name: (dict(dec.probs), dict(dec.meta)) for name, dec in res.items()}
        for res in d.decide_batch(PROBE, qs, level)  # type: ignore[arg-type]
    ]


@pytest.mark.parametrize("level", ["raw", "L0", "L1"])
def test_round_trip_gives_identical_decisions(tmp_path: Path, level: str) -> None:
    d = _fitted()
    d.save(tmp_path / "calib")
    loaded = Decider.load(tmp_path / "calib", _backend())
    qs = [ROUTE, REFUND, URGENCY]
    assert _probs(loaded, qs, level) == _probs(d, qs, level)  # exact, not approximate


def test_load_restores_settings(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    loaded = Decider.load(tmp_path / "c", _backend(), max_questions=5, max_batch=7)
    assert loaded.prior_strength == 0.7
    assert loaded.max_questions == 5
    assert loaded.max_batch == 7


def test_save_writes_only_calibration_files(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    assert sorted(os.listdir(tmp_path / "c")) == sorted([JSON_FILE, ARRAYS_FILE])


def test_partial_calibration_round_trips(tmp_path: Path) -> None:
    d = Decider(_backend())
    d.fit_prior(STATES, [ROUTE])  # prior only
    d.fit_temperature(STATES, REFUND, [i % 2 == 0 for i in range(60)])  # temperature only
    d.save(tmp_path / "c")
    loaded = Decider.load(tmp_path / "c", _backend())
    assert _probs(loaded, [ROUTE], "L0") == _probs(d, [ROUTE], "L0")
    assert _probs(loaded, [REFUND], "L1") == _probs(d, [REFUND], "L1")
    with pytest.raises(NotFittedError):
        loaded.decide("s", [ROUTE], level="L1")  # ROUTE never had a temperature


def test_empty_decider_round_trips(tmp_path: Path) -> None:
    Decider(_backend()).save(tmp_path / "c")
    loaded = Decider.load(tmp_path / "c", _backend())
    assert _probs(loaded, [ROUTE], "L0") == _probs(Decider(_backend()), [ROUTE], "L0")


def test_load_refuses_other_model_or_revision(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    with pytest.raises(ArtifactError, match="model"):
        Decider.load(tmp_path / "c", _backend(model_id="other-model"))
    with pytest.raises(ArtifactError, match="revision"):
        Decider.load(tmp_path / "c", _backend(revision="abc123"))


def test_save_refuses_non_empty_directory(tmp_path: Path) -> None:
    d = _fitted()
    d.save(tmp_path / "c")
    with pytest.raises(ArtifactError):
        d.save(tmp_path / "c")


def test_load_respects_max_bytes(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    with pytest.raises(ArtifactError, match="size"):
        Decider.load(tmp_path / "c", _backend(), max_bytes=50)


def test_loaded_decider_can_be_refitted(tmp_path: Path) -> None:
    _fitted().save(tmp_path / "c")
    loaded = Decider.load(tmp_path / "c", _backend())
    loaded.fit_prior(STATES[:10], [REFUND])  # refitting the prior discards its temperature
    with pytest.raises(NotFittedError):
        loaded.decide("s", [REFUND], level="L1")
