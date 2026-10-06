"""Standard logging (M7.1): useful records, no handlers configured, never any state text."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from brier import Choice, Decider, Noul
from brier.backends.fake import FakeBackend

ROUTE = Choice("Which team?", ["billing", "technical", "sales"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
MARKER = "zq-private-state-marker"  # must never reach a log record
STATES = [f"{MARKER} customer message {i}" for i in range(75)]
ROUTE_LABELS = [ROUTE.options[i % 3] for i in range(75)]


def _texts(records: list[logging.LogRecord]) -> str:
    return "\n".join(f"{r.name} {r.getMessage()} {r.args!r}" for r in records)


def _fitted(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> Decider:
    caplog.set_level(logging.DEBUG, logger="brier")
    d = Decider(FakeBackend(hidden_size=16, num_layers=8))
    d.fit_prior(STATES, [ROUTE, REFUND])
    d.fit_temperature(STATES, ROUTE, ROUTE_LABELS)
    d.fit_head(STATES, ROUTE, ROUTE_LABELS)
    d.decide_batch(STATES[:10], [ROUTE], level="L2")
    d.decide_batch(STATES[:10], [ROUTE, REFUND], level="L0")
    d.decide_batch(STATES[:10], [ROUTE], level="L1")
    d.save(tmp_path / "c")
    Decider.load(tmp_path / "c", FakeBackend(hidden_size=16, num_layers=8))
    return d


def test_library_installs_only_a_null_handler() -> None:
    handlers = logging.getLogger("brier").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    assert not any(type(h) is not logging.NullHandler for h in handlers)


def test_fits_and_artifacts_are_logged_at_info(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _fitted(tmp_path, caplog)
    info = [r for r in caplog.records if r.levelno == logging.INFO]
    text = _texts(info)
    assert "fit_prior" in text
    assert "'route'" in text
    assert "route, refund on 75 states" in text  # fit_prior lists the fitted questions
    assert "75 states" in text
    assert "fit_temperature" in text
    assert "T=" in text
    assert "fit_head" in text
    assert "layer=" in text
    assert "solver=" in text
    assert "oof_accuracy=" in text
    assert "saved calibration" in text
    assert "loaded calibration" in text
    assert all(r.name.startswith("brier.") for r in info)


def test_decisions_are_not_logged_at_info(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="brier")
    d = Decider(FakeBackend())
    d.decide_batch(STATES[:20], [ROUTE, REFUND], level="L0")
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def test_no_state_text_in_any_record(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _fitted(tmp_path, caplog)
    assert caplog.records  # the flow above did log something
    assert MARKER not in _texts(caplog.records)


def test_benchmark_logs_each_level(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from brier.bench.run import run_task
    from brier.bench.tasks import make_banking20

    caplog.set_level(logging.INFO, logger="brier")
    rows = [(f"{MARKER} intent_{c} message {i}", f"intent_{c}") for c in "abc" for i in range(40)]
    task = make_banking20(rows, top_k=3, n_pool=10, n_cal=60)
    run_task(FakeBackend(), task, ["raw", "L0"], tmp_path / "out", n_resamples=10)
    text = _texts(caplog.records)
    assert "level raw" in text
    assert "level L0" in text
    assert MARKER not in text


def test_unpinned_revision_warning() -> None:
    from brier.backends.hf import _warn_if_unpinned

    with pytest.warns(UserWarning, match="revision"):
        _warn_if_unpinned("some/model", None)
    _warn_if_unpinned("some/model", "abc123")  # pinned: no warning (pytest errors on any)
