"""`brier check`: conformance report for a backend (M6.1)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest

from brier import Choice, Noul, Score
from brier.__main__ import main
from brier.backends.fake import FakeBackend
from brier.check import SANITY, CheckReport, check_backend
from brier.errors import TokenizationError


def _gold_text(question: Choice | Noul | Score, gold: object) -> str:
    if isinstance(question, Choice):
        return str(gold)
    if isinstance(question, Noul):
        return "Yes" if gold else "No"
    assert question.labels is not None
    return question.labels[int(gold) - 1]  # type: ignore[call-overload]


def _knowing_backend() -> FakeBackend:
    """A fake model that knows every sanity answer."""
    gold = {state: _gold_text(q, y) for q, items in SANITY for state, y in items}

    def content(state: str, option: str) -> float:
        return 5.0 if gold.get(state) == option else 0.0

    return FakeBackend(content=content, revision="abc123")


def _status(report: CheckReport, name: str) -> str:
    return next(i.status for i in report.items if i.name == name)


@dataclass(frozen=True)
class _NoLetters(FakeBackend):
    def label_token_ids(self, labels: Sequence[str]) -> list[int]:
        if "A" in labels:
            raise TokenizationError("'A' is two tokens")
        return super().label_token_ids(labels)


@dataclass(frozen=True)
class _NoHidden(FakeBackend):
    def hidden_states(
        self, prompts: Sequence[str], layers: Sequence[int]
    ) -> npt.NDArray[np.float32]:
        raise RuntimeError("no hidden states")


@dataclass(frozen=True)
class _BatchDependent(FakeBackend):
    """Results change with batch size, like a padding bug."""

    def label_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> npt.NDArray[np.float64]:
        out = super().label_logprobs(prompts, token_ids)
        if len(prompts) > 1:
            out[:, 0] += 2.0
        return out


@dataclass(frozen=True)
class _NaN(FakeBackend):
    def label_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> npt.NDArray[np.float64]:
        out = super().label_logprobs(prompts, token_ids)
        out[0, 0] = np.nan
        return out


def test_sanity_set_is_well_formed() -> None:
    kinds = {type(q) for q, _ in SANITY}
    assert kinds == {Choice, Noul, Score}
    for q, items in SANITY:
        assert len(items) >= 4
        for _, gold in items:
            if isinstance(q, Choice):
                assert gold in q.options
            elif isinstance(q, Noul):
                assert isinstance(gold, bool)
            else:
                assert 1 <= int(gold) <= q.levels  # type: ignore[call-overload]


def test_good_backend_passes_everything() -> None:
    report = check_backend(_knowing_backend())
    assert report.ok
    assert all(i.status == "pass" for i in report.items), report.items
    assert report.levels == {"raw": True, "L0": True, "L1": True, "L2": True}
    assert report.sanity["raw_accuracy"] == 1.0
    assert report.sanity["L0_accuracy"] == 1.0
    assert report.sanity["L0_flip_rate"] == 0.0
    assert report.sanity["ms_per_prompt"] >= 0.0


def test_score_letter_fallback_is_reported_not_failed() -> None:
    report = check_backend(_knowing_backend())
    detail = next(i.detail for i in report.items if i.name == "labels.score")
    assert "letter" in detail  # FakeBackend has no single-token "10", like real tokenizers


def test_unpinned_revision_warns() -> None:
    report = check_backend(FakeBackend())
    assert _status(report, "revision") == "warn"
    assert report.ok  # a warning is not a failure


def test_weak_model_warns_on_sanity() -> None:
    report = check_backend(FakeBackend(content=lambda s, o: 0.0, revision="x"))
    assert _status(report, "sanity") == "warn"
    assert report.ok


def test_choice_labels_failing_fails_choice_levels() -> None:
    report = check_backend(_NoLetters(revision="x"))
    assert _status(report, "labels.choice") == "fail"
    assert _status(report, "labels.noul") == "pass"
    assert not report.ok
    assert report.levels["raw"] is False


def test_missing_hidden_states_disables_l2_only() -> None:
    report = check_backend(_NoHidden(revision="x"))
    assert _status(report, "hidden_states") == "fail"
    assert report.levels == {"raw": True, "L0": True, "L1": True, "L2": False}
    assert "no hidden states" in next(i.detail for i in report.items if i.name == "hidden_states")


def test_batch_dependent_results_fail_consistency() -> None:
    report = check_backend(_BatchDependent(revision="x"))
    assert _status(report, "batch_consistency") == "fail"
    assert not report.ok


def test_non_finite_logprobs_fail() -> None:
    report = check_backend(_NaN(revision="x"))
    assert _status(report, "batch_consistency") == "fail"


def test_report_to_dict_is_json_serialisable() -> None:
    d = check_backend(_knowing_backend()).to_dict()
    again = json.loads(json.dumps(d))
    assert again["model"] == {"id": "fake", "revision": "abc123"}
    assert again["ok"] is True
    assert {i["name"] for i in again["checks"]} >= {"labels.choice", "hidden_states", "sanity"}


def test_cli_prints_report_and_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import brier.__main__ as cli

    monkeypatch.setattr(cli, "_make_backend", lambda args: _knowing_backend())
    assert main(["check", "some/model"]) == 0
    out = capsys.readouterr().out
    assert "labels.choice" in out
    assert "PASS" in out
    assert "Supported levels: raw, L0, L1, L2" in out

    monkeypatch.setattr(cli, "_make_backend", lambda args: _NoLetters(revision="x"))
    assert main(["check", "some/model"]) == 1


def test_cli_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import brier.__main__ as cli

    monkeypatch.setattr(cli, "_make_backend", lambda args: _knowing_backend())
    assert main(["check", "some/model", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_reports_load_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import brier.__main__ as cli
    from brier.errors import BrierError

    def boom(args: object) -> FakeBackend:
        raise BrierError("x/y has no chat template")

    monkeypatch.setattr(cli, "_make_backend", boom)
    assert main(["check", "x/y"]) == 1
    assert "no chat template" in capsys.readouterr().out


def test_cli_passes_backend_options(monkeypatch: pytest.MonkeyPatch) -> None:
    import brier.__main__ as cli

    seen = {}

    def fake_hf(model_id: str, **kwargs: object) -> FakeBackend:
        seen.update(kwargs, model_id=model_id)
        return _knowing_backend()

    monkeypatch.setattr("brier.backends.hf.HFBackend", fake_hf)
    args = ["check", "m/x", "--revision", "abc", "--dtype", "float32", "--device", "cpu"]
    args += ["--batch-size", "4", "--attn-implementation", "eager"]
    assert cli.main(args) == 0
    assert seen == {
        "model_id": "m/x",
        "revision": "abc",
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 4,
        "attn_implementation": "eager",
    }


@dataclass(frozen=True)
class _WrongShape(FakeBackend):
    def hidden_states(
        self, prompts: Sequence[str], layers: Sequence[int]
    ) -> npt.NDArray[np.float32]:
        return np.zeros((1, 1, 4), dtype=np.float32)  # one layer instead of all requested


def test_wrong_hidden_state_shape_fails() -> None:
    report = check_backend(_WrongShape(revision="x"))
    assert _status(report, "hidden_states") == "fail"
    assert report.levels["L2"] is False


def test_cli_json_load_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import brier.__main__ as cli

    def boom(args: object) -> FakeBackend:
        raise OSError("not found")

    monkeypatch.setattr(cli, "_make_backend", boom)
    assert main(["check", "x/y", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert "not found" in out["load_error"]


def test_python_m_brier_runs() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "brier", "check", "--help"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert "--revision" in out.stdout


@dataclass(frozen=True)
class _NoLayers(FakeBackend):
    num_layers: int = 0  # default_layers raises on a model without blocks


def test_unusable_layer_count_is_reported_not_raised() -> None:
    report = check_backend(_NoLayers(revision="x"))
    assert _status(report, "hidden_states") == "fail"


def test_cli_survives_unencodable_error_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    import brier.__main__ as cli

    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding="cp1252")  # strict, like a Windows console
    monkeypatch.setattr(sys, "stdout", console)

    def boom(args: object) -> FakeBackend:
        raise TokenizationError("label \u0120A is two tokens")

    monkeypatch.setattr(cli, "_make_backend", boom)
    assert main(["check", "x/y"]) == 1
    console.flush()
    assert b"\\u0120A is two tokens" in raw.getvalue()


def test_cli_works_when_streams_cannot_be_reconfigured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io
    import sys

    import brier.__main__ as cli

    out = io.StringIO()  # has no reconfigure(), like some embedded consoles
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(cli, "_make_backend", lambda args: _knowing_backend())
    assert main(["check", "some/model"]) == 0
    assert "Result: OK" in out.getvalue()
