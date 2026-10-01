import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brier import Decision
from brier.errors import BrierError


def test_choice_decision_answer_and_confidence() -> None:
    d = Decision("route", "choice", {"billing": 0.7, "technical": 0.2, "sales": 0.1}, "L0")
    assert d.answer == "billing"
    assert d.confidence == pytest.approx(0.7)
    assert d.level == "L0"
    assert d.meta == {}


def test_noul_decision_exposes_p_yes_and_bool_answer() -> None:
    d = Decision("refund", "noul", {"yes": 0.3, "no": 0.7}, "raw")
    assert d.p_yes == pytest.approx(0.3)
    assert d.answer is False
    assert Decision("r", "noul", {"yes": 0.9, "no": 0.1}, "raw").answer is True


def test_score_decision_expected_and_mode() -> None:
    # expected = 1*0.1 + 2*0.2 + 3*0.7 = 2.6; mode = 3
    d = Decision("urgency", "score", {"1": 0.1, "2": 0.2, "3": 0.7}, "L1")
    assert d.expected == pytest.approx(2.6)
    assert d.answer == 3


def test_ties_resolve_to_first_listed() -> None:
    assert Decision("c", "choice", {"a": 0.5, "b": 0.5}, "raw").answer == "a"
    assert Decision("s", "score", {"1": 0.5, "2": 0.5}, "raw").answer == 1


def test_p_yes_only_on_noul() -> None:
    d = Decision("c", "choice", {"a": 1.0, "b": 0.0}, "raw")
    with pytest.raises(BrierError):
        _ = d.p_yes


def test_expected_only_on_score() -> None:
    d = Decision("c", "choice", {"a": 1.0, "b": 0.0}, "raw")
    with pytest.raises(BrierError):
        _ = d.expected


@pytest.mark.parametrize(
    "probs",
    [
        {"a": 0.6, "b": 0.6},
        {"a": math.nan, "b": 0.5},
        {"a": math.inf, "b": 0.0},
        {"a": 1.5, "b": -0.5},
        {},
    ],
    ids=["sum-not-one", "nan", "inf", "negative", "empty"],
)
def test_invalid_probs_rejected(probs: dict[str, float]) -> None:
    with pytest.raises(BrierError):
        Decision("c", "choice", probs, "raw")


def test_invalid_level_rejected() -> None:
    with pytest.raises(BrierError):
        Decision("c", "choice", {"a": 1.0}, "L9")  # type: ignore[arg-type]


def test_invalid_type_rejected() -> None:
    with pytest.raises(BrierError):
        Decision("c", "multi", {"a": 1.0}, "raw")  # type: ignore[arg-type]


def test_decision_is_frozen_and_probs_copied() -> None:
    probs = {"a": 0.4, "b": 0.6}
    d = Decision("c", "choice", probs, "raw", meta={"model_id": "m"})
    probs["a"] = 0.9
    assert d.probs["a"] == pytest.approx(0.4)
    with pytest.raises(AttributeError):
        d.level = "L0"  # type: ignore[misc]
    with pytest.raises(TypeError):
        d.probs["a"] = 1.0  # type: ignore[index]


@given(
    st.lists(st.floats(min_value=1e-6, max_value=1.0), min_size=2, max_size=26).map(
        lambda w: [x / sum(w) for x in w]
    )
)
def test_confidence_is_max_prob_and_answer_is_argmax(ps: list[float]) -> None:
    probs = {f"o{i}": p for i, p in enumerate(ps)}
    d = Decision("c", "choice", probs, "raw")
    assert d.confidence == max(ps)
    assert d.probs[d.answer] == max(ps)
