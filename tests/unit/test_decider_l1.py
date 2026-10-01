import hashlib

import numpy as np
import pytest

from brier import Choice, Decider, Noul, Score
from brier.backends.fake import FakeBackend
from brier.errors import InsufficientDataError, NotFittedError, QuestionError

ROUTE = Choice("Which team?", ["billing", "technical", "sales", "other"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
URGENCY = Score("How urgent?", levels=5, name="urgency")
STATES = [f"customer message {i}" for i in range(120)]


def _unit(*parts: object) -> float:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0**64


def _overconfident() -> FakeBackend:
    # Large-scale random content: L0 is very sure of itself.
    return FakeBackend(content=lambda state, option: 12.0 * _unit(state, option))


def _noisy_labels(d: Decider, q: Choice, states: list[str]) -> list[str]:
    """Labels that agree with L0's answer only ~60 % of the time."""
    labels = []
    for i, res in enumerate(d.decide_batch(states[:64], [q]) + d.decide_batch(states[64:], [q])):
        top = res[q.name].answer
        other = [o for o in q.options if o != top][i % (len(q.options) - 1)]
        labels.append(top if _unit("label", i) < 0.6 else other)
    return labels


def _nll(decisions: list[dict], name: str, labels: list[str]) -> float:  # type: ignore[type-arg]
    return float(
        -np.mean([np.log(d[name].probs[y]) for d, y in zip(decisions, labels, strict=True)])
    )


def _batch(d: Decider, q: Choice, level: str) -> list[dict]:  # type: ignore[type-arg]
    return d.decide_batch(STATES[:64], [q], level) + d.decide_batch(STATES[64:], [q], level)  # type: ignore[arg-type]


def test_l1_requires_fit() -> None:
    with pytest.raises(NotFittedError):
        Decider(FakeBackend()).decide("s", [ROUTE], level="L1")


def test_fit_temperature_softens_overconfident_l0_and_improves_nll() -> None:
    d = Decider(_overconfident())
    labels = _noisy_labels(d, ROUTE, STATES)
    d.fit_temperature(STATES, ROUTE, labels)
    l0, l1 = _batch(d, ROUTE, "L0"), _batch(d, ROUTE, "L1")
    t = l1[0]["route"].meta["temperature"]
    assert t > 1.0
    assert all(a["route"].answer == b["route"].answer for a, b in zip(l0, l1, strict=True))
    assert np.mean([r["route"].confidence for r in l1]) < np.mean(
        [r["route"].confidence for r in l0]
    )
    assert _nll(l1, "route", labels) < _nll(l0, "route", labels)
    assert l1[0]["route"].level == "L1"
    assert l1[0]["route"].meta["n_forward"] == 4
    assert l1[0]["route"].meta["prior_applied"] is False


def test_label_types_per_question() -> None:
    d = Decider(FakeBackend())
    d.fit_temperature(STATES[:50], REFUND, [i % 3 == 0 for i in range(50)])
    d.fit_temperature(STATES[:50], URGENCY, [i % 5 + 1 for i in range(50)])
    res = d.decide("s", [REFUND, URGENCY], level="L1")
    assert res["refund"].level == res["urgency"].level == "L1"
    assert res["refund"].meta["temperature"] > 0


@pytest.mark.parametrize(
    ("question", "labels"),
    [
        (ROUTE, ["refunds"] * 50),  # not an option
        (ROUTE, [0] * 50),  # index instead of option string
        (REFUND, [1] * 50),  # int instead of bool
        (REFUND, ["yes"] * 50),  # string instead of bool
        (URGENCY, [6] * 50),  # level out of range
        (URGENCY, [0] * 50),
        (URGENCY, [True] * 50),  # bool is not a level
        (URGENCY, ["3"] * 50),
    ],
)
def test_bad_labels_rejected(question, labels) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(QuestionError):
        Decider(FakeBackend()).fit_temperature(STATES[:50], question, labels)


def test_label_count_must_match_states() -> None:
    with pytest.raises(QuestionError):
        Decider(FakeBackend()).fit_temperature(STATES[:50], REFUND, [True] * 49)


def test_needs_fifty_labels() -> None:
    with pytest.raises(InsufficientDataError):
        Decider(FakeBackend()).fit_temperature(STATES[:49], REFUND, [True, False] * 24 + [True])


def test_question_must_be_a_question() -> None:
    with pytest.raises(QuestionError):
        Decider(FakeBackend()).fit_temperature(STATES[:50], "refund", [True] * 50)  # type: ignore[arg-type]


def test_temperature_keyed_by_whole_question() -> None:
    d = Decider(FakeBackend())
    d.fit_temperature(STATES[:50], ROUTE, [ROUTE.options[i % 4] for i in range(50)])
    d.decide("s", [ROUTE], level="L1")
    reordered = Choice(ROUTE.text, ["sales", "billing", "technical", "other"], name="route")
    with pytest.raises(NotFittedError):
        d.decide("s", [reordered], level="L1")


def test_all_requested_questions_need_a_temperature() -> None:
    d = Decider(FakeBackend())
    d.fit_temperature(STATES[:50], REFUND, [i % 2 == 0 for i in range(50)])
    with pytest.raises(NotFittedError):
        d.decide("s", [REFUND, ROUTE], level="L1")


def test_temperature_fitted_on_top_of_prior() -> None:
    d = Decider(FakeBackend(label_prior={"Yes": 2.0}))
    d.fit_prior(STATES, [REFUND])
    d.fit_temperature(STATES[:50], REFUND, [i % 2 == 0 for i in range(50)])
    res = d.decide("s", [REFUND], level="L1")["refund"]
    assert res.meta["prior_applied"] is True


def test_refitting_prior_discards_stale_temperature() -> None:
    d = Decider(FakeBackend())
    d.fit_temperature(STATES[:50], REFUND, [i % 2 == 0 for i in range(50)])
    d.decide("s", [REFUND], level="L1")
    d.fit_prior(STATES, [REFUND])
    with pytest.raises(NotFittedError):
        d.decide("s", [REFUND], level="L1")


def test_l2_still_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        Decider(FakeBackend()).decide("s", [REFUND], level="L2")
