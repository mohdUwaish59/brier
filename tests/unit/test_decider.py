import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from brier import Choice, Decider, Decision, Noul, Score
from brier.backends.fake import FakeBackend
from brier.errors import (
    BrierError,
    InputTooLargeError,
    InsufficientDataError,
    NotFittedError,
    QuestionError,
)

ROUTE = Choice("Which team?", ["billing", "technical", "sales", "other"], name="route")
REFUND = Noul("Is this a refund request?", name="refund")
URGENCY = Score("How urgent?", levels=5, name="urgency")
QS = [ROUTE, REFUND, URGENCY]
POOL = [f"customer message {i}" for i in range(100)]


def test_decide_returns_decision_per_question() -> None:
    res = Decider(FakeBackend()).decide("My card was charged twice", QS)
    assert set(res) == {"route", "refund", "urgency"}
    assert all(isinstance(d, Decision) for d in res.values())
    assert res["route"].type == "choice"
    assert list(res["route"].probs) == list(ROUTE.options)
    assert res["route"].answer in ROUTE.options
    assert list(res["refund"].probs) == ["yes", "no"]
    assert isinstance(res["refund"].answer, bool)
    assert list(res["urgency"].probs) == ["1", "2", "3", "4", "5"]
    assert 1.0 <= res["urgency"].expected <= 5.0
    assert all(d.level == "L0" for d in res.values())


def test_meta_records_provenance() -> None:
    res = Decider(FakeBackend()).decide("s", QS, level="L0")
    assert res["route"].meta["model_id"] == "fake"
    assert res["route"].meta["revision"] is None
    assert res["route"].meta["n_forward"] == 4  # one per rotation
    assert res["refund"].meta["n_forward"] == 1
    assert res["route"].meta["prior_applied"] is False
    raw = Decider(FakeBackend()).decide("s", QS, level="raw")
    assert raw["route"].meta["n_forward"] == 1
    assert "prior_applied" not in raw["route"].meta


def test_decide_equals_first_row_of_decide_batch() -> None:
    d = Decider(FakeBackend(position_bias=(1.0,)))
    one = d.decide("s1", QS)
    batch = d.decide_batch(["s1", "s2"], QS)
    assert len(batch) == 2
    for name in one:
        assert one[name].probs == batch[0][name].probs


def test_decide_batch_empty_states() -> None:
    assert Decider(FakeBackend()).decide_batch([], QS) == []


def test_l0_removes_position_bias_raw_does_not() -> None:
    b = FakeBackend(
        position_bias=(3.0, 0.0, 0.0, 0.0),
        content=lambda state, option: 1.0 if option == "sales" else 0.0,
    )
    d = Decider(b)
    assert d.decide("s", [ROUTE], level="raw")["route"].answer == "billing"
    assert d.decide("s", [ROUTE], level="L0")["route"].answer == "sales"


def test_fit_prior_applies_to_l0_only() -> None:
    d = Decider(FakeBackend(label_prior={"Yes": 2.0}))
    before = np.mean([d.decide(s, [REFUND])["refund"].p_yes for s in POOL])
    d.fit_prior(POOL, [REFUND])
    after = [d.decide(s, [REFUND])["refund"] for s in POOL]
    assert all(x.meta["prior_applied"] is True for x in after)
    assert abs(np.mean([x.p_yes for x in after]) - 0.5) < abs(before - 0.5) / 2
    raw = d.decide(POOL[0], [REFUND], level="raw")["refund"]
    assert raw.p_yes == pytest.approx(
        Decider(FakeBackend(label_prior={"Yes": 2.0}))
        .decide(POOL[0], [REFUND], "raw")["refund"]
        .p_yes
    )


def test_prior_is_keyed_by_whole_question_not_name() -> None:
    d = Decider(FakeBackend())
    d.fit_prior(POOL[:10], [ROUTE])
    other = Choice("Which team?", ["sales", "billing", "technical", "other"], name="route")
    assert d.decide("s", [ROUTE])["route"].meta["prior_applied"] is True
    assert d.decide("s", [other])["route"].meta["prior_applied"] is False


def test_score_prior_off_by_default() -> None:
    d = Decider(FakeBackend())
    d.fit_prior(POOL[:10], [URGENCY])
    assert d.decide("s", [URGENCY])["urgency"].meta["prior_applied"] is False
    d2 = Decider(FakeBackend(), score_prior=True)
    d2.fit_prior(POOL[:10], [URGENCY])
    assert d2.decide("s", [URGENCY])["urgency"].meta["prior_applied"] is True


def test_prior_strength_zero_is_rotation_only() -> None:
    b = FakeBackend(label_prior={"Yes": 2.0})
    d = Decider(b, prior_strength=0.0)
    d.fit_prior(POOL[:10], [REFUND])
    plain = Decider(b).decide("s", [REFUND])["refund"].p_yes
    assert d.decide("s", [REFUND])["refund"].p_yes == pytest.approx(plain)


@pytest.mark.parametrize("lam", [-0.5, 1.5, True])
def test_prior_strength_validated(lam: object) -> None:
    with pytest.raises(BrierError):
        Decider(FakeBackend(), prior_strength=lam)  # type: ignore[arg-type]


def test_fit_prior_needs_states() -> None:
    with pytest.raises(InsufficientDataError):
        Decider(FakeBackend()).fit_prior([], [REFUND])


@pytest.mark.parametrize("level", ["L1", "L2"])
def test_unfitted_levels_raise(level: str) -> None:
    with pytest.raises(NotFittedError):
        Decider(FakeBackend()).decide("s", QS, level=level)  # type: ignore[arg-type]


def test_unknown_level_rejected() -> None:
    with pytest.raises(BrierError):
        Decider(FakeBackend()).decide("s", QS, level="L9")  # type: ignore[arg-type]


def test_question_limits() -> None:
    qs = [Noul(f"q{i}?", name=f"q{i}") for i in range(33)]
    with pytest.raises(InputTooLargeError):
        Decider(FakeBackend()).decide("s", qs)
    Decider(FakeBackend()).decide("s", qs[:32])
    with pytest.raises(InputTooLargeError):
        Decider(FakeBackend(), max_questions=2).decide("s", qs[:3])


def test_batch_limits() -> None:
    with pytest.raises(InputTooLargeError):
        Decider(FakeBackend()).decide_batch(["s"] * 65, [REFUND])
    assert len(Decider(FakeBackend()).decide_batch(["s"] * 64, [REFUND])) == 64
    with pytest.raises(InputTooLargeError):
        Decider(FakeBackend(), max_batch=2).decide_batch(["s"] * 3, [REFUND])


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_limits_validated(bad: object) -> None:
    with pytest.raises(BrierError):
        Decider(FakeBackend(), max_questions=bad)  # type: ignore[arg-type]
    with pytest.raises(BrierError):
        Decider(FakeBackend(), max_batch=bad)  # type: ignore[arg-type]


def test_question_validation() -> None:
    d = Decider(FakeBackend())
    with pytest.raises(QuestionError):
        d.decide("s", [])
    with pytest.raises(QuestionError):
        d.decide("s", [Noul("a?", name="x"), Noul("b?", name="x")])
    with pytest.raises(QuestionError):
        d.decide("s", ["not a question"])  # type: ignore[list-item]
    with pytest.raises(QuestionError):
        d.decide(None, [REFUND])  # type: ignore[arg-type]
    with pytest.raises(QuestionError):
        d.decide("s", None)  # type: ignore[arg-type]
    with pytest.raises(QuestionError):
        d.decide("s", "refund")  # type: ignore[arg-type]
    with pytest.raises(QuestionError):
        d.decide_batch("not a list of states", [REFUND])  # type: ignore[arg-type]


def test_deterministic() -> None:
    a = Decider(FakeBackend()).decide("s", QS)
    b = Decider(FakeBackend()).decide("s", QS)
    assert {k: v.probs for k, v in a.items()} == {k: v.probs for k, v in b.items()}


@settings(max_examples=25, deadline=None)
@given(st.text(max_size=40), st.sampled_from(["raw", "L0"]))
def test_every_decision_is_a_distribution(state: str, level: str) -> None:
    for dec in Decider(FakeBackend(position_bias=(0.7, -0.3))).decide(state, QS, level).values():  # type: ignore[arg-type]
        assert sum(dec.probs.values()) == pytest.approx(1.0, abs=1e-9)
