import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from declib import Choice, Noul, Score
from declib.backends.base import Backend
from declib.backends.fake import FakeBackend
from declib.errors import DeclibError, TokenizationError
from declib.prompts import labels, render

ROUTE = Choice("Which team?", ["billing", "technical", "sales"], name="route")


def _lp(b: FakeBackend, prompt: str, labs: tuple[str, ...]) -> np.ndarray:
    return b.label_logprobs([prompt], b.label_token_ids(labs))[0]


def test_satisfies_backend_protocol() -> None:
    b: Backend = FakeBackend()
    assert b.model_id == "fake"
    assert b.revision is None
    assert b.num_layers == 8


def test_label_token_ids_distinct_and_known() -> None:
    b = FakeBackend()
    ids = b.label_token_ids(["A", "B", "Yes", "No", "1", "9"])
    assert len(set(ids)) == 6


@pytest.mark.parametrize("labs", [["A", "A"], ["A", "zzz"], ["10"]], ids=["dup", "unknown", "10"])
def test_label_token_ids_strict(labs: list[str]) -> None:
    with pytest.raises(TokenizationError):
        FakeBackend().label_token_ids(labs)


def test_logprobs_shape_dtype_and_full_vocab_normalisation() -> None:
    b = FakeBackend()
    prompts = [render("s1", ROUTE), render("s2", ROUTE)]
    out = b.label_logprobs(prompts, b.label_token_ids(labels(ROUTE)))
    assert out.shape == (2, 3)
    assert out.dtype == np.float64
    assert np.all(out < 0)
    # gathered from a full-vocab log-softmax, so the label mass is below 1
    assert np.all(np.exp(out).sum(axis=1) < 1)


def test_deterministic_across_instances() -> None:
    p = render("same state", ROUTE)
    labs = labels(ROUTE)
    np.testing.assert_array_equal(_lp(FakeBackend(), p, labs), _lp(FakeBackend(), p, labs))
    assert not np.array_equal(_lp(FakeBackend(seed=1), p, labs), _lp(FakeBackend(), p, labs))


def test_content_follows_option_not_position() -> None:
    # Without bias, rotating the options moves each option's logit with it.
    b = FakeBackend()
    labs = labels(ROUTE)
    lp0 = _lp(b, render("s", ROUTE), labs)
    lp1 = _lp(b, render("s", ROUTE, shift=1), labs)  # shows technical, sales, billing
    np.testing.assert_allclose(lp1, np.roll(lp0, -1))


def test_position_bias_raises_logit_at_that_position() -> None:
    labs = labels(ROUTE)
    p = render("s", ROUTE)
    base = _lp(FakeBackend(), p, labs)
    biased = _lp(FakeBackend(position_bias=(2.0, 0.0, 0.0)), p, labs)
    diff = biased - base
    # logit differences between labels change by exactly the bias
    assert (diff[0] - diff[1]) == pytest.approx(2.0)
    assert (diff[1] - diff[2]) == pytest.approx(0.0)


def test_label_prior_applies_per_label() -> None:
    q = Noul("Refund?", name="r")
    p = render("s", q)
    base = _lp(FakeBackend(), p, ("Yes", "No"))
    biased = _lp(FakeBackend(label_prior={"Yes": 1.5}), p, ("Yes", "No"))
    assert (biased[0] - biased[1]) - (base[0] - base[1]) == pytest.approx(1.5)


def test_custom_content_function() -> None:
    b = FakeBackend(content=lambda state, option: 5.0 if option == "sales" else 0.0)
    lp = _lp(b, render("s", ROUTE), labels(ROUTE))
    assert int(np.argmax(lp)) == 2


def test_score_uses_label_text_or_level() -> None:
    seen: list[tuple[str, str]] = []

    def content(state: str, option: str) -> float:
        seen.append((state, option))
        return 0.0

    b = FakeBackend(content=content)
    q = Score("Urgent?", levels=3, name="u", labels=["low", "mid", "high"])
    _lp(b, render("st", q), labels(q))
    assert seen == [("st", "low"), ("st", "mid"), ("st", "high")]
    seen.clear()
    _lp(b, render("st", Noul("q?", name="n")), ("Yes", "No"))
    assert seen == [("st", "Yes"), ("st", "No")]


def test_state_is_unescaped_before_content() -> None:
    seen: list[str] = []
    b = FakeBackend(content=lambda state, option: seen.append(state) or 0.0)
    _lp(b, render("a < b & c", ROUTE), labels(ROUTE))
    assert set(seen) == {"a < b & c"}


def test_hidden_states_shape_dtype_deterministic() -> None:
    b = FakeBackend(hidden_size=4)
    p = [render("s1", ROUTE), render("s2", ROUTE)]
    h = b.hidden_states(p, [1, 5])
    assert h.shape == (2, 2, 4)
    assert h.dtype == np.float32
    np.testing.assert_array_equal(h, FakeBackend(hidden_size=4).hidden_states(p, [1, 5]))
    assert not np.array_equal(h[0], h[1])


@pytest.mark.parametrize("layers", [[-1], [8], []])
def test_hidden_states_rejects_bad_layers(layers: list[int]) -> None:
    with pytest.raises(DeclibError):
        FakeBackend().hidden_states(["p"], layers)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_rejects_non_finite_config(bad: float) -> None:
    with pytest.raises(DeclibError):
        FakeBackend(position_bias=(bad,))
    with pytest.raises(DeclibError):
        FakeBackend(label_prior={"A": bad})


@given(st.text(max_size=50))
def test_logprobs_finite_for_any_state(state: str) -> None:
    lp = _lp(FakeBackend(), render(state, ROUTE), labels(ROUTE))
    assert np.all(np.isfinite(lp))
