import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from declib import Choice, Noul, Score
from declib._math import norm
from declib.backends.fake import FakeBackend
from declib.debias import apply_prior, combine, fit_prior, l0_logprobs, unrotate
from declib.errors import DeclibError, QuestionError
from declib.prompts import render
from declib.readout import raw_logprobs

ROUTE = Choice("Which team?", ["billing", "technical", "sales", "other"], name="route")
STATES = [f"customer message {i}" for i in range(200)]
BIASED = FakeBackend(position_bias=(1.5, 0.5, 0.0, -0.5))


# ---------- raw ----------


def test_raw_is_softmax_over_label_tokens() -> None:
    b = FakeBackend()
    out = raw_logprobs(b, ["s"], ROUTE)
    z = b.label_logprobs([render("s", ROUTE)], b.label_token_ids(["A", "B", "C", "D"]))
    np.testing.assert_allclose(out, norm(z))
    assert out.shape == (1, 4)


def test_raw_rows_sum_to_one_for_every_type() -> None:
    b = FakeBackend()
    for q in [ROUTE, Noul("Refund?", name="r"), Score("Urgent?", levels=5, name="u")]:
        out = raw_logprobs(b, STATES[:5], q)
        np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0, atol=1e-12)


def test_score_falls_back_to_letters_when_digit_not_single_token() -> None:
    # FakeBackend has no "10" token, like real tokenizers.
    seen: list[str] = []
    b = FakeBackend(content=lambda state, option: seen.append(option) or 0.0)
    out = raw_logprobs(b, ["s"], Score("Urgent?", levels=10, name="u"))
    assert out.shape == (1, 10)
    assert seen == [str(i) for i in range(1, 11)]  # lettered options "A. 1" .. "J. 10"


def test_raw_shift_reads_display_positions() -> None:
    b = FakeBackend()
    a = raw_logprobs(b, ["s"], ROUTE, shift=1)
    z = b.label_logprobs([render("s", ROUTE, shift=1)], b.label_token_ids(["A", "B", "C", "D"]))
    np.testing.assert_allclose(a, norm(z))


# ---------- unrotate / combine ----------


def test_unrotate_maps_positions_back_to_options() -> None:
    # shift 1 shows options (1, 2, 3, 0) at positions (0, 1, 2, 3)
    pos = np.log(np.array([[0.1, 0.2, 0.3, 0.4]]))
    opt = np.exp(unrotate(pos, 1))
    # option k sat at position (k - 1) mod 4
    np.testing.assert_allclose(opt, [[0.4, 0.1, 0.2, 0.3]])


def test_combine_is_normalised_geometric_mean() -> None:
    a = np.log(np.array([[0.5, 0.5]]))
    b = np.log(np.array([[0.9, 0.1]]))
    # geometric means: sqrt(.45)=0.6708, sqrt(.05)=0.2236 -> normalised .75, .25
    np.testing.assert_allclose(np.exp(combine([a, b])), [[0.75, 0.25]])


def test_combine_rejects_empty() -> None:
    with pytest.raises(DeclibError):
        combine([])


# ---------- L0 rotations ----------


def _flip_rate(fn: object) -> float:
    rev = Choice(ROUTE.text, list(reversed(ROUTE.options)), name="route")
    a = np.array(ROUTE.options)[np.argmax(fn(BIASED, STATES, ROUTE), axis=1)]  # type: ignore[operator]
    b = np.array(rev.options)[np.argmax(fn(BIASED, STATES, rev), axis=1)]  # type: ignore[operator]
    return float(np.mean(a != b))


def test_acceptance_l0_flip_rate_below_raw() -> None:
    raw_flip = _flip_rate(raw_logprobs)
    l0_flip = _flip_rate(l0_logprobs)
    assert raw_flip > 0.2  # the position bias really flips raw answers
    assert l0_flip < raw_flip
    assert l0_flip == 0.0  # additive bias + all K shifts cancels exactly


@settings(max_examples=30, deadline=None)
@given(st.permutations(ROUTE.options), st.text(max_size=30))
def test_l0_equivariant_to_option_permutation(perm: list[str], state: str) -> None:
    q2 = Choice(ROUTE.text, perm, name="route")
    p1 = dict(zip(ROUTE.options, np.exp(l0_logprobs(BIASED, [state], ROUTE)[0]), strict=True))
    p2 = dict(zip(perm, np.exp(l0_logprobs(BIASED, [state], q2)[0]), strict=True))
    for opt in ROUTE.options:
        assert p1[opt] == pytest.approx(p2[opt], abs=1e-12)


def test_l0_recovers_content_under_position_bias() -> None:
    # Only "sales" has content; a big bias at position A must not win under L0.
    b = FakeBackend(
        position_bias=(3.0, 0.0, 0.0, 0.0),
        content=lambda state, option: 1.0 if option == "sales" else 0.0,
    )
    assert int(np.argmax(raw_logprobs(b, ["s"], ROUTE)[0])) == 0
    assert int(np.argmax(l0_logprobs(b, ["s"], ROUTE)[0])) == 2


def test_l0_subset_of_shifts_and_validation() -> None:
    out = l0_logprobs(BIASED, ["s"], ROUTE, shifts=[0, 2])
    np.testing.assert_allclose(np.exp(out).sum(), 1.0)
    for bad in ([], [0, 0], [4], [-1]):
        with pytest.raises(QuestionError):
            l0_logprobs(BIASED, ["s"], ROUTE, shifts=bad)


def test_l0_does_not_rotate_noul_or_score() -> None:
    b = FakeBackend(position_bias=(1.0, 0.0))
    for q in [Noul("Refund?", name="r"), Score("Urgent?", levels=3, name="u")]:
        np.testing.assert_allclose(l0_logprobs(b, STATES[:3], q), raw_logprobs(b, STATES[:3], q))
        with pytest.raises(QuestionError):
            l0_logprobs(b, ["s"], q, shifts=[0])


# ---------- prior ----------


def test_fit_prior_is_mean_probability_clipped_and_normalised() -> None:
    logp = np.log(np.array([[0.8, 0.2], [0.6, 0.4]]))
    np.testing.assert_allclose(fit_prior(logp), [0.7, 0.3])
    tiny = np.log(np.array([[1.0 - 1e-12, 1e-12]]))
    pi = fit_prior(tiny)
    assert pi[1] == pytest.approx(1e-6, rel=1e-3)
    assert pi.sum() == pytest.approx(1.0)


def test_apply_prior_hand_computed() -> None:
    # q ∝ p / π: (0.6/0.75, 0.4/0.25) = (0.8, 1.6) -> (1/3, 2/3)
    logp = np.log(np.array([[0.6, 0.4]]))
    out = np.exp(apply_prior(logp, np.array([0.75, 0.25])))
    np.testing.assert_allclose(out, [[1 / 3, 2 / 3]])


def test_apply_prior_lambda_zero_and_uniform_prior_are_identity() -> None:
    logp = norm(np.array([[0.3, -1.0, 2.0]]))
    np.testing.assert_allclose(apply_prior(logp, np.array([0.5, 0.3, 0.2]), lam=0.0), logp)
    np.testing.assert_allclose(apply_prior(logp, np.full(3, 1 / 3)), logp, atol=1e-12)


@pytest.mark.parametrize("lam", [-0.1, 1.1, math.nan, True])
def test_apply_prior_rejects_lambda_outside_unit_interval(lam: float) -> None:
    with pytest.raises(DeclibError):
        apply_prior(np.log(np.array([[0.5, 0.5]])), np.array([0.5, 0.5]), lam=lam)


def test_apply_prior_rejects_shape_mismatch_and_bad_prior() -> None:
    logp = np.log(np.array([[0.5, 0.5]]))
    with pytest.raises(DeclibError):
        apply_prior(logp, np.array([0.2, 0.3, 0.5]))
    with pytest.raises(DeclibError):
        apply_prior(logp, np.array([0.0, 1.0]))
    with pytest.raises(DeclibError):
        fit_prior(np.empty((0, 2)))


def test_prior_correction_removes_label_bias_on_noul() -> None:
    q = Noul("Refund?", name="r")
    b = FakeBackend(label_prior={"Yes": 2.0})
    raw = raw_logprobs(b, STATES, q)
    corrected = apply_prior(raw, fit_prior(raw))
    raw_gap = abs(np.exp(raw[:, 0]).mean() - 0.5)
    new_gap = abs(np.exp(corrected[:, 0]).mean() - 0.5)
    # Dividing by the mean probability only partly cancels an additive logit bias
    # (here 0.80 -> 0.61 vs 0.51 unbiased), so assert a large reduction, not removal.
    assert raw_gap > 0.25
    assert new_gap < raw_gap / 2


@given(
    st.lists(st.lists(st.floats(-20, 0), min_size=3, max_size=3), min_size=1, max_size=20),
    st.floats(0, 1),
)
def test_prior_outputs_are_distributions(rows: list[list[float]], lam: float) -> None:
    logp = norm(np.array(rows))
    out = apply_prior(logp, fit_prior(logp), lam=lam)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0, atol=1e-9)


def test_tokenization_error_propagates_for_non_score() -> None:
    from declib.errors import TokenizationError

    class NoYes(FakeBackend):
        def label_token_ids(self, labels: list[str]) -> list[int]:  # type: ignore[override]
            raise TokenizationError("no single-token Yes")

    with pytest.raises(TokenizationError):
        raw_logprobs(NoYes(), ["s"], Noul("Refund?", name="r"))


def test_apply_prior_accepts_numpy_scalar_lambda() -> None:
    logp = np.log(np.array([[0.6, 0.4]]))
    out = apply_prior(logp, np.array([0.75, 0.25]), lam=np.float32(1.0))
    np.testing.assert_allclose(np.exp(out), [[1 / 3, 2 / 3]], rtol=1e-6)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_fit_prior_rejects_non_finite(bad: float) -> None:
    with pytest.raises(DeclibError):
        fit_prior(np.array([[bad, 0.0]]))
