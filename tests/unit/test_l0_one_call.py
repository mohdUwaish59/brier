"""M8.1 (ADR-0010 phase A): L0 sends all rotations of a question in one backend call."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest

from brier import Choice, Noul, Score
from brier._math import norm
from brier.backends.fake import FakeBackend
from brier.debias import combine, combine_rotated, l0_logprobs, rotated_prompts, unrotate
from brier.errors import BrierError
from brier.prompts import render
from brier.readout import raw_logprobs

ROUTE = Choice("Which team?", ["a", "b", "c", "d", "e"], name="route")
STATES = [f"message number {i}" for i in range(7)]
BIASED = FakeBackend(position_bias=(1.5, 0.5, 0.0, -0.5, -1.0), label_prior={"B": 0.3})


@dataclass(frozen=True)
class _Counting(FakeBackend):
    calls: list[int] = field(default_factory=list)

    def label_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> npt.NDArray[np.float64]:
        self.calls.append(len(prompts))
        return super().label_logprobs(prompts, token_ids)


def _per_rotation(states: list[str], q: Choice, shifts: list[int]) -> np.ndarray:
    """The pre-M8.1 implementation: one backend call per rotation."""
    return combine([unrotate(raw_logprobs(BIASED, states, q, s), s) for s in shifts])


@pytest.mark.parametrize("shifts", [None, [0], [0, 2], [1, 3, 4], [0, 1, 2, 3, 4]])
def test_one_call_matches_per_rotation_bit_for_bit(shifts: list[int] | None) -> None:
    s_list = list(range(5)) if shifts is None else shifts
    got = l0_logprobs(BIASED, STATES, ROUTE, shifts=shifts)
    np.testing.assert_array_equal(got, _per_rotation(STATES, ROUTE, s_list))


def test_all_rotations_go_in_one_call() -> None:
    backend = _Counting(position_bias=(1.0, 0.0, -1.0))
    q = Choice("Pick", ["x", "y", "z"], name="q")
    l0_logprobs(backend, STATES, q)
    assert backend.calls == [len(STATES) * 3]


def test_noul_and_score_still_use_the_raw_readout() -> None:
    noul = Noul("Refund?", name="refund")
    score = Score("Urgency", levels=10, name="u")  # 10 levels: lettered fallback
    for q in (noul, score):
        np.testing.assert_array_equal(
            l0_logprobs(BIASED, STATES, q), raw_logprobs(BIASED, STATES, q)
        )


def test_rotated_prompts_are_shift_major() -> None:
    prompts = rotated_prompts(STATES[:2], ROUTE, [0, 3])
    assert prompts == [
        render(STATES[0], ROUTE, shift=0),
        render(STATES[1], ROUTE, shift=0),
        render(STATES[0], ROUTE, shift=3),
        render(STATES[1], ROUTE, shift=3),
    ]


def test_combine_rotated_reshapes_and_unrotates() -> None:
    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(3, 5)) for _ in range(2)]
    got = combine_rotated(np.concatenate(blocks), [1, 4], n_states=3)
    # each block is normalised first, as raw_logprobs did before M8.1
    want = combine([unrotate(norm(blocks[0]), 1), unrotate(norm(blocks[1]), 4)])
    np.testing.assert_array_equal(got, want)


def test_combine_rotated_rejects_a_wrong_row_count() -> None:
    with pytest.raises(BrierError, match="backend returned"):
        combine_rotated(np.zeros((5, 5)), [0, 1], n_states=3)


def test_empty_states() -> None:
    assert l0_logprobs(BIASED, [], ROUTE).shape == (0, 5)
