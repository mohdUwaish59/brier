"""The :class:`Decider`: the orchestration class most users touch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from brier._math import FloatArray
from brier.backends.base import Backend
from brier.debias import apply_prior, fit_prior, l0_logprobs
from brier.decision import Decision, Level, QuestionType
from brier.errors import (
    BrierError,
    InputTooLargeError,
    InsufficientDataError,
    NotFittedError,
    QuestionError,
)
from brier.questions import Choice, Noul, Question, Score, validate_questions
from brier.readout import raw_logprobs

_LEVELS = ("raw", "L0", "L1", "L2")


def _check_limit(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BrierError(f"{what} must be a positive int")
    return value


class Decider:
    """Answer typed questions about states with calibrated probabilities.

    Parameters
    ----------
    backend : Backend
        Model access (e.g. :class:`brier.backends.hf.HFBackend`).
    prior_strength : float
        λ in ``[0, 1]`` for the L0 batch prior correction (ADR-0005).
    score_prior : bool
        Fit and apply the L0 prior to Score questions too (off by default).
    max_questions, max_batch : int
        Per-call limits (SPEC §5). The per-prompt token cap is the backend's.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        prior_strength: float = 1.0,
        score_prior: bool = False,
        max_questions: int = 32,
        max_batch: int = 64,
    ) -> None:
        if isinstance(prior_strength, bool) or not 0.0 <= float(prior_strength) <= 1.0:
            raise BrierError("prior_strength must be in [0, 1]")
        self.backend = backend
        self.prior_strength = float(prior_strength)
        self.score_prior = bool(score_prior)
        self.max_questions = _check_limit(max_questions, "max_questions")
        self.max_batch = _check_limit(max_batch, "max_batch")
        # Keyed by the whole question (text, options, name), never by name alone.
        self._priors: dict[Question, FloatArray] = {}

    def decide(
        self, state: str, questions: Sequence[Question], level: Level = "L0"
    ) -> dict[str, Decision]:
        """Decide every question for one state.

        Parameters
        ----------
        state : str
            The text the questions are about (untrusted; see THREAT_MODEL T3).
        questions : Sequence of Choice, Noul or Score
            Questions with unique names.
        level : {"raw", "L0", "L1", "L2"}
            Correction level.

        Returns
        -------
        dict of str to Decision
            Keyed by question name.

        Raises
        ------
        QuestionError, InputTooLargeError, NotFittedError, TokenizationError
            On invalid input, exceeded limits, unfitted levels or unusable labels.
        """
        return self.decide_batch([state], questions, level)[0]

    def decide_batch(
        self, states: Sequence[str], questions: Sequence[Question], level: Level = "L0"
    ) -> list[dict[str, Decision]]:
        """Decide every question for each state (see :meth:`decide`)."""
        states = self._check_states(states)
        if len(states) > self.max_batch:
            raise InputTooLargeError(f"{len(states)} states exceed max_batch={self.max_batch}")
        self._check_questions(questions)
        if level not in _LEVELS:
            raise BrierError(f"unknown level {level!r}")
        if level in ("L1", "L2"):
            raise NotFittedError(f"level {level} has not been fitted")
        out: list[dict[str, Decision]] = [{} for _ in states]
        if not states:
            return out
        for q in questions:
            logp, meta = self._logprobs(states, q, level)
            keys = _answer_keys(q)
            for i, row in enumerate(np.exp(logp)):
                probs = dict(zip(keys, row.tolist(), strict=True))
                out[i][q.name] = Decision(q.name, _type(q), probs, level, meta)
        return out

    def fit_prior(self, states: Sequence[str], questions: Sequence[Question]) -> None:
        """Estimate each question's L0 label prior from unlabelled states.

        Score questions are skipped unless ``score_prior=True``.

        Raises
        ------
        InsufficientDataError
            If ``states`` is empty.
        """
        states = self._check_states(states)
        if not states:
            raise InsufficientDataError("fit_prior needs at least one state")
        validate_questions(questions)
        for q in questions:
            if isinstance(q, Score) and not self.score_prior:
                continue
            self._priors[q] = fit_prior(l0_logprobs(self.backend, states, q))

    def _logprobs(
        self, states: list[str], q: Question, level: Level
    ) -> tuple[FloatArray, dict[str, object]]:
        meta: dict[str, object] = {
            "model_id": self.backend.model_id,
            "revision": self.backend.revision,
        }
        if level == "raw":
            meta["n_forward"] = 1
            return raw_logprobs(self.backend, states, q), meta
        logp = l0_logprobs(self.backend, states, q)
        prior = self._priors.get(q)
        meta["n_forward"] = len(q.options) if isinstance(q, Choice) else 1
        meta["prior_applied"] = prior is not None
        if prior is not None:
            logp = apply_prior(logp, prior, lam=self.prior_strength)
        return logp, meta

    def _check_questions(self, questions: Sequence[Question]) -> None:
        if not isinstance(questions, Sequence):
            raise QuestionError("questions must be a sequence of questions")
        if not questions:
            raise QuestionError("at least one question is required")
        if len(questions) > self.max_questions:
            raise InputTooLargeError(
                f"{len(questions)} questions exceed max_questions={self.max_questions}"
            )
        if not all(isinstance(q, (Choice, Noul, Score)) for q in questions):
            raise QuestionError("questions must be Choice, Noul or Score instances")
        validate_questions(questions)

    @staticmethod
    def _check_states(states: Sequence[str]) -> list[str]:
        if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
            raise QuestionError("states must be a sequence of strings")
        if not all(isinstance(s, str) for s in states):
            raise QuestionError("every state must be a string")
        return list(states)


def _type(q: Question) -> QuestionType:
    if isinstance(q, Choice):
        return "choice"
    return "noul" if isinstance(q, Noul) else "score"


def _answer_keys(q: Question) -> list[str]:
    if isinstance(q, Choice):
        return list(q.options)
    if isinstance(q, Noul):
        return ["yes", "no"]
    return [str(i) for i in range(1, q.levels + 1)]
