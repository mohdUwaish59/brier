"""The :class:`Decider`: the orchestration class most users touch."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from brier._math import FloatArray
from brier.artifacts import (
    DEFAULT_MAX_BYTES,
    Artifact,
    Calibration,
    load_artifact,
    save_artifact,
)
from brier.backends.base import Backend
from brier.calibrate.temperature import MIN_ITEMS, apply_temperature, fit_temperature
from brier.debias import apply_prior, fit_prior, l0_logprobs
from brier.decision import Decision, Level, QuestionType
from brier.errors import (
    ArtifactError,
    BrierError,
    InputTooLargeError,
    InsufficientDataError,
    NotFittedError,
    QuestionError,
)
from brier.heads._features import IntArray
from brier.heads.fitted import L2Head
from brier.heads.select import default_layers, select_head
from brier.prompts import render
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
        self._temperatures: dict[Question, float] = {}
        self._heads: dict[Question, L2Head] = {}

    def save(self, path: str | Path) -> None:
        """Save fitted calibration (priors, temperatures, ``prior_strength``), never weights.

        Parameters
        ----------
        path : str or Path
            Directory to create (must be empty if it exists). See ADR-0003.

        Raises
        ------
        ArtifactError
            If ``path`` is a file or a non-empty directory.
        """
        questions = list(dict.fromkeys([*self._priors, *self._temperatures, *self._heads]))
        calibrations = tuple(
            Calibration(q, self._priors.get(q), self._temperatures.get(q), self._heads.get(q))
            for q in questions
        )
        artifact = Artifact(
            self.backend.model_id, self.backend.revision, self.prior_strength, calibrations
        )
        save_artifact(path, artifact)

    @classmethod
    def load(
        cls,
        path: str | Path,
        backend: Backend,
        *,
        score_prior: bool = False,
        max_questions: int = 32,
        max_batch: int = 64,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> Decider:
        """Create a Decider from a saved calibration artifact.

        The artifact must have been fitted on ``backend``'s exact model id and revision and
        with the current prompt templates; it is validated as untrusted input
        (THREAT_MODEL T2). ``prior_strength`` comes from the artifact. With
        ``revision=None`` the match is by model name only and does not pin weights; pin a
        commit SHA. The checksum guards integrity, not authorship.

        Parameters
        ----------
        path : str or Path
            Artifact directory written by :meth:`save`.
        backend : Backend
            Model access; must match the artifact's model.
        score_prior, max_questions, max_batch : see :class:`Decider`
        max_bytes : int
            Size cap for the artifact files (default 100 MB).

        Raises
        ------
        ArtifactError
            If the artifact is invalid or does not match the backend or templates.
        """
        artifact = load_artifact(
            path, model_id=backend.model_id, revision=backend.revision, max_bytes=max_bytes
        )
        decider = cls(
            backend,
            prior_strength=artifact.prior_strength,
            score_prior=score_prior,
            max_questions=max_questions,
            max_batch=max_batch,
        )
        for c in artifact.calibrations:
            if c.prior is not None:
                decider._priors[c.question] = c.prior
            if c.temperature is not None:
                decider._temperatures[c.question] = c.temperature
            if c.head is not None:
                if c.head.layer >= backend.num_layers:
                    raise ArtifactError(
                        f"head layer {c.head.layer} does not exist in a "
                        f"{backend.num_layers}-layer model"
                    )
                decider._heads[c.question] = c.head
        return decider

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
        if level == "L2":
            missing = [q.name for q in questions if q not in self._heads]
            if missing:
                raise NotFittedError(f"no L2 head fitted for {missing}")
        if level == "L1":
            missing = [q.name for q in questions if q not in self._temperatures]
            if missing:
                raise NotFittedError(f"no L1 temperature fitted for {missing}")
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
            self._temperatures.pop(q, None)  # fitted on top of the old prior: now stale

    def fit_temperature(
        self, states: Sequence[str], question: Question, labels: Sequence[object]
    ) -> None:
        """Fit the L1 temperature of one question on labelled states.

        ``T`` is fitted on top of L0 (including the prior, if fitted). Fit the prior
        first: refitting it later discards the temperature.

        Parameters
        ----------
        states : Sequence[str]
            At least 50 labelled states.
        question : Choice, Noul or Score
            The question.
        labels : Sequence
            One answer per state, typed like :attr:`Decision.answer`: an option string
            (Choice), a ``bool`` (Noul) or an int level ``1..L`` (Score).

        Raises
        ------
        InsufficientDataError
            If there are fewer than 50 labelled states.
        QuestionError
            If a label does not match the question type or the counts differ.
        """
        states = self._check_states(states)
        y = _label_indices(question, labels, len(states))
        if len(states) < MIN_ITEMS:
            raise InsufficientDataError(f"need at least {MIN_ITEMS} labelled states")
        logp, _ = self._logprobs(states, question, "L0")
        self._temperatures[question] = fit_temperature(logp, y)

    def fit_head(
        self,
        states: Sequence[str],
        question: Question,
        labels: Sequence[object],
        layers: Sequence[int] | None = None,
    ) -> None:
        """Fit the L2 hidden-state head of one question (METHODS.md, L2).

        Reads the hidden state at the last prompt token for each candidate layer, selects
        layer, solver and alpha by stratified 5-fold out-of-fold NLL, refits on all labels and
        fits the L2 temperature on the out-of-fold scores.

        Parameters
        ----------
        states : Sequence[str]
            Labelled states: at least 60, and at least 5 per answer.
        question : Choice, Noul or Score
            The question (prompt rendered without rotation; Score with digit labels).
        labels : Sequence
            One answer per state, typed like :attr:`Decision.answer`.
        layers : Sequence[int] or None
            Candidate blocks; default every 2nd block from 40 % to 90 % depth.

        Raises
        ------
        InsufficientDataError
            With too few labels overall or for some answer.
        QuestionError, BrierError
            On malformed labels or layers.

        Warns
        -----
        UserWarning
            If the L2 temperature hit its lower search bound (tiny out-of-fold score
            gaps): the head may be overconfident.
        """
        states = self._check_states(states)
        y = _label_indices(question, labels, len(states))
        n_layers = self.backend.num_layers
        cand = default_layers(n_layers) if layers is None else list(layers)
        if (
            not cand
            or len(set(cand)) != len(cand)
            or not all(
                isinstance(b, int) and not isinstance(b, bool) and 0 <= b < n_layers for b in cand
            )
        ):
            raise BrierError(f"layers must be distinct ints in [0, {n_layers})")
        prompts = [render(s, question) for s in states]
        hidden = self.backend.hidden_states(prompts, cand)
        head = L2Head.from_selection(select_head(hidden, cand, y, len(_answer_keys(question))))
        # Lower bound: T could not sharpen enough, risk of overconfidence. (Upper bound means
        # ~uniform predictions: an uninformative but honest head, so no warning.)
        if head.temperature_at_bound and head.temperature < 1.0:
            warnings.warn(
                f"L2 temperature for {question.name!r} hit its lower search bound, "
                "so the head may be overconfident",
                UserWarning,
                stacklevel=2,
            )
        self._heads[question] = head

    def _logprobs(
        self, states: list[str], q: Question, level: Level
    ) -> tuple[FloatArray, dict[str, object]]:
        meta: dict[str, object] = {
            "model_id": self.backend.model_id,
            "revision": self.backend.revision,
        }
        if level == "L2":
            head = self._heads[q]
            hidden = self.backend.hidden_states([render(s, q) for s in states], [head.layer])
            meta.update(
                n_forward=1, layer=head.layer, solver=head.solver, temperature=head.temperature
            )
            return head.log_probs(hidden[:, 0]), meta
        if level == "raw":
            meta["n_forward"] = 1
            return raw_logprobs(self.backend, states, q), meta
        logp = l0_logprobs(self.backend, states, q)
        prior = self._priors.get(q)
        meta["n_forward"] = len(q.options) if isinstance(q, Choice) else 1
        meta["prior_applied"] = prior is not None
        if prior is not None:
            logp = apply_prior(logp, prior, lam=self.prior_strength)
        if level == "L1":
            temperature = self._temperatures[q]
            meta["temperature"] = temperature
            logp = apply_temperature(logp, temperature)
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


def _label_index(q: Question, label: object) -> int:
    """Index of ``label`` in the question's answer order (see ``fit_temperature``)."""
    if isinstance(q, Choice):
        if isinstance(label, str) and label in q.options:
            return list(q.options).index(label)
        raise QuestionError(f"label {label!r} is not one of the options of {q.name!r}")
    if isinstance(q, Noul):
        if isinstance(label, (bool, np.bool_)):
            return 0 if label else 1
        raise QuestionError(f"Noul labels must be bool, got {label!r}")
    if (
        isinstance(label, (bool, np.bool_))
        or not isinstance(label, (int, np.integer))
        or not 1 <= label <= q.levels
    ):
        raise QuestionError(f"Score labels must be ints in 1..{q.levels}, got {label!r}")
    return int(label) - 1


def _label_indices(question: Question, labels: Sequence[object], n_states: int) -> IntArray:
    """Validate and convert labels typed like ``Decision.answer`` to answer indices."""
    if not isinstance(question, (Choice, Noul, Score)):
        raise QuestionError("question must be a Choice, Noul or Score instance")
    items = list(labels)
    if len(items) != n_states:
        raise QuestionError(f"{len(items)} labels for {n_states} states")
    return np.array([_label_index(question, label) for label in items], dtype=np.int64)
