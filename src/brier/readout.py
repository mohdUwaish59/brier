"""Level ``raw``: label-token log-probs restricted and renormalised (METHODS.md)."""

from __future__ import annotations

from collections.abc import Sequence

from brier._math import FloatArray, norm
from brier.backends.base import Backend
from brier.errors import TokenizationError
from brier.prompts import labels, render
from brier.questions import Question, Score


def resolve_labels(backend: Backend, question: Question) -> tuple[bool, list[int]]:
    """Label token ids, falling back to letters for Score if a digit is not one token.

    Returns
    -------
    score_letters : bool
        Whether Score levels must be rendered as lettered options.
    token_ids : list of int
        Token id of each label, in display-position order.
    """
    try:
        return False, backend.label_token_ids(labels(question))
    except TokenizationError:
        if not isinstance(question, Score):
            raise
    return True, backend.label_token_ids(labels(question, score_letters=True))


def raw_logprobs(
    backend: Backend, states: Sequence[str], question: Question, shift: int = 0
) -> FloatArray:
    """``(n_states, K)`` log-probs over display positions: ``norm(z)``.

    Parameters
    ----------
    backend : Backend
        Model access.
    states : Sequence[str]
        State texts.
    question : Choice, Noul or Score
        The question.
    shift : int
        Option rotation (Choice only); columns are display positions, not options.

    Returns
    -------
    numpy.ndarray
        float64 array; each row sums to 1 in probability space.
    """
    score_letters, token_ids = resolve_labels(backend, question)
    prompts = [render(s, question, shift=shift, score_letters=score_letters) for s in states]
    return norm(backend.label_logprobs(prompts, token_ids))
