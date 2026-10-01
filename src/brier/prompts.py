"""Prompt rendering. The exact strings here are covered by snapshot tests.

``render`` returns the *user message*. The backend wraps it in the model's chat
template with :data:`SYSTEM` as the system message and :data:`ANSWER_PREFIX` as the
start of the assistant turn; label tokens are read at the position right after it.
Changing any template string is a breaking change for stored artifacts (ADR-0004).
"""

from __future__ import annotations

import html
import string
from collections.abc import Sequence

from brier.errors import QuestionError
from brier.questions import Choice, Noul, Question

SYSTEM = "You answer multiple-choice questions about the text in <state> tags."
ANSWER_PREFIX = "Answer:"

_LETTERS = string.ascii_uppercase
_ANSWER_LETTER = "Answer with the letter only."


def escape_state(state: str) -> str:
    """Escape ``&``, ``<`` and ``>`` so the state cannot open or close a ``<state>`` block."""
    return html.escape(state, quote=False)


def rotate(options: Sequence[str], shift: int) -> tuple[str, ...]:
    """Display order for ``shift``: position ``j`` shows option ``(j + shift) mod K``."""
    return tuple(options[shift:]) + tuple(options[:shift])


def labels(question: Question, score_letters: bool = False) -> tuple[str, ...]:
    """Answer labels read from the model, in display-position order.

    Parameters
    ----------
    question : Choice, Noul or Score
        The question.
    score_letters : bool
        Use letters instead of digits for Score (when a digit is not a single token).

    Returns
    -------
    tuple of str
        ``A``.. for Choice, ``Yes``/``No`` for Noul, ``1``..``L`` (or ``A``..) for Score.
    """
    if isinstance(question, Choice):
        return tuple(_LETTERS[: len(question.options)])
    if isinstance(question, Noul):
        return ("Yes", "No")
    if score_letters:
        return tuple(_LETTERS[: question.levels])
    return tuple(str(i) for i in range(1, question.levels + 1))


def render(state: str, question: Question, shift: int = 0, score_letters: bool = False) -> str:
    """Render the user message for one state and question.

    Parameters
    ----------
    state : str
        Untrusted text; escaped before rendering.
    question : Choice, Noul or Score
        The question.
    shift : int
        Option rotation, ``0 <= shift < K`` (Choice only).
    score_letters : bool
        Render Score levels as lettered options (see :func:`labels`).

    Returns
    -------
    str
        The user message.

    Raises
    ------
    QuestionError
        If ``state`` is not a string or ``shift`` is invalid for the question.
    """
    if not isinstance(state, str):
        raise QuestionError("state must be a string")
    lines = ["<state>", escape_state(state), "</state>", f"Question: {question.text}"]
    if isinstance(shift, bool) or not isinstance(shift, int):
        raise QuestionError("shift must be an int")
    if isinstance(question, Choice):
        if not 0 <= shift < len(question.options):
            raise QuestionError(f"shift must be in [0, {len(question.options)})")
        shown = rotate(question.options, shift)
        lines += [f"{lab}. {opt}" for lab, opt in zip(labels(question), shown, strict=True)]
        lines.append(_ANSWER_LETTER)
        return "\n".join(lines)
    if shift != 0:
        raise QuestionError("only Choice questions can be rotated")
    if isinstance(question, Noul):
        lines.append("Answer Yes or No.")
        return "\n".join(lines)
    levels = [str(i) for i in range(1, question.levels + 1)]
    if score_letters:
        texts = question.labels or levels
        lines += [f"{lab}. {t}" for lab, t in zip(labels(question, True), texts, strict=True)]
        lines.append(_ANSWER_LETTER)
    else:
        if question.labels:
            lines += [f"{lv}. {t}" for lv, t in zip(levels, question.labels, strict=True)]
        else:
            lines.append(f"Scale: 1 (lowest) to {question.levels} (highest).")
        lines.append("Answer with the number only.")
    return "\n".join(lines)
