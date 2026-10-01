"""Typed questions: :class:`Choice`, :class:`Noul` and :class:`Score`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from brier.errors import QuestionError

MAX_OPTIONS = 26
MAX_OPTION_CHARS = 500
MAX_SCORE_LEVELS = 10


def _check_str(value: object, what: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise QuestionError(f"{what} must be a non-empty string")


def _check_strings(values: object, what: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise QuestionError(f"{what} must be a sequence of strings")
    out = tuple(values)
    for v in out:
        _check_str(v, what)
        if len(v) > MAX_OPTION_CHARS:
            raise QuestionError(f"{what} must be at most {MAX_OPTION_CHARS} characters")
        if v.splitlines() != [v]:
            # A line break could fake another option's label line ("x\nB. y").
            raise QuestionError(f"{what} must not contain line breaks")
    return out


@dataclass(frozen=True)
class Choice:
    """Pick one of 2-26 options.

    Parameters
    ----------
    text : str
        The question.
    options : Sequence[str]
        Unique, non-empty option strings (at most 500 characters each).
    name : str
        Key of this question in the result.
    """

    text: str
    options: Sequence[str]
    name: str

    def __post_init__(self) -> None:
        _check_str(self.text, "text")
        _check_str(self.name, "name")
        options = _check_strings(self.options, "options")
        if not 2 <= len(options) <= MAX_OPTIONS:
            raise QuestionError(f"Choice needs 2-{MAX_OPTIONS} options, got {len(options)}")
        if len(set(options)) != len(options):
            raise QuestionError("options must be unique")
        object.__setattr__(self, "options", options)


@dataclass(frozen=True)
class Noul:
    """A yes/no question.

    Parameters
    ----------
    text : str
        The question.
    name : str
        Key of this question in the result.
    """

    text: str
    name: str

    def __post_init__(self) -> None:
        _check_str(self.text, "text")
        _check_str(self.name, "name")


@dataclass(frozen=True)
class Score:
    """Rate on integer levels ``1..levels``.

    Parameters
    ----------
    text : str
        The question.
    levels : int
        Number of levels, between 2 and 10.
    name : str
        Key of this question in the result.
    labels : Sequence[str] or None
        Optional description of each level, one per level, lowest first.
    """

    text: str
    levels: int
    name: str
    labels: Sequence[str] | None = None

    def __post_init__(self) -> None:
        _check_str(self.text, "text")
        _check_str(self.name, "name")
        if isinstance(self.levels, bool) or not isinstance(self.levels, int):
            raise QuestionError("levels must be an int")
        if not 2 <= self.levels <= MAX_SCORE_LEVELS:
            raise QuestionError(f"Score needs 2-{MAX_SCORE_LEVELS} levels, got {self.levels}")
        if self.labels is not None:
            labels = _check_strings(self.labels, "labels")
            if len(labels) != self.levels:
                raise QuestionError(f"labels needs {self.levels} entries, got {len(labels)}")
            object.__setattr__(self, "labels", labels)


Question = Choice | Noul | Score


def validate_questions(questions: Sequence[Question]) -> None:
    """Raise :class:`QuestionError` if two questions in one call share a ``name``."""
    names = [q.name for q in questions]
    if len(set(names)) != len(names):
        raise QuestionError("question names must be unique within one call")
