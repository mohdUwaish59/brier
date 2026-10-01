"""The :class:`Decision` result type."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from declib.errors import DeclibError

Level = Literal["raw", "L0", "L1", "L2"]
QuestionType = Literal["choice", "noul", "score"]

_LEVELS = ("raw", "L0", "L1", "L2")
_TYPES = ("choice", "noul", "score")
_SUM_TOL = 1e-9


@dataclass(frozen=True)
class Decision:
    """Answer distribution for one question.

    Parameters
    ----------
    name : str
        The question's name.
    type : {"choice", "noul", "score"}
        The question type.
    probs : Mapping[str, float]
        Probability of each answer, in question order. Keys are the option strings
        for Choice, ``"yes"``/``"no"`` for Noul and ``"1"``..``"L"`` for Score.
    level : {"raw", "L0", "L1", "L2"}
        The correction level that produced it.
    meta : Mapping[str, object]
        Provenance (model id, revision, layer, number of forward passes).

    Raises
    ------
    DeclibError
        If ``probs`` is empty, non-finite, negative or does not sum to 1 (within 1e-9),
        or if ``type`` or ``level`` is unknown.
    """

    name: str
    type: QuestionType
    probs: Mapping[str, float]
    level: Level
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in _TYPES:
            raise DeclibError(f"unknown question type {self.type!r}")
        if self.level not in _LEVELS:
            raise DeclibError(f"unknown level {self.level!r}")
        probs = {k: float(v) for k, v in self.probs.items()}
        values = probs.values()
        if not probs or not all(math.isfinite(p) and p >= 0.0 for p in values):
            raise DeclibError(f"{self.name}: probabilities must be finite and non-negative")
        if abs(math.fsum(values) - 1.0) > _SUM_TOL:
            raise DeclibError(f"{self.name}: probabilities must sum to 1")
        object.__setattr__(self, "probs", MappingProxyType(probs))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    @property
    def _top(self) -> str:
        # max() keeps the first key on ties, so ties go to the first-listed answer.
        return max(self.probs, key=self.probs.__getitem__)

    @property
    def answer(self) -> str | bool | int:
        """Most probable answer: option string, ``bool`` for Noul, ``int`` level for Score."""
        if self.type == "noul":
            return self._top == "yes"
        if self.type == "score":
            return int(self._top)
        return self._top

    @property
    def confidence(self) -> float:
        """Probability of the answer (the maximum probability)."""
        return self.probs[self._top]

    @property
    def p_yes(self) -> float:
        """Probability of "yes" (Noul only)."""
        if self.type != "noul":
            raise DeclibError("p_yes is only defined for Noul decisions")
        return self.probs["yes"]

    @property
    def expected(self) -> float:
        """Expected level (Score only)."""
        if self.type != "score":
            raise DeclibError("expected is only defined for Score decisions")
        return math.fsum(int(k) * p for k, p in self.probs.items())
