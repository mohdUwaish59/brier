"""Deterministic backend for tests: known content signal plus configurable biases."""

from __future__ import annotations

import hashlib
import html
import math
import re
import string
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from brier._math import norm
from brier.errors import BrierError, TokenizationError

# No "10": a two-digit level is never a single token, as in real tokenizers.
_VOCAB = (*string.ascii_uppercase, "Yes", "No", *"123456789", "the", "I", "Sure", ".")
_IDS = {tok: i for i, tok in enumerate(_VOCAB)}
_OPTION_LINE = re.compile(r"^([A-Z]|[1-9])\. (.*)$")
_FILLER_LOGIT = 0.0  # logit of every non-label token: leaves some mass off the labels


def _hash_unit(*parts: object) -> float:
    """Stable value in [0, 1) from ``parts`` (``hash()`` is salted per process)."""
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


@dataclass(frozen=True)
class FakeBackend:
    """Test backend with a known content signal and configurable biases.

    The logit of the label at display position ``j`` is
    ``content(state, option_j) + position_bias[j] + label_prior[label]``.

    Parameters
    ----------
    position_bias : Sequence[float]
        Added to the logit of the label at each display position (missing = 0).
    label_prior : Mapping[str, float]
        Added to the logit of each label token (missing = 0).
    content : callable or None
        ``(state, option_text) -> float``; the option text is the rendered option or
        level label, or the label itself (Noul, unlabelled Score). Default: a stable
        hash of ``(seed, state, option)`` in [-2, 2).
    seed : int
        Seed for the default content and hidden states.
    hidden_size, num_layers : int
        Shape of :meth:`hidden_states`.
    """

    position_bias: Sequence[float] = ()
    label_prior: Mapping[str, float] = field(default_factory=dict)
    content: Callable[[str, str], float] | None = None
    seed: int = 0
    hidden_size: int = 16
    num_layers: int = 8
    model_id: str = "fake"
    revision: str | None = None

    def __post_init__(self) -> None:
        values = [*self.position_bias, *self.label_prior.values()]
        if not all(math.isfinite(v) for v in values):
            raise BrierError("FakeBackend biases must be finite")

    def label_token_ids(self, labels: Sequence[str]) -> list[int]:
        """Vocabulary id of each label (strict: known and distinct)."""
        if len(set(labels)) != len(labels):
            raise TokenizationError("labels must be distinct")
        missing = [lab for lab in labels if lab not in _IDS]
        if missing:
            raise TokenizationError(f"not a single token: {missing}")
        return [_IDS[lab] for lab in labels]

    def label_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> npt.NDArray[np.float64]:
        """Full-vocabulary log-softmax gathered at ``token_ids``."""
        out = np.empty((len(prompts), len(token_ids)), dtype=np.float64)
        for i, prompt in enumerate(prompts):
            state, shown = self._parse(prompt)
            logits = np.full(len(_VOCAB), _FILLER_LOGIT)
            for j, tid in enumerate(token_ids):
                label = _VOCAB[tid]
                logits[tid] = (
                    self._content(state, shown.get(label, label))
                    + (self.position_bias[j] if j < len(self.position_bias) else 0.0)
                    + self.label_prior.get(label, 0.0)
                )
            out[i] = norm(logits)[list(token_ids)]
        return out

    def hidden_states(
        self, prompts: Sequence[str], layers: Sequence[int]
    ) -> npt.NDArray[np.float32]:
        """Hash-seeded Gaussian features, deterministic per (seed, prompt, layer)."""
        if not layers or not all(0 <= b < self.num_layers for b in layers):
            raise BrierError(f"layers must be non-empty and in [0, {self.num_layers})")
        out = np.empty((len(prompts), len(layers), self.hidden_size), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            for j, layer in enumerate(layers):
                rng = np.random.default_rng(int(_hash_unit(self.seed, prompt, layer) * 2**63))
                out[i, j] = rng.standard_normal(self.hidden_size)
        return out

    def _content(self, state: str, option: str) -> float:
        if self.content is not None:
            return float(self.content(state, option))
        return 4.0 * _hash_unit(self.seed, state, option) - 2.0

    @staticmethod
    def _parse(prompt: str) -> tuple[str, dict[str, str]]:
        """Unescaped state and ``{label: shown text}`` from a rendered prompt."""
        head, _, rest = prompt.partition("\n</state>\n")
        state = html.unescape(head.removeprefix("<state>\n"))
        shown = {}
        for line in rest.splitlines():
            m = _OPTION_LINE.match(line)
            if m:
                shown[m.group(1)] = m.group(2)
        return state, shown
