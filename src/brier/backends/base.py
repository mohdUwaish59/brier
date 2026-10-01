"""The Backend protocol: the only boundary between brier and model code (ADR-0002)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Backend(Protocol):
    """A causal LM that can score label tokens and expose hidden states.

    ``prompts`` are user messages from :func:`brier.prompts.render`; the backend wraps
    them in its chat template with ``prompts.SYSTEM`` and ``prompts.ANSWER_PREFIX``.
    """

    model_id: str
    revision: str | None
    num_layers: int

    def label_token_ids(self, labels: Sequence[str]) -> list[int]:
        """Token id of each label as it appears right after the answer prefix.

        Raises
        ------
        TokenizationError
            If any label is not a single, distinct token.
        """
        ...

    def label_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> npt.NDArray[np.float64]:
        """``(n_prompts, n_labels)`` log-probs of ``token_ids`` at the next position.

        Taken from a full-vocabulary log-softmax, then gathered; batched internally.
        """
        ...

    def hidden_states(
        self, prompts: Sequence[str], layers: Sequence[int]
    ) -> npt.NDArray[np.float32]:
        """``(n_prompts, n_layers, d)`` residual-stream state at the last prompt token.

        May stop the forward pass after ``max(layers)``.
        """
        ...
