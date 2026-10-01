"""Hugging Face ``transformers`` backend.

The only module that may import torch/transformers, and only lazily (ADR-0002).
Security (THREAT_MODEL T1): ``trust_remote_code=False`` and safetensors only.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from brier.errors import BrierError, InputTooLargeError, TokenizationError
from brier.prompts import ANSWER_PREFIX, SYSTEM

# Placeholder for the user message when splitting the chat template (private-use chars).
_SENTINEL = chr(0xE000) + "brier-user" + chr(0xE000)
_DTYPES = ("float32", "bfloat16", "float16")
# A token covers at most this many characters in practice; cheap pre-check before tokenising.
_MAX_CHARS_PER_TOKEN = 32


class _StopForward(Exception):  # noqa: N818 - control flow, not an error
    """Raised by a hook to stop the forward pass after the deepest requested layer."""


class HFBackend:
    """Causal LM loaded with ``transformers``.

    Parameters
    ----------
    model_id : str
        Hugging Face model id. The model must have a chat template.
    revision : str or None
        Commit sha to pin (recommended).
    device : str or None
        Torch device; default ``"cuda"`` if available, else ``"cpu"``.
    dtype : {"float32", "bfloat16", "float16"} or None
        Default ``"bfloat16"`` on CUDA, ``"float32"`` on CPU.
    batch_size : int
        Prompts per forward pass.
    max_prompt_tokens : int
        Hard cap on one encoded prompt (default: 8,192-token state plus room for the
        template and question), lowered to the model's context length if smaller.

    Raises
    ------
    BrierError
        If the ``hf`` extra is missing, an argument is invalid or the model has no
        usable chat template.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        *,
        device: str | None = None,
        dtype: str | None = None,
        batch_size: int = 8,
        max_prompt_tokens: int = 9216,
    ) -> None:
        try:
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as e:  # pragma: no cover - depends on installed extras
            raise BrierError('HFBackend needs the hf extra: pip install "brier[hf]"') from e
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise BrierError("batch_size must be a positive int")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or ("bfloat16" if device.startswith("cuda") else "float32")
        if dtype not in _DTYPES:
            raise BrierError(f"dtype must be one of {_DTYPES}")

        self.model_id = model_id
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._torch: Any = torch
        self.tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=False
        )
        self.model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=getattr(torch, dtype),
        )
        self.model.to(device).eval()
        self._decoder: Any = self.model.get_decoder()
        self.layers: Any = self._decoder.layers
        self.num_layers: int = len(self.layers)
        self._prefix, self._suffix = self._split_template()
        pad = self.tokenizer.pad_token_id
        pad = pad if pad is not None else self.tokenizer.eos_token_id
        if pad is None:
            raise BrierError(f"{model_id} tokenizer has neither a pad nor an eos token")
        self._pad_id: int = pad
        context = getattr(self.model.config, "max_position_embeddings", None) or max_prompt_tokens
        self.max_prompt_tokens: int = min(max_prompt_tokens, context)

    def _split_template(self) -> tuple[list[int], list[int]]:
        """Token ids before and after the user message (suffix ends with the answer prefix)."""
        if not getattr(self.tokenizer, "chat_template", None):
            raise BrierError(f"{self.model_id} has no chat template")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _SENTINEL},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(text, str) or text.count(_SENTINEL) != 1:
            raise BrierError("chat template must contain the user message exactly once")
        before, after = text.split(_SENTINEL)
        return self._ids(before), self._ids(after + ANSWER_PREFIX)

    def _ids(self, text: str, *, split_special: bool = False) -> list[int]:
        ids: list[int] = self.tokenizer(
            text, add_special_tokens=False, split_special_tokens=split_special
        )["input_ids"]
        return ids

    def encode(self, prompt: str) -> list[int]:
        """Token ids of the full chat prompt for one user message.

        Special-token text inside the user message (e.g. ``<|im_end|>``) is tokenised
        as plain text, so state content cannot inject control tokens.

        Raises
        ------
        InputTooLargeError
            If the encoded prompt exceeds ``max_prompt_tokens`` (never truncated).
        """
        if len(prompt) > self.max_prompt_tokens * _MAX_CHARS_PER_TOKEN:
            raise InputTooLargeError("prompt is too long")
        ids = self._prefix + self._ids(prompt, split_special=True) + self._suffix
        if len(ids) > self.max_prompt_tokens:
            raise InputTooLargeError(
                f"prompt is {len(ids)} tokens, limit is {self.max_prompt_tokens}"
            )
        return ids

    def label_token_ids(self, labels: Sequence[str]) -> list[int]:
        """Token id of each label as it appears right after ``"Answer:"`` (``" A"``)."""
        base = self._ids(ANSWER_PREFIX)
        out = []
        for label in labels:
            ids = self._ids(f"{ANSWER_PREFIX} {label}")
            if len(ids) != len(base) + 1 or ids[: len(base)] != base:
                raise TokenizationError(f"label {label!r} is not a single token after the prefix")
            out.append(ids[-1])
        if len(set(out)) != len(out):
            raise TokenizationError("labels must map to distinct tokens")
        return out

    def _batches(self, prompts: Sequence[str]) -> Any:
        """Yield left-padded ``(input_ids, attention_mask, position_ids)`` tensors."""
        torch = self._torch
        for start in range(0, len(prompts), self.batch_size):
            encoded = [self.encode(p) for p in prompts[start : start + self.batch_size]]
            width = max(len(e) for e in encoded)
            ids = torch.tensor([[self._pad_id] * (width - len(e)) + e for e in encoded])
            mask = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in encoded])
            # Models default to arange positions, which ignores left padding.
            pos = (mask.cumsum(-1) - 1).clamp(min=0)
            yield ids.to(self.device), mask.to(self.device), pos.to(self.device)

    def label_logprobs(
        self, prompts: Sequence[str], token_ids: Sequence[int]
    ) -> npt.NDArray[np.float64]:
        """``(n_prompts, n_labels)`` full-vocabulary log-probs gathered at ``token_ids``."""
        torch = self._torch
        index = torch.tensor(list(token_ids), device=self.device)
        rows = []
        with torch.inference_mode():
            for ids, mask, pos in self._batches(prompts):
                out = self.model(
                    input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=1
                )
                logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                rows.append(logp[:, index].double().cpu().numpy())
        result: npt.NDArray[np.float64] = (
            np.concatenate(rows) if rows else np.empty((0, len(token_ids)))
        )
        return result

    def hidden_states(
        self, prompts: Sequence[str], layers: Sequence[int]
    ) -> npt.NDArray[np.float32]:
        """``(n_prompts, n_layers, d)`` residual stream after each block, last token.

        The forward pass stops after the deepest requested block.
        """
        if not layers or not all(isinstance(b, int) and 0 <= b < self.num_layers for b in layers):
            raise BrierError(f"layers must be non-empty ints in [0, {self.num_layers})")
        torch = self._torch
        wanted = sorted(set(layers))
        deepest = wanted[-1]
        captured: dict[int, Any] = {}

        def hook(block: int) -> Any:
            def fn(_module: Any, _args: Any, output: Any) -> None:
                h = output[0] if isinstance(output, tuple) else output
                captured[block] = h[:, -1, :].float().cpu()
                if block == deepest:
                    raise _StopForward

            return fn

        handles = [self.layers[b].register_forward_hook(hook(b)) for b in wanted]
        rows = []
        try:
            with torch.inference_mode():
                for ids, mask, pos in self._batches(prompts):
                    captured.clear()
                    with contextlib.suppress(_StopForward):
                        self._decoder(
                            input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False
                        )
                    rows.append(torch.stack([captured[b] for b in layers], dim=1).numpy())
        finally:
            for handle in handles:
                handle.remove()
        if not rows:
            return np.empty((0, len(layers), self.model.config.hidden_size), dtype=np.float32)
        result: npt.NDArray[np.float32] = np.concatenate(rows).astype(np.float32)
        return result
