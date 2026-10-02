"""Prompt format selection in HFBackend (ADR-0008), with a stub tokenizer: no torch needed."""

from __future__ import annotations

from typing import Any

import pytest

from brier.backends.hf import _leading_special_ids, _template_texts
from brier.errors import BrierError
from brier.prompts import ANSWER_PREFIX, SYSTEM


class _Tok:
    """Character-level stand-in for a Hugging Face tokenizer."""

    def __init__(
        self, chat_template: str | None = None, bos: int | None = None, eos: int | None = None
    ):
        self.chat_template = chat_template
        self._bos, self._eos = bos, eos

    def __call__(
        self, text: str, add_special_tokens: bool = True, **_: Any
    ) -> dict[str, list[int]]:
        ids = [1000 + ord(c) for c in text]
        if add_special_tokens:
            ids = ([self._bos] if self._bos is not None else []) + ids
            ids += [self._eos] if self._eos is not None else []
        return {"input_ids": ids}

    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        assert self.chat_template is not None
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "".join(parts) + "<assistant>"


def test_no_chat_template_uses_plain_format() -> None:
    fmt, before, after = _template_texts(_Tok())
    assert fmt == "plain"
    assert before == f"{SYSTEM}\n\n"
    assert after == f"\n{ANSWER_PREFIX}"


def test_chat_template_is_used_when_present() -> None:
    fmt, before, after = _template_texts(_Tok(chat_template="x"))
    assert fmt == "chat"
    assert before == f"<system>{SYSTEM}</system><user>"
    assert after == f"</user><assistant>{ANSWER_PREFIX}"


def test_empty_chat_template_counts_as_none() -> None:
    assert _template_texts(_Tok(chat_template=""))[0] == "plain"


def test_chat_template_must_contain_the_user_message_once() -> None:
    class Dup(_Tok):
        def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
            return messages[1]["content"] * 2

    with pytest.raises(BrierError, match="exactly once"):
        _template_texts(Dup(chat_template="x"))


@pytest.mark.parametrize(
    ("bos", "eos", "want"),
    [(None, None, []), (1, None, [1]), (1, 2, [1]), (None, 2, [])],
)
def test_leading_special_ids(bos: int | None, eos: int | None, want: list[int]) -> None:
    assert _leading_special_ids(_Tok(bos=bos, eos=eos)) == want


def test_leading_special_ids_refuses_an_unreadable_tokenizer() -> None:
    from brier.errors import TokenizationError

    class Odd(_Tok):
        def __call__(
            self, text: str, add_special_tokens: bool = True, **_: Any
        ) -> dict[str, list[int]]:
            return {"input_ids": [7] if add_special_tokens else [1000 + ord(c) for c in text]}

    with pytest.raises(TokenizationError, match="special tokens"):
        _leading_special_ids(Odd())
