import string

import pytest
from hypothesis import given
from hypothesis import strategies as st

from brier import Choice, Noul, Score
from brier.errors import BrierError, QuestionError
from brier.questions import validate_questions


def test_choice_happy_path_stores_options_as_tuple() -> None:
    q = Choice("Which team?", ["billing", "technical"], name="route")
    assert q.text == "Which team?"
    assert q.options == ("billing", "technical")
    assert q.name == "route"


@pytest.mark.parametrize("k", [2, 26])
def test_choice_accepts_option_count_bounds(k: int) -> None:
    q = Choice("q", list(string.ascii_lowercase[:k]), name="c")
    assert len(q.options) == k


@pytest.mark.parametrize("k", [0, 1, 27])
def test_choice_rejects_option_count_outside_range(k: int) -> None:
    opts = [f"o{i}" for i in range(k)]
    with pytest.raises(QuestionError):
        Choice("q", opts, name="c")


@pytest.mark.parametrize(
    "options",
    [["a", "a"], ["a", ""], ["a", "   "], ["a", "x" * 501]],
    ids=["duplicate", "empty", "blank", "too-long"],
)
def test_choice_rejects_bad_options(options: list[str]) -> None:
    with pytest.raises(QuestionError):
        Choice("q", options, name="c")


def test_choice_accepts_500_char_option() -> None:
    Choice("q", ["a", "x" * 500], name="c")


def test_choice_rejects_bare_string_as_options() -> None:
    # A str is a sequence of chars; "ab" must not silently become ("a", "b").
    with pytest.raises(QuestionError):
        Choice("q", "ab", name="c")  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["", "   \n"])
def test_empty_text_rejected_for_every_type(text: str) -> None:
    with pytest.raises(QuestionError):
        Choice(text, ["a", "b"], name="c")
    with pytest.raises(QuestionError):
        Noul(text, name="n")
    with pytest.raises(QuestionError):
        Score(text, levels=3, name="s")


@pytest.mark.parametrize("name", ["", "  "])
def test_empty_name_rejected(name: str) -> None:
    with pytest.raises(QuestionError):
        Noul("q", name=name)


def test_non_string_text_rejected() -> None:
    with pytest.raises(QuestionError):
        Noul(123, name="n")  # type: ignore[arg-type]


def test_question_error_is_a_brier_error() -> None:
    assert issubclass(QuestionError, BrierError)


def test_questions_are_frozen() -> None:
    q = Noul("q", name="n")
    with pytest.raises(AttributeError):
        q.text = "other"  # type: ignore[misc]


@pytest.mark.parametrize("levels", [2, 10])
def test_score_accepts_level_bounds(levels: int) -> None:
    assert Score("q", levels=levels, name="s").levels == levels


@pytest.mark.parametrize("levels", [1, 11, 0, -3])
def test_score_rejects_levels_outside_range(levels: int) -> None:
    with pytest.raises(QuestionError):
        Score("q", levels=levels, name="s")


@pytest.mark.parametrize("levels", [True, 3.0, "3"])
def test_score_rejects_non_int_levels(levels: object) -> None:
    with pytest.raises(QuestionError):
        Score("q", levels=levels, name="s")  # type: ignore[arg-type]


def test_score_labels_stored_as_tuple() -> None:
    q = Score("q", levels=3, name="s", labels=["low", "mid", "high"])
    assert q.labels == ("low", "mid", "high")


@pytest.mark.parametrize(
    "labels",
    [["low", "high"], ["low", "", "high"], ["low", "mid", "x" * 501]],
    ids=["wrong-length", "empty", "too-long"],
)
def test_score_rejects_bad_labels(labels: list[str]) -> None:
    with pytest.raises(QuestionError):
        Score("q", levels=3, name="s", labels=labels)


def test_validate_questions_accepts_unique_names() -> None:
    validate_questions([Noul("a?", name="a"), Score("b?", levels=3, name="b")])


def test_validate_questions_rejects_duplicate_names() -> None:
    with pytest.raises(QuestionError):
        validate_questions([Noul("a?", name="x"), Noul("b?", name="x")])


@given(
    st.lists(
        st.text(min_size=1, max_size=20).filter(lambda s: s.strip() and s.splitlines() == [s]),
        min_size=2,
        max_size=26,
    )
)
def test_choice_validation_matches_uniqueness(options: list[str]) -> None:
    if len(set(options)) == len(options):
        assert Choice("q", options, name="c").options == tuple(options)
    else:
        with pytest.raises(QuestionError):
            Choice("q", options, name="c")


@pytest.mark.parametrize("brk", ["\n", "\r", "\u2028", "\x85"])
def test_options_and_labels_reject_line_breaks(brk: str) -> None:
    # "billing\nB. sales" would render as a fake "B." line.
    with pytest.raises(QuestionError):
        Choice("q", [f"billing{brk}B. sales", "tech"], name="c")
    with pytest.raises(QuestionError):
        Score("q", levels=2, name="s", labels=[f"low{brk}x", "high"])


def test_question_text_may_contain_newlines() -> None:
    assert Noul("line one\nline two", name="n").text == "line one\nline two"
