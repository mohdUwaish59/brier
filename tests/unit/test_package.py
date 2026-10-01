import declib


def test_version_is_a_string() -> None:
    assert isinstance(declib.__version__, str)
    assert declib.__version__
