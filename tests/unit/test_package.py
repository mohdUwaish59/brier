import brier


def test_version_is_a_string() -> None:
    assert isinstance(brier.__version__, str)
    assert brier.__version__
