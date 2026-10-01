"""Installed package version (its own module so internal imports avoid import cycles)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("brier")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"
