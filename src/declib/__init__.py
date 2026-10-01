"""declib: calibrated typed decisions from open LLMs."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("declib")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
