"""declib: calibrated typed decisions from open LLMs."""

from importlib.metadata import PackageNotFoundError, version

from declib.decision import Decision
from declib.questions import Choice, Noul, Score

try:
    __version__ = version("declib")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["Choice", "Decision", "Noul", "Score", "__version__"]
