"""brier: calibrated typed decisions from open LLMs."""

from importlib.metadata import PackageNotFoundError, version

from brier.decider import Decider
from brier.decision import Decision
from brier.questions import Choice, Noul, Score

try:
    __version__ = version("brier")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["Choice", "Decider", "Decision", "Noul", "Score", "__version__"]
