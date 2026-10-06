"""brier: calibrated typed decisions from open LLMs."""

import logging

from brier._version import __version__
from brier.decider import Decider
from brier.decision import Decision
from brier.questions import Choice, Noul, Score

__all__ = ["Choice", "Decider", "Decision", "Noul", "Score", "__version__"]

# A library never configures logging; the application decides what to show (SPEC, Logging).
logging.getLogger(__name__).addHandler(logging.NullHandler())
