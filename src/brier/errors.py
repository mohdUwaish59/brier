"""Exceptions raised by brier. All derive from :class:`BrierError`."""


class BrierError(Exception):
    """Base class for every error raised by brier."""


class QuestionError(BrierError, ValueError):
    """A question (or set of questions) is malformed."""


class InputTooLargeError(BrierError, ValueError):
    """An input exceeds a configured size limit."""


class TokenizationError(BrierError):
    """A label is not a single, distinct token where it is read."""


class NotFittedError(BrierError, RuntimeError):
    """A correction level was requested before it was fitted."""


class InsufficientDataError(BrierError, ValueError):
    """Too few (labelled) items to fit a correction level."""


class ArtifactError(BrierError):
    """A calibration artifact is invalid or does not match the backend."""
