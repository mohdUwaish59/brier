"""Exceptions raised by declib. All derive from :class:`DeclibError`."""


class DeclibError(Exception):
    """Base class for every error raised by declib."""


class QuestionError(DeclibError, ValueError):
    """A question (or set of questions) is malformed."""


class InputTooLargeError(DeclibError, ValueError):
    """An input exceeds a configured size limit."""


class TokenizationError(DeclibError):
    """A label is not a single, distinct token where it is read."""


class NotFittedError(DeclibError, RuntimeError):
    """A correction level was requested before it was fitted."""


class InsufficientDataError(DeclibError, ValueError):
    """Too few (labelled) items to fit a correction level."""


class ArtifactError(DeclibError):
    """A calibration artifact is invalid or does not match the backend."""
