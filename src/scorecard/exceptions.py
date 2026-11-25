class DataValidationError(Exception):
    """Raised when a raw file or column is missing or breaks a rule."""


class SelectionError(Exception):
    """Raised when variable selection cannot produce a usable model."""
