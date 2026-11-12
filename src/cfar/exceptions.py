"""Custom exception hierarchy for cfar."""

from __future__ import annotations

__all__ = ["CFARError", "ValidationError", "CalibrationError"]


class CFARError(Exception):
    """Base class for all cfar errors."""


class ValidationError(CFARError):
    """Raised when user-supplied input violates the cfar contract."""


class CalibrationError(CFARError):
    """Raised when statistical calibration cannot produce a valid cutoff."""
