"""Score adapters: convert raw model signals to bounded [0, 1] confidence scores.

All adapters return ``float64`` NumPy arrays strictly bounded in ``[0.0, 1.0]``
and reject ``NaN`` inputs. The library never touches text or PII — only these
bounded numeric scores.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit

from cfar.exceptions import ValidationError

__all__ = ["from_logits", "from_cosine", "from_logprobs"]


def _as_array(values: ArrayLike) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"scores must be numeric: {exc}") from exc
    if arr.size and bool(np.isnan(arr).any()):
        raise ValidationError("scores contain NaN values")
    return arr


def _bounded(arr: np.ndarray) -> np.ndarray:
    """Clip to [0, 1] and guard against NaN produced by +Inf/-Inf mixtures."""
    out = np.clip(arr, 0.0, 1.0)
    if out.size and bool(np.isnan(out).any()):
        raise ValidationError("adapter produced NaN; check for +Inf/-Inf mixtures")
    return out


def from_logits(logits: ArrayLike, temperature: float = 1.0) -> np.ndarray:
    """Map raw logits to probabilities via ``expit(logits / temperature)``.

    Args:
        logits: array-like of unbounded raw logits (±Inf is tolerated and
            clipped to the [0, 1] output range).
        temperature: temperature applied before the sigmoid, must be > 0.

    Returns:
        ``float64`` array strictly bounded in ``[0.0, 1.0]``.
    """
    try:
        t = float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"temperature must be a float: {exc}") from exc
    if not t > 0.0:
        raise ValidationError(f"temperature must be > 0, got {temperature}")
    return _bounded(expit(_as_array(logits) / t))


def from_cosine(similarities: ArrayLike) -> np.ndarray:
    """Map cosine similarities from ``[-1, 1]`` to ``[0, 1]`` via ``(s + 1) / 2``.

    Values outside ``[-1, 1]`` are clipped to the output bounds.
    """
    return _bounded((_as_array(similarities) + 1.0) / 2.0)


def from_logprobs(logprobs: Sequence[Sequence[float]] | ArrayLike) -> np.ndarray:
    """Convert per-token log-probabilities to sequence-level confidence.

    For a 2-D input (one row per sequence) each row is reduced to the geometric
    mean token probability ``exp(mean(logprobs))``; a 1-D input is
    exponentiated element-wise.
    """
    arr = _as_array(logprobs)
    if arr.ndim == 1:
        return _bounded(np.exp(arr))
    if arr.ndim == 2:
        if arr.shape[1] == 0:
            raise ValidationError("logprob sequences must be non-empty")
        return _bounded(np.exp(arr.mean(axis=1)))
    raise ValidationError(f"logprobs must be 1-D or 2-D, got {arr.ndim}-D")
