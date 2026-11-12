"""Runtime drift monitoring against a stored calibration baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from cfar.exceptions import ValidationError
from cfar.manifest import CalibrationManifest

__all__ = ["DriftReport", "DriftMonitor"]

_MIN_LIVE_SAMPLES = 20


@dataclass(frozen=True)
class DriftReport:
    """Result of comparing a live H0 sample against the calibrated baseline."""

    route: str
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    baseline_median: float
    live_median: float
    drift_detected: bool


class DriftMonitor:
    """Two-sample drift monitor over the manifest's H0 percentile fingerprints.

    Compares a live benign sample against the 101-point baseline fingerprint
    stored at calibration time using the two-sample Kolmogorov-Smirnov test and
    the Wasserstein distance. Drift is flagged when the KS p-value falls below
    ``alpha``.
    """

    def __init__(self, manifest: CalibrationManifest, alpha: float = 0.01) -> None:
        try:
            level = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"alpha must be a float: {exc}") from exc
        if not 0.0 < level < 1.0:
            raise ValidationError(f"alpha must satisfy 0 < a < 1, got {alpha}")
        self._manifest = manifest
        self._alpha = level
        self._baselines: dict[str, np.ndarray] = {
            route: np.asarray(artifact.h0_quantiles, dtype=np.float64)
            for route, artifact in manifest.routes.items()
        }

    @property
    def alpha(self) -> float:
        return self._alpha

    @classmethod
    def from_manifest(cls, path: str | Path, alpha: float = 0.01) -> "DriftMonitor":
        """Build a monitor directly from a manifest JSON file."""
        return cls(CalibrationManifest.load(path), alpha=alpha)

    def check(self, route: str, live_h0_scores: ArrayLike) -> DriftReport:
        """Compare a live benign sample against the stored baseline.

        Args:
            route: calibrated route name.
            live_h0_scores: fresh 1-D sample of benign scores (n >= 20) in
                ``[0, 1]``.

        Returns:
            Frozen :class:`DriftReport` with KS and Wasserstein statistics.

        Raises:
            ValidationError: unknown route or invalid live sample.
        """
        if route not in self._baselines:
            raise ValidationError(
                f"unknown route {route!r}; calibrated routes: {sorted(self._baselines)}"
            )
        try:
            live = np.array(live_h0_scores, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"live scores must be numeric: {exc}") from exc
        if live.ndim != 1:
            raise ValidationError(f"live scores must be 1-D, got {live.ndim}-D")
        if live.size < _MIN_LIVE_SAMPLES:
            raise ValidationError(
                f"drift checks need at least {_MIN_LIVE_SAMPLES} live scores, "
                f"got {live.size}"
            )
        if not np.isfinite(live).all():
            raise ValidationError("live scores must be finite (no NaN or +/-Inf)")
        if live.min() < 0.0 or live.max() > 1.0:
            raise ValidationError("live scores must lie within [0.0, 1.0]")
        baseline = self._baselines[route]
        ks = stats.ks_2samp(baseline, live)
        wasserstein = stats.wasserstein_distance(baseline, live)
        return DriftReport(
            route=route,
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            wasserstein_distance=float(wasserstein),
            baseline_median=float(np.median(baseline)),
            live_median=float(np.median(live)),
            drift_detected=bool(ks.pvalue < self._alpha),
        )
