"""cfar — Constant False Alarm Rate calibration and gating for AI systems."""

from __future__ import annotations

# NOTE: __version__ must be defined before the submodule imports below;
# cfar.calibrator reads it at import time.
__version__ = "0.1.0"

from cfar.adapters import from_cosine, from_logits, from_logprobs
from cfar.calibrator import CalibrationEvidence, MultiRouteCalibrator, ThresholdCalibrator
from cfar.exceptions import CFARError, CalibrationError, ValidationError
from cfar.estimators import EstimationMethod, EstimationResult, estimate_threshold
from cfar.manifest import CalibrationManifest, RouteArtifact, RuntimeGate
from cfar.monitor import DriftMonitor, DriftReport

__all__ = [
    "CFARError",
    "CalibrationError",
    "ValidationError",
    "from_logits",
    "from_cosine",
    "from_logprobs",
    "EstimationMethod",
    "EstimationResult",
    "estimate_threshold",
    "RouteArtifact",
    "CalibrationManifest",
    "RuntimeGate",
    "CalibrationEvidence",
    "ThresholdCalibrator",
    "MultiRouteCalibrator",
    "DriftMonitor",
    "DriftReport",
    "__version__",
]
