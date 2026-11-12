"""Calibration manifest: serializable route artifacts and the runtime gate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from cfar.exceptions import ValidationError

__all__ = ["RouteArtifact", "CalibrationManifest", "RuntimeGate"]


@dataclass(frozen=True)
class RouteArtifact:
    """Immutable calibration record for a single route."""

    route: str
    threshold: float
    ci_low: float
    ci_high: float
    target_pfa: float
    simulated_pfa: float
    method: str
    n_samples: int
    n_simulations: int
    ks_pvalue: float
    h1_recall: float | None
    h0_quantiles: list[float] = field(repr=False)
    """101-point percentile fingerprint ([0..100]) of the baseline H0 scores."""
    tail_params: dict[str, float]


@dataclass
class CalibrationManifest:
    """Versioned, JSON-serializable collection of per-route calibration artifacts."""

    cfar_version: str
    created_at: str
    """UTC ISO-8601 creation timestamp."""
    system_pfa_budget: float | None
    """Joint Sidak FWER budget, or None when per-route target PFA was used."""
    routes: dict[str, RouteArtifact]

    def to_dict(self) -> dict[str, Any]:
        """Plain (JSON-ready) dictionary representation of the manifest."""
        return {
            "cfar_version": self.cfar_version,
            "created_at": self.created_at,
            "system_pfa_budget": self.system_pfa_budget,
            "routes": {name: asdict(artifact) for name, artifact in self.routes.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationManifest":
        """Rebuild a manifest from its dictionary representation."""
        try:
            routes = {
                name: RouteArtifact(
                    route=str(item["route"]),
                    threshold=float(item["threshold"]),
                    ci_low=float(item["ci_low"]),
                    ci_high=float(item["ci_high"]),
                    target_pfa=float(item["target_pfa"]),
                    simulated_pfa=float(item["simulated_pfa"]),
                    method=str(item["method"]),
                    n_samples=int(item["n_samples"]),
                    n_simulations=int(item["n_simulations"]),
                    ks_pvalue=float(item["ks_pvalue"]),
                    h1_recall=(
                        None if item.get("h1_recall") is None else float(item["h1_recall"])
                    ),
                    h0_quantiles=[float(v) for v in item["h0_quantiles"]],
                    tail_params={
                        str(k): float(v) for k, v in item.get("tail_params", {}).items()
                    },
                )
                for name, item in data["routes"].items()
            }
            return cls(
                cfar_version=str(data["cfar_version"]),
                created_at=str(data["created_at"]),
                system_pfa_budget=(
                    None
                    if data.get("system_pfa_budget") is None
                    else float(data["system_pfa_budget"])
                ),
                routes=routes,
            )
        except KeyError as exc:
            raise ValidationError(f"manifest is missing required field: {exc}") from exc

    def save(self, path: str | Path) -> Path:
        """Serialize the manifest to a JSON file and return its path."""
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationManifest":
        """Load a manifest from a JSON file."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValidationError(f"cannot read manifest file {str(path)!r}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"manifest file {str(path)!r} is not valid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


class RuntimeGate:
    """Latency-critical route gate backed by a pre-cached threshold dict.

    Lookup is a plain ``dict`` access per route, keeping ``evaluate()`` in the
    microsecond range. Scores for routes absent from the manifest are ignored.
    """

    def __init__(self, manifest: CalibrationManifest) -> None:
        self._manifest = manifest
        self._thresholds: dict[str, float] = {
            route: float(artifact.threshold)
            for route, artifact in manifest.routes.items()
        }

    @property
    def manifest(self) -> CalibrationManifest:
        return self._manifest

    @property
    def routes(self) -> tuple[str, ...]:
        return tuple(self._thresholds)

    @classmethod
    def from_manifest(cls, path: str | Path) -> "RuntimeGate":
        """Build a gate directly from a manifest JSON file."""
        return cls(CalibrationManifest.load(path))

    def evaluate(
        self, scores: Mapping[str, float], min_margin: float = 0.0
    ) -> list[str]:
        """Return triggered routes sorted by descending score.

        A route triggers when its score exceeds the calibrated threshold.
        When ``min_margin > 0``, only routes within ``min_margin`` of the top
        triggered score are kept.

        Args:
            scores: mapping of route name to a confidence score in ``[0, 1]``.
            min_margin: minimum score relative to the top triggered route.

        Returns:
            Names of triggered routes, best score first.

        Raises:
            ValidationError: on non-numeric or out-of-range scores.
        """
        if min_margin < 0.0:
            raise ValidationError(f"min_margin must be >= 0, got {min_margin}")
        thresholds = self._thresholds
        triggered: list[tuple[str, float]] = []
        for route, score in scores.items():
            threshold = thresholds.get(route)
            if threshold is None:
                continue
            try:
                value = float(score)
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    f"score for route {route!r} must be numeric: {exc}"
                ) from exc
            if value != value or value < 0.0 or value > 1.0:
                raise ValidationError(
                    f"score for route {route!r} must be a finite value in [0, 1], "
                    f"got {score!r}"
                )
            if value > threshold:
                triggered.append((route, value))
        if not triggered:
            return []
        triggered.sort(key=lambda item: item[1], reverse=True)
        if min_margin > 0.0:
            floor = triggered[0][1] - min_margin
            triggered = [item for item in triggered if item[1] >= floor]
        return [route for route, _ in triggered]
