"""Tests for cfar.monitor — drift detection against manifest baselines."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from cfar import (
    CalibrationManifest,
    DriftMonitor,
    MultiRouteCalibrator,
    RouteArtifact,
    ValidationError,
)


def baseline_scores(n: int = 3000, seed: int = 71) -> np.ndarray:
    return expit(np.random.default_rng(seed).normal(0.0, 1.0, size=n))


def make_manifest() -> CalibrationManifest:
    scores = baseline_scores()
    fingerprint = [float(v) for v in np.percentile(scores, np.linspace(0.0, 100.0, 101))]
    artifact = RouteArtifact(
        route="router",
        threshold=0.8382,
        ci_low=0.82,
        ci_high=0.86,
        target_pfa=0.05,
        simulated_pfa=0.05,
        method="evt_montecarlo",
        n_samples=3000,
        n_simulations=100_000,
        ks_pvalue=0.4,
        h1_recall=0.95,
        h0_quantiles=fingerprint,
        tail_params={"xi": -0.2, "sigma": 0.6, "u": 1.03, "tail_fraction": 0.15},
    )
    return CalibrationManifest(
        cfar_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
        system_pfa_budget=None,
        routes={"router": artifact},
    )


class TestDriftMonitor:
    def test_stationary_scores_report_no_drift(self):
        monitor = DriftMonitor(make_manifest())
        report = monitor.check("router", baseline_scores(n=500, seed=99))
        assert report.route == "router"
        assert report.drift_detected is False
        assert report.ks_pvalue > 0.01
        assert 0.0 <= report.ks_statistic <= 1.0
        assert report.wasserstein_distance >= 0.0
        assert report.baseline_median == pytest.approx(0.5, abs=0.05)
        assert report.live_median == pytest.approx(0.5, abs=0.05)

    def test_shifted_scores_report_drift(self):
        monitor = DriftMonitor(make_manifest())
        live = expit(np.random.default_rng(100).normal(1.0, 1.0, size=800))
        report = monitor.check("router", live)
        assert report.drift_detected is True
        assert report.ks_pvalue < 0.01
        assert report.live_median > report.baseline_median
        assert report.wasserstein_distance > 0.05

    def test_input_validation(self):
        monitor = DriftMonitor(make_manifest())
        with pytest.raises(ValidationError):
            monitor.check("router", baseline_scores(n=19, seed=1))
        with pytest.raises(ValidationError):
            monitor.check("unknown", baseline_scores(n=50, seed=1))
        nan_scores = baseline_scores(n=50, seed=2)
        nan_scores[0] = np.nan
        with pytest.raises(ValidationError):
            monitor.check("router", nan_scores)
        out_of_range = baseline_scores(n=50, seed=3)
        out_of_range[0] = 1.5
        with pytest.raises(ValidationError):
            monitor.check("router", out_of_range)

    def test_invalid_alpha_rejected(self):
        with pytest.raises(ValidationError):
            DriftMonitor(make_manifest(), alpha=0.0)

    def test_from_manifest(self, tmp_path):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(
            {"router": baseline_scores(seed=71)}
        )
        path = evidence.save_manifest(tmp_path / "manifest.json")
        monitor = DriftMonitor.from_manifest(path, alpha=0.01)
        report = monitor.check("router", baseline_scores(n=300, seed=72))
        assert report.drift_detected is False
