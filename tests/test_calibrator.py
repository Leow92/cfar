"""Tests for cfar.calibrator — budgeting, recall, advisories, reports, gate."""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
from scipy.special import expit

from cfar import (
    CalibrationManifest,
    MultiRouteCalibrator,
    RuntimeGate,
    ThresholdCalibrator,
    ValidationError,
    estimate_threshold,
)

ROUTES = ("intent_router", "guardrail")


def h0(n: int = 2000, seed: int = 11) -> np.ndarray:
    return expit(np.random.default_rng(seed).normal(0.0, 1.0, size=n))


def h1(n: int = 2000, seed: int = 12) -> np.ndarray:
    return expit(np.random.default_rng(seed).normal(3.5, 1.0, size=n))


def make_h0_scores() -> dict[str, np.ndarray]:
    return {"intent_router": h0(seed=11), "guardrail": h0(seed=13)}


class TestMultiRouteCalibrator:
    def test_per_route_target_pfa(self):
        calibrator = MultiRouteCalibrator(
            target_pfa={"intent_router": 0.05, "guardrail": 0.01}
        )
        evidence = calibrator.calibrate(make_h0_scores())
        assert set(evidence.thresholds) == set(ROUTES)
        assert 0.0 < evidence.thresholds["intent_router"] < 1.0
        assert evidence.thresholds["guardrail"] > evidence.thresholds["intent_router"]
        assert evidence.manifest.routes["intent_router"].target_pfa == pytest.approx(0.05)

    def test_scalar_target_pfa_applies_to_all_routes(self):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(make_h0_scores())
        for artifact in evidence.manifest.routes.values():
            assert artifact.target_pfa == pytest.approx(0.05)

    def test_pfa_arguments_are_mutually_exclusive(self):
        with pytest.raises(ValidationError):
            MultiRouteCalibrator(target_pfa=0.05, system_pfa=0.05)
        with pytest.raises(ValidationError):
            MultiRouteCalibrator()

    def test_invalid_pfa_values_rejected(self):
        with pytest.raises(ValidationError):
            MultiRouteCalibrator(target_pfa=0.7)
        with pytest.raises(ValidationError):
            MultiRouteCalibrator(system_pfa=-0.1)

    def test_sidak_family_wise_budget(self):
        evidence = MultiRouteCalibrator(system_pfa=0.05).calibrate(make_h0_scores())
        assert evidence.manifest.system_pfa_budget == pytest.approx(0.05)
        expected = 1.0 - (1.0 - 0.05) ** (1.0 / len(ROUTES))
        for artifact in evidence.manifest.routes.values():
            assert artifact.target_pfa == pytest.approx(expected)
        solo = MultiRouteCalibrator(system_pfa=0.05).calibrate({"solo": h0(seed=21)})
        assert solo.manifest.routes["solo"].target_pfa == pytest.approx(0.05)

    def test_h1_recall_matches_manual_computation(self):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(
            make_h0_scores(), h1_scores={"intent_router": h1()}
        )
        artifact = evidence.manifest.routes["intent_router"]
        manual = float(np.mean(h1() > artifact.threshold))
        assert artifact.h1_recall == pytest.approx(manual)
        assert artifact.h1_recall > 0.90
        assert not any(
            "recall" in a.lower() for a in evidence.advisories["intent_router"]
        )
        assert evidence.manifest.routes["guardrail"].h1_recall is None

    def test_low_recall_advisory(self):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(
            {"router": h0(seed=31)}, h1_scores={"router": h0(seed=32)}
        )
        assert evidence.manifest.routes["router"].h1_recall < 0.90
        assert any("recall" in a.lower() for a in evidence.advisories["router"])

    def test_ks_advisory_on_misspecified_fit(self):
        rng = np.random.default_rng(41)
        bimodal = np.concatenate(
            [rng.uniform(0.0, 0.35, 2000), rng.uniform(0.65, 1.0, 2000)]
        )
        evidence = MultiRouteCalibrator(
            target_pfa=0.05, method="beta_montecarlo"
        ).calibrate({"bimodal": bimodal})
        assert evidence.manifest.routes["bimodal"].ks_pvalue < 0.05
        assert any("ks" in a.lower() for a in evidence.advisories["bimodal"])

    def test_summary_table(self):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(
            make_h0_scores(), h1_scores={"intent_router": h1()}
        )
        text = evidence.summary()
        assert "Route" in text and "Target PFA" in text and "H1 Recall" in text
        for route in ROUTES:
            assert route in text

    def test_manifest_json_roundtrip(self, tmp_path):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(make_h0_scores())
        path = evidence.save_manifest(tmp_path / "manifest.json")
        assert path.exists()
        loaded = CalibrationManifest.load(path)
        assert loaded == evidence.manifest
        raw = json.loads(path.read_text())
        assert raw["cfar_version"] == "0.1.0"
        assert raw["system_pfa_budget"] is None
        assert len(raw["routes"]["intent_router"]["h0_quantiles"]) == 101

    def test_export_report_is_self_contained(self, tmp_path):
        evidence = MultiRouteCalibrator(target_pfa=0.05).calibrate(
            make_h0_scores(), h1_scores={"intent_router": h1()}
        )
        path = evidence.export_report(tmp_path / "report.html")
        assert path.exists()
        report = path.read_text()
        assert report.startswith("<!DOCTYPE html>")
        assert "<svg" in report
        assert "intent_router" in report
        assert "</html>" in report
        for forbidden in ("<script", "href=", "src="):
            assert forbidden not in report


class TestThresholdCalibrator:
    def test_single_route_calibration(self):
        evidence = ThresholdCalibrator(target_pfa=0.05).calibrate(
            h0(seed=51), h1_scores=h1(seed=52)
        )
        assert set(evidence.thresholds) == {ThresholdCalibrator.default_route}
        assert evidence.manifest.routes["default"].h1_recall is not None

    def test_matches_direct_estimator(self):
        scores = h0(seed=53)
        evidence = ThresholdCalibrator(target_pfa=0.05).calibrate(scores)
        direct = estimate_threshold(scores, 0.05)
        assert evidence.thresholds["default"] == pytest.approx(direct.threshold)


class TestRuntimeGate:
    def _evidence(self):
        calibrator = MultiRouteCalibrator(target_pfa={"a": 0.05, "b": 0.05})
        return calibrator.calibrate({"a": h0(seed=61), "b": h0(seed=62)})

    def test_trigger_and_ranking(self):
        evidence = self._evidence()
        gate = RuntimeGate(evidence.manifest)
        ta, tb = evidence.thresholds["a"], evidence.thresholds["b"]
        assert gate.evaluate({"a": ta + 0.01, "b": tb + 0.02}) == ["b", "a"]
        assert gate.evaluate({"a": ta - 0.01, "b": tb - 0.01}) == []

    def test_min_margin_filters_runner_ups(self):
        evidence = self._evidence()
        gate = RuntimeGate(evidence.manifest)
        ta, tb = evidence.thresholds["a"], evidence.thresholds["b"]
        scores = {"a": ta + 0.001, "b": tb + 0.10}
        assert gate.evaluate(scores, min_margin=0.05) == ["b"]
        assert gate.evaluate(scores, min_margin=0.0) == ["b", "a"]

    def test_unknown_routes_ignored(self):
        gate = RuntimeGate(self._evidence().manifest)
        assert gate.evaluate({"unknown_route": 0.99}) == []

    def test_invalid_scores_rejected(self):
        gate = RuntimeGate(self._evidence().manifest)
        with pytest.raises(ValidationError):
            gate.evaluate({"a": 1.5})
        with pytest.raises(ValidationError):
            gate.evaluate({"a": float("nan")})
        with pytest.raises(ValidationError):
            gate.evaluate({"a": 0.5}, min_margin=-0.1)

    def test_from_manifest(self, tmp_path):
        evidence = self._evidence()
        path = evidence.save_manifest(tmp_path / "manifest.json")
        gate = RuntimeGate.from_manifest(path)
        ta = evidence.thresholds["a"]
        assert gate.evaluate({"a": ta + 0.01}) == ["a"]

    def test_lookup_latency(self):
        evidence = self._evidence()
        gate = RuntimeGate(evidence.manifest)
        scores = {
            "a": evidence.thresholds["a"] + 0.01,
            "b": evidence.thresholds["b"] + 0.01,
        }
        gate.evaluate(scores)  # warm-up
        iterations = 2000
        start = time.perf_counter()
        for _ in range(iterations):
            gate.evaluate(scores)
        average = (time.perf_counter() - start) / iterations
        assert average < 1e-4  # < 0.1 ms per evaluate() call
