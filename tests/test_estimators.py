"""Tests for cfar.estimators — validation, accuracy, determinism, speed."""

from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit

from cfar import CalibrationError, EstimationResult, ValidationError, estimate_threshold


def beta_scores(n: int = 4000, a: float = 2.0, b: float = 8.0, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).beta(a, b, size=n)


def logit_normal_scores(
    n: int = 4000, mu: float = 0.0, sigma: float = 1.0, seed: int = 7
) -> np.ndarray:
    return expit(np.random.default_rng(seed).normal(mu, sigma, size=n))


class TestValidation:
    def test_too_few_samples(self):
        with pytest.raises(ValidationError):
            estimate_threshold(np.full(29, 0.5), 0.05)

    def test_nan_rejected(self):
        scores = beta_scores()
        scores[0] = np.nan
        with pytest.raises(ValidationError):
            estimate_threshold(scores, 0.05)

    def test_out_of_range_rejected(self):
        scores = beta_scores()
        scores[0] = 1.5
        with pytest.raises(ValidationError):
            estimate_threshold(scores, 0.05)

    def test_pfa_bounds(self):
        scores = beta_scores()
        for bad in (0.0, 0.5, -0.1, 1.0):
            with pytest.raises(ValidationError):
                estimate_threshold(scores, bad)

    def test_unknown_method(self):
        with pytest.raises(ValidationError):
            estimate_threshold(beta_scores(), 0.05, method="gumbel")

    def test_constant_scores_rejected(self):
        with pytest.raises(ValidationError):
            estimate_threshold(np.full(100, 0.42), 0.05)

    def test_2d_rejected(self):
        with pytest.raises(ValidationError):
            estimate_threshold(np.zeros((40, 2)), 0.05)


class TestEmpirical:
    def test_tracks_target_pfa_on_beta_data(self):
        scores = beta_scores()
        result = estimate_threshold(scores, 0.05, method="empirical")
        analytic = stats.beta.ppf(0.95, 2.0, 8.0)
        assert isinstance(result, EstimationResult)
        assert result.method == "empirical"
        assert result.threshold == pytest.approx(analytic, abs=0.02)
        assert result.simulated_pfa == pytest.approx(0.05, abs=0.02)
        assert result.ci_low <= result.threshold <= result.ci_high
        assert result.n_samples == 4000
        assert result.n_simulations == 0
        assert result.ks_pvalue == 1.0
        assert result.tail_params == {}

    def test_extreme_pfa(self):
        scores = beta_scores(n=20000, seed=3)
        result = estimate_threshold(scores, 0.01, method="empirical")
        assert result.threshold == pytest.approx(stats.beta.ppf(0.99, 2.0, 8.0), abs=0.02)


class TestBetaMonteCarlo:
    def test_tracks_target_pfa_on_beta_data(self):
        scores = beta_scores()
        result = estimate_threshold(scores, 0.05, method="beta_montecarlo")
        analytic = stats.beta.ppf(0.95, 2.0, 8.0)
        assert result.method == "beta_montecarlo"
        assert result.threshold == pytest.approx(analytic, abs=0.02)
        assert result.simulated_pfa == pytest.approx(0.05, abs=0.01)
        assert result.tail_params["a"] == pytest.approx(2.0, abs=0.4)
        assert result.tail_params["b"] == pytest.approx(8.0, abs=1.0)
        assert result.ks_pvalue > 0.05
        assert result.n_simulations == 100_000
        assert result.mc_samples.size == 100_000
        assert result.mc_samples.min() >= 0.0
        assert result.mc_samples.max() <= 1.0


class TestEVTMonteCarlo:
    def test_tracks_target_pfa_on_logit_normal_data(self):
        scores = logit_normal_scores()
        result = estimate_threshold(scores, 0.05)
        analytic = expit(stats.norm.ppf(0.95))
        assert result.method == "evt_montecarlo"
        assert result.threshold == pytest.approx(analytic, abs=0.02)
        assert result.simulated_pfa == pytest.approx(0.05, abs=0.01)
        assert set(result.tail_params) == {"xi", "sigma", "u", "tail_fraction"}
        assert 0.0 <= result.ks_pvalue <= 1.0
        assert result.ci_low <= result.ci_high
        assert result.n_simulations == 100_000
        assert result.mc_samples.min() >= 0.0 and result.mc_samples.max() <= 1.0

    def test_threshold_stays_bounded_on_spiky_data(self):
        rng = np.random.default_rng(3)
        scores = np.concatenate([rng.beta(1.5, 8.0, 3500), rng.uniform(0.9, 1.0, 500)])
        result = estimate_threshold(scores, 0.01)
        assert 0.0 < result.threshold <= 1.0
        assert abs(result.simulated_pfa - 0.01) < 0.005

    def test_degenerate_tail_raises(self):
        scores = np.concatenate([np.full(90, 0.1), np.full(10, 0.9)])
        with pytest.raises(CalibrationError):
            estimate_threshold(scores, 0.05)

    def test_fallback_quantile_for_short_tails(self):
        scores = logit_normal_scores(n=40, seed=5)
        result = estimate_threshold(scores, 0.05, n_simulations=2000, n_bootstrap=100)
        assert 0.0 < result.threshold < 1.0

    def test_deterministic_given_seed(self):
        scores = logit_normal_scores()
        first = estimate_threshold(scores, 0.05, random_state=123)
        second = estimate_threshold(scores, 0.05, random_state=123)
        assert first.threshold == second.threshold
        assert first.ci_low == second.ci_low
        assert first.ci_high == second.ci_high
        other = estimate_threshold(scores, 0.05, random_state=124)
        assert other.threshold != first.threshold

    def test_accepts_generator_random_state(self):
        scores = logit_normal_scores(n=500, seed=9)
        result = estimate_threshold(scores, 0.05, random_state=np.random.default_rng(0))
        assert 0.0 < result.threshold < 1.0

    def test_calibration_speed(self):
        scores = logit_normal_scores(n=2000, seed=2)
        start = time.perf_counter()
        estimate_threshold(scores, 0.05)
        assert time.perf_counter() - start < 2.0
