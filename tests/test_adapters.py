"""Tests for cfar.adapters — bounds, edge cases, and validation."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import expit

from cfar import ValidationError, from_cosine, from_logits, from_logprobs


class TestFromLogits:
    def test_zero_logit_maps_to_half(self):
        out = from_logits([0.0])
        assert out.dtype == np.float64
        assert out[0] == pytest.approx(0.5)

    def test_bounds_and_extremes(self):
        out = from_logits([-100.0, -10.0, 0.0, 10.0, 100.0])
        assert np.all(out >= 0.0) and np.all(out <= 1.0)
        assert out[0] < 1e-40
        assert out[4] == 1.0
        assert out[1] < 1e-4
        assert out[3] > 1.0 - 1e-4

    def test_matches_expit(self):
        logits = np.linspace(-5.0, 5.0, 101)
        np.testing.assert_allclose(from_logits(logits), expit(logits))

    def test_temperature_controls_sharpness(self):
        logits = np.array([2.0, -2.0])
        hot = from_logits(logits, temperature=10.0)
        cold = from_logits(logits, temperature=0.1)
        assert np.all(np.abs(hot - 0.5) < np.abs(cold - 0.5))

    def test_infinite_logits_stay_bounded(self):
        out = from_logits([np.inf, -np.inf])
        assert out.tolist() == [1.0, 0.0]

    def test_nonpositive_temperature_rejected(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValidationError):
                from_logits([0.0], temperature=bad)

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            from_logits([0.0, float("nan")])

    def test_accepts_python_lists(self):
        out = from_logits([0.0, 1.0])
        assert out.shape == (2,)


class TestFromCosine:
    def test_endpoints_and_midpoint(self):
        out = from_cosine([-1.0, 0.0, 1.0])
        assert out.tolist() == [0.0, 0.5, 1.0]

    def test_out_of_range_clipped(self):
        out = from_cosine([-3.0, 2.0])
        assert out.tolist() == [0.0, 1.0]

    def test_linear_mapping(self):
        assert from_cosine([0.25])[0] == pytest.approx(0.625)

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            from_cosine([0.3, float("nan")])


class TestFromLogprobs:
    def test_1d_exponentiates(self):
        out = from_logprobs([0.0, math.log(0.5), -200.0])
        assert out[0] == pytest.approx(1.0)
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(0.0, abs=1e-12)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_2d_geometric_mean_per_sequence(self):
        lp = [
            [math.log(0.5), math.log(0.5)],
            [math.log(0.25), math.log(1.0)],
        ]
        out = from_logprobs(lp)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(0.5)  # sqrt(0.25 * 1.0)

    def test_2d_matches_manual_geometric_mean(self):
        rng = np.random.default_rng(0)
        lp = rng.uniform(-6.0, -0.1, size=(50, 12))
        expected = np.exp(lp.mean(axis=1))
        np.testing.assert_allclose(from_logprobs(lp), expected)

    def test_nan_rejected(self):
        with pytest.raises(ValidationError):
            from_logprobs([0.0, float("nan")])

    def test_empty_sequences_rejected(self):
        with pytest.raises(ValidationError):
            from_logprobs(np.empty((2, 0)))

    def test_3d_rejected(self):
        with pytest.raises(ValidationError):
            from_logprobs(np.zeros((2, 2, 2)))
