"""Statistical threshold (cutoff) estimation from benign H0 score samples.

Given a sample of benign / near-miss (H0) confidence scores, these estimators
derive a cutoff ``gamma`` such that ``P(score > gamma | H0) ~= target_pfa``.

Three strategies are available:

* ``empirical`` -- direct sample quantile plus a vectorized bootstrap CI.
* ``beta_montecarlo`` -- MLE Beta fit of the full score distribution, a KS
  goodness-of-fit check, and a Monte Carlo draw from the fitted law.
* ``evt_montecarlo`` (default) -- peaks-over-threshold Extreme Value Theory.
  Scores are mapped to logit space so tail extrapolation can never leave
  ``[0, 1]``; upper-tail exceedances are modelled with a Generalized Pareto
  Distribution (GPD, ``scipy.stats.genpareto``, ``floc=0``) and a hybrid
  Monte Carlo simulation (empirical body + GPD tail) yields the cutoff.
  Bootstrap CIs refit the GPD with a closed-form probability-weighted
  moments (PWM) estimator inside the loop so calibration stays fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from scipy.special import expit, logit

from cfar.exceptions import CalibrationError, ValidationError

__all__ = ["EstimationMethod", "EstimationResult", "estimate_threshold"]

EstimationMethod = Literal["empirical", "beta_montecarlo", "evt_montecarlo"]

_EPS = 1e-7
_MIN_SAMPLES = 30
_MIN_TAIL_EXCEEDANCES = 10
_MIN_SIMULATIONS = 1_000
_MIN_BOOTSTRAP = 100
_METHODS: tuple[str, ...] = ("empirical", "beta_montecarlo", "evt_montecarlo")


@dataclass(frozen=True, eq=False)
class EstimationResult:
    """Outcome of a single-route threshold estimation."""

    threshold: float
    """Cutoff gamma; H0 scores above it count as false alarms."""
    ci_low: float
    ci_high: float
    """95% bootstrap confidence interval for ``threshold``."""
    target_pfa: float
    """Requested probability of false alarm."""
    simulated_pfa: float
    """False-alarm rate observed on the samples behind ``threshold``."""
    method: EstimationMethod
    """Estimation strategy used."""
    n_samples: int
    """Number of H0 calibration scores."""
    n_simulations: int
    """Monte Carlo draws behind gamma (0 for the empirical method)."""
    ks_pvalue: float
    """Goodness-of-fit p-value (1.0 for the non-parametric empirical method)."""
    tail_params: dict[str, float]
    """Fitted tail parameters (empty for the empirical method)."""
    mc_samples: np.ndarray = field(repr=False)
    """Monte Carlo sample behind gamma (the H0 sample itself for empirical)."""


def _validate_scores(scores: ArrayLike) -> np.ndarray:
    try:
        arr = np.array(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"scores must be numeric: {exc}") from exc
    if arr.ndim != 1:
        raise ValidationError(f"scores must be a 1-D array, got {arr.ndim}-D")
    if arr.size < _MIN_SAMPLES:
        raise ValidationError(
            f"calibration needs at least {_MIN_SAMPLES} scores, got {arr.size}"
        )
    if not np.isfinite(arr).all():
        raise ValidationError("scores must be finite (no NaN or +/-Inf)")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValidationError("scores must lie within [0.0, 1.0]")
    if arr.min() == arr.max():
        raise ValidationError("scores are constant; a cutoff cannot be calibrated")
    return arr


def _validate_pfa(target_pfa: float) -> float:
    try:
        pfa = float(target_pfa)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"target_pfa must be a float: {exc}") from exc
    if not 0.0 < pfa < 0.5:
        raise ValidationError(f"target_pfa must satisfy 0 < p < 0.5, got {pfa}")
    return pfa


def _as_rng(random_state: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _bootstrap_quantile_ci(
    scores: np.ndarray, q: float, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Vectorized bootstrap CI for the q-quantile of ``scores``."""
    n = scores.size
    chunk = max(1, 2_000_000 // max(n, 1))
    pieces: list[np.ndarray] = []
    drawn = 0
    while drawn < n_bootstrap:
        rows = min(chunk, n_bootstrap - drawn)
        idx = rng.integers(0, n, size=(rows, n))
        pieces.append(np.quantile(scores[idx], q, axis=1))
        drawn += rows
    boot = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
    lo, hi = np.quantile(boot, (0.025, 0.975))
    return float(lo), float(hi)


def _estimate_empirical(
    scores: np.ndarray, pfa: float, n_bootstrap: int, rng: np.random.Generator
) -> EstimationResult:
    q = 1.0 - pfa
    threshold = float(np.quantile(scores, q))
    ci_low, ci_high = _bootstrap_quantile_ci(scores, q, n_bootstrap, rng)
    return EstimationResult(
        threshold=threshold,
        ci_low=ci_low,
        ci_high=ci_high,
        target_pfa=pfa,
        simulated_pfa=float(np.mean(scores > threshold)),
        method="empirical",
        n_samples=int(scores.size),
        n_simulations=0,
        ks_pvalue=1.0,
        tail_params={},
        mc_samples=scores,
    )


def _estimate_beta(
    scores: np.ndarray,
    pfa: float,
    n_simulations: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> EstimationResult:
    clipped = np.clip(scores, _EPS, 1.0 - _EPS)
    try:
        a, b, _, _ = stats.beta.fit(clipped, floc=0.0, fscale=1.0)
    except (ValueError, RuntimeError) as exc:
        raise CalibrationError(f"Beta fit failed: {exc}") from exc
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
        raise CalibrationError("Beta fit produced invalid parameters")
    ks = stats.kstest(clipped, "beta", args=(a, b))
    mc = stats.beta.rvs(a, b, size=n_simulations, random_state=rng)
    threshold = float(np.quantile(mc, 1.0 - pfa))
    ci_low, ci_high = _bootstrap_quantile_ci(scores, 1.0 - pfa, n_bootstrap, rng)
    return EstimationResult(
        threshold=threshold,
        ci_low=ci_low,
        ci_high=ci_high,
        target_pfa=pfa,
        simulated_pfa=float(np.mean(mc > threshold)),
        method="beta_montecarlo",
        n_samples=int(scores.size),
        n_simulations=int(n_simulations),
        ks_pvalue=float(ks.pvalue),
        tail_params={"a": float(a), "b": float(b)},
        mc_samples=mc,
    )


def _extract_tail(
    z: np.ndarray, tail_fraction: float
) -> tuple[float, np.ndarray, float]:
    """Return ``(u, exceedances, empirical_tail_fraction)`` for logit-space scores."""
    u = float(np.quantile(z, 1.0 - tail_fraction))
    exceedances = z[z > u] - u
    if exceedances.size < _MIN_TAIL_EXCEEDANCES:
        u = float(np.quantile(z, 0.80))
        exceedances = z[z > u] - u
    if exceedances.size < 2:
        raise CalibrationError(
            "tail extraction failed: fewer than 2 exceedances above the tail threshold"
        )
    if float(exceedances.min()) == float(exceedances.max()):
        raise CalibrationError("degenerate tail: exceedances have zero variance")
    return u, exceedances, exceedances.size / z.size


def _gpd_pwm(x: np.ndarray) -> tuple[float, float]:
    """Closed-form probability-weighted moments (PWM) GPD estimator.

    Returns ``(c, sigma)`` in ``scipy.stats.genpareto`` parameterization
    (``cdf(x) = 1 - (1 + c*x/sigma)**(-1/c)``; ``c > 0`` is heavy-tailed).
    With L-moments ``l1``, ``l2`` and ``tau = l2 / l1``::

        c = 2 - 1/tau        sigma = l1 * (1 - tau) / tau
    """
    xs = np.sort(x)
    k = xs.size
    b0 = float(xs.mean())
    b1 = float(np.dot(np.arange(k) / (k - 1), xs) / k)
    l1 = b0
    l2 = 2.0 * b1 - b0
    if l1 <= 0.0 or l2 <= 0.0:
        raise CalibrationError("PWM estimation failed: non-positive L-moments")
    tau = l2 / l1
    if not 1.0 / 3.0 < tau < 1.0:
        raise CalibrationError("PWM estimation failed: L-CV out of range")
    c = 2.0 - 1.0 / tau
    sigma = l1 * (1.0 - tau) / tau
    if not np.isfinite(c) or not np.isfinite(sigma) or sigma <= 0.0 or c <= -1.0:
        raise CalibrationError("PWM estimation failed: invalid GPD parameters")
    return float(c), float(sigma)


def _hybrid_mc(
    z: np.ndarray,
    u: float,
    c: float,
    sigma: float,
    p_tail: float,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Hybrid Monte Carlo scores: empirical body + GPD tail, mapped back via expit."""
    body = z[z <= u]
    if body.size == 0:
        raise CalibrationError("tail extraction produced an empty body")
    tail_mask = rng.random(n_simulations) < p_tail
    n_tail = int(np.count_nonzero(tail_mask))
    z_mc = np.empty(n_simulations, dtype=np.float64)
    z_mc[~tail_mask] = body[rng.integers(0, body.size, size=n_simulations - n_tail)]
    if n_tail:
        z_mc[tail_mask] = u + stats.genpareto.rvs(
            c, loc=0.0, scale=sigma, size=n_tail, random_state=rng
        )
    return expit(z_mc)


def _bootstrap_gamma(
    z_b: np.ndarray, u_b: float, exceedances: np.ndarray, target_pfa: float
) -> float:
    """Semi-analytic mixture quantile for one bootstrap resample (no MC needed)."""
    q = 1.0 - target_pfa
    n = z_b.size
    k = exceedances.size
    if k < 2 or float(exceedances.min()) == float(exceedances.max()):
        return float(np.quantile(expit(z_b), q))
    try:
        c_b, sigma_b = _gpd_pwm(exceedances)
    except CalibrationError:
        return float(np.quantile(expit(z_b), q))
    p_tail = k / n
    if q > 1.0 - p_tail:
        # The quantile sits in the GPD tail component of the mixture.
        x_q = stats.genpareto.ppf(1.0 - target_pfa / p_tail, c_b, loc=0.0, scale=sigma_b)
        z_q = u_b + float(x_q)
    else:
        body = z_b[z_b <= u_b]
        if body.size == 0:
            return float(np.quantile(expit(z_b), q))
        level = min(q / max(1.0 - p_tail, 1e-12), 1.0)
        z_q = float(np.quantile(body, level))
    gamma = float(expit(z_q))
    if not np.isfinite(gamma):
        return float(np.quantile(expit(z_b), q))
    return gamma


def _evt_bootstrap_ci(
    scores: np.ndarray,
    target_pfa: float,
    tail_fraction: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap CI that refits the GPD via the fast PWM estimator inside the loop."""
    n = scores.size
    gammas = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resample = scores[rng.integers(0, n, size=n)]
        z_b = logit(np.clip(resample, _EPS, 1.0 - _EPS))
        u_b = float(np.quantile(z_b, 1.0 - tail_fraction))
        exceedances = z_b[z_b > u_b] - u_b
        if exceedances.size < _MIN_TAIL_EXCEEDANCES:
            u_b = float(np.quantile(z_b, 0.80))
            exceedances = z_b[z_b > u_b] - u_b
        gammas[i] = _bootstrap_gamma(z_b, u_b, exceedances, target_pfa)
    lo, hi = np.quantile(gammas, (0.025, 0.975))
    return float(lo), float(hi)


def _estimate_evt(
    scores: np.ndarray,
    pfa: float,
    n_simulations: int,
    n_bootstrap: int,
    tail_fraction: float,
    rng: np.random.Generator,
) -> EstimationResult:
    z = logit(np.clip(scores, _EPS, 1.0 - _EPS))
    u, exceedances, p_tail = _extract_tail(z, tail_fraction)
    try:
        c, _, sigma = stats.genpareto.fit(exceedances, floc=0.0)
    except (ValueError, RuntimeError) as exc:
        raise CalibrationError(f"GPD fit failed: {exc}") from exc
    if not (np.isfinite(c) and np.isfinite(sigma) and sigma > 0.0):
        raise CalibrationError("GPD fit produced invalid parameters")
    ks = stats.kstest(exceedances, "genpareto", args=(c, 0.0, sigma))
    mc = _hybrid_mc(z, u, float(c), float(sigma), p_tail, n_simulations, rng)
    threshold = float(np.quantile(mc, 1.0 - pfa))
    ci_low, ci_high = _evt_bootstrap_ci(scores, pfa, tail_fraction, n_bootstrap, rng)
    return EstimationResult(
        threshold=threshold,
        ci_low=ci_low,
        ci_high=ci_high,
        target_pfa=pfa,
        simulated_pfa=float(np.mean(mc > threshold)),
        method="evt_montecarlo",
        n_samples=int(scores.size),
        n_simulations=int(n_simulations),
        ks_pvalue=float(ks.pvalue),
        tail_params={
            "xi": float(c),
            "sigma": float(sigma),
            "u": float(u),
            "tail_fraction": float(p_tail),
        },
        mc_samples=mc,
    )


def estimate_threshold(
    scores: ArrayLike,
    target_pfa: float,
    method: EstimationMethod = "evt_montecarlo",
    *,
    n_simulations: int = 100_000,
    n_bootstrap: int = 300,
    tail_fraction: float = 0.15,
    random_state: int | np.random.Generator | None = 42,
) -> EstimationResult:
    """Estimate the cutoff ``gamma`` for one route from its H0 score sample.

    Args:
        scores: 1-D array of benign H0 confidence scores in ``[0, 1]``
            (at least 30 samples, finite, non-constant).
        target_pfa: desired false-alarm probability, ``0 < p < 0.5``.
        method: ``"empirical"``, ``"beta_montecarlo"`` or ``"evt_montecarlo"``.
        n_simulations: Monte Carlo draws for the simulation-based methods.
        n_bootstrap: bootstrap resamples behind the 95% CI.
        tail_fraction: fraction of the upper tail modelled by the GPD
            (EVT method only).
        random_state: seed (default 42 for reproducible audits), a
            ``numpy.random.Generator``, or ``None`` for OS entropy.

    Returns:
        Frozen :class:`EstimationResult` with the cutoff and its uncertainty.

    Raises:
        ValidationError: invalid inputs.
        CalibrationError: the statistical fit failed or is degenerate.
    """
    if method not in _METHODS:
        raise ValidationError(f"unknown method {method!r}; expected one of {_METHODS}")
    arr = _validate_scores(scores)
    pfa = _validate_pfa(target_pfa)
    if method != "empirical" and n_simulations < _MIN_SIMULATIONS:
        raise ValidationError(
            f"n_simulations must be >= {_MIN_SIMULATIONS} for {method!r}, "
            f"got {n_simulations}"
        )
    if n_bootstrap < _MIN_BOOTSTRAP:
        raise ValidationError(f"n_bootstrap must be >= {_MIN_BOOTSTRAP}, got {n_bootstrap}")
    if method == "evt_montecarlo" and not 0.0 < tail_fraction < 0.5:
        raise ValidationError(
            f"tail_fraction must satisfy 0 < f < 0.5, got {tail_fraction}"
        )
    rng = _as_rng(random_state)
    if method == "empirical":
        return _estimate_empirical(arr, pfa, n_bootstrap, rng)
    if method == "beta_montecarlo":
        return _estimate_beta(arr, pfa, n_simulations, n_bootstrap, rng)
    return _estimate_evt(arr, pfa, n_simulations, n_bootstrap, tail_fraction, rng)
