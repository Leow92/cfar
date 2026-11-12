# cfar — Constant False Alarm Rate gating for AI systems

`cfar` is a zero-bloat, model-agnostic statistical calibration and runtime
gating library for AI agent orchestrators, semantic routers, and guardrails.
It replaces arbitrary confidence thresholds (`if score > 0.75`) with cutoffs
γ derived from the distribution of benign / near-miss (H0) scores, each
carrying an explicit, testable probability of false alarm (PFA).

- Runtime dependencies: `numpy` + `scipy`. Nothing else.
- Operates on 1-D arrays of bounded `[0, 1]` confidence scores — zero PII,
  zero text touch.
- Deterministic calibration (seeded `np.random.default_rng`), JSON manifests,
  self-contained HTML/SVG audit reports.

## The problem

Production agent stacks gate on magic numbers. A `0.75` cutoff means nothing
statistically: it varies per model, per prompt template, and per score scale,
and nobody knows the resulting false-alarm rate.

| hand-tuned threshold | cfar cutoff |
|---|---|
| no statistical meaning | `P(score > γ \| H0) ≈ target PFA` |
| per-model folklore | calibrated from your own benign traffic |
| no alarm budget | Šidák-corrected system-wide PFA budget |
| silent degradation | KS + Wasserstein drift monitoring |

`cfar` inverts the question: *given the benign score distribution, where must
the cutoff sit so that the false-alarm probability stays under budget?*

## Architecture

```
OFFLINE CALIBRATION
  raw model signals (logits / cosine similarities / token logprobs)
      │   adapters: from_logits · from_cosine · from_logprobs
      ▼
  bounded [0, 1] confidence scores
      │   H0 = benign / near-miss sample per route (+ optional H1 = signal)
      ▼
  estimators:  empirical ─────────────────┐
               beta_montecarlo ───────────┤   γ, 95% bootstrap CI,
               evt_montecarlo (default) ──┘   simulated PFA, KS goodness-of-fit
      ▼
  CalibrationManifest (JSON) ──▶ audit report: self-contained HTML + inline SVG

RUNTIME
  live scores ─────▶ RuntimeGate.evaluate() ──▶ triggered routes (score > γ, ranked, margin-filtered)
  live H0 sample ──▶ DriftMonitor.check() ────▶ DriftReport (two-sample KS + Wasserstein)
```

## Installation

```bash
pip install cfar        # from PyPI (once published)
pip install -e .        # from a source checkout
```

Requires Python >= 3.10, `numpy >= 1.24`, `scipy >= 1.10`.

## Quickstart

```python
import numpy as np
from scipy.special import expit
from cfar import MultiRouteCalibrator, RuntimeGate, DriftMonitor

rng = np.random.default_rng(0)

# 1. Collect bounded [0, 1] confidence scores per route (H0 = benign traffic).
h0 = {
    "router": expit(rng.normal(0.0, 1.0, size=5000)),    # e.g. via from_logits(...)
    "guard":  expit(rng.normal(-0.3, 1.0, size=5000)),   # e.g. via from_cosine(...)
}
h1 = {"router": expit(rng.normal(3.5, 1.0, size=2000))}  # optional signal sample

# 2. Calibrate cutoffs under a 5% system-wide false-alarm budget (Šidák FWER).
evidence = MultiRouteCalibrator(system_pfa=0.05).calibrate(h0, h1_scores=h1)
print(evidence.summary())
evidence.save_manifest("manifest.json")
evidence.export_report("audit_report.html")   # self-contained HTML + inline SVG

# 3. Gate at runtime — no magic numbers.
gate = RuntimeGate.from_manifest("manifest.json")
triggered = gate.evaluate({"router": 0.91, "guard": 0.12}, min_margin=0.05)

# 4. Monitor the benign score distribution for drift.
live_h0 = expit(rng.normal(0.0, 1.0, size=500))
report = DriftMonitor.from_manifest("manifest.json").check("router", live_h0)
print(report.drift_detected, report.ks_pvalue)
```

## Estimation methods

| method | model | best for |
|---|---|---|
| `empirical` | none (direct quantile) | any shape; no extrapolation beyond the sample |
| `beta_montecarlo` | `Beta(a, b)` on `[0, 1]` | smooth unimodal score distributions |
| `evt_montecarlo` (default) | Generalized Pareto tail in logit space | heavy or bounded tails; extrapolates without ever leaving `[0, 1]` |

All methods return γ with a 95% bootstrap CI, the simulated PFA, and a KS
goodness-of-fit p-value (1.0 for the non-parametric `empirical` method).

### EVT detail (default)

Scores are mapped to unconstrained logit space `z = logit(s)` so tail
extrapolation can never produce values outside `[0, 1]` when mapped back via
`expit`. Exceedances above `u = quantile(z, 1 - tail_fraction)` (falling back
to the 0.80 quantile when the tail is too thin) are fitted with a Generalized
Pareto Distribution (`scipy.stats.genpareto.fit(..., floc=0)`); a hybrid
Monte Carlo simulation — empirical body with probability `1 - tail_fraction`,
`u + GPD(ξ, σ)` tail with probability `tail_fraction` — yields
`γ = quantile(s_MC, 1 - target_pfa)`. Bootstrap CIs refit the GPD with a
closed-form probability-weighted-moments (PWM) estimator inside the loop, so
a full calibration finishes in well under 250 ms.

## System-wide PFA budget (Šidák)

With `M` routes and a joint budget `system_pfa = α`, each route receives

```
α_route = 1 − (1 − α)^(1/M)
```

so the probability that *any* route raises a false alarm stays ≤ α.

## Multi-tool queries

A query can legitimately require several routes at once — "look up the
population of Tokyo, then multiply it by twelve" needs both the KB sub-agent
and the calculator. `RuntimeGate.evaluate()` handles this by design:

- **Union semantics** — it returns *every* route whose score exceeds its own
  γ, ranked best-first, not just the top match.
- **The budget already covers it** — Šidák prices the event "a benign query
  triggers *any* route" per query, not per firing, so traffic that
  legitimately calls several tools never breaks the system PFA budget.
- **`min_margin` is the eagerness dial** — runner-ups more than `min_margin`
  below the leader are dropped. Genuinely blended queries score close on
  both heads and survive; a faint secondary echo is suppressed.

```python
triggered = gate.evaluate({"calculator": 0.64, "kb_agent": 0.61})
# ['calculator', 'kb_agent'] -> dispatch both; dependency order is yours
```

Dependency ordering (e.g. the KB result feeding the calculator) is
orchestrator-side: cfar decides *which* routes fire, your execution plan
decides how they run. See `examples/` for runnable end-to-end demos,
including `multi_tool_query.py`.

## Adapters

```python
from cfar import from_logits, from_cosine, from_logprobs

scores = from_logits(logits, temperature=1.0)   # expit(logits / T)
scores = from_cosine(cosine_sims)                # (s + 1) / 2, clipped to [0, 1]
scores = from_logprobs(logprobs_2d)              # exp(mean(logprob)) per sequence
```

## Drift monitoring

`DriftMonitor.check(route, live_h0_scores)` compares a live benign sample
(n ≥ 20) against the 101-point percentile fingerprint stored in the manifest
using the two-sample Kolmogorov–Smirnov test and the Wasserstein distance.
`drift_detected` is raised when the KS p-value falls below `alpha` (default
`0.01`).

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
