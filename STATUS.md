# cfar — Implementation Status

> Last updated: **2026-11-12** · Package version: **0.1.0** · Overall status: **complete, all tests green, validated on real models**

Quick links: [Modules](#module-status) · [Feature checklist](#feature-checklist) · [Verification](#last-verification-run) · [Known limitations](#known-limitations--backlog) · [Changelog](#changelog) · [Maintaining this file](#maintaining-this-file)

---

## At a glance

| Item | Value |
|---|---|
| Version | 0.1.0 |
| Python | >= 3.10 (verified on 3.14.7) |
| Runtime dependencies | `numpy>=1.24.0`, `scipy>=1.10.0` — nothing else |
| Test suite | **59 / 59 passing** (`pytest -v`, 2.9 s) |
| Packaging | `pip install -e .` verified (src layout, PEP 561 `py.typed`) |
| Real-model validation | 2 open-source decision models on real data — see [Verification](#last-verification-run) |
| License | Apache-2.0 |
| Repo state | Initial commit created on `main` (package + tests + examples) |

---

## Module status

| Module | Status | Contents |
|---|---|---|
| `src/cfar/exceptions.py` | done | `CFARError`, `ValidationError`, `CalibrationError` |
| `src/cfar/adapters.py` | done | `from_logits`, `from_cosine`, `from_logprobs` — bounded `[0,1]` float64, NaN-rejecting |
| `src/cfar/estimators.py` | done | `estimate_threshold` + `EstimationResult`; `empirical`, `beta_montecarlo`, `evt_montecarlo` (default); PWM GPD bootstrap CI |
| `src/cfar/manifest.py` | done | `RouteArtifact`, `CalibrationManifest` (JSON save/load), `RuntimeGate` |
| `src/cfar/calibrator.py` | done | `MultiRouteCalibrator` (Šidák FWER), `ThresholdCalibrator`, `CalibrationEvidence` (summary / manifest / HTML+SVG report) |
| `src/cfar/monitor.py` | done | `DriftMonitor`, `DriftReport` (two-sample KS + Wasserstein) |
| `src/cfar/__init__.py` | done | Public API re-exports, `__version__` |
| `tests/` (4 files) | done | 59 tests: adapters, estimators, calibrator/gate, monitor |
| `examples/` (2 demos) | done | Real-model integration demos: LLM logit router (Qwen2.5-0.5B-Instruct) + embedding semantic router (all-MiniLM-L6-v2) on real data (Dolly-15k / GSM8K / SQuAD) |
| `pyproject.toml` | done | Metadata, deps, `pythonpath=["src"]` for pytest |
| `README.md` | done | Problem statement, ASCII architecture, quickstart |
| `LICENSE` | done | Full Apache-2.0 text |

---

## Feature checklist

### Adapters
- [x] `from_logits` — `expit(logits / temperature)`, temperature > 0 validated
- [x] `from_cosine` — `(s + 1) / 2` with clipping
- [x] `from_logprobs` — `exp(mean(logprob))` per sequence (2-D) or `exp(x)` (1-D)
- [x] All outputs strictly bounded `[0.0, 1.0]`, float64; NaN inputs rejected

### Estimators
- [x] Input contract: 1-D, n >= 30, finite, in `[0,1]`, non-constant; `0 < target_pfa < 0.5`
- [x] `empirical` — direct quantile + vectorized bootstrap 95% CI
- [x] `beta_montecarlo` — Beta MLE fit (`floc=0, fscale=1`), KS GOF, MC quantile
- [x] `evt_montecarlo` (default) — logit-space tail, GPD fit (`floc=0`), hybrid MC (empirical body + GPD tail), 0.80-quantile fallback for thin tails
- [x] Degenerate-tail and constant-input guards (`CalibrationError` / `ValidationError`)
- [x] Fast bootstrap CI via closed-form PWM GPD estimator (`c = 2 - 1/tau`, scipy POT convention)
- [x] Deterministic by default (`random_state=42`); accepts seed, `Generator`, or `None`

### Manifest & runtime gate
- [x] `RouteArtifact` frozen dataclass (threshold, CI, PFAs, KS p-value, H1 recall, 101-point `h0_quantiles` fingerprint, tail params)
- [x] `CalibrationManifest.to_dict / save / load / from_dict` — JSON round-trip tested
- [x] `RuntimeGate` — pre-cached `{route: threshold}` dict; `evaluate()` returns triggered routes ranked by score, `min_margin` filtering, unknown routes ignored

### Calibration
- [x] `target_pfa` (scalar or per-route dict) XOR `system_pfa` — mutually exclusive, validated
- [x] Šidák FWER: `alpha_route = 1 - (1 - alpha_sys)^(1/M)`
- [x] `h1_recall = mean(h1 > gamma)`; advisories when recall < 0.90 or KS p < 0.05
- [x] `CalibrationEvidence.summary()` — fixed-width ASCII table
- [x] `export_report()` — self-contained HTML, inline SVG histograms (H0/H1 + dashed γ line), no external resources

### Monitoring
- [x] `DriftMonitor.check()` — n >= 20 live scores vs 101-point baseline fingerprint
- [x] Two-sample KS + Wasserstein distance; `drift_detected = ks_pvalue < alpha` (default 0.01)

---

## Last verification run

| Check | Result |
|---|---|
| `pytest -v` | **59 passed, 0 failed, 0 warnings** (2.92 s, Python 3.14.7 / numpy 2.5.4 / scipy 1.18.1) |
| `pip install -e .` | OK — metadata + src layout valid |
| EVT accuracy (logit-normal, PFA 0.05) | γ = 0.838 vs analytic 0.8382; simulated PFA = 0.050 |
| Empirical / beta accuracy (Beta(2,8), PFA 0.05) | threshold within tolerance of `beta.ppf(0.95)`; simulated PFA ≈ 0.05 |
| Calibration speed (EVT, n = 2000–5000) | ~120 ms per route (budget: < 250 ms) |
| `RuntimeGate.evaluate()` latency | < 0.1 ms per call (dict-cached lookup) |
| Drift detection (synthetic) | stationary → no drift (KS p = 0.93); shifted → drift flagged |

### Real-model validation (`examples/`, 2026-11-12)

Orchestrator demo: 1 tool (calculator) + 1 sub-agent (KB endpoint); real data
(H0 = Dolly-15k instructions, H1 = GSM8K math / SQuAD factual questions);
labels by dataset provenance; `system_pfa = 0.05` (Šidák → 2.53% per route).

| Check | Result |
|---|---|
| Demo A — Qwen2.5-0.5B-Instruct yes/no log-odds router (`from_logits`) | Pipeline works end-to-end (1,700 real queries, MPS). PFA held exactly (sim = 0.0253 = target). **Router cannot separate routes** (H1 recall 0.008 / 0.000) — advisories fired correctly. Prompt-change drift detected (KS p = 1.3e-23); same-prompt control clean (p = 0.81). |
| Demo B — all-MiniLM-L6-v2 embedding semantic router (`from_cosine`) | **Calculator route production-viable**: H1 recall 0.736 @ 2.53% PFA; 3/3 math and 3/3 benign held-out queries routed correctly; 1,700 queries scored in 2.9 s. KB route weak (recall 0.064) — flagged by advisory. Cosmetic route-description reword correctly NOT flagged as drift (p = 0.074, distribution unmoved). |
| Interpretation | cfar held the false-alarm budget on every real model and quantified which routes each decision model can actually support — instead of silently misrouting behind a hand-tuned threshold. |

---

## Known limitations & backlog

- [ ] Not yet published to PyPI (`pip install cfar` in README is aspirational)
- [x] Initial commit created (v0.1.0: package + tests + examples)
- [ ] `beta_montecarlo` bootstrap CI uses the empirical quantile bootstrap (no per-resample Beta refit) — documented trade-off for speed
- [ ] EVT bootstrap CI uses a semi-analytic mixture quantile (PWM refit) rather than a full MC per resample — documented trade-off for speed
- [ ] KS p-values with fitted parameters are approximate (standard GOF caveat)
- [ ] Example demos label routes by dataset provenance (heuristic ground truth, not per-query human annotation)
- [ ] KB-route decision model in the demos is weak (embedding similarity to a single description); needs a stronger router (few-shot LLM, better descriptions, or a trained head) — cfar's audit quantifies the gap rather than hiding it
- [ ] Possible future work: Python 3.10–3.13 CI matrix, ruff/mypy config, publish workflow

---

## Changelog

### 0.1.0 — 2026-11-12
- Initial implementation: adapters, three estimators (empirical / beta MC / EVT MC), Šidák multi-route budgeting, JSON manifests, `RuntimeGate`, `DriftMonitor`, HTML+SVG audit reports.
- 59-test suite, all passing.
- Real-model validation demos added (`examples/`): cfar audited two real open-source decision models (Qwen2.5-0.5B-Instruct logit router; all-MiniLM-L6-v2 embedding router) on real data (Dolly-15k / GSM8K / SQuAD). False-alarm budgets held exactly in all runs; the advisory system correctly exposed decision models that cannot separate their routes, and drift detection correctly flagged a real prompt change while clearing an unchanged control.
- Notable fixes during development:
  - PWM GPD estimator sign convention corrected to scipy's POT parameterization (`c = 2 - 1/tau`).
  - Degenerate-input guards switched from `var() == 0` to exact `min == max` (float variance of constants is not exactly zero).

---

## Maintaining this file

When making changes, update:

1. **Last updated** date (top) and **At a glance** table.
2. **Module status** — flip entries or add new modules/rows.
3. **Feature checklist** — tick `[x]` items or add `[ ]` ones for planned work.
4. **Last verification run** — replace with fresh `pytest -v` numbers after any change.
5. **Known limitations & backlog** — move finished items out, add new ones.
6. **Changelog** — add a `### <version> — <date>` section per release (Keep-a-Changelog style).
