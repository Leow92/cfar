"""Multi-route threshold calibration with auditable evidence artifacts."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike

from cfar import __version__
from cfar.estimators import EstimationMethod, EstimationResult, estimate_threshold
from cfar.exceptions import ValidationError
from cfar.manifest import CalibrationManifest, RouteArtifact

__all__ = ["CalibrationEvidence", "MultiRouteCalibrator", "ThresholdCalibrator"]

_H1_RECALL_TARGET = 0.90
_KS_WARN_LEVEL = 0.05


def _check_pfa_value(value: object, label: str, upper: float) -> None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{label} must be a float: {exc}") from exc
    if not 0.0 < number < upper:
        raise ValidationError(f"{label} must satisfy 0 < p < {upper}, got {number}")


def _validate_h1(scores: ArrayLike) -> np.ndarray:
    """Validate an optional H1 (signal) score sample."""
    try:
        arr = np.array(scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"h1 scores must be numeric: {exc}") from exc
    if arr.ndim != 1:
        raise ValidationError(f"h1 scores must be 1-D, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValidationError("h1 scores must not be empty")
    if not np.isfinite(arr).all():
        raise ValidationError("h1 scores must be finite (no NaN or +/-Inf)")
    if arr.min() < 0.0 or arr.max() > 1.0:
        raise ValidationError("h1 scores must lie within [0.0, 1.0]")
    return arr


def _route_advisories(result: EstimationResult, h1_recall: float | None) -> list[str]:
    """Human-readable advisory strings for one calibrated route."""
    advisories: list[str] = []
    if h1_recall is not None and h1_recall < _H1_RECALL_TARGET:
        advisories.append(
            f"H1 recall {h1_recall:.3f} is below the {_H1_RECALL_TARGET:.2f} target; "
            "the cutoff may reject too much legitimate traffic. Consider a higher "
            "target_pfa, additional calibration data, or better score separation."
        )
    if result.ks_pvalue < _KS_WARN_LEVEL:
        advisories.append(
            f"Tail goodness-of-fit KS p-value {result.ks_pvalue:.3g} is below "
            f"{_KS_WARN_LEVEL:.2f}; the fitted tail model may be misspecified. "
            "Consider method='empirical' or collecting more calibration samples."
        )
    return advisories


_HTML_HEAD = (
    "<!DOCTYPE html>\n"
    "<html lang='en'>\n"
    "<head>\n"
    "<meta charset='utf-8'>\n"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
    "<title>CFAR Calibration Report</title>\n"
    "<style>\n"
    ":root { color-scheme: light; }\n"
    "* { box-sizing: border-box; }\n"
    "body { margin: 0; padding: 2.5rem 1.25rem; background: #f1f5f9; color: #0f172a; "
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }\n"
    "main { max-width: 60rem; margin: 0 auto; }\n"
    "h1 { font-size: 1.45rem; margin: 0 0 0.3rem; }\n"
    "h2 { font-size: 1.05rem; margin: 0 0 0.9rem; }\n"
    "p.meta { color: #64748b; font-size: 0.85rem; margin: 0 0 1.4rem; }\n"
    "table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0; "
    "border-radius: 8px; overflow: hidden; font-size: 0.88rem; font-variant-numeric: tabular-nums; }\n"
    "th { text-align: left; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; "
    "color: #64748b; background: #f8fafc; padding: 0.55rem 0.75rem; border-bottom: 1px solid #e2e8f0; }\n"
    "td { padding: 0.55rem 0.75rem; border-bottom: 1px solid #f1f5f9; }\n"
    "tr:last-child td { border-bottom: none; }\n"
    "section.route { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; "
    "padding: 1.2rem 1.2rem 1.4rem; margin: 1.2rem 0; }\n"
    ".cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.6rem, 1fr)); gap: 0.6rem; margin: 0 0 1rem; }\n"
    ".card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.7rem; }\n"
    ".card .k { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }\n"
    ".card .v { font-size: 1.0rem; font-weight: 600; margin-top: 0.18rem; font-variant-numeric: tabular-nums; }\n"
    ".advisory { border-left: 3px solid #f59e0b; background: #fffbeb; color: #92400e; "
    "padding: 0.5rem 0.75rem; border-radius: 0 6px 6px 0; font-size: 0.84rem; margin: 0.4rem 0; }\n"
    "svg { max-width: 100%; height: auto; display: block; margin-top: 0.5rem; }\n"
    "footer { color: #94a3b8; font-size: 0.75rem; margin-top: 2rem; }\n"
    "</style>\n"
    "</head>\n"
    "<body>\n"
    "<main>\n"
)


def _histogram_svg(
    h0: np.ndarray,
    gamma: float,
    h1: np.ndarray | None = None,
    width: int = 760,
    height: int = 300,
) -> str:
    """Self-contained inline SVG histogram of H0 (and optional H1) scores."""
    n_bins = 40
    counts0, _ = np.histogram(h0, bins=n_bins, range=(0.0, 1.0))
    counts1: np.ndarray | None = None
    if h1 is not None and np.size(h1) > 0:
        counts1, _ = np.histogram(h1, bins=n_bins, range=(0.0, 1.0))
    peak = 1
    for counts in (counts0, counts1):
        if counts is not None and counts.size:
            peak = max(peak, int(counts.max()))
    left, right, top, bottom = 56.0, 20.0, 34.0, 46.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_w = plot_w / n_bins
    parts: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img'>",
        f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
        f"<rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' "
        "fill='#f8fafc' stroke='#e2e8f0'/>",
    ]
    for i, count in enumerate(counts0):
        if count <= 0:
            continue
        bar_h = plot_h * count / peak
        parts.append(
            f"<rect x='{left + i * bar_w + 0.6:.2f}' y='{top + plot_h - bar_h:.2f}' "
            f"width='{bar_w - 1.2:.2f}' height='{bar_h:.2f}' fill='#6366f1' "
            "fill-opacity='0.9'/>"
        )
    if counts1 is not None:
        for i, count in enumerate(counts1):
            if count <= 0:
                continue
            bar_h = plot_h * count / peak
            parts.append(
                f"<rect x='{left + i * bar_w + 0.6:.2f}' y='{top + plot_h - bar_h:.2f}' "
                f"width='{bar_w - 1.2:.2f}' height='{bar_h:.2f}' fill='#f59e0b' "
                "fill-opacity='0.55'/>"
            )
    gx = left + gamma * plot_w
    parts.append(
        f"<line x1='{gx:.2f}' y1='{top}' x2='{gx:.2f}' y2='{top + plot_h}' "
        "stroke='#dc2626' stroke-width='2' stroke-dasharray='7 5'/>"
    )
    label_x = min(max(gx, left + 34.0), left + plot_w - 34.0)
    parts.append(
        f"<text x='{label_x:.2f}' y='{top - 9:.2f}' font-size='12' fill='#dc2626' "
        f"text-anchor='middle' font-family='monospace'>&gamma; = {gamma:.4f}</text>"
    )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        tx = left + tick * plot_w
        parts.append(
            f"<line x1='{tx:.2f}' y1='{top + plot_h}' x2='{tx:.2f}' "
            f"y2='{top + plot_h + 5.0}' stroke='#94a3b8'/>"
        )
        parts.append(
            f"<text x='{tx:.2f}' y='{top + plot_h + 19.0:.2f}' font-size='11' "
            f"fill='#64748b' text-anchor='middle' font-family='monospace'>{tick:.2f}</text>"
        )
    parts.append(
        f"<text x='{left - 8.0:.2f}' y='{top + 4.0:.2f}' font-size='11' fill='#64748b' "
        f"text-anchor='end' font-family='monospace'>{peak}</text>"
    )
    parts.append(
        f"<text x='{left - 8.0:.2f}' y='{top + plot_h + 4.0:.2f}' font-size='11' "
        "fill='#64748b' text-anchor='end' font-family='monospace'>0</text>"
    )
    parts.append(
        f"<text x='{left + plot_w / 2.0:.2f}' y='{height - 8.0:.2f}' font-size='11' "
        "fill='#475569' text-anchor='middle' font-family='monospace'>confidence "
        "score</text>"
    )
    legend_right = left + plot_w
    parts.append(
        f"<rect x='{legend_right - 214.0:.2f}' y='10' width='10' height='10' "
        "fill='#6366f1' fill-opacity='0.9'/>"
    )
    parts.append(
        f"<text x='{legend_right - 199.0:.2f}' y='19' font-size='11' fill='#475569' "
        "font-family='monospace'>H0 benign</text>"
    )
    if counts1 is not None:
        parts.append(
            f"<rect x='{legend_right - 112.0:.2f}' y='10' width='10' height='10' "
            "fill='#f59e0b' fill-opacity='0.7'/>"
        )
        parts.append(
            f"<text x='{legend_right - 97.0:.2f}' y='19' font-size='11' "
            "fill='#475569' font-family='monospace'>H1 signal</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


@dataclass
class CalibrationEvidence:
    """Auditable outcome of a calibration run."""

    manifest: CalibrationManifest
    thresholds: dict[str, float]
    advisories: dict[str, list[str]]
    h0_scores: dict[str, np.ndarray] = field(repr=False)
    h1_scores: dict[str, np.ndarray | None] = field(repr=False)

    def summary(self) -> str:
        """Fixed-width ASCII table of thresholds, PFAs, recall, and advisories."""
        m = self.manifest
        header = (
            f"{'Route':<20} {'Target PFA':>11} {'Threshold':>11} "
            f"{'95% CI':<23} {'Sim PFA':>9} {'H1 Recall':>10}"
        )
        major = "=" * len(header)
        minor = "-" * len(header)
        lines = [f"CFAR Calibration Summary (cfar v{m.cfar_version})", major, header, minor]
        for route, artifact in m.routes.items():
            recall = "-" if artifact.h1_recall is None else f"{artifact.h1_recall:.4f}"
            ci = f"[{artifact.ci_low:.4f}, {artifact.ci_high:.4f}]"
            lines.append(
                f"{route:<20} {artifact.target_pfa:>11.4f} {artifact.threshold:>11.4f} "
                f"{ci:<23} {artifact.simulated_pfa:>9.4f} {recall:>10}"
            )
        lines.append(minor)
        flagged = [(route, items) for route, items in self.advisories.items() if items]
        if flagged:
            lines.append("Advisories:")
            for route, items in flagged:
                for item in items:
                    lines.append(f"  {route}: {item}")
        else:
            lines.append("Advisories: none.")
        budget = (
            "unset" if m.system_pfa_budget is None
            else f"{m.system_pfa_budget:.4f} (Sidak FWER)"
        )
        lines.append(f"System PFA budget: {budget}")
        return "\n".join(lines)

    def save_manifest(self, path: str | Path) -> Path:
        """Serialize the calibration manifest to JSON and return its path."""
        return self.manifest.save(path)

    def export_report(self, path: str | Path) -> Path:
        """Write a self-contained HTML audit report with inline SVG histograms."""
        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._render_report(), encoding="utf-8")
        return target

    def _render_report(self) -> str:
        esc = html.escape
        m = self.manifest
        budget = "unset" if m.system_pfa_budget is None else f"{m.system_pfa_budget:.4f}"
        parts: list[str] = [
            _HTML_HEAD,
            "<h1>CFAR Calibration Report</h1>",
            f"<p class='meta'>cfar v{esc(m.cfar_version)} &middot; created "
            f"{esc(m.created_at)} &middot; system PFA budget: {esc(budget)}</p>",
            self._overview_table(),
        ]
        for route, artifact in m.routes.items():
            parts.append(f"<section class='route'><h2>{esc(route)}</h2>")
            parts.append(self._cards(artifact))
            for item in self.advisories.get(route, []):
                parts.append(f"<div class='advisory'>{esc(item)}</div>")
            h1 = self.h1_scores.get(route)
            parts.append(_histogram_svg(self.h0_scores[route], artifact.threshold, h1))
            parts.append("</section>")
        parts.append(
            f"<footer>Self-contained audit report generated by cfar "
            f"v{esc(m.cfar_version)} &middot; Apache-2.0 &middot; no external "
            "resources.</footer>"
        )
        parts.append("</main>\n</body>\n</html>\n")
        return "".join(parts)

    def _overview_table(self) -> str:
        esc = html.escape
        rows: list[str] = []
        for route, artifact in self.manifest.routes.items():
            recall = (
                "&mdash;" if artifact.h1_recall is None else f"{artifact.h1_recall:.4f}"
            )
            rows.append(
                "<tr>"
                f"<td>{esc(route)}</td><td>{esc(artifact.method)}</td>"
                f"<td>{artifact.target_pfa:.4f}</td><td>{artifact.threshold:.4f}</td>"
                f"<td>[{artifact.ci_low:.4f}, {artifact.ci_high:.4f}]</td>"
                f"<td>{artifact.simulated_pfa:.4f}</td><td>{recall}</td>"
                f"<td>{artifact.ks_pvalue:.4g}</td>"
                "</tr>"
            )
        return (
            "<table><thead><tr>"
            "<th>Route</th><th>Method</th><th>Target PFA</th>"
            "<th>Threshold &gamma;</th><th>95% CI</th><th>Sim PFA</th>"
            "<th>H1 Recall</th><th>KS p</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

    def _cards(self, artifact: RouteArtifact) -> str:
        esc = html.escape
        entries = [
            ("Threshold (\u03b3)", f"{artifact.threshold:.4f}"),
            ("95% CI", f"[{artifact.ci_low:.4f}, {artifact.ci_high:.4f}]"),
            ("Target PFA", f"{artifact.target_pfa:.4f}"),
            ("Simulated PFA", f"{artifact.simulated_pfa:.4f}"),
            ("H1 Recall", "-" if artifact.h1_recall is None else f"{artifact.h1_recall:.4f}"),
            ("KS p-value", f"{artifact.ks_pvalue:.4g}"),
            ("Method", artifact.method),
            ("Calibration samples", f"{artifact.n_samples:,}"),
            ("MC simulations", f"{artifact.n_simulations:,}"),
        ]
        cells = "".join(
            f"<div class='card'><div class='k'>{esc(key)}</div>"
            f"<div class='v'>{esc(value)}</div></div>"
            for key, value in entries
        )
        return f"<div class='cards'>{cells}</div>"


class MultiRouteCalibrator:
    """Calibrate cutoffs for many routes under per-route or system-wide PFA budgets.

    Pass exactly one of:

    * ``target_pfa`` -- a float applied to every route, or a mapping with one
      entry per route.
    * ``system_pfa`` -- a joint family-wise budget split across ``M`` routes
      via the Sidak correction ``alpha_route = 1 - (1 - alpha) ** (1 / M)``.
    """

    def __init__(
        self,
        *,
        target_pfa: float | Mapping[str, float] | None = None,
        system_pfa: float | None = None,
        method: EstimationMethod = "evt_montecarlo",
        n_simulations: int = 100_000,
        n_bootstrap: int = 300,
        tail_fraction: float = 0.15,
        random_state: int | np.random.Generator | None = 42,
    ) -> None:
        if (target_pfa is None) == (system_pfa is None):
            raise ValidationError(
                "provide exactly one of target_pfa (float or per-route mapping) "
                "or system_pfa"
            )
        if system_pfa is not None:
            _check_pfa_value(system_pfa, "system_pfa", upper=1.0)
        if target_pfa is not None:
            if isinstance(target_pfa, Mapping):
                for route, value in target_pfa.items():
                    _check_pfa_value(value, f"target_pfa for route {route!r}", upper=0.5)
            else:
                _check_pfa_value(target_pfa, "target_pfa", upper=0.5)
        self._target_pfa = target_pfa
        self._system_pfa = None if system_pfa is None else float(system_pfa)
        self._method: EstimationMethod = method
        self._n_simulations = int(n_simulations)
        self._n_bootstrap = int(n_bootstrap)
        self._tail_fraction = float(tail_fraction)
        self._random_state = random_state

    def calibrate(
        self,
        h0_scores: Mapping[str, ArrayLike],
        h1_scores: Mapping[str, ArrayLike] | None = None,
    ) -> CalibrationEvidence:
        """Calibrate every route and return auditable :class:`CalibrationEvidence`.

        Args:
            h0_scores: mapping of route name to its benign H0 score sample.
            h1_scores: optional mapping of route name to a signal (H1) sample;
                routes may be omitted.

        Returns:
            :class:`CalibrationEvidence` with thresholds, advisories, manifest,
            and report/serialization helpers.
        """
        if not isinstance(h0_scores, Mapping) or not h0_scores:
            raise ValidationError(
                "h0_scores must be a non-empty mapping of route name to score array"
            )
        if h1_scores is not None and not isinstance(h1_scores, Mapping):
            raise ValidationError("h1_scores must be a mapping of route name to score array")
        if h1_scores is not None:
            unknown = sorted(set(h1_scores) - set(h0_scores))
            if unknown:
                raise ValidationError(
                    f"h1_scores contains routes absent from h0_scores: {unknown}"
                )
        route_pfas = self._resolve_route_pfas(h0_scores)
        artifacts: dict[str, RouteArtifact] = {}
        advisories: dict[str, list[str]] = {}
        h0_arrays: dict[str, np.ndarray] = {}
        h1_arrays: dict[str, np.ndarray | None] = {}
        for route, h0 in h0_scores.items():
            result = estimate_threshold(
                h0,
                route_pfas[route],
                self._method,
                n_simulations=self._n_simulations,
                n_bootstrap=self._n_bootstrap,
                tail_fraction=self._tail_fraction,
                random_state=self._random_state,
            )
            h0_arr = np.asarray(h0, dtype=np.float64)
            h1_arr: np.ndarray | None = None
            h1_recall: float | None = None
            if h1_scores is not None and route in h1_scores:
                h1_arr = _validate_h1(h1_scores[route])
                h1_recall = float(np.mean(h1_arr > result.threshold))
            advisories[route] = _route_advisories(result, h1_recall)
            fingerprint = np.percentile(h0_arr, np.linspace(0.0, 100.0, 101))
            artifacts[route] = RouteArtifact(
                route=route,
                threshold=result.threshold,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                target_pfa=result.target_pfa,
                simulated_pfa=result.simulated_pfa,
                method=result.method,
                n_samples=result.n_samples,
                n_simulations=result.n_simulations,
                ks_pvalue=result.ks_pvalue,
                h1_recall=h1_recall,
                h0_quantiles=[float(v) for v in fingerprint],
                tail_params=dict(result.tail_params),
            )
            h0_arrays[route] = h0_arr
            h1_arrays[route] = h1_arr
        manifest = CalibrationManifest(
            cfar_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            system_pfa_budget=self._system_pfa,
            routes=artifacts,
        )
        return CalibrationEvidence(
            manifest=manifest,
            thresholds={route: artifact.threshold for route, artifact in artifacts.items()},
            advisories=advisories,
            h0_scores=h0_arrays,
            h1_scores=h1_arrays,
        )

    def _resolve_route_pfas(
        self, h0_scores: Mapping[str, ArrayLike]
    ) -> dict[str, float]:
        if self._system_pfa is not None:
            m = len(h0_scores)
            return {
                route: 1.0 - (1.0 - self._system_pfa) ** (1.0 / m)
                for route in h0_scores
            }
        if isinstance(self._target_pfa, Mapping):
            missing = sorted(set(h0_scores) - set(self._target_pfa))
            extra = sorted(set(self._target_pfa) - set(h0_scores))
            if missing:
                raise ValidationError(f"target_pfa mapping is missing routes: {missing}")
            if extra:
                raise ValidationError(
                    f"target_pfa mapping contains unknown routes: {extra}"
                )
            return {route: float(self._target_pfa[route]) for route in h0_scores}
        assert self._target_pfa is not None
        return {route: float(self._target_pfa) for route in h0_scores}


class ThresholdCalibrator:
    """Single-route convenience wrapper around :class:`MultiRouteCalibrator`."""

    default_route = "default"

    def __init__(
        self,
        target_pfa: float,
        method: EstimationMethod = "evt_montecarlo",
        *,
        n_simulations: int = 100_000,
        n_bootstrap: int = 300,
        tail_fraction: float = 0.15,
        random_state: int | np.random.Generator | None = 42,
    ) -> None:
        self._multi = MultiRouteCalibrator(
            target_pfa=target_pfa,
            method=method,
            n_simulations=n_simulations,
            n_bootstrap=n_bootstrap,
            tail_fraction=tail_fraction,
            random_state=random_state,
        )

    def calibrate(
        self, h0_scores: ArrayLike, h1_scores: ArrayLike | None = None
    ) -> CalibrationEvidence:
        """Calibrate the single route and return :class:`CalibrationEvidence`."""
        h1 = None if h1_scores is None else {self.default_route: h1_scores}
        return self._multi.calibrate({self.default_route: h0_scores}, h1_scores=h1)
