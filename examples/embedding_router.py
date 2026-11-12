"""Second real decision model for cfar: an embedding-based semantic router.

Companion to ``real_model_router.py`` (LLM logit router). Same real data,
same cfar pipeline, but the decision model is the standard semantic-router
design: sentence-transformers all-MiniLM-L6-v2 (open-source, 22M params)
computes the cosine similarity between each query and a route description;
``cfar.from_cosine`` normalizes it to [0, 1] and cfar derives the cutoffs.

Real data (cached under ``examples/data/`` by real_model_router.py):
  H0  benign      Databricks Dolly-15k instructions (filtered)
  H1  calculator  GSM8K grade-school math problems
  H1  kb_agent    SQuAD v2 dev factual questions

Example-only dependency (the cfar package itself needs only numpy+scipy):
    pip install sentence-transformers

Run:
    python examples/embedding_router.py
"""

from __future__ import annotations

import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from cfar import DriftMonitor, MultiRouteCalibrator, RuntimeGate, from_cosine

from real_model_router import OUT_DIR, load_corpus

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

ROUTE_DESCRIPTIONS = {
    "calculator": (
        "a user request that requires performing arithmetic calculations "
        "or solving a math problem"
    ),
    "kb_agent": (
        "a factual question that requires looking up information in a "
        "knowledge base to answer"
    ),
}

# Same routing intent, reworded: simulates a post-deploy route-description edit.
DRIFT_DESCRIPTION = "a user request that needs some math done"


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    n_benign, n_h1, n_heldout = (600, 250, 150) if device == "mps" else (350, 150, 100)

    benign, math, knowledge = load_corpus()
    benign_cal = benign[:n_benign]
    benign_held = benign[n_benign : n_benign + n_heldout]
    benign_demo = benign[n_benign + n_heldout : n_benign + n_heldout + 3]
    math_cal = math[:n_h1]
    math_demo = math[n_h1 : n_h1 + 3]
    kb_cal = knowledge[:n_h1]
    kb_demo = knowledge[n_h1 : n_h1 + 3]

    print(f"loading {MODEL_ID} on {device} ...")
    model = SentenceTransformer(MODEL_ID, device=device)

    def route_scores(queries: list[str], description: str) -> np.ndarray:
        query_emb = model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )
        route_emb = model.encode(
            [description], normalize_embeddings=True, show_progress_bar=False
        )[0]
        return from_cosine(query_emb @ route_emb)

    start = time.perf_counter()
    h0_scores = {
        "calculator": route_scores(benign_cal, ROUTE_DESCRIPTIONS["calculator"]),
        "kb_agent": route_scores(benign_cal, ROUTE_DESCRIPTIONS["kb_agent"]),
    }
    h1_scores = {
        "calculator": route_scores(math_cal, ROUTE_DESCRIPTIONS["calculator"]),
        "kb_agent": route_scores(kb_cal, ROUTE_DESCRIPTIONS["kb_agent"]),
    }
    print(
        f"scored {2 * (n_benign + n_h1)} queries in "
        f"{time.perf_counter() - start:.1f}s"
    )

    print("\nscore overview (from_cosine of query/route similarity):")
    for route in ("calculator", "kb_agent"):
        h0, h1 = h0_scores[route], h1_scores[route]
        print(
            f"  {route:10s} H0 mean={h0.mean():.3f} p95={np.quantile(h0, 0.95):.3f}"
            f" | H1 mean={h1.mean():.3f} p05={np.quantile(h1, 0.05):.3f}"
        )

    # Calibrate: at most 5% of benign queries may trigger ANY route (Sidak).
    evidence = MultiRouteCalibrator(system_pfa=0.05).calibrate(
        h0_scores, h1_scores=h1_scores
    )
    print()
    print(evidence.summary())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = evidence.save_manifest(OUT_DIR / "embedding_manifest.json")
    evidence.export_report(OUT_DIR / "embedding_report.html")
    print(f"\nmanifest: {manifest_path}")

    # Runtime: gate the same held-out real queries.
    gate = RuntimeGate.from_manifest(manifest_path)
    handlers = {"calculator": "calculator_tool", "kb_agent": "kb_agent_endpoint"}

    print("\nruntime routing on held-out real queries:")
    traffic = (
        [("benign  ", q) for q in benign_demo]
        + [("math    ", q) for q in math_demo]
        + [("knowledge", q) for q in kb_demo]
    )
    for kind, query in traffic:
        scores = {
            route: float(route_scores([query], description)[0])
            for route, description in ROUTE_DESCRIPTIONS.items()
        }
        triggered = gate.evaluate(scores)
        action = (
            " -> ".join(handlers[route] for route in triggered)
            if triggered
            else "answer_inline"
        )
        shown = ", ".join(f"{route}: {score:.3f}" for route, score in scores.items())
        print(f"  [{kind}] {query[:62]!r}")
        print(f"      scores({shown}) -> {action}")

    # Ops: drift monitoring after a (simulated) route-description edit.
    monitor = DriftMonitor.from_manifest(manifest_path, alpha=0.01)
    control = route_scores(benign_held, ROUTE_DESCRIPTIONS["calculator"])
    changed = route_scores(benign_held, DRIFT_DESCRIPTION)
    control_report = monitor.check("calculator", control)
    changed_report = monitor.check("calculator", changed)
    print("\ndrift monitoring (calculator head, held-out benign traffic):")
    print(
        f"  same description   : detected={control_report.drift_detected} "
        f"ks_p={control_report.ks_pvalue:.3f}"
    )
    print(
        f"  changed description: detected={changed_report.drift_detected} "
        f"ks_p={changed_report.ks_pvalue:.2e} "
        f"wasserstein={changed_report.wasserstein_distance:.3f}"
    )
    if changed_report.drift_detected:
        print("  -> old cutoffs no longer hold their PFA guarantee: recalibrate")


if __name__ == "__main__":
    main()
