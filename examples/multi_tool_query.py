"""Multi-tool queries: one query, several routes (kb_agent + calculator).

``RuntimeGate.evaluate()`` returns ALL triggered routes ranked by score, so a
query that needs both the knowledge-base sub-agent and the calculator tool
triggers both. The joint Sidak budget already covers multi-triggering: the
guarantee bounds the probability that a benign query triggers ANY route —
the false-alarm event is counted per query, not per route firing — so
traffic that legitimately needs several tools never breaks the system PFA
budget.

Reuses the embedding router and the manifest calibrated by
``embedding_router.py`` (run that first). Then:

  1. routes real multi-tool queries (fact lookup + computation),
  2. dispatches in dependency order: kb_agent retrieves the fact, the
     calculator consumes it (stub retrieval, real arithmetic),
  3. shows the ``min_margin`` dial for suppressing weak secondary triggers.

Run:
    python examples/multi_tool_query.py
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from cfar import RuntimeGate, from_cosine

from embedding_router import MODEL_ID, ROUTE_DESCRIPTIONS
from real_model_router import OUT_DIR

# Each query needs a fact from the KB AND a computation on that fact.
# (fact = what the kb_agent stub "retrieves"; compute = what the calculator
# tool does with it — the arithmetic itself is real.)
MULTI_TOOL_QUERIES = [
    (
        "Search the knowledge base for the population of Tokyo, "
        "then multiply the number by twelve.",
        13_960_000,
        lambda fact: fact * 12,
    ),
    (
        "Look up the year the Eiffel Tower was completed in the knowledge "
        "base and calculate how many years ago that was.",
        1889,
        lambda fact: 2026 - fact,
    ),
    (
        "Find the speed of light in the knowledge base, then divide it by "
        "one hundred.",
        299_792_458,
        lambda fact: fact / 100,
    ),
    (
        "Retrieve the number of bones in the human body from the knowledge "
        "base and compute half of it.",
        206,
        lambda fact: fact / 2,
    ),
]


def main() -> None:
    manifest_path = OUT_DIR / "embedding_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("manifest missing: run examples/embedding_router.py first")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL_ID, device=device)
    gate = RuntimeGate.from_manifest(manifest_path)

    print("calibrated cutoffs (Sidak-split 5% system budget):")
    for route, artifact in gate.manifest.routes.items():
        print(f"  {route:10s} gamma = {artifact.threshold:.4f}")

    def route_scores(query: str) -> dict[str, float]:
        query_emb = model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores: dict[str, float] = {}
        for route, description in ROUTE_DESCRIPTIONS.items():
            route_emb = model.encode(
                [description], normalize_embeddings=True, show_progress_bar=False
            )[0]
            scores[route] = float(from_cosine(query_emb @ route_emb))
        return scores

    print("\nmulti-tool queries (fact lookup + computation):")
    for query, fact, compute in MULTI_TOOL_QUERIES:
        scores = route_scores(query)
        triggered = gate.evaluate(scores)  # min_margin=0: keep every trigger
        margined = gate.evaluate(scores, min_margin=0.05)  # suppress runner-ups
        shown = ", ".join(f"{route}: {score:.3f}" for route, score in scores.items())
        print(f"\n  query: {query!r}")
        print(f"    scores({shown})")
        print(f"    triggered (min_margin=0.00): {triggered}")
        print(f"    triggered (min_margin=0.05): {margined}")
        if set(triggered) == {"kb_agent", "calculator"}:
            value = compute(fact)
            print(
                f"    dispatch: kb_agent -> retrieved {fact:,}; "
                f"calculator -> {value:,.2f}"
            )
        elif triggered:
            print(f"    dispatch: {' -> '.join(triggered)} (single-route path)")
        else:
            print("    dispatch: answer_inline")


if __name__ == "__main__":
    main()
