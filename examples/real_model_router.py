"""Real-model integration demo for cfar.

Orchestrator with 1 tool (calculator) and 1 sub-agent (knowledge-base
endpoint). The decision model is a real open-source LLM:
Qwen2.5-0.5B-Instruct (Apache-2.0, ungated). For every query it answers two
yes/no routing questions; the yes-vs-no log-odds (a logit) are converted to
bounded [0, 1] confidence scores via ``cfar.from_logits``, and cfar derives
the cutoffs from the score distributions.

Real data (downloaded on first run, cached under ``examples/data/``):
  H0  benign      Databricks Dolly-15k instructions (Apache-2.0) filtered to
                  non-numeric, non-question creative/brainstorm tasks
  H1  calculator  GSM8K grade-school math problems
  H1  kb_agent    SQuAD v2 dev factual questions

Mocked parts (stated honestly): route labels come from dataset provenance
(heuristic ground truth, not per-query human annotation), and the tool and
sub-agent are stubs. The model, its scores, and the queries are real.

Example-only dependencies (the cfar package itself needs only numpy+scipy):
    pip install torch transformers

Run:
    python examples/real_model_router.py
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfar import DriftMonitor, MultiRouteCalibrator, RuntimeGate, from_logits

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "output"

SOURCES: dict[str, list[str]] = {
    "gsm8k_test.jsonl": [
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
    ],
    "squad_dev.json": [
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
    ],
    "dolly_15k.jsonl": [
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
    ],
}

QUESTION_START = re.compile(
    r"^(what|who|whom|when|where|which|why|how|name|is|are|was|were|does|do|"
    r"did|can|could|would|should|will|has|have)\b",
    re.IGNORECASE,
)

ROUTE_PROMPTS = {
    "calculator": (
        "You are the router of an AI assistant. Does the user query below "
        "require performing an arithmetic calculation with a calculator tool? "
        "Answer strictly yes or no.\n\nQuery: {q}"
    ),
    "kb_agent": (
        "You are the router of an AI assistant. Does the user query below "
        "require looking up facts in a knowledge base before it can be "
        "answered? Answer strictly yes or no.\n\nQuery: {q}"
    ),
}

# Same routing intent, different wording: simulates a post-deploy prompt change.
DRIFT_PROMPT = (
    "Quick check: is a calculator needed to handle this? Reply with yes or "
    "no only.\nUser query: {q}"
)


def fetch(name: str) -> Path:
    """Download a dataset on first use and cache it under examples/data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / name
    if target.exists():
        return target
    for url in SOURCES[name]:
        try:
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, target)
            return target
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            print(f"  failed ({exc}); trying next mirror")
    raise RuntimeError(f"could not download {name}")


def load_corpus() -> tuple[list[str], list[str], list[str]]:
    """Return (benign, math, knowledge) query lists, deduplicated and shuffled."""
    rng = random.Random(13)

    qa_categories = {"open_qa", "general_qa", "closed_qa"}
    benign: list[str] = []
    seen: set[str] = set()
    for line in fetch("dolly_15k.jsonl").read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        instruction = item["instruction"].strip()
        if item.get("context"):
            continue  # tasks with extra context are noisier; keep it simple
        if item.get("category") in qa_categories:
            continue  # factual questions belong to the KB route, not benign
        if any(char.isdigit() for char in instruction):
            continue  # numeric tasks would contaminate the calculator H0
        if QUESTION_START.match(instruction):
            continue
        if not 8 <= len(instruction.split()) <= 60:
            continue
        key = instruction.lower()
        if key in seen:
            continue
        seen.add(key)
        benign.append(instruction)
    rng.shuffle(benign)

    math = [
        json.loads(line)["question"].strip()
        for line in fetch("gsm8k_test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rng.shuffle(math)

    squad = json.loads(fetch("squad_dev.json").read_text(encoding="utf-8"))
    knowledge = list(
        {
            qa["question"].strip()
            for article in squad["data"]
            for paragraph in article["paragraphs"]
            for qa in paragraph["qas"]
        }
    )
    rng.shuffle(knowledge)
    return benign, math, knowledge


class LogitRouter:
    """Real decision model: a small instruct LLM scoring yes/no log-odds."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        print(f"loading {model_id} on {self.device} ({dtype}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.yes_id = self._token_id("yes")
        self.no_id = self._token_id("no")

    def _token_id(self, word: str) -> int:
        ids = self.tok(word, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"{word!r} is not a single token: {ids}")
        return ids[0]

    @torch.no_grad()
    def logodds(
        self, queries: list[str], template: str, batch_size: int = 32
    ) -> np.ndarray:
        """Yes-vs-no log-odds per query (higher = the route should fire)."""
        scores: list[float] = []
        for start in range(0, len(queries), batch_size):
            batch = queries[start : start + batch_size]
            prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": template.format(q=q)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for q in batch
            ]
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**enc).logits[:, -1, :]
            diff = logits[:, self.yes_id] - logits[:, self.no_id]
            scores.extend(diff.float().tolist())
        return np.asarray(scores, dtype=np.float64)


def main() -> None:
    on_mps = torch.backends.mps.is_available()
    n_benign, n_h1, n_heldout = (600, 250, 150) if on_mps else (350, 150, 100)

    benign, math, knowledge = load_corpus()
    print(
        f"corpus: {len(benign)} benign | {len(math)} math | "
        f"{len(knowledge)} knowledge queries"
    )

    benign_cal = benign[:n_benign]
    benign_held = benign[n_benign : n_benign + n_heldout]
    benign_demo = benign[n_benign + n_heldout : n_benign + n_heldout + 3]
    math_cal = math[:n_h1]
    math_demo = math[n_h1 : n_h1 + 3]
    kb_cal = knowledge[:n_h1]
    kb_demo = knowledge[n_h1 : n_h1 + 3]

    router = LogitRouter()

    start = time.perf_counter()
    # H0: benign traffic scored by both routing heads.
    h0_logits = {
        "calculator": router.logodds(benign_cal, ROUTE_PROMPTS["calculator"]),
        "kb_agent": router.logodds(benign_cal, ROUTE_PROMPTS["kb_agent"]),
    }
    # H1: genuine route traffic, for recall auditing.
    h1_logits = {
        "calculator": router.logodds(math_cal, ROUTE_PROMPTS["calculator"]),
        "kb_agent": router.logodds(kb_cal, ROUTE_PROMPTS["kb_agent"]),
    }
    print(
        f"scored {2 * (n_benign + n_h1)} queries in "
        f"{time.perf_counter() - start:.1f}s"
    )

    h0_scores = {route: from_logits(values) for route, values in h0_logits.items()}
    h1_scores = {route: from_logits(values) for route, values in h1_logits.items()}

    print("\nscore overview (from_logits of yes/no log-odds):")
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
    manifest_path = evidence.save_manifest(OUT_DIR / "manifest.json")
    evidence.export_report(OUT_DIR / "report.html")
    print(f"\nmanifest: {manifest_path}")

    # Runtime: gate real held-out queries through the stub orchestrator.
    gate = RuntimeGate.from_manifest(manifest_path)
    handlers = {"calculator": "calculator_tool", "kb_agent": "kb_agent_endpoint"}

    print("\nruntime routing on held-out real queries:")
    traffic = (
        [("benign  ", q) for q in benign_demo]
        + [("math    ", q) for q in math_demo]
        + [("knowledge", q) for q in kb_demo]
    )
    for kind, query in traffic:
        logits = {
            route: float(router.logodds([query], prompt)[0])
            for route, prompt in ROUTE_PROMPTS.items()
        }
        scores = {route: float(from_logits(value)) for route, value in logits.items()}
        triggered = gate.evaluate(scores)
        action = (
            " -> ".join(handlers[route] for route in triggered)
            if triggered
            else "answer_inline"
        )
        shown = ", ".join(f"{route}: {score:.3f}" for route, score in scores.items())
        print(f"  [{kind}] {query[:62]!r}")
        print(f"      scores({shown}) -> {action}")

    # Ops: drift monitoring after a (simulated) prompt-template deploy.
    monitor = DriftMonitor.from_manifest(manifest_path, alpha=0.01)
    control = from_logits(router.logodds(benign_held, ROUTE_PROMPTS["calculator"]))
    changed = from_logits(router.logodds(benign_held, DRIFT_PROMPT))
    control_report = monitor.check("calculator", control)
    changed_report = monitor.check("calculator", changed)
    print("\ndrift monitoring (calculator head, held-out benign traffic):")
    print(
        f"  same prompt   : detected={control_report.drift_detected} "
        f"ks_p={control_report.ks_pvalue:.3f}"
    )
    print(
        f"  changed prompt: detected={changed_report.drift_detected} "
        f"ks_p={changed_report.ks_pvalue:.2e} "
        f"wasserstein={changed_report.wasserstein_distance:.3f}"
    )
    if changed_report.drift_detected:
        print("  -> old cutoffs no longer hold their PFA guarantee: recalibrate")


if __name__ == "__main__":
    main()
