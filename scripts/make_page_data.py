"""Build the JSON the results page reads.

Everything comes out of the two run files in results/. The page does the
amortisation arithmetic itself, from per-question costs derived here, so a
reader can move the question count and watch the crossover move rather than
being handed one number to trust.

    python3 scripts/make_page_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data"
RUNS = ["results/mem0-n20.json", "results/baselines-n20.json"]

# The order they are worth comparing in: the thing being tested, the cheap
# baseline it has to beat, then the two obvious alternatives.
ORDER = ["mem0", "bm25", "recency_window", "full_context"]
LABELS = {
    "mem0": "mem0",
    "bm25": "keyword search",
    "recency_window": "last N turns",
    "full_context": "the whole conversation",
}


def main() -> None:
    systems, pricing, dataset = {}, None, None
    for rel in RUNS:
        run = json.loads((ROOT / rel).read_text())
        pricing = pricing or run["config"]["pricing"]
        dataset = dataset or run["dataset"]
        for name, s in run["summary"].items():
            n = s["n"]
            ingest_tok = s["ingest_prompt_tokens"] + s["ingest_completion_tokens"]
            systems[name] = {
                "key": name,
                "label": LABELS.get(name, name),
                "n_questions": n,
                # Charged once, before any question is asked.
                "ingest_tokens": ingest_tok,
                "ingest_usd": s["usd_ingest_DERIVED"],
                "ingest_llm_calls": s["ingest_llm_calls"],
                "ingest_wall_s": s["ingest_wall_s"],
                # Charged again for every question.
                "per_q_tokens": (s["query_prompt_tokens"] + s["query_completion_tokens"]) / n,
                "per_q_usd": s["usd_query_total_DERIVED"] / n,
                "accuracy": s["accuracy_llm_judge"],
                "accuracy_strict": s["accuracy_deterministic"],
                "solved": s["solved_llm_judge"],
                "p50_retrieve_ms": s["p50_retrieve_ms"],
                "p50_query_ms": s["p50_query_ms"],
                "mean_context_chars": s["mean_context_chars"],
                "usd_per_solved": s["usd_per_solved_task_DERIVED"],
            }

    payload = {
        "dataset": dataset,
        # A price sheet is an input, not a measurement. The page says so where
        # it shows dollars.
        "pricing": pricing,
        "systems": [systems[k] for k in ORDER if k in systems],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.json"
    path.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"{path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} kB")

    m, b = systems["mem0"], systems["bm25"]
    by_tok = m["ingest_tokens"] / (b["per_q_tokens"] - m["per_q_tokens"])
    print(f"crossover against keyword search, counting every question: {by_tok:.0f}")


if __name__ == "__main__":
    main()
