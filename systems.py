"""The systems under test.

Every system implements the same two methods and is handed exactly the same
material: the same conversation, the same turns, the same session dates, in the
same order. Nothing after `retrieve()` differs between systems: the same answer
model, the same answer prompt, the same temperature, the same seed, the same judge.

Contract:
    ingest(turns)      -> builds whatever index/memory the system needs
    retrieve(question)  -> a context string handed to the shared answerer
"""

import shutil
from pathlib import Path


def render(turns):
    return "\n".join(t["render"] for t in turns)


class FullContext:
    """Honest ceiling baseline: put the whole transcript in the prompt.

    Worth including because on LoCoMo-length conversations it is not obviously
    beatable, and a memory system that does not beat it has to justify itself on
    cost or latency instead of accuracy.
    """

    name = "full_context"

    def __init__(self, cfg):
        self.turns = []

    def ingest(self, turns):
        self.turns = turns

    def retrieve(self, q):
        return render(self.turns)


class RecencyWindow:
    """Naive baseline: the last N turns. What you get with no memory system at all."""

    name = "recency_window"

    def __init__(self, cfg):
        self.n = cfg["retrieval"]["recency_turns"]
        self.turns = []

    def ingest(self, turns):
        self.turns = turns

    def retrieve(self, q):
        return render(self.turns[-self.n :])


class BM25:
    """Lexical retrieval over the raw transcript. No LLM anywhere in the loop."""

    name = "bm25"

    def __init__(self, cfg):
        self.k = cfg["retrieval"]["top_k"]
        self.turns = []
        self.index = None

    @staticmethod
    def _tok(s):
        return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split() if w]

    def ingest(self, turns):
        from rank_bm25 import BM25Okapi

        self.turns = turns
        self.index = BM25Okapi([self._tok(t["render"]) for t in turns])

    def retrieve(self, q):
        scores = self.index.get_scores(self._tok(q))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.k]
        return render([self.turns[i] for i in sorted(top)])


class Mem0:
    """Mem0 OSS (`pip install mem0ai`), fully local: Ollama LLM + Ollama embedder
    + local Qdrant. No mem0 cloud, no API key, no network egress."""

    name = "mem0"

    def __init__(self, cfg):
        from mem0 import Memory

        mc = cfg["mem0"]
        store = Path(mc["vector_store_path"]).resolve()
        if store.exists():
            shutil.rmtree(store)  # fresh store every run, no leakage between runs
        self.cfg = cfg
        self.mc = mc
        self.user_id = "locomo"
        self.k = cfg["retrieval"]["top_k"]
        self.mem = Memory.from_config(
            {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": cfg["models"]["memory_llm"]["model"],
                        "temperature": cfg["models"]["memory_llm"]["temperature"],
                        "ollama_base_url": cfg["backend"]["base_url"],
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": cfg["models"]["embedder"]["model"],
                        "embedding_dims": cfg["models"]["embedder"]["dims"],
                        "ollama_base_url": cfg["backend"]["base_url"],
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "memharness",
                        "path": str(store),
                        "embedding_model_dims": cfg["models"]["embedder"]["dims"],
                    },
                },
            }
        )

    def ingest(self, turns):
        # One add() per session, in chronological order, which is how a real agent
        # would write to memory as the conversation happens.
        by_session = {}
        for t in turns:
            by_session.setdefault(t["session"], []).append(t)
        import time

        sids = sorted(by_session, key=lambda s: int(s))
        for i, sid in enumerate(sids, 1):
            msgs = []
            for t in by_session[sid]:
                if self.mc["role_mapping"] == "all_user":
                    role = "user"
                else:
                    role = "user" if t["is_a"] else "assistant"
                msgs.append({"role": role, "content": t["render"]})
            t0 = time.perf_counter()
            self.mem.add(messages=msgs, user_id=self.user_id, infer=self.mc["infer"])
            # Printed per session so a partial ingest is still a usable measurement
            # if the run has to be cut short.
            print(f"    ingest session {sid} ({i}/{len(sids)}, {len(msgs)} turns) "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)

    def retrieve(self, q):
        r = self.mem.search(query=q, top_k=self.k, filters={"user_id": self.user_id})
        items = r.get("results", r) if isinstance(r, dict) else r
        return "\n".join(f"- {m.get('memory', '')}" for m in items)


REGISTRY = {c.name: c for c in (FullContext, RecencyWindow, BM25, Mem0)}
