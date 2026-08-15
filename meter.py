"""Metered Ollama client.

Every chat/embed call made by anything in this process, including the calls mem0
makes inside its own extraction pipeline, is recorded with real token counts
(Ollama returns prompt_eval_count / eval_count) and wall-clock latency, attributed
to whichever (system, stage) is currently active.

This is what makes the write-side cost of a memory system visible. Published agent
memory tables report retrieval accuracy and search latency; the tokens burned at
ingest usually do not appear anywhere.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import ollama


@dataclass
class Call:
    system: str
    stage: str  # ingest | retrieve | answer | judge
    kind: str  # chat | embed
    model: str
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float


@dataclass
class Meter:
    calls: list = field(default_factory=list)
    _system: str = "?"
    _stage: str = "?"

    @contextmanager
    def scope(self, system: str, stage: str):
        prev = (self._system, self._stage)
        self._system, self._stage = system, stage
        try:
            yield
        finally:
            self._system, self._stage = prev

    def record(self, kind, model, pt, ct, wall_ms):
        self.calls.append(
            Call(self._system, self._stage, kind, model, int(pt or 0), int(ct or 0), wall_ms)
        )

    def totals(self, system=None, stage=None):
        sel = [
            c
            for c in self.calls
            if (system is None or c.system == system) and (stage is None or c.stage == stage)
        ]
        return {
            "calls": len(sel),
            "prompt_tokens": sum(c.prompt_tokens for c in sel),
            "completion_tokens": sum(c.completion_tokens for c in sel),
            "embed_tokens": sum(c.prompt_tokens for c in sel if c.kind == "embed"),
            "wall_ms": sum(c.wall_ms for c in sel),
        }

    def latencies(self, system, stage):
        return [c.wall_ms for c in self.calls if c.system == system and c.stage == stage]


METER = Meter()


class MeteredClient(ollama.Client):
    """Drop-in ollama.Client that reports every call to the global METER."""

    def chat(self, *a, **kw):
        t0 = time.perf_counter()
        r = super().chat(*a, **kw)
        ms = (time.perf_counter() - t0) * 1000
        model = kw.get("model") or (a[0] if a else "?")
        METER.record("chat", model, getattr(r, "prompt_eval_count", 0), getattr(r, "eval_count", 0), ms)
        return r

    def embed(self, *a, **kw):
        t0 = time.perf_counter()
        r = super().embed(*a, **kw)
        ms = (time.perf_counter() - t0) * 1000
        model = kw.get("model") or (a[0] if a else "?")
        METER.record("embed", model, getattr(r, "prompt_eval_count", 0), 0, ms)
        return r


def install():
    """Route mem0's internal Ollama clients through the meter.

    mem0 does `from ollama import Client` at module import time, so patching
    `ollama.Client` alone is not enough; the already-bound names are replaced too.
    """
    ollama.Client = MeteredClient
    import mem0.embeddings.ollama as _e
    import mem0.llms.ollama as _l

    _l.Client = MeteredClient
    _e.Client = MeteredClient
