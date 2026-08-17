"""Tests for the token meter.

The ingest-cost column is the one number in this harness that nobody else
publishes, and every claim built on it depends on two things being true: that a
call is attributed to the system and stage that was active when it happened, and
that ingest and query totals do not leak into each other. If attribution is
wrong, the crossover arithmetic is wrong, and it fails silently because the
totals still add up.

These run without a live Ollama server. Only the python package is imported.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meter import Call, Meter  # noqa: E402


@pytest.fixture()
def meter():
    return Meter()


def test_records_nothing_before_any_call(meter):
    assert meter.totals() == {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "embed_tokens": 0,
        "wall_ms": 0,
    }


def test_attributes_a_call_to_the_active_scope(meter):
    with meter.scope("mem0", "ingest"):
        meter.record("chat", "m", 100, 20, 5.0)

    call = meter.calls[0]
    assert (call.system, call.stage) == ("mem0", "ingest")


def test_separates_ingest_from_query_totals(meter):
    # The whole finding is that these two are different sizes. If they leak into
    # each other the crossover is meaningless.
    with meter.scope("mem0", "ingest"):
        meter.record("chat", "m", 1000, 500, 1.0)
    with meter.scope("mem0", "answer"):
        meter.record("chat", "m", 10, 5, 1.0)

    assert meter.totals(system="mem0", stage="ingest")["prompt_tokens"] == 1000
    assert meter.totals(system="mem0", stage="answer")["prompt_tokens"] == 10
    assert meter.totals(system="mem0")["prompt_tokens"] == 1010


def test_separates_one_system_from_another(meter):
    with meter.scope("mem0", "ingest"):
        meter.record("chat", "m", 500, 0, 1.0)
    with meter.scope("bm25", "ingest"):
        meter.record("chat", "m", 7, 0, 1.0)

    assert meter.totals(system="mem0")["prompt_tokens"] == 500
    assert meter.totals(system="bm25")["prompt_tokens"] == 7


def test_nested_scopes_restore_the_outer_one(meter):
    # mem0's ingest calls happen inside our ingest scope; a nested retrieve must
    # not permanently steal attribution from it.
    with meter.scope("mem0", "ingest"):
        meter.record("chat", "m", 1, 0, 1.0)
        with meter.scope("mem0", "retrieve"):
            meter.record("embed", "e", 2, 0, 1.0)
        meter.record("chat", "m", 4, 0, 1.0)

    assert meter.totals(stage="ingest")["prompt_tokens"] == 5
    assert meter.totals(stage="retrieve")["prompt_tokens"] == 2


def test_scope_restores_even_when_the_call_raises(meter):
    # A failed model call must not leave every later call misattributed.
    with meter.scope("mem0", "ingest"):
        with pytest.raises(RuntimeError):
            with meter.scope("mem0", "answer"):
                raise RuntimeError("upstream refused")
        meter.record("chat", "m", 9, 0, 1.0)

    assert meter.totals(stage="ingest")["prompt_tokens"] == 9
    assert meter.totals(stage="answer")["calls"] == 0


def test_embed_tokens_exclude_chat_tokens(meter):
    # Embeddings are priced differently, so collapsing them into prompt_tokens
    # would misprice every system that embeds at ingest.
    with meter.scope("mem0", "ingest"):
        meter.record("chat", "m", 100, 10, 1.0)
        meter.record("embed", "e", 40, 0, 1.0)

    totals = meter.totals(stage="ingest")
    assert totals["prompt_tokens"] == 140
    assert totals["embed_tokens"] == 40


def test_missing_token_counts_become_zero_not_none(meter):
    # Ollama omits the field on some responses. A None here would poison every
    # sum downstream rather than costing one call.
    with meter.scope("mem0", "answer"):
        meter.record("chat", "m", None, None, 1.0)

    assert meter.totals()["prompt_tokens"] == 0
    assert meter.totals()["completion_tokens"] == 0
    assert isinstance(meter.calls[0], Call)


def test_latencies_are_per_system_and_stage(meter):
    with meter.scope("mem0", "answer"):
        meter.record("chat", "m", 1, 1, 12.5)
        meter.record("chat", "m", 1, 1, 7.5)
    with meter.scope("bm25", "answer"):
        meter.record("chat", "m", 1, 1, 99.0)

    assert meter.latencies("mem0", "answer") == [12.5, 7.5]
    assert meter.latencies("bm25", "answer") == [99.0]


def test_a_call_outside_any_scope_is_still_recorded(meter):
    # Losing it would undercount. It lands under the placeholder system so it
    # shows up as unattributed rather than vanishing.
    meter.record("chat", "m", 3, 0, 1.0)

    assert meter.totals()["calls"] == 1
    assert meter.calls[0].system == "?"


def test_ingest_dominates_query_in_the_shape_the_finding_claims(meter):
    """The headline is that reading a conversation costs far more than answering
    questions about it. This pins the arithmetic that claim is computed from."""
    with meter.scope("mem0", "ingest"):
        meter.record("chat", "m", 61_705, 14_745, 1.0)
    for _ in range(20):
        with meter.scope("mem0", "answer"):
            meter.record("chat", "m", 296, 6, 1.0)

    ingest = meter.totals(system="mem0", stage="ingest")
    answer = meter.totals(system="mem0", stage="answer")
    ingest_total = ingest["prompt_tokens"] + ingest["completion_tokens"]
    answer_total = answer["prompt_tokens"] + answer["completion_tokens"]

    assert ingest_total == 76_450
    assert answer_total == 6_040
    assert ingest_total / answer_total > 12
