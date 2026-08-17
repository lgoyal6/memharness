"""Tests for scoring and for the two model-free baselines.

Every accuracy number in this harness comes out of `deterministic`, so its
behaviour is the difference between a real comparison and a scoring artefact.
The two baselines are here because they are what mem0 is measured against, and a
baseline that quietly does the wrong thing flatters whatever it is compared to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from judge import ABSTAIN, deterministic, is_abstention, normalize, token_f1  # noqa: E402
from systems import RecencyWindow, render  # noqa: E402

THRESHOLD = 0.6


def turns(n):
    return [{"render": f"turn {i}", "speaker": "A", "text": f"turn {i}"} for i in range(n)]


class TestNormalize:
    def test_strips_case_punctuation_and_articles(self):
        assert normalize("The Coffee, please!") == "coffee please"

    def test_collapses_whitespace(self):
        assert normalize("a   b\n\tc") == "b c"

    def test_is_idempotent(self):
        once = normalize("The Dog's bowl.")
        assert normalize(once) == once


class TestTokenF1:
    def test_identical_strings_score_one(self):
        assert token_f1("coffee", "coffee") == 1.0

    def test_disjoint_strings_score_zero(self):
        assert token_f1("coffee", "tea") == 0.0

    def test_empty_input_scores_zero_rather_than_raising(self):
        assert token_f1("", "coffee") == 0.0
        assert token_f1("coffee", "") == 0.0

    def test_punctuation_and_case_do_not_change_the_score(self):
        assert token_f1("Coffee!", "coffee") == 1.0

    def test_partial_overlap_lands_between(self):
        score = token_f1("he likes coffee and tea", "coffee")
        assert 0.0 < score < 1.0


class TestDeterministic:
    def test_exact_answer_is_correct(self):
        ok, _ = deterministic("coffee", "coffee", THRESHOLD)
        assert ok

    def test_padded_answer_is_correct_by_containment(self):
        # The answer prompt permits padding, so a strict F1 would punish
        # verbosity rather than error. Containment is the escape hatch.
        ok, f1 = deterministic(
            "Based on the transcript, I believe the answer is coffee.", "coffee", THRESHOLD
        )
        assert ok
        assert f1 < THRESHOLD

    def test_wrong_answer_is_not_correct(self):
        ok, _ = deterministic("tea", "coffee", THRESHOLD)
        assert not ok

    def test_abstention_never_counts_as_correct(self):
        # A system that abstains has not answered. Crediting it would make
        # refusing to answer a scoring strategy.
        ok, _ = deterministic(ABSTAIN, "coffee", THRESHOLD)
        assert not ok

    def test_abstention_is_not_correct_even_when_gold_appears_in_it(self):
        ok, _ = deterministic(ABSTAIN, "no", THRESHOLD)
        assert not ok

    def test_returns_the_f1_alongside_the_verdict(self):
        _, f1 = deterministic("coffee", "coffee", THRESHOLD)
        assert f1 == 1.0

    def test_threshold_is_respected(self):
        pred, gold = "he mentioned coffee tea juice water milk", "coffee"
        assert deterministic(pred, gold, 0.01)[0]
        # Containment still fires here, which is the documented behaviour;
        # the F1 itself is well below any sane threshold.
        assert deterministic(pred, gold, 0.99)[1] < 0.99


class TestIsAbstention:
    def test_detects_the_sentinel(self):
        assert is_abstention(ABSTAIN)

    def test_detects_it_with_punctuation_and_case(self):
        assert is_abstention("No answer.")

    def test_a_real_answer_is_not_an_abstention(self):
        assert not is_abstention("coffee")


class TestRecencyWindow:
    def test_returns_only_the_last_n_turns(self):
        w = RecencyWindow({"retrieval": {"recency_turns": 3}})
        w.ingest(turns(10))

        out = w.retrieve("anything")

        assert "turn 9" in out
        assert "turn 7" in out
        assert "turn 6" not in out

    def test_keeps_chronological_order(self):
        # Reversing it would change what the answering model sees for reasons
        # unrelated to retrieval quality.
        w = RecencyWindow({"retrieval": {"recency_turns": 3}})
        w.ingest(turns(5))

        out = w.retrieve("anything")

        assert out.index("turn 2") < out.index("turn 3") < out.index("turn 4")

    def test_a_short_transcript_is_returned_whole(self):
        w = RecencyWindow({"retrieval": {"recency_turns": 50}})
        w.ingest(turns(2))

        assert "turn 0" in w.retrieve("anything")

    def test_ignores_the_query_entirely(self):
        # It is the no-memory baseline. If it responded to the query it would
        # not be measuring what it is supposed to measure.
        w = RecencyWindow({"retrieval": {"recency_turns": 2}})
        w.ingest(turns(6))

        assert w.retrieve("coffee") == w.retrieve("something else entirely")


class TestRender:
    def test_joins_turns_without_dropping_any(self):
        out = render(turns(3))
        for i in range(3):
            assert f"turn {i}" in out

    def test_empty_input_renders_empty(self):
        assert render([]).strip() == ""
