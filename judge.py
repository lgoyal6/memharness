"""Scoring. Two scorers, both applied to every system, both reported.

`deterministic` needs no model at all, so the harness produces a defensible table
even with zero API keys and no GPU. `llm_judge` is pluggable: point `models.judge`
at any Ollama model, or swap the `chat` callable for a hosted one.
"""

import re
import string
from collections import Counter

_ARTICLES = {"a", "an", "the"}


def normalize(s):
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def deterministic(pred, gold, threshold):
    """Correct if gold is contained in the prediction, or token-F1 clears threshold.

    Containment is included because the answer prompt permits padding words and
    a strict F1 would punish a system for verbosity rather than for being wrong.
    """
    if is_abstention(pred):
        return False, 0.0
    f1 = token_f1(pred, gold)
    contained = bool(normalize(gold)) and normalize(gold) in normalize(pred)
    return (contained or f1 >= threshold), f1


ABSTAIN = "NO ANSWER"


def is_abstention(pred):
    return normalize(pred) in ("no answer", "no answer no answer")


def llm_judge(chat, prompt_tmpl, question, gold, pred):
    # An abstention is never correct. Applied identically to every system, and it
    # exists because small judges will happily mark "NO ANSWER" as CORRECT.
    if is_abstention(pred):
        return False
    out = chat(prompt_tmpl.format(question=question, gold=gold, prediction=pred))
    return out.strip().upper().startswith("CORRECT")
