"""The grounding checker: does a cited clause actually support an answer sentence?

A sentence is supported when, after normalisation, it is a substring of the cited clause, or when its
content tokens appear in the clause in the same order (longest common subsequence ratio) above a
threshold, provided that every number, amount, percentage and date in the sentence also appears in the
clause and that no negation word was introduced. Numbers are checked exactly because they are what an
LLM most often gets subtly wrong, and a paraphrase that changes "30 days" to "45 days" must fail even
though almost every token still matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clausewise.schemas import Clause
from clausewise.textutil import STOPWORDS, stem, tokenize

NEGATIONS: frozenset[str] = frozenset({"not", "no", "never", "neither", "nor", "without", "unless", "except"})
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_PUNCT_RE = re.compile(r"[^a-z0-9%\s]")
_WS_RE = re.compile(r"\s+")
_GROUNDING_STOP = STOPWORDS - NEGATIONS


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace so wrapping and quoting do not matter."""
    text = text.lower().replace("\n", " ")
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def grounding_tokens(text: str) -> list[str]:
    """Stemmed content tokens that keep negation words, so an inserted "not" is visible to the checker."""
    return [stem(t) for t in tokenize(text) if t not in _GROUNDING_STOP]


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in _NUMBER_RE.findall(text)}


def lcs_alignment(a: list[str], b: list[str]) -> list[int]:
    """Indices in ``b`` of a longest common subsequence with ``a`` (classic O(len(a) * len(b)) table).

    The indices are what the negation check needs: they delimit the stretch of the clause that the
    sentence was taken from, so a "not" the clause has inside that stretch, or one the sentence added,
    can be counted.
    """
    if not a or not b:
        return []
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            table[i][j] = table[i + 1][j + 1] + 1 if a[i] == b[j] else max(table[i + 1][j], table[i][j + 1])
    out: list[int] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(j)
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    return len(lcs_alignment(a, b))


@dataclass(frozen=True)
class Support:
    """Checker verdict for one (sentence, clause) pair."""

    score: float
    reason: str

    @property
    def supported(self) -> bool:
        """True when the sentence is verbatim or an ordered token match above the threshold."""
        return self.score >= 1.0 or self.reason == "ordered-token-match"


def support_score(sentence: str, clause_text: str, threshold: float = 0.85) -> Support:
    """Score how well ``clause_text`` supports ``sentence``; 1.0 is a verbatim match, 0.0 a contradiction."""
    s_norm, c_norm = normalize(sentence), normalize(clause_text)
    if not s_norm:
        return Support(0.0, "empty")
    if s_norm in c_norm:
        return Support(1.0, "verbatim")
    missing_numbers = _numbers(s_norm) - _numbers(c_norm)
    if missing_numbers:
        return Support(0.0, f"numbers not in clause: {', '.join(sorted(missing_numbers))}")
    s_tokens, c_tokens = grounding_tokens(sentence), grounding_tokens(clause_text)
    if not s_tokens:
        return Support(0.0, "no content tokens")
    c_set = set(c_tokens)
    new_negations = [t for t in s_tokens if t in NEGATIONS and t not in c_set]
    if new_negations:
        return Support(0.0, f"negation not in clause: {', '.join(new_negations)}")
    aligned = lcs_alignment(s_tokens, c_tokens)
    ratio = len(aligned) / len(s_tokens)
    if ratio < threshold:
        return Support(ratio, "insufficient overlap")
    span_negations = sum(1 for t in c_tokens[aligned[0] : aligned[-1] + 1] if t in NEGATIONS)
    sentence_negations = sum(1 for t in s_tokens if t in NEGATIONS)
    if span_negations != sentence_negations:
        return Support(0.0, f"negation mismatch: clause span has {span_negations}, sentence has {sentence_negations}")
    return Support(ratio, "ordered-token-match")


def best_support(sentence: str, clauses: list[Clause], threshold: float = 0.85) -> tuple[Support, Clause | None]:
    """The best supporting clause among the cited ones."""
    best: tuple[Support, Clause | None] = (Support(0.0, "no citations"), None)
    for clause in clauses:
        sup = support_score(sentence, clause.text, threshold)
        if sup.score > best[0].score or best[1] is None:
            best = (sup, clause)
        if sup.score >= 1.0:
            break
    return best
