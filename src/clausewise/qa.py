"""Grounded extractive question answering over one contract.

The answerer never writes a sentence of its own. It retrieves candidate clauses, scores their sentences
against the question and returns the best sentence verbatim with its clause id. Scoring is an
IDF-weighted token overlap computed over the document itself: a question word that appears in every
clause ("agreement", "party") carries almost no weight, a word that appears in one clause ("uptime",
"deposit") decides the match. Questions that ask for a value (a duration, an amount, a date, a
jurisdiction, a percentage) are restricted to sentences that actually contain a value of that type; when
the retrieved clauses hold none, the search widens to the whole document before the answerer gives up.
Yes/no questions are answered by the risk rules on the clause that owns the topic, so the answer and the
risk report can never disagree. Every returned sentence still passes through the grounding checker; that
checker is the contract the optional LLM path must honour too.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from clausewise.config import Settings, load_settings
from clausewise.entities import extract_entities, extract_roles
from clausewise.grounding import best_support
from clausewise.retrieval import ClauseRetriever
from clausewise.risk import detect_clause
from clausewise.schemas import Answer, AnswerSentence, Clause, Entity
from clausewise.segment import body_text
from clausewise.textutil import content_tokens, normalize_ws, split_sentences

_TYPE_RULES: tuple[tuple[str, str], ...] = (
    ("jurisdiction", r"\bgovern|\blaws?\b|jurisdiction|legal system"),
    ("percentage", r"uptime|availability|percent|service level"),
    ("date", r"\bwhen\b.*\b(signed|effect|dated)\b|effective date|what date|which date"),
    ("parties", r"\bwho (are|signed|entered)\b|\bparties\b|which (entities|companies)"),
    ("boolean", r"^(is|does|will|are|has)\b"),
    ("money_or_fees", r"\bcap\b|limited to|maximum aggregate|at most|ceiling"),
    (
        "duration",
        r"how long|how many (days|months|years|hours)|notice|period|duration|within what time|how quickly"
        r"|deadline|\bdue\b|leave|\brun\b|\blast\b|survival",
    ),
    ("money", r"how much|\bfees?\b|price|\brent\b|salary|deposit|\bpaid\b|\bpay\b|cost|amount|insurance|cover|\bsum\b"),
)
_TYPE_RES = [(t, re.compile(p, re.I)) for t, p in _TYPE_RULES]
_ENTITY_FOR_TYPE = {
    "jurisdiction": "jurisdiction",
    "percentage": "percentage",
    "date": "date",
    "money": "money",
    "duration": "duration",
}
_BOOLEAN_TOPICS: tuple[tuple[str, str, str], ...] = (
    (
        r"renew|roll over|evergreen",
        "auto_renewal",
        r"renew|expir|initial term|shall commence|continue for|remains in force|term of this agreement",
    ),
    (r"liab|uncapped|ceiling|exposure", "unlimited_liability", r"liab"),
    (r"compet", "non_compete", r"compet"),
)
SYNONYMS: dict[str, tuple[str, ...]] = {
    "price": ("fee",),
    "cost": ("fee",),
    "bill": ("invoic",),
    "settle": ("pay",),
    "deal": ("agreement",),
    "contract": ("agreement",),
    "roll": ("renew",),
    "evergreen": ("renew",),
    "walk": ("terminat",),
    "end": ("terminat", "expiry"),
    "barred": ("not",),
    "secret": ("confidential",),
    "space": ("premis",),
    "sign": ("made", "dated", "effective"),
    "quickly": ("within",),
    "deadline": ("within",),
    "guarante": ("uptime", "availability"),
    "commit": ("uptime", "availability"),
    "govern": ("laws",),
    "law": ("laws",),
    "legal": ("laws",),
    "system": ("laws",),
    "entity": ("between",),
    "company": ("between",),
    "announce": ("notic",),
    "advance": ("notic",),
    "ceiling": ("exceed", "limit"),
    "exposure": ("liability",),
    "uncapped": ("unlimited",),
    "inventor": ("intellectual",),
    "ownership": ("intellectual", "retain"),
    "disagreement": ("dispute",),
    "heard": ("court", "arbitration"),
    "protection": ("indemnif",),
    "infringement": ("indemnif",),
    "transferred": ("assign",),
    "someone": ("assign",),
    "misuse": ("suspend",),
    "injunction": ("injunctive",),
    "property": ("premis",),
    "let": ("premis",),
    "hire": ("solicit",),
    "scope": ("services",),
    "role": ("position", "duties"),
    "report": ("reporting",),
    "work": ("place", "duties"),
    "last": ("term", "continue", "remain", "survive"),
    "run": ("term", "continue"),
    "year": ("annum", "year"),
    "keep": ("remain", "survive"),
    "kept": ("remain", "survive"),
    "due": ("within", "payable"),
    "initial": ("initial", "commence"),
    "convenience": ("convenience",),
    "ended": ("terminat",),
    "duration": ("term", "continue", "commence"),
    "date": ("made", "dated", "entered", "effective"),
    "effect": ("effective", "made", "entered", "dated"),
    "obligation": ("obligation", "survive", "remain"),
    "protected": ("confidential", "survive", "remain"),
    "survival": ("survive", "remain", "continue"),
    "interpret": ("construed", "governed"),
    "document": ("agreement",),
    "pick": ("governed",),
}
ABSTAIN_VALUE = 0.12
ABSTAIN_CLAUSE = 0.3


@dataclass
class _Candidate:
    clause: Clause
    sentence: str
    overlap: float
    typed: bool
    score: float


@dataclass(frozen=True)
class _QToken:
    token: str
    synonyms: tuple[str, ...]


def infer_answer_type(question: str) -> str:
    """Decide what kind of short answer the question wants."""
    for atype, pattern in _TYPE_RES:
        if pattern.search(question.strip()):
            return atype
    return "clause"


def _matches(q: str, tokens: set[str]) -> bool:
    if len(q) <= 2:
        return q in tokens
    return any(t.startswith(q) or (len(t) > 2 and q.startswith(t)) for t in tokens)


def question_tokens(question: str) -> list[_QToken]:
    """Content tokens of the question with their synonym expansions."""
    out: list[_QToken] = []
    for t in content_tokens(question):
        syns = tuple(s for key, syns in SYNONYMS.items() if t.startswith(key) or key.startswith(t) for s in syns)
        out.append(_QToken(t, syns))
    return out


class _Doc:
    """Per-document statistics shared by all questions."""

    def __init__(self, clauses: list[Clause]) -> None:
        self.clauses = clauses
        self.tokens = [set(content_tokens(c.text)) for c in clauses]
        n = len(clauses)
        self.idf: dict[str, float] = {}
        for toks in self.tokens:
            for t in toks:
                self.idf[t] = self.idf.get(t, 0.0) + 1.0
        for t, df in self.idf.items():
            self.idf[t] = math.log((n + 1) / (df + 0.5)) + 0.1
        self.max_idf = math.log((n + 1) / 0.5) + 0.1

    def weight(self, q: _QToken) -> float:
        """IDF of the question token, or of its best in-document synonym, or the cap when absent."""
        candidates = [self.idf[t] for t in self.idf if _matches(q.token, {t})]
        for s in q.synonyms:
            candidates.extend(self.idf[t] for t in self.idf if _matches(s, {t}))
        return max(candidates) if candidates else self.max_idf

    def overlap(self, qtoks: list[_QToken], tokens: set[str]) -> float:
        total = sum(self.weight(q) for q in qtoks)
        if total == 0:
            return 0.0
        hit = sum(
            self.weight(q) for q in qtoks if _matches(q.token, tokens) or any(_matches(s, tokens) for s in q.synonyms)
        )
        return hit / total


def _entities_in(sentence: str, etype: str) -> list[Entity]:
    return [e for e in extract_entities(sentence) if e.type == etype]


def _keyword_positions(sentence: str, qtoks: list[_QToken], doc: _Doc) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for m in re.finditer(r"[a-z]+", sentence.lower()):
        word = m.group(0)
        for q in qtoks:
            if _matches(q.token, {word}) or any(_matches(s, {word}) for s in q.synonyms):
                out.append((m.start(), doc.weight(q)))
                break
    return out


def _nearest_entity(sentence: str, ents: list[Entity], qtoks: list[_QToken], doc: _Doc) -> Entity:
    """The entity closest to the anchor keyword: the heaviest question word in the sentence, earliest on ties.

    "If availability falls below 99.95%, ... a service credit of 5%" must yield 99.95% for a question about
    availability even though "service" sits right next to 5%; anchoring on the most specific word does that.
    """
    positions = _keyword_positions(sentence, qtoks, doc)
    if not positions:
        return ents[0]
    anchor = min(positions, key=lambda pw: (-pw[1], pw[0]))[0]
    return min(ents, key=lambda e: (abs(e.start - anchor), e.start))


class Answerer:
    """Extractive answerer for one document."""

    def __init__(self, clauses: list[Clause], retriever: ClauseRetriever, settings: Settings | None = None) -> None:
        self.clauses = clauses
        self.retriever = retriever
        self.settings = settings or load_settings()
        self.doc = _Doc(clauses)

    def _typed(self, sentence: str, clause: Clause, answer_type: str) -> bool:
        if answer_type == "money_or_fees":
            has_fee_months = "fees" in sentence.lower() and bool(_entities_in(sentence, "duration"))
            return (
                bool(_entities_in(sentence, "money"))
                or has_fee_months
                or any(f.risk_type == "unlimited_liability" for f in detect_clause(clause))
            )
        etype = _ENTITY_FOR_TYPE.get(answer_type)
        return bool(etype and _entities_in(sentence, etype))

    def _score_clauses(
        self, question: str, answer_type: str, clauses: list[Clause], qtoks: list[_QToken]
    ) -> list[_Candidate]:
        out: list[_Candidate] = []
        value_question = answer_type in _ENTITY_FOR_TYPE or answer_type == "money_or_fees"
        for rank, clause in enumerate(clauses):
            body = body_text(clause)
            clause_overlap = self.doc.overlap(qtoks, set(content_tokens(f"{clause.heading or ''} {body}")))
            for a, b in split_sentences(body):
                sentence = normalize_ws(body[a:b])
                overlap = self.doc.overlap(qtoks, set(content_tokens(sentence)))
                typed = self._typed(sentence, clause, answer_type) if value_question else False
                score = 0.7 * overlap + 0.3 * clause_overlap - 0.02 * rank
                if value_question:
                    score = score if typed else score * 0.25
                out.append(_Candidate(clause, sentence, overlap, typed, score))
        return sorted(out, key=lambda c: -c.score)

    def _candidates(self, question: str, answer_type: str, k: int) -> tuple[list[_Candidate], list[str]]:
        """Score every sentence of every clause, in retrieval order; the rank feeds a small penalty."""
        qtoks = question_tokens(question)
        ranked = [self.clauses[i] for i in self.retriever.rank(question)]
        if answer_type == "date" and self.clauses and self.clauses[0] in ranked:
            ranked = [self.clauses[0], *(c for c in ranked if c is not self.clauses[0])]
        cands = self._score_clauses(question, answer_type, ranked, qtoks)
        return cands, [c.clause_id for c in ranked[:k]]

    def _short_answer(self, cand: _Candidate, answer_type: str, question: str) -> str | None:
        if answer_type == "money_or_fees":
            if any(f.risk_type == "unlimited_liability" for f in detect_clause(cand.clause)):
                return "unlimited"
            if money := _entities_in(cand.sentence, "money"):
                return money[0].value
            if (dur := _entities_in(cand.sentence, "duration")) and "fees" in cand.sentence.lower():
                return f"{dur[0].value} fees"
            return None
        etype = _ENTITY_FOR_TYPE.get(answer_type)
        if etype and (ents := _entities_in(cand.sentence, etype)):
            return _nearest_entity(cand.sentence, ents, question_tokens(question), self.doc).value
        return None

    def _boolean(self, question: str, k: int) -> Answer | None:
        """Yes/no questions about renewal, liability caps and non-competes, answered by the risk rules."""
        topic = next((t for t in _BOOLEAN_TOPICS if re.search(t[0], question, re.I)), None)
        if topic is None:
            return None
        _q, risk, cue = topic
        ranked = self.retriever.search(question, k=len(self.clauses))
        owner = next((c for c in ranked if re.search(cue, body_text(c), re.I)), None)
        if owner is None:
            return Answer(
                question=question,
                answer_type="boolean",
                sentences=[],
                retrieved=[c.clause_id for c in ranked[:k]],
                abstained=True,
            )
        findings = [f for f in detect_clause(owner) if f.risk_type == risk]
        if findings:
            sentence = findings[0].evidence
        else:
            body = body_text(owner)
            sentence = next(
                (normalize_ws(body[a:b]) for a, b in split_sentences(body) if re.search(cue, body[a:b], re.I)),
                normalize_ws(body),
            )
        return self._finish(
            question, "boolean", [(sentence, owner)], "yes" if findings else "no", [c.clause_id for c in ranked[:k]]
        )

    def _parties_answer(self, question: str) -> Answer | None:
        preamble = self.clauses[0] if self.clauses else None
        if preamble is None:
            return None
        parties = sorted({e.value for e in extract_entities(preamble.text) if e.type == "party"})
        if not parties:
            return None
        sentence = next(
            (
                normalize_ws(preamble.text[a:b])
                for a, b in split_sentences(preamble.text)
                if parties[0] in normalize_ws(preamble.text[a:b])
            ),
            None,
        )
        if sentence is None:
            return None
        return self._finish(question, "parties", [(sentence, preamble)], " | ".join(parties), [preamble.clause_id])

    def _finish(
        self, question: str, answer_type: str, picks: list[tuple[str, Clause]], short: str | None, retrieved: list[str]
    ) -> Answer:
        accepted: list[AnswerSentence] = []
        rejected: list[AnswerSentence] = []
        for sentence, clause in picks:
            sup, _ = best_support(sentence, [clause], self.settings.grounding_threshold)
            item = AnswerSentence(
                text=sentence, citations=[clause.clause_id], grounded=sup.supported, support=min(1.0, sup.score)
            )
            (accepted if sup.supported else rejected).append(item)
        return Answer(
            question=question,
            answer_type=answer_type,
            sentences=accepted,
            short_answer=short if accepted else None,
            rejected=rejected,
            retrieved=retrieved,
            method="extractive",
            abstained=not accepted,
        )

    def answer(self, question: str, k: int = 5) -> Answer:
        """Answer ``question`` from the document, or abstain."""
        answer_type = infer_answer_type(question)
        if not self.clauses:
            return Answer(question=question, answer_type=answer_type, sentences=[], abstained=True)
        if answer_type == "parties" and (ans := self._parties_answer(question)):
            return ans
        if answer_type == "boolean":
            if ans := self._boolean(question, k):
                return ans
            answer_type = "clause"
        cands, retrieved = self._candidates(question, answer_type, k)
        value_question = answer_type in _ENTITY_FOR_TYPE or answer_type == "money_or_fees"
        best = cands[0] if cands else None
        weak = (
            best is None
            or (value_question and (not best.typed or best.overlap < ABSTAIN_VALUE))
            or (not value_question and best.overlap < ABSTAIN_CLAUSE)
        )
        if best is None or weak:
            return Answer(question=question, answer_type=answer_type, sentences=[], retrieved=retrieved, abstained=True)
        picks = [(best.sentence, best.clause)]
        if (
            not value_question
            and len(cands) > 1
            and cands[1].clause.clause_id == best.clause.clause_id
            and cands[1].score >= 0.8 * best.score
        ):
            picks.append((cands[1].sentence, cands[1].clause))
        return self._finish(question, answer_type, picks, self._short_answer(best, answer_type, question), retrieved)


def roles_of(text: str) -> tuple[str, str]:
    """Convenience re-export so callers of the QA module do not need the entities module."""
    return extract_roles(text)
