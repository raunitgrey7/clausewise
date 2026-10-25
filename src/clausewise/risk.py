"""Risk playbook detection with sentence-level evidence.

Each rule works one sentence at a time: a sentence fires when it matches a positive pattern and none of the
rule's veto patterns. Sentence scope is the whole trick. "The liability of each party shall not be
unlimited and shall be capped at ..." shares every keyword with a genuinely uncapped clause, and only a
veto evaluated on the same sentence can tell them apart. Every finding carries the matched sentence as
evidence so a reviewer can check the call in seconds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clausewise.entities import _JURISDICTION, normalize_jurisdiction
from clausewise.playbook import DEFAULT_PLAYBOOK, Playbook
from clausewise.schemas import Clause, RiskFinding
from clausewise.segment import body_text
from clausewise.textutil import normalize_ws, sentences


@dataclass(frozen=True)
class Rule:
    """Positive and veto patterns for one risk type."""

    risk_type: str
    positives: tuple[str, ...]
    vetoes: tuple[str, ...]
    description: str

    def compiled(self) -> tuple[list[re.Pattern[str]], list[re.Pattern[str]]]:
        """Compile the positive and veto patterns, case-insensitive."""
        return [re.compile(p, re.I) for p in self.positives], [re.compile(v, re.I) for v in self.vetoes]


RULES: tuple[Rule, ...] = (
    Rule(
        "unlimited_liability",
        (
            r"\bliability\b[^;]{0,80}\bshall be unlimited\b",
            r"\bunlimited in amount\b",
            r"\bwithout (?:any )?limitation as to amount\b",
            r"\bnothing in this agreement shall limit or exclude the liability of\b",
        ),
        (
            r"\bnot be unlimited\b",
            r"\bread as making\b",
            r"\bcapped at\b",
            r"\bis limited to\b",
            r"\bexcept\b[^.]*\bunlimited\b",
        ),
        "a party's liability is uncapped",
    ),
    Rule(
        "auto_renewal",
        (
            r"\bautomatically renew\b",
            r"\brenew automatically\b",
            r"\bextended automatically\b",
            r"\bsuccessive (?:periods|terms) of\b",
            r"\bconsecutive renewal terms\b",
        ),
        (r"\bnot (?:automatically )?renew\b", r"\bno automatic renewal\b", r"\bnot renew automatically\b"),
        "the agreement renews by default unless a party acts",
    ),
    Rule(
        "unilateral_termination",
        (
            r"^(?!either party|each party|both parties|neither)[A-Z][A-Za-z ]{2,30}? "
            r"(?:may terminate this agreement|reserves the right to terminate this agreement)\b[^.]*?"
            r"\b(?:for convenience|for any reason|at any time|at its sole discretion)\b",
            r"\bshall have no right to terminate\b",
            r"\bmay terminate (?:this agreement )?only for\b",
        ),
        (r"\beither party\b", r"\beach party\b", r"\bboth parties\b", r"\bneither\b", r"\bequal terms\b"),
        "only one party may terminate for convenience",
    ),
    Rule(
        "non_compete",
        (r"\b(?:shall not|agrees not to|covenants that it shall not)\b[^.]{0,320}\b(?:compet\w+|competitor)\b",),
        (
            r"\bnothing in this agreement\b",
            r"\bremains? free to compete\b",
            r"\bno non-compete\b",
            r"\bdoes not restrict\b",
            r"\bno covenant not to compete\b",
            r"\bends on termination\b",
        ),
        "a non-compete restriction continues after termination",
    ),
    Rule(
        "ip_assignment",
        (
            r"\b(?:assigns?|assigned|vest in)\b[^.]*"
            r"\b(?:pre-existing|background|whether or not|all present and future|all right, title and interest)\b",
            r"\b(?:pre-existing|background|whether or not)\b[^.]*\b(?:is hereby assigned|assigns? to|shall vest in)\b",
        ),
        (
            r"\bexcluding\b[^.]*\bpre-existing\b",
            r"\bretains? all rights?\b",
            r"\bremains the property of\b",
            r"\bcreated specifically for\b",
            r"\bnothing in this agreement shall operate to assign\b",
            r"\bno assignment of intellectual property\b",
            r"\bretains ownership\b",
        ),
        "pre-existing or unrelated intellectual property is assigned away",
    ),
)
_COMPILED = {rule.risk_type: rule.compiled() for rule in RULES}
_POST_TERM = re.compile(
    r"\b(?:after|following)\b[^.]{0,60}\b(?:termination|end of this agreement|the agreement ends|expiry)\b"
    r"|\b(?:after|following) (?:its|the) termination\b",
    re.I,
)


_MARKER = re.compile(r"^(?:\d{1,2}\.\d{1,2}|\(\w\))\s+")


def _matches(rule: Rule, sentence: str) -> str | None:
    positives, vetoes = _COMPILED[rule.risk_type]
    flat = _MARKER.sub("", normalize_ws(sentence))
    if any(v.search(flat) for v in vetoes):
        return None
    for p in positives:
        m = p.search(flat)
        if m:
            return flat
    return None


def _non_compete_hit(flat: str) -> bool:
    return bool(_POST_TERM.search(flat))


def _unilateral_hits(clause_sentences: list[str]) -> list[str]:
    """Unilateral termination needs a one-sided convenience grant; a restriction on the other party adds weight."""
    rule = RULES[2]
    hits = [h for s in clause_sentences if (h := _matches(rule, s))]
    positives, _ = _COMPILED[rule.risk_type]
    has_grant = any(positives[0].search(h) for h in hits)
    return hits if has_grant else []


def _governing_law_findings(clause: Clause, playbook: Playbook) -> list[RiskFinding]:
    m = _JURISDICTION.search(clause.text)
    if not m:
        return []
    name = normalize_jurisdiction(m.group("j"))
    if name in playbook.acceptable_jurisdictions:
        return []
    pos = playbook.positions["governing_law"]
    sentence = next(
        (normalize_ws(s) for s in sentences(clause.text) if m.group("j").split()[0] in s), normalize_ws(m.group(0))
    )
    return [
        RiskFinding(
            risk_type="governing_law",
            clause_id=clause.clause_id,
            severity=pos.severity,
            weight=pos.weight,  # type: ignore[arg-type]
            evidence=sentence,
            explanation=f"{clause.clause_id} ({clause.heading or clause.label or 'governing law'}): governing law is "
            f"{name}, which is outside the accepted set ({', '.join(sorted(playbook.acceptable_jurisdictions))}).",
            playbook_position=pos.position,
        )
    ]


def detect_clause(clause: Clause, playbook: Playbook = DEFAULT_PLAYBOOK) -> list[RiskFinding]:
    """Run every rule on one clause."""
    findings: list[RiskFinding] = []
    sents = sentences(body_text(clause))
    for rule in RULES:
        if rule.risk_type == "unilateral_termination":
            hits = _unilateral_hits(sents)
        else:
            hits = [h for s in sents if (h := _matches(rule, s))]
            if rule.risk_type == "non_compete":
                hits = [h for h in hits if _non_compete_hit(h)]
        if hits:
            pos = playbook.positions[rule.risk_type]
            label = clause.heading or clause.label or "clause"
            findings.append(
                RiskFinding(
                    risk_type=rule.risk_type,
                    clause_id=clause.clause_id,
                    severity=pos.severity,
                    weight=pos.weight,  # type: ignore[arg-type]
                    evidence=hits[0],
                    explanation=f"{clause.clause_id} ({label}): {rule.description}.",
                    playbook_position=pos.position,
                )
            )
    findings.extend(_governing_law_findings(clause, playbook))
    return findings


def detect_risks(clauses: list[Clause], playbook: Playbook = DEFAULT_PLAYBOOK) -> list[RiskFinding]:
    """Run the playbook over a whole document."""
    out: list[RiskFinding] = []
    for clause in clauses:
        out.extend(detect_clause(clause, playbook))
    return out


def risk_score(findings: list[RiskFinding]) -> int:
    """Sum of finding weights, capped at 100."""
    return min(100, sum(f.weight for f in findings))
