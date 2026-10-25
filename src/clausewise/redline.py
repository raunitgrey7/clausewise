"""Redline suggestions: replace a risky clause with the playbook fallback and show the diff.

The proposed text is the playbook's fallback language with the contract's own role names substituted,
so the suggestion reads as if it belonged to the document. The benchmark checks the loop closes: running
the detector on the proposed text must produce no finding of the same type.
"""

from __future__ import annotations

import difflib
import re

from clausewise.playbook import DEFAULT_PLAYBOOK, Playbook
from clausewise.schemas import Clause, Redline, RiskFinding
from clausewise.textutil import sentences

_ASSIGN_RE = re.compile(
    r"\b([A-Z][A-Za-z ]{2,30}?) (?:hereby )?(?:irrevocably )?assigns? to ([A-Z][A-Za-z ]{2,30}?)[ ,]"
)
_VEST_RE = re.compile(
    r"\b(?:created|developed|conceived)[^.]*?\bby ([A-Z][A-Za-z ]{2,30}?)[ ,][^.]*?"
    r"assigned to ([A-Z][A-Za-z ]{2,30}?)\b"
)
_NOTICE_RE = re.compile(r"\b((?:[a-z]+ )?\(?\d+\)? (?:days?|months?))\b", re.I)
_GRANT_RE = re.compile(r"\b([A-Z][A-Za-z ]{2,30}?) (?:may terminate|reserves the right to terminate)")


def _roles_for(clause: Clause, roles: tuple[str, str]) -> dict[str, str]:
    """Work out which role is which for the placeholders in the fallback language."""
    a, b = roles
    assignor, assignee = b, a
    m = _ASSIGN_RE.search(clause.text) or _VEST_RE.search(clause.text)
    if m:
        assignor, assignee = m.group(1).strip(), m.group(2).strip()
    notice = "thirty (30) days"
    if (n := _NOTICE_RE.search(clause.text)) and "notice" in clause.text.lower():
        notice = n.group(1)
    return {"A": a, "B": b, "ASSIGNOR": assignor, "ASSIGNEE": assignee, "notice": notice}


def _fill(template: str, values: dict[str, str]) -> str:
    return re.sub(r"\{([A-Za-z_]+)\}", lambda m: values.get(m.group(1), m.group(0)), template)


def _diff(clause_id: str, original: str, proposed: str) -> str:
    lines = difflib.unified_diff(
        sentences(original),
        sentences(proposed),
        fromfile=f"{clause_id} current",
        tofile=f"{clause_id} proposed",
        lineterm="",
        n=0,
    )
    return "\n".join(lines)


def suggest_redline(
    clause: Clause, finding: RiskFinding, roles: tuple[str, str], playbook: Playbook = DEFAULT_PLAYBOOK
) -> Redline:
    """Build the redline for one finding."""
    pos = playbook.positions[finding.risk_type]
    proposed = _fill(pos.fallback, _roles_for(clause, roles))
    rationale = (
        f"{finding.explanation} Replace with the playbook position: {pos.position} "
        f'The evidence sentence was: "{finding.evidence}"'
    )
    return Redline(
        clause_id=clause.clause_id,
        risk_type=finding.risk_type,
        original=clause.text,
        proposed=proposed,
        diff=_diff(clause.clause_id, clause.text, proposed),
        rationale=rationale,
    )


def suggest_redlines(clauses: list[Clause], findings: list[RiskFinding], roles: tuple[str, str],
                     playbook: Playbook = DEFAULT_PLAYBOOK) -> list[Redline]:  # fmt: skip
    """One redline per finding, in finding order."""
    by_id = {c.clause_id: c for c in clauses}
    return [suggest_redline(by_id[f.clause_id], f, roles, playbook) for f in findings if f.clause_id in by_id]
