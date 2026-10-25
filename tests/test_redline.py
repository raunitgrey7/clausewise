import pytest

from clausewise.redline import suggest_redline, suggest_redlines
from clausewise.risk import detect_clause, detect_risks
from clausewise.schemas import RISK_TYPES, Clause

EXAMPLES = {
    "unlimited_liability": "The liability of Customer under this Agreement shall be unlimited.",
    "auto_renewal": "Thereafter this Agreement shall automatically renew for successive periods of one (1) year.",
    "unilateral_termination": "Customer may terminate this Agreement at any time for any reason by giving Supplier 30 days written notice. Supplier may terminate this Agreement only for material breach.",
    "non_compete": "Supplier shall not, for a period of 12 months after its termination, engage in any business that competes with Customer in India.",
    "ip_assignment": "Supplier hereby irrevocably assigns to Customer all present and future intellectual property rights, including Supplier's pre-existing intellectual property.",
    "governing_law": "This Agreement shall be governed by the laws of the State of Delaware.",
}


@pytest.mark.parametrize("risk_type", RISK_TYPES)
def test_redline_resolves_each_risk(risk_type):
    text = EXAMPLES[risk_type]
    cl = Clause(clause_id="c03", start=0, end=len(text), heading=None, text=text)
    finding = next(f for f in detect_clause(cl) if f.risk_type == risk_type)
    red = suggest_redline(cl, finding, ("Supplier", "Customer"))
    proposed = Clause(clause_id="c03", start=0, end=len(red.proposed), heading=None, text=red.proposed)
    assert not any(f.risk_type == risk_type for f in detect_clause(proposed))
    assert red.diff.startswith("--- c03 current") and "+" in red.diff
    assert finding.evidence in red.rationale


def test_roles_and_notice_are_substituted():
    text = EXAMPLES["unilateral_termination"]
    cl = Clause(clause_id="c07", start=0, end=len(text), heading="Termination", text=text)
    red = suggest_redline(cl, detect_clause(cl)[0], ("Supplier", "Customer"))
    assert "30 days" in red.proposed and "{" not in red.proposed


def test_ip_assignor_and_assignee_taken_from_clause():
    text = EXAMPLES["ip_assignment"]
    cl = Clause(clause_id="c09", start=0, end=len(text), heading=None, text=text)
    red = suggest_redline(cl, detect_clause(cl)[0], ("Provider", "Client"))
    assert "Supplier assigns to Customer" in red.proposed


def test_one_redline_per_finding(contract):
    findings = detect_risks(contract.clauses)
    redlines = suggest_redlines(contract.clauses, findings, (contract.parties[0].role, contract.parties[1].role))
    assert [(r.clause_id, r.risk_type) for r in redlines] == [(f.clause_id, f.risk_type) for f in findings]
