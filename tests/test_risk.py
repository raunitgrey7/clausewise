import json

import pytest

from clausewise.corpus.generator import generate_corpus
from clausewise.playbook import DEFAULT_PLAYBOOK, Playbook
from clausewise.risk import detect_clause, detect_risks, risk_score
from clausewise.schemas import Clause


def clause(text, heading=None, cid="c05"):
    return Clause(clause_id=cid, start=0, end=len(text), heading=heading, text=text)


def types(text, heading=None, playbook=DEFAULT_PLAYBOOK):
    return sorted({f.risk_type for f in detect_clause(clause(text, heading), playbook)})


@pytest.mark.parametrize(
    "text",
    [
        "The liability of Customer under this Agreement shall be unlimited.",
        "Supplier shall be liable for all losses suffered by Customer arising out of this Agreement, without any limitation as to amount.",
        "Nothing in this Agreement shall limit or exclude the liability of Customer to Supplier, which shall be unlimited in amount.",
    ],
)
def test_unlimited_liability_positive(text):
    assert types(text) == ["unlimited_liability"]


@pytest.mark.parametrize(
    "text",
    [
        "The liability of each party shall not be unlimited and shall in all cases be capped at INR 5,00,000.",
        "Nothing in this clause shall be read as making either party's liability unlimited; the aggregate liability of each party is limited to INR 5,00,000.",
        "Neither party's liability shall be unlimited: each party's aggregate liability is limited to USD 50,000, except in the case of fraud.",
        "In no event shall either party's aggregate liability under this Agreement exceed USD 50,000.",
    ],
)
def test_unlimited_liability_hard_negatives(text):
    assert "unlimited_liability" not in types(text)


@pytest.mark.parametrize(
    "text",
    [
        "Thereafter this Agreement shall automatically renew for successive periods of one (1) year unless either party gives notice.",
        "Upon expiry of the initial term, this Agreement shall renew automatically for further terms of 12 months each.",
        "This Agreement shall be extended automatically for consecutive renewal terms of one (1) year.",
    ],
)
def test_auto_renewal_positive(text):
    assert types(text) == ["auto_renewal"]


@pytest.mark.parametrize(
    "text",
    [
        "This Agreement shall not renew automatically; any renewal shall require a new written agreement.",
        "Renewal must be agreed in writing at least 60 days before expiry; there is no automatic renewal.",
        "The parties may renew this Agreement for a further 12 months by mutual written agreement, failing which it shall expire.",
    ],
)
def test_auto_renewal_hard_negatives(text):
    assert types(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Customer may terminate this Agreement at any time for any reason by giving Supplier 30 days written notice. Supplier may terminate this Agreement only for material breach by Customer.",
        "Landlord may terminate this Agreement for convenience on 60 days written notice to Tenant. Tenant shall have no right to terminate this Agreement for convenience.",
        "7.1 Licensor reserves the right to terminate this Agreement, in whole or in part, at its sole discretion upon 45 days notice to Subscriber. Subscriber may terminate only for an uncured material breach.",
    ],
)
def test_unilateral_termination_positive(text):
    assert types(text) == ["unilateral_termination"]


@pytest.mark.parametrize(
    "text",
    [
        "Either party may terminate this Agreement for convenience by giving the other party not less than 30 days prior written notice.",
        "Either party may terminate for convenience on 30 days written notice; no party shall have any greater or lesser right to terminate than the other.",
        "Termination for convenience is available to both parties on equal terms upon 30 days written notice, and neither Supplier nor Customer may terminate at its sole discretion without such notice.",
        "This Agreement may be terminated by either party without cause on 30 days written notice to the other party.",
        "Neither party may terminate this Agreement before the end of the term except for material breach.",
    ],
)
def test_unilateral_termination_negatives(text):
    assert "unilateral_termination" not in types(text)


@pytest.mark.parametrize(
    "text",
    [
        "Employee shall not, during the term of this Agreement and for a period of twelve (12) months after its termination, directly or indirectly engage in any business that competes with the business of Company within India.",
        "For 6 months following termination of this Agreement for any reason, Supplier shall not carry on any business which competes with Customer in India.",
    ],
)
def test_non_compete_positive(text):
    assert types(text) == ["non_compete"]


@pytest.mark.parametrize(
    "text",
    [
        "During the term of this Agreement, Supplier shall not provide services to any direct competitor of Customer. This restriction ends on termination of this Agreement and does not restrict Supplier's activities after termination.",
        "Nothing in this Agreement shall prevent Employee from engaging in any business after termination, including with competitors of Company; the parties agree that no non-compete restriction applies.",
        "The parties acknowledge that this Agreement contains no covenant not to compete, and Employee remains free to compete with Company both during and after the term.",
    ],
)
def test_non_compete_negatives(text):
    assert types(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "Employee hereby irrevocably assigns to Company, with full title guarantee, all present and future intellectual property rights in any work created by Employee, whether or not created in the course of this Agreement, including Employee's pre-existing intellectual property.",
        "All intellectual property conceived by Supplier during the term, whether or not related to the services and including background intellectual property, shall vest in and is hereby assigned to Customer absolutely.",
    ],
)
def test_ip_assignment_positive(text):
    assert types(text) == ["ip_assignment"]


@pytest.mark.parametrize(
    "text",
    [
        "Upon payment in full, Supplier assigns to Customer all intellectual property rights in the deliverables created specifically for Customer, excluding Supplier's pre-existing materials, in which Supplier retains all rights.",
        "Nothing in this Agreement shall operate to assign any pre-existing intellectual property of either party; each party retains ownership of its background intellectual property.",
        "Each party retains all rights in its pre-existing intellectual property. Supplier grants Customer a non-exclusive licence to use the deliverables.",
    ],
)
def test_ip_assignment_negatives(text):
    assert types(text) == []


def test_governing_law_uses_playbook():
    text = "This Agreement shall be governed by the laws of the State of New York, without regard to its conflict of laws principles."
    assert types(text) == ["governing_law"]
    assert types("This Agreement shall be governed by the laws of India.") == []
    lenient = Playbook(acceptable_jurisdictions=frozenset({"New York"}))
    assert types(text, playbook=lenient) == []


def test_heading_line_does_not_mask_rule():
    text = "ARTICLE V. TERMINATION\nClient may terminate this Agreement at any time for any reason by giving Provider 90 days written notice. Provider may terminate this Agreement only for material breach."
    assert types(text, heading="TERMINATION") == ["unilateral_termination"]


def test_findings_carry_evidence_and_positions():
    f = detect_clause(clause("The liability of Customer under this Agreement shall be unlimited.", "Liability"))[0]
    assert f.evidence.startswith("The liability of Customer")
    assert f.severity == "high" and f.weight == 30 and "c05" in f.explanation
    assert f.playbook_position == DEFAULT_PLAYBOOK.positions["unlimited_liability"].position


def test_risk_score_is_capped():
    f = detect_clause(clause("The liability of Customer under this Agreement shall be unlimited."))
    assert risk_score(f) == 30
    assert risk_score(f * 5) == 100
    assert risk_score([]) == 0


def test_playbook_from_json(tmp_path):
    path = tmp_path / "pb.json"
    path.write_text(
        json.dumps(
            {
                "acceptable_jurisdictions": ["Germany"],
                "positions": {"auto_renewal": {"severity": "high", "weight": 40, "position": "x", "fallback": "y"}},
            }
        )
    )
    pb = Playbook.from_json(path)
    assert pb.acceptable_jurisdictions == frozenset({"Germany"})
    assert pb.positions["auto_renewal"].weight == 40
    assert pb.positions["unlimited_liability"] == DEFAULT_PLAYBOOK.positions["unlimited_liability"]


def test_corpus_precision_and_recall(corpus):
    tp = fp = fn = 0
    for c in corpus:
        found = {(f.risk_type, f.clause_id) for f in detect_risks(c.clauses)}
        gold = {(r.risk_type, r.clause_id) for r in c.risks}
        tp += len(found & gold)
        fp += len(found - gold)
        fn += len(gold - found)
    assert tp / (tp + fp) >= 0.97
    assert tp / (tp + fn) >= 0.95


def test_detection_is_deterministic():
    cs = generate_corpus(42, 10)
    first = [[f.model_dump() for f in detect_risks(c.clauses)] for c in cs]
    second = [[f.model_dump() for f in detect_risks(c.clauses)] for c in cs]
    assert first == second
