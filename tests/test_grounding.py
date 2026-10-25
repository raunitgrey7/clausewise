from hypothesis import given, settings
from hypothesis import strategies as st

from clausewise.grounding import best_support, lcs_length, normalize, support_score
from clausewise.schemas import Clause
from clausewise.textutil import sentences

CLAUSE = (
    "Termination. Either party may terminate this Agreement for convenience by giving the other party not less "
    "than thirty (30) days prior written notice. Termination shall not affect accrued rights."
)


def test_verbatim_sentence_is_supported():
    sup = support_score(
        "Either party may terminate this Agreement for convenience by giving the other party not less than thirty (30) days prior written notice.",
        CLAUSE,
    )
    assert sup.supported and sup.score == 1.0 and sup.reason == "verbatim"


def test_case_and_whitespace_do_not_matter():
    assert support_score("TERMINATION SHALL NOT\nAFFECT   accrued rights", CLAUSE).supported


def test_number_change_is_rejected():
    sup = support_score(
        "Either party may terminate by giving not less than forty five (45) days prior written notice.", CLAUSE
    )
    assert not sup.supported and "45" in sup.reason


def test_inserted_negation_is_rejected():
    sup = support_score("Termination shall never affect accrued rights.", CLAUSE)
    assert not sup.supported and "negation" in sup.reason


def test_light_paraphrase_with_same_tokens_is_accepted():
    sup = support_score(
        "either party may terminate this agreement for convenience by giving the other party not less than thirty (30) days prior written notice",
        CLAUSE,
    )
    assert sup.supported


def test_fabricated_sentence_is_rejected():
    sup = support_score("The Supplier shall provide all upgrades free of charge for the life of the Agreement.", CLAUSE)
    assert not sup.supported and sup.reason == "insufficient overlap"


def test_sentence_from_other_clause_is_rejected():
    assert not support_score("Rent is payable monthly in advance to the Landlord.", CLAUSE).supported


def test_empty_sentence():
    assert support_score("   ", CLAUSE).reason == "empty"


def test_lcs_length():
    assert lcs_length(["a", "b", "c", "d"], ["a", "x", "c", "d"]) == 3
    assert lcs_length([], ["a"]) == 0


def test_normalize_strips_punctuation():
    assert normalize('The "Agreement"\n is   made.') == "the agreement is made"


def test_best_support_picks_the_supporting_clause():
    c1 = Clause(clause_id="c01", start=0, end=10, heading=None, text="Rent is payable monthly in advance.")
    c2 = Clause(clause_id="c02", start=20, end=30, heading=None, text=CLAUSE)
    sup, clause = best_support("Termination shall not affect accrued rights.", [c1, c2])
    assert clause is c2 and sup.supported
    sup, clause = best_support("anything", [])
    assert clause is None and not sup.supported


@settings(max_examples=60, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_any_corpus_sentence_is_supported_by_its_clause(index: int):
    from clausewise.corpus.generator import generate_contract

    c = generate_contract(42, index % 400)
    for clause in c.clauses[2:6]:
        for s in sentences(clause.text)[:3]:
            if len(s.split()) >= 4:
                assert support_score(s, clause.text).supported
