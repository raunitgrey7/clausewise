import random
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clausewise.corpus.values import duration, format_date, money
from clausewise.entities import extract_entities, extract_roles, normalize_jurisdiction


def values(text, etype):
    return [e.value for e in extract_entities(text) if e.type == etype]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("not less than thirty (30) days prior written notice", ["30 days"]),
        ("within 45 days of receipt", ["45 days"]),
        ("for a period of twenty four months after termination", ["24 months"]),
        ("thirty (30)\ndays after the wrap", ["30 days"]),
        ("one (1) year thereafter", ["1 year"]),
        ("within 72 hours of becoming aware", ["72 hours"]),
        ("in the twelve (12) months immediately preceding", ["12 months"]),
    ],
)
def test_durations(text, expected):
    assert values(text, "duration") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a fee of INR 12,50,000 per annum", ["INR 1250000"]),
        ("rent of Rs. 45,000 per month", ["INR 45000"]),
        ("US$ 25,000 payable", ["USD 25000"]),
        ("$98,000 per year", ["USD 98000"]),
        ("GBP 10,000 and EUR 15,000 and SGD 30,000", ["GBP 10000", "EUR 15000", "SGD 30000"]),
        ("GBP\n323,000 per year", ["GBP 323000"]),
    ],
)
def test_money(text, expected):
    assert values(text, "money") == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("made on 12 March 2025 between", ["2025-03-12"]),
        ("entered into as of March 12, 2025 (the", ["2025-03-12"]),
        ("dated 12th day of March, 2025 and", ["2025-03-12"]),
        ("dated 2025-03-12.", ["2025-03-12"]),
        ("on 31 February 2025 (invalid)", []),
        ("the Digital Personal Data Protection Act, 2023 applies", []),
    ],
)
def test_dates(text, expected):
    assert values(text, "date") == expected


def test_percentages():
    assert values("uptime of at least 99.95%, a credit of 10 per cent", "percentage") == ["99.95%", "10%"]
    assert values("clause 7.1 applies", "percentage") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("shall be governed by the laws of India. Each party", ["India"]),
        ("construed in accordance with the laws of the State of New York, without regard", ["New York"]),
        ("governed by and construed in\naccordance with the laws of England and Wales.", ["England and Wales"]),
        ("governed by the laws of the Dubai International Financial Centre.", ["DIFC"]),
        ("a company incorporated under the laws of India with its registered office", []),
        ("governed by the laws of Ruritania.", ["Ruritania"]),
    ],
)
def test_jurisdictions(text, expected):
    assert values(text, "jurisdiction") == expected


def test_normalize_jurisdiction_generic():
    assert normalize_jurisdiction("the State of Texas") == "Texas"
    assert normalize_jurisdiction("the Republic of Ireland") == "Ireland"


def test_parties_from_preamble_and_signature():
    text = (
        "THIS AGREEMENT is made on 1 May 2024 between Zenith Labs Private Limited, a company incorporated under the laws "
        'of India ("Supplier"), and Meera Iyer, an individual residing at 4 MG Road, Pune ("Customer").\n\n'
        "Signed for and on behalf of Zenith Labs Private Limited\nName: X"
    )
    assert values(text, "party") == ["Zenith Labs Private Limited", "Meera Iyer", "Zenith Labs Private Limited"]


def test_parties_tolerate_typo_in_description():
    text = 'between Sterling Networks Private Limited, a cmopany registered in India ("Provider"), and Beta LLC, a company incorporated under the laws of Delaware ("Customer").'
    assert values(text, "party") == ["Sterling Networks Private Limited", "Beta LLC"]


def test_extract_roles(contract):
    assert extract_roles(contract.text) == (contract.parties[0].role, contract.parties[1].role)
    assert extract_roles("no quotes here at all") == ("Party A", "Party B")


def test_entities_attributed_to_clauses(contract):
    ents = extract_entities(contract.text, contract.clauses)
    assert ents and all(e.clause_id for e in ents)


def test_corpus_level_entity_f1(corpus):
    tp = fp = fn = 0
    for c in corpus:
        pred = extract_entities(c.text)
        gold = list(c.entities)
        matched = set()
        for p in pred:
            hit = next(
                (
                    i
                    for i, g in enumerate(gold)
                    if i not in matched
                    and g.type == p.type
                    and g.value == p.value
                    and g.start < p.end
                    and p.start < g.end
                ),
                None,
            )
            if hit is None:
                fp += 1
            else:
                matched.add(hit)
                tp += 1
        fn += len(gold) - len(matched)
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert f1 >= 0.97


@settings(max_examples=60, deadline=None)
@given(
    st.sampled_from(["INR", "USD", "GBP", "EUR", "SGD"]),
    st.integers(min_value=1, max_value=99_999_999),
    st.sampled_from(["indian", "western", "code"]),
    st.integers(min_value=0, max_value=1000),
)
def test_money_round_trip(currency, amount, style, seed):
    v = money(random.Random(seed), currency, amount, style)
    assert values(f"a fee of {v.text} per year", "money") == [v.normalized]


@settings(max_examples=60, deadline=None)
@given(
    st.integers(min_value=1, max_value=999),
    st.sampled_from(["days", "months", "years", "hours"]),
    st.sampled_from(["words_parens", "digits", "words"]),
)
def test_duration_round_trip(n, unit, numstyle):
    v = duration(n, unit, numstyle)
    assert values(f"within {v.text} of notice", "duration") == [v.normalized]


@settings(max_examples=40, deadline=None)
@given(
    st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31)), st.sampled_from(["dmy", "mdy", "legal", "iso"])
)
def test_date_round_trip(d, fmt):
    assert values(f"made on {format_date(d, fmt)} between", "date") == [d.isoformat()]
