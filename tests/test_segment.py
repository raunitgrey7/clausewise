import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clausewise.segment import body_text, classify_line, segment, split_inline_heading


@pytest.mark.parametrize(
    ("line", "kind", "heading"),
    [
        ("3. Term", "heading", "Term"),
        ("3. Term. This Agreement shall commence on the Effective Date.", "heading", "Term"),
        ("Section 4. Ending this Agreement", "heading", "Ending this Agreement"),
        ("Clause 7: Return or Destruction", "heading", "Return or Destruction"),
        ("ARTICLE IV. LIMITATION OF LIABILITY", "heading", "LIMITATION OF LIABILITY"),
        ("GOVERNING LAW", "heading", "GOVERNING LAW"),
        ("7.1 Either party may terminate.", "text", None),
        ("(a) is or becomes public;", "text", None),
        ("30 days written notice shall be given.", "text", None),
        ("Page 2 of 5", "page", None),
        ("IN WITNESS WHEREOF the parties have signed.", "signature", None),
        ("WHEREAS the Supplier provides services;", "recital", None),
        ("BETWEEN:", "text", None),
        ("", "blank", None),
    ],
)
def test_classify_line(line, kind, heading):
    assert classify_line(line) == (kind, heading)


def test_split_inline_heading():
    assert split_inline_heading("Term. This Agreement shall commence.") == ("Term", "This Agreement shall commence.")
    assert split_inline_heading("Confidential Information") == ("Confidential Information", None)
    assert split_inline_heading("the parties agree that nothing here applies.")[0] == ""


def test_segment_recovers_gold_boundaries(corpus):
    hits = total = 0
    for c in corpus:
        pred = {c.text.count("\n", 0, cl.start) for cl in segment(c.text)}
        gold = {c.text.count("\n", 0, cl.start) for cl in c.clauses}
        hits += len(pred & gold)
        total += len(gold)
    assert hits / total >= 0.98


def test_segment_ids_are_sequential_and_headings_found(contract):
    pred = segment(contract.text)
    assert [c.clause_id for c in pred] == [f"c{i:02d}" for i in range(1, len(pred) + 1)]
    gold_headings = {cl.heading.upper() for cl in contract.clauses if cl.heading}
    assert gold_headings <= {cl.heading.upper() for cl in pred if cl.heading}


def test_page_markers_are_not_part_of_any_clause():
    text = "TITLE\n\nThis Agreement is between A and B.\n\n1. Term\n\nIt lasts a year.\n\nPage 1 of 2\n\n2. Rent\n\nRent is due monthly."
    pred = segment(text)
    assert [c.heading for c in pred] == [None, "Term", "Rent"]
    assert all("Page 1 of 2" not in c.text for c in pred)


def test_cross_reference_at_line_start_is_not_a_boundary():
    text = "TITLE\n\nPreamble here.\n\n1. Liability\n\nLiability is capped as set out in\nSection 9 (Notices) of this Agreement and nothing more."
    assert len(segment(text)) == 2


def test_unheaded_paragraph_after_body_starts_clause():
    text = "TITLE\n\nPreamble.\n\n1. Term\n\nThe term is one year.\n\nEither party may terminate on notice.\n\n3. Law\n\nIndian law."
    assert [c.heading for c in segment(text)] == [None, "Term", None, "Law"]


def test_body_text_strips_heading_line_and_inline_heading():
    pred = segment("TITLE\n\nPre.\n\n2. Term. This Agreement lasts two years.\n\n3. Rent\n\nRent is due.")
    assert body_text(pred[1]) == "This Agreement lasts two years."
    assert body_text(pred[2]) == "Rent is due."


def test_empty_and_blank_input():
    assert segment("") == []
    assert segment("   \n\n ") == []


@settings(max_examples=150, deadline=None)
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=600))
def test_segment_never_crashes_and_spans_are_consistent(text: str):
    clauses = segment(text)
    for a, b in itertools.pairwise(clauses):
        assert a.end <= b.start
    for c in clauses:
        assert 0 <= c.start < c.end <= len(text)
        assert text[c.start : c.end] == c.text
