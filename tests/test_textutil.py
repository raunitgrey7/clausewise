from hypothesis import given
from hypothesis import strategies as st

from clausewise.textutil import (
    content_tokens,
    int_to_words,
    normalize_ws,
    sentences,
    split_sentences,
    stem,
    tokenize,
    words_to_int,
)


def test_tokenize_lowercases_and_keeps_decimals():
    assert tokenize("Uptime of 99.9% per Month") == ["uptime", "of", "99.9", "per", "month"]


def test_stem_maps_inflections_to_one_root():
    roots = {stem(w) for w in ("terminate", "terminated", "terminating", "terminates")}
    assert roots == {"terminat"}
    assert stem("days") == "days"  # short tokens are untouched


def test_content_tokens_drop_stopwords_and_question_words():
    toks = content_tokens("What is the governing law of the agreement?")
    assert "what" not in toks and "agreement" not in toks
    assert "governing" in toks or "govern" in toks


def test_split_sentences_respects_abbreviations_and_subnumbers():
    text = "Acme Pvt. Ltd. shall pay. It is due within thirty (30) days. See clause 7.1 for details."
    assert sentences(text) == [
        "Acme Pvt. Ltd. shall pay.",
        "It is due within thirty (30) days.",
        "See clause 7.1 for details.",
    ]


def test_split_sentences_blank_line_is_boundary():
    assert sentences("First paragraph\n\nSECOND PARA") == ["First paragraph", "SECOND PARA"]


def test_split_sentences_spans_index_into_text():
    text = "  One. Two?  "
    assert [text[a:b] for a, b in split_sentences(text)] == ["One.", "Two?"]


def test_normalize_ws_collapses_newlines():
    assert normalize_ws("thirty (30)\n  days") == "thirty (30) days"


@given(st.integers(min_value=0, max_value=999))
def test_number_words_round_trip(n: int):
    assert words_to_int(int_to_words(n).split()) == n


def test_words_to_int_rejects_non_numbers():
    assert words_to_int(["banana"]) is None
    assert words_to_int([]) is None
