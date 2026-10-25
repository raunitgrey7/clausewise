import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clausewise.config import Settings
from clausewise.embeddings import embedder_name, load_embedder
from clausewise.retrieval import BM25Index, ClauseRetriever, TfidfIndex, index_text, reciprocal_rank_fusion
from clausewise.schemas import Clause

DOCS = [
    "Rent is payable monthly in advance to the Landlord.",
    "Either party may terminate this Agreement on thirty days written notice.",
    "This Agreement shall be governed by the laws of India.",
]


def test_bm25_prefers_matching_document():
    idx = BM25Index(DOCS)
    scores = idx.scores("which law governs the agreement")
    assert scores.argmax() == 2
    assert idx.scores("zebra").sum() == 0


def test_tfidf_prefers_matching_document():
    idx = TfidfIndex(DOCS)
    assert idx.scores("notice to terminate").argmax() == 1


def test_rrf_fuses_and_breaks_ties_by_first_ranking():
    assert reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]]) == [0, 2, 1]
    assert reciprocal_rank_fusion([[1, 0], [1, 0]]) == [1, 0]
    assert reciprocal_rank_fusion([]) == []


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.text(alphabet="abcdefghij ", min_size=1, max_size=40), min_size=1, max_size=6),
    st.integers(min_value=0, max_value=5),
)
def test_bm25_unique_token_ranks_its_document_first(docs, pick):
    docs = [d + f" uniqueterm{i}" for i, d in enumerate(docs)]
    target = pick % len(docs)
    scores = BM25Index(docs).scores(f"uniqueterm{target}")
    assert scores.argmax() == target


def make_clauses(texts, headings=None):
    out, pos = [], 0
    for i, t in enumerate(texts):
        h = headings[i] if headings else None
        out.append(Clause(clause_id=f"c{i + 1:02d}", start=pos, end=pos + len(t), heading=h, text=t))
        pos += len(t) + 2
    return out


@pytest.mark.parametrize("mode", ["bm25", "tfidf", "hybrid", "dense"])
def test_retriever_modes_without_embedder(mode):
    r = ClauseRetriever(make_clauses(DOCS), None, mode)
    assert r.search("governing law", k=1)[0].clause_id == "c03"
    assert len(r.rank("rent")) == 3
    expected = {"bm25": "bm25", "tfidf": "tfidf", "hybrid": "bm25+tfidf", "dense": "tfidf (dense unavailable)"}[mode]
    assert r.backend == expected


def test_retriever_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ClauseRetriever(make_clauses(DOCS), None, "magic")


def test_retriever_handles_empty_document():
    r = ClauseRetriever([], None, "hybrid")
    assert r.rank("anything") == [] and r.search("anything") == []


def test_index_text_repeats_heading():
    cl = make_clauses(["Rent. Rent is due monthly."], ["Rent"])[0]
    assert index_text(cl).startswith("Rent. Rent. ")


def test_tfidf_mode_setting_disables_dense(settings):
    s = Settings(embeddings="tfidf", data_dir=settings.data_dir)
    assert load_embedder(s) is None
    assert embedder_name(None) == "tfidf-fallback"


def test_dense_mode_requires_cached_model(settings):
    s = Settings(embeddings="dense", embed_model="minishlab/this-model-does-not-exist", data_dir=settings.data_dir)
    assert load_embedder(s) is None


@pytest.mark.skipif(
    os.environ.get("CLAUSEWISE_TEST_DENSE") != "1", reason="set CLAUSEWISE_TEST_DENSE=1 to test the cached dense model"
)
def test_dense_backend_when_cached(settings):
    s = Settings(embeddings="dense", data_dir=settings.data_dir)
    emb = load_embedder(s)
    assert emb is not None
    r = ClauseRetriever(make_clauses(DOCS), emb, "hybrid")
    assert r.backend == "bm25+dense"
    assert r.search("which jurisdiction's law applies", k=1)[0].clause_id == "c03"
