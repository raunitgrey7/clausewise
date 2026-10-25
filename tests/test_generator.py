import gzip
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from clausewise.config import DEFAULT_CORPUS_SIZE
from clausewise.corpus.generator import corpus_path, generate_contract, generate_corpus, load_or_generate
from clausewise.corpus.plans import PLANS
from clausewise.schemas import RISK_TYPES


def test_default_corpus_size_meets_spec():
    assert DEFAULT_CORPUS_SIZE >= 300


def test_same_seed_same_corpus(corpus):
    again = generate_corpus(42, len(corpus))
    assert [c.model_dump_json() for c in again] == [c.model_dump_json() for c in corpus]


def test_different_seed_different_text(corpus):
    other = generate_contract(43, 0)
    assert other.text != corpus[0].text


def test_all_contract_types_present(corpus):
    assert {c.contract_type for c in corpus} == set(PLANS)


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=0, max_value=5000), st.integers(min_value=0, max_value=10_000))
def test_any_contract_satisfies_invariants(index: int, seed: int):
    c = generate_contract(seed, index)
    assert c.clauses[0].label == "preamble" and c.clauses[-1].label == "signature"
    for a, b in zip(c.clauses, c.clauses[1:], strict=False):
        assert a.end <= b.start
    for cl in c.clauses:
        assert c.text[cl.start : cl.end] == cl.text
    ids = {cl.clause_id for cl in c.clauses}
    assert all(r.clause_id in ids for r in c.risks)
    assert all(q.gold_clause_id in ids for q in c.questions if q.answerable)


def test_entity_spans_contain_their_values(corpus):
    for c in corpus:
        for e in c.entities:
            span = " ".join(c.text[e.start : e.end].split())
            if e.type == "party":
                assert span == e.value
            elif e.type == "money":
                assert any(ch.isdigit() for ch in span)
            elif e.type == "duration":
                assert span.split()[-1].rstrip(".").lower().startswith(e.value.split()[-1][:3])


def test_risk_variants_map_to_planted_risks(corpus):
    for c in corpus:
        variants = json.loads(str(c.style["variants"]))
        planted = {r.risk_type for r in c.risks}
        assert (variants.get("term") == "auto_renew") == ("auto_renewal" in planted)
        assert (variants.get("liability") == "unlimited") == ("unlimited_liability" in planted)
        assert (variants.get("ip") == "broad") == ("ip_assignment" in planted)
        assert (variants.get("termination") == "unilateral") == ("unilateral_termination" in planted)
        assert (variants.get("non_compete") == "post_term") == ("non_compete" in planted)


def test_every_risk_type_is_planted_somewhere(corpus):
    planted = {r.risk_type for c in corpus for r in c.risks}
    assert planted == set(RISK_TYPES)


def test_questions_have_one_unanswerable_each(corpus):
    for c in corpus:
        assert sum(1 for q in c.questions if not q.answerable) == 1
        assert sum(1 for q in c.questions if q.answerable) >= 3


def test_noise_only_in_noisy_contracts(corpus):
    for c in corpus:
        dropped = [
            cl for cl in c.clauses if cl.heading is None and cl.label not in {"preamble", "recitals", "signature"}
        ]
        if float(c.style["heading_dropout"]) == 0:
            assert not dropped
    assert any(float(c.style["heading_dropout"]) > 0 for c in corpus)
    assert any(float(c.style["heading_dropout"]) == 0 for c in corpus)


def test_load_or_generate_caches_and_reloads(tmp_path):
    first = load_or_generate(tmp_path, 7, 5)
    path = corpus_path(tmp_path, 7, 5)
    assert path.exists()
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        assert sum(1 for _ in fh) == 5
    second = load_or_generate(tmp_path, 7, 5)
    assert [c.text for c in first] == [c.text for c in second]


@pytest.mark.parametrize("ctype", sorted(PLANS))
def test_mandatory_labels_always_present(corpus, ctype):
    mandatory = {label for label, p in PLANS[ctype] if p >= 1.0}
    for c in corpus:
        if c.contract_type == ctype:
            assert mandatory <= {cl.label for cl in c.clauses}
