import pytest

from clausewise.pipeline import infer_contract_type


def test_analyze_returns_full_result(analyzer, contract):
    analysis = analyzer.analyze(contract.text)
    r = analysis.result
    assert r.contract_type == contract.contract_type
    assert len(r.clauses) >= len(contract.clauses) - 2
    assert all(c.label and 0 < c.confidence <= 1 for c in r.clauses)
    assert len(r.redlines) == len(r.risks)
    assert 0 <= r.risk_score <= 100 and r.summary
    assert analysis.roles == (contract.parties[0].role, contract.parties[1].role)


def test_pipeline_recovers_planted_risks_end_to_end(analyzer, corpus):
    tp = fn = fp = 0
    for c in corpus:
        found = {f.risk_type for f in analyzer.analyze(c.text).result.risks}
        gold = {r.risk_type for r in c.risks}
        tp += len(found & gold)
        fn += len(gold - found)
        fp += len(found - gold)
    assert tp / (tp + fn) >= 0.9
    assert tp / (tp + fp) >= 0.95


def test_contract_type_inference(corpus):
    correct = sum(infer_contract_type(c.text, [cl.label or "" for cl in c.clauses]) == c.contract_type for c in corpus)
    assert correct / len(corpus) >= 0.95
    assert infer_contract_type("random text", ["rent"]) == "lease"
    assert infer_contract_type("random text", ["definitions"]) == "nda"


def test_text_too_long_is_rejected(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze("x" * (analyzer.settings.max_text_chars + 1))


def test_ask_over_analysis(analyzer, contract):
    analysis = analyzer.analyze(contract.text)
    jurisdiction = next(e.value for e in contract.entities if e.type == "jurisdiction")
    ans = analyzer.ask(analysis.clauses, "Which law governs this agreement?")
    assert not ans.abstained and ans.method == "extractive" and ans.short_answer == jurisdiction


def test_llm_flag_falls_back_when_disabled(analyzer, contract):
    analysis = analyzer.analyze(contract.text)
    ans = analyzer.ask(analysis.clauses, "Which law governs this agreement?", llm=True)
    assert ans.method == "extractive"


def test_binary_garbage_does_not_crash(analyzer):
    text = "\x00\x01\x02 " * 10 + "�‮" * 50 + "\n" * 20 + "9" * 300
    r = analyzer.analyze(text).result
    assert r.risks == [] and r.risk_score == 0
