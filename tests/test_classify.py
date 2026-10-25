from sklearn.metrics import f1_score

from clausewise.classify import ClauseClassifier, dense_features, rule_label, split_contracts
from clausewise.schemas import Clause


def test_split_is_deterministic_and_disjoint(corpus):
    train, test = split_contracts(corpus, 42)
    train2, test2 = split_contracts(corpus, 42)
    assert [c.contract_id for c in train] == [c.contract_id for c in train2]
    assert [c.contract_id for c in test] == [c.contract_id for c in test2]
    assert not {c.contract_id for c in train} & {c.contract_id for c in test}
    assert len(test) == round(0.3 * len(corpus))


def test_split_changes_with_seed(corpus):
    _, a = split_contracts(corpus, 1)
    _, b = split_contracts(corpus, 2)
    assert [c.contract_id for c in a] != [c.contract_id for c in b]


def test_classifier_learns_labels(corpus):
    train, test = split_contracts(corpus, 42)
    clf = ClauseClassifier().fit([c.clauses for c in train])
    y_true = [cl.label for c in test for cl in c.clauses]
    y_pred = clf.predict_docs([c.clauses for c in test])
    assert f1_score(y_true, y_pred, average="macro") >= 0.9


def test_body_only_classifier_works_without_headings(corpus):
    train, test = split_contracts(corpus, 42)
    strip = lambda docs: [[cl.model_copy(update={"heading": None}) for cl in doc] for doc in docs]  # noqa: E731
    clf = ClauseClassifier(use_heading=False).fit(strip([c.clauses for c in train]))
    y_true = [cl.label for c in test for cl in c.clauses]
    y_pred = clf.predict_docs(strip([c.clauses for c in test]))
    assert f1_score(y_true, y_pred, average="macro") >= 0.85


def test_predict_returns_confidences(corpus, analyzer):
    preds = analyzer.classifier.predict(corpus[0].clauses)
    assert len(preds) == len(corpus[0].clauses)
    assert all(0 < conf <= 1 for _label, conf in preds)
    assert analyzer.classifier.predict([]) == []


def test_rule_baseline_recognises_common_headings():
    mk = lambda h, t: Clause(clause_id="c01", start=0, end=len(t), heading=h, text=t)  # noqa: E731
    assert rule_label(mk("Limitation of Liability", "x"), 3, 10) == "liability"
    assert rule_label(mk("Governing Law", "x"), 3, 10) == "governing_law"
    assert rule_label(mk(None, "TITLE\n\nThis Agreement is made"), 0, 10) == "preamble"
    assert rule_label(mk(None, "WHEREAS the parties"), 1, 10) == "recitals"
    assert rule_label(mk(None, "IN WITNESS WHEREOF"), 9, 10) == "signature"


def test_dense_features_are_bounded():
    cl = Clause(
        clause_id="c02",
        start=0,
        end=10,
        heading="Fees",
        text="Fees. INR 5,00,000 within 30 days, 12% per annum, 12 March 2025, laws of India",
    )
    feats = dense_features(cl, 1, 5)
    assert all(0.0 <= f <= 1.0 for f in feats)
    assert feats[5:9] == [1.0, 1.0, 1.0, 1.0]  # money, duration, date, percent cues
