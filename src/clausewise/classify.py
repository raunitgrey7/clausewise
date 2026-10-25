"""Clause classification: deterministic features plus a small linear model.

Two feature families feed a multinomial logistic regression: TF-IDF over the clause body and (when the
pipeline is allowed to see it) over the heading, and a short dense vector of structural cues (relative
position in the document, length, presence of money, durations, dates, percentages, a governing-law
phrase). The dense cues are what let the model keep working when headings are missing or generic
("General", "Miscellaneous"). A heading-keyword rule baseline is kept so the benchmark can show what the
trained model adds over the obvious approach.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from clausewise.schemas import Clause
from clausewise.segment import body_text

_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z-]+\b"
_CUES: tuple[tuple[str, str], ...] = (
    ("money", r"\b(?:INR|Rs\.|USD|US\$|GBP|EUR|SGD|S\$)\s*\d|\$\d"),
    ("duration", r"\b\d+\)?\s+(?:days?|months?|years?|hours?)\b"),
    ("date", r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"),
    ("percent", r"\d\s*(?:%|per cent)"),
    ("laws_of", r"\blaws of\b"),
    ("whereas", r"\bWHEREAS\b"),
    ("witness", r"\bIN WITNESS WHEREOF\b"),
    ("quoted_def", r'"[A-Z][A-Za-z ]+" (?:means|has the meaning)'),
    ("shall_not", r"\bshall not\b"),
    ("either_party", r"\beither party\b"),
)
_CUE_RES = [(name, re.compile(pattern)) for name, pattern in _CUES]

HEADING_RULES: tuple[tuple[str, str], ...] = (
    ("definitions", r"definition|interpretation"),
    ("confidentiality", r"confidential|non-disclosure"),
    ("exclusions", r"exclusion|exception|excluded"),
    ("term", r"^term\b|duration|renewal"),
    ("termination", r"terminat|ending"),
    ("liability", r"liabilit"),
    ("indemnification", r"indemn"),
    ("ip", r"intellectual|ownership|proprietary"),
    ("warranties", r"warrant"),
    ("non_compete", r"compet|restrictive"),
    ("non_solicit", r"solicit"),
    ("governing_law", r"governing|applicable law|^law$"),
    ("dispute_resolution", r"dispute|arbitration"),
    ("assignment", r"assign|sublet|transfer"),
    ("notices", r"notice|communication"),
    ("entire_agreement", r"entire|whole agreement|general|miscellaneous"),
    ("severability", r"sever|invalidity"),
    ("force_majeure", r"force majeure|beyond control|excused"),
    ("insurance", r"insurance"),
    ("data_protection", r"data"),
    ("services", r"services"),
    ("payment", r"fee|payment|charges"),
    ("subscription", r"subscription|access|licence grant|provision of the service"),
    ("acceptable_use", r"acceptable|restrictions|customer obligations"),
    ("sla", r"service level|availability|support"),
    ("duties", r"duties|position|role|appointment"),
    ("compensation", r"remuneration|salary|compensation|pay"),
    ("benefits", r"benefit|leave|holiday"),
    ("premises", r"premises|demise|grant of lease|permitted use"),
    ("rent", r"rent"),
    ("security_deposit", r"deposit"),
    ("maintenance", r"maintenance|repair|condition"),
    ("return_of_materials", r"return"),
    ("remedies", r"remed|injunctive|equitable"),
)
_HEADING_RULES = [(label, re.compile(pattern, re.I)) for label, pattern in HEADING_RULES]


def rule_label(clause: Clause, index: int, total: int) -> str:
    """Baseline: label from the heading keyword table, with preamble/recitals/signature by position cues."""
    text = clause.text
    if index == 0:
        return "preamble"
    if text.lstrip().startswith(("WHEREAS", "RECITALS", "BACKGROUND")):
        return "recitals"
    if "IN WITNESS WHEREOF" in text or (index == total - 1 and not clause.heading):
        return "signature"
    if clause.heading:
        for label, pattern in _HEADING_RULES:
            if pattern.search(clause.heading):
                return label
    return "entire_agreement"


def dense_features(clause: Clause, index: int, total: int) -> list[float]:
    """Structural cues that do not depend on vocabulary."""
    text = clause.text
    n_tokens = max(1, len(text.split()))
    feats = [
        index / max(1, total - 1),
        1.0 if index == 0 else 0.0,
        1.0 if index == total - 1 else 0.0,
        min(1.0, np.log1p(n_tokens) / 7.0),
        1.0 if clause.heading else 0.0,
    ]
    feats.extend(1.0 if pattern.search(text) else 0.0 for _name, pattern in _CUE_RES)
    return feats


@dataclass
class _Prepared:
    bodies: list[str]
    headings: list[str]
    dense: np.ndarray


def _prepare(docs: list[list[Clause]]) -> _Prepared:
    bodies: list[str] = []
    headings: list[str] = []
    dense: list[list[float]] = []
    for doc in docs:
        for i, clause in enumerate(doc):
            bodies.append(body_text(clause))
            headings.append(clause.heading or "")
            dense.append(dense_features(clause, i, len(doc)))
    return _Prepared(bodies, headings, np.asarray(dense, dtype=np.float64))


class ClauseClassifier:
    """TF-IDF (body and optionally heading) + dense cues -> logistic regression."""

    def __init__(self, use_heading: bool = True, seed: int = 42, c: float = 8.0) -> None:
        self.use_heading = use_heading
        self.body_vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=60_000, token_pattern=_TOKEN_PATTERN
        )
        self.head_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, token_pattern=_TOKEN_PATTERN)
        self.model = LogisticRegression(C=c, max_iter=5000, random_state=seed)
        self.classes_: list[str] = []

    def _matrix(self, prep: _Prepared, fit: bool) -> sparse.csr_matrix:
        if fit:
            body = self.body_vec.fit_transform(prep.bodies)
            head = self.head_vec.fit_transform(prep.headings) if self.use_heading else None
        else:
            body = self.body_vec.transform(prep.bodies)
            head = self.head_vec.transform(prep.headings) if self.use_heading else None
        blocks = [body, sparse.csr_matrix(prep.dense * 2.0)]
        if head is not None:
            blocks.insert(1, head)
        return sparse.hstack(blocks).tocsr()

    def fit(self, docs: list[list[Clause]]) -> ClauseClassifier:
        """Train on documents whose clauses carry gold labels."""
        prep = _prepare(docs)
        labels = [clause.label or "unknown" for doc in docs for clause in doc]
        self.model.fit(self._matrix(prep, fit=True), labels)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_proba_docs(self, docs: list[list[Clause]]) -> np.ndarray:
        """Class probabilities for every clause of every document, in order."""
        prep = _prepare(docs)
        return self.model.predict_proba(self._matrix(prep, fit=False))

    def predict(self, doc: list[Clause]) -> list[tuple[str, float]]:
        """(label, confidence) for each clause of one document."""
        if not doc:
            return []
        proba = self.predict_proba_docs([doc])
        idx = proba.argmax(axis=1)
        return [(self.classes_[int(i)], float(proba[row, i])) for row, i in enumerate(idx)]

    def predict_docs(self, docs: list[list[Clause]]) -> list[str]:
        """Flat list of predicted labels for several documents (benchmark use)."""
        if not docs:
            return []
        proba = self.predict_proba_docs(docs)
        return [self.classes_[int(i)] for i in proba.argmax(axis=1)]


def split_contracts[T](items: list[T], seed: int, test_fraction: float = 0.3) -> tuple[list[T], list[T]]:
    """Deterministic train/test split by contract."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(items))
    n_test = round(len(items) * test_fraction)
    test_idx = set(order[:n_test].tolist())
    train = [x for i, x in enumerate(items) if i not in test_idx]
    test = [x for i, x in enumerate(items) if i in test_idx]
    return train, test
