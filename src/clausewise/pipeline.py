"""End-to-end analysis: text in, clauses, labels, entities, risks, redlines and answers out.

The ``Analyzer`` owns the trained classifier and the embedding backend so that the API, the CLI and the
benchmark all run exactly the same code path. Training happens once per process on the seeded corpus (a
few seconds) and is cached on disk by seed and corpus size, which keeps the API start-up fast and the
benchmark reproducible.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib

from clausewise import __version__
from clausewise.classify import ClauseClassifier, split_contracts
from clausewise.config import Settings, load_settings
from clausewise.corpus.generator import load_or_generate
from clausewise.embeddings import Embedder, embedder_name, load_embedder
from clausewise.entities import extract_entities, extract_roles
from clausewise.playbook import DEFAULT_PLAYBOOK, Playbook
from clausewise.qa import Answerer
from clausewise.redline import suggest_redlines
from clausewise.retrieval import ClauseRetriever
from clausewise.risk import detect_risks, risk_score
from clausewise.schemas import AnalysisResult, Answer, ClassifiedClause, Clause, Contract
from clausewise.segment import segment

log = logging.getLogger(__name__)

_TYPE_CUES: tuple[tuple[str, frozenset[str]], ...] = (
    ("lease", frozenset({"rent", "premises", "security_deposit", "maintenance"})),
    ("employment", frozenset({"compensation", "duties", "benefits"})),
    ("saas", frozenset({"subscription", "sla", "acceptable_use", "data_protection"})),
    ("msa", frozenset({"services"})),
)
_TITLE_CUES: tuple[tuple[str, str], ...] = (
    ("lease", "LEASE"), ("lease", "RENT"), ("employment", "EMPLOYMENT"), ("saas", "SAAS"),
    ("saas", "SOFTWARE AS A SERVICE"), ("saas", "CLOUD"), ("msa", "SERVICES AGREEMENT"),
    ("nda", "NON-DISCLOSURE"), ("nda", "CONFIDENTIALITY AGREEMENT"),
)  # fmt: skip


def infer_contract_type(text: str, labels: list[str]) -> str:
    """Contract type from the title line, falling back to which clause types are present."""
    title = text.strip().split("\n", 1)[0].upper()
    for ctype, cue in _TITLE_CUES:
        if cue in title:
            return ctype
    present = set(labels)
    for ctype, cues in _TYPE_CUES:
        if present & cues:
            return ctype
    return "nda"


@dataclass
class Analysis:
    """Internal result that keeps the clause objects for follow-up questions."""

    result: AnalysisResult
    clauses: list[ClassifiedClause]
    roles: tuple[str, str]


class Analyzer:
    """Trained pipeline, shared by API, CLI and benchmark."""

    def __init__(self, settings: Settings | None = None, playbook: Playbook = DEFAULT_PLAYBOOK,
                 classifier: ClauseClassifier | None = None, embedder: Embedder | None = None) -> None:  # fmt: skip
        self.settings = settings or load_settings()
        self.playbook = playbook
        self.classifier = classifier or load_or_train_classifier(self.settings)
        self.embedder = embedder if embedder is not None else load_embedder(self.settings)
        log.info("analyzer ready", extra={"embeddings": embedder_name(self.embedder)})

    @property
    def backend(self) -> str:
        """Name of the embedding backend in use."""
        return embedder_name(self.embedder)

    def analyze(self, text: str) -> Analysis:
        """Run segmentation, classification, entities, risks and redlines on ``text``."""
        if len(text) > self.settings.max_text_chars:
            raise ValueError(f"text longer than {self.settings.max_text_chars} characters")
        raw = segment(text)
        predictions = self.classifier.predict(raw)
        clauses = [
            ClassifiedClause(**c.model_dump(exclude={"label"}), label=label, confidence=conf)
            for c, (label, conf) in zip(raw, predictions, strict=True)
        ]
        entities = extract_entities(text, clauses)  # type: ignore[arg-type]
        findings = detect_risks(clauses, self.playbook)  # type: ignore[arg-type]
        roles = extract_roles(text)
        redlines = suggest_redlines(clauses, findings, roles, self.playbook)  # type: ignore[arg-type]
        ctype = infer_contract_type(text, [c.label or "" for c in clauses])
        score = risk_score(findings)
        summary = (
            f"{ctype.upper()} with {len(clauses)} clauses, {len(findings)} playbook finding(s), "
            f"risk score {score}/100; "
            f"roles {roles[0]} and {roles[1]}."
        )
        result = AnalysisResult(
            contract_type=ctype, clauses=clauses, entities=entities, risks=findings, redlines=redlines,  # type: ignore[arg-type]
            risk_score=score, summary=summary,
        )  # fmt: skip
        return Analysis(result=result, clauses=clauses, roles=roles)

    def retriever(self, clauses: list[Clause], mode: str = "hybrid") -> ClauseRetriever:
        """Build a retriever over ``clauses`` with the analyzer's embedding backend."""
        return ClauseRetriever(clauses, self.embedder, mode)

    def ask(self, clauses: list[Clause], question: str, mode: str = "hybrid", llm: bool = False) -> Answer:
        """Answer ``question`` over already-segmented clauses; ``llm`` is honoured only when enabled in settings."""
        retriever = self.retriever(clauses, mode)
        if llm and self.settings.llm_enabled:
            from clausewise.llm import LLMAnswerer

            return LLMAnswerer(clauses, retriever, self.settings).answer(question)
        return Answerer(clauses, retriever, self.settings).answer(question)

    def analyze_and_ask(self, text: str, question: str) -> Answer:
        """Convenience for the API: segment then answer."""
        return self.ask(segment(text), question)


def classifier_cache_path(settings: Settings) -> Path:
    """Where the trained classifier for these settings is cached."""
    key = hashlib.sha256(f"{__version__}:{settings.seed}:{settings.corpus_size}".encode()).hexdigest()[:12]
    return settings.data_dir / f"classifier_{key}.joblib"


def train_classifier(contracts: list[Contract], settings: Settings, use_heading: bool = True) -> ClauseClassifier:
    """Train on the training split of ``contracts`` (the benchmark evaluates on the rest)."""
    train, _test = split_contracts(contracts, settings.seed)
    docs = [c.clauses for c in train]
    if not use_heading:
        docs = [[cl.model_copy(update={"heading": None}) for cl in doc] for doc in docs]
    return ClauseClassifier(use_heading=use_heading, seed=settings.seed).fit(docs)


def load_or_train_classifier(settings: Settings) -> ClauseClassifier:
    """Load the cached classifier or train it on the seeded corpus and cache it."""
    path = classifier_cache_path(settings)
    if path.exists():
        return joblib.load(path)
    contracts = load_or_generate(settings.data_dir, settings.seed, settings.corpus_size)
    clf = train_classifier(contracts, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    return clf
