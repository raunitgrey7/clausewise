"""Typed data contracts shared by the generator, the pipeline, the benchmark and the API.

Pydantic models are used instead of dataclasses because the same objects are serialised to the corpus
cache, returned by FastAPI and compared in the benchmark; one schema with validation keeps the three
consumers honest about field names and types.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ContractType = Literal["nda", "msa", "saas", "employment", "lease"]
RiskType = Literal[
    "unlimited_liability",
    "auto_renewal",
    "unilateral_termination",
    "non_compete",
    "ip_assignment",
    "governing_law",
]
Severity = Literal["high", "medium", "low"]
EntityType = Literal["party", "date", "money", "duration", "jurisdiction", "percentage"]

RISK_TYPES: tuple[str, ...] = (
    "unlimited_liability",
    "auto_renewal",
    "unilateral_termination",
    "non_compete",
    "ip_assignment",
    "governing_law",
)


class Party(BaseModel):
    """A contracting party as named in the preamble."""

    name: str
    role: str


class Entity(BaseModel):
    """A typed value found in the text, with character offsets into the full contract text."""

    type: EntityType
    value: str = Field(description="Normalised value, for example '30 days', 'INR 1250000', '2025-03-12'")
    start: int
    end: int
    clause_id: str | None = None


class Clause(BaseModel):
    """A contiguous span of a contract with an optional heading."""

    clause_id: str
    start: int
    end: int
    heading: str | None = None
    text: str
    label: str | None = None


class PlantedRisk(BaseModel):
    """Ground truth: a risk the generator deliberately wrote into a clause."""

    risk_type: RiskType
    clause_id: str


class Question(BaseModel):
    """A benchmark question with its gold clause and, when the question has a short answer, its value."""

    question: str
    gold_clause_id: str | None
    gold_label: str
    answer_type: str
    gold_value: str | None = None
    answerable: bool = True


class Contract(BaseModel):
    """A generated contract with complete ground truth."""

    contract_id: str
    contract_type: ContractType
    title: str
    text: str
    clauses: list[Clause]
    parties: list[Party]
    entities: list[Entity]
    risks: list[PlantedRisk]
    questions: list[Question]
    style: dict[str, str | int | float | bool]


class RiskFinding(BaseModel):
    """A playbook rule that fired on a clause, with the evidence that made it fire."""

    risk_type: RiskType
    clause_id: str
    severity: Severity
    weight: int
    evidence: str
    explanation: str
    playbook_position: str


class Redline(BaseModel):
    """A suggested replacement for a risky clause, with a unified diff and the reason."""

    clause_id: str
    risk_type: RiskType
    original: str
    proposed: str
    diff: str
    rationale: str


class ClassifiedClause(Clause):
    """A clause after classification; ``label`` is the predicted type and ``confidence`` its probability."""

    confidence: float = 0.0


class AnalysisResult(BaseModel):
    """Everything the pipeline derives from a contract text."""

    contract_type: ContractType
    clauses: list[ClassifiedClause]
    entities: list[Entity]
    risks: list[RiskFinding]
    redlines: list[Redline]
    risk_score: int = Field(ge=0, le=100)
    summary: str


class AnswerSentence(BaseModel):
    """One sentence of an answer, with the clause ids that support it and the checker verdict."""

    text: str
    citations: list[str]
    grounded: bool
    support: float = Field(ge=0.0, le=1.0)


class Answer(BaseModel):
    """A grounded answer. ``sentences`` only contains sentences that passed the grounding checker."""

    question: str
    answer_type: str
    sentences: list[AnswerSentence]
    short_answer: str | None = None
    rejected: list[AnswerSentence] = Field(default_factory=list)
    retrieved: list[str] = Field(default_factory=list)
    method: Literal["extractive", "llm"] = "extractive"
    abstained: bool = False
