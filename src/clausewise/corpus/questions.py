"""Question bank for the grounded QA benchmark.

Paraphrases deliberately use vocabulary that the clauses do not ("roll over", "evergreen", "barred from
competing", "contract price") so that lexical retrieval cannot win by string matching alone and the
dense backend has something to add. Each spec names the clause label that holds the answer and, when
the answer is a value, the placeholder key whose normalised form is the gold short answer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuestionSpec:
    """A family of paraphrased questions sharing one gold clause label and answer type."""

    label: str
    answer_type: str
    value_key: str | None
    paraphrases: tuple[str, ...]
    variants: tuple[str, ...] | None = None
    true_variants: tuple[str, ...] | None = None
    contract_types: tuple[str, ...] | None = None


QUESTION_BANK: tuple[QuestionSpec, ...] = (
    QuestionSpec(
        "governing_law",
        "jurisdiction",
        "jurisdiction",
        (
            "Which law governs this agreement?",
            "What is the governing law of the contract?",
            "Under the laws of which jurisdiction will this agreement be construed?",
            "Which country's legal system applies to this contract?",
            "What legal system do the parties pick for interpreting the document?",
        ),
    ),
    QuestionSpec(
        "termination",
        "duration",
        "notice",
        (
            "How much notice is required to terminate the agreement?",
            "What is the notice period for termination?",
            "How many days' notice must a party give to end the contract?",
            "How long in advance must a party announce that it is walking away?",
        ),
    ),
    QuestionSpec(
        "termination",
        "clause",
        None,
        (
            "On what grounds can the agreement be terminated?",
            "Can either side terminate for convenience?",
            "What happens when the relationship is ended early?",
        ),
    ),
    QuestionSpec(
        "term",
        "duration",
        "term",
        (
            "How long is the initial term of the agreement?",
            "What is the duration of the contract?",
            "For how long does this agreement run?",
            "How many years or months does the deal last?",
        ),
    ),
    QuestionSpec(
        "term",
        "boolean",
        None,
        (
            "Does the agreement renew automatically?",
            "Is there an automatic renewal provision?",
            "Will the contract roll over at the end of the term?",
            "Is this an evergreen agreement?",
        ),
        true_variants=("auto_renew",),
    ),
    QuestionSpec(
        "liability",
        "boolean",
        None,
        (
            "Is either party's liability unlimited?",
            "Does the agreement leave liability uncapped for anyone?",
            "Is there a party whose exposure has no ceiling?",
        ),
        true_variants=("unlimited",),
    ),
    QuestionSpec(
        "liability",
        "money_or_fees",
        "cap",
        (
            "What is the cap on liability?",
            "To what amount is liability limited?",
            "What is the maximum aggregate liability under the agreement?",
            "How much can one side be made to pay at most?",
        ),
        variants=("capped", "hard_negative"),
    ),
    QuestionSpec(
        "payment",
        "money",
        "fee",
        (
            "What are the fees payable under this agreement?",
            "How much does the customer pay per year?",
            "What is the annual fee for the services?",
            "What is the contract price?",
        ),
    ),
    QuestionSpec(
        "payment",
        "duration",
        "payment_days",
        (
            "When are invoices due?",
            "What are the payment terms for invoices?",
            "Within how many days must invoices be paid?",
            "How long does the customer have to settle a bill?",
        ),
    ),
    QuestionSpec(
        "rent",
        "money",
        "rent",
        (
            "What is the monthly rent?",
            "How much rent is payable each month?",
            "What does the tenant pay for the space?",
        ),
    ),
    QuestionSpec(
        "security_deposit",
        "money",
        "deposit",
        (
            "What is the security deposit?",
            "How much deposit must the tenant pay?",
            "What sum is held as security under the lease?",
        ),
    ),
    QuestionSpec(
        "compensation",
        "money",
        "salary",
        (
            "What is the annual salary?",
            "How much is the employee paid per year?",
            "What base pay does the employee receive?",
        ),
    ),
    QuestionSpec(
        "confidentiality",
        "duration",
        "conf_duration",
        (
            "How long do the confidentiality obligations last?",
            "For what period must confidential information be protected after termination?",
            "What is the survival period of the confidentiality clause?",
            "How many years must secrets be kept once the deal ends?",
        ),
    ),
    QuestionSpec(
        "non_compete",
        "duration",
        "noncompete",
        (
            "How long is the non-compete restriction after termination?",
            "For what period is the restricted party barred from competing?",
            "What is the duration of the restrictive covenant?",
        ),
        variants=("post_term",),
    ),
    QuestionSpec(
        "non_compete",
        "boolean",
        None,
        (
            "Is there a non-compete restriction that continues after termination?",
            "Is the party prevented from working for competitors once the agreement ends?",
            "Does a post-termination non-compete apply?",
        ),
        true_variants=("post_term",),
    ),
    QuestionSpec(
        "sla",
        "percentage",
        "sla",
        (
            "What uptime does the provider commit to?",
            "What is the availability service level?",
            "What monthly uptime percentage is guaranteed?",
        ),
    ),
    QuestionSpec(
        "data_protection",
        "duration",
        "breach_hours",
        (
            "How quickly must the provider notify a personal data breach?",
            "What is the data breach notification deadline?",
            "Within what time must a security incident be reported to the customer?",
        ),
    ),
    QuestionSpec(
        "insurance",
        "money",
        "insurance",
        (
            "What level of insurance cover is required?",
            "What is the minimum insurance limit per claim?",
            "How much insurance must be maintained?",
        ),
    ),
    QuestionSpec(
        "benefits",
        "duration",
        "leave_days",
        ("How many days of annual leave does the employee get?", "What is the paid leave entitlement?"),
    ),
    QuestionSpec(
        "return_of_materials",
        "duration",
        "return_days",
        (
            "How quickly must confidential materials be returned or destroyed?",
            "What is the deadline for certifying destruction of confidential information?",
        ),
    ),
    QuestionSpec(
        "preamble",
        "date",
        "effective_date",
        (
            "When was the agreement signed?",
            "What is the effective date?",
            "On what date does the agreement take effect?",
        ),
    ),
    QuestionSpec(
        "preamble",
        "parties",
        None,
        (
            "Who are the parties to this agreement?",
            "Which entities entered into this contract?",
            "Who signed this deal?",
        ),
    ),
    QuestionSpec(
        "ip",
        "clause",
        None,
        (
            "Who owns the intellectual property created under the agreement?",
            "What happens to intellectual property rights in the deliverables?",
            "Does the agreement transfer ownership of inventions?",
        ),
    ),
    QuestionSpec(
        "force_majeure",
        "clause",
        None,
        (
            "What happens if a party cannot perform because of events beyond its control?",
            "Is there a force majeure provision?",
            "How are natural disasters or strikes dealt with?",
        ),
    ),
    QuestionSpec(
        "dispute_resolution",
        "clause",
        None,
        ("How are disputes resolved?", "Is there an arbitration clause?", "Where would a disagreement be heard?"),
    ),
    QuestionSpec(
        "indemnification",
        "clause",
        None,
        (
            "Who indemnifies whom under this agreement?",
            "What are the indemnity obligations?",
            "Is there protection against third-party infringement claims?",
        ),
    ),
    QuestionSpec(
        "assignment",
        "clause",
        None,
        (
            "Can the agreement be assigned to a third party?",
            "Is consent needed to assign or subcontract?",
            "Can rights under the contract be transferred to someone else?",
        ),
    ),
    QuestionSpec(
        "notices",
        "clause",
        None,
        ("How must notices be given under the agreement?", "When is a notice deemed received?"),
    ),
    QuestionSpec(
        "maintenance",
        "clause",
        None,
        ("Who is responsible for repairs to the premises?", "What are the tenant's maintenance obligations?"),
    ),
    QuestionSpec(
        "duties",
        "clause",
        None,
        ("What is the employee's role?", "To whom does the employee report?", "Where is the normal place of work?"),
    ),
    QuestionSpec(
        "warranties",
        "clause",
        None,
        ("What warranties does the supplier give?", "Are implied warranties excluded?"),
    ),
    QuestionSpec(
        "acceptable_use",
        "clause",
        None,
        ("What uses of the service are prohibited?", "Can the provider suspend access for misuse?"),
    ),
    QuestionSpec(
        "subscription",
        "clause",
        None,
        ("What rights does the customer get to use the service?", "Is the licence to the software exclusive?"),
    ),
    QuestionSpec(
        "exclusions",
        "clause",
        None,
        (
            "What information is excluded from confidentiality obligations?",
            "Is publicly available information treated as confidential?",
        ),
    ),
    QuestionSpec(
        "remedies",
        "clause",
        None,
        ("Can the disclosing party seek an injunction?", "What remedies are available for breach of the NDA?"),
    ),
    QuestionSpec(
        "premises",
        "clause",
        None,
        ("What may the premises be used for?", "Which property is being let?"),
    ),
    QuestionSpec(
        "services",
        "clause",
        None,
        ("What services will be provided?", "How are changes to the scope of work handled?"),
    ),
    QuestionSpec(
        "non_solicit",
        "clause",
        None,
        ("Can either party hire the other's employees?", "Is there a non-solicitation restriction?"),
    ),
)

VALUE_ANSWER_TYPES: frozenset[str] = frozenset(
    {"duration", "money", "percentage", "date", "jurisdiction", "boolean", "parties", "money_or_fees"}
)
