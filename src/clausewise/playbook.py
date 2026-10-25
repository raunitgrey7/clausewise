"""The risk playbook: the positions a reviewer holds and the severities attached to departures from them.

The playbook is plain data so that a legal team can change a position (for example the set of acceptable
governing-law jurisdictions) without touching detection logic. It is also the single source of truth for
the generator's governing-law ground truth, which is why it lives in its own dependency-free module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Position:
    """What the playbook wants for one risk type."""

    severity: str
    weight: int
    position: str
    fallback: str


@dataclass(frozen=True)
class Playbook:
    """Reviewer positions keyed by risk type plus the acceptable governing-law jurisdictions."""

    acceptable_jurisdictions: frozenset[str] = frozenset({"India", "England and Wales", "Singapore"})
    positions: dict[str, Position] = field(
        default_factory=lambda: {
            "unlimited_liability": Position(
                "high",
                30,
                "Aggregate liability of each party is capped; only fraud, wilful misconduct and death or personal "
                "injury are carved out.",
                "Except for liability arising from fraud, wilful misconduct, or death or personal injury caused by "
                "negligence, the total aggregate liability of each party under or in connection with this Agreement "
                "shall not exceed the total fees paid or payable under this Agreement in the twelve (12) months "
                "preceding the event giving rise to the claim. Neither party shall be liable for any indirect or "
                "consequential loss.",
            ),
            "ip_assignment": Position(
                "high",
                25,
                "Each party keeps its pre-existing and background intellectual property; only deliverables created "
                "specifically under the agreement transfer, and only on payment.",
                "Each party retains all right, title and interest in its pre-existing and background intellectual "
                "property. Upon payment in full, {ASSIGNOR} assigns to {ASSIGNEE} the intellectual property rights in "
                "the deliverables created specifically for {ASSIGNEE} under this Agreement, excluding {ASSIGNOR}'s "
                "pre-existing materials, which are licensed to {ASSIGNEE} on a non-exclusive, perpetual basis to the "
                "extent incorporated in the deliverables.",
            ),
            "unilateral_termination": Position(
                "medium",
                15,
                "Termination for convenience, if available, is available to both parties on the same notice.",
                "Either party may terminate this Agreement for convenience by giving the other party not less than "
                "{notice} prior written notice. Either party may terminate this Agreement with immediate effect by "
                "written notice if the other party commits a material breach which is not remedied within thirty "
                "(30) days of written notice requiring remedy.",
            ),
            "non_compete": Position(
                "medium",
                15,
                "No post-termination non-compete; confidentiality and non-solicitation protect legitimate interests.",
                "During the term of this Agreement, {B} shall not provide services to a direct competitor of {A} "
                "without {A}'s prior written consent. This restriction ends on termination of this Agreement and "
                "nothing in this Agreement restricts {B}'s activities after termination.",
            ),
            "auto_renewal": Position(
                "low",
                10,
                "Renewal requires the written agreement of both parties, or at minimum a reminder notice before the "
                "non-renewal deadline.",
                "On expiry of the initial term this Agreement shall terminate unless the parties agree in writing to "
                "extend it. Any extension of the term requires a written agreement signed by both parties.",
            ),
            "governing_law": Position(
                "medium",
                15,
                "Governing law is India, England and Wales or Singapore.",
                "This Agreement and any dispute or claim arising out of or in connection with it shall be governed by "
                "and construed in accordance with the laws of India. Each party irrevocably submits to the exclusive "
                "jurisdiction of the courts at New Delhi.",
            ),
        }
    )

    @classmethod
    def from_json(cls, path: Path) -> Playbook:
        """Load a playbook from JSON; missing positions fall back to the defaults."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        default = cls()
        positions = dict(default.positions)
        for key, value in raw.get("positions", {}).items():
            positions[key] = Position(**value)
        return cls(
            acceptable_jurisdictions=frozenset(raw.get("acceptable_jurisdictions", default.acceptable_jurisdictions)),
            positions=positions,
        )


DEFAULT_PLAYBOOK = Playbook()
