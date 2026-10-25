"""Contract plans: which labels each contract type contains, in what order, with what probability.

The inclusion probabilities and the risk-variant mixtures below are the statistical specification of
the corpus. They are documented in docs/EVALUATION.md and must be changed there too if edited here.
"""

from __future__ import annotations

# (label, probability of inclusion), in drafting order. Labels in BOILERPLATE may be shuffled.
PLANS: dict[str, tuple[tuple[str, float], ...]] = {
    "nda": (
        ("definitions", 0.9), ("confidentiality", 1.0), ("exclusions", 0.85), ("term", 1.0),
        ("return_of_materials", 0.8), ("remedies", 0.7), ("liability", 0.5), ("governing_law", 1.0),
        ("dispute_resolution", 0.7), ("assignment", 0.5), ("notices", 0.8), ("entire_agreement", 0.9),
        ("severability", 0.8),
    ),
    "msa": (
        ("definitions", 0.9), ("services", 1.0), ("payment", 1.0), ("term", 1.0), ("termination", 1.0),
        ("confidentiality", 0.9), ("ip", 1.0), ("warranties", 0.8), ("indemnification", 0.8), ("liability", 1.0),
        ("insurance", 0.5), ("non_compete", 0.5), ("non_solicit", 0.5), ("force_majeure", 0.7),
        ("governing_law", 1.0), ("dispute_resolution", 0.8), ("assignment", 0.8), ("notices", 0.8),
        ("entire_agreement", 0.9), ("severability", 0.8),
    ),
    "saas": (
        ("definitions", 0.9), ("subscription", 1.0), ("acceptable_use", 0.9), ("sla", 0.9), ("payment", 1.0),
        ("term", 1.0), ("termination", 1.0), ("data_protection", 0.9), ("confidentiality", 0.9), ("ip", 1.0),
        ("warranties", 0.7), ("indemnification", 0.7), ("liability", 1.0), ("force_majeure", 0.6),
        ("governing_law", 1.0), ("dispute_resolution", 0.7), ("assignment", 0.7), ("notices", 0.8),
        ("entire_agreement", 0.9), ("severability", 0.8),
    ),
    "employment": (
        ("definitions", 0.4), ("duties", 1.0), ("compensation", 1.0), ("benefits", 0.9), ("term", 0.9),
        ("termination", 1.0), ("confidentiality", 0.9), ("ip", 0.9), ("non_compete", 0.7), ("non_solicit", 0.5),
        ("governing_law", 1.0), ("dispute_resolution", 0.6), ("notices", 0.7), ("entire_agreement", 0.8),
        ("severability", 0.8),
    ),
    "lease": (
        ("definitions", 0.6), ("premises", 1.0), ("rent", 1.0), ("security_deposit", 0.9), ("term", 1.0),
        ("termination", 0.9), ("maintenance", 0.9), ("insurance", 0.5), ("liability", 0.5), ("assignment", 0.9),
        ("governing_law", 1.0), ("dispute_resolution", 0.7), ("notices", 0.8), ("entire_agreement", 0.8),
        ("severability", 0.7),
    ),
}  # fmt: skip

BOILERPLATE: frozenset[str] = frozenset(
    {
        "governing_law",
        "dispute_resolution",
        "assignment",
        "notices",
        "entire_agreement",
        "severability",
        "force_majeure",
    }
)

CONTRACT_TYPE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("nda", 0.20), ("msa", 0.25), ("saas", 0.20), ("employment", 0.20), ("lease", 0.15),
)  # fmt: skip

# Risk-bearing labels: variant name -> probability. Variants named in clause_templates.py.
VARIANT_MIX: dict[str, tuple[tuple[str, float], ...]] = {
    "term": (("fixed", 0.45), ("auto_renew", 0.35), ("hard_negative", 0.20)),
    "termination": (("mutual", 0.35), ("for_cause_only", 0.15), ("unilateral", 0.30), ("hard_negative", 0.20)),
    "liability": (("capped", 0.50), ("unlimited", 0.30), ("hard_negative", 0.20)),
    "ip": (("retain", 0.35), ("deliverables", 0.20), ("broad", 0.30), ("hard_negative", 0.15)),
    "non_compete": (("post_term", 0.55), ("during_term", 0.25), ("hard_negative", 0.20)),
}

ROLES: dict[str, tuple[tuple[str, str], ...]] = {
    "nda": (("Disclosing Party", "Receiving Party"), ("Discloser", "Recipient")),
    "msa": (("Supplier", "Customer"), ("Service Provider", "Client")),
    "saas": (("Provider", "Customer"), ("Licensor", "Subscriber")),
    "employment": (("Company", "Employee"), ("Employer", "Employee")),
    "lease": (("Landlord", "Tenant"), ("Lessor", "Lessee")),
}

# Which role assigns intellectual property to which, per contract type ("A" or "B").
IP_ROLES: dict[str, tuple[str, str]] = {"msa": ("A", "B"), "saas": ("B", "A"), "employment": ("B", "A")}

TITLES: dict[str, tuple[str, ...]] = {
    "nda": ("NON-DISCLOSURE AGREEMENT", "CONFIDENTIALITY AGREEMENT", "MUTUAL NON-DISCLOSURE AGREEMENT"),
    "msa": ("MASTER SERVICES AGREEMENT", "SERVICES AGREEMENT", "PROFESSIONAL SERVICES AGREEMENT"),
    "saas": ("SOFTWARE AS A SERVICE AGREEMENT", "SAAS SUBSCRIPTION AGREEMENT", "CLOUD SERVICES AGREEMENT"),
    "employment": ("EMPLOYMENT AGREEMENT", "CONTRACT OF EMPLOYMENT", "EMPLOYMENT CONTRACT"),
    "lease": ("LEASE AGREEMENT", "LEASE DEED", "COMMERCIAL LEASE AGREEMENT", "RENT AGREEMENT"),
}
AGREEMENT_NOUN: dict[str, str] = {
    "nda": "Non-Disclosure Agreement",
    "msa": "Master Services Agreement",
    "saas": "Subscription Agreement",
    "employment": "Employment Agreement",
    "lease": "Lease",
}

RECITALS: dict[str, tuple[tuple[str, ...], ...]] = {
    "nda": (
        (
            "WHEREAS {A} possesses certain confidential and proprietary information relating to its business;",
            "WHEREAS {B} wishes to receive such information for the purpose of evaluating a potential business "
            "relationship between the parties (the Purpose);",
        ),
        (
            "WHEREAS the parties wish to explore a commercial relationship and in connection with it {A} may "
            "disclose Confidential Information to {B};",
            "WHEREAS {B} is willing to receive and protect such information on the terms set out below;",
        ),
    ),
    "msa": (
        (
            "WHEREAS {A} is engaged in the business of providing {service_area} services;",
            "WHEREAS {B} wishes to engage {A} to provide such services on the terms of this Agreement;",
        ),
        (
            "WHEREAS {B} requires {service_area} services and {A} has the expertise to provide them;",
            "WHEREAS the parties wish to set out the terms on which {A} will provide services to {B};",
        ),
    ),
    "saas": (
        (
            "WHEREAS {A} operates a cloud-hosted software platform for {service_area};",
            "WHEREAS {B} wishes to subscribe to the platform on the terms of this Agreement;",
        ),
        (
            "WHEREAS {A} makes its software available to customers as a service over the internet;",
            "WHEREAS {B} wishes to obtain access to the software for its internal business purposes;",
        ),
    ),
    "employment": (
        (
            "WHEREAS {A} wishes to employ {B} and {B} wishes to accept employment with {A} on the terms set out "
            "in this Agreement;",
        ),
        (
            "WHEREAS {A} carries on the business of {service_area} and wishes to engage {B} in the role described "
            "below;",
        ),
    ),
    "lease": (
        (
            "WHEREAS {A} is the owner of the Premises described below;",
            "WHEREAS {B} wishes to take the Premises on lease and {A} has agreed to grant such lease on the terms "
            "set out in this Lease;",
        ),
        ("WHEREAS {A} has agreed to let and {B} has agreed to take the Premises on the terms set out below;",),
    ),
}
