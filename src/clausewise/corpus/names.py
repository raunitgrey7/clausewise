"""Name, address and jurisdiction pools for the corpus generator.

Indian and Western pools are mixed on purpose: the target user reviews contracts in Delhi, so Indian
company suffixes ("Private Limited", "Pvt. Ltd.") and INR amounts with lakh grouping must be first-class,
while cross-border paper (US, UK, Singapore) is what makes the governing-law playbook rule meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPANY_PREFIXES: tuple[str, ...] = (
    "Zenith", "Kalpana", "Brightwater", "Aarohan", "Northgate", "Meridian", "Saffron", "Indus", "Cobalt",
    "Lotus", "Harbour", "Quantum", "Sterling", "Everest", "Nimbus", "Trident", "Orchid", "Granite", "Pioneer",
    "Vedant", "Aster", "Kestrel", "Marigold", "Suryoday", "Bluegrass", "Ironwood", "Nalanda", "Peregrine",
    "Tamarind", "Westbrook", "Chandra", "Halcyon", "Juniper", "Kaveri", "Lumen", "Mandala", "Oakridge",
)  # fmt: skip
COMPANY_SECTORS: tuple[str, ...] = (
    "Technologies", "Systems", "Analytics", "Logistics", "Consulting", "Software", "Infotech", "Solutions",
    "Holdings", "Ventures", "Digital", "Labs", "Industries", "Enterprises", "Networks", "Capital", "Media",
)  # fmt: skip

FIRST_NAMES: tuple[str, ...] = (
    "Arjun", "Meera", "Rohan", "Priya", "Vikram", "Ananya", "Kabir", "Nisha", "Aditya", "Sneha", "Rahul",
    "Divya", "Siddharth", "Pooja", "Imran", "Fatima", "Daniel", "Sarah", "Michael", "Emily", "Thomas", "Hannah",
    "Oliver", "Chloe", "Wei", "Mei Ling", "Rajesh", "Lakshmi", "Farhan", "Zara",
)  # fmt: skip
LAST_NAMES: tuple[str, ...] = (
    "Sharma", "Verma", "Iyer", "Nair", "Patel", "Reddy", "Khan", "Gupta", "Mehta", "Bose", "Chaudhary",
    "Pillai", "Desai", "Kulkarni", "Thakur", "Singh", "Smith", "Johnson", "Williams", "Brown", "Taylor",
    "Anderson", "Clarke", "Tan", "Lim", "Ng", "Mueller", "Fischer",
)  # fmt: skip

CITIES: tuple[str, ...] = (
    "New Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Gurugram", "Noida", "Kolkata",
    "Ahmedabad", "London", "Singapore", "New York", "San Francisco", "Berlin", "Sydney",
)  # fmt: skip
STREETS: tuple[str, ...] = (
    "MG Road", "Nehru Place", "Cyber City", "Bandra Kurla Complex", "Koramangala", "Hitec City",
    "Anna Salai", "Sector 62", "Park Street", "Baker Street", "Market Street", "Collins Avenue",
    "Orchard Road", "Friedrichstrasse", "George Street", "Connaught Place",
)  # fmt: skip

CITIES_BY_COUNTRY: dict[str, tuple[str, ...]] = {
    "India": ("New Delhi", "Mumbai", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Gurugram", "Noida", "Kolkata"),
    "the United States": ("New York", "San Francisco"),
    "England and Wales": ("London",),
    "Singapore": ("Singapore",),
    "Germany": ("Berlin",),
    "Australia": ("Sydney",),
}

JOB_TITLES: tuple[str, ...] = (
    "Software Engineer", "Senior Data Analyst", "Product Manager", "Legal Counsel", "Operations Lead",
    "Account Executive", "Machine Learning Engineer", "Finance Manager", "Quality Assurance Lead",
)  # fmt: skip
MANAGER_TITLES: tuple[str, ...] = (
    "Chief Technology Officer", "Head of Engineering", "Director of Operations", "Chief Executive Officer",
    "Vice President of Sales", "General Counsel",
)  # fmt: skip
PERMITTED_USES: tuple[str, ...] = (
    "office and administrative purposes", "residential purposes", "retail purposes",
    "warehousing and light storage", "a software development centre",
)  # fmt: skip
SERVICE_AREAS: tuple[str, ...] = (
    "software development", "data analytics", "cloud infrastructure", "management consulting",
    "digital marketing", "information technology", "payroll processing",
)  # fmt: skip
TERRITORIES: tuple[str, ...] = (
    "India", "the Republic of India", "the United Kingdom", "the United States", "the Asia Pacific region",
    "any territory in which {A} carries on business", "the National Capital Region of Delhi",
)  # fmt: skip
DATA_REGIONS: tuple[str, ...] = ("India", "the European Union", "Singapore", "the United States")
ARBITRATION: tuple[tuple[str, str], ...] = (
    ("the Singapore International Arbitration Centre", "Singapore"),
    ("the London Court of International Arbitration", "London"),
    ("the Mumbai Centre for International Arbitration", "Mumbai"),
    ("the Delhi International Arbitration Centre", "New Delhi"),
    ("the International Chamber of Commerce", "Paris"),
)


@dataclass(frozen=True)
class Jurisdiction:
    """A governing-law option. ``name`` is the normalised value the entity extractor must produce."""

    name: str
    law_phrase: str
    courts: str
    weight: float


JURISDICTIONS: tuple[Jurisdiction, ...] = (
    Jurisdiction("India", "India", "the courts at New Delhi", 0.40),
    Jurisdiction("England and Wales", "England and Wales", "the courts of England and Wales", 0.12),
    Jurisdiction("Singapore", "Singapore", "the courts of Singapore", 0.10),
    Jurisdiction("New York", "the State of New York", "the state and federal courts located in New York County", 0.10),
    Jurisdiction("Delaware", "the State of Delaware", "the courts of the State of Delaware", 0.08),
    Jurisdiction("California", "the State of California", "the courts located in San Francisco County", 0.06),
    Jurisdiction("Germany", "the Federal Republic of Germany", "the courts of Berlin", 0.05),
    Jurisdiction("New South Wales", "New South Wales, Australia", "the courts of New South Wales", 0.05),
    Jurisdiction("DIFC", "the Dubai International Financial Centre", "the DIFC Courts", 0.04),
)

# Company suffix pools keyed by the country of incorporation.
COMPANY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "India": ("Private Limited", "Pvt. Ltd.", "Limited"),
    "the United States": ("Inc.", "LLC", "Corporation"),
    "England and Wales": ("Limited", "Ltd.", "plc"),
    "Singapore": ("Pte. Ltd.",),
    "Germany": ("GmbH",),
    "Australia": ("Pty Ltd",),
}
INCORPORATION_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("India", 0.55),
    ("the United States", 0.15),
    ("England and Wales", 0.12),
    ("Singapore", 0.10),
    ("Germany", 0.04),
    ("Australia", 0.04),
)
