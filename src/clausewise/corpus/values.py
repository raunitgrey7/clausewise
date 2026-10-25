"""Value generation and formatting with ground-truth normalisation.

Every placeholder value is a :class:`Value`: the text that goes into the contract, the entity type it
represents (or None) and the normalised form the extractor is expected to recover. Formatting variety
(``thirty (30) days`` against ``30 days``, ``INR 12,50,000`` against ``Rs. 12,50,000``) is decided per
contract so that one document is internally consistent, as real drafting is.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

from clausewise.textutil import int_to_words

MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)  # fmt: skip


@dataclass(frozen=True)
class Value:
    """A rendered placeholder value. ``text`` may itself contain placeholders (nested rendering)."""

    text: str
    entity_type: str | None = None
    normalized: str | None = None


def plain(text: str) -> Value:
    """A value that is not an entity."""
    return Value(text)


def _group_indian(n: int) -> str:
    s = str(n)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join([*parts, tail])


def _group_western(n: int) -> str:
    return f"{n:,}"


def money(rng: random.Random, currency: str, amount: int, style: str) -> Value:
    """Format an amount in one of the drafting styles; normalised as ``CUR <integer>``."""
    if currency == "INR":
        grouped = _group_indian(amount) if style != "western" else _group_western(amount)
        prefix = rng.choice(("INR ", "Rs. ")) if style != "code" else "INR "
        text = f"{prefix}{grouped}"
    else:
        grouped = _group_western(amount)
        symbol = {"USD": ("USD ", "US$ "), "GBP": ("GBP ",), "EUR": ("EUR ",), "SGD": ("SGD ", "S$ ")}[currency]
        text = f"{rng.choice(symbol)}{grouped}"
    return Value(text, "money", f"{currency} {amount}")


def duration(n: int, unit: str, numstyle: str) -> Value:
    """Format a duration such as ``thirty (30) days``; normalised as ``30 days``."""
    unit_text = unit if n != 1 else unit[:-1]
    if numstyle == "words_parens" and n <= 999:
        text = f"{int_to_words(n)} ({n}) {unit_text}"
    elif numstyle == "words" and n <= 999:
        text = f"{int_to_words(n)} {unit_text}"
    else:
        text = f"{n} {unit_text}"
    return Value(text, "duration", f"{n} {unit_text}")


def percentage(rng: random.Random, pct: str) -> Value:
    """Format a percentage as ``99.9%`` or ``99.9 per cent``; normalised with the percent sign."""
    text = f"{pct}%" if rng.random() < 0.7 else f"{pct} per cent"
    return Value(text, "percentage", f"{pct}%")


def format_date(d: date, fmt: str) -> str:
    """Render a date in one of four legal drafting formats."""
    if fmt == "dmy":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    if fmt == "mdy":
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if fmt == "legal":
        suffix = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
        return f"{d.day}{suffix} day of {MONTHS[d.month - 1]}, {d.year}"
    return d.isoformat()


def random_date(rng: random.Random, fmt: str) -> Value:
    """A date between 2023-01-01 and 2026-06-30 in the given format; normalised as ISO."""
    start = date(2023, 1, 1)
    d = start + timedelta(days=rng.randrange(0, 1277))
    return Value(format_date(d, fmt), "date", d.isoformat())


def round_to(n: int, step: int) -> int:
    """Round ``n`` to the nearest multiple of ``step`` (at least one step)."""
    return max(step, round(n / step) * step)


def pick_currency(rng: random.Random, incorporation: str) -> str:
    """Currency follows the paying party's country most of the time."""
    if incorporation == "India":
        return rng.choices(["INR", "USD"], weights=[0.85, 0.15])[0]
    table = {
        "the United States": "USD",
        "England and Wales": "GBP",
        "Singapore": "SGD",
        "Germany": "EUR",
        "Australia": "USD",
    }
    return table.get(incorporation, "USD")


def amount_for(rng: random.Random, currency: str, kind: str) -> int:
    """Draw a plausible amount for a fee, rent, salary, deposit, cap or insurance limit."""
    scale = 80 if currency == "INR" else 1
    ranges = {
        "fee": (10_000, 500_000, 1_000),
        "rent": (400, 6_000, 50),
        "salary": (40_000, 250_000, 1_000),
        "insurance": (100_000, 2_000_000, 50_000),
        "cap": (25_000, 1_000_000, 5_000),
    }
    lo, hi, step = ranges[kind]
    value = rng.randint(lo, hi) * scale
    return round_to(value, step * scale)
