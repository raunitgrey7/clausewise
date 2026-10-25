"""Entity extraction: durations, amounts, dates, percentages, governing-law jurisdictions and parties.

Pure regular expressions with explicit normalisation. The patterns tolerate newlines inside values
because PDF-extracted contracts wrap mid-phrase ("thirty (30)\\ndays"). Jurisdiction extraction is
context bound ("governed by the laws of ...") so that "a company incorporated under the laws of India" in
the preamble is not mistaken for the governing law.
"""

from __future__ import annotations

import re
from datetime import date

from clausewise.corpus.names import JURISDICTIONS
from clausewise.corpus.values import MONTHS
from clausewise.schemas import Clause, Entity
from clausewise.textutil import NUMBER_WORDS, normalize_ws, words_to_int

_UNIT = r"(?P<unit>days?|months?|years?|hours?|weeks?)"
_WORDNUM = r"(?:" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r"|hundred)"
_DUR_WORDS_PARENS = re.compile(rf"\b(?P<words>{_WORDNUM}(?:[\s-]+{_WORDNUM})*)\s*\((?P<digits>\d+)\)\s+{_UNIT}\b", re.I)
_DUR_DIGITS = re.compile(rf"(?<![\w.])(?P<digits>\d{{1,4}})\s+{_UNIT}\b", re.I)
_DUR_WORDS = re.compile(rf"\b(?P<words>{_WORDNUM}(?:[\s-]+{_WORDNUM})*)\s+{_UNIT}\b", re.I)
_MONEY = re.compile(r"(?P<cur>INR|Rs\.|USD|US\$|GBP|EUR|SGD|S\$|\$)\s*(?P<amt>\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+)")
_MONTH = "|".join(MONTHS)
_DATE_DMY = re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?(?P<m>{_MONTH}),?\s+(?P<y>\d{{4}})\b")
_DATE_MDY = re.compile(rf"\b(?P<m>{_MONTH})\s+(?P<d>\d{{1,2}}),\s+(?P<y>\d{{4}})\b")
_DATE_ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_PERCENT = re.compile(r"(?<![\w.])(?P<num>\d{1,3}(?:\.\d+)?)\s*(?:%|per\s+cent\b|percent\b)")
_KNOWN_LAW = "|".join(re.escape(j.law_phrase) for j in sorted(JURISDICTIONS, key=lambda j: -len(j.law_phrase)))
_JURISDICTION = re.compile(
    r"(?:governed\s+by|construed\s+in\s+accordance\s+with)\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?"
    rf"the\s+laws\s+of\s+(?P<j>(?:{_KNOWN_LAW})|(?:the\s+)?[A-Z][A-Za-z\s]*?)(?=[.,;]|\s+without\b)".replace(
        "\\ ", "\\s+"
    )
)
_PARTY_INTRO = re.compile(
    r"(?:\bbetween\s+|\band\s+|\(\d\)\s+)(?P<name>[A-Z][\w.&'\s]{2,80}?),\s+(?:an?\s+[a-z]+\b|having\b)"
)
_PARTY_SIGN = re.compile(r"for\s+and\s+on\s+behalf\s+of\s+(?P<name>[A-Z][^\n]+)")
_JUR_BY_PHRASE = {j.law_phrase.lower(): j.name for j in JURISDICTIONS}
_ROLE_QUOTE = re.compile(r'"(?:the\s+)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)"')
_ROLE_STOP = frozenset({"Agreement", "Effective Date", "Purpose", "Lease", "Premises", "Services", "Service"})


def _unit_text(n: int, unit: str) -> str:
    base = unit.lower().rstrip("s") + "s"
    return base[:-1] if n == 1 else base


def _add(found: dict[tuple[int, int], Entity], etype: str, value: str, start: int, end: int) -> None:
    for s, e in list(found):
        if s < end and start < e:
            if (e - s) >= (end - start):
                return
            del found[(s, e)]
    found[(start, end)] = Entity(type=etype, value=value, start=start, end=end)  # type: ignore[arg-type]


def _durations(text: str, found: dict[tuple[int, int], Entity]) -> None:
    for m in _DUR_WORDS_PARENS.finditer(text):
        n = int(m.group("digits"))
        _add(found, "duration", f"{n} {_unit_text(n, m.group('unit'))}", m.start(), m.end())
    for m in _DUR_DIGITS.finditer(text):
        n = int(m.group("digits"))
        _add(found, "duration", f"{n} {_unit_text(n, m.group('unit'))}", m.start(), m.end())
    for m in _DUR_WORDS.finditer(text):
        n = words_to_int(m.group("words").split())
        if n is not None:
            _add(found, "duration", f"{n} {_unit_text(n, m.group('unit'))}", m.start(), m.end())


def _money(text: str, found: dict[tuple[int, int], Entity]) -> None:
    table = {"Rs.": "INR", "US$": "USD", "$": "USD", "S$": "SGD"}
    for m in _MONEY.finditer(text):
        cur = table.get(m.group("cur"), m.group("cur"))
        amount = int(float(m.group("amt").replace(",", "")))
        _add(found, "money", f"{cur} {amount}", m.start(), m.end())


def _dates(text: str, found: dict[tuple[int, int], Entity]) -> None:
    for pattern in (_DATE_DMY, _DATE_MDY, _DATE_ISO):
        for m in pattern.finditer(text):
            month = m.group("m")
            month_n = MONTHS.index(month) + 1 if month in MONTHS else int(month)
            try:
                iso = date(int(m.group("y")), month_n, int(m.group("d"))).isoformat()
            except ValueError:
                continue
            _add(found, "date", iso, m.start(), m.end())


def _percentages(text: str, found: dict[tuple[int, int], Entity]) -> None:
    for m in _PERCENT.finditer(text):
        _add(found, "percentage", f"{m.group('num')}%", m.start(), m.end())


def normalize_jurisdiction(phrase: str) -> str:
    """Map "the State of New York" to "New York", "New South Wales, Australia" to "New South Wales", etc."""
    key = normalize_ws(phrase).lower()
    if key in _JUR_BY_PHRASE:
        return _JUR_BY_PHRASE[key]
    cleaned = re.sub(
        r"^the\s+(state|federal\s+republic|republic|commonwealth)\s+of\s+", "", normalize_ws(phrase), flags=re.I
    )
    cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.I)
    return cleaned.split(",")[0].strip()


def _jurisdictions(text: str, found: dict[tuple[int, int], Entity]) -> None:
    for m in _JURISDICTION.finditer(text):
        _add(found, "jurisdiction", normalize_jurisdiction(m.group("j")), m.start("j"), m.end("j"))


def _parties(text: str, found: dict[tuple[int, int], Entity]) -> None:
    for pattern in (_PARTY_INTRO, _PARTY_SIGN):
        for m in pattern.finditer(text):
            name = normalize_ws(m.group("name"))
            if 2 <= len(name.split()) <= 8:
                _add(found, "party", name, m.start("name"), m.start("name") + len(m.group("name").rstrip()))


def extract_entities(text: str, clauses: list[Clause] | None = None) -> list[Entity]:
    """Extract all entities from ``text`` and attribute each to the clause whose span contains it."""
    found: dict[tuple[int, int], Entity] = {}
    _durations(text, found)
    _money(text, found)
    _dates(text, found)
    _percentages(text, found)
    _jurisdictions(text, found)
    _parties(text, found)
    out = sorted(found.values(), key=lambda e: (e.start, e.end))
    if clauses:
        for e in out:
            e.clause_id = next((c.clause_id for c in clauses if c.start <= e.start < c.end), None)
    return out


def extract_roles(text: str) -> tuple[str, str]:
    """The two defined party roles (for example ("Supplier", "Customer")) from the preamble.

    Falls back to ("Party A", "Party B") when the document does not define roles in quotes.
    """
    head = text[:2500]
    roles: list[str] = []
    for m in _ROLE_QUOTE.finditer(head):
        role = normalize_ws(m.group(1))
        if role not in _ROLE_STOP and role not in roles:
            roles.append(role)
        if len(roles) == 2:
            break
    if len(roles) < 2:
        roles = ([*roles, "Party A", "Party B"])[:2]
    return roles[0], roles[1]
