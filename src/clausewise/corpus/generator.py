"""Seeded contract generator with complete ground truth.

A contract is assembled in four steps: draw a plan (type, roles, style, which optional clauses appear),
draw a context of values (parties, dates, amounts, durations, jurisdiction), pick a variant for each
risk-bearing clause, then render. Rendering goes through one builder that records the character span of
every placeholder value, so the entity ground truth is exact by construction rather than re-derived by
regex afterwards. Line wrapping replaces spaces with newlines and never changes a character offset.

Each contract is generated from ``random.Random(seed * 1_000_003 + index)`` so that any single contract
can be regenerated without regenerating the ones before it.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from clausewise.corpus import names
from clausewise.corpus.clause_templates import ClauseTemplate, templates_for
from clausewise.corpus.plans import (
    AGREEMENT_NOUN,
    BOILERPLATE,
    CONTRACT_TYPE_WEIGHTS,
    IP_ROLES,
    PLANS,
    RECITALS,
    ROLES,
    TITLES,
    VARIANT_MIX,
)
from clausewise.corpus.questions import QUESTION_BANK, QuestionSpec
from clausewise.corpus.values import Value, amount_for, duration, money, percentage, pick_currency, plain, random_date
from clausewise.playbook import DEFAULT_PLAYBOOK
from clausewise.schemas import Clause, Contract, Entity, Party, PlantedRisk, Question

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]+(?::[a-z_]+)?)\}")
_NUMBERING_STYLES: tuple[tuple[str, float], ...] = (
    ("arabic", 0.35),
    ("section", 0.20),
    ("article", 0.15),
    ("clause", 0.10),
    ("none", 0.20),
)


@dataclass
class _Builder:
    """Accumulates contract text and the spans of typed values."""

    parts: list[str] = field(default_factory=list)
    length: int = 0
    entities: list[tuple[str, str, int, int]] = field(default_factory=list)

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.length += len(text)

    def text(self) -> str:
        return "".join(self.parts)


@dataclass
class _Style:
    numbering: str
    inline: bool
    upper: bool
    subnumber: bool
    wrap: int
    page_markers: bool
    numstyle: str
    datefmt: str
    moneystyle: str
    blank_after_heading: bool
    heading_dropout: float
    typo_rate: float


def _roman(n: int) -> str:
    table = ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
    out = ""
    for value, sym in table:
        while n >= value:
            out += sym
            n -= value
    return out


def _weighted(rng: random.Random, options: tuple[tuple[str, float], ...]) -> str:
    return rng.choices([o for o, _ in options], weights=[w for _, w in options])[0]


def _draw_style(rng: random.Random) -> _Style:
    numbering = _weighted(rng, _NUMBERING_STYLES)
    noisy = rng.random() < 0.5
    numbered = numbering in {"arabic", "section", "clause"}
    return _Style(
        numbering=numbering,
        inline=numbered and rng.random() < 0.35,
        upper=numbering in {"article", "none"} or rng.random() < 0.3,
        subnumber=rng.random() < 0.25,
        wrap=rng.choice((0, 0, 72, 80, 100)),
        page_markers=rng.random() < 0.3,
        numstyle=_weighted(rng, (("words_parens", 0.5), ("digits", 0.35), ("words", 0.15))),
        datefmt=rng.choice(("dmy", "mdy", "legal", "iso")),
        moneystyle=rng.choice(("indian", "indian", "western", "code")),
        blank_after_heading=rng.random() < 0.7,
        heading_dropout=0.15 if noisy else 0.0,
        typo_rate=0.01 if noisy else 0.0,
    )


def _company_name(rng: random.Random, country: str) -> str:
    suffix = rng.choice(names.COMPANY_SUFFIXES[country])
    return f"{rng.choice(names.COMPANY_PREFIXES)} {rng.choice(names.COMPANY_SECTORS)} {suffix}"


def _person_name(rng: random.Random) -> str:
    return f"{rng.choice(names.FIRST_NAMES)} {rng.choice(names.LAST_NAMES)}"


def _address(rng: random.Random, country: str | None = None) -> str:
    cities = names.CITIES_BY_COUNTRY.get(country or "", names.CITIES)
    return f"{rng.randint(1, 450)} {rng.choice(names.STREETS)}, {rng.choice(cities)}"


def _party_description(rng: random.Random, individual: bool, country: str) -> str:
    if individual:
        return rng.choice(
            (
                f"an individual residing at {_address(rng, country)}",
                f"having their residential address at {_address(rng, country)}",
            )
        )
    return rng.choice(
        (
            f"a company incorporated under the laws of {country} with its registered office at "
            f"{_address(rng, country)}",
            f"a company registered in {country} whose principal place of business is at {_address(rng, country)}",
        )
    )


def _draw_jurisdiction(rng: random.Random) -> names.Jurisdiction:
    return rng.choices(names.JURISDICTIONS, weights=[j.weight for j in names.JURISDICTIONS])[0]


def _build_context(rng: random.Random, ctype: str, roles: tuple[str, str], style: _Style) -> dict[str, Value]:
    """Draw every placeholder value for one contract."""
    a_country = _weighted(rng, names.INCORPORATION_WEIGHTS)
    b_country = _weighted(rng, names.INCORPORATION_WEIGHTS)
    b_individual = ctype == "employment" or (ctype == "lease" and rng.random() < 0.5)
    a_name = _company_name(rng, a_country)
    b_name = _person_name(rng) if b_individual else _company_name(rng, b_country)
    currency = pick_currency(rng, b_country if ctype in {"msa", "saas"} else a_country)
    jur = _draw_jurisdiction(rng)
    arb = rng.choice(names.ARBITRATION)
    unilateral_is_a = rng.random() < 0.5
    ctx: dict[str, Value] = {
        "A": plain(roles[0]),
        "B": plain(roles[1]),
        "U": plain(roles[0] if unilateral_is_a else roles[1]),
        "V": plain(roles[1] if unilateral_is_a else roles[0]),
        "A_name": Value(a_name, "party", a_name),
        "B_name": Value(b_name, "party", b_name),
        "A_desc": plain(_party_description(rng, False, a_country)),
        "B_desc": plain(_party_description(rng, b_individual, b_country)),
        "effective_date": random_date(rng, style.datefmt),
        "jurisdiction": Value(jur.law_phrase, "jurisdiction", jur.name),
        "law": plain("the laws of {jurisdiction}"),
        "courts": plain(jur.courts),
        "arb_institution": plain(arb[0]),
        "seat": plain(arb[1]),
        "city": plain(rng.choice(names.CITIES)),
        "territory": plain(rng.choice(names.TERRITORIES)),
        "data_region": plain(rng.choice(names.DATA_REGIONS)),
        "job_title": plain(rng.choice(names.JOB_TITLES)),
        "manager_title": plain(rng.choice(names.MANAGER_TITLES)),
        "permitted_use": plain(rng.choice(names.PERMITTED_USES)),
        "service_area": plain(rng.choice(names.SERVICE_AREAS)),
        "premises_address": plain(f"Unit {rng.randint(1, 40)}, {_address(rng)}"),
        "payment_frequency": plain(rng.choice(("monthly", "quarterly", "annually"))),
        "rent_day": plain(rng.choice(("first", "fifth", "seventh", "tenth"))),
    }
    ip_roles = IP_ROLES.get(ctype, ("A", "B"))
    ctx["ASSIGNOR"] = ctx[ip_roles[0]]
    ctx["ASSIGNEE"] = ctx[ip_roles[1]]
    _add_durations(rng, ctx, style.numstyle)
    _add_amounts(rng, ctx, currency, ctype, style.moneystyle)
    for key, choices in (
        ("sla", ("99.5", "99.9", "99.95")),
        ("credit_pct", ("5", "10")),
        ("interest_rate", ("8", "12", "18")),
        ("escalation", ("5", "7", "10")),
        ("bonus_pct", ("10", "15", "20")),
    ):
        ctx[key] = percentage(rng, rng.choice(choices))
    return ctx


def _add_durations(rng: random.Random, ctx: dict[str, Value], numstyle: str) -> None:
    days = {
        "notice": (15, 30, 45, 60, 90),
        "cure": (14, 30),
        "nonrenew_notice": (30, 60, 90),
        "payment_days": (15, 30, 45, 60),
        "claim_days": (30,),
        "fm_days": (30, 60, 90),
        "dispute_days": (30, 45),
        "return_days": (10, 14, 30),
        "grace_days": (5, 7, 10),
        "deposit_refund_days": (30, 60),
        "leave_days": (18, 21, 24),
        "post_days": (2, 3),
    }
    for key, options in days.items():
        ctx[key] = duration(rng.choice(options), "days", numstyle)
    for key, options in {"noncompete": (6, 12, 18, 24), "nonsolicit": (12, 18), "cap_months": (12,)}.items():
        ctx[key] = duration(rng.choice(options), "months", numstyle)
    for key, options in {"conf_duration": (2, 3, 5), "insurance_years": (2, 3)}.items():
        ctx[key] = duration(rng.choice(options), "years", numstyle)
    for key, options in {
        "breach_hours": (24, 48, 72),
        "response_hours": (1, 2, 4),
        "maintenance_notice": (48, 72),
        "inspection_notice": (24, 48),
    }.items():
        ctx[key] = duration(rng.choice(options), "hours", numstyle)
    term_months = rng.choice((12, 24, 36, 60))
    if term_months % 12 == 0 and rng.random() < 0.6:
        ctx["term"] = duration(term_months // 12, "years", numstyle)
    else:
        ctx["term"] = duration(term_months, "months", numstyle)
    renewal_months = rng.choice((12, 24))
    if rng.random() < 0.5:
        ctx["renewal"] = duration(renewal_months // 12, "years", numstyle)
    else:
        ctx["renewal"] = duration(renewal_months, "months", numstyle)


def _add_amounts(rng: random.Random, ctx: dict[str, Value], currency: str, ctype: str, moneystyle: str) -> None:
    for key in ("fee", "rent", "salary", "insurance"):
        ctx[key] = money(rng, currency, amount_for(rng, currency, key), moneystyle)
    deposit = amount_for(rng, currency, "rent") * rng.choice((2, 3, 6))
    ctx["deposit"] = money(rng, currency, deposit, moneystyle)
    if ctype in {"msa", "saas"} and rng.random() < 0.6:
        months = ctx["cap_months"].normalized or ""
        ctx["cap"] = Value(
            "the total fees paid or payable by {B} under this Agreement in the {cap_months} immediately "
            "preceding the event giving rise to the claim",
            None,
            f"{months} fees",
        )
    else:
        ctx["cap"] = money(rng, currency, amount_for(rng, currency, "cap"), moneystyle)


def _render(template: str, ctx: dict[str, Value], refs: dict[str, str], b: _Builder, used: set[str]) -> None:
    """Append ``template`` to the builder, substituting placeholders and recording entity spans."""
    pos = 0
    for m in _PLACEHOLDER_RE.finditer(template):
        b.add(template[pos : m.start()])
        key = m.group(1)
        if key.startswith("ref:"):
            b.add(refs.get(key[4:], "the applicable provisions of this Agreement"))
        else:
            value = ctx[key]
            used.add(key)
            start = b.length
            _render(value.text, ctx, refs, b, used)
            if value.entity_type:
                b.entities.append((value.entity_type, value.normalized or value.text, start, b.length))
        pos = m.end()
    b.add(template[pos:])


def _choose_plan(rng: random.Random, ctype: str) -> list[str]:
    labels = [label for label, p in PLANS[ctype] if rng.random() < p]
    if rng.random() < 0.5:
        tail = [lab for lab in labels if lab in BOILERPLATE]
        rng.shuffle(tail)
        it = iter(tail)
        labels = [next(it) if lab in BOILERPLATE else lab for lab in labels]
    return labels


def _choose_templates(rng: random.Random, ctype: str, labels: list[str]) -> list[ClauseTemplate]:
    chosen: list[ClauseTemplate] = []
    for label in labels:
        options = templates_for(label, ctype)
        if label in VARIANT_MIX:
            variant = _weighted(rng, VARIANT_MIX[label])
            options = [t for t in options if t.variant == variant]
        chosen.append(rng.choice(options))
    return chosen


def _heading_line(style: _Style, k: int, heading: str) -> str:
    text = heading.upper() if style.upper else heading
    if style.numbering == "arabic":
        return f"{k}. {text}"
    if style.numbering == "section":
        return f"Section {k}. {text}"
    if style.numbering == "clause":
        return f"Clause {k}: {text}"
    if style.numbering == "article":
        return f"ARTICLE {_roman(k)}. {text}"
    return text


def _ref_text(style: _Style, k: int, heading: str) -> str:
    if style.numbering == "arabic":
        return f"clause {k}"
    if style.numbering == "section":
        return f"Section {k}"
    if style.numbering == "clause":
        return f"Clause {k}"
    if style.numbering == "article":
        return f"Article {_roman(k)}"
    return f"the {heading} clause"


def _clause_units(rng: random.Random, t: ClauseTemplate) -> list[str]:
    units = [rng.choice(slot) for slot in t.slots]
    units.extend(rng.choice(slot) for slot in t.optional if rng.random() < 0.5)
    return units


def _wrap_text(text: str, width: int) -> str:
    """Hard-wrap each line at ``width`` by turning spaces into newlines; offsets are preserved."""
    if width <= 0:
        return text
    out_lines: list[str] = []
    for line in text.split("\n"):
        chars = list(line)
        last_break = -1
        col = 0
        for i, ch in enumerate(chars):
            col += 1
            if ch == " ":
                last_break = i
            if col > width and last_break >= 0:
                chars[last_break] = "\n"
                col = i - last_break
                last_break = -1
        out_lines.append("".join(chars))
    return "\n".join(out_lines)


@dataclass
class _Draft:
    """Mutable state while one contract is being written."""

    b: _Builder = field(default_factory=_Builder)
    clauses: list[Clause] = field(default_factory=list)
    used: dict[str, set[str]] = field(default_factory=dict)
    variants: dict[str, str] = field(default_factory=dict)
    risks: list[PlantedRisk] = field(default_factory=list)

    def begin(self) -> int:
        return self.b.length

    def entities_sorted(self) -> list[tuple[str, str, int, int]]:
        return sorted(self.b.entities, key=lambda e: (e[2], e[3]))

    def finish(self, label: str, heading: str | None, start: int) -> Clause:
        cid = f"c{len(self.clauses) + 1:02d}"
        clause = Clause(
            clause_id=cid, start=start, end=self.b.length, heading=heading, text=self.b.text()[start:], label=label
        )
        self.clauses.append(clause)
        return clause


def _write_preamble(rng: random.Random, d: _Draft, ctype: str, ctx: dict[str, Value], refs: dict[str, str]) -> None:
    title = rng.choice(TITLES[ctype])
    noun = AGREEMENT_NOUN[ctype]
    used: set[str] = set()
    start = d.begin()
    variant = rng.randrange(3)
    if variant == 0:
        d.b.add(f"{title}\n\n")
        _render(
            f'THIS {noun.upper()} (the "Agreement") is made on {{effective_date}} between {{A_name}}, {{A_desc}} '
            f'("{{A}}"), and {{B_name}}, {{B_desc}} ("{{B}}").',
            ctx,
            refs,
            d.b,
            used,
        )
    elif variant == 1:
        d.b.add(f"{title}\n\n")
        _render(
            f'This {noun} (the "Agreement") is entered into as of {{effective_date}} (the "Effective Date")\n\n'
            f'BETWEEN:\n\n(1) {{A_name}}, {{A_desc}} (the "{{A}}"); and\n\n(2) {{B_name}}, {{B_desc}} (the "{{B}}").',
            ctx,
            refs,
            d.b,
            used,
        )
    else:
        d.b.add(f"{title}\n\n")
        _render(
            "This Agreement is dated {effective_date} and is made between {A_name}, {A_desc}, hereinafter referred "
            'to as the "{A}", and {B_name}, {B_desc}, hereinafter referred to as the "{B}". {A} and {B} are each a '
            '"party" and together the "parties".',
            ctx,
            refs,
            d.b,
            used,
        )
    clause = d.finish("preamble", None, start)
    d.used[clause.clause_id] = used


def _write_recitals(rng: random.Random, d: _Draft, ctype: str, ctx: dict[str, Value], refs: dict[str, str]) -> None:
    d.b.add("\n\n")
    used: set[str] = set()
    start = d.begin()
    heading = None
    if rng.random() < 0.5:
        heading = rng.choice(("RECITALS", "BACKGROUND", "WHEREAS"))
        d.b.add(f"{heading}\n\n")
    for line in rng.choice(RECITALS[ctype]):
        _render(line, ctx, refs, d.b, used)
        d.b.add("\n\n")
    d.b.add(
        rng.choice(
            (
                "NOW, THEREFORE, in consideration of the mutual covenants set out below, the parties agree as follows:",
                "NOW IT IS HEREBY AGREED as follows:",
                "The parties therefore agree as follows:",
            )
        )
    )
    clause = d.finish("recitals", heading, start)
    d.used[clause.clause_id] = used


def _write_clause(
    rng: random.Random,
    d: _Draft,
    style: _Style,
    k: int,
    t: ClauseTemplate,
    heading: str,
    ctx: dict[str, Value],
    refs: dict[str, str],
) -> None:
    used: set[str] = set()
    start = d.begin()
    units = _clause_units(rng, t)
    line = _heading_line(style, k, heading)
    dropped = not style.inline and rng.random() < style.heading_dropout
    if dropped:
        for i, unit in enumerate(units):
            _render(unit, ctx, refs, d.b, used)
            if i < len(units) - 1:
                d.b.add(" ")
    elif style.inline:
        d.b.add(f"{line}. ")
        for i, unit in enumerate(units):
            _render(unit, ctx, refs, d.b, used)
            if i < len(units) - 1:
                d.b.add(" ")
    else:
        d.b.add(line + ("\n\n" if style.blank_after_heading else "\n"))
        for i, unit in enumerate(units):
            if style.subnumber and style.numbering in {"arabic", "section", "clause"}:
                d.b.add(f"{k}.{i + 1} ")
            elif style.subnumber:
                d.b.add(f"({chr(97 + i)}) ")
            _render(unit, ctx, refs, d.b, used)
            if i < len(units) - 1:
                d.b.add("\n" if style.subnumber else " ")
    clause = d.finish(t.label, None if dropped else heading, start)
    d.used[clause.clause_id] = used
    d.variants[t.label] = t.variant
    if t.risk:
        d.risks.append(PlantedRisk(risk_type=t.risk, clause_id=clause.clause_id))  # type: ignore[arg-type]


def _write_signature(rng: random.Random, d: _Draft, ctx: dict[str, Value], refs: dict[str, str]) -> None:
    d.b.add("\n\n")
    used: set[str] = set()
    start = d.begin()
    d.b.add(
        rng.choice(
            (
                "IN WITNESS WHEREOF the parties have executed this Agreement on the date first written above.",
                "IN WITNESS WHEREOF, the parties have caused this Agreement to be signed by their duly authorised "
                "representatives.",
            )
        )
    )
    for role in ("A", "B"):
        d.b.add("\n\n")
        _render(f"Signed for and on behalf of {{{role}_name}}", ctx, refs, d.b, used)
        d.b.add(f"\nName: {_person_name(rng)}\nTitle: {rng.choice(('Director', 'Authorised Signatory', 'Partner'))}")
    clause = d.finish("signature", None, start)
    d.used[clause.clause_id] = used


def _make_questions(rng: random.Random, d: _Draft, ctype: str, ctx: dict[str, Value]) -> list[Question]:
    by_label: dict[str, Clause] = {}
    for c in d.clauses:
        by_label.setdefault(c.label or "", c)
    candidates: list[tuple[QuestionSpec, Clause]] = []
    absent: list[QuestionSpec] = []
    for spec in QUESTION_BANK:
        if spec.contract_types and ctype not in spec.contract_types:
            continue
        clause = by_label.get(spec.label)
        if clause is None:
            absent.append(spec)
            continue
        if spec.variants and d.variants.get(spec.label) not in spec.variants:
            continue
        if spec.value_key and spec.value_key not in d.used[clause.clause_id]:
            continue
        candidates.append((spec, clause))
    rng.shuffle(candidates)
    questions = [_question_from(rng, spec, clause, d, ctx) for spec, clause in candidates[:8]]
    if absent:
        spec = rng.choice(absent)
        questions.append(
            Question(
                question=rng.choice(spec.paraphrases),
                gold_clause_id=None,
                gold_label=spec.label,
                answer_type=spec.answer_type,
                gold_value=None,
                answerable=False,
            )
        )
    return questions


def _question_from(
    rng: random.Random, spec: QuestionSpec, clause: Clause, d: _Draft, ctx: dict[str, Value]
) -> Question:
    value: str | None = None
    if spec.answer_type == "boolean":
        value = "yes" if d.variants.get(spec.label) in (spec.true_variants or ()) else "no"
    elif spec.answer_type == "parties":
        value = " | ".join(sorted([ctx["A_name"].text, ctx["B_name"].text]))
    elif spec.value_key:
        value = ctx[spec.value_key].normalized
    return Question(
        question=rng.choice(spec.paraphrases),
        gold_clause_id=clause.clause_id,
        gold_label=spec.label,
        answer_type=spec.answer_type,
        gold_value=value,
        answerable=True,
    )


def _page_marker(d: _Draft, style: _Style, k: int, total: int, every: int) -> None:
    if style.page_markers and k % every == 0:
        d.b.add(f"\n\nPage {k // every} of {math.ceil(total / every) + 1}")


def generate_contract(seed: int, index: int) -> Contract:
    """Generate contract number ``index`` of the corpus for ``seed``."""
    rng = random.Random(seed * 1_000_003 + index)
    ctype = _weighted(rng, CONTRACT_TYPE_WEIGHTS)
    roles = rng.choice(ROLES[ctype])
    style = _draw_style(rng)
    ctx = _build_context(rng, ctype, roles, style)
    labels = _choose_plan(rng, ctype)
    templates = _choose_templates(rng, ctype, labels)
    headings = [rng.choice(t.headings) for t in templates]
    refs = {t.label: _ref_text(style, k, h) for k, (t, h) in enumerate(zip(templates, headings, strict=True), start=1)}
    d = _Draft()
    _write_preamble(rng, d, ctype, ctx, refs)
    if rng.random() < 0.85:
        _write_recitals(rng, d, ctype, ctx, refs)
    every = rng.choice((4, 5, 6))
    for k, (t, heading) in enumerate(zip(templates, headings, strict=True), start=1):
        d.b.add("\n\n")
        _write_clause(rng, d, style, k, t, heading, ctx, refs)
        _page_marker(d, style, k, len(templates), every)
    _write_signature(rng, d, ctx, refs)
    text = _wrap_text(d.b.text(), style.wrap)
    text = _inject_typos(rng, text, style.typo_rate, d)
    clauses = [c.model_copy(update={"text": text[c.start : c.end]}) for c in d.clauses]
    entities = _entities(d, text, clauses)
    return Contract(
        contract_id=f"{ctype}-{index:04d}",
        contract_type=ctype,  # type: ignore[arg-type]
        title=text.split("\n", 1)[0],
        text=text,
        clauses=clauses,
        parties=[Party(name=ctx["A_name"].text, role=roles[0]), Party(name=ctx["B_name"].text, role=roles[1])],
        entities=entities,
        risks=d.risks + _governing_law_risk(ctx, clauses),
        questions=_make_questions(rng, d, ctype, ctx),
        style={**style.__dict__, "roles": f"{roles[0]}/{roles[1]}", "variants": json.dumps(d.variants, sort_keys=True)},
    )


_TYPO_WORD = re.compile(r"\b[a-z]{6,}\b")


def _inject_typos(rng: random.Random, text: str, rate: float, d: _Draft) -> str:
    """Swap two adjacent interior letters in a fraction of long lowercase words.

    Entity spans are left untouched so the gold values stay exact; swaps preserve length so no offset
    moves. This mimics OCR and typing noise in scanned or retyped contracts.
    """
    if rate <= 0:
        return text
    protected = [(s, e) for _t, _v, s, e in d.b.entities]
    chars = list(text)
    for m in _TYPO_WORD.finditer(text):
        if rng.random() >= rate or any(s <= m.start() < e for s, e in protected):
            continue
        i = rng.randint(m.start() + 1, m.end() - 3)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _entities(d: _Draft, text: str, clauses: list[Clause]) -> list[Entity]:
    out: list[Entity] = []
    for etype, value, start, end in d.entities_sorted():
        cid = next((c.clause_id for c in clauses if c.start <= start < c.end), None)
        out.append(Entity(type=etype, value=value, start=start, end=end, clause_id=cid))  # type: ignore[arg-type]
    return out


def _governing_law_risk(ctx: dict[str, Value], clauses: list[Clause]) -> list[PlantedRisk]:
    clause = next((c for c in clauses if c.label == "governing_law"), None)
    if clause is None or ctx["jurisdiction"].normalized in DEFAULT_PLAYBOOK.acceptable_jurisdictions:
        return []
    return [PlantedRisk(risk_type="governing_law", clause_id=clause.clause_id)]


def generate_corpus(seed: int, size: int) -> list[Contract]:
    """Generate ``size`` contracts deterministically."""
    return [generate_contract(seed, i) for i in range(size)]


def corpus_path(data_dir: Path, seed: int, size: int) -> Path:
    """Cache location for a corpus."""
    return data_dir / f"corpus_seed{seed}_n{size}.jsonl.gz"


def load_or_generate(data_dir: Path, seed: int, size: int) -> list[Contract]:
    """Load the cached corpus for (seed, size) or generate and cache it."""
    path = corpus_path(data_dir, seed, size)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return [Contract.model_validate_json(line) for line in fh if line.strip()]
    contracts = generate_corpus(seed, size)
    data_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for c in contracts:
            fh.write(c.model_dump_json() + "\n")
    return contracts
