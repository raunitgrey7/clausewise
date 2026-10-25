"""Clause segmentation: turn raw contract text into clause spans.

The segmenter is line based. Each line is classified as a heading (numbered, "Section n.", "Clause n:",
"ARTICLE n." or a bare upper-case title), a recital opener, the signature block opener, a page marker or
ordinary text; a new clause starts at every heading-like line. Working on lines rather than on sentence
boundaries is what lets inline headings ("3. Term. This Agreement...") and sub-numbered paragraphs
("3.1", "(a)") be handled with the same rules, and it keeps offsets exact because nothing is rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clausewise.schemas import Clause

_NUMBERED = re.compile(r"^(?P<num>\d{1,2})\.\s+(?P<rest>\S.*)$")
_SECTION = re.compile(r"^Section\s+(?P<num>\d{1,2})\.\s+(?P<rest>\S.*)$")
_CLAUSE = re.compile(r"^Clause\s+(?P<num>\d{1,2}):\s+(?P<rest>\S.*)$")
_ARTICLE = re.compile(r"^ARTICLE\s+(?P<num>[IVXLC]+)\.\s+(?P<rest>\S.*)$")
_UPPER_TITLE = re.compile(r"^[A-Z][A-Z0-9 ,&'()/-]{1,70}$")
_PAGE_MARKER = re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$")
_RECITAL = re.compile(r"^(WHEREAS|RECITALS|BACKGROUND)\b")
_SIGNATURE = re.compile(r"^IN WITNESS WHEREOF\b")
_HEAD_WORD = re.compile(r"^(?:[A-Z][A-Za-z'-]*|and|or|of|the|this|to|for|on|in|by|with|at|a|an|&|[A-Z0-9]+)$")
_MAX_HEADING_WORDS = 8


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


def _lines(text: str) -> list[_Line]:
    out: list[_Line] = []
    pos = 0
    for raw in text.split("\n"):
        out.append(_Line(pos, pos + len(raw), raw))
        pos += len(raw) + 1
    return out


def _is_title_case(words: list[str]) -> bool:
    return bool(words) and len(words) <= _MAX_HEADING_WORDS and all(_HEAD_WORD.match(w) for w in words)


def split_inline_heading(rest: str) -> tuple[str, str | None]:
    """Split "Term. This Agreement..." into ("Term", "This Agreement...").

    Returns ``(heading, None)`` when the whole text is a heading, and ``("", rest)`` when no heading can
    be isolated (the line is body text that happens to start with a number).
    """
    stripped = rest.rstrip()
    words = stripped.split()
    if _is_title_case(words):
        return stripped.rstrip("."), None
    m = re.match(r"^(?P<head>[^.]{1,80}?)\.\s+(?P<body>\S.*)$", stripped, re.S)
    if m and _is_title_case(m.group("head").split()):
        return m.group("head"), m.group("body")
    return "", rest


def classify_line(text: str) -> tuple[str, str | None]:
    """Classify one line; returns (kind, heading). Kinds: heading, recital, signature, page, blank, text."""
    s = text.strip()
    if not s:
        return "blank", None
    if _PAGE_MARKER.match(s):
        return "page", None
    if _SIGNATURE.match(s):
        return "signature", None
    for pattern in (_NUMBERED, _SECTION, _CLAUSE, _ARTICLE):
        m = pattern.match(s)
        if m:
            heading, _body = split_inline_heading(m.group("rest"))
            if heading:
                return "heading", heading
            return "text", None
    if _RECITAL.match(s):
        return "recital", None
    if _UPPER_TITLE.match(s) and len(s.split()) <= _MAX_HEADING_WORDS and not s.endswith(":"):
        return "heading", s
    return "text", None


def _trim_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def segment(text: str) -> list[Clause]:
    """Split ``text`` into clauses. Returns an empty list for blank input."""
    if not text.strip():
        return []
    lines = _lines(text)
    boundaries: list[tuple[int, str | None]] = []
    first_content = next(i for i, ln in enumerate(lines) if ln.text.strip())
    boundaries.append((lines[first_content].start, None))
    seen_recital = False
    seen_heading = False
    prev_kind = "blank"
    body_lines_since_boundary = 0
    for i, ln in enumerate(lines):
        kind, heading = classify_line(ln.text)
        if i == first_content and kind == "heading":
            prev_kind = kind
            continue  # the document title is part of the preamble
        if kind == "heading" and heading and heading.upper() in {"RECITALS", "BACKGROUND", "WHEREAS"}:
            kind = "recital"
        if kind == "recital" and (seen_recital or i == first_content):
            kind = "text"
        if kind == "recital":
            seen_recital = True
            boundaries.append((ln.start, None))
            body_lines_since_boundary = 0
        elif kind in {"heading", "signature"}:
            seen_heading = seen_heading or kind == "heading"
            boundaries.append((ln.start, heading))
            body_lines_since_boundary = 0
        elif kind == "text":
            if _starts_unheaded_clause(ln.text, prev_kind, seen_heading, body_lines_since_boundary):
                boundaries.append((ln.start, None))
                body_lines_since_boundary = 0
            body_lines_since_boundary += 1
        if kind != "page":
            prev_kind = kind
    return _build(text, lines, boundaries)


_PARAGRAPH_START = re.compile(r"^[A-Z][a-z]")
_CONTINUATION = re.compile(r"^(?:\(\w\)|\d{1,2}\.\d|WHEREAS|NOW\b|BETWEEN\b|Name:|Title:|Signed\b)")


def _starts_unheaded_clause(line: str, prev_kind: str, seen_heading: bool, body_lines: int) -> bool:
    """A paragraph after a blank line, past the first heading, that follows body text and is not a list item.

    This is the fallback for clauses whose heading was lost (scanning, copy-paste). It only fires once a
    numbered clause has already started, so the preamble and recitals cannot be split by it.
    """
    s = line.strip()
    if not seen_heading or prev_kind != "blank" or body_lines == 0:
        return False
    return bool(_PARAGRAPH_START.match(s)) and not _CONTINUATION.match(s)


def _build(text: str, lines: list[_Line], boundaries: list[tuple[int, str | None]]) -> list[Clause]:
    page_spans = [(ln.start, ln.end) for ln in lines if classify_line(ln.text)[0] == "page"]
    clauses: list[Clause] = []
    for idx, (start, heading) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
        for p_start, _p_end in page_spans:
            if start < p_start < end:
                end = min(end, p_start)
        end = _trim_end(text, start, end)
        if end <= start:
            continue
        clauses.append(
            Clause(clause_id=f"c{len(clauses) + 1:02d}", start=start, end=end, heading=heading, text=text[start:end])
        )
    return clauses


def body_text(clause: Clause) -> str:
    """Clause text with the heading line removed, used by the classifier's body features."""
    if not clause.heading:
        return clause.text
    first_nl = clause.text.find("\n")
    first_line = clause.text if first_nl < 0 else clause.text[:first_nl]
    _heading, inline_body = split_inline_heading(
        re.sub(r"^(\d{1,2}\.|Section\s+\d+\.|Clause\s+\d+:|ARTICLE\s+[IVXLC]+\.)\s+", "", first_line)
    )
    if inline_body is not None:
        return (inline_body + (clause.text[first_nl:] if first_nl >= 0 else "")).strip()
    return clause.text[first_nl + 1 :].strip() if first_nl >= 0 else ""
