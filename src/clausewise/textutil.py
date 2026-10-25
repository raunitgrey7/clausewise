"""Shared text primitives: tokenisation, light stemming, sentence splitting and number words.

Every stage of the pipeline (segmentation, classification features, BM25, the grounding checker and the
extractive answerer) must agree on what a token and a sentence are, otherwise a sentence that the answerer
copies verbatim could fail the grounding check on a tokenisation mismatch. Keeping the primitives here and
nowhere else is what makes the extractive path provably self-consistent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_WS_RE = re.compile(r"\s+")

STOPWORDS: frozenset[str] = frozenset(
    (
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "as",
        "is",
        "are",
        "be",
        "been",
        "being",
        "was",
        "were",
        "shall",
        "will",
        "may",
        "must",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "their",
        "there",
        "which",
        "who",
        "whom",
        "whose",
        "any",
        "all",
        "each",
        "either",
        "neither",
        "both",
        "such",
        "than",
        "then",
        "into",
        "under",
        "over",
        "upon",
        "within",
        "without",
        "between",
        "before",
        "after",
        "during",
        "about",
        "against",
        "so",
        "if",
        "not",
        "no",
        "nor",
        "other",
        "party",
        "parties",
        "agreement",
        "hereof",
        "hereunder",
        "herein",
        "hereby",
        "thereof",
        "whether",
        "per",
        "what",
        "how",
        "when",
        "where",
        "does",
        "do",
        "did",
        "can",
        "could",
        "should",
        "would",
        "have",
        "has",
        "had",
        "get",
        "gets",
    )
)

# Abbreviations that end with a full stop but do not end a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {"pvt", "ltd", "inc", "no", "rs", "mr", "mrs", "ms", "dr", "st", "e.g", "i.e", "vs", "pte", "co", "corp", "llc"}
)

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
NUMBER_WORDS: dict[str, int] = {**_UNITS, **_TENS}


def normalize_ws(text: str) -> str:
    """Collapse runs of whitespace (including newlines inside wrapped values) into single spaces."""
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens. Decimal numbers such as 99.9 stay as one token."""
    return _TOKEN_RE.findall(text.lower())


def stem(token: str) -> str:
    """A deliberately conservative suffix stripper.

    It only normalises plurals and the common verb endings, so "terminates", "terminated" and
    "terminating" meet at "terminat" while short tokens are never touched. Anything smarter (Porter,
    Snowball) would need another dependency and would make the grounding checker harder to explain.
    """
    if len(token) <= 4 or token[0].isdigit():
        return token
    root = token
    for suffix in ("ation", "ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            root = token[: -len(suffix)]
            root = root + "y" if suffix == "ies" else root
            break
    if len(root) > 5 and root.endswith("e"):
        root = root[:-1]  # "terminate" and "terminated" both become "terminat"
    return root


def content_tokens(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed, in order of appearance."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def _is_sentence_end(text: str, idx: int) -> bool:
    """Return True if the full stop at ``idx`` terminates a sentence."""
    before = text[:idx]
    word = re.search(r"([A-Za-z.]+)$", before)
    if word and word.group(1).lower().rstrip(".") in _ABBREVIATIONS:
        return False
    if word and len(word.group(1)) == 1 and word.group(1).isupper():
        return False  # initials such as "J. Smith"
    if re.search(r"\(\w\)$", before):
        return False  # list markers such as "(a)." are rare; keep the list in one sentence
    after = text[idx + 1 :]
    if re.match(r"\d", after):
        return False  # decimal number or sub-clause number such as 7.1
    return bool(re.match(r"\s*$", after) or re.match(r"\s+[\"(A-Z0-9]", after) or re.match(r"\s*\n", after))


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentences in ``text``.

    Spans exclude leading and trailing whitespace. The splitter is punctuation driven: a full stop,
    question mark, semicolon-free exclamation or a blank line ends a sentence unless the stop belongs to a
    known abbreviation, an initial or a number.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch == ".":
            boundary = _is_sentence_end(text, i)
        elif ch in "?!":
            boundary = True
        else:
            boundary = ch == "\n" and text[i : i + 2] == "\n\n"
        if boundary:
            seg = text[start : i + 1]
            s_off = len(seg) - len(seg.lstrip())
            e_off = len(seg) - len(seg.rstrip())
            if seg.strip():
                spans.append((start + s_off, i + 1 - e_off))
            start = i + 1
        i += 1
    tail = text[start:]
    if tail.strip():
        s_off = len(tail) - len(tail.lstrip())
        spans.append((start + s_off, start + len(tail.rstrip())))
    return spans


def sentences(text: str) -> list[str]:
    """Sentence strings, convenience wrapper around :func:`split_sentences`."""
    return [text[a:b] for a, b in split_sentences(text)]


def words_to_int(words: Iterable[str]) -> int | None:
    """Parse "thirty", "twenty four" or "one hundred twenty" into an int; None when not a number phrase."""
    total = 0
    seen = False
    for w in words:
        w = w.lower().replace("-", " ")
        for part in w.split():
            if part in NUMBER_WORDS:
                total += NUMBER_WORDS[part]
                seen = True
            elif part == "hundred" and seen:
                total *= 100
            elif part in {"and"}:
                continue
            else:
                return None
    return total if seen else None


def int_to_words(value: int) -> str:
    """Render 0..999 in words, legal drafting style ("twenty four", "one hundred twenty")."""
    if value < 0 or value > 999:
        raise ValueError("int_to_words supports 0..999")
    units = {v: k for k, v in _UNITS.items()}
    tens = {v: k for k, v in _TENS.items()}
    parts: list[str] = []
    if value >= 100:
        parts.append(f"{units[value // 100]} hundred")
        value %= 100
        if value == 0:
            return parts[0]
    if value < 20:
        parts.append(units[value])
    else:
        parts.append(tens[value - value % 10] + (f" {units[value % 10]}" if value % 10 else ""))
    return " ".join(parts)
