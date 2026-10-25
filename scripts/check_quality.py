"""Repository quality gates that ruff does not cover.

1. No function longer than 60 lines (the charter's limit), measured with the ast module.
2. No em dash or en dash anywhere in tracked text files.
3. No placeholder markers (the usual two upper-case words) in shipped source.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FUNCTION_LINES = 60
DASHES = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
MARKERS = re.compile(r"\b(" + "TO" + "DO|FIX" + "ME)\b")
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".html",
    ".js",
    ".css",
    ".json",
    ".txt",
    "Makefile",
    "Dockerfile",
}
SKIP_DIRS = {".venv", ".git", "data", ".pytest_cache", ".ruff_cache", ".hypothesis", "node_modules", "__pycache__"}


def iter_files() -> list[Path]:
    """Tracked-looking files under the repository root."""
    out = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in TEXT_SUFFIXES):
            out.append(path)
    return out


def long_functions(path: Path) -> list[str]:
    """Functions whose body spans more than MAX_FUNCTION_LINES lines."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > MAX_FUNCTION_LINES:
                problems.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name} is {length} lines")
    return problems


def main() -> int:
    """Run all checks and print every violation."""
    problems: list[str] = []
    for path in iter_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), start=1):
            if DASHES.search(line):
                problems.append(f"{path.relative_to(ROOT)}:{i} contains an em or en dash")
        if path.suffix == ".py":
            problems.extend(long_functions(path))
            if "tests" not in path.parts and path.name != "check_quality.py" and MARKERS.search(text):
                problems.append(f"{path.relative_to(ROOT)} contains a placeholder marker")
    for p in problems:
        print(p)
    print(f"check_quality: {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
