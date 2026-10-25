"""Fail when the results table in README.md differs from benchmarks/results.md.

The benchmark writes results.md; README.md embeds the same table between two marker comments. This check
is what makes charter rule 3 ("every number in the README was measured") enforceable by CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def main() -> int:
    """Compare the embedded table with the generated one."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    generated = (ROOT / "benchmarks" / "results.md").read_text(encoding="utf-8").strip()
    if START not in readme or END not in readme:
        print("README.md is missing the results markers")
        return 1
    embedded = readme.split(START, 1)[1].split(END, 1)[0].strip()
    if embedded != generated:
        print("README.md results table does not match benchmarks/results.md; run scripts/update_readme.py")
        return 1
    print("README results table matches benchmarks/results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
