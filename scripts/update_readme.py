"""Copy benchmarks/results.md into README.md between the results markers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- results:start -->"
END = "<!-- results:end -->"


def main() -> int:
    """Rewrite the README block."""
    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    table = (ROOT / "benchmarks" / "results.md").read_text(encoding="utf-8").strip()
    head, rest = readme.split(START, 1)
    _old, tail = rest.split(END, 1)
    readme_path.write_text(f"{head}{START}\n{table}\n{END}{tail}", encoding="utf-8")
    print("README.md updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
