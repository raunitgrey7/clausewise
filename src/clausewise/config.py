"""Runtime settings read from the environment.

Everything has a working default so that ``uv run pytest`` and ``make bench`` need no configuration. The
settings object is deliberately a frozen dataclass rather than a Pydantic settings model: it has no
validation needs beyond type coercion and it keeps the API surface small.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

DEFAULT_SEED = 42
DEFAULT_CORPUS_SIZE = 400
DEFAULT_EMBED_MODEL = "minishlab/potion-base-8M"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw.strip() else default


@dataclass(frozen=True)
class Settings:
    """Process-wide configuration, built once by :func:`load_settings`."""

    seed: int = DEFAULT_SEED
    corpus_size: int = DEFAULT_CORPUS_SIZE
    data_dir: Path = field(default_factory=lambda: REPO_ROOT / "data")
    embeddings: str = "auto"
    embed_model: str = DEFAULT_EMBED_MODEL
    grounding_threshold: float = 0.85
    max_text_chars: int = 400_000
    api_key: str | None = None
    llm_enabled: bool = False
    llm_model: str = "claude-haiku-5-5"
    log_level: str = "INFO"


def load_settings() -> Settings:
    """Read settings from ``CLAUSEWISE_*`` and ``APP_API_KEY`` environment variables."""
    data_dir = os.environ.get("CLAUSEWISE_DATA_DIR")
    api_key = os.environ.get("APP_API_KEY")
    return Settings(
        seed=_env_int("CLAUSEWISE_SEED", DEFAULT_SEED),
        corpus_size=_env_int("CLAUSEWISE_CORPUS_SIZE", DEFAULT_CORPUS_SIZE),
        data_dir=Path(data_dir) if data_dir else REPO_ROOT / "data",
        embeddings=os.environ.get("CLAUSEWISE_EMBEDDINGS", "auto").lower(),
        embed_model=os.environ.get("CLAUSEWISE_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        grounding_threshold=_env_float("CLAUSEWISE_GROUNDING_THRESHOLD", 0.85),
        max_text_chars=_env_int("CLAUSEWISE_MAX_TEXT_CHARS", 400_000),
        api_key=api_key if api_key else None,
        llm_enabled=os.environ.get("CLAUSEWISE_LLM", "0") in {"1", "true", "yes"},
        llm_model=os.environ.get("CLAUSEWISE_LLM_MODEL", "claude-haiku-5-5"),
        log_level=os.environ.get("CLAUSEWISE_LOG_LEVEL", "INFO").upper(),
    )
