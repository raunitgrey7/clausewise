"""Shared fixtures: an isolated data directory, a small corpus and a trained analyzer.

Tests never touch the repository's data/ directory and never need the network: the embedding backend is
forced to TF-IDF unless a test opts into the dense model explicitly (and skips when it is not cached).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("CLAUSEWISE_EMBEDDINGS", "tfidf")
os.environ["CLAUSEWISE_CORPUS_SIZE"] = "80"
os.environ["CLAUSEWISE_SEED"] = "42"
os.environ.pop("APP_API_KEY", None)
os.environ.pop("CLAUSEWISE_LLM", None)

from clausewise.config import Settings, load_settings
from clausewise.corpus.generator import generate_corpus
from clausewise.pipeline import Analyzer
from clausewise.schemas import Contract

SMALL_CORPUS_SIZE = 80


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data")
    os.environ["CLAUSEWISE_DATA_DIR"] = str(path)
    return path


@pytest.fixture(scope="session")
def settings(data_dir: Path) -> Settings:
    return load_settings()


@pytest.fixture(scope="session")
def corpus() -> list[Contract]:
    return generate_corpus(42, SMALL_CORPUS_SIZE)


@pytest.fixture(scope="session")
def analyzer(settings: Settings) -> Analyzer:
    return Analyzer(settings)


@pytest.fixture(scope="session")
def contract(corpus: list[Contract]) -> Contract:
    """An MSA with planted risks, used by several modules."""
    return next(c for c in corpus if c.contract_type == "msa" and len(c.risks) >= 2)


def contracts_with_risk(corpus: list[Contract], risk_type: str) -> list[Contract]:
    """Helper for tests that need a contract with a specific planted risk."""
    return [c for c in corpus if any(r.risk_type == risk_type for r in c.risks)]
