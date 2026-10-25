"""Dense embedding backend with an explicit, logged fallback.

The default model is minishlab/potion-base-8M (model2vec static embeddings, 30 MB, CPU only, 256 dims).
It is loaded from the local Hugging Face cache first; a download is attempted only in ``auto`` mode and
only once per process. When no model is available the retriever falls back to TF-IDF cosine similarity
and says so in the logs and in the benchmark output, so a number is never silently produced by a
different backend than the one named next to it.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from clausewise.config import Settings, load_settings

log = logging.getLogger(__name__)
_MODEL_FILES = ("model.safetensors", "tokenizer.json", "config.json")


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised vectors."""

    name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim) with unit-norm rows."""
        ...


class Model2VecEmbedder:
    """Static token embeddings averaged per text (model2vec)."""

    def __init__(self, model_name: str, allow_download: bool) -> None:
        from model2vec import StaticModel

        self.name = f"model2vec:{model_name}"
        try:
            from pathlib import Path

            from huggingface_hub import hf_hub_download

            paths = [hf_hub_download(model_name, f, local_files_only=True) for f in _MODEL_FILES]
            self._model = StaticModel.from_pretrained(Path(paths[0]).parent)
        except Exception as exc:
            if not allow_download:
                raise RuntimeError(f"embedding model {model_name} not cached and downloads disabled") from exc
            log.warning("embedding model %s not in local cache, downloading once", model_name)
            self._model = StaticModel.from_pretrained(model_name, force_download=False)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Unit-norm embeddings, one row per text."""
        vectors = np.asarray(self._model.encode(texts), dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


_CACHE: dict[str, Embedder | None] = {}


def load_embedder(settings: Settings | None = None) -> Embedder | None:
    """Return the configured embedder or None when the TF-IDF fallback must be used.

    ``CLAUSEWISE_EMBEDDINGS``: ``auto`` (default) uses the cached model or downloads it once, ``dense``
    requires the model to be cached, ``tfidf`` and ``none`` skip dense retrieval entirely.
    """
    settings = settings or load_settings()
    key = f"{settings.embeddings}:{settings.embed_model}"
    if key in _CACHE:
        return _CACHE[key]
    embedder: Embedder | None = None
    if settings.embeddings in {"auto", "dense"}:
        try:
            embedder = Model2VecEmbedder(settings.embed_model, allow_download=settings.embeddings == "auto")
        except Exception as exc:
            log.warning("dense embeddings unavailable (%s); falling back to TF-IDF", exc)
            embedder = None
    _CACHE[key] = embedder
    return embedder


def embedder_name(embedder: Embedder | None) -> str:
    """Human-readable backend name for logs, health checks and benchmark output."""
    return embedder.name if embedder is not None else "tfidf-fallback"
