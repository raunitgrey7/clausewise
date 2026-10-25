# syntax=docker/dockerfile:1
# Stage 1: resolve and install dependencies into a self-contained virtual environment.
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra embeddings
# Pre-download the embedding model so the container runs fully offline. If the build has no network the
# step is skipped and the service falls back to TF-IDF (reported by /health).
RUN HF_HOME=/app/hf uv run --no-sync python -c "from model2vec import StaticModel; StaticModel.from_pretrained('minishlab/potion-base-8M', force_download=False)" || echo "embedding model not downloaded; TF-IDF fallback"

# Stage 2: minimal runtime image, non-root user, no build tools.
FROM python:3.13-slim AS runtime
RUN groupadd --system app && useradd --system --gid app --home /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/hf /app/hf
COPY --chown=app:app src ./src
RUN mkdir -p /app/data && chown app:app /app/data
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf \
    HF_HUB_OFFLINE=1 \
    CLAUSEWISE_DATA_DIR=/app/data \
    CLAUSEWISE_EMBEDDINGS=auto
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["clausewise", "serve", "--host", "0.0.0.0", "--port", "8000"]
