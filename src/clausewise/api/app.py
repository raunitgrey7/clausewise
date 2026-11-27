"""The FastAPI application.

``create_app`` builds everything from settings so tests can construct an app with a fake API key or a
pre-trained analyzer. Request ids are read from ``X-Request-ID`` or generated, echoed back in the response
and attached to every log line; ``APP_API_KEY`` turns on header authentication for the ``/api`` routes
while ``/health`` and ``/metrics`` stay open for probes and scrapers.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clausewise import __version__
from clausewise.api import metrics
from clausewise.config import Settings, load_settings
from clausewise.corpus.generator import load_or_generate
from clausewise.logging_setup import configure_logging, request_id_var
from clausewise.pipeline import Analysis, Analyzer
from clausewise.retrieval import MODES
from clausewise.schemas import AnalysisResult, Answer

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
_SAMPLE_COUNT = 24


class AnalyzeRequest(BaseModel):
    """Body of POST /api/analyze."""

    text: str = Field(min_length=20, description="Full contract text")


class AskRequest(BaseModel):
    """Body of POST /api/ask."""

    text: str = Field(min_length=20)
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field(default="hybrid", pattern="^(bm25|tfidf|dense|hybrid)$")
    llm: bool = Field(default=False, description="Use the optional LLM answerer when CLAUSEWISE_LLM=1")


class HealthResponse(BaseModel):
    """Body of GET /health."""

    status: str
    version: str
    embeddings: str
    seed: int
    auth_enabled: bool
    llm_enabled: bool


class SampleSummary(BaseModel):
    """One corpus contract in the sample list."""

    contract_id: str
    contract_type: str
    title: str


class SampleText(BaseModel):
    """A corpus contract's text."""

    contract_id: str
    contract_type: str
    text: str


class _AnalysisCache:
    """Small LRU keyed by text hash so analyze-then-ask does not segment twice."""

    def __init__(self, analyzer: Analyzer, size: int = 32) -> None:
        self.analyzer = analyzer
        self.size = size
        self._items: OrderedDict[str, Analysis] = OrderedDict()

    def get(self, text: str) -> Analysis:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]
        analysis = self.analyzer.analyze(text)
        self._items[key] = analysis
        if len(self._items) > self.size:
            self._items.popitem(last=False)
        return analysis


class _State:
    """Lazily built heavy objects shared by the route handlers."""

    def __init__(self, settings: Settings, analyzer: Analyzer | None) -> None:
        self.settings = settings
        self._analyzer = analyzer
        self._cache: _AnalysisCache | None = None
        self._samples: list[SampleText] | None = None

    @property
    def analyzer_backend(self) -> str:
        return self._analyzer.backend if self._analyzer is not None else "not loaded yet"

    @property
    def cache(self) -> _AnalysisCache:
        if self._cache is None:
            if self._analyzer is None:
                self._analyzer = Analyzer(self.settings)
            self._cache = _AnalysisCache(self._analyzer)
        return self._cache

    @property
    def samples(self) -> list[SampleText]:
        if self._samples is None:
            contracts = load_or_generate(self.settings.data_dir, self.settings.seed, self.settings.corpus_size)
            self._samples = [
                SampleText(contract_id=c.contract_id, contract_type=c.contract_type, text=c.text)
                for c in contracts[:_SAMPLE_COUNT]
            ]
        return self._samples


def _auth_dependency(settings: Settings) -> Callable[..., None]:
    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    return require_key


def _install_observability(app: FastAPI) -> None:
    """Request ids, JSON access logs and Prometheus counters for every request."""

    @app.middleware("http")
    async def observe(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled error", extra={"path": request.url.path})
            response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        duration = time.perf_counter() - start
        path = request.url.path if request.url.path.startswith(("/api", "/health", "/metrics")) else "/static"
        metrics.REQUESTS.labels(request.method, path, str(response.status_code)).inc()
        metrics.LATENCY.labels(path).observe(duration)
        response.headers["X-Request-ID"] = rid
        log.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration * 1000, 2),
            },
        )
        request_id_var.reset(token)
        return response


def _register_core_routes(app: FastAPI, state: _State) -> None:
    """Health, metrics and the dashboard; these stay open for probes and scrapers."""
    settings = state.settings

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            embeddings=state.analyzer_backend,
            seed=settings.seed,
            auth_enabled=bool(settings.api_key),
            llm_enabled=settings.llm_enabled,
        )

    @app.get("/metrics")
    def prometheus() -> Response:
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _check_size(text: str, settings: Settings) -> None:
    if len(text) > settings.max_text_chars:
        raise HTTPException(status_code=413, detail=f"text longer than {settings.max_text_chars} characters")


def _register_api_routes(app: FastAPI, state: _State) -> None:
    """The authenticated analysis and question routes."""
    auth = Depends(_auth_dependency(state.settings))

    @app.get("/api/samples", response_model=list[SampleSummary], dependencies=[auth])
    def samples() -> list[SampleSummary]:
        return [
            SampleSummary(contract_id=s.contract_id, contract_type=s.contract_type, title=s.text.split("\n", 1)[0])
            for s in state.samples
        ]

    @app.get("/api/samples/{contract_id}", response_model=SampleText, dependencies=[auth])
    def sample(contract_id: str) -> SampleText:
        match = next((s for s in state.samples if s.contract_id == contract_id), None)
        if match is None:
            raise HTTPException(status_code=404, detail="unknown sample")
        return match

    @app.post("/api/analyze", response_model=AnalysisResult, dependencies=[auth])
    def analyze(body: AnalyzeRequest) -> AnalysisResult:
        _check_size(body.text, state.settings)
        result = state.cache.get(body.text).result
        metrics.ANALYSES.inc()
        for finding in result.risks:
            metrics.FINDINGS.labels(finding.risk_type).inc()
        return result

    @app.post("/api/ask", response_model=Answer, dependencies=[auth])
    def ask(body: AskRequest) -> Answer:
        _check_size(body.text, state.settings)
        if body.mode not in MODES:
            raise HTTPException(status_code=422, detail=f"mode must be one of {MODES}")
        analysis = state.cache.get(body.text)
        answer = state.cache.analyzer.ask(analysis.clauses, body.question, body.mode, llm=body.llm)  # type: ignore[arg-type]
        metrics.ANSWERS.labels("abstained" if answer.abstained else "answered").inc()
        metrics.REJECTIONS.inc(len(answer.rejected))
        return answer


def create_app(settings: Settings | None = None, analyzer: Analyzer | None = None) -> FastAPI:
    """Build the application; heavy objects are created lazily on first use unless provided."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="clausewise", version=__version__, docs_url="/docs", redoc_url=None)
    state = _State(settings, analyzer)
    _install_observability(app)
    _register_core_routes(app, state)
    _register_api_routes(app, state)
    return app


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start uvicorn with the default settings."""
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_config=None)
