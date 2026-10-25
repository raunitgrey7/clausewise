"""Prometheus metrics for the API.

Counters and histograms are module level so that one registry serves the whole process; the middleware
records request counts and latencies, the routes record domain events (analyses, findings, answers).
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

REQUESTS = Counter("clausewise_http_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram(
    "clausewise_http_request_duration_seconds", "Request latency", ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)  # fmt: skip
ANALYSES = Counter("clausewise_analyses_total", "Contracts analysed")
FINDINGS = Counter("clausewise_risk_findings_total", "Playbook findings", ["risk_type"])
ANSWERS = Counter("clausewise_qa_answers_total", "Questions answered", ["outcome"])
REJECTIONS = Counter("clausewise_grounding_rejections_total", "Answer sentences rejected by the grounding checker")


def render() -> tuple[bytes, str]:
    """Prometheus text exposition and its content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
