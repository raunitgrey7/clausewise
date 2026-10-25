import dataclasses

import pytest
from fastapi.testclient import TestClient

from clausewise.api.app import create_app


@pytest.fixture(scope="module")
def client(settings, analyzer):
    return TestClient(create_app(settings, analyzer=analyzer))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["embeddings"] and body["auth_enabled"] is False
    assert "X-Request-ID" in r.headers


def test_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_metrics_endpoint(client):
    client.get("/health")
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "clausewise_http_requests_total" in r.text


def test_analyze_and_ask(client, contract):
    r = client.post("/api/analyze", json={"text": contract.text})
    assert r.status_code == 200
    body = r.json()
    assert body["contract_type"] == contract.contract_type and body["clauses"]
    before = client.get("/metrics").text
    assert "clausewise_analyses_total" in before
    jurisdiction = next(e.value for e in contract.entities if e.type == "jurisdiction")
    r = client.post(
        "/api/ask", json={"text": contract.text, "question": "Which law governs this agreement?", "mode": "bm25"}
    )
    assert r.status_code == 200
    ans = r.json()
    assert ans["abstained"] is False and ans["sentences"][0]["grounded"] is True
    assert ans["short_answer"] == jurisdiction


def test_ask_abstains_on_nonsense(client, contract):
    r = client.post("/api/ask", json={"text": contract.text, "question": "What colour is the moon on Tuesdays?"})
    assert r.status_code == 200 and r.json()["abstained"] is True


def test_validation_errors(client):
    assert client.post("/api/analyze", json={"text": "short"}).status_code == 422
    assert client.post("/api/ask", json={"text": "x" * 30, "question": "a"}).status_code == 422
    assert (
        client.post("/api/ask", json={"text": "x" * 30, "question": "valid question", "mode": "nope"}).status_code
        == 422
    )
    assert (
        client.post("/api/analyze", content=b"not json", headers={"content-type": "application/json"}).status_code
        == 422
    )


def test_payload_too_large(settings, analyzer):
    small = dataclasses.replace(settings, max_text_chars=100)
    c = TestClient(create_app(small, analyzer=analyzer))
    assert c.post("/api/analyze", json={"text": "y" * 101}).status_code == 413


def test_samples(client):
    r = client.get("/api/samples")
    assert r.status_code == 200 and len(r.json()) > 0
    first = r.json()[0]["contract_id"]
    assert client.get(f"/api/samples/{first}").json()["text"]
    assert client.get("/api/samples/nope-0000").status_code == 404


def test_dashboard_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "clausewise" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_api_key_protection(settings, analyzer, contract):
    secured = dataclasses.replace(settings, api_key="secret")
    c = TestClient(create_app(secured, analyzer=analyzer))
    assert c.get("/health").status_code == 200
    assert c.get("/api/samples").status_code == 401
    assert c.post("/api/analyze", json={"text": contract.text}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.post("/api/analyze", json={"text": contract.text}, headers={"X-API-Key": "secret"}).status_code == 200


def test_garbage_input_returns_result_not_500(client):
    r = client.post("/api/analyze", json={"text": "�\u0000 garbage " * 20})
    assert r.status_code == 200 and r.json()["risks"] == []
