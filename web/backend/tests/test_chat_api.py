"""HTTP-layer tests for /api/jobs/{id}/chat -- distinct from
test_chat_orchestrator.py, which tests the orchestrator function directly.
This exercises FastAPI's request handling: validation, the missing-API-key
guard, and a full round trip through run_in_threadpool with a mocked
orchestrator call.
"""

from __future__ import annotations

import json

import httpx
import pytest


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_chat_requires_message_and_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app import store

    store.DATA_ROOT = tmp_path
    from app.main import app

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        job = (await client.post("/api/jobs")).json()

        empty = await client.post(f"/api/jobs/{job['id']}/chat", json={"message": "  "})
        assert empty.status_code == 400

        no_key = await client.post(f"/api/jobs/{job['id']}/chat", json={"message": "hello"})
        assert no_key.status_code == 503

        missing_job = await client.post("/api/jobs/does-not-exist/chat", json={"message": "hello"})
        assert missing_job.status_code == 404


@pytest.mark.anyio
async def test_chat_round_trip_through_http(tmp_path, monkeypatch):
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    from app import store

    store.DATA_ROOT = tmp_path
    from app import main as main_module

    def fake_handle_chat_turn(job_id: str, message: str, model: str | None = None) -> dict:
        return {
            "timestamp": "2026-01-01T00:00:00Z",
            "user_message": message,
            "reply": "mocked reply",
            "orchestrator_tool_calls": [],
            "subagent_traces": {},
            "canvas": None,
        }

    monkeypatch.setattr(main_module, "handle_chat_turn", fake_handle_chat_turn)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main_module.app), base_url="http://test") as client:
        job = (await client.post("/api/jobs")).json()

        response = await client.post(f"/api/jobs/{job['id']}/chat", json={"message": "hi there"})
        assert response.status_code == 200
        body = response.json()
        assert body["reply"] == "mocked reply"
        assert body["user_message"] == "hi there"

        history = await client.get(f"/api/jobs/{job['id']}/chat")
        assert history.status_code == 200
        # fake_handle_chat_turn bypasses chat_store, so history is legitimately
        # empty here -- this just confirms the endpoint itself works.
        assert history.json() == {"turns": []}
