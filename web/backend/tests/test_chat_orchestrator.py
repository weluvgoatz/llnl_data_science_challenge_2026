"""Verifies the chat orchestrator's wiring -- tool dispatch, job_id binding,
nested subagent delegation, canvas-directive capture, and persistence --
without a live OpenAI API key. A scripted fake client stands in for the
model's *decisions* (which tool to call, in what order), but every tool it
"decides" to call is the real Python function running against a real,
already-classified job directory, so this exercises the actual pipeline
data path end to end, not just message plumbing.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fakes standing in for the OpenAI SDK's chat.completions response shape.
# ---------------------------------------------------------------------------


@dataclass
class FakeFunctionCall:
    name: str
    arguments: str  # JSON-encoded, matching the real SDK


@dataclass
class FakeToolCall:
    function: FakeFunctionCall
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeCompletionsAPI:
    def __init__(self, script: dict[str, list[FakeResponse]]):
        # script keys are substrings matched against the system prompt (the
        # first message, which run_tool_loop always prepends), so the fake
        # can tell an orchestrator call apart from a subagent call sharing
        # the same fake client.
        self._script = {key: list(value) for key, value in script.items()}
        self.calls: list[dict[str, Any]] = []

    def create(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> FakeResponse:
        system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        self.calls.append({"system": system, "messages": messages, **kwargs})
        for key, queue in self._script.items():
            if key in system:
                if not queue:
                    raise AssertionError(f"Fake client ran out of scripted responses for system containing {key!r}")
                return queue.pop(0)
        raise AssertionError(f"No scripted response matches system prompt: {system[:80]!r}")


class FakeChatAPI:
    def __init__(self, script: dict[str, list[FakeResponse]]):
        self.completions = FakeCompletionsAPI(script)


class FakeOpenAIClient:
    def __init__(self, script: dict[str, list[FakeResponse]]):
        self.chat = FakeChatAPI(script)


def tool_use(name: str, **input_kwargs: Any) -> FakeResponse:
    call = FakeToolCall(function=FakeFunctionCall(name=name, arguments=json.dumps(input_kwargs)))
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=None, tool_calls=[call]), finish_reason="tool_calls")])


def final_text(text: str) -> FakeResponse:
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=text, tool_calls=None), finish_reason="stop")])


# ---------------------------------------------------------------------------
# A real, minimal job directory with an actual classification JSON.
# ---------------------------------------------------------------------------


@pytest.fixture
def real_job(tmp_path, monkeypatch):
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    from app import store

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    root = store.job_dir(job["id"])

    design = {
        "junctions": [{"id": 0, "position": [0, 0, 0], "indices": [0, 0, 0]},
                       {"id": 1, "position": [10, 0, 0], "indices": [1, 0, 0]}],
        "struts": [{"id": 0, "junction0": 0, "junction1": 1, "thickness": 0.1}],
        "unit_cells": [{"id": 0, "struts": [0], "indices": [0, 0, 0]}],
    }
    (root / "uploads").mkdir(parents=True, exist_ok=True)
    design_path = root / "uploads" / "design.json"
    design_path.write_text(json.dumps(design), encoding="utf-8")

    classification = {
        "struts": [
            {
                "id": 0,
                "p0": [0.0, 0.0, 0.0],
                "p1": [10.0, 0.0, 0.0],
                "verdict": "present",
                "design_thickness": 0.1,
                "evidence": {
                    "reason": "as_built_edge_matched",
                    "mean_density": 45000.0,
                    "density_cutoff": 38000.0,
                    "bow_um": 10.0,
                    "bow_threshold_um": 212.0,
                    "measured_radius_um": 180.0,
                    "nominal_radius_um": 212.0,
                },
            }
        ],
        "meta": {"counts": {"present": 1}, "n": 1, "volume_shape_zyx": [10, 10, 10]},
    }
    (root / "defects").mkdir(parents=True, exist_ok=True)
    classification_path = root / "defects" / "classification.json"
    classification_path.write_text(json.dumps(classification), encoding="utf-8")

    job["files"] = [{"id": "designfile", "name": "design.json", "size": design_path.stat().st_size,
                      "path": "uploads/design.json", "kind": "json", "summary": "test"}]
    job["defects"] = {
        "status": "complete",
        "path": "defects/classification.json",
        "versions": [{"id": 1, "label": "initial", "path": "defects/classification.json",
                       "params": {}, "counts": {"present": 1}, "n": 1, "createdAt": store.now()}],
        "activeVersionId": 1,
    }
    store.save_job(job)
    return job["id"]


# ---------------------------------------------------------------------------
# The actual wiring test.
# ---------------------------------------------------------------------------


def test_chat_turn_delegates_and_captures_canvas(real_job, monkeypatch):
    from app import chat_store
    from app.agents import orchestrator, subagents

    job_id = real_job

    script = {
        subagents.REPORT_SYSTEM_PROMPT[:40]: [
            tool_use("explain_strut", strut_id=0),
            final_text("Strut 0 is present: mean density 45000 is above the cutoff 38000, and bow 10um is well under the 212um bend threshold."),
        ],
        orchestrator.ORCHESTRATOR_SYSTEM_PROMPT[:40]: [
            tool_use("report_agent", request="Why was strut 0 classified as present?"),
            tool_use("mount_surface", component="DefectView", select_strut_ids=[0], filter_verdicts=["present"]),
            final_text("Strut 0 is present because its measured density (45,000) is above the specimen's cutoff (38,000) and its bow (10 µm) is far under the 212 µm bend threshold."),
        ],
    }
    fake_client = FakeOpenAIClient(script)
    monkeypatch.setattr(orchestrator, "_client", lambda: fake_client)

    turn = orchestrator.handle_chat_turn(job_id, "Why was strut 0 classified as present?", model="fake-model")

    # 1. The real tool actually ran against the real job data (not a stub).
    report_trace = turn["subagent_traces"]["report_agent"][0]["tool_calls"]
    assert report_trace[0]["tool"] == "explain_strut"
    assert report_trace[0]["input"] == {"strut_id": 0}
    assert report_trace[0]["output"]["strut"]["verdict"] == "present"
    assert report_trace[0]["output"]["strut"]["evidence"]["mean_density"] == 45000.0
    assert report_trace[0]["is_error"] is False

    # 2. The orchestrator's own tool-call log recorded the delegation + mount call.
    tool_names = [c["tool"] for c in turn["orchestrator_tool_calls"]]
    assert tool_names == ["report_agent", "mount_surface"]

    # 3. mount_surface validated real job state (defects.status == "complete")
    # and the resulting directive was captured verbatim.
    assert turn["mount"] == {
        "component": "DefectView",
        "props": {"select_strut_ids": [0], "filter_verdicts": ["present"]},
    }

    # 4. The final reply is the orchestrator's last text block.
    assert "45,000" in turn["reply"]

    # 5. Everything persisted: conversation resumable, audit log has the full trace.
    conversation = chat_store.load_conversation(job_id)
    assert conversation[0] == {"role": "user", "content": "Why was strut 0 classified as present?"}
    assert any(m["role"] == "assistant" for m in conversation)

    audit = chat_store.load_audit(job_id)
    assert len(audit) == 1
    assert audit[0]["reply"] == turn["reply"]
    assert audit[0]["subagent_traces"]["report_agent"][0]["tool_calls"][0]["tool"] == "explain_strut"


def test_unknown_tool_name_is_reported_not_crashed(real_job, monkeypatch):
    """If the model calls a tool name that isn't in the dispatch table, the
    loop must feed back a well-formed error instead of raising."""
    from app.agents import orchestrator

    job_id = real_job
    script = {
        orchestrator.ORCHESTRATOR_SYSTEM_PROMPT[:40]: [
            tool_use("nonexistent_tool", foo="bar"),
            final_text("Sorry, something went wrong internally."),
        ],
    }
    fake_client = FakeOpenAIClient(script)
    monkeypatch.setattr(orchestrator, "_client", lambda: fake_client)

    turn = orchestrator.handle_chat_turn(job_id, "do something", model="fake-model")
    assert turn["orchestrator_tool_calls"][0]["is_error"] is True
    assert "Unknown tool" in turn["orchestrator_tool_calls"][0]["output"]["error"]


def test_tool_error_from_real_function_is_surfaced(real_job, monkeypatch):
    """A real ToolError (e.g. asking about a strut id that doesn't exist)
    must reach the model as an error, not crash the turn."""
    from app.agents import orchestrator, subagents

    job_id = real_job
    script = {
        subagents.REPORT_SYSTEM_PROMPT[:40]: [
            tool_use("explain_strut", strut_id=999),
            final_text("That strut id doesn't exist in this classification."),
        ],
        orchestrator.ORCHESTRATOR_SYSTEM_PROMPT[:40]: [
            tool_use("report_agent", request="explain strut 999"),
            final_text("There's no strut 999 in this job."),
        ],
    }
    fake_client = FakeOpenAIClient(script)
    monkeypatch.setattr(orchestrator, "_client", lambda: fake_client)

    turn = orchestrator.handle_chat_turn(job_id, "why is strut 999 missing?", model="fake-model")
    report_trace = turn["subagent_traces"]["report_agent"][0]["tool_calls"]
    assert report_trace[0]["is_error"] is True
    assert "No strut with id 999" in report_trace[0]["output"]["error"]

    # A plain Q&A turn with no mount_surface/unmount_surface call must NOT
    # include a "mount" key at all -- that's what tells the frontend "leave
    # whatever's on screen alone", distinct from an explicit unmount.
    assert "mount" not in turn


def test_unmount_surface_clears_with_explicit_null(real_job, monkeypatch):
    from app.agents import orchestrator

    job_id = real_job
    script = {
        orchestrator.ORCHESTRATOR_SYSTEM_PROMPT[:40]: [
            tool_use("unmount_surface"),
            final_text("Cleared the view."),
        ],
    }
    fake_client = FakeOpenAIClient(script)
    monkeypatch.setattr(orchestrator, "_client", lambda: fake_client)

    turn = orchestrator.handle_chat_turn(job_id, "hide that", model="fake-model")
    assert "mount" in turn  # key present...
    assert turn["mount"] is None  # ...with an explicit null, not just absent


def test_mount_surface_rejects_defect_view_before_analysis(monkeypatch, tmp_path):
    """mount_surface must validate real job state itself, not just trust the
    model -- asking for DefectView before analysis has run should come back
    as a clear ToolError, not a crash or a fabricated view."""
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    from app import store

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    job["files"] = [{"id": "f1", "name": "scan.tif", "size": 1, "path": "uploads/scan.tif", "kind": "tiff", "summary": "test"}]
    store.save_job(job)

    from app.agents import orchestrator

    job_id = job["id"]
    script = {
        orchestrator.ORCHESTRATOR_SYSTEM_PROMPT[:40]: [
            tool_use("mount_surface", component="DefectView"),
            final_text("Analysis hasn't been run yet -- want me to start it?"),
        ],
    }
    fake_client = FakeOpenAIClient(script)
    monkeypatch.setattr(orchestrator, "_client", lambda: fake_client)

    turn = orchestrator.handle_chat_turn(job_id, "show me the defects", model="fake-model")
    assert turn["orchestrator_tool_calls"][0]["is_error"] is True
    assert "hasn't been run yet" in turn["orchestrator_tool_calls"][0]["output"]["error"]
    # mount_surface raised before producing a directive -- nothing should mount.
    assert "mount" not in turn
