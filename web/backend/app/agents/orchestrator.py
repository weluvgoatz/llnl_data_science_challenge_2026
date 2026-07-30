"""The top-level orchestrator: interprets a chat message, delegates to
detection_agent / report_agent / plot_agent, and mounts/unmounts a single UI
surface. It has no domain tools of its own besides delegation and
mount_surface/unmount_surface -- it never computes a pipeline number itself.
"""

from __future__ import annotations

import os
import time
from typing import Any

import openai

from . import subagents, surfaces
from .runtime import run_tool_loop
from .. import chat_store
from ..store import load_job

DEFAULT_MODEL = "gpt-5.4"  # matches the model already used by .codex/agents/strut_error_detection_agent.toml

ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator for Lattice Lens, an X-ray CT inspection tool for 3D-printed lattice structures. There are no fixed tabs or a persistent viewer -- the main surface starts empty (just the uploaded file list) and you are the only thing that puts anything else there. You do not compute or explain anything yourself -- you have exactly three specialist subagents and two surface-control tools, and your job is judgment about who to call and what to show:

- detection_agent: owns the numerical pipeline for the current job -- starting the initial analysis, checking its status, (re)running defect classification with different thresholds, generating the per-defect validation image gallery, and exporting 3D models. Call it for anything that changes/(re)produces pipeline output, or to check real progress.
- report_agent: reads already-computed result metadata (per-strut evidence, defect hotspots by unit cell, measured-vs-nominal thickness) and explains findings in plain language, citing real numbers. It also generates the full Lattice NDE report (Markdown + PDF, three sections: input metadata & statistics, output statistics & plots, and a situational analysis of what the defect pattern implies about the print process) -- delegate to it when the user asks for "the report" / "the NDE report" / to generate/make/write a report, once analysis has completed. Relay the Markdown path and PDF artifact it returns, and mount ReportView if the user's request also implies wanting to see it (not just generate it).
- plot_agent: analyzes result data and renders the most suitable chart as a PNG.
- mount_surface(component, ...): put one of four surfaces in the main area -- ModelViewer (an uploaded STL, file_id), DefectView (the classified 3D lattice, optional version_id/filter_verdicts/select_strut_ids), ReportView (the NDE report, no args), or DataViz (a TIFF slice / design-JSON graph via file_id+slice_index, or a generated plot/gallery image via artifact_id). For a TIFF specifically: showing it plainly (e.g. "show me the TIFF") means file_id only -- do NOT set show_tilt_pane. Only set show_tilt_pane=true when the user explicitly asks about tilt/alignment correction (e.g. "fix the tilt", "is this scan tilted") -- that opens a side pane with the correction process/result next to the still-untouched original; it never replaces the main view.
- unmount_surface(): clear the main area back to the file list.

Rules:
1. Never state a number yourself that a subagent didn't return. If you don't know something, delegate -- don't guess.
2. Delegate with a clear, self-contained natural-language request; a subagent has no memory of this conversation, only what you tell it in `request`.
3. You may call multiple subagents in one turn (e.g. report_agent to explain a hotspot, then plot_agent to chart it).
4. Mount something whenever the answer is better shown than told -- the user asking "show me X" or "render Y" is a direct instruction to call mount_surface, not just describe X in text. After a subagent produces an artifact, mount it by `artifact_id` (found in the subagent's tool-call output -- never a filesystem path).
5. mount_surface validates preconditions itself and returns a clear error if something isn't ready (analysis still running, no design JSON uploaded, etc.) -- relay that error plainly rather than guessing why it failed or silently retrying.
6. This job's TIFF/STL/JSON files can be shown (ModelViewer, DataViz) at any time, even before analysis has run -- only DefectView needs a completed classification. If the user asks for something that isn't ready yet, say so and offer what IS available right now (e.g. "the model view works now; defect queries need analysis to run first").
7. run_initial_analysis (via detection_agent) starts the pipeline in the background and returns immediately -- it does not mean analysis is done. Don't claim results are ready until a status check or a later message confirms it.
8. Starting or checking analysis (run_initial_analysis, get_job_status) is NEVER by itself a reason to call mount_surface or unmount_surface. Whatever the user is currently looking at must stay exactly as it is through the whole analyzing -> analyzed transition -- report progress in your reply text only, the same way rule 4's "mount when it's better shown" does not apply here. Only mount/unmount when the user's message is itself a request to look at, hide, or replace something.
9. Keep your final reply conversational and concise; the full evidence trail is stored separately for audit, so you don't need to repeat every number a subagent returned -- just the ones that actually answer the question.
10. Stay in scope: this assistant is for additive-manufacturing lattice inspection -- the AM/CT workflow, this job's files and results, defect analysis, and closely related engineering/ML background (e.g. how CT scanning or tilt correction works, why a strut might be classified as thin, what a file format here is for). Judge by substance, not keywords -- something on-topic in spirit should never be declined just because it doesn't match an exact phrase. For a request that's clearly outside that domain (general knowledge, unrelated how-tos, casual chit-chat, help with something else entirely), don't answer it -- politely decline and redirect to what you can actually help with, e.g. "I'm focused on additive-manufacturing lattice inspection, so I can't help with that. But I can explain the AM/CT workflow, show your model, run the analysis, or answer questions about detected defects." """

ORCHESTRATOR_TOOLS: list[dict[str, Any]] = [
    {
        "name": "detection_agent",
        "description": (
            "Delegate to the detection agent: it owns starting/checking analysis, defect classification "
            "(including re-running it with different thresholds), generating the per-defect validation image "
            "gallery, and exporting 3D models. Give it a clear, self-contained instruction in plain language."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "What you need the detection agent to do."}},
            "required": ["request"],
        },
    },
    {
        "name": "report_agent",
        "description": (
            "Delegate to the report agent: it reads the classification result's metadata (per-strut evidence, "
            "hotspot groupings, thickness comparisons) and explains/narrates findings, and it generates the full "
            "Lattice NDE report (Markdown + PDF) when asked. Give it a clear, self-contained question or request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "What you need the report agent to explain."}},
            "required": ["request"],
        },
    },
    {
        "name": "plot_agent",
        "description": (
            "Delegate to the plot agent: it analyzes the result data and produces the most suitable chart "
            "(verdict counts, defect hotspots, thickness distribution, or version-comparison) as a PNG. Give it "
            "a clear, self-contained charting request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "What you need plotted."}},
            "required": ["request"],
        },
    },
    {
        "name": "mount_surface",
        "description": "Put one surface in the main area, replacing whatever (if anything) is currently mounted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {"type": "string", "enum": sorted(surfaces.VALID_COMPONENTS)},
                "file_id": {"type": "string", "description": "An uploaded file: ModelViewer needs an STL, DataViz needs a TIFF or design JSON."},
                "artifact_id": {"type": "string", "description": "A generated artifact: ModelViewer for a generated .stl model part, DataViz for a generated plot/gallery image (never a .ply -- no in-browser viewer for those, mention the download instead)."},
                "version_id": {"type": "integer", "description": "DefectView: which classification version (omit for the active one)."},
                "filter_verdicts": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["present", "missing", "bent", "thin", "disconnected"]},
                    "description": "DefectView: which verdict categories to show.",
                },
                "select_strut_ids": {"type": "array", "items": {"type": "integer"}, "description": "DefectView: struts to highlight."},
                "slice_index": {"type": "integer", "description": "DataViz on a TIFF: which slice."},
                "show_tilt_pane": {
                    "type": "boolean",
                    "description": (
                        "DataViz on a TIFF only. Leave unset/false for a plain 'show me the TIFF' request -- "
                        "that renders just the raw slice. Set true only when the user explicitly asks about "
                        "tilt correction; opens a side pane with the correction progress/result next to the "
                        "still-unmodified original, it does not replace the main view."
                    ),
                },
            },
            "required": ["component"],
        },
    },
    {
        "name": "unmount_surface",
        "description": "Clear the main area back to the file list.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _client() -> openai.OpenAI:
    return openai.OpenAI()  # reads OPENAI_API_KEY from the environment


def _describe_files(job: dict[str, Any]) -> str:
    """A short, real inventory of this job's uploaded files -- without this,
    the orchestrator has no way to know a TIFF/STL/JSON exists at all, so a
    generic reference ("this scan", "the model") would force it to ask the
    user for a file_id it could otherwise resolve on its own (mount_surface
    already auto-resolves when exactly one file of the requested kind
    exists)."""
    if not job["files"]:
        return "No files uploaded yet."
    return "\n".join(
        f"- {f['name']} (kind={f['kind']}, id={f['id']}, {f.get('summary', '')})" for f in job["files"]
    )


def handle_chat_turn(job_id: str, user_message: str, model: str | None = None) -> dict[str, Any]:
    """Run one chat turn end to end: load conversation state, let the
    orchestrator delegate to subagents and mount/unmount a surface as
    needed, persist the updated conversation and an audit record, and
    return the turn's result.
    """
    client = _client()
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    job = load_job(job_id)
    system = (
        f"{ORCHESTRATOR_SYSTEM_PROMPT}\n\n"
        f"Current job files:\n{_describe_files(job)}\n\n"
        "When the user refers to a file generically (\"this scan\", \"the TIFF\", \"the model\") and exactly "
        "one file of that kind is listed above, use it directly -- mount_surface and detection_agent resolve "
        "file_id on their own when there's only one candidate, so don't ask the user to repeat what's already "
        "listed here."
    )

    messages = chat_store.load_conversation(job_id)
    messages.append({"role": "user", "content": user_message})

    mount_holder: dict[str, Any] = {"mounted": "__unset__"}
    subagent_traces: dict[str, list[dict[str, Any]]] = {}

    def _dispatch_subagent(name: str, request: str) -> dict[str, Any]:
        result = subagents.run_subagent(name, job_id, request, client, model)
        subagent_traces.setdefault(name, []).append({"request": request, **result})
        # What the orchestrator sees as the "tool result" is the subagent's
        # own final answer plus its tool log, so the orchestrator can cite
        # specifics (ids, numbers) without re-deriving them.
        return {"reply": result["final_text"], "tool_calls": result["tool_calls"]}

    def _mount_surface(**kwargs: Any) -> dict[str, Any]:
        result = surfaces.mount_surface(job_id, **kwargs)
        mount_holder["mounted"] = result["mounted"]
        return result

    def _unmount_surface() -> dict[str, Any]:
        result = surfaces.unmount_surface(job_id)
        mount_holder["mounted"] = result["mounted"]
        return result

    dispatch = {
        "detection_agent": lambda request: _dispatch_subagent("detection_agent", request),
        "report_agent": lambda request: _dispatch_subagent("report_agent", request),
        "plot_agent": lambda request: _dispatch_subagent("plot_agent", request),
        "mount_surface": _mount_surface,
        "unmount_surface": _unmount_surface,
    }

    result = run_tool_loop(
        client=client,
        model=model,
        system=system,
        tools=ORCHESTRATOR_TOOLS,
        dispatch=dispatch,
        messages=messages,
    )

    chat_store.save_conversation(job_id, messages)

    turn_record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "user_message": user_message,
        "reply": result["final_text"],
        "orchestrator_tool_calls": result["tool_calls"],
        "subagent_traces": subagent_traces,
    }
    # Only include "mount" at all if mount_surface/unmount_surface was
    # actually called this turn -- key ABSENT means "leave whatever's
    # currently mounted alone", which is distinct from key PRESENT with
    # value null (explicit unmount_surface -> clear back to the file list).
    # Collapsing these to the same value would make a plain Q&A turn
    # (no mount call) indistinguishable from "hide what you're looking at".
    if mount_holder["mounted"] != "__unset__":
        turn_record["mount"] = mount_holder["mounted"]
    chat_store.append_audit(job_id, turn_record)
    return turn_record
