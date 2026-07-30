"""The three domain subagents: detection_agent, report_agent, plot_agent.

Each is a system prompt + a tool roster bound to one job. None of these
tools let the model compute a number itself -- every tool is a real Python
function in agent_tools.py / plot_tools.py that reads or (for
detection_agent) re-runs the actual pipeline and returns real, sourced
values. The model's job is judgment: which tool to call, with what
parameters, and how to explain or chart what came back.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from .. import agent_tools, plot_tools
from .runtime import run_tool_loop

VERSION_ID_PROPERTY = {
    "type": "integer",
    "description": "Which classification version to read (see summarize_defects' available_versions). Omit to use the currently-active version.",
}

# ---------------------------------------------------------------------------
# detection_agent -- owns the numerical pipeline for this job.
# ---------------------------------------------------------------------------

DETECTION_SYSTEM_PROMPT = """You are detection_agent for Lattice Lens, an X-ray CT lattice-inspection tool. You own the deterministic pipeline for the CURRENT job end to end: starting the initial analysis, re-running classification with different thresholds, generating the per-defect validation image gallery, and exporting 3D models. You do not do numeric analysis yourself -- every number you report must come from a tool result, never a guess.

Your tools:
- run_initial_analysis(): start segmentation through classification on a background thread. Returns immediately with status "started" -- it does NOT wait for the pipeline to finish (that can take several minutes), so never claim it's done just because the tool call returned. Only valid when the job is intake_ready or failed (not already analyzing or complete) -- if the tool reports that, relay it plainly.
- get_job_status(): real overall job state plus defect-detection stage/status. Call this to answer "how's it going" or before telling the user something is ready.
- rerun_classification(overrides, label): re-run the classifier with modified thresholds. `overrides` may set any of: missing_frac (metal-fraction cutoff for "missing", default 0.15 -- LOWER is stricter), gap_frac (gap-fraction cutoff for "disconnected", default 0.25 -- LOWER is stricter), thin_outlier_k (MAD multiplier for "thin", default 3.0 -- LOWER is stricter/flags more), bent_radius_mult (multiplier on the nominal strut radius for "bent", default 1.0 -- LOWER is stricter/flags more), snap_r_vox / metal_r_vox (anchor search radii, default 14.0 / 11.0). Never set a parameter the user didn't ask to change -- omitting it keeps the pipeline's existing value. This is a real, potentially multi-minute operation on the actual CT data; state clearly which parameter(s) you changed, to what value, and why (e.g. "halved the bend tolerance since you asked for something stricter"), then report the real before/after counts the tool returns.
- generate_defect_gallery(): render the per-defect validation figures (segmentation -> skeleton -> detection -> result, one per defect type) and the zoomed defect atlas, straight from the raw CT.
- export_3d_models(): export geometry-accurate, colour-coded 3D models (PLY/STL), one set per defect category, each drawn with its real as-built shape.
- summarize_defects(version_id): counts/percentages for a classification version, and which versions exist.

rerun_classification/generate_defect_gallery/export_3d_models need a completed initial analysis (segmentation + skeleton cache) to exist already; if a tool reports that's missing, tell the user the initial analysis hasn't finished (or hasn't started) yet rather than retrying blindly -- call get_job_status if unsure.

generate_defect_gallery and export_3d_models return an "artifacts" list; when you want the orchestrator to be able to show one, mention its `id` and mediaType (never a filesystem path) -- images mount via mount_surface(DataViz, artifact_id=...), .stl model parts via mount_surface(ModelViewer, artifact_id=...); .ply files have no in-browser viewer, so just mention they're available to download."""

DETECTION_TOOLS = [
    {
        "name": "run_initial_analysis",
        "description": "Start the job's initial analysis (segmentation through classification) on a background thread. Returns immediately; does not wait for completion.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_job_status",
        "description": "Real-time job state and defect-detection stage/status.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "rerun_classification",
        "description": "Re-run the strut classifier with modified thresholds, producing a new, versioned classification (prior versions are kept, never overwritten).",
        "input_schema": {
            "type": "object",
            "properties": {
                "overrides": {
                    "type": "object",
                    "description": "Only include the parameters you actually want to change.",
                    "properties": {
                        "missing_frac": {"type": "number"},
                        "gap_frac": {"type": "number"},
                        "thin_outlier_k": {"type": "number"},
                        "bent_radius_mult": {"type": "number"},
                        "snap_r_vox": {"type": "number"},
                        "metal_r_vox": {"type": "number"},
                    },
                    "additionalProperties": False,
                },
                "label": {"type": "string", "description": "Short human-readable label for this version, e.g. 'stricter bend threshold'."},
            },
            "required": [],
        },
    },
    {
        "name": "generate_defect_gallery",
        "description": "Render the per-defect pipeline-validation figures and the zoomed defect atlas from the raw CT.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "export_3d_models",
        "description": "Export geometry-accurate, colour-coded 3D models (PLY/STL) per defect category.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "summarize_defects",
        "description": "Per-category counts/percentages for a classification version, plus which versions exist.",
        "input_schema": {"type": "object", "properties": {"version_id": VERSION_ID_PROPERTY}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# report_agent -- reads result metadata, explains, never mutates.
# ---------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """You are report_agent for Lattice Lens. You read the CURRENT job's already-computed result metadata and explain findings in plain language -- you never modify the pipeline and never invent a number.

Your tools:
- explain_strut(strut_id, version_id): the verdict and the exact measured evidence for one designed strut -- this is how you answer "why was this strut classified as X".
- defect_hotspots(top_n, version_id): the design's own unit cells ranked by defect rate, with real grid positions and centroids -- this is how you answer "where are defects concentrated".
- compare_thickness(strut_id, verdict, version_id): measured as-built radius vs. the pipeline's nominal radius and the design file's own nominal thickness value, for one strut or aggregated over a verdict category -- this is how you answer "is the printed thickness what the design specified".
- summarize_defects(version_id): overall counts/percentages and which classification versions exist.

Every tool result carries a "source" field naming exactly what pipeline artifact it came from; ground your explanation in it. If a tool returns a "note" field (e.g. compare_thickness's caveat that the pipeline's nominal radius and the design file's own nominal thickness are two different figures that don't have to agree, or the known measurement artifact near thick junctions), pass that nuance on -- do not smooth it into one confident number, and do not silently pick one figure as "the" design spec when the tool told you there are two. If evidence is genuinely ambiguous, say so plainly instead of guessing."""

REPORT_TOOLS = [
    {
        "name": "explain_strut",
        "description": "The verdict and exact measured evidence for one designed strut.",
        "input_schema": {
            "type": "object",
            "properties": {"strut_id": {"type": "integer"}, "version_id": VERSION_ID_PROPERTY},
            "required": ["strut_id"],
        },
    },
    {
        "name": "defect_hotspots",
        "description": "Rank the design's own unit cells by defect rate, with grid position and centroid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "How many top cells to return (default 10)."},
                "version_id": VERSION_ID_PROPERTY,
            },
            "required": [],
        },
    },
    {
        "name": "compare_thickness",
        "description": "Measured as-built radius vs. nominal figures, for one strut (pass strut_id) or aggregated over a verdict category (pass verdict).",
        "input_schema": {
            "type": "object",
            "properties": {
                "strut_id": {"type": "integer"},
                "verdict": {"type": "string", "enum": ["present", "missing", "bent", "thin", "disconnected"]},
                "version_id": VERSION_ID_PROPERTY,
            },
            "required": [],
        },
    },
    {
        "name": "summarize_defects",
        "description": "Per-category counts/percentages for a classification version, plus which versions exist.",
        "input_schema": {"type": "object", "properties": {"version_id": VERSION_ID_PROPERTY}, "required": []},
    },
]


# ---------------------------------------------------------------------------
# plot_agent -- analyzes result data, renders the most suitable PNG.
# ---------------------------------------------------------------------------

PLOT_SYSTEM_PROMPT = """You are plot_agent for Lattice Lens. You analyze the CURRENT job's result data and produce the single most suitable chart for the question, as a PNG saved on disk -- you never modify the pipeline.

Your plotting tools each compute the real numbers AND render them (matplotlib, colours matching the app's verdict palette everywhere else):
- plot_verdict_counts(version_id): bar chart of strut counts by verdict.
- plot_hotspots_map(top_n, version_id): horizontal bar chart of the unit cells with the highest defect rate -- use this for "where are defects concentrated".
- plot_thickness_distribution(verdict, version_id): histogram of measured as-built radius (optionally filtered to one verdict) with the pipeline's nominal radius marked -- use this for "is the thickness what was specified".
- plot_version_comparison(): grouped bar chart of verdict counts across every rerun version (only works once at least 2 versions exist -- if it errors because there's only one version, say so, don't fabricate a comparison).

Read-only tools, if you need the underlying numbers to decide what to plot or how to caption it (these do not plot anything):
- defect_hotspots(top_n, version_id), compare_thickness(strut_id, verdict, version_id), summarize_defects(version_id).

Pick the chart that actually answers the question -- do not default to plot_verdict_counts for everything. Report back the artifact's `id` (from the tool result's `artifact` object -- this is what the orchestrator needs to display it, never the filesystem path) and a one-line, numerically-grounded caption of what it shows; do not restate every value the chart already displays."""

PLOT_TOOLS = [
    {
        "name": "plot_verdict_counts",
        "description": "Bar chart of strut counts by verdict.",
        "input_schema": {"type": "object", "properties": {"version_id": VERSION_ID_PROPERTY}, "required": []},
    },
    {
        "name": "plot_hotspots_map",
        "description": "Horizontal bar chart of the unit cells with the highest defect rate.",
        "input_schema": {
            "type": "object",
            "properties": {"top_n": {"type": "integer"}, "version_id": VERSION_ID_PROPERTY},
            "required": [],
        },
    },
    {
        "name": "plot_thickness_distribution",
        "description": "Histogram of measured as-built radius, with the nominal radius marked.",
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["present", "missing", "bent", "thin", "disconnected"]},
                "version_id": VERSION_ID_PROPERTY,
            },
            "required": [],
        },
    },
    {
        "name": "plot_version_comparison",
        "description": "Grouped bar chart of verdict counts across every rerun version.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "defect_hotspots",
        "description": "Read-only: rank unit cells by defect rate without plotting.",
        "input_schema": {
            "type": "object",
            "properties": {"top_n": {"type": "integer"}, "version_id": VERSION_ID_PROPERTY},
            "required": [],
        },
    },
    {
        "name": "compare_thickness",
        "description": "Read-only: measured vs. nominal radius numbers without plotting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "strut_id": {"type": "integer"},
                "verdict": {"type": "string", "enum": ["present", "missing", "bent", "thin", "disconnected"]},
                "version_id": VERSION_ID_PROPERTY,
            },
            "required": [],
        },
    },
    {
        "name": "summarize_defects",
        "description": "Read-only: overall counts/percentages and which versions exist.",
        "input_schema": {"type": "object", "properties": {"version_id": VERSION_ID_PROPERTY}, "required": []},
    },
]


def _bind(job_id: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    return functools.partial(fn, job_id)


def _detection_dispatch(job_id: str) -> dict[str, Callable[..., Any]]:
    return {
        "run_initial_analysis": _bind(job_id, agent_tools.run_initial_analysis),
        "get_job_status": _bind(job_id, agent_tools.get_job_status),
        "rerun_classification": _bind(job_id, agent_tools.rerun_classification),
        "generate_defect_gallery": _bind(job_id, agent_tools.generate_defect_gallery),
        "export_3d_models": _bind(job_id, agent_tools.export_3d_models),
        "summarize_defects": _bind(job_id, agent_tools.summarize_defects),
    }


def _report_dispatch(job_id: str) -> dict[str, Callable[..., Any]]:
    return {
        "explain_strut": _bind(job_id, agent_tools.explain_strut),
        "defect_hotspots": _bind(job_id, agent_tools.defect_hotspots),
        "compare_thickness": _bind(job_id, agent_tools.compare_thickness),
        "summarize_defects": _bind(job_id, agent_tools.summarize_defects),
    }


def _plot_dispatch(job_id: str) -> dict[str, Callable[..., Any]]:
    return {
        "plot_verdict_counts": _bind(job_id, plot_tools.plot_verdict_counts),
        "plot_hotspots_map": _bind(job_id, plot_tools.plot_hotspots_map),
        "plot_thickness_distribution": _bind(job_id, plot_tools.plot_thickness_distribution),
        "plot_version_comparison": _bind(job_id, plot_tools.plot_version_comparison),
        "defect_hotspots": _bind(job_id, agent_tools.defect_hotspots),
        "compare_thickness": _bind(job_id, agent_tools.compare_thickness),
        "summarize_defects": _bind(job_id, agent_tools.summarize_defects),
    }


SUBAGENTS: dict[str, dict[str, Any]] = {
    "detection_agent": {"system": DETECTION_SYSTEM_PROMPT, "tools": DETECTION_TOOLS, "dispatch": _detection_dispatch},
    "report_agent": {"system": REPORT_SYSTEM_PROMPT, "tools": REPORT_TOOLS, "dispatch": _report_dispatch},
    "plot_agent": {"system": PLOT_SYSTEM_PROMPT, "tools": PLOT_TOOLS, "dispatch": _plot_dispatch},
}


def run_subagent(name: str, job_id: str, request: str, client: Any, model: str) -> dict[str, Any]:
    """Run one subagent's own, independent tool-calling loop on `request`.
    It has no memory of the orchestrator's conversation -- only what
    `request` tells it -- matching a real "separate context window" worker."""
    spec = SUBAGENTS[name]
    messages: list[dict[str, Any]] = [{"role": "user", "content": request}]
    return run_tool_loop(
        client=client,
        model=model,
        system=spec["system"],
        tools=spec["tools"],
        dispatch=spec["dispatch"](job_id),
        messages=messages,
    )
