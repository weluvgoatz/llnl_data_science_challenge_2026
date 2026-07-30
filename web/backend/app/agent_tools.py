"""Deterministic, evidence-only tools the chat orchestrator calls.

None of these do numeric analysis of their own beyond simple, auditable
aggregation (grouping, counting, unit conversion) over numbers the pipeline
already computed and wrote to disk. Every result carries a "source" field
naming the artifact it was read from, so an agent's answer can always be
traced back to a real pipeline output -- nothing here invents a figure.
"""

from __future__ import annotations

import json
import shutil
import statistics
import threading
from pathlib import Path
from typing import Any

from . import defect_detection as dd
from . import workflow
from .store import job_dir, load_job, now, update_job


class ToolError(Exception):
    """A well-formed "can't do that" response, not a crash."""


def load_classification(job: dict, version_id: int | None = None) -> dict:
    defects = job.get("defects") or {}
    if version_id is None:
        path = defects.get("path")
    else:
        versions = defects.get("versions") or []
        match = next((v for v in versions if v["id"] == version_id), None)
        if not match:
            raise ToolError(f"No classification version {version_id} for this job.")
        path = match["path"]
    if not path:
        raise ToolError("Defect classification has not completed for this job yet.")
    full = job_dir(job["id"]) / path
    if not full.is_file():
        raise ToolError(f"Classification file missing on disk: {path}")
    return json.loads(full.read_text(encoding="utf-8"))


def load_design(job: dict) -> dict:
    design_item = next((f for f in job["files"] if f["kind"] == "json"), None)
    if not design_item:
        raise ToolError("This job has no design JSON uploaded.")
    return json.loads((job_dir(job["id"]) / design_item["path"]).read_text(encoding="utf-8"))


def summarize_defects(job_id: str, version_id: int | None = None) -> dict:
    """Overall per-category counts/percentages for a classification version,
    plus which versions exist (id, label, params) so a caller can ask for a
    specific one instead of always the currently-active version."""
    job = load_job(job_id)
    data = load_classification(job, version_id)
    counts = data["meta"]["counts"]
    n = data["meta"]["n"]
    defective = n - counts.get("present", 0)
    versions = (job.get("defects") or {}).get("versions") or []
    return {
        "source": "classification JSON meta.counts",
        "version_id": version_id if version_id is not None else (job.get("defects") or {}).get("activeVersionId"),
        "counts": counts,
        "n": n,
        "defect_count": defective,
        "defect_rate": defective / n if n else None,
        "available_versions": [{"id": v["id"], "label": v["label"], "params": v["params"]} for v in versions],
    }


def explain_strut(job_id: str, strut_id: int, version_id: int | None = None) -> dict:
    """Return the verdict and the exact measured evidence for one designed strut."""
    job = load_job(job_id)
    data = load_classification(job, version_id)
    strut = next((s for s in data["struts"] if s.get("id") == strut_id), None)
    if strut is None:
        raise ToolError(f"No strut with id {strut_id} in this classification.")
    return {
        "source": "classification JSON produced by unified_defects_accurate.py",
        "strut": strut,
    }


def defect_hotspots(job_id: str, top_n: int = 10, version_id: int | None = None) -> dict:
    """Rank the design's own unit cells by defect rate.

    Groups struts by the design JSON's `unit_cells` (each cell already lists
    the strut ids that belong to it, and a grid [i,j,k] location) -- this
    reuses the pipeline's existing grouping rather than inventing a spatial
    clustering.
    """
    job = load_job(job_id)
    data = load_classification(job, version_id)
    design = load_design(job)
    verdict_of = {s["id"]: s["verdict"] for s in data["struts"]}
    endpoints_of = {s["id"]: (s["p0"], s["p1"]) for s in data["struts"]}

    cells = []
    for cell in design.get("unit_cells", []):
        ids = [sid for sid in cell.get("struts", []) if sid in verdict_of]
        if not ids:
            continue
        counts: dict[str, int] = {}
        for sid in ids:
            v = verdict_of[sid]
            counts[v] = counts.get(v, 0) + 1
        defective = len(ids) - counts.get("present", 0)
        pts = [pt for sid in ids for pt in endpoints_of[sid]]
        centroid = [sum(axis) / len(pts) for axis in zip(*pts)] if pts else None
        cells.append(
            {
                "unit_cell_id": cell["id"],
                "grid_indices": cell.get("indices"),
                "n_struts": len(ids),
                "counts": counts,
                "defect_count": defective,
                "defect_fraction": defective / len(ids),
                "centroid_xyz": centroid,
            }
        )
    cells.sort(key=lambda c: c["defect_fraction"], reverse=True)
    return {
        "source": "design JSON unit_cells grouping + classification verdicts",
        "n_cells_with_struts": len(cells),
        "top_cells": cells[:top_n],
    }


THICKNESS_NOTE = (
    "measured_radius_um comes from a 2x-downsampled distance transform along the "
    "strut's own as-built centerline (only present when an as-built edge was "
    "matched -- missing/disconnected struts have no real edge to measure). "
    "nominal_radius_um is the physical strut radius the pipeline's own "
    "thresholds (e.g. the bend check) are computed against "
    "(config.STRUT_DIAMETER_UM / 2). design_thickness is the separate, "
    "unconverted nominal value carried in the design file itself. These two "
    "'nominal' figures are not guaranteed to agree -- report both, do not "
    "silently pick one. Known artifact (also documented for the lattice-3d-"
    "modeler skill): a strut whose centerline runs close to a thick junction "
    "can show an inflated measured_radius_um from the transform bleeding into "
    "the junction, not a real thickened strut -- treat values well above "
    "nominal_radius_um * 1.25 as suspect for this reason, especially on short "
    "struts or struts near a high-degree node, rather than as evidence the "
    "strut is oversized."
)


def compare_thickness(
    job_id: str,
    strut_id: int | None = None,
    verdict: str | None = None,
    version_id: int | None = None,
) -> dict:
    """Compare measured as-built radius to the design/nominal figures.

    Pass strut_id for one strut's detail, or verdict (e.g. "thin") for
    population statistics over every strut with that verdict that has a
    measured radius. All numbers are read directly from the classification
    JSON's per-strut evidence.
    """
    job = load_job(job_id)
    data = load_classification(job, version_id)
    struts = data["struts"]

    if strut_id is not None:
        strut = next((s for s in struts if s.get("id") == strut_id), None)
        if strut is None:
            raise ToolError(f"No strut with id {strut_id} in this classification.")
        ev = strut.get("evidence", {})
        return {
            "source": "classification JSON evidence",
            "note": THICKNESS_NOTE,
            "strut_id": strut_id,
            "verdict": strut["verdict"],
            "design_thickness": strut.get("design_thickness"),
            "measured_radius_um": ev.get("measured_radius_um"),
            "nominal_radius_um": ev.get("nominal_radius_um"),
        }

    pool = [s for s in struts if verdict is None or s["verdict"] == verdict]
    if not pool:
        raise ToolError(f"No struts with verdict={verdict!r} in this classification.")
    radii = [
        s["evidence"]["measured_radius_um"]
        for s in pool
        if s.get("evidence", {}).get("measured_radius_um") is not None
    ]
    if not radii:
        raise ToolError(
            f"No struts with a measured radius for verdict={verdict!r} "
            "(they may all lack an as-built edge match)."
        )
    nominal = next(
        (
            s["evidence"]["nominal_radius_um"]
            for s in pool
            if s.get("evidence", {}).get("nominal_radius_um") is not None
        ),
        None,
    )
    return {
        "source": f"classification JSON evidence, aggregated over {len(radii)} struts"
        + (f" with verdict={verdict!r}" if verdict else ""),
        "note": THICKNESS_NOTE,
        "verdict_filter": verdict,
        "n_measured": len(radii),
        "n_total_in_group": len(pool),
        "mean_measured_radius_um": statistics.mean(radii),
        "median_measured_radius_um": statistics.median(radii),
        "min_measured_radius_um": min(radii),
        "max_measured_radius_um": max(radii),
        "nominal_radius_um": nominal,
    }


ALLOWED_OVERRIDES = set(dd.TUNABLE_PARAMS)


def rerun_classification(
    job_id: str,
    overrides: dict[str, float] | None = None,
    label: str | None = None,
) -> dict:
    """Re-run unified_defects_accurate.py with modified thresholds, versioned.

    Never overwrites a prior run: writes a new classification_v<n>.json,
    appends it to job["defects"]["versions"], and makes it the active
    version -- every earlier version stays on disk and in the version list.
    Requires the job's initial analysis (segmentation + skeleton cache) to
    have already completed; this only redoes the classification stage, which
    is the stage these thresholds actually govern.

    This does real, potentially multi-minute work (re-registers the design,
    re-anchors every node, re-classifies every strut) -- callers running this
    from a request handler should do so on a background thread and poll
    job["defects"], the same way the initial analysis run already works.
    """
    overrides = overrides or {}
    bad = set(overrides) - ALLOWED_OVERRIDES
    if bad:
        raise ToolError(f"Unknown parameter(s): {sorted(bad)}. Allowed: {sorted(ALLOWED_OVERRIDES)}")

    job = load_job(job_id)
    defects = job.get("defects") or {}
    versions = defects.get("versions") or []
    if not versions:
        raise ToolError("No completed initial analysis to rerun classification against.")

    base, stk, design_json_abs = dd.resolve_specimen_paths(job)
    mask = stk / f"{base}_segmented_clean.tif"
    skelc = stk / f"{base}_segmented_clean.skelcoords.npz"
    if not mask.is_file() or not skelc.is_file():
        raise ToolError("Segmentation/skeleton cache missing -- run the initial analysis first.")

    params = dict(dd.DEFAULT_PARAMS)
    params.update(overrides)

    env = dd.build_env(stk, base, design_json_abs, overrides)
    dd.run_script("unified_defects_accurate.py", env)
    try:
        dd.run_script("bent_struts.py", env)
    except Exception:
        pass  # supplementary bow/tortuosity detail; classification already succeeded

    classified = stk / f"{base}_unified_defects_accurate.json"
    if not classified.is_file():
        raise ToolError("Rerun finished without producing a classification JSON.")
    payload = json.loads(classified.read_text(encoding="utf-8"))

    root = job_dir(job_id)
    next_id = max(v["id"] for v in versions) + 1
    dest_relative = f"defects/classification_v{next_id}.json"
    (root / dest_relative).write_bytes(classified.read_bytes())

    version_entry = {
        "id": next_id,
        "label": label or (f"rerun with {overrides}" if overrides else "rerun (defaults)"),
        "path": dest_relative,
        "params": params,
        "counts": payload["meta"]["counts"],
        "n": payload["meta"]["n"],
        "createdAt": now(),
    }
    versions.append(version_entry)

    def apply(current: dict[str, Any]) -> None:
        current_defects = dict(current.get("defects") or {})
        current_defects["versions"] = versions
        current_defects["activeVersionId"] = next_id
        current_defects["path"] = dest_relative
        current["defects"] = current_defects

    update_job(job_id, apply)

    previous = versions[-2] if len(versions) > 1 else None
    return {
        "source": "unified_defects_accurate.py rerun",
        "version": version_entry,
        "previous_counts": previous["counts"] if previous else None,
    }


# viz_defect_atlas.py, viz_pipeline_perdefect.py, and export_realistic_models.py
# (unlike every classification-stage script) write to a FIXED, shared path
# under analysis/defect_detection/ rather than a per-job LATTICE_STK path --
# they were built for one CLI/Codex user at a time. Until they're
# parameterized like the rest of the pipeline, this lock serializes calls to
# them so two jobs' galleries/models can never interleave and corrupt each
# other; only one such job runs at a time across the whole backend.
_SHARED_OUTPUT_LOCK = threading.Lock()


def _require_segmented(job: dict) -> tuple[str, Path, str]:
    base, stk, design_json_abs = dd.resolve_specimen_paths(job)
    mask = stk / f"{base}_segmented_clean.tif"
    if not mask.is_file():
        raise ToolError("Segmentation missing -- run the initial analysis first.")
    return base, stk, design_json_abs


def _move_and_register(job_id: str, src: Path, dest_subdir: str, caption: str, media_type: str) -> dict[str, Any]:
    root = job_dir(job_id)
    dest_dir = root / dest_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    src.replace(dest)
    relative = str(dest.relative_to(root))
    artifact = {
        "id": f"{dest_subdir}_{dest.name}",
        "name": dest.name,
        "caption": caption,
        "path": relative,
        "mediaType": media_type,
    }
    def apply(current: dict[str, Any]) -> None:
        artifacts = [a for a in current.get("artifacts", []) if a["id"] != artifact["id"]]
        artifacts.append(artifact)
        current["artifacts"] = artifacts

    update_job(job_id, apply)
    return artifact


def generate_defect_gallery(job_id: str) -> dict:
    """Render the per-defect pipeline-validation figures and the zoomed CT
    atlas (detection_agent). Requires a completed classification.

    Every panel is measured from the same raw scan the classifier used --
    see .agents/skills/defect_visualizer/SKILL.md. Serialized (see
    _SHARED_OUTPUT_LOCK) because the underlying scripts share one repo-wide
    output path.
    """
    job = load_job(job_id)
    base, stk, design_json_abs = _require_segmented(job)
    env = dd.build_env(stk, base, design_json_abs)

    with _SHARED_OUTPUT_LOCK:
        for stale in list(dd.DD_ROOT.glob("VIZ_9_defect_atlas.png")) + list(dd.DD_ROOT.glob("VIZ_10_pipeline_*.png")):
            stale.unlink(missing_ok=True)

        dd.run_script("viz_defect_atlas.py", env)
        dd.run_script("viz_pipeline_perdefect.py", env)

        produced = sorted(dd.DD_ROOT.glob("VIZ_9_defect_atlas.png")) + sorted(dd.DD_ROOT.glob("VIZ_10_pipeline_*.png"))
        artifacts = [
            _move_and_register(job_id, f, "gallery", f"Defect gallery: {f.stem.replace('_', ' ')}", "image/png")
            for f in produced
        ]
    if not artifacts:
        raise ToolError("Gallery scripts ran but produced no recognizable output files.")
    return {"source": "viz_defect_atlas.py + viz_pipeline_perdefect.py", "artifacts": artifacts}


def export_3d_models(job_id: str) -> dict:
    """Export geometry-accurate, colour-coded 3D models (PLY/STL) where each
    defect carries its real as-built shape (detection_agent). Requires a
    completed classification. See
    .agents/skills/lattice_3d_modeler/SKILL.md for what "geometry-accurate"
    means per category. Serialized (see _SHARED_OUTPUT_LOCK).
    """
    job = load_job(job_id)
    base, stk, design_json_abs = _require_segmented(job)
    env = dd.build_env(stk, base, design_json_abs)

    outdir = dd.DD_ROOT
    parts_dir = outdir / "MODEL_realistic_parts"
    fixed_names = (
        "MODEL_lattice_realistic.ply",
        "MODEL_lattice_realistic_coloured.stl",
        "MODEL_defects_only.ply",
    )

    with _SHARED_OUTPUT_LOCK:
        for name in fixed_names:
            (outdir / name).unlink(missing_ok=True)
        if parts_dir.is_dir():
            shutil.rmtree(parts_dir)

        dd.run_script("export_realistic_models.py", env)

        produced = [outdir / name for name in fixed_names if (outdir / name).is_file()]
        if parts_dir.is_dir():
            produced.extend(sorted(parts_dir.glob("*")))

        media_by_suffix = {".ply": "application/octet-stream", ".stl": "model/stl"}
        artifacts = [
            _move_and_register(
                job_id, f, "models", f"3D model: {f.name}", media_by_suffix.get(f.suffix, "application/octet-stream")
            )
            for f in produced
        ]
    if not artifacts:
        raise ToolError("Export script ran but produced no recognizable output files.")
    return {"source": "export_realistic_models.py", "artifacts": artifacts}


def run_initial_analysis(job_id: str) -> dict[str, Any]:
    """Start this job's initial analysis (segmentation through classification)
    on a background thread and return immediately -- this never blocks
    waiting for the multi-minute pipeline. Call get_job_status afterward (or
    on a later turn) to check progress; the job must be in "intake_ready" or
    "failed" state, i.e. not already analyzing or complete.
    """
    try:
        job = workflow.start_analysis(job_id)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {"source": "job state", "status": "started", "job_state": job["state"]}


def get_job_status(job_id: str) -> dict[str, Any]:
    """Real-time status for this job: overall state, and (once started) the
    defect-detection stage/status. Use this to answer "how's it going" or to
    check whether a surface (e.g. the defect view) is ready to show yet."""
    job = load_job(job_id)
    defects = job.get("defects") or {}
    return {
        "source": "job state",
        "job_state": job["state"],
        "job_error": job.get("error"),
        "defects_status": defects.get("status"),
        "defects_stage": defects.get("stage"),
        "defects_error": defects.get("error"),
        "has_report": job.get("report") is not None,
        "has_design_json": any(f["kind"] == "json" for f in job["files"]),
        "has_tiff": any(f["kind"] == "tiff" for f in job["files"]),
    }
