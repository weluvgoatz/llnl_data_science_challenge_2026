"""mount_surface / unmount_surface: the orchestrator's UI-state tools.

These don't compute anything about the lattice -- they validate that a
requested surface's data actually exists for the current job (a design JSON,
a completed classification, a generated artifact, a report, ...) and pass
through exactly what the frontend needs to render it. Every failure is a
ToolError with a message written for a user, not a stack trace, so a
precondition that isn't met yet (analysis still running, no design JSON
uploaded) comes back as something the orchestrator can relay plainly instead
of a generic tool-call failure.
"""

from __future__ import annotations

from typing import Any

from ..store import load_job


class ToolError(Exception):
    """A well-formed "can't show that (yet)" response, not a crash."""


VALID_COMPONENTS = {"ModelViewer", "DefectView", "ReportView", "DataViz"}


def mount_surface(
    job_id: str,
    component: str,
    file_id: str | None = None,
    artifact_id: str | None = None,
    version_id: int | None = None,
    filter_verdicts: list[str] | None = None,
    select_strut_ids: list[int] | None = None,
    slice_index: int | None = None,
    show_tilt_pane: bool | None = None,
) -> dict[str, Any]:
    if component not in VALID_COMPONENTS:
        raise ToolError(f"Unknown component {component!r}. Valid components: {sorted(VALID_COMPONENTS)}")
    job = load_job(job_id)

    if component == "ModelViewer":
        # Accepts either an uploaded STL (file_id) or a generated model
        # artifact (artifact_id, e.g. from detection_agent's export_3d_models
        # -- the per-category .stl parts) -- both render through the same
        # Three.js STL loader, just from a different URL.
        if artifact_id is not None:
            artifact = _find_artifact(job, artifact_id)
            if artifact["mediaType"] != "model/stl":
                raise ToolError(
                    f"Artifact {artifact_id!r} is a {artifact['mediaType']}, not an STL -- "
                    "PLY models don't have an in-browser viewer here; point the user at the download instead."
                )
            return _directive("ModelViewer", artifact_id=artifact_id)
        file = _find_file(job, file_id, kind="stl")
        return _directive("ModelViewer", file_id=file["id"])

    if component == "DataViz":
        if artifact_id is not None:
            _find_artifact(job, artifact_id)
            return _directive("DataViz", artifact_id=artifact_id)
        file = _find_file(job, file_id, kind=None)
        if file["kind"] not in ("tiff", "json"):
            raise ToolError(
                "DataViz needs a TIFF (renders a slice) or a design JSON (renders its strut graph), "
                "or pass artifact_id for a generated plot/gallery image."
            )
        if show_tilt_pane and file["kind"] != "tiff":
            raise ToolError("show_tilt_pane only applies to a TIFF file.")
        return _directive("DataViz", file_id=file["id"], slice_index=slice_index, show_tilt_pane=show_tilt_pane)

    if component == "DefectView":
        defects = job.get("defects")
        has_json = any(f["kind"] == "json" for f in job["files"])
        if not defects:
            if job["state"] in ("new", "intake_ready"):
                raise ToolError("Analysis hasn't been run yet. Ask me to run the analysis first.")
            if job["state"] == "analyzing":
                raise ToolError("Analysis is still running. I'll let you know when defect detection is ready.")
            if not has_json:
                raise ToolError(
                    "No design JSON was uploaded for this job, so defect detection was skipped -- the rest of "
                    "the analysis completed, but there's no classification to show. I can show the raw scan "
                    "(DataViz on the TIFF) instead."
                )
            raise ToolError("Defect classification is not available for this job.")
        if defects.get("status") == "running":
            raise ToolError(f"Defect detection is still running (stage: {defects.get('stage', 'unknown')}). I'll let you know when it's ready.")
        if defects.get("status") != "complete":
            raise ToolError(f"Defect detection failed: {defects.get('error') or 'unknown error'}.")
        if version_id is not None:
            versions = defects.get("versions") or []
            if not any(v["id"] == version_id for v in versions):
                raise ToolError(f"No classification version {version_id} for this job.")
        return _directive(
            "DefectView",
            version_id=version_id,
            filter_verdicts=filter_verdicts,
            select_strut_ids=select_strut_ids,
        )

    if component == "ReportView":
        if not job.get("report"):
            raise ToolError("No report yet -- it's generated once analysis completes.")
        return _directive("ReportView")

    raise ToolError(f"Unhandled component {component!r}")  # unreachable given the check above


def unmount_surface(job_id: str) -> dict[str, Any]:
    load_job(job_id)  # 404s (via store.job_dir) if the job doesn't exist
    return {"mounted": None}


def _find_file(job: dict, file_id: str | None, kind: str | None) -> dict:
    if not file_id:
        candidates = [f for f in job["files"] if kind is None or f["kind"] == kind]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ToolError(f"No {kind or 'matching'} file in this job.")
        raise ToolError(
            "Multiple files match -- specify file_id. Candidates: "
            + ", ".join(f"{f['id']} ({f['name']})" for f in candidates)
        )
    file = next((f for f in job["files"] if f["id"] == file_id), None)
    if not file:
        raise ToolError(f"No file with id {file_id!r} in this job.")
    if kind and file["kind"] != kind:
        raise ToolError(f"File {file_id!r} is a {file['kind']}, not a {kind}.")
    return file


def _find_artifact(job: dict, artifact_id: str) -> dict:
    artifact = next((a for a in job["artifacts"] if a["id"] == artifact_id), None)
    if not artifact:
        raise ToolError(f"No artifact with id {artifact_id!r} in this job.")
    return artifact


def _directive(component: str, **props: Any) -> dict[str, Any]:
    return {"mounted": {"component": component, "props": {k: v for k, v in props.items() if v is not None}}}
