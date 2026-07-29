"""Deterministic preprocessing and fallback classification for Run Analysis."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .store import job_dir, load_job, save_job

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DD_ROOT = _REPO_ROOT / "analysis" / "defect_detection"
_SRC_ROOT = _REPO_ROOT / "web" / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import mcp_server as _segmentation_tools  # noqa: E402

STAGE_TIMEOUT_SECONDS = 3600


@dataclass(frozen=True)
class PreparedDefectInputs:
    base: str
    stack_dir: Path
    raw: Path
    design: Path
    mask: Path
    skeleton: Path
    graph: Path
    classifier_graph: Path
    classification: Path
    bend_detail: Path

    def environment(self) -> dict[str, str]:
        return {
            "LATTICE_BASE": self.base,
            "LATTICE_STK": str(self.stack_dir),
            "LATTICE_DESIGN_JSON": str(self.design),
            "LATTICE_PREBUILT_GRAPH": str(self.classifier_graph),
            "PYTHONIOENCODING": "utf-8",
        }


def _set_defects(job_id: str, **fields: Any) -> None:
    job = load_job(job_id)
    defects = dict(job.get("defects") or {})
    defects.update(fields)
    job["defects"] = defects
    save_job(job)


def _run_script(script: str, env: dict[str, str], timeout: int = STAGE_TIMEOUT_SECONDS) -> None:
    proc = subprocess.run(
        [sys.executable, str(_DD_ROOT / script)],
        cwd=str(_DD_ROOT),
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "no output")[-1500:]
        raise RuntimeError(f"{script} failed: {detail}")


def _valid_tiff(path: Path) -> bool:
    try:
        with tifffile.TiffFile(path) as image:
            return bool(image.pages)
    except (OSError, tifffile.TiffFileError):
        return False


def _valid_skeleton(path: Path) -> bool:
    try:
        with np.load(path) as cached:
            coords = cached["coords"]
        return coords.ndim == 2 and coords.shape[1] == 3 and len(coords) > 0
    except (OSError, ValueError, KeyError):
        return False


def _valid_graph(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return bool(payload["junctions"]) and isinstance(payload["struts"], list)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _valid_classifier_graph(path: Path) -> bool:
    try:
        with np.load(path) as cached:
            nodes, edges, bow = cached["nodes"], cached["edges"], cached["bow"]
        return (
            nodes.ndim == 2
            and nodes.shape[1] == 3
            and edges.ndim == 2
            and edges.shape[1] == 2
            and len(edges) == len(bow)
        )
    except (OSError, ValueError, KeyError):
        return False


def prepare_defect_inputs(
    job_id: str,
    tiff_relative_path: str,
    design_json_absolute_path: str,
) -> PreparedDefectInputs:
    """Run segmentation, skeletonization, and graph construction without Codex."""
    root = job_dir(job_id)
    base = Path(tiff_relative_path).stem
    stack = root / "defects" / "tif_stacks"
    stack.mkdir(parents=True, exist_ok=True)
    raw = stack / f"{base}.tif"
    if not raw.exists():
        raw.symlink_to((root / tiff_relative_path).resolve())

    prepared = PreparedDefectInputs(
        base=base,
        stack_dir=stack,
        raw=raw,
        design=Path(design_json_absolute_path).resolve(),
        mask=stack / f"{base}_segmented_clean.tif",
        skeleton=stack / f"{base}_segmented_clean.skelcoords.npz",
        graph=stack / f"{base}_segmented_clean_asbuilt_graph_cleaned.json",
        classifier_graph=stack / f"{base}_classifier_graph.npz",
        classification=stack / f"{base}_unified_defects_accurate.json",
        bend_detail=stack / f"{base}_asbuilt_bent.json",
    )
    env = prepared.environment()

    segmented = stack / f"{base}_segmented.tif"
    if not _valid_tiff(prepared.mask):
        _set_defects(job_id, status="running", stage="segmenting", error=None)
        if not _valid_tiff(segmented):
            result = _segmentation_tools.segment_tiff(
                str(raw), str(segmented), adaptive=True
            )
            if isinstance(result, str) and result.startswith("Error"):
                raise RuntimeError(f"TIFF segmentation failed: {result}")

        _set_defects(job_id, stage="cleaning")
        proc = subprocess.run(
            [
                sys.executable,
                "-u",
                str(_REPO_ROOT / "analysis" / "clean_segmentation.py"),
                "--mask",
                str(segmented),
                "--out",
                str(prepared.mask),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not _valid_tiff(prepared.mask):
            raise RuntimeError(
                f"clean_segmentation.py failed: {(proc.stderr or proc.stdout)[-1500:]}"
            )

    if not _valid_skeleton(prepared.skeleton):
        _set_defects(job_id, status="running", stage="skeletonizing", error=None)
        _run_script("prepare_skeleton.py", env)
        if not _valid_skeleton(prepared.skeleton):
            raise RuntimeError("skeletonization did not create a valid coordinate cache")

    if not (
        _valid_graph(prepared.graph)
        and _valid_classifier_graph(prepared.classifier_graph)
    ):
        _set_defects(job_id, status="running", stage="building_graph", error=None)
        _run_script("prepare_classifier_graph.py", env)
        if not _valid_graph(prepared.graph) or not _valid_classifier_graph(
            prepared.classifier_graph
        ):
            raise RuntimeError("graph construction did not create valid graph artifacts")

    _set_defects(job_id, status="running", stage="classifying", error=None)
    return prepared


def validate_classification(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        struts = payload["struts"]
        meta = payload["meta"]
        if not isinstance(struts, list) or not struts:
            raise ValueError("struts must be a non-empty list")
        if int(meta["n"]) != len(struts) or not isinstance(meta["counts"], dict):
            raise ValueError("classification metadata is inconsistent")
        allowed = {"present", "missing", "bent", "thin", "disconnected"}
        if any(item.get("verdict") not in allowed for item in struts):
            raise ValueError("classification contains an unknown verdict")
        return payload
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"invalid classification JSON: {exc}") from exc


def publish_classification(job_id: str, prepared: PreparedDefectInputs) -> None:
    validate_classification(prepared.classification)
    root = job_dir(job_id)
    destination_relative = "defects/classification.json"
    (root / destination_relative).write_bytes(prepared.classification.read_bytes())
    _set_defects(
        job_id,
        status="complete",
        stage="complete",
        error=None,
        path=destination_relative,
    )


def run_deterministic_classification(
    job_id: str, prepared: PreparedDefectInputs
) -> None:
    """Fallback used only when the downstream Codex phase is unavailable."""
    _set_defects(job_id, status="running", stage="classifying", error=None)
    _run_script("unified_defects_accurate.py", prepared.environment())
    _set_defects(job_id, stage="bend_detail")
    try:
        _run_script("bent_struts.py", prepared.environment())
    except Exception:
        pass
    publish_classification(job_id, prepared)


def run_strut_defect_detection(
    job_id: str,
    tiff_relative_path: str,
    design_json_absolute_path: str,
) -> None:
    """Backward-compatible deterministic full-pipeline entry point."""
    try:
        prepared = prepare_defect_inputs(
            job_id, tiff_relative_path, design_json_absolute_path
        )
        run_deterministic_classification(job_id, prepared)
    except Exception as exc:
        _set_defects(job_id, status="failed", error=str(exc))
