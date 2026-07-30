"""Runs the repository's own strut defect-detection pipeline (the same stages
the strut_error_detection_agent performs) as a fixed, deterministic subprocess
chain, scoped to one job's uploaded TIFF + design JSON.

Stages: segment (adaptive Otsu) -> clean -> skeletonize -> build as-built graph
-> classify every designed strut as present/missing/bent/thin/disconnected.
Each stage is the exact script the agent itself calls via MCP
(web/src/mcp_server.py's segment_tiff, analysis/clean_segmentation.py, and the
analysis/defect_detection/*.py scripts), just invoked directly instead of
through an LLM tool-call loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .store import job_dir, now, update_job

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DD_ROOT = _REPO_ROOT / "analysis" / "defect_detection"
_SRC_ROOT = _REPO_ROOT / "web" / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_DD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DD_ROOT))

import mcp_server as _segmentation_tools  # noqa: E402  (repo's web/src/mcp_server.py)
import config as _dd_config  # noqa: E402  (analysis/defect_detection/config.py)

STAGE_TIMEOUT_SECONDS = 3600

# Classification thresholds that config.py exposes as env-overridable knobs
# (analysis/defect_detection/config.py), keyed by the short name the chat
# tools use -> (env var name, current default read from that same config
# module). Single source of truth: if config.py's defaults change, these
# follow automatically instead of drifting out of sync.
TUNABLE_PARAMS: dict[str, tuple[str, float]] = {
    "missing_frac": ("LATTICE_MISSING_FRAC", _dd_config.MISSING_FRAC),
    "gap_frac": ("LATTICE_GAP_FRAC", _dd_config.GAP_FRAC),
    "thin_outlier_k": ("LATTICE_THIN_OUTLIER_K", _dd_config.THIN_OUTLIER_K),
    "bent_radius_mult": ("LATTICE_BENT_RADIUS_MULT", _dd_config.BENT_RADIUS_MULT),
    "snap_r_vox": ("LATTICE_SNAP_R_VOX", _dd_config.SNAP_R_VOX),
    "metal_r_vox": ("LATTICE_METAL_R_VOX", _dd_config.METAL_R_VOX),
}
DEFAULT_PARAMS: dict[str, float] = {name: default for name, (_, default) in TUNABLE_PARAMS.items()}


def resolve_specimen_paths(job: dict) -> tuple[str, Path, str]:
    """Derive (base, stk_dir, design_json_abs_path) for a job the same way
    run_strut_defect_detection does, so a later rerun targets the exact same
    working directory without needing the original call's arguments again."""
    root = job_dir(job["id"])
    tiff_item = next((f for f in job["files"] if f["kind"] == "tiff"), None)
    design_item = next((f for f in job["files"] if f["kind"] == "json"), None)
    if not tiff_item or not design_item:
        raise ValueError("Job is missing a TIFF and/or design JSON input")
    base = Path(tiff_item["path"]).stem
    stk = root / "defects" / "tif_stacks"
    design_json_abs = str((root / design_item["path"]).resolve())
    return base, stk, design_json_abs


def build_env(stk: Path, base: str, design_json_abs: str, overrides: dict[str, float] | None = None) -> dict[str, str]:
    """Environment for a pipeline-script subprocess, with optional
    classification-threshold overrides (see TUNABLE_PARAMS)."""
    env = dict(os.environ)
    env["LATTICE_STK"] = str(stk)
    env["LATTICE_BASE"] = base
    env["LATTICE_DESIGN_JSON"] = design_json_abs
    env["PYTHONIOENCODING"] = "utf-8"
    for name, value in (overrides or {}).items():
        env_name, _ = TUNABLE_PARAMS[name]
        env[env_name] = str(value)
    return env


def _set_defects(job_id: str, **fields: Any) -> None:
    # Atomic (see store.update_job) -- this fires many times per pipeline
    # run and can overlap a lingering tilt-check thread for another file.
    def apply(current: dict[str, Any]) -> None:
        defects = dict(current.get("defects") or {})
        defects.update(fields)
        current["defects"] = defects

    update_job(job_id, apply)


def _run_script(script: str, env: dict[str, str]) -> None:
    proc = subprocess.run(
        [sys.executable, str(_DD_ROOT / script)],
        cwd=str(_DD_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=STAGE_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed: {(proc.stderr or proc.stdout)[-1500:]}")


run_script = _run_script  # public alias for agent_tools.rerun_classification
DD_ROOT = _DD_ROOT  # public alias: where the shared-output-path scripts write


def run_strut_defect_detection(
    job_id: str,
    tiff_relative_path: str,
    design_json_absolute_path: str,
) -> None:
    """Classify every designed strut in a job's TIFF against its design JSON.

    Writes progress into job["defects"] as it goes (status/stage), and never
    raises: failures are recorded in job["defects"] so a defect-detection
    failure never takes down the rest of the analysis run.
    """
    root = job_dir(job_id)
    _set_defects(job_id, status="running", stage="segmenting", error=None)

    try:
        base = Path(tiff_relative_path).stem
        stk = root / "defects" / "tif_stacks"
        stk.mkdir(parents=True, exist_ok=True)
        raw_link = stk / f"{base}.tif"
        if not raw_link.exists():
            raw_link.symlink_to((root / tiff_relative_path).resolve())

        env = build_env(stk, base, design_json_absolute_path)

        segmented = stk / f"{base}_segmented.tif"
        _segmentation_tools.segment_tiff(str(raw_link), str(segmented), adaptive=True)

        _set_defects(job_id, stage="cleaning")
        cleaned = stk / f"{base}_segmented_clean.tif"
        proc = subprocess.run(
            [
                sys.executable,
                str(_REPO_ROOT / "analysis" / "clean_segmentation.py"),
                "--mask", str(segmented),
                "--out", str(cleaned),
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"clean_segmentation.py failed: {proc.stderr[-1500:]}")

        _set_defects(job_id, stage="skeletonizing")
        _run_script("skel_to_json.py", env)

        _set_defects(job_id, stage="building_graph")
        _run_script("clean_and_compare.py", env)

        _set_defects(job_id, stage="classifying")
        _run_script("unified_defects_accurate.py", env)

        _set_defects(job_id, stage="bend_detail")
        try:
            _run_script("bent_struts.py", env)
        except Exception:
            pass  # supplementary bow/tortuosity detail; classification already succeeded

        classified = stk / f"{base}_unified_defects_accurate.json"
        if not classified.is_file():
            raise RuntimeError("Pipeline finished without producing a classification JSON")

        destination_relative = "defects/classification.json"
        (root / destination_relative).write_bytes(classified.read_bytes())
        payload = json.loads(classified.read_text(encoding="utf-8"))

        initial_version = {
            "id": 1,
            "label": "initial (default thresholds)",
            "path": destination_relative,
            "params": dict(DEFAULT_PARAMS),
            "counts": payload["meta"]["counts"],
            "n": payload["meta"]["n"],
            "createdAt": now(),
        }
        _set_defects(
            job_id,
            status="complete",
            stage="complete",
            error=None,
            path=destination_relative,
            versions=[initial_version],
            activeVersionId=1,
        )
    except Exception as exc:
        _set_defects(job_id, status="failed", error=str(exc))
