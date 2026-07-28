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

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .store import job_dir, load_job, save_job

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DD_ROOT = _REPO_ROOT / "analysis" / "defect_detection"
_SRC_ROOT = _REPO_ROOT / "web" / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import mcp_server as _segmentation_tools  # noqa: E402  (repo's web/src/mcp_server.py)

STAGE_TIMEOUT_SECONDS = 3600


def _set_defects(job_id: str, **fields: Any) -> None:
    job = load_job(job_id)
    defects = dict(job.get("defects") or {})
    defects.update(fields)
    job["defects"] = defects
    save_job(job)


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

        env = dict(os.environ)
        env["LATTICE_STK"] = str(stk)
        env["LATTICE_BASE"] = base
        env["LATTICE_DESIGN_JSON"] = design_json_absolute_path
        env["PYTHONIOENCODING"] = "utf-8"

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

        _set_defects(
            job_id,
            status="complete",
            stage="complete",
            error=None,
            path=destination_relative,
        )
    except Exception as exc:
        _set_defects(job_id, status="failed", error=str(exc))
