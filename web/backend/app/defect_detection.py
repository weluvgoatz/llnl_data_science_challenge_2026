"""Runs the repository's v2 strut defect-detection pipeline (detect_v2.py),
scoped to one job's uploaded TIFF + design JSON.

v2 is self-contained: given a raw TIF + registered design JSON it segments
(global Otsu, analysis/defect_detection/v2/segment.py), skeletonises, builds
the as-built graph, and classifies every designed strut as
present/missing/bent/thin/disconnected -- one script, no separate
segment/clean/skeletonize/graph-build/classify chain. See
analysis/defect_detection/v2/INTEGRATION.md for the full contract.

v2's own mask/skeleton caches are namespaced (*_v2.tif / *_v2.skelcoords.npz,
see detect_v2.py) so this never reads a cache left by the old v1 script chain
(unified_defects_accurate.py and friends), which is no longer invoked from
this module at all.

v2 writes its classification to a path shared across all jobs
(analysis/defect_detection/v2/strut_classification_v2.json) rather than a
per-job path, so every call into v2 is serialized (_V2_LOCK) and the result is
copied into the job's own directory before the lock is released -- otherwise
two jobs analyzing concurrently could read each other's output.

_v2_to_v1_schema() adapts v2's record shape to the field names the rest of
the app (agent_tools.py, plot_tools.py, the frontend) already expect (`id`
not `i`, `evidence.measured_radius_um` not `dia_um`, etc.), so nothing
downstream of defects/classification.json had to change.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .store import job_dir, now, update_job

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DD_ROOT = _REPO_ROOT / "analysis" / "defect_detection"
_V2_ROOT = _DD_ROOT / "v2"

if str(_DD_ROOT) not in sys.path:
    sys.path.insert(0, str(_DD_ROOT))

# detect_v2.py now does in one subprocess call what used to be four separate
# stages (segment/skeleton/graph/classify), each with its own 3600s budget --
# doubled so a large scan's combined first run (segmentation + skeletonising,
# both one-off and cached after) has the same headroom the old chain had in
# total, not per-stage.
STAGE_TIMEOUT_SECONDS = 7200

# v2's detect_v2.py/export_3d.py write to a fixed path under v2/ rather than a
# per-job path (see module docstring) -- shared across every job/thread in
# this backend process, so only one v2 run may be in flight at a time.
_V2_LOCK = threading.Lock()

# v1's unified_defects_accurate.py exposed these as env-overridable knobs
# (config.py's LATTICE_MISSING_FRAC etc.). v2 (detect_v2.py) has no equivalent
# per-parameter tuning hooks -- its thresholds are derived from the scan's own
# measured diameter distribution (median - 3*MAD) or are fixed physical
# quantities (e.g. bend = 1 strut radius), not simple fractions. Kept as an
# empty table (rather than removed) so agent_tools.rerun_classification's
# "unknown parameter" check has something real to validate against and a
# threshold-rerun request fails with a clear, honest error instead of
# silently doing nothing or falling back to v1.
TUNABLE_PARAMS: dict[str, tuple[str, float]] = {}
DEFAULT_PARAMS: dict[str, float] = {}


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


def _run_script(script: str, env: dict[str, str], root: Path = _DD_ROOT) -> None:
    proc = subprocess.run(
        [sys.executable, str(root / script)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=STAGE_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script} failed: {(proc.stderr or proc.stdout)[-1500:]}")


run_script = _run_script  # public alias for agent_tools.rerun_classification
DD_ROOT = _DD_ROOT  # public alias: where the shared-output-path v1 viz scripts write
V2_ROOT = _V2_ROOT  # public alias: where v2's shared-output-path scripts write
V2_LOCK = _V2_LOCK  # public alias: serializes access to V2_ROOT's shared outputs
MASK_SUFFIX = "_segmented_clean_v2.tif"  # public alias: matches detect_v2.MASK_P's naming
SKELC_SUFFIX = "_segmented_clean_v2.skelcoords.npz"  # public alias: matches detect_v2.SKELC's naming


def v2_to_v1_schema(v2_payload: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
    """Adapt detect_v2.py's strut_classification_v2.json record shape to the
    field names agent_tools.py / plot_tools.py / the frontend already read
    from a v1 classification JSON (`id` not `i`, `evidence.measured_radius_um`
    not `dia_um`, ...), so nothing downstream of defects/classification.json
    needed to change to consume v2's output.

    v2's verdict vocabulary (present/missing/bent/thin/disconnected) is
    already identical to v1's, so `verdict` passes through unchanged.
    """
    design_struts_by_id = {s["id"]: s for s in design.get("struts", [])}
    thresholds = v2_payload["meta"]["thresholds"]
    nominal_radius_um = thresholds["nominal_diameter_um"] / 2

    # v2's own "reason"-equivalent: which measurement decided the verdict.
    # Not machine-checked downstream (no consumer branches on it -- see
    # frontend/src/types.ts's StrutEvidence.reason, only ever displayed/read
    # by an LLM tool call), just kept for parity with the v1 evidence shape.
    def reason_for(r: dict[str, Any]) -> str:
        if r.get("node0") == "absent" or r.get("node1") == "absent":
            return "no_metal_at_anchor"
        if r["verdict"] in ("present", "thin", "bent"):
            return "as_built_edge_matched"
        return "material_sample_between_anchors"

    out_struts = []
    for r in v2_payload["struts"]:
        sid = r["i"]
        design_strut = design_struts_by_id.get(sid, {})
        dia_um = r.get("dia_um")
        out_struts.append(
            {
                "id": sid,
                "p0": r["p0"],
                "p1": r["p1"],
                "verdict": r["verdict"],
                "design_thickness": design_strut.get("thickness"),
                "evidence": {
                    "reason": reason_for(r),
                    "measured_radius_um": (dia_um / 2) if dia_um is not None else None,
                    "nominal_radius_um": nominal_radius_um,
                    "bow_um": r.get("bow_um"),
                    "bow_threshold_um": thresholds.get("bent_bow_um"),
                    "subtype": r.get("subtype"),
                    "region": r.get("context"),
                    "node0_state": r.get("node0"),
                    "node1_state": r.get("node1"),
                    "gap_um": r.get("gap_um"),
                    "surviving_piece_um": r.get("stub_um"),
                    "diameter_thinnest_um": r.get("dia_min_um"),
                    "graph_hops": r.get("hops"),
                    "detached_at_joint": bool(r.get("flag_phantom_edge", False)),
                },
            }
        )

    return {
        "struts": out_struts,
        "meta": {
            "model": "v2",
            "counts": v2_payload["meta"]["counts"],
            "counts_interior": v2_payload["meta"].get("counts_interior"),
            "n": v2_payload["meta"]["n"],
            "volume_shape_zyx": v2_payload["meta"]["volume_shape_zyx"],
            "thresholds": thresholds,
        },
    }


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
    _set_defects(job_id, status="running", stage="detecting", error=None)

    try:
        base = Path(tiff_relative_path).stem
        stk = root / "defects" / "tif_stacks"
        stk.mkdir(parents=True, exist_ok=True)
        raw_link = stk / f"{base}.tif"
        if not raw_link.exists():
            raw_link.symlink_to((root / tiff_relative_path).resolve())

        env = build_env(stk, base, design_json_absolute_path)

        # detect_v2.py is self-contained: it segments (global Otsu), caches
        # the mask under stk/{base}_segmented_clean_v2.tif, skeletonises,
        # builds the as-built graph, and classifies -- one script, one stage,
        # from the app's point of view. It writes to a path shared across
        # every job (V2_ROOT), so this is locked and the result copied into
        # the job's own directory before another job's run can overwrite it.
        _set_defects(job_id, stage="detecting")
        with _V2_LOCK:
            _run_script("detect_v2.py", env, root=_V2_ROOT)
            v2_out = _V2_ROOT / "strut_classification_v2.json"
            if not v2_out.is_file():
                raise RuntimeError("detect_v2.py finished without producing strut_classification_v2.json")
            v2_payload = json.loads(v2_out.read_text(encoding="utf-8"))
            # audit copy of v2's own record, keyed by this job's scan -- kept
            # under stk (job-scoped) alongside the mask/skeleton cache.
            (stk / f"{base}_strut_classification_v2.json").write_text(
                json.dumps(v2_payload), encoding="utf-8"
            )

        _set_defects(job_id, stage="classifying")
        design = json.loads(Path(design_json_absolute_path).read_text(encoding="utf-8"))
        payload = v2_to_v1_schema(v2_payload, design)

        destination_relative = "defects/classification.json"
        (root / destination_relative).write_text(json.dumps(payload), encoding="utf-8")

        initial_version = {
            "id": 1,
            "label": "initial (v2 detector)",
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
