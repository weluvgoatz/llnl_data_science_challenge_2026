from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

_MATPLOTLIB_CONFIG = Path(__file__).resolve().parents[2] / ".matplotlib"
_MATPLOTLIB_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CONFIG))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

# web/analysis is a sibling of web/backend; make it importable to reuse the
# repository's own tilt-estimation algorithm instead of duplicating it.
_ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from analysis.tiff_tilt import _forward_rotation_matrix, estimate_tilt  # noqa: E402

from .store import job_dir, load_job, update_job


def inspect_upload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return {"kind": "json", "summary": f"Valid {type(payload).__name__} JSON"}
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as image:
            if not image.pages:
                raise ValueError("TIFF contains no pages")
            first = image.pages[0]
            return {
                "kind": "tiff",
                "pageCount": len(image.pages),
                "width": first.imagewidth,
                "height": first.imagelength,
                "summary": f"{len(image.pages)} slices · {first.imagewidth}×{first.imagelength}",
            }
    if suffix == ".stl":
        size = path.stat().st_size
        if size < 84:
            text = path.read_bytes()[:256].lstrip().lower()
            if not text.startswith(b"solid"):
                raise ValueError("STL is empty or malformed")
        triangles = None
        with path.open("rb") as stream:
            header = stream.read(84)
        if len(header) == 84:
            candidate = struct.unpack("<I", header[80:84])[0]
            if 84 + candidate * 50 == size:
                triangles = candidate
        return {
            "kind": "stl",
            "triangleCount": triangles,
            "summary": f"{triangles:,} triangles" if triangles is not None else "ASCII STL mesh",
        }
    raise ValueError("Supported file types are .json, .tif, .tiff, and .stl")


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    values = frame.astype(np.float32)
    low, high = np.percentile(values, (1, 99))
    if high <= low:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0, 1)


def render_tiff_slice(source: Path, destination: Path, index: int) -> None:
    with tifffile.TiffFile(source) as image:
        if index < 0 or index >= len(image.pages):
            raise IndexError(index)
        frame = _normalize_frame(image.pages[index].asarray())
    fig, axis = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    axis.imshow(frame, cmap="gray", origin="lower")
    axis.set_title(f"CT slice {index}")
    axis.set_xlabel("X pixel")
    axis.set_ylabel("Y pixel")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, facecolor="#f4f3ef")
    plt.close(fig)


# estimate_tilt() requires a binary mask (it fits lines through repeated
# lattice edges); raw uploads are grayscale, so tilt detection runs against
# an Otsu-thresholded copy while the correction itself is applied to the
# original grayscale volume so the comparison still looks like CT data.
TILT_THRESHOLD_DEGREES = 0.1
_TILT_SAMPLE_STRIDE = 40


def _binarize_tiff(source: Path, destination: Path) -> None:
    with tifffile.TiffFile(source) as image:
        pages = image.pages
        sample = np.stack(
            [pages[index].asarray() for index in range(0, len(pages), _TILT_SAMPLE_STRIDE)]
        )
        threshold = threshold_otsu(sample)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tifffile.TiffWriter(destination) as writer:
            for page in pages:
                frame = (page.asarray() >= threshold).astype(np.uint8)
                writer.write(frame, contiguous=True)


def _rotate_grayscale_tiff(
    source: Path,
    destination: Path,
    zy_angle_degrees: float,
    zx_angle_degrees: float,
) -> None:
    with tifffile.TiffFile(source) as image:
        shape = (len(image.pages),) + image.pages[0].shape
        dtype = image.pages[0].asarray().dtype
        volume = np.empty(shape, dtype=dtype)
        for index, page in enumerate(image.pages):
            volume[index] = page.asarray()

    # Voxel spacing is assumed isotropic: uploads carry no physical
    # calibration, so the scale terms in the general tilt-tool formula
    # reduce to the identity and can be skipped here.
    rotation = _forward_rotation_matrix(zy_angle_degrees, zx_angle_degrees)
    inverse = np.linalg.inv(rotation)
    center = (np.asarray(shape, dtype=float) - 1.0) / 2.0
    offset = center - inverse @ center
    background = int(np.percentile(volume[::_TILT_SAMPLE_STRIDE], 1))
    corrected = ndi.affine_transform(
        volume,
        inverse,
        offset=offset,
        output_shape=shape,
        order=1,
        mode="constant",
        cval=background,
        prefilter=False,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(destination, corrected, photometric="minisblack")


def check_and_correct_tilt(job_id: str, file_id: str) -> None:
    # Every mutation below goes through update_job (atomic load+mutate+save
    # under one lock) instead of separate load_job()/save_job() calls --
    # this runs as a background thread that can overlap with the analysis
    # pipeline's own background thread touching the same job.json, and a
    # stale full-dict save from whichever thread finishes last would
    # silently discard the other's fields (this is exactly how "analyzing"
    # was observed reverting to "intake_ready" moments after starting).
    job = load_job(job_id)
    item = next((entry for entry in job["files"] if entry["id"] == file_id), None)
    if not item or item["kind"] != "tiff":
        return
    root = job_dir(job_id)
    source = root / item["path"]

    def mark_checking(current: dict[str, Any]) -> None:
        current_item = next(entry for entry in current["files"] if entry["id"] == file_id)
        current_item["tiltStatus"] = "checking"
        current_item["tiltError"] = None

    update_job(job_id, mark_checking)

    try:
        binary_path = root / "tilt" / f"{file_id}-binary.tif"
        _binarize_tiff(source, binary_path)
        estimate = estimate_tilt(binary_path)
        binary_path.unlink(missing_ok=True)
        not_tilted = (
            abs(estimate.zy.angle_degrees) <= TILT_THRESHOLD_DEGREES
            and abs(estimate.zx.angle_degrees) <= TILT_THRESHOLD_DEGREES
        )

        def apply_estimate(current: dict[str, Any]) -> None:
            current_item = next(entry for entry in current["files"] if entry["id"] == file_id)
            current_item["tiltZY"] = estimate.zy.angle_degrees
            current_item["tiltZX"] = estimate.zx.angle_degrees
            if not_tilted:
                current_item["tiltStatus"] = "not_tilted"

        update_job(job_id, apply_estimate)
        if not_tilted:
            return

        corrected_relative = f"tilt/{file_id}-corrected.tif"
        _rotate_grayscale_tiff(
            source,
            root / corrected_relative,
            estimate.zy.angle_degrees,
            estimate.zx.angle_degrees,
        )

        def apply_corrected(current: dict[str, Any]) -> None:
            current_item = next(entry for entry in current["files"] if entry["id"] == file_id)
            current_item["correctedPath"] = corrected_relative
            current_item["tiltStatus"] = "corrected"

        update_job(job_id, apply_corrected)
    except Exception as exc:

        def apply_failure(current: dict[str, Any]) -> None:
            current_item = next(entry for entry in current["files"] if entry["id"] == file_id)
            current_item["tiltStatus"] = "failed"
            current_item["tiltError"] = str(exc)

        update_job(job_id, apply_failure)


def _run_external_harness(root: Path) -> bool:
    command = os.environ.get("CODEX_ANALYSIS_COMMAND", "").strip()
    if not command:
        return False
    prompt = (
        "Analyze the lattice files in uploads/. Use the configured MCP tools and "
        "NDE report skill. Write PNG results to analysis/, a Markdown report to "
        "report/report.md, and artifact-manifest.json at the job root."
    )
    completed = subprocess.run(
        [part.format(job_dir=str(root), prompt=prompt) for part in shlex.split(command)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("CODEX_ANALYSIS_TIMEOUT", "900")),
        check=False,
    )
    (root / "codex.log").write_text(
        completed.stdout + "\n--- stderr ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"Codex harness exited with status {completed.returncode}")
    return True


def _discover_outputs(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    artifacts = []
    for path in sorted((root / "analysis").glob("*.png")):
        artifacts.append(
            {
                "id": path.stem,
                "name": path.name,
                "caption": path.stem.replace("_", " ").title(),
                "path": str(path.relative_to(root)),
                "mediaType": "image/png",
            }
        )
    report_path = root / "report" / "report.md"
    report = None
    if report_path.is_file():
        report = {
            "name": "Lattice NDE report.md",
            "path": str(report_path.relative_to(root)),
            "mediaType": "text/markdown",
        }
    return artifacts, report


def _run_builtin(root: Path, job: dict[str, Any]) -> None:
    tiffs = [item for item in job["files"] if item["kind"] == "tiff"]
    for item in tiffs:
        count = item["pageCount"]
        indices = sorted({0, count // 2, count - 1})
        for index in indices:
            destination = root / "analysis" / f"{Path(item['name']).stem}_slice_{index}.png"
            render_tiff_slice(root / item["path"], destination, index)

    lines = [
        "# Lattice non-destructive evaluation report",
        "",
        "## Input inventory",
        "",
        "| File | Type | Details |",
        "| --- | --- | --- |",
    ]
    for item in job["files"]:
        lines.append(f"| {item['name']} | {item['kind'].upper()} | {item['summary']} |")
    lines.extend(
        [
            "",
            "## Visual analysis",
            "",
            (
                "Representative CT slices were generated for visual inspection."
                if tiffs
                else "No TIFF volume was supplied, so no CT slice images were generated."
            ),
            "",
            "STL meshes remain available in the interactive viewer. JSON inputs were "
            "validated and retained for an external agent, but are not rendered.",
            "",
            "## Workflow note",
            "",
            "This report was generated by the built-in deterministic workflow. Configure "
            "`CODEX_ANALYSIS_COMMAND` to route the same job through Codex, the MCP analysis "
            "tools, and the project NDE report skill.",
        ]
    )
    report_path = root / "report" / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def start_analysis(job_id: str) -> dict[str, Any]:
    """Validate + flip a job into 'analyzing' and start run_analysis on a
    background thread, returning immediately. Shared by the HTTP endpoint
    and the chat-callable run_initial_analysis tool, so both trigger the
    exact same backgrounded, never-blocking path. Raises ValueError (not an
    HTTP-specific exception) if the job isn't in a startable state -- each
    caller translates that to its own error type. The state check + flip
    happens atomically inside update_job so two near-simultaneous triggers
    (e.g. a double chat request) can't both pass the check.
    """

    def begin(current: dict[str, Any]) -> None:
        if current["state"] not in {"intake_ready", "failed"}:
            raise ValueError(f"Analysis is unavailable in state {current['state']!r} (needs intake_ready or failed)")
        current["state"] = "analyzing"
        current["error"] = None

    job = update_job(job_id, begin)
    threading.Thread(target=run_analysis, args=(job_id,), daemon=True).start()
    return job


def run_analysis(job_id: str) -> None:
    # Only ever invoked as start_analysis's background-thread target, which
    # has already validated the job and atomically set state="analyzing" --
    # no redundant state check/save here. (A redundant save here used to
    # race with other background threads touching the same job.json, e.g. a
    # tilt check still finishing from upload, silently reverting state.)
    job = load_job(job_id)
    root = job_dir(job_id)
    try:
        external = _run_external_harness(root)
        if not external:
            _run_builtin(root, job)
        artifacts, report = _discover_outputs(root)
        if external and not report:
            raise RuntimeError("Codex completed without creating report/report.md")

        tiff_item = next((entry for entry in job["files"] if entry["kind"] == "tiff"), None)
        design_item = next((entry for entry in job["files"] if entry["kind"] == "json"), None)
        if tiff_item and design_item:
            from .defect_detection import run_strut_defect_detection

            run_strut_defect_detection(
                job_id,
                tiff_item["path"],
                str((root / design_item["path"]).resolve()),
            )

        def finish(current: dict[str, Any]) -> None:
            current["artifacts"] = artifacts
            current["report"] = report
            current["state"] = "complete"

        update_job(job_id, finish)
    except Exception as exc:
        message = str(exc)

        def fail(current: dict[str, Any]) -> None:
            current["state"] = "failed"
            current["error"] = message

        update_job(job_id, fail)
