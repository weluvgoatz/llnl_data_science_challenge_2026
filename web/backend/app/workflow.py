from __future__ import annotations

import json
import os
import shlex
import struct
import subprocess
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

from .store import job_dir, load_job, save_job


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


def run_analysis(job_id: str) -> None:
    job = load_job(job_id)
    if job["state"] not in {"intake_ready", "analyzing", "failed"}:
        return
    job["state"] = "analyzing"
    job["error"] = None
    save_job(job)
    root = job_dir(job_id)
    try:
        external = _run_external_harness(root)
        if not external:
            _run_builtin(root, job)
        artifacts, report = _discover_outputs(root)
        if external and not report:
            raise RuntimeError("Codex completed without creating report/report.md")
        job = load_job(job_id)
        job["artifacts"] = artifacts
        job["report"] = report
        job["state"] = "complete"
        save_job(job)
    except Exception as exc:
        job = load_job(job_id)
        job["state"] = "failed"
        job["error"] = str(exc)
        save_job(job)
