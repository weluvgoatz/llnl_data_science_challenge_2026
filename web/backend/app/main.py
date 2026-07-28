from __future__ import annotations

import os
import re
import threading
import uuid
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

from .store import create_job, job_dir, load_job, safe_child, save_job
from .workflow import check_and_correct_tilt, inspect_upload, render_tiff_slice, run_analysis


app = FastAPI(title="Lattice Lens API", version="0.1.0")
cors_origins = [
    origin.strip()
    for origin in os.environ.get(
        "LATTICE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024


def get_job_or_404(job_id: str) -> dict:
    try:
        return load_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc


def public_job(job: dict) -> dict:
    job_id = job["id"]
    result = dict(job)
    result["files"] = [
        {
            **item,
            "contentUrl": f"/api/jobs/{job_id}/files/{item['id']}",
            "sliceUrl": (
                f"/api/jobs/{job_id}/files/{item['id']}/slice"
                if item["kind"] == "tiff"
                else None
            ),
            "correctedSliceUrl": (
                f"/api/jobs/{job_id}/files/{item['id']}/corrected-slice"
                if item.get("tiltStatus") == "corrected"
                else None
            ),
        }
        for item in job["files"]
    ]
    result["artifacts"] = [
        {**item, "downloadUrl": f"/api/jobs/{job_id}/artifacts/{item['id']}"}
        for item in job["artifacts"]
    ]
    if job["report"]:
        result["report"] = {
            **job["report"],
            "downloadUrl": f"/api/jobs/{job_id}/report?download=true",
            "previewUrl": f"/api/jobs/{job_id}/report",
        }
    defects = job.get("defects")
    if defects:
        result["defects"] = {
            **defects,
            "dataUrl": f"/api/jobs/{job_id}/defects" if defects.get("status") == "complete" else None,
        }
    return result


def stream_file(path: Path, media_type: str, filename: str | None = None) -> StreamingResponse:
    async def content():
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                yield chunk

    headers = {"Content-Length": str(path.stat().st_size)}
    if filename:
        headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(filename)}"
    return StreamingResponse(content(), media_type=media_type, headers=headers)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/jobs", status_code=201)
async def new_job() -> dict:
    return public_job(create_job())


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    return public_job(get_job_or_404(job_id))


@app.post("/api/jobs/{job_id}/files")
async def upload_files(job_id: str, files: list[UploadFile] = File(...)) -> dict:
    job = get_job_or_404(job_id)
    if job["state"] not in {"new", "intake_ready"}:
        raise HTTPException(409, "Files cannot be changed after analysis starts")
    if not files:
        raise HTTPException(400, "Choose at least one file")

    root = job_dir(job_id)
    staged: list[tuple[Path, dict]] = []
    destinations: list[Path] = []
    try:
        for upload in files:
            original = Path(upload.filename or "").name
            if not original:
                raise ValueError("Every upload needs a filename")
            safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", original)
            file_id = uuid.uuid4().hex
            destination = root / "uploads" / f"{file_id}-{safe_name}"
            destinations.append(destination)
            size = 0
            with destination.open("wb") as stream:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise ValueError(f"{original} exceeds the 2 GiB limit")
                    stream.write(chunk)
            if size == 0:
                raise ValueError(f"{original} is empty")
            metadata = inspect_upload(destination)
            record = {
                "id": file_id,
                "name": original,
                "size": size,
                "path": str(destination.relative_to(root)),
                **metadata,
            }
            if record["kind"] == "tiff":
                record["tiltStatus"] = "pending"
            staged.append((destination, record))
    except Exception as exc:
        for path in destinations:
            path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc

    job["files"].extend(record for _, record in staged)
    job["state"] = "intake_ready"
    job["error"] = None
    save_job(job)
    for _, record in staged:
        if record["kind"] == "tiff":
            threading.Thread(
                target=check_and_correct_tilt,
                args=(job_id, record["id"]),
                daemon=True,
            ).start()
    return public_job(job)


@app.get("/api/jobs/{job_id}/files/{file_id}")
async def get_file(job_id: str, file_id: str) -> StreamingResponse:
    job = get_job_or_404(job_id)
    item = next((entry for entry in job["files"] if entry["id"] == file_id), None)
    if not item:
        raise HTTPException(404, "File not found")
    media = {
        "stl": "model/stl",
        "tiff": "image/tiff",
        "json": "application/json",
    }[item["kind"]]
    return stream_file(safe_child(job_id, item["path"]), media, item["name"])


@app.get("/api/jobs/{job_id}/files/{file_id}/slice")
async def get_slice(job_id: str, file_id: str, index: int = 0) -> StreamingResponse:
    job = get_job_or_404(job_id)
    item = next((entry for entry in job["files"] if entry["id"] == file_id), None)
    if not item or item["kind"] != "tiff":
        raise HTTPException(404, "TIFF file not found")
    if index < 0 or index >= item["pageCount"]:
        raise HTTPException(400, "Slice index is out of range")
    preview = job_dir(job_id) / "previews" / f"{file_id}-{index}.png"
    if not preview.is_file():
        render_tiff_slice(safe_child(job_id, item["path"]), preview, index)
    return stream_file(preview, "image/png")


@app.get("/api/jobs/{job_id}/files/{file_id}/corrected-slice")
async def get_corrected_slice(job_id: str, file_id: str, index: int = 0) -> StreamingResponse:
    job = get_job_or_404(job_id)
    item = next((entry for entry in job["files"] if entry["id"] == file_id), None)
    if not item or item.get("tiltStatus") != "corrected" or not item.get("correctedPath"):
        raise HTTPException(404, "Tilt-corrected TIFF not available")
    if index < 0 or index >= item["pageCount"]:
        raise HTTPException(400, "Slice index is out of range")
    preview = job_dir(job_id) / "previews" / f"{file_id}-corrected-{index}.png"
    if not preview.is_file():
        render_tiff_slice(safe_child(job_id, item["correctedPath"]), preview, index)
    return stream_file(preview, "image/png")


@app.post("/api/jobs/{job_id}/analysis", status_code=202)
async def analyze(job_id: str) -> dict:
    job = get_job_or_404(job_id)
    if job["state"] not in {"intake_ready", "failed"}:
        raise HTTPException(409, "Analysis is unavailable in the current state")
    job["state"] = "analyzing"
    job["error"] = None
    save_job(job)
    threading.Thread(target=run_analysis, args=(job_id,), daemon=True).start()
    return public_job(job)


@app.get("/api/jobs/{job_id}/artifacts/{artifact_id}")
async def artifact(job_id: str, artifact_id: str) -> StreamingResponse:
    job = get_job_or_404(job_id)
    item = next((entry for entry in job["artifacts"] if entry["id"] == artifact_id), None)
    if not item:
        raise HTTPException(404, "Artifact not found")
    return stream_file(
        safe_child(job_id, item["path"]),
        item["mediaType"],
        item["name"],
    )


@app.get("/api/jobs/{job_id}/defects")
async def get_defects(job_id: str) -> StreamingResponse:
    job = get_job_or_404(job_id)
    defects = job.get("defects")
    if not defects or defects.get("status") != "complete" or not defects.get("path"):
        raise HTTPException(404, "Defect classification not available")
    return stream_file(safe_child(job_id, defects["path"]), "application/json")


@app.get("/api/jobs/{job_id}/report")
async def report(job_id: str, download: bool = False):
    job = get_job_or_404(job_id)
    if not job["report"]:
        raise HTTPException(404, "Report not available")
    path = safe_child(job_id, job["report"]["path"])
    if download:
        return stream_file(path, "text/markdown", job["report"]["name"])
    return PlainTextResponse(path.read_text(encoding="utf-8"), media_type="text/markdown")
