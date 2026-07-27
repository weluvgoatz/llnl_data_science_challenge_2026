from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATA_ROOT = Path(
    os.environ.get(
        "LATTICE_JOB_ROOT",
        Path(__file__).resolve().parents[2] / "job-data",
    )
).resolve()
_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id: str) -> Path:
    try:
        uuid.UUID(job_id)
    except ValueError as exc:
        raise KeyError(job_id) from exc
    return DATA_ROOT / job_id


def create_job() -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    root = job_dir(job_id)
    for name in ("uploads", "analysis", "report", "previews"):
        (root / name).mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "state": "new",
        "createdAt": now(),
        "updatedAt": now(),
        "files": [],
        "artifacts": [],
        "report": None,
        "error": None,
    }
    save_job(job)
    return job


def load_job(job_id: str) -> dict[str, Any]:
    path = job_dir(job_id) / "job.json"
    if not path.is_file():
        raise KeyError(job_id)
    with _LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict[str, Any]) -> None:
    root = job_dir(job["id"])
    root.mkdir(parents=True, exist_ok=True)
    job["updatedAt"] = now()
    path = root / "job.json"
    temporary = root / ".job.json.tmp"
    with _LOCK:
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)


def safe_child(job_id: str, relative_path: str) -> Path:
    root = job_dir(job_id).resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents:
        raise ValueError("Invalid artifact path")
    return candidate
