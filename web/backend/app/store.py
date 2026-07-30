from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


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


def update_job(job_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Atomically load -> mutate -> save one job, holding the lock across the
    whole read-modify-write.

    Several background threads touch the same job concurrently in this app
    (a per-file tilt check, the analysis pipeline, its defect-detection
    stage updates) -- calling load_job() and save_job() as two separate
    steps leaves a window where thread A's save can silently clobber fields
    thread B already wrote, because each save writes the *whole* dict from
    whatever snapshot it loaded. `mutate` receives the freshly loaded dict
    and edits it in place; the return value is the saved job.
    """
    root = job_dir(job_id)
    path = root / "job.json"
    with _LOCK:
        if not path.is_file():
            raise KeyError(job_id)
        job = json.loads(path.read_text(encoding="utf-8"))
        mutate(job)
        job["updatedAt"] = now()
        temporary = root / ".job.json.tmp"
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)
        return job


def safe_child(job_id: str, relative_path: str) -> Path:
    root = job_dir(job_id).resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents:
        raise ValueError("Invalid artifact path")
    return candidate
