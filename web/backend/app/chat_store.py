"""Per-job persistence for the chat orchestrator: the raw conversation
(so a job's chat can resume across requests) and an append-only audit log
(one record per turn, with every subagent's full tool-call trace) so any
reply can be traced back to the pipeline output it came from."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .store import job_dir

_LOCK = threading.RLock()


def _chat_dir(job_id: str) -> Path:
    directory = job_dir(job_id) / "chat"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_conversation(job_id: str) -> list[dict[str, Any]]:
    path = _chat_dir(job_id) / "messages.json"
    if not path.is_file():
        return []
    with _LOCK:
        return json.loads(path.read_text(encoding="utf-8"))


def save_conversation(job_id: str, messages: list[dict[str, Any]]) -> None:
    directory = _chat_dir(job_id)
    path = directory / "messages.json"
    temporary = directory / ".messages.json.tmp"
    with _LOCK:
        temporary.write_text(json.dumps(messages, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)


def append_audit(job_id: str, turn_record: dict[str, Any]) -> None:
    path = _chat_dir(job_id) / "audit.jsonl"
    with _LOCK, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(turn_record, default=str) + "\n")


def load_audit(job_id: str) -> list[dict[str, Any]]:
    path = _chat_dir(job_id) / "audit.jsonl"
    if not path.is_file():
        return []
    with _LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
