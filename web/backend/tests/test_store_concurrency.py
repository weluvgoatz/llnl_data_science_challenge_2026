"""Regression test for a real bug found while verifying the chat-triggered
analysis flow: two background threads (a per-file tilt check, and the
analysis pipeline) each did load_job() -> mutate a few fields -> save_job()
as separate steps. Whichever thread's save() ran last silently overwrote
the *other* thread's changes, because save_job() writes the whole dict from
whatever snapshot that thread loaded. Observed symptom: job.state flipped
from "analyzing" back to "intake_ready" seconds after starting, because a
tilt-check thread's stale save (loaded before analysis started) landed
after analysis's save. store.update_job() fixes this by holding one lock
across the whole read-modify-write; this test proves it under real thread
concurrency, not just by inspecting the code.
"""

from __future__ import annotations

import threading


def test_concurrent_update_job_calls_do_not_clobber_each_other(tmp_path, monkeypatch):
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    from app import store

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    job["files"] = [{"id": "f1", "kind": "tiff", "tiltStatus": "pending"}]
    store.save_job(job)

    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def set_state_analyzing() -> None:
        try:
            barrier.wait(timeout=5)

            def mutate(current: dict) -> None:
                # Simulate real work happening between load and save, which
                # is exactly the window that used to let the other thread's
                # write land first and then get overwritten.
                import time

                time.sleep(0.05)
                current["state"] = "analyzing"

            store.update_job(job["id"], mutate)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def set_tilt_status() -> None:
        try:
            barrier.wait(timeout=5)

            def mutate(current: dict) -> None:
                import time

                time.sleep(0.05)
                current["files"][0]["tiltStatus"] = "not_tilted"

            store.update_job(job["id"], mutate)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    t1 = threading.Thread(target=set_state_analyzing)
    t2 = threading.Thread(target=set_tilt_status)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors, f"unexpected errors: {errors}"

    final = store.load_job(job["id"])
    # BOTH writes must have survived -- this is the actual bug: before the
    # fix, whichever thread's save() ran last would silently discard the
    # other's field, so this would show state=="intake_ready" (its original
    # value, clobbered) or tiltStatus=="pending" (never updated), depending
    # on scheduling. With update_job, both threads' changes must be present.
    assert final["state"] == "analyzing"
    assert final["files"][0]["tiltStatus"] == "not_tilted"


def test_update_job_rejects_invalid_transition_atomically(tmp_path, monkeypatch):
    """The mutate callback can veto an update by raising -- e.g.
    start_analysis's real "already analyzing" guard -- and the job file
    must be left completely unchanged (no partial write)."""
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    from app import store

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    job["state"] = "analyzing"
    store.save_job(job)

    def mutate(current: dict) -> None:
        if current["state"] != "intake_ready":
            raise ValueError("not startable")
        current["state"] = "analyzing"

    try:
        store.update_job(job["id"], mutate)
        assert False, "expected ValueError"
    except ValueError:
        pass

    final = store.load_job(job["id"])
    assert final["state"] == "analyzing"  # unchanged, not corrupted
