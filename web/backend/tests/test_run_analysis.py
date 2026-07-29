from __future__ import annotations

import subprocess
from pathlib import Path

from app.defect_detection import PreparedDefectInputs


def _prepared(root: Path) -> PreparedDefectInputs:
    stack = root / "defects" / "tif_stacks"
    stack.mkdir(parents=True, exist_ok=True)
    base = "scan"
    return PreparedDefectInputs(
        base=base,
        stack_dir=stack,
        raw=stack / f"{base}.tif",
        design=root / "uploads" / "design.json",
        mask=stack / f"{base}_segmented_clean.tif",
        skeleton=stack / f"{base}_segmented_clean.skelcoords.npz",
        graph=stack / f"{base}_segmented_clean_asbuilt_graph_cleaned.json",
        classifier_graph=stack / f"{base}_classifier_graph.npz",
        classification=stack / f"{base}_unified_defects_accurate.json",
        bend_detail=stack / f"{base}_asbuilt_bent.json",
    )


def _job(root: Path) -> dict:
    uploads = root / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "scan.tif").touch()
    (uploads / "design.json").write_text("{}", encoding="utf-8")
    return {
        "files": [
            {
                "kind": "tiff",
                "path": "uploads/scan.tif",
                "name": "scan.tif",
                "pageCount": 1,
                "summary": "scan",
            },
            {
                "kind": "json",
                "path": "uploads/design.json",
                "name": "design.json",
                "summary": "design",
            },
        ]
    }


def test_codex_phase_receives_prepared_artifacts_and_forbids_preprocessing(
    tmp_path, monkeypatch
):
    from app import workflow

    prepared = _prepared(tmp_path)
    job = _job(tmp_path)
    monkeypatch.setenv("CODEX_ANALYSIS_COMMAND", "fake-codex {prompt}")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, "complete", "")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    assert workflow._run_external_harness(tmp_path, job, prepared)

    prompt = captured["argv"][1]
    assert "Deterministic preprocessing is already complete" in prompt
    assert "Do not call segment_tiff, skeletonize, build_asbuilt_graph" in prompt
    assert "Do not spawn the self-contained strut_error_detection_agent" in prompt
    assert str(prepared.mask) in prompt
    assert str(prepared.skeleton) in prompt
    assert str(prepared.graph) in prompt
    assert str(prepared.classifier_graph) in prompt
    assert captured["env"]["LATTICE_STK"] == str(prepared.stack_dir)
    assert captured["env"]["LATTICE_PREBUILT_GRAPH"] == str(
        prepared.classifier_graph
    )


def test_codex_timeout_preserves_partial_log(tmp_path, monkeypatch):
    from app import workflow

    prepared = _prepared(tmp_path)
    job = _job(tmp_path)
    monkeypatch.setenv("CODEX_ANALYSIS_COMMAND", "fake-codex {prompt}")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(
            args[0], 900, output="partial progress", stderr="still working"
        )

    monkeypatch.setattr(workflow.subprocess, "run", timeout)
    try:
        workflow._run_external_harness(tmp_path, job, prepared)
    except RuntimeError as exc:
        assert "timed out" in str(exc)
    else:
        raise AssertionError("expected timeout")

    log = (tmp_path / "codex.log").read_text(encoding="utf-8")
    assert "partial progress" in log
    assert "still working" in log
    assert "Timed out" in log


def test_run_analysis_prepares_before_codex_and_falls_back(
    tmp_path, monkeypatch
):
    from app import defect_detection, store, workflow

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    root = store.job_dir(job["id"])
    job.update(_job(root))
    job["state"] = "intake_ready"
    store.save_job(job)
    prepared = _prepared(root)
    calls = []

    def prepare(*args):
        calls.append("prepare")
        return prepared

    def codex(*args):
        calls.append("codex")
        raise RuntimeError("agent unavailable")

    def classify(*args):
        calls.append("fallback-classification")

    def builtin(root_path, current_job):
        calls.append("builtin-report")
        (root_path / "analysis" / "fallback.png").write_bytes(b"png")
        (root_path / "report" / "report.md").write_text(
            "# fallback\n", encoding="utf-8"
        )

    monkeypatch.setattr(defect_detection, "prepare_defect_inputs", prepare)
    monkeypatch.setattr(
        defect_detection, "run_deterministic_classification", classify
    )
    monkeypatch.setattr(workflow, "_run_external_harness", codex)
    monkeypatch.setattr(workflow, "_run_builtin", builtin)

    workflow.run_analysis(job["id"])

    finished = store.load_job(job["id"])
    assert calls == [
        "prepare",
        "codex",
        "fallback-classification",
        "builtin-report",
    ]
    assert finished["state"] == "complete"
    assert finished["defects"]["codexFallback"] is True
    assert "agent unavailable" in finished["defects"]["codexError"]


def test_run_analysis_uses_codex_for_tiff_without_design(tmp_path, monkeypatch):
    from app import store, workflow

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    root = store.job_dir(job["id"])
    uploads = root / "uploads"
    (uploads / "scan.tif").touch()
    job["files"] = [
        {
            "kind": "tiff",
            "path": "uploads/scan.tif",
            "name": "scan.tif",
            "pageCount": 1,
            "summary": "scan",
        }
    ]
    job["state"] = "intake_ready"
    store.save_job(job)
    calls = []

    def codex(root_path, current_job, prepared=None):
        calls.append("codex")
        (root_path / "analysis" / "codex.png").write_bytes(b"png")
        (root_path / "report" / "report.md").write_text(
            "# Codex analysis\n", encoding="utf-8"
        )
        return True

    def builtin(*args):
        calls.append("builtin")

    monkeypatch.setattr(workflow, "_run_external_harness", codex)
    monkeypatch.setattr(workflow, "_run_builtin", builtin)

    workflow.run_analysis(job["id"])

    finished = store.load_job(job["id"])
    assert calls == ["codex"]
    assert finished["state"] == "complete"
    assert finished["report"] is not None
    assert [item["name"] for item in finished["artifacts"]] == ["codex.png"]
