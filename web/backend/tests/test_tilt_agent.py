from __future__ import annotations

import json
import subprocess

import numpy as np
import tifffile


def _report(zy: float = 0.25, zx: float = -0.5) -> dict:
    return {
        "input_path": "/tmp/segmented.tif",
        "output_path": "/tmp/corrected.tif",
        "shape": [3, 4, 4],
        "dtype": "uint8",
        "estimated_zy_degrees": zy,
        "estimated_zx_degrees": zx,
        "applied_zy_degrees": zy,
        "applied_zx_degrees": zx,
        "residual_zy_degrees": 0.01,
        "residual_zx_degrees": -0.02,
        "middle_fraction": 1 / 3,
        "voxel_spacing": [1.0, 1.0, 1.0],
    }


def _write_source(path) -> None:
    tifffile.imwrite(
        path,
        np.arange(48, dtype=np.uint16).reshape(3, 4, 4),
        photometric="minisblack",
    )


def _write_binary(path) -> None:
    volume = np.zeros((3, 4, 4), dtype=np.uint8)
    volume[:, 1:3, 1:3] = 255
    tifffile.imwrite(path, volume, photometric="minisblack")


def test_tilt_agent_prompt_and_artifact_validation(tmp_path, monkeypatch):
    from app import workflow

    source = tmp_path / "source with spaces.tif"
    segmented = tmp_path / "tilt" / "segmented.tif"
    corrected = tmp_path / "tilt" / "corrected.tif"
    report_path = tmp_path / "tilt" / "report.json"
    log_path = tmp_path / "tilt" / "codex.log"
    _write_source(source)
    monkeypatch.setenv("CODEX_TILT_COMMAND", "fake-codex {prompt}")
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        segmented.parent.mkdir(parents=True, exist_ok=True)
        _write_binary(segmented)
        _write_binary(corrected)
        report_path.write_text(json.dumps(_report()), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, "agent complete", "")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    result = workflow._run_tilt_agent(
        tmp_path,
        source,
        segmented,
        corrected,
        report_path,
        log_path,
    )

    prompt = captured["argv"][1]
    assert "Spawn the custom `tiff_tilt_correction_agent` subagent" in prompt
    assert "segment_tiff" in prompt
    assert "adaptive=true" in prompt
    assert "Correct both Z-Y and Z-X" in prompt
    assert str(source.resolve()) in prompt
    assert captured["kwargs"]["cwd"] == tmp_path
    assert result["estimated_zx_degrees"] == -0.5
    assert "agent complete" in log_path.read_text(encoding="utf-8")


def test_check_and_correct_tilt_falls_back_and_exposes_level_segmentation(
    tmp_path,
    monkeypatch,
):
    from app import store, workflow
    from app.main import public_job

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    root = store.job_dir(job["id"])
    source = root / "uploads" / "scan.tif"
    _write_source(source)
    job["files"].append(
        {
            "id": "scan-id",
            "name": "scan.tif",
            "size": source.stat().st_size,
            "path": "uploads/scan.tif",
            "kind": "tiff",
            "pageCount": 3,
            "width": 4,
            "height": 4,
            "summary": "3 slices",
            "tiltStatus": "pending",
        }
    )
    store.save_job(job)

    def failed_agent(*args, **kwargs):
        raise RuntimeError("Codex unavailable")

    def successful_fallback(source_path, segmented, corrected, report_path):
        segmented.parent.mkdir(parents=True, exist_ok=True)
        _write_binary(segmented)
        _write_binary(corrected)
        report = _report(zy=0.05, zx=-0.04)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr(workflow, "_run_tilt_agent", failed_agent)
    monkeypatch.setattr(
        workflow,
        "_run_deterministic_tilt_fallback",
        successful_fallback,
    )
    workflow.check_and_correct_tilt(job["id"], "scan-id")

    finished = store.load_job(job["id"])
    item = finished["files"][0]
    assert item["tiltStatus"] == "not_tilted"
    assert item["tiltZY"] == 0.05
    assert item["tiltZX"] == -0.04
    assert item["correctedPath"] == "tilt/scan-id-corrected.tif"
    assert item["tiltError"] is None
    assert not (root / "tilt" / "scan-id-segmented.tif").exists()
    assert public_job(finished)["files"][0]["correctedSliceUrl"]


def test_check_and_correct_tilt_reports_agent_and_fallback_failures(
    tmp_path,
    monkeypatch,
):
    from app import store, workflow

    store.DATA_ROOT = tmp_path
    job = store.create_job()
    root = store.job_dir(job["id"])
    source = root / "uploads" / "scan.tif"
    _write_source(source)
    job["files"].append(
        {
            "id": "scan-id",
            "name": "scan.tif",
            "size": source.stat().st_size,
            "path": "uploads/scan.tif",
            "kind": "tiff",
            "pageCount": 3,
            "width": 4,
            "height": 4,
            "summary": "3 slices",
            "tiltStatus": "pending",
        }
    )
    store.save_job(job)

    monkeypatch.setattr(
        workflow,
        "_run_tilt_agent",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("agent boom")),
    )
    monkeypatch.setattr(
        workflow,
        "_run_deterministic_tilt_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fallback boom")),
    )
    workflow.check_and_correct_tilt(job["id"], "scan-id")

    item = store.load_job(job["id"])["files"][0]
    assert item["tiltStatus"] == "failed"
    assert "agent boom" in item["tiltError"]
    assert "fallback boom" in item["tiltError"]
    assert "correctedPath" not in item
