import json
import asyncio

import httpx
import numpy as np
import pytest
import tifffile


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_upload_gating_and_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    monkeypatch.delenv("CODEX_ANALYSIS_COMMAND", raising=False)
    from app import store

    store.DATA_ROOT = tmp_path
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        job = (await client.post("/api/jobs")).json()
        assert job["state"] == "new"

        response = await client.post(
            f"/api/jobs/{job['id']}/files",
            files=[("files", ("model.json", json.dumps({"nodes": []}), "application/json"))],
        )
        assert response.status_code == 200
        assert response.json()["state"] == "intake_ready"

        response = await client.post(f"/api/jobs/{job['id']}/analysis")
        assert response.status_code == 202
        complete = response.json()
        for _ in range(100):
            complete = (await client.get(f"/api/jobs/{job['id']}")).json()
            if complete["state"] != "analyzing":
                break
            await asyncio.sleep(0.02)
        assert complete["state"] == "complete"
        assert complete["report"] is not None


@pytest.mark.anyio
async def test_tiff_slice_and_invalid_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("LATTICE_JOB_ROOT", str(tmp_path))
    monkeypatch.setenv("CODEX_TILT_COMMAND", "")
    from app import store

    store.DATA_ROOT = tmp_path
    from app.main import app

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        job = (await client.post("/api/jobs")).json()
        tif_path = tmp_path / "tiny.tif"
        tifffile.imwrite(
            tif_path,
            np.arange(48, dtype=np.uint16).reshape(3, 4, 4),
            photometric="minisblack",
        )
        with tif_path.open("rb") as source:
            response = await client.post(
                f"/api/jobs/{job['id']}/files",
                files=[("files", ("tiny.tif", source, "image/tiff"))],
            )
        item = response.json()["files"][0]
        preview = await client.get(
            f"/api/jobs/{job['id']}/files/{item['id']}/slice?index=1"
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"

        other = (await client.post("/api/jobs")).json()
        bad = await client.post(
            f"/api/jobs/{other['id']}/files",
            files=[("files", ("notes.txt", b"nope", "text/plain"))],
        )
        assert bad.status_code == 400


@pytest.mark.anyio
async def test_health_reports_limits_and_cors(tmp_path, monkeypatch):
    from app import main, store

    store.DATA_ROOT = tmp_path
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(main, "MAX_TIFF_EXPANDED_BYTES", 64 * 1024 * 1024)
    monkeypatch.setattr(main, "STORAGE_MODE", "ephemeral")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/health",
            headers={"Origin": "https://weluvgoatz.github.io"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "maxUploadBytes": 64 * 1024 * 1024,
        "maxTiffExpandedBytes": 64 * 1024 * 1024,
        "storage": "ephemeral",
    }
    assert response.headers["access-control-allow-origin"] == "https://weluvgoatz.github.io"


@pytest.mark.anyio
async def test_upload_and_expanded_tiff_limits(tmp_path, monkeypatch):
    from app import main, store

    store.DATA_ROOT = tmp_path
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 8)
    monkeypatch.setattr(main, "MAX_TIFF_EXPANDED_BYTES", 16)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=main.app), base_url="http://test"
    ) as client:
        job = (await client.post("/api/jobs")).json()
        response = await client.post(
            f"/api/jobs/{job['id']}/files",
            files=[("files", ("large.json", b'{"nodes":[]}', "application/json"))],
        )
        assert response.status_code == 413
        assert "upload limit" in response.json()["detail"]

        monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024 * 1024)
        tiff_path = tmp_path / "expanded.tif"
        tifffile.imwrite(
            tiff_path,
            np.zeros((3, 4, 4), dtype=np.uint16),
            photometric="minisblack",
        )
        with tiff_path.open("rb") as source:
            response = await client.post(
                f"/api/jobs/{job['id']}/files",
                files=[("files", ("expanded.tif", source, "image/tiff"))],
            )

    assert response.status_code == 413
    assert "expands to" in response.json()["detail"]
