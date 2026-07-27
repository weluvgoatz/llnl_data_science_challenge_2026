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
