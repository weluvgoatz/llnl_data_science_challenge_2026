# Note on tools

For tools implemented by Daniel within here, look at https://docs.google.com/document/d/16k8c3e-Ywrf_nNyZUjGjZT0ZnSHd2YaEr3NrTz2SjWw/edit?usp=sharing

# Lattice Lens

Lattice Lens is a four-step React and FastAPI application for uploading,
inspecting, analyzing, and reporting on lattice data.

## Features

- Mixed `.json`, `.tif`, `.tiff`, and `.stl` uploads with server-side validation
- Workflow routes that unlock only after the preceding stage succeeds
- Browser-native Three.js STL inspection
- TIFF slice inspection with an interactive stack slider
- Background analysis jobs, generated PNG gallery, and artifact downloads
- Markdown NDE report preview and download
- Durable per-job state that survives browser refreshes
- Optional Codex CLI harness for the repository MCP tools and report skill

JSON files are validated and passed through to the workflow but are not rendered.

## Run locally

Use two terminals from the repository root.

```bash
cd web/backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

```bash
cd web/frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` to the backend on port 8000.

## GitHub Pages

The frontend is deployed by `.github/workflows/deploy-pages.yml`. GitHub Pages
cannot run FastAPI, Codex, or MCP servers, so the backend must be hosted
separately on a service that supports persistent Python processes and job
storage.

In the GitHub repository, add this Actions variable:

```text
VITE_API_BASE_URL=https://your-backend.example.com
```

Configure the backend to accept requests from the Pages origin:

```bash
export LATTICE_CORS_ORIGINS=https://weluvgoatz.github.io
```

Then choose **GitHub Actions** under **Settings → Pages → Build and
deployment → Source** and run the **Deploy Lattice Lens to GitHub Pages**
workflow. The frontend will be published at:

```text
https://weluvgoatz.github.io/llnl_data_science_challenge_2026/
```

## Codex harness

Without additional configuration, the backend uses a deterministic local
workflow that creates representative TIFF slice images and a Markdown report.
This makes the complete UI usable without model credentials.

To run classification, visualization, and report generation through Codex, set
a command template before starting the backend:

```bash
export CODEX_ANALYSIS_COMMAND='codex exec --ephemeral --sandbox workspace-write {prompt}'
export CODEX_ANALYSIS_TIMEOUT=900
python -m uvicorn app.main:app --reload
```

For jobs containing both a TIFF and design JSON, Run Analysis first performs
segmentation, skeletonization, and as-built graph construction deterministically.
Those stages run before the Codex timeout and are reused after a retry. Codex is
then given the validated artifacts and is prohibited from repeating
preprocessing. The command runs inside the individual job directory, where Codex
can discover the repository configuration, defect tools, and local report
skills. The harness requires Codex to place:

- PNG outputs in `analysis/`
- The report at `report/report.md`

The complete Codex event log is retained as `codex.log` inside the job directory.
Use the narrowest sandbox that permits the configured MCP workflow. Do not use a
sandbox bypass for uploaded or otherwise untrusted data.

If Codex is unavailable, times out, or returns incomplete artifacts, Run Analysis
reuses the deterministic preprocessing, falls back to deterministic
classification, and generates the basic local report. A TIFF without a design
JSON continues to use the basic local report without defect classification.

TIFF uploads use the project-scoped `tiff_tilt_correction_agent` automatically.
The parent Codex turn delegates to that agent, which calls the `segment_tiff`
MCP tool in adaptive mode before correcting both tilt axes. Override its command
or timeout when needed:

```bash
export CODEX_TILT_COMMAND='codex exec --ephemeral --sandbox workspace-write {prompt}'
export CODEX_TILT_TIMEOUT=900
```

If that Codex run is unavailable or fails validation, the backend falls back to
the same segmentation and binary tilt-correction implementation locally. Both
paths expose a segmented processed TIFF, including when no meaningful tilt is
detected. Per-upload Codex logs are stored under the job's `tilt/` directory.

## Test and build

```bash
cd web/backend
python -m pytest -q
```

```bash
cd web/frontend
npm run build
```

Job data defaults to `web/job-data/`. Override it with `LATTICE_JOB_ROOT`.
