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

To run the analysis stage through Codex instead, set a command template before
starting the backend:

```bash
export CODEX_ANALYSIS_COMMAND='codex exec --ephemeral --sandbox workspace-write {prompt}'
export CODEX_ANALYSIS_TIMEOUT=900
python -m uvicorn app.main:app --reload
```

The command runs inside the individual job directory. Because that directory is
inside this Git repository, Codex can discover the repository configuration,
the `segmentation-tools` MCP server, and the local NDE report skill. The harness
requires Codex to place:

- PNG outputs in `analysis/`
- The report at `report/report.md`

The complete Codex event log is retained as `codex.log` inside the job directory.
Use the narrowest sandbox that permits the configured MCP workflow. Do not use a
sandbox bypass for uploaded or otherwise untrusted data.

## Chat: detection_agent / report_agent / plot_agent

Once a job's initial analysis has completed, the workbench's chat panel is
backed by an orchestrator (`app/agents/orchestrator.py`) that delegates to
three subagents, each a scoped tool-calling loop against real pipeline
output -- no subagent invents a number, every tool result traces back to a
real classification JSON, design JSON, or rendered artifact:

- **detection_agent** -- re-runs classification with modified thresholds
  (versioned, never overwrites a prior run), generates the per-defect
  validation gallery, exports 3D models.
- **report_agent** -- reads per-strut evidence, defect hotspots by unit
  cell, and measured-vs-nominal thickness to explain findings.
- **plot_agent** -- renders the chart that actually answers the question
  (verdict counts, hotspots, thickness distribution, version comparison).

This requires an OpenAI API key with function-calling access:

```bash
export OPENAI_API_KEY=sk-...
# optional, defaults to gpt-5.4 (matching .codex/agents/strut_error_detection_agent.toml):
export OPENAI_MODEL=gpt-5.4
python -m uvicorn app.main:app --reload
```

Without it, `POST /api/jobs/{id}/chat` returns `503` -- the rest of the app
(upload, inspect, run analysis, download the report) works exactly the same
either way, since the deterministic pipeline never depends on chat. Every
chat turn's full tool-call trace (per subagent) is persisted to
`job-data/<job>/chat/audit.jsonl` for audit, alongside the resumable
conversation in `chat/messages.json`.

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
