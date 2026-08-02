# Lattice Lens

Lattice Lens is a chat-driven inspection tool for X-ray CT scans of
additively-manufactured lattice structures. Upload a CT scan (`.tif`), its
design graph (`.json`), and optionally its CAD mesh (`.stl`); the app
segments and skeletonizes the scan, classifies every designed strut as
**present / missing / bent / thin / disconnected**, and lets you explore the
result — as a 3D defect view, charts, or a generated NDE report — through
either the UI directly or a chat panel backed by an LLM orchestrator.

This README covers the web app specifically (`web/`). The repository also
contains a separate, older Codex-CLI-based agentic layer (MCP servers +
skills, operating on the same underlying scan data) — see
[`AGENTS.md`](../AGENTS.md) at the repo root for that. The two don't share a
runtime; this doc is about the product in `web/`.

---

## 1. Clone

```bash
git clone https://github.com/weluvgoatz/llnl_data_science_challenge_2026.git
cd llnl_data_science_challenge_2026
git checkout agentic-system   # the active development branch (this app);
                               # main is an older, simpler, non-chat version
```

`data/` is stored in **Git LFS** (see `.gitattributes`) — install
[`git-lfs`](https://git-lfs.com/) before cloning, or run `git lfs pull`
afterward, or the sample CT/design-JSON files under `data/` will be
pointer stubs instead of real data.

---

## 2. Repository layout

```
llnl_data_science_challenge_2026/
├── AGENTS.md                    Codex-CLI agentic architecture (separate from web/)
├── README.md                    LLNL's own Data Science Challenge instructions
├── requirements.txt              deps for the Codex-side MCP servers (src/)
│
├── .codex/agents/                Codex subagent config (.toml)
├── .agents/skills/                Codex skills (SKILL.md + scripts/)
├── src/                          Codex-facing MCP servers (FastMCP)
│   ├── mcp_server.py                  segment_tiff, skeletonize, segment_ct_dataset, ...
│   ├── defect_mcp_server.py           classify_lattice_defects, summarize_lattice_defects, ...
│   ├── defect_report_agent.py         reconstructed report-building library (also used by web/)
│   └── skeletonization.py
│
├── analysis/                     the detection algorithms (importable, not web-specific)
│   ├── defect_detection/
│   │   ├── config.py                   central paths/constants, env-overridable
│   │   ├── unified_defects_accurate.py the ORIGINAL (v1) fixed-fraction classifier -- superseded
│   │   ├── v2/                         detect_v2.py: the CURRENT self-contained detector
│   │   │   ├── segment.py                    global-Otsu segmentation
│   │   │   ├── detect_v2.py                  segment -> skeletonize -> graph -> classify, one script
│   │   │   ├── run_pipeline.py               CLI entry point (see "Run the detector standalone" below)
│   │   │   ├── export_3d.py                  colour-coded PLY export
│   │   │   ├── export_defect_list.py         defective_struts.json/.csv, the downstream contract
│   │   │   ├── capture_defect_samples.py     zoomed raw-CT sample images per defect class
│   │   │   └── INTEGRATION.md                the v2 output contract, read this first
│   │   ├── agents/                     defect_results_reporting_agent_v2.py / _v3.py --
│   │   │                               spatial-concentration + exterior-face/volume analysis,
│   │   │                               reconstructed to back web/backend/app/report_tools.py
│   │   ├── defect_results_reporting_agent.py   the report-agent entry point (DefectResultsReportingAgent)
│   │   └── sample_output/              a real example classification JSON, for local testing
│   ├── clean_segmentation.py, segment_to_tif.py    v1 pipeline stages (pre-v2)
│   └── tiff_tilt.py etc. do NOT live here -- see web/analysis/ below
│
├── data/                         sample CT scans / design graphs / STLs (Git LFS)
│   └── missing_struts/                 the primary challenge specimen (tif_stacks/, registered_jsons/, stls/)
│
├── evals/, presentation/, images/      challenge material, not part of the running app
│
└── web/                          <-- THE WEB APP (this README's subject)
    ├── analysis/                       tilt-correction helpers used only by the backend
    │   └── tiff_tilt.py, tiff_rotation.py, tiff_skeletonization.py
    │       NOTE: this is a DIFFERENT directory from the repo-root analysis/ above --
    │       same name, unrelated content (tilt/rotation vs. defect detection). Don't confuse them.
    ├── src/                            a second, web-specific copy of the MCP-style segmentation
    │   │                               helpers (mcp_server.py, defect_mcp_server.py, skeletonization.py) --
    │   │                               kept import-compatible with the repo-root src/, used by
    │   │                               older code paths; the current chat pipeline (detect_v2.py)
    │   │                               doesn't depend on this.
    ├── backend/                        FastAPI app
    │   ├── requirements.txt
    │   ├── app/
    │   │   ├── main.py                     HTTP API (FastAPI routes)
    │   │   ├── store.py                    job state persistence (job-data/<job_id>/job.json)
    │   │   ├── workflow.py                 upload intake, tilt-check, the automatic per-analysis report
    │   │   ├── defect_detection.py         wires detect_v2.py into a job; v1<->v2 schema adapter
    │   │   ├── agent_tools.py              evidence/versioning tools (explain_strut, rerun_classification, ...)
    │   │   ├── plot_tools.py               matplotlib chart tools
    │   │   ├── report_tools.py             generate_report_data / finalize_report (the NDE report)
    │   │   ├── chat_store.py               conversation + audit-log persistence
    │   │   └── agents/                     the chat orchestrator
    │   │       ├── orchestrator.py               top-level: delegates to subagents, controls the UI surface
    │   │       ├── subagents.py                  detection_agent / report_agent / plot_agent definitions
    │   │       ├── runtime.py                    the shared OpenAI tool-calling loop
    │   │       └── surfaces.py                   mount_surface/unmount_surface validation
    │   └── tests/                          pytest suite (schema-fixture-based, no OpenAI key needed)
    └── frontend/                       React + Vite + TypeScript
        ├── package.json
        ├── vite.config.ts                  dev-server proxy: /api -> http://127.0.0.1:8000
        └── src/
            ├── App.tsx                         top-level layout, file list, surface mounting
            ├── api.ts                          typed fetch wrappers for the backend
            ├── types.ts                        shared TS types (mirrors backend JSON shapes)
            ├── defects.ts                      verdict colours/labels/descriptions
            └── components/
                ├── ChatPanel.tsx                    the chat UI + reasoning-trace viewer
                ├── DefectViewer.tsx                 3D classified-lattice view (Three.js)
                ├── DesignGraphViewer.tsx            design-JSON graph view
                ├── StlViewer.tsx                    STL mesh view (Three.js)
                └── ZoomableImage.tsx                 lightbox for PNG artifacts/galleries
```

---

## 3. Running it

### Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.11+ | backend |
| Node.js | 20+ | frontend |
| `git-lfs` | any | `data/` sample files |
| An OpenAI API key | — | chat panel only (see below); everything else works without it |

Set up a Python environment once (either works):

```bash
# conda (matches the challenge's own setup)
conda create -n dssi_env python=3.11 -y
conda activate dssi_env

# -- or a plain venv --
python3 -m venv .venv && source .venv/bin/activate
```

### 3.1 Backend

```bash
cd web/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Runs from `web/backend` as the working directory. On startup it needs no
external services except (optionally) the OpenAI API — job data, uploaded
files, generated plots/reports, and caches all live on local disk under
`web/job-data/` by default (override with `LATTICE_JOB_ROOT`).

### 3.2 Frontend

In a second terminal:

```bash
cd web/frontend
npm install
npm run dev
```

Open **http://localhost:5173** — Vite proxies `/api/*` requests to
`http://127.0.0.1:8000` (the backend above), configured in
`web/frontend/vite.config.ts`.

### 3.3 Enabling the chat panel (detection_agent / report_agent / plot_agent)

Without an API key, the app is still fully usable — upload files, view the
TIFF/STL/design graph, run analysis (the real `detect_v2.py` pipeline runs
either way), view the classified 3D lattice, and download the automatic
per-analysis report. Only the **chat panel** needs an LLM:

```bash
export OPENAI_API_KEY=sk-...
# optional, defaults to gpt-5.4:
export OPENAI_MODEL=gpt-5.4
```

Set this before starting `uvicorn`. Without it, `POST /api/jobs/{id}/chat`
returns `503` and the chat panel is disabled client-side; nothing else is
affected.

Once running, asking the chat panel to "run the analysis," "show me the
defect view," "generate the report," or "plot the defect hotspots" delegates
to `detection_agent` / `report_agent` / `plot_agent` respectively — see
[`web/backend/app/agents/subagents.py`](backend/app/agents/subagents.py) for
each one's exact tool roster and system prompt.

### 3.4 Run the detector standalone (no web app at all)

If you just want the classification JSON for a scan, without the API/UI:

```bash
export LATTICE_BASE="my_scan"                          # file name, no extension
export LATTICE_STK="data/my_scan/tif_stacks"           # dir holding my_scan.tif
export LATTICE_DESIGN_JSON="data/my_scan/design.json"  # registered design graph

cd analysis/defect_detection/v2
python run_pipeline.py            # detect + defect list (default)
python run_pipeline.py --viz      # + the 3D colour-coded model
python run_pipeline.py --all      # + validation panels + samples + 3D model
```

First run self-segments and skeletonizes (several minutes on a ~1 GB scan,
several GB of RAM); both are cached (`<base>_segmented_clean_v2.tif` /
`.skelcoords.npz`) so later runs are fast. With no env vars set, it runs
against the challenge specimen already in `data/`. See
[`analysis/defect_detection/v2/INTEGRATION.md`](../analysis/defect_detection/v2/INTEGRATION.md)
for the exact output contract.

---

## 4. Environment variables reference

| Variable | Default | Effect |
|---|---|---|
| `OPENAI_API_KEY` | unset | required for the chat panel; everything else works without it |
| `OPENAI_MODEL` | `gpt-5.4` | model used by the orchestrator and all three subagents |
| `LATTICE_JOB_ROOT` | `web/job-data/` | where per-job uploads/results/state are stored |
| `LATTICE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated allowed origins; set this when the frontend is hosted separately from the backend (e.g. GitHub Pages) |
| `LATTICE_BASE`, `LATTICE_STK`, `LATTICE_DESIGN_JSON` | challenge specimen | point the standalone CLI detector (3.4) or the Codex/MCP tools at a different scan |
| `LATTICE_VOXEL_UM`, `LATTICE_STRUT_DIAMETER_UM` | `58.1` / `424.0` | physical calibration, only if your scan's specimen differs |
| `CODEX_ANALYSIS_COMMAND` | unset | opt-in: route the *automatic per-upload* analysis stage through a Codex CLI invocation instead of the deterministic built-in workflow (see `workflow.py`); unrelated to the chat panel |

---

## 5. Testing

```bash
cd web/backend
python -m pytest -q          # 11 tests, fixture-based, no OpenAI key or real CT data needed
```

```bash
cd web/frontend
npx tsc -b                   # typecheck
npm run build                # production build (also typechecks)
```

---

## 6. Deployment

The frontend deploys to **GitHub Pages** via
[`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml)
(triggers on push to `main`); the backend is meant to run on a host that
supports a persistent Python process — GitHub Pages cannot run FastAPI. A
Render blueprint for the backend exists at `render.yaml` **on `main`**
(not on `agentic-system` — the two branches' deploy setups have diverged;
check `main` before changing deployment config).

**Currently live at https://weluvgoatz.github.io/llnl_data_science_challenge_2026/
is a build of `main`, not `agentic-system`** — `main` is an earlier,
simpler version of this app with no chat panel, no `detect_v2.py`, and no
report agent. Getting this branch's work live requires reconciling it into
`main` first; see `git log --oneline main..agentic-system` for the delta.
