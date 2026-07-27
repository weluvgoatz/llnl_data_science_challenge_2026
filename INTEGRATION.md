# Integration guide — Strut Defect Detection module

This is the **strut error detection + validation + stats** piece of the team's
agentic workflow. If you're building another part (data loader, dashboard,
research agent), this tells you how to plug into it.

**One-line contract:** give it a **TIFF** (+ the defect-free **design JSON**) →
get back a **classification JSON** where every strut is labelled
`present | missing | bent | thin | disconnected`. Everything downstream just
reads that JSON.

---

## 1. Setup (once)
```bash
pip install -r requirements.txt
```
Register the two MCP servers in your Codex config (`~/.codex/config.toml`):
```toml
[mcp_servers.ct_segmentation]
command = "python"
args = ["src/mcp_server.py"]

[mcp_servers.lattice_defects]
command = "python"
args = ["src/defect_mcp_server.py"]
```

## 2. Point it at your scan (no code edits)
```bash
export LATTICE_BASE="my_scan"                 # file name, no extension
export LATTICE_STK="data/my_scan/tif_stacks"  # folder with the .tif + outputs
export LATTICE_DESIGN_JSON="data/my_scan/design_graph.json"
```
Defaults reproduce the challenge specimen, so with no env vars it "just runs".

## 3. Run it — pick one
- **Easiest:** invoke the subagent `strut_error_detection_agent` — this IS the
  "Strut Error Detection" box; it does everything (segment → skeletonize → graph →
  classify with evidence + confidence → report).
- **Tool by tool** (MCP): `build_asbuilt_graph` → `classify_lattice_defects` →
  `detect_bent_struts` → `summarize_lattice_defects`.
- **CLI:** run the scripts in `analysis/defect_detection/` in that order.

## 4. What you get back (the hand-off format)
`<base>_unified_defects_accurate.json`:
```json
{
  "struts": [
    { "p0": [x, y, z], "p1": [x, y, z], "verdict": "present",
      "confidence": 0.94, "evidence": { "rule": "as-built strut exists ...",
      "bow_um": 61, "bent_threshold_um": 212, "density": 51200, "thin_cutoff": 39055 } },
    { "p0": [x, y, z], "p1": [x, y, z], "verdict": "missing",
      "confidence": 0.71, "evidence": { "metal_fraction": 0.043, "missing_threshold": 0.15 } }
  ],
  "meta": {
    "counts": { "present": 17253, "missing": 410, "bent": 365,
                "thin": 287, "disconnected": 153 },
    "n": 18468,
    "volume_shape_zyx": [761, 815, 837]
  }
}
```
- `p0`, `p1` = the strut's two joint positions in **[x, y, z] scan voxels**.
- `verdict` = one of `present | missing | bent | thin | disconnected`.
- `confidence` = 0–1 (how decisively the measurement cleared its threshold);
  the Validation agent should review anything below ~0.2.
- `evidence` = the raw measured value(s) and threshold behind the verdict.
- Bend detail (bow in µm, tortuosity) is in `<base>_asbuilt_bent.json`.

**Downstream agents (Statistical Analysis, Dashboard) only need this JSON.**
Call `summarize_lattice_defects(<path>)` for the counts/percentages table.

## 5. The decision rules (so results are explainable)
Call the `describe_thresholds` tool, or:
| Verdict | Rule |
| :--- | :--- |
| Missing | metal along the strut < 15% (or a joint sits on no metal) |
| Disconnected (broken) | one continuous gap ≥ 25% of the strut length |
| Thin | strut density < median − 3·MAD of all struts (robust low outlier) |
| Bent | peak bow off the strut's own axis > 1 radius = 212 µm |
| Present (complete) | none of the above |
Thresholds scale with the part (radius, length, the specimen's own density), so
they transfer to a new scan without re-tuning.

## 6. Where this fits the team workflow
| Workflow box | Provided here |
| :--- | :--- |
| Visualization Agent (Segment / Visualize Slice / Skeletonize) | MCP tools + `defect_visualizer` skill |
| **Strut Error Detection Agent** | `defect_classifier` skill, `classify_lattice_defects`, `detect_bent_struts` — **core** |
| Highest-FG / anchoring | metal-anchoring inside the classifier |
| Strut Validation | evidence figures via `defect_visualizer`; method validates vs raw CT |
| Statistical Analysis | `summarize_lattice_defects` + the JSON |
| Dashboard | 3D models via `lattice_3d_modeler`, interactive HTML, colour key |

**Still to build by teammates:** Data Loader Tool (file validation), Dashboard UI,
optional Tilt Tool (the classifier is already tilt-tolerant) and Research Agent.

---
Full architecture + threshold rationale: see [`AGENTS.md`](AGENTS.md) and
[`analysis/defect_detection/PROCESS.md`](analysis/defect_detection/PROCESS.md).
