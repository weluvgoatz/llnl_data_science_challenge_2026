# Agentic architecture — reference-free lattice defect detection

This repo is packaged as a **Codex agentic system**: the scientific pipeline is
exposed as MCP **tools**, documented as **skills**, and orchestrated by
**subagents**, so you can run defect detection by asking an agent instead of
hand-running scripts — and reuse the pieces to build your own agents.

## The problem
Given an X-ray CT scan of a metal additively-manufactured octet-truss lattice,
find and classify every strut as **present, missing, bent, thin, or
disconnected**, using only the scan as ground truth (reference-free — no labels,
no trusting a paper). The full method (thresholds, why each is set where it is) is
in [`analysis/defect_detection/PROCESS.md`](analysis/defect_detection/PROCESS.md).

## Layout
```
.codex/agents/            Codex subagents (.toml, developer_instructions)
  segmentation_agent.toml       closed-loop Otsu segmentation of a .tif
  strut_error_detection_agent.toml  the "Strut Error Detection" node: detect +
                                    classify every strut with evidence + confidence
.agents/skills/           Codex skills (SKILL.md + optional scripts/)
  threshold_optimizer/          pick + justify a segmentation threshold
  nde_report_expert/            NDE report from volume/mask/skeleton
  defect_classifier/            reference-free classify every strut
  defect_visualizer/            per-defect validation figures + atlas + maps
  lattice_3d_modeler/           geometry-accurate coloured 3D models
src/                      MCP servers (FastMCP)
  mcp_server.py                 segment_ct_dataset, segment_tiff, skeletonize
  defect_mcp_server.py          the defect pipeline as callable tools
  skeletonization.py            skeletonize helper
analysis/defect_detection/  the algorithms (importable, specimen-parameterized)
  config.py                     central paths/constants (env-overridable)
  skel_to_json.py, clean_and_compare.py, unified_defects_accurate.py,
  bent_struts.py, split_defects.py, viz_*.py, export_*.py, PROCESS.md
data/                     sample CT / design graphs (Git LFS)
```

## Pipeline (tools compose in this order)
```
segment_tiff / segmentation_agent      -> <base>_segmented_clean.tif
skeletonize                            -> <base>_...skelcoords.npz
build_asbuilt_graph                    -> cleaned as-built graph JSON
classify_lattice_defects   ---+        -> <base>_unified_defects_accurate.json
detect_bent_struts            |        -> <base>_asbuilt_bent.json
render_defect_visualizations  |        -> VIZ_*.png
export_defect_3d_models       |        -> MODEL_*.ply / .stl
summarize_lattice_defects  <--+        -> counts + percentages
describe_thresholds                    -> the exact decision rules
```

## Decision rules (reference-free, topology-first)
| Verdict | Rule | Threshold origin |
| :--- | :--- | :--- |
| Missing | metal along strut < 15% (or joint on no metal) | above the joint-metal floor |
| Disconnected | one continuous gap >= 25% of length | fraction of strut length |
| Thin | density < median - 3*MAD of all struts | computed from the specimen |
| Bent | peak bow off PCA axis > 1 radius = 212 um | design radius (+3sigma check) |
| Present | none of the above | - |

## Run it on a different scan (no code edits)
Every stage reads its paths from `analysis/defect_detection/config.py`, which is
environment-overridable:
```
export LATTICE_BASE="my_scan"                 # file name, no extension
export LATTICE_STK="data/my_scan/tif_stacks"  # dir with the .tif + derived files
export LATTICE_DESIGN_JSON="data/my_scan/design_graph.json"
```
Defaults reproduce the challenge specimen, so no env vars = original behaviour.

## Register the MCP servers with Codex
Add to your Codex MCP config (e.g. `~/.codex/config.toml`):
```toml
[mcp_servers.ct_segmentation]
command = "python"
args = ["src/mcp_server.py"]

[mcp_servers.lattice_defects]
command = "python"
args = ["src/defect_mcp_server.py"]
```
Then invoke the `strut_error_detection_agent` subagent, or activate a skill and
let the agent call the tools.

## Setup
```
pip install -r requirements.txt
```

## Result on the challenge specimen (18,468 struts)
Present 93.42% · Missing 2.22% · Bent 1.98% · Thin 1.55% · Disconnected 0.83%
-> **6.58% defective** (metal-anchored classifier, validated against the raw CT).
