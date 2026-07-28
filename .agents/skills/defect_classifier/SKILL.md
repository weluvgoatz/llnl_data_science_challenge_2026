---
name: lattice-defect-classifier
description: Reference-free classification of every strut in an X-ray CT lattice as present, missing, bent, thin, or disconnected — grounded in the raw scan (no ground-truth labels), using a metal-anchored, topology-first classifier with physically-derived thresholds.
---

# Lattice Defect Classification Protocol

You are the **Reference-Free Lattice Defect Classifier**. Given an X-ray CT scan
of an octet-truss lattice, you decide, for every designed strut, whether it is
**present, missing, bent, thin, or disconnected** — using only the scan itself as
ground truth (never a paper or prior labels). The MCP tools live in
`src/defect_mcp_server.py`; the scripts live in `analysis/defect_detection/`.

Select the specimen once with environment variables (defaults reproduce the
challenge specimen), so every stage operates on the same scan:
```
export LATTICE_BASE="<scan file name without extension>"
export LATTICE_STK="<dir holding the .tif stack and derived files>"
export LATTICE_DESIGN_JSON="<repo-relative path to defect-free design graph JSON>"
```

## The pipeline (run in order; each stage caches its output)

### Step 1 — Segmentation
Isolate metal from background. Normally the upstream segmentation stage provides
this; if not, use the `segment_tiff` MCP tool with an Otsu threshold. Clean with
small-object removal + binary closing. Output: `<base>_segmented_clean.tif`.
Keep any boundary "cooked"/low-density layer — it is a REAL specimen feature, not
an artifact.

### Step 2 — Skeletonization → as-built graph
Call `build_asbuilt_graph()`. It thins the metal to a 1-voxel medial axis, counts
26-neighbour degrees (degree 2 = strut interior, degree >=3 = junction), labels
components, then cleans the graph: union-find node merge (D=20 vox), degree-2
dissolution, spur pruning (30 vox), short-artifact removal. Output: the cleaned
graph JSON (nodes + struts with length, thickness, mean density, bow).

### Step 3 — Classify
Call `classify_lattice_defects()`. It:
1. Registers the defect-free **design** graph to the as-built graph (affine ICP).
2. **Anchors** every designed node to real metal — snap to the nearest as-built
   node, else to the local metal centroid, else the node is genuinely absent.
   (This anchoring is what removes registration-drift false "missing" at edges.)
3. Classifies each designed strut **topology-first, material-second** with the
   rules below (call `describe_thresholds()` for the authoritative statement).

| Verdict | Rule | Threshold origin |
| :--- | :--- | :--- |
| **Missing** | metal along strut < 15% (or joint on no metal) | just above joint-metal floor |
| **Disconnected** | one continuous gap >= 25% of length | fraction of strut length |
| **Thin** | density < median - 3*MAD of all struts | computed from this specimen |
| **Bent** | peak bow off PCA axis > 1 radius = 212 um | design radius; +3sigma cross-check |
| **Present** | none of the above | - |

Output: `<base>_unified_defects_accurate.json`.

### Step 4 — Bend detail (optional but recommended)
Call `detect_bent_struts()` for per-strut bow, tortuosity, and `max_dev_um`.

### Step 5 — Report
Call `summarize_lattice_defects(<classification json>)` for the count/percentage
table, then write a short Markdown report: the per-category counts, the total
defect rate, and one sentence of method per defect type.

# Technical Constraints
- The classifier is **reference-free**: validate conclusions against the raw CT
  (render the flagged struts and look), never against a published defect list.
- Use 26-connectivity for all 3D component analysis (diagonal struts).
- Anchor to real metal before deciding "missing" — un-anchored registration drift
  invents a false missing shell at the specimen boundary.
- Thresholds scale with the part (strut radius, strut length, the specimen's own
  density distribution), so they transfer to other scans without re-tuning.
- Do not force a single connected component: missing/disconnected struts are
  genuine, expected disconnections.
