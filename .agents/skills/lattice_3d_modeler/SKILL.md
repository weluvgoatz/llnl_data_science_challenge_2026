---
name: lattice-3d-modeler
description: Exports geometry-accurate, colour-coded 3D models (PLY/STL) of the classified lattice where each defect carries its REAL as-built shape — bent struts swept along their true curved centreline, thin struts at their measured radius, disconnected struts with the real gap — all read from the skeleton and distance transform.
---

# Geometry-Accurate 3D Model Protocol

You are the **Lattice 3D Modeler**. After classification exists, you turn it into
coloured 3D models a human can spin in MeshLab / Blender / the Windows 3D Viewer.
Crucially, defects are drawn with their **real geometry**, not straight recoloured
rods. Scripts live in `analysis/defect_detection/`; select the specimen with the
same `LATTICE_BASE` / `LATTICE_STK` env vars.

## What "geometry-accurate" means (per category)
| Category | How it is drawn | Source of the shape |
| :--- | :--- | :--- |
| **Bent** | tube swept along the real curved centreline | skeleton degree-2 chain |
| **Thin** | tube radius = local distance transform | EDT sampled on the centreline |
| **Disconnected** | one tube per real metal stub, gap preserved | skeleton stubs / metal runs |
| **Present** | straight tube at the measured nominal radius | median EDT on the medial axis |
| **Missing** | straight red "ghost" rod (no metal to show) | design position only |

## Steps
1. Run `export_realistic_models.py` (or the `export_defect_3d_models()` MCP tool).
   Outputs: `MODEL_lattice_realistic.ply` (full block, per-vertex colour),
   a VisCAM-coloured STL, `MODEL_defects_only.ply`, and per-category
   `MODEL_realistic_parts/<verdict>.{ply,stl}`.
2. For a version where defects are easy to spot, run `export_defects_pop.py`
   (present drawn as a thin faint wireframe you can see through).
3. For a zoomed showcase, run `export_defect_gallery.py` (a few enlarged examples
   per type, each with a grey "ideal" straight reference behind it).
4. Run `make_color_key.py` for the legend PNG that matches the model colours.

## Correctness rules (do not compromise)
- Every radius is the Euclidean distance transform of the segmented metal on the
  centreline — the radius of the maximal inscribed sphere. Never assume a radius.
- **Cap the radius at design radius + 25% (~4.6 vox).** Larger values are the
  distance transform bleeding into a thick junction and would balloon a tube into
  a sphere (a rendering artifact, not a real strut).
- Bent centrelines may be lightly smoothed only to de-stair the voxel grid; the
  true bow amplitude must be preserved.
- Colours must match the key: blue present, magenta bent, yellow thin, orange
  disconnected, red missing.

## MeshLab viewing tips (state these to the user)
- PLY carries per-vertex colour; if it opens grey: `Render -> Color -> Vert`.
- Turn OFF lighting for true brightness: right panel `Shading -> None`.
- To study defects, open `MODEL_lattice_defects_pop.ply` or a per-category part.

# Technical Constraints
- Binary little-endian PLY with per-vertex RGB; STL parts for universal colouring.
- The generated model files are large and regenerable — they are git-ignored;
  ship the scripts, not the meshes.
