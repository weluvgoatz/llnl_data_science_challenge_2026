---
name: lattice-defect-3d-model
description: Build or serve the colour-coded 3-D model of a classified lattice, where defects are shown as real measured geometry (bent struts actually curve, thin struts are actually thin, severed struts are actually split) — for MeshLab or a web viewer. ON REQUEST ONLY; never part of a routine detection run.
---

# Lattice Defect 3-D Model

A colour-coded model of the classified lattice in which **defects are geometry,
not just colour**. Requires a completed classification
(`v2/strut_classification_v2.json`).

## ON REQUEST ONLY

**Do not build this, list its paths, or put it in a reply unless somebody
explicitly asks** for a visualisation / 3-D model / picture of the lattice /
"something to look at". It is the slowest artefact and the largest output, and a
routine detection run has no use for it.

Once built it **persists**. If the files already exist, serve them — do not
rebuild.

## Build

```bash
cd analysis/defect_detection/v2
python export_3d.py                     # or: python run_pipeline.py --viz
```

## What it writes (`v2/model3d/`)

| file | size | use |
|---|---|---|
| `lattice_full.ply` | ~53 MB | all struts — **MeshLab** |
| `lattice_defects_only.ply` | ~9 MB | defects only — **web viewers** |
| `parts/<verdict>.ply` | — | one per category, toggle independently |
| `color_key.png` | — | legend |

Serve `lattice_defects_only.ply` to a browser (small, loads fast) and
`lattice_full.ply` for desktop inspection.

## Colours and what the geometry shows

| | colour | rendering |
|---|---|---|
| present | blue | tube at its measured radius |
| missing | red | **kept as a full straight tube on purpose** — not removed, so you can see where the strut belongs |
| disconnected | orange | **one tube per surviving connected piece** — the break is a real hole |
| thin | yellow | radius read from the distance transform at every step, so hairlines are thin and necked struts pinch |
| bent | magenta | swept along the strut's **actual skeleton polyline** — the real curvature |

## Two things to tell the viewer

1. **MeshLab: set Render → Shading → None.** Otherwise lighting washes the
   vertex colours out and everything looks grey.
2. For struts **detached at a joint**, the crack is hairline — the tube visibly
   stops short of its node rather than showing a mid-span gap, because that is
   where the severance physically is. No gap is ever exaggerated to make it more
   visible; the geometry is what was measured.

## Accuracy notes for anyone editing this

- Radii come from the distance transform of the segmented mask and are **capped
  at 1.25× the nominal strut radius**. At a junction the transform reports the
  radius of the whole node blob; without the cap, strut ends inflate into blobs.
- Disconnected struts are rendered by **connected-component labelling** of the
  corridor metal, not by sampling presence along the line — a presence-based
  centreline cannot show a joint detachment at all.
- Bent struts use the stored skeleton polyline, i.e. the exact curve the bow
  measurement came from, so the picture and the number agree.
- Coordinates are scan voxels (x, y, z).
