# 3-D lattice defect models

Colour-coded 3-D models of the classified octet-truss lattice
(`210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices`).

**Defects are shown as real measured geometry, not just colour** — bent struts
actually curve, thin struts actually are thin, severed struts actually are split.

## The two files

| file | size | contains | use it for |
|---|---|---|---|
| `lattice_3d_ALL_struts_color_coded.ply` | 80 MB | all **18,468** struts | the full picture — MeshLab / desktop / web on request |
| `lattice_3d_DEFECTS_ONLY_color_coded.ply` | 22 MB | only the **1,822** defective struts | fast loading, presentations, default web view |

`COLOR_KEY.png` is the legend.

Tubes are **14-sided**, so they read as round rather than faceted when you zoom in.
The polygon budget is spent on the defects rather than spread evenly: a healthy
strut is straight and gains nothing from extra samples along it, while a bent one
needs them to render its curve and a necked one to render its pinch.

| | axial samples | vertices |
|---|---:|---:|
| present | 6 | 1,430,582 |
| disconnected | 26 | 258,088 |
| thin | 22 | 167,378 |
| bent | 30 | 100,500 |
| missing | 2 | 11,490 |

**Serving this on the web:** 80 MB is heavy for a browser. Load
`DEFECTS_ONLY` (22 MB) by default and fetch the full model only when the user
asks to see the whole lattice. If it still needs to be lighter, converting to
glTF with Draco compression typically cuts a mesh like this by 5-10x.

## How to open them

**MeshLab** (or any mesh viewer): File → Import Mesh → pick the `.ply`.

> ### ⚠️ Set Render → Shading → None
> Otherwise the lighting washes out the vertex colours and the whole thing looks
> grey. This is the single most common "it's broken" report — it isn't, it's the
> shading.

**In a browser:** these are standard binary PLY files with per-vertex colour, so
three.js `PLYLoader` reads them directly. Use the DEFECTS_ONLY file for the web —
9 MB loads fast, 53 MB does not.

## What the colours mean

| colour | verdict | count | what the geometry shows |
|---|---|---|---|
| 🔵 blue | present | 16,646 | healthy strut at its measured radius |
| 🔴 red | missing | 383 | **deliberately kept as a full straight tube** so you can see where the strut *should* be — it was never printed |
| 🟠 orange | disconnected | 569 | **one tube per surviving piece** — the break is a real hole in the mesh |
| 🟡 yellow | thin | 548 | radius measured at every step, so hairlines are thin and necked struts pinch mid-span |
| 🟣 magenta | bent | 322 | swept along the strut's **actual** centreline, so you see the true curvature |

## Two honest notes

**Missing struts are not removed.** That's on purpose — a hole in the model tells
you nothing, whereas a red tube shows exactly which strut is absent and where it
belongs.

**Struts detached at a joint** (462 of the 569 disconnected) have a *hairline*
crack. Those render as a tube that visibly **stops short of its node** rather
than a mid-span gap, because that's where the severance physically is. No gap was
ever exaggerated to make it more visible — you're looking at what was measured.

## Reading the numbers responsibly

1,822 struts are flagged (9.87% of 18,468), but **640 of those are interior**.
The rest sit near the specimen's outer faces, where the design file expects
struts the printer never built that far out. The metal really is absent there —
but that's the edge of the part being under-built, not individual struts failing.

So: **1,822 flagged, 640 genuine individual defects.** Quote both.

## Regenerating these

```bash
cd analysis/defect_detection/v2
python run_pipeline.py --viz
```

Takes about two minutes once the scan has been classified. Output lands in
`model3d/` (plus per-category `.ply` files if you want to toggle individual
defect types); these two were copied from there and renamed.
