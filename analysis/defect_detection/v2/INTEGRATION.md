# Detection Subagent — integration guide

For teammates wiring this into the pipeline or the website. If you just want to
run it, see `README.md`.

## Contract

**In:** a raw CT `.tif` stack + the registered defect-free design `.json`
**Out:** `defect_list/defective_struts.json` (and `.csv`)

Nothing else is required. This subagent does its own segmentation and
skeletonisation — do not wait on another agent for a mask.

```bash
export LATTICE_BASE="my_scan"                    # my_scan.tif lives in LATTICE_STK
export LATTICE_STK="/path/to/tif_stacks"
export LATTICE_DESIGN_JSON="relative/path/registered.json"
export LATTICE_VOXEL_UM=58.1                     # set if your scan differs
export LATTICE_STRUT_DIAMETER_UM=424.0           # set if your specimen differs

cd analysis/defect_detection/v2
python run_pipeline.py
```

First run segments + skeletonises (several minutes) and caches both; later runs
are fast.

## What downstream agents read

### `strut_classification_v2.json` — every strut
```json
{"i": 10, "verdict": "missing", "subtype": "dropped",
 "p0": [64.0, 55.0, 109.5], "p1": [100.62, 57.12, 66.5],
 "node0": "junction", "node1": "junction",
 "context": "boundary", "gap_um": 2059.0, "stub_um": 0.0,
 "hops": 2, "prof": "###########.................############"}
```

### `defect_list/defective_struts.json` — the 1822 defective only
Same information plus microns, midpoint, `z_slice_range` (which TIF slices to
open), `length_um`, and `orientation` (`vertical` / `flat`). The file documents
its own fields in a `documentation` object at the top — read that rather than
guessing.

`defective_struts.csv` is the same rows flat, for Excel or pandas.

## Verdicts

| verdict | meaning |
|---|---|
| `present` | healthy |
| `missing` | no strut was built between the two nodes |
| `disconnected` | built but severed — no connected metal path node to node |
| `thin` | some sustained section below `median − 3·MAD` of this scan's diameters |
| `bent` | centreline bows more than one strut radius off its own chord |

Subtypes: `dropped` (both nodes printed) · `node_lost` (one node never printed) ·
`void` (neither). Node states: `junction` / `partial` / `absent`.

## THE ONE THING TO GET RIGHT WHEN REPORTING

Every strut carries `region`: `interior` or `boundary`.

**Boundary struts sit near an outer face where the design extends past what was
actually printed.** The metal really is absent or severed there, but it reflects
the under-built outer shell — not an isolated manufacturing defect. Quoting the
total as "defects found" overstates the case.

Reference specimen:

| | total | interior |
|---|---:|---:|
| missing | 383 | 74 |
| disconnected | 569 | 31 |
| thin | 548 | 419 |
| bent | 322 | 116 |
| **defective** | **1822 (9.87%)** | **640** |

Report both, labelled. `defective_struts.csv` has a `region` column to filter on.

## 3-D visualisation — on request only

The colour-coded model is **not** built by a normal run. When a user asks to see
the lattice:

```bash
python run_pipeline.py --viz        # or: python export_3d.py
```

Writes to `model3d/`:
- `lattice_defects_only.ply` (~9 MB) — **serve this to the browser**
- `lattice_full.ply` (~53 MB) — for MeshLab
- `color_key.png` — legend

Colours: blue present · red missing · orange disconnected · yellow thin ·
magenta bent. Defects are real geometry — bent struts actually curve, thin struts
are actually thin, severed struts are actually split. Missing struts are
deliberately kept as full red tubes so you can see where they belong.

Two notes for whoever builds the viewer: PLY with per-vertex colour loads
directly in three.js (`PLYLoader`); and if a user opens the file in MeshLab tell
them to set **Render → Shading → None**, or lighting washes the colours out.

Once built the files persist — serve them, don't rebuild.

## Validation

```bash
python validate_v2.py thin 20            # panels drawn from the stored measurements
python validate_v2.py disconnected 20
python validate_v2.py missing 20 --orient vertical
python export_samples.py                 # 50 individual PNGs per category
```

Manually reviewed 200/200 panels on the reference specimen — all correct. Every
panel is drawn **from the stored record**, so a panel cannot disagree with its
verdict.

**On a new specimen, render a `thin` and a `disconnected` sheet before quoting
numbers.** Those two are the most sensitive to segmentation quality; `missing`
and `bent` are the most robust.

## Files

| file | purpose |
|---|---|
| `run_pipeline.py` | entry point — start here |
| `detect_v2.py` | the detector |
| `segment.py` | raw volume → clean metal mask |
| `validate_v2.py` | validation panels |
| `export_defect_list.py` | the downstream contract (JSON + CSV) |
| `export_3d.py` | 3-D model (on request) |
| `export_samples.py`, `make_review_grids.py` | bulk manual review |
| `verify_segmentation.py` | proves segmentation matches the validated mask |

## Reproducibility

Verified end-to-end on the reference specimen:
- raw TIF → mask: **voxel-exact** (57,443,060 metal voxels, Otsu 40127)
- mask → skeleton: **identical** (767,611 voxels)
- skeleton → verdicts: **bit-identical** to the manually validated baseline

`REF_*` constants at the top of `detect_v2.py` record the validated static
values. Calibration scales from them and prints its factors at startup — they are
exactly `1.0000` for the reference specimen, which is the regression test.
