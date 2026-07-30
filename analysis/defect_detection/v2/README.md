# Strut defect detector — v2 (self-contained)

**Input:** a raw CT `.tif` stack + the registered design `.json`
**Output:** every strut classified, with the evidence behind each verdict

Nothing outside this folder is needed. Segmentation, skeletonisation, graph
extraction, registration and classification all happen here, and the slow steps
cache themselves so they run once.

## Run it

```bash
# 1. classify (segments + skeletonises automatically if not already cached)
python detect_v2.py

# 2. look at the results yourself
python validate_v2.py missing 20        # 20 random missing struts
python validate_v2.py disconnected 10
python validate_v2.py thin 10
python validate_v2.py bent 5
python validate_v2.py missing 5 --orient vertical   # z-climbing only
python validate_v2.py missing 5 --subtype void      # both nodes never printed

# 3. export the defect list (JSON + CSV, with coordinates)
python export_defect_list.py

# 4. bulk samples for manual review (50 per category, one PNG each)
python export_samples.py
```

## Point it at a different scan

```bash
export LATTICE_BASE="my_scan"                 # expects my_scan.tif in LATTICE_STK
export LATTICE_STK="/path/to/tif_stacks"
export LATTICE_DESIGN_JSON="relative/path/to/registered.json"
export LATTICE_VOXEL_UM=58.1                  # <- set these two if the
export LATTICE_STRUT_DIAMETER_UM=424.0        #    specimen differs
python detect_v2.py
```

Everything else adapts on its own: the lattice pitch is read from the design
JSON, the fused end-caps are auto-detected, and the thin cutoff is recomputed
from that scan's own strut population.

`LATTICE_CAP_Z="70,690"` overrides cap auto-detection if you want to pin an
exact window.

## What each verdict means

| verdict | meaning |
|---|---|
| `missing` | no strut was built between the two nodes |
| `disconnected` | a strut was built but is severed — no single connected piece of metal reaches node to node |
| `thin` | some sustained section is below `median − 3·MAD` of this scan's diameters |
| `bent` | the centreline bows more than one strut radius off the chord between its own ends |
| `present` | none of the above |

Subtypes: `dropped` (both nodes printed), `node_lost` (one node never printed),
`void` (neither printed). Nodes are reported as `junction` / `partial` /
`absent`, and each strut is `interior` or `boundary`.

**Read `region` before quoting counts.** `boundary` struts sit near an outer
face where the design extends past what was actually printed — the metal really
is absent, but it reflects the under-built shell, not an isolated defect.

## How it works

1. **Segment** — Otsu on a 4× downsample, drop blobs < 20 voxels, close pinholes
2. **Skeletonise** — 1-voxel centreline of the metal
3. **Graph** — skeleton → real nodes and real connections; merge fragments,
   dissolve degree-2 chains, prune spurs
4. **Register** — affine ICP fits the design JSON to the as-built graph, then
   every design node is anchored onto real metal
5. **Connectivity** — for each design strut, label the metal inside a corridor
   around it and ask whether **one connected component reaches from node to
   node**. Presence is not connection: a strut detached at a joint leaves only a
   hairline crack that a step-by-step presence test walks straight over.
6. **Split** — broken struts are `missing` if no surviving piece > half a strut
   diameter exists mid-span, else `disconnected`
7. **Geometry** — connected struts get bow (from their own skeleton polyline)
   and diameter (EDT along the centreline)

Every strut's record stores the measurements its verdict came from, and the
validation panels are drawn **from those records** — so a panel cannot disagree
with its verdict.

## Files

| file | purpose |
|---|---|
| `detect_v2.py` | the detector; writes `strut_classification_v2.json` |
| `segment.py` | raw volume → clean metal mask |
| `validate_v2.py` | validation panels, drawn from stored measurements |
| `export_defect_list.py` | defect list with coordinates (JSON + CSV) |
| `export_samples.py` | 50 individual sample PNGs per category |
| `make_review_grids.py` | packs samples into 2×2 grids for fast review |
| `verify_segmentation.py` | proves the built-in segmentation matches the validated mask |
| `recheck4.py` | thin-slab vs thick-slab check for disputed struts |

## Validation status

Manually reviewed **200/200 panels** (50 per category) on the reference specimen
`210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices` — all correct.

Reference counts (18,468 struts): present 16646 · missing 383 · disconnected 569
· thin 548 · bent 322 · **defective 1822 (9.87%)**.

Regression-tested after making the constants dynamic: **bit-identical**. The
built-in segmentation was verified **voxel-exact** against the validated mask
(57,443,060 metal voxels, Otsu 40127).

`REF_*` constants at the top of `detect_v2.py` record the validated static
values; the calibration scales from them and is exactly 1.0 for this specimen.

**On a new specimen, run a validation sheet before trusting the numbers.**
Thin and disconnected are the most segmentation-sensitive; missing and bent are
the most robust.
