# Start here

Finds defective struts in a lattice CT scan. You give it a scan and the design
file; it tells you which struts are broken and where.

## What you need

Two files:
1. the CT scan — a `.tif` stack
2. the registered design — a `.json`

That's it. It does its own segmentation.

## Run it

```bash
cd analysis/defect_detection/v2
python run_pipeline.py
```

First run takes a few minutes (it segments and skeletonises the scan, then saves
those so later runs are quick).

## What you get

**`defect_list/defective_struts.csv`** — open it in Excel. One row per bad strut:

| column | meaning |
|---|---|
| `defect` | `missing`, `disconnected`, `thin`, or `bent` |
| `x0,y0,z0` → `x1,y1,z1` | the strut's two ends, in scan voxels |
| `z_slice_from`, `z_slice_to` | **which TIF slices to open** to look at it |
| `region` | `interior` or `boundary` — read the warning below |

`defective_struts.json` has the same thing plus microns and the measurement
behind each verdict. `strut_classification_v2.json` has all struts, healthy ones
included.

## What the four defects mean

| | |
|---|---|
| **missing** | the strut was never built |
| **disconnected** | it was built but it's snapped — no metal path from one end to the other |
| **thin** | too skinny somewhere along its length |
| **bent** | bowed off its straight line |

## ⚠️ One thing to know before you quote numbers

Every strut is marked `interior` or `boundary`.

**Boundary** ones sit at the outer surface, where the design file expects struts
the printer never built that far out. The metal really is missing — but that's
the edge of the part being under-built, not individual struts failing.

So report both:

> 1,822 struts flagged (9.87%), of which **640 are interior** — those are the
> real individual defects.

Filter the `region` column to separate them.

## Different scan?

```bash
export LATTICE_BASE="my_scan"                  # my_scan.tif
export LATTICE_STK="/path/to/folder"
export LATTICE_DESIGN_JSON="path/to/design.json"
export LATTICE_VOXEL_UM=58.1                   # only if yours differs
export LATTICE_STRUT_DIAMETER_UM=424.0         # only if yours differs
python run_pipeline.py
```

Everything else figures itself out from the scan and the design file.

**On a scan nobody has checked before, look at some results first:**

```bash
python validate_v2.py thin 20
python validate_v2.py disconnected 20
```

That saves pictures to `panels/` showing each strut with the numbers behind its
verdict. `thin` and `disconnected` are the ones most affected by segmentation
quality, so check those two. `missing` and `bent` hold up better.

## Want a 3D picture?

```bash
python run_pipeline.py --viz
```

Saves `model3d/lattice_full.ply` — open in MeshLab and **set Render → Shading →
None** (otherwise the colours look grey). Blue = healthy, red = missing,
orange = disconnected, yellow = thin, magenta = bent. Bent struts really curve
and thin struts really are thin, so you can see the damage, not just the colour.

It's off by default because it's slow and the file is big.

## More detail

- `README.md` — how it works, all the options
- `INTEGRATION.md` — wiring it into the pipeline or a website
