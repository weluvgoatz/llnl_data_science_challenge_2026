---
name: strut-defect-detection
description: Detect and classify every strut in an X-ray CT lattice as present, missing, disconnected, thin, or bent — self-contained from a raw TIF plus the registered design JSON, using connectivity (not presence) to find severed struts, with every verdict carrying the measurement it was decided on. Manually validated 200/200 panels.
---

# Strut Defect Detection (v2)

Given a raw lattice CT `.tif` and its registered design `.json`, classify every
designed strut. Everything lives in `analysis/defect_detection/v2/`; read
`v2/README.md` for the full description.

## Select the specimen

```bash
export LATTICE_BASE="<scan name without extension>"
export LATTICE_STK="<dir holding the .tif>"
export LATTICE_DESIGN_JSON="<repo-relative registered design JSON>"
export LATTICE_VOXEL_UM=58.1            # set if the scan differs
export LATTICE_STRUT_DIAMETER_UM=424.0  # set if the specimen differs
```

Everything else adapts on its own: the lattice pitch is read from the design
JSON, the fused end-caps are auto-detected, and the thin cutoff is recomputed
from that scan's own strut population.

## Run

```bash
cd analysis/defect_detection/v2
python run_pipeline.py                  # detect + defect list
python run_pipeline.py --validate 20    # + validation panels
```

The first run segments and skeletonises (slow) and caches both. Later runs reuse
the caches. **Do not build the 3-D model here** — that is the
`lattice-defect-3d-model` skill and is on request only.

## The five verdicts

| verdict | test |
|---|---|
| `missing` | no strut was built between the two nodes |
| `disconnected` | built but severed — no single connected piece of metal reaches node to node |
| `thin` | some sustained section below `median − 3·MAD` of this scan's diameters |
| `bent` | centreline bows more than one strut radius off the chord between its own ends |
| `present` | none of the above |

Subtypes: `dropped` (both nodes printed), `node_lost` (one node never printed),
`void` (neither). Node states: `junction` / `partial` / `absent`.

## The one idea that matters

**Presence is not connection.** A strut that detached at its joint leaves a
hairline crack, so metal exists at every position along it — a step-by-step
presence test walks straight over the break and calls it healthy. Instead, label
the metal inside a corridor around the strut and ask whether **one connected
component reaches from node to node**. That cannot be fooled by a crack of any
width. This is why v2 finds 569 disconnected where the earlier presence-based
version found 34.

Two consequences worth knowing:
- A skeleton edge only proves connection if **both its ends are materially fused
  to their junction cores**; otherwise it is a phantom edge from a broken stub's
  tip being merged into a node cluster (462 of those here).
- Bow must be measured on the strut's **own skeleton polyline**. A
  metal-centroid path gets dragged sideways by neighbouring struts near
  junctions, which once inflated `bent` from 365 to 3054.

## Output

- `v2/strut_classification_v2.json` — every strut with the measurements behind
  its verdict (metal profile, gap, stub, diameters, bow, graph hops, node
  states, region)
- `v2/defect_list/defective_struts.json` + `.csv` — defective struts only, with
  both endpoints in voxels and microns and the TIF slice range to inspect each

## Reporting rule

Every strut is `interior` or `boundary`. **Boundary struts sit where the design
extends past the under-built outer shell** — the metal really is absent, but it
is shell fragmentation, not an isolated defect. Report interior and total
separately, always. Validated specimen: 1822 defective (9.87%), 541 interior.

## Validate before you trust

```bash
python validate_v2.py thin 20
python validate_v2.py disconnected 20
```

Panels are drawn **from the stored measurements**, so a panel cannot contradict
its verdict. On any scan you have not seen, check `thin` and `disconnected`
first — they are the most segmentation-sensitive. `missing` and `bent` are the
most robust. State plainly that a new specimen is unvalidated until inspected.

## Memory limits (these will bite you)

- Whole-volume distance transform asks ~5.8 GB and dies → use chunked
  `edt_radii`.
- `skimage.remove_small_objects` dies on a volume this size (bincounts every
  voxel) → `segment.py` labels explicitly instead.
- Hold at most one full-resolution array plus the mask.
