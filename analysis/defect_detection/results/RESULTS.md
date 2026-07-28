# Defect results — 210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices

Source of truth: `210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_unified_defects_accurate.json`

**18468 struts total · 1123 defective (6.08%)**

| Category | Count | % of all | Deep-interior (trustworthy) |
|---|---:|---:|---:|
| present | 17345 | 93.92% | 11135 |
| missing | 427 | 2.31% | 71 |
| disconnected | 44 | 0.24% | 4 |
| thin | 287 | 1.55% | 166 |
| bent | 365 | 1.98% | 107 |

## How each call is made

Missing/disconnected are decided by the size of the largest BREAK. The strut is followed with a disk held perpendicular to it, not by sampling the bare centre line: a real strut often sits a few voxels off the registered line, and a line sample reads that offset as a break. (A 3-D box would instead bridge genuine breaks, so the disk is kept flat.)

- **missing** — empty over >= 50% of the span.
- **disconnected** — a break >= 424 um (one strut diameter), but less than half the span.
- **thin** — tube diameter below median - 3*MAD of all struts.
- **bent** — centerline bows > 1 strut radius (212 um) off the straight axis.
- **present** — none of the above.

## Read this before quoting numbers

Only **71 of 427** missing struts sit deep inside the printed part. The rest are near an outer face, where the design extends past what was actually built - those are boundary artifacts, not dropped struts. Quote the deep-interior count for verified defects and the full count only as a raw design-comparison number.

The fused end-caps (z < ~60, ~73% solid metal vs ~4% in open lattice) are excluded from the trustworthy counts - struts there are fused into a slab and cannot be judged individually.

`defects.csv` lists every defective strut; use `deep_interior = 1` to filter.