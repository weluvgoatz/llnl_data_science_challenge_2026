# Defective struts — 210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices

**1822 defective struts** out of 18468 (9.87%).

## Files
- `defective_struts.json` — full records + inline documentation
- `defective_struts.csv` — same rows, flat, for Excel

## Counts
| defect | total | interior | boundary | vertical | flat |
|---|---:|---:|---:|---:|---:|
| missing | 383 | 74 | 309 | 354 | 29 |
| disconnected | 569 | 31 | 538 | 251 | 318 |
| thin | 548 | 419 | 129 | 157 | 391 |
| bent | 322 | 116 | 206 | 313 | 9 |

## Subtypes (missing / disconnected)
- **missing**: 354 dropped, 29 node_lost
- **disconnected**: 566 -, 3 node_lost

## Coordinates
`x0,y0,z0 -> x1,y1,z1` are the strut's two ends in scan voxels. **z is the TIF slice number** (1 voxel = 58.1 um), so to inspect a strut open the slices in `z_slice_from`..`z_slice_to`.

## Read before quoting numbers
`region=boundary` struts sit near an outer face where the design extends past what was actually printed. The metal really is absent/severed, but it reflects the under-built shell rather than an isolated defect — **640 of 1822** defects are `interior` and individually defensible.

`detached_at_joint=1` (disconnected) means the strut body is still in place but cracked off at a joint — `gap_um` can read ~0 because the crack is short, yet no connected metal path exists. These are real severances, verified by connected-component labelling.

Each `strut_id` indexes the design JSON strut list and matches the `i####` in the manual-validation panel filenames.