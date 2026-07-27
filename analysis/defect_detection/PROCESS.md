# As-Built Lattice Defect Detection — Full Process

Goal: turn the CT scan (`.tif`) into a graph JSON (nodes + struts) that is
faithful to the actual printed part ("as-built"), then compare it to the
no-defect ground-truth design to count missing / disconnected / thin struts.

Input: `data/missing_struts/tif_stacks/…0point5dash1….tif` (761×815×837 uint16,
~1 GB, the 0.5%-missing specimen).

---

## Two approaches (why we ended up with the skeleton one)

1. **Periodic-grid** (earlier): assume a perfect octet grid, infer its spacing
   from the data, predict where every strut *should* be, then check the scan.
   Fast, but it *imposes* an ideal lattice.
2. **Skeleton-based** (final): extract the graph *directly from the material*.
   Faithful to what is actually there — present struts are edges, missing struts
   are absent, broken struts show as loose ends. This is the one we kept.

---

## The skeleton pipeline, step by step

### Stage 0 — Segmentation (`segment_to_tif.py`, `clean_segmentation.py`)
- Otsu-threshold the raw CT → binary mask (material vs air). Otsu = 40127,
  ~11% material.
- Clean it: remove speckle blobs < 20 voxels, morphological closing. Output:
  `…_segmented_clean.tif`. This reduces skeleton noise later.

### Stage 1 — Skeletonize (`skel_to_json.py`)
- `skimage.morphology.skeletonize` (3D, Lee's method) on the full-res mask.
- Fat struts (~7 voxels thick) collapse to 1-voxel centerlines.
- Result: 767,611 skeleton voxels. (Slow: ~8 min. Cached to `.skelcoords.npz`
  so reruns are instant.)

### Stage 2 — Skeleton → graph
- Degree of every skeleton voxel = count of its 26-neighbors (via convolution).
- Classify voxels: degree 1 = endpoint, degree 2 = mid-strut path, degree ≥3 =
  junction (branch point).
- **Nodes** = connected clusters of (branch + endpoint) voxels; position =
  centroid (computed sparsely over node voxels only, for memory).
- **Edges** = connected runs of degree-2 path voxels; each edge's two end nodes
  are found by dilating the node labels by 1 voxel and reading which nodes each
  path touches.

### Stage 3 — Measure each strut
- **Length** = number of path voxels.
- **Thickness** = distance transform of the mask (computed at 2× downsampling
  for memory), sampled along the strut.
- **Density** = raw CT intensity sampled along the strut.

### Stage 4 — First cleanup + write JSON
- Merge node fragments within a radius; prune short spurs.
- Write JSON in the ground-truth schema (`junctions`, `struts`, `unit_cells`)
  plus measured fields. Output: `…_asbuilt_graph.json`.
- Raw result: **4850 nodes, 18823 struts** — but inflated by artifacts.

### Engineering problems hit (and fixes)
| Problem | Fix |
|---|---|
| Full-res distance transform allocated 5.8 GB → OOM | compute thickness on a 2× volume |
| `center_of_mass` raveled the full 519M array → OOM | compute centroids sparsely (node voxels only) |
| Three 2 GB label arrays held at once | free them before the distance transform |
| `mask.shape` used after `del mask` | capture the shape first |
| numpy `int64` not JSON-serializable | custom encoder |
| Re-skeletonizing (8 min) on every failure | cache the skeleton coords |

---

## Stage 5 — Graph cleanup (`clean_and_compare.py`)

The raw graph over-counted (18,823 > ideal 18,468) because skeletonization adds
artifacts. We diagnosed it: only 16,504 edges had real strut length (~55 vox);
the rest were short. Fixes:

1. **Merge node tangles** — at a real junction 12 struts meet and the skeleton
   knots up into several fragments; merge fragments within 20 voxels into one
   node (distinct nodes are ~55 voxels apart, so this is safe).
2. **Dedupe** — one strut per node pair.
3. **Dissolve degree-2 nodes** — a real strut sometimes gets a false branch-point
   in its middle, splitting it into two edges; merge them back into one strut.
4. **Prune short spurs** — dangling edges < 30 voxels.
5. **Drop residual short artifact edges** between two junctions.

Result: **3658 nodes, 17,878 struts, median node degree 12** (the exact octet
interior value) — and now **fewer** struts than the ideal 18,468, as expected
for a defective part. Output: `…_asbuilt_graph_cleaned.json`.

---

## Stage 6 — Compare to ground truth (`clean_and_compare.py`)

- **Register** the ground-truth design onto the as-built graph with a similarity
  ICP (scale + translation) — needed because the reference JSON is ~0.8% scaled
  and shifted relative to the scan.
- **Match** each of the 18,468 designed struts to the nearest as-built strut.
  No match → missing; matched + abnormal → defective; matched + normal → present.

---

## Stage 7 — Split missing vs disconnected (`split_defects.py`)

Graph-matching can't tell a truly-absent strut from a broken one (both lack a
complete edge). So for each designed strut we sample the *actual material* along
its registered path in the mask:
- path essentially empty → **missing**
- partial material with a contiguous gap → **disconnected**
- continuous but low density → **thin**
- continuous + full density → **present**

The gap-detection tolerance (mask dilation) strongly affects the
missing/disconnected split: dilation=2 bridges small breaks (few disconnected);
dilation=1 keeps them visible (more disconnected).

---

## Results

Cleaned as-built graph: **3658 nodes, 17,878 struts** (ideal: 3430 / 18,468).

Defect breakdown (dilation=1, sensitive):

| Verdict | Count | % |
|---|---:|---:|
| Present | 17,614 | 95.38% |
| Thin | 42 | 0.23% |
| Disconnected | 375 | 2.03% |
| Missing | 437 | 2.37% |
| **Total defects** | **854** | **4.62%** |

Cross-method comparison (total defect rate is robust; the split is not):

| Method | Missing | Disconnected | Total |
|---|---:|---:|---:|
| skeleton graph (edge-match) | 5.15%* | (lumped) | 6.06% |
| material sample, dilation=2 | 2.15% | 0.99% | 3.55% |
| material sample, dilation=1 | 2.37% | 2.03% | 4.62% |
| paper (manual review) | 0.57% | 5.13% | 5.70% |

\*graph method lumps disconnected into missing.

**Honest conclusion:** the *total* defect rate is reliable (~4–6%, paper 5.7%);
the missing-vs-disconnected split is threshold/registration-dependent and cannot
be pinned down precisely by an automated pipeline without manual review.

---

## Files

Scripts (`analysis/defect_detection/`):
- `skel_to_json.py` — segmented TIF → as-built graph JSON
- `clean_and_compare.py` — clean the graph + compare to ground truth
- `split_defects.py` — split into missing / disconnected / thin
- (earlier, grid-based: `reffree_defect_detection.py`, `tif_to_lattice_json.py`,
  `analyze_lattice_json.py`, `reference_based_registered.py`)

Data (`data/missing_struts/tif_stacks/`):
- `…_asbuilt_graph.json` — raw as-built graph
- `…_asbuilt_graph_cleaned.json` — cleaned graph (the accurate one)
