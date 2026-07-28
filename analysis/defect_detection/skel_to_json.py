"""Build a 100%-faithful AS-BUILT lattice JSON directly from the scan.

Instead of assuming an ideal periodic grid, this extracts the graph from the
actual segmented material via skeletonization:

  segmented mask -> 3D skeleton (centerlines) -> graph (branch points = nodes,
  paths = struts) -> merge node fragments -> prune noise spurs ->
  measure each strut (length, thickness from EDT, density from raw CT) ->
  JSON in the ground-truth schema.

The result represents EXACTLY what is in the scan: present struts are edges,
missing struts are simply absent, disconnected struts appear as broken/loose
ends, and boundary nodes are captured as they are.

Usage (from repo root):
    python analysis/defect_detection/skel_to_json.py \
        --mask "..._segmented_clean.tif" --raw "....tif"
"""

import argparse
import gc
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from config import MASK as CFG_MASK, RAW as CFG_RAW   # specimen defaults (env-overridable)

CONN = np.ones((3, 3, 3), int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", default=str(CFG_MASK), help="segmented binary .tif")
    ap.add_argument("--raw", default=str(CFG_RAW), help="raw CT .tif (for density)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--d-merge", type=float, default=12.0,
                    help="merge node fragments closer than this (full-res vox)")
    ap.add_argument("--spur-len", type=float, default=20.0,
                    help="prune dangling edges shorter than this (vox)")
    ap.add_argument("--skel-cache", default=None,
                    help="path to cache/reuse the skeleton coordinates (.npz)")
    args = ap.parse_args()

    mask_path = Path(args.mask).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else \
        mask_path.with_name(mask_path.stem + "_asbuilt_graph.json")

    print("loading mask ...")
    mask = tifffile.imread(mask_path) > 0
    vol_shape = tuple(int(x) for x in mask.shape)   # keep before mask is freed
    print(f"  {mask.shape}, material {100*mask.mean():.2f}%")

    # Skeleton cache (store just the True-voxel coordinates; tiny + fast) so a
    # failure downstream never forces another slow skeletonization.
    cache = Path(args.skel_cache) if args.skel_cache else \
        mask_path.with_suffix(".skelcoords.npz")
    if cache.exists():
        print(f"loading cached skeleton from {cache.name} ...")
        z = np.load(cache); coords = z["coords"]
        skel = np.zeros(mask.shape, bool); skel[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    else:
        print("skeletonizing (full resolution - this is the slow step) ...")
        skel = skeletonize(mask)
        np.savez_compressed(cache, coords=np.argwhere(skel))
        print(f"  cached skeleton to {cache.name}")
    print(f"  skeleton voxels: {int(skel.sum()):,}")

    # --- degree of every skeleton voxel ---
    K = np.ones((3, 3, 3), int); K[1, 1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), K, mode="constant") * skel
    node_vox = skel & (deg != 2)               # branch(>=3)+end(1)+isolated(0)
    path_vox = skel & (deg == 2)
    del deg; gc.collect()

    # --- node labels + SPARSE centroids (over node voxels only, not full vol) ---
    node_lbl, n_nodes = ndi.label(node_vox, structure=CONN)
    nvz, nvy, nvx = np.where(node_vox)
    nvl = node_lbl[nvz, nvy, nvx]
    cnt = np.bincount(nvl, minlength=n_nodes + 1).astype(float); cnt[cnt == 0] = 1
    cz = np.bincount(nvl, weights=nvz.astype(float), minlength=n_nodes + 1) / cnt
    cy = np.bincount(nvl, weights=nvy.astype(float), minlength=n_nodes + 1) / cnt
    cx = np.bincount(nvl, weights=nvx.astype(float), minlength=n_nodes + 1) / cnt
    centroids = np.stack([cz, cy, cx], 1)[1:]   # drop background label 0

    # --- node id at each path voxel (dilate node labels by 1) ---
    pv = np.where(path_vox)
    node_lbl_dil = ndi.grey_dilation(node_lbl, footprint=CONN)
    nlab = node_lbl_dil[pv]
    del node_lbl, node_lbl_dil, node_vox, nvz, nvy, nvx, nvl; gc.collect()

    # --- edge labels + segment id at each path voxel (node labels now freed) ---
    edge_lbl, n_seg = ndi.label(path_vox, structure=CONN)
    elab = edge_lbl[pv]
    del edge_lbl, path_vox, skel; gc.collect()
    print(f"  raw: {n_nodes} node-clusters, {n_seg} path-segments")

    # --- thickness via a 2x-downsampled distance transform (memory-safe) ---
    # The graph is full-res; only this thickness measurement is at half-res,
    # which is plenty for a strut ~7 voxels thick. Full-res EDT needs ~10 GB.
    print("  distance transform (thickness, 2x) ...")
    m2 = mask[::2, ::2, ::2]
    edt2 = ndi.distance_transform_edt(m2)
    edt_pv = edt2[pv[0] // 2, pv[1] // 2, pv[2] // 2] * 2.0
    del edt2, m2, mask; gc.collect()

    print("  sampling raw density ...")
    raw = tifffile.imread(args.raw)
    raw_pv = raw[pv].astype(np.float64)
    del raw; gc.collect()

    # per-segment aggregates
    counts = np.bincount(elab, minlength=n_seg + 1).astype(float)
    counts[counts == 0] = 1
    seg_len = np.bincount(elab, minlength=n_seg + 1)
    seg_edt = np.bincount(elab, weights=edt_pv, minlength=n_seg + 1) / counts
    seg_raw = np.bincount(elab, weights=raw_pv, minlength=n_seg + 1) / counts

    # per-segment connected node labels (the 2 ends)
    pairs = np.unique(np.stack([elab, nlab], 1)[nlab > 0], axis=0)
    seg_to_nodes = defaultdict(set)
    for e, n in pairs:
        seg_to_nodes[int(e)].add(int(n) - 1)   # 0-based node id

    edges = []   # [a, b, length, thickness, density]
    for e, ns in seg_to_nodes.items():
        if len(ns) == 2:
            a, b = sorted(ns)
            edges.append([a, b, int(seg_len[e]), float(seg_edt[e]), float(seg_raw[e])])
    print(f"  edges connecting 2 nodes: {len(edges)}")

    # --- merge node fragments ---
    parent = list(range(len(centroids)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cKDTree(centroids).query_pairs(args.d_merge):
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(len(centroids)):
        groups[find(i)].append(i)
    new_ids = {root: i for i, root in enumerate(groups)}
    new_centroids = np.array([centroids[idx].mean(0) for idx in groups.values()])
    remap = {i: new_ids[find(i)] for i in range(len(centroids))}
    merged = []
    for a, b, L, th, de in edges:
        na, nb = remap[a], remap[b]
        if na != nb:
            merged.append([min(na, nb), max(na, nb), L, th, de])
    print(f"  after node merge: {len(new_centroids)} nodes, {len(merged)} edges")

    # --- prune short spurs (dangling short edges) ---
    edges = merged
    changed = True
    while changed:
        changed = False
        d = defaultdict(int)
        for a, b, L, th, de in edges:
            d[a] += 1; d[b] += 1
        keep = []
        for e in edges:
            a, b, L, th, de = e
            if L < args.spur_len and (d[a] == 1 or d[b] == 1):
                changed = True
            else:
                keep.append(e)
        edges = keep
    used = sorted({a for a, b, *_ in edges} | {b for a, b, *_ in edges})
    idmap = {old: i for i, old in enumerate(used)}
    nodes = new_centroids[used]
    edges = [[idmap[a], idmap[b], L, th, de] for a, b, L, th, de in edges]
    print(f"  after spur prune: {len(nodes)} nodes, {len(edges)} edges")

    # --- infer grid (for indices + unit_cells, to match schema) ---
    half = np.zeros(3); origin = np.zeros(3)
    for ax in range(3):
        c = nodes[:, ax]; hs = np.linspace(30, 45, 300)
        R = [np.abs(np.exp(1j * 2 * np.pi * c / h).mean()) for h in hs]
        h = hs[int(np.argmax(R))]; half[ax] = h
        ang = np.angle(np.exp(1j * 2 * np.pi * c / h).mean())
        origin[ax] = (ang / (2 * np.pi)) * h % h
    gidx = np.round((nodes - origin) / half).astype(int)
    gidx -= gidx.min(0)

    # --- build JSON (ground-truth schema + measured fields) ---
    junctions = [{"id": i,
                  "position": [float(nodes[i][2]), float(nodes[i][1]), float(nodes[i][0])],
                  "indices": [int(gidx[i][0]), int(gidx[i][1]), int(gidx[i][2])]}
                 for i in range(len(nodes))]

    # unit cells by binning node grid index // 2 (clamped), then group struts
    ncell = [max(gidx[:, k].max() // 2, 1) for k in range(3)]
    def cell_of_edge(a, b):
        m = (gidx[a] + gidx[b]) / 2
        return tuple(min(int(m[k] // 2), ncell[k] - 1) for k in range(3))
    cells = {}
    struts = []
    cell_struts = defaultdict(list)
    for sid, (a, b, L, th, de) in enumerate(edges):
        c = cell_of_edge(a, b)
        cid = cells.setdefault(c, len(cells))
        struts.append({"id": sid, "unit_cell_edge_idx": len(cell_struts[cid]),
                       "junction0": a, "junction1": b,
                       "thickness": th, "length": L, "mean_density": de})
        cell_struts[cid].append(sid)
    cbase = [min(c[k] for c in cells) for k in range(3)]
    unit_cells = [{"id": cid, "struts": cell_struts[cid],
                   "indices": [c[0]-cbase[0], c[1]-cbase[1], c[2]-cbase[2]]}
                  for c, cid in cells.items()]

    out = {"junctions": junctions, "struts": struts, "unit_cells": unit_cells,
           "meta": {"source": mask_path.name, "method": "skeleton graph (as-built)",
                    "cell_spacing_vox": [float(x) for x in (2 * half)],
                    "volume_shape": list(vol_shape)}}

    def _np(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"not serializable: {type(o)}")
    out_path.write_text(json.dumps(out, default=_np), encoding="utf-8")

    degs = defaultdict(int)
    for a, b, *_ in edges:
        degs[a] += 1; degs[b] += 1
    dv = np.array([degs[i] for i in range(len(nodes))])
    print(f"\nAS-BUILT GRAPH: {len(junctions)} junctions, {len(struts)} struts, "
          f"{len(unit_cells)} unit_cells")
    print(f"  median node degree {int(np.median(dv))} (octet interior = 12)")
    print(f"Saved: {out_path.name}  ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
