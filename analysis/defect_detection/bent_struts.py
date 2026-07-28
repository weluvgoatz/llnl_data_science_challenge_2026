"""Detect BENT struts from the CT scan.

A bent strut is present and continuous, but its centerline curves away from the
straight line between its two joints. Because the skeleton traces the ACTUAL
centerline (curve included), we can measure the bend directly:

    max lateral deviation = the largest perpendicular distance from any point on
                            the strut's real centerline to the straight chord
                            joining its two end nodes.

A perfectly straight strut deviates ~0-1 voxel (discretisation only); a bent
strut bows several voxels off axis.

We also compute tortuosity = arc length / chord length as a cross-check. Arc
length is obtained by ordering the strut's voxels along the chord (projection
parameter) and summing true Euclidean steps - correct for any strut that does
not fold back on itself.

The graph is rebuilt from the cached skeleton with per-strut VOXEL TRACKING, so
that after the cleanup (node merging, degree-2 dissolution) each final strut
still knows exactly which centerline voxels belong to it.

Usage:
    python analysis/defect_detection/bent_struts.py
"""

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import tifffile

from config import ROOT, STK, MASK, SKELC as SKEL_CACHE, BENT_JSON as OUT

CONN = np.ones((3, 3, 3), int)
D_MERGE = 20.0        # merge node fragments (distinct joints are ~55 vox apart)
SPUR_LEN = 30.0       # prune dangling edges shorter than this
MIN_STRUT = 30.0      # drop sub-length artifact edges between two junctions
VOXEL_UM = 58.1       # micron per voxel
STRUT_RADIUS_VOX = 424.0 / 2 / VOXEL_UM   # design radius ~3.65 voxels
K_OUT = 3.0           # statistical outlier strength


def main():
    print("loading skeleton (cached) ...")
    shape = tifffile.TiffFile(MASK).series[0].shape
    coords = np.load(SKEL_CACHE)["coords"]
    skel = np.zeros(shape, bool)
    skel[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    print(f"  volume {shape}, skeleton voxels {int(skel.sum()):,}")

    # ---- degree ----
    K = np.ones((3, 3, 3), int); K[1, 1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), K, mode="constant") * skel
    node_vox = skel & (deg != 2)
    path_vox = skel & (deg == 2)
    del deg; gc.collect()

    # ---- nodes: labels, sparse centroids, and per-node voxel coords ----
    node_lbl, n_nodes = ndi.label(node_vox, structure=CONN)
    nvz, nvy, nvx = np.where(node_vox)
    nvl = node_lbl[nvz, nvy, nvx]
    cnt = np.bincount(nvl, minlength=n_nodes + 1).astype(float); cnt[cnt == 0] = 1
    cz = np.bincount(nvl, weights=nvz.astype(float), minlength=n_nodes + 1) / cnt
    cy = np.bincount(nvl, weights=nvy.astype(float), minlength=n_nodes + 1) / cnt
    cx = np.bincount(nvl, weights=nvx.astype(float), minlength=n_nodes + 1) / cnt
    centroids = np.stack([cz, cy, cx], 1)[1:]          # (n_nodes,3) in (z,y,x)
    node_pts = np.stack([nvz, nvy, nvx], 1).astype(np.float32)

    # group node voxels by node id (for including dissolved-node voxels later)
    order = np.argsort(nvl, kind="stable")
    nvl_s = nvl[order]; node_pts_s = node_pts[order]
    bounds = np.searchsorted(nvl_s, np.arange(n_nodes + 2))
    node_coords = {i: node_pts_s[bounds[i + 1]:bounds[i + 2]] for i in range(n_nodes)}

    # ---- node id touching each path voxel (dilate node labels by 1) ----
    pv = np.where(path_vox)
    node_lbl_dil = ndi.grey_dilation(node_lbl, footprint=CONN)
    nlab = node_lbl_dil[pv]
    del node_lbl, node_lbl_dil, node_vox, nvz, nvy, nvx, nvl, node_pts
    gc.collect()

    # ---- path segments ----
    edge_lbl, n_seg = ndi.label(path_vox, structure=CONN)
    elab = edge_lbl[pv]
    path_pts = np.stack(pv, 1).astype(np.float32)      # (M,3) in (z,y,x)
    del edge_lbl, path_vox, skel; gc.collect()
    print(f"  {n_nodes} node-clusters, {n_seg} path-segments")

    # group path voxels by segment id
    order = np.argsort(elab, kind="stable")
    elab_s = elab[order]; path_pts_s = path_pts[order]
    sb = np.searchsorted(elab_s, np.arange(n_seg + 2))
    seg_coords = {e: path_pts_s[sb[e + 1]:sb[e + 2]] for e in range(n_seg)}

    # segment -> its two end nodes
    pairs = np.unique(np.stack([elab, nlab], 1)[nlab > 0], axis=0)
    seg_nodes = defaultdict(set)
    for e, nlb in pairs:
        seg_nodes[int(e)].add(int(nlb) - 1)

    # ---- build edges carrying their voxel sets ----
    edges = {}          # eid -> dict(u, v, vox=[arrays])
    adj = defaultdict(set)
    eid = 0
    for e, ns in seg_nodes.items():
        if len(ns) != 2:
            continue
        u, v = sorted(ns)
        # seg_coords is 0-indexed for connected-component label e (labels start at 1)
        vox = seg_coords.get(e - 1, np.empty((0, 3), np.float32))
        edges[eid] = {"u": u, "v": v, "vox": [vox]}
        adj[u].add(eid); adj[v].add(eid)
        eid += 1
    print(f"  raw edges (2 nodes): {len(edges)}")

    # ---- 1. merge node fragments ----
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cKDTree(centroids).query_pairs(D_MERGE):
        parent[find(a)] = find(b)
    groups = defaultdict(list)
    for i in range(n_nodes):
        groups[find(i)].append(i)
    gid = {r: i for i, r in enumerate(groups)}
    NP = np.array([centroids[idx].mean(0) for idx in groups.values()])
    node_group_coords = {gid[r]: np.concatenate([node_coords[i] for i in idx])
                         for r, idx in groups.items()}
    remap = {i: gid[find(i)] for i in range(n_nodes)}

    new_edges = {}; new_adj = defaultdict(set); seen = {}
    ne = 0
    for e in edges.values():
        u, v = remap[e["u"]], remap[e["v"]]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue                       # dedupe: one strut per node pair
        seen[key] = ne
        new_edges[ne] = {"u": key[0], "v": key[1], "vox": e["vox"]}
        new_adj[key[0]].add(ne); new_adj[key[1]].add(ne)
        ne += 1
    edges, adj = new_edges, new_adj
    print(f"  after node merge + dedupe: {len(NP)} nodes, {len(edges)} struts")

    # ---- 2. dissolve degree-2 nodes (re-join split struts, keep their voxels) ----
    def other(e, n):
        return edges[e]["v"] if edges[e]["u"] == n else edges[e]["u"]
    stack = [n for n in adj if len(adj[n]) == 2]
    while stack:
        n = stack.pop()
        if len(adj[n]) != 2:
            continue
        e1, e2 = list(adj[n])
        a, b = other(e1, n), other(e2, n)
        if a == b:
            for e in (e1, e2):
                edges.pop(e, None)
            adj[a].discard(e1); adj[a].discard(e2); adj[n].clear()
            continue
        merged_vox = edges[e1]["vox"] + edges[e2]["vox"]
        if n in node_group_coords and len(node_group_coords[n]):
            merged_vox = merged_vox + [node_group_coords[n]]   # the false mid-node's voxels
        nid = ne; ne += 1
        edges[nid] = {"u": min(a, b), "v": max(a, b), "vox": merged_vox}
        for e in (e1, e2):
            edges.pop(e, None)
        adj[a].discard(e1); adj[a].add(nid)
        adj[b].discard(e2); adj[b].add(nid)
        adj[n].clear()
        if len(adj[a]) == 2: stack.append(a)
        if len(adj[b]) == 2: stack.append(b)
    print(f"  after dissolving deg-2 nodes: {len(edges)} struts")

    # ---- 3. prune spurs, 4. drop short artifact edges ----
    def chord_len(e):
        return float(np.linalg.norm(NP[e["u"]] - NP[e["v"]]))
    changed = True
    while changed:
        changed = False
        deg2 = defaultdict(int)
        for e in edges.values():
            deg2[e["u"]] += 1; deg2[e["v"]] += 1
        for k in list(edges):
            e = edges[k]
            if chord_len(e) < SPUR_LEN and (deg2[e["u"]] == 1 or deg2[e["v"]] == 1):
                edges.pop(k); changed = True
    deg2 = defaultdict(int)
    for e in edges.values():
        deg2[e["u"]] += 1; deg2[e["v"]] += 1
    for k in list(edges):
        e = edges[k]
        if chord_len(e) < MIN_STRUT and deg2[e["u"]] >= 3 and deg2[e["v"]] >= 3:
            edges.pop(k)
    print(f"  CLEANED: {len({e['u'] for e in edges.values()} | {e['v'] for e in edges.values()})} nodes, "
          f"{len(edges)} struts")

    # ================= BEND MEASUREMENT =================
    print("\nmeasuring bend (max lateral deviation from the chord) ...")
    results = []
    skipped = 0
    for k, e in edges.items():
        p0, p1 = NP[e["u"]], NP[e["v"]]
        node_chord = float(np.linalg.norm(p1 - p0))
        # keep only physically plausible struts (a real octet strut is ~55 vox);
        # anything much longer is a wandering artifact edge, not a strut.
        if not (40.0 <= node_chord <= 75.0):
            skipped += 1
            continue
        Q = np.concatenate(e["vox"]).astype(float) if len(e["vox"]) else np.empty((0, 3))
        if len(Q) < 10:
            skipped += 1
            continue

        # Order the centreline voxels along the strut's OWN axis, found by PCA.
        # (Using the node-centroid chord is unreliable: a junction centroid can
        # sit off-axis or, for a badly merged node group, in empty space.)
        cen = Q.mean(0)
        X = Q - cen
        axis = np.linalg.svd(X, full_matrices=False)[2][0]     # 1st principal direction
        Qs = Q[np.argsort(X @ axis)]

        # CONTIGUITY CHECK: a genuine centreline is an unbroken 26-connected
        # chain, so consecutive voxels are <= sqrt(3) ~ 1.73 apart. A big jump
        # means this "strut" is a fragmented / artifact edge -> reject it.
        steps = np.linalg.norm(np.diff(Qs, axis=0), axis=1)
        if len(steps) == 0 or steps.max() > 3.0:
            skipped += 1
            continue

        # bow measured against the line joining the path's own two ends
        a, b = Qs[0], Qs[-1]
        ab = b - a
        span = float(np.linalg.norm(ab))
        if not (40.0 <= span <= 75.0):        # must be a physically real strut
            skipped += 1
            continue
        tt = np.clip(((Qs - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        proj = a[None] + tt[:, None] * ab[None]
        max_dev = float(np.linalg.norm(Qs - proj, axis=1).max())

        arc = float(steps.sum())              # true arc length along the chain
        tort = arc / span                     # >= 1 by construction

        results.append({"u": e["u"], "v": e["v"], "chord": node_chord, "span": span,
                        "arc": arc, "tortuosity": tort, "max_dev": max_dev,
                        "max_dev_um": max_dev * VOXEL_UM,
                        "p0": [float(x) for x in p0], "p1": [float(x) for x in p1],
                        "mid": ((p0 + p1) / 2).tolist()})
    print(f"  measured {len(results)} struts  (skipped {skipped} implausible/short edges)")

    dev = np.array([r["max_dev"] for r in results])
    tor = np.array([r["tortuosity"] for r in results])
    med = float(np.median(dev)); mad = float(np.median(np.abs(dev - med)))
    sigma = 1.4826 * mad if mad > 0 else dev.std()
    thr_stat = med + K_OUT * sigma
    thr_phys = STRUT_RADIUS_VOX
    thr = max(thr_stat, thr_phys)

    print("\n--- deviation distribution (voxels) ---")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q:2d}: {np.percentile(dev, q):.2f} vox ({np.percentile(dev,q)*VOXEL_UM:.0f} um)")
    print(f"  median {med:.2f}, robust sigma {sigma:.2f}")
    print(f"  statistical threshold (median+{K_OUT:g}sigma): {thr_stat:.2f} vox")
    print(f"  physical threshold (1 strut radius):        {thr_phys:.2f} vox")
    print(f"  => BENT if max deviation > {thr:.2f} vox ({thr*VOXEL_UM:.0f} um)")

    bent = dev > thr
    n = len(results)
    print("\n" + "=" * 60)
    print("BENT STRUT RESULT")
    print("=" * 60)
    print(f"  struts measured : {n}")
    print(f"  BENT            : {int(bent.sum())}  ({100*bent.mean():.2f}%)")
    print(f"  straight        : {int((~bent).sum())}  ({100*(~bent).mean():.2f}%)")
    print(f"  worst bend      : {dev.max():.2f} vox ({dev.max()*VOXEL_UM:.0f} um)")
    print(f"  median tortuosity {np.median(tor):.4f} (1.0 = perfectly straight)")

    for r, b in zip(results, bent):
        r["bent"] = bool(b)
    OUT.write_text(json.dumps(
        {"struts": results,
         "meta": {"threshold_vox": thr, "threshold_um": thr * VOXEL_UM,
                  "voxel_um": VOXEL_UM, "n_bent": int(bent.sum()), "n_struts": n}}),
        encoding="utf-8")
    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
