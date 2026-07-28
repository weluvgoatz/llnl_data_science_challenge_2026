"""ACCURATE unified defect classification — anchored to real metal.

The earlier version sampled along *registered blueprint* lines, which drift off
the real struts at the specimen edges and produce a false "missing" shell. This
version removes that error:

  1. Rebuild the as-built graph from the scan (cached skeleton) with bow + density.
  2. Register the design to the as-built graph (affine ICP - better edge fit).
  3. For every designed node, find an ANCHOR ON REAL METAL:
       - snap to the nearest as-built node, OR
       - if none is near (a boundary node my extraction missed), snap to the
         centroid of local metal, OR
       - if there is no metal there at all -> the node is genuinely absent.
  4. Classify each designed strut by REAL TOPOLOGY first, material second:
       - as-built edge exists between the two anchors  -> present (bent / thin)
       - no edge, but metal spans the anchors          -> present (extraction gap)
       - no edge, gap in the middle                    -> disconnected
       - no edge, empty                                -> missing
     Because anchors sit on real metal, registration drift can no longer create
     phantom defects at the boundary.
"""

import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from config import ROOT, STK, BASE, DESIGN_JSON as GT
MASK = STK / f"{BASE}_segmented_clean.tif"
RAW = STK / f"{BASE}.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
OUT = STK / f"{BASE}_unified_defects_accurate.json"

CONN = np.ones((3, 3, 3), int)
UM = 58.1
D_MERGE = 20.0
SPUR_LEN = 30.0
MIN_STRUT = 30.0
STRUT_RADIUS = 424.0 / 2 / UM          # ~3.65 vox
BENT_THR = STRUT_RADIUS
SNAP_R = 14.0                          # snap design node to as-built node within this
METAL_R = 11                           # else search this-radius ball for metal
K_OUT = 3.0
MISSING_FRAC = 0.15
GAP_FRAC = 0.25
DISK_R = 4.0                           # perpendicular search radius when tracking a strut
MISSING_GAP = 0.50                     # empty over >= half the span -> nothing is there
DISCONNECT_GAP_UM = 424.0              # a break >= one strut diameter is a real break


def longest_gap(hit):
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1; best = max(best, cur)
    return best


def _perp_axes(u):
    t = np.array([1., 0, 0]) if abs(u[0]) < 0.9 else np.array([0., 1, 0])
    e1 = np.cross(u, t); e1 /= np.linalg.norm(e1) + 1e-9
    return e1, np.cross(u, e1)


_DISK = None


def _disk_offsets():
    """integer (a,b) offsets filling a radius-DISK_R disk, nearest-first."""
    global _DISK
    if _DISK is None:
        r = int(np.ceil(DISK_R))
        pts = [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1)
               if a * a + b * b <= DISK_R * DISK_R]
        pts.sort(key=lambda ab: ab[0] ** 2 + ab[1] ** 2)
        _DISK = pts
    return _DISK


def disk_profile(pa, pb, metal, shape, K=60):
    """Is there metal in the plane PERPENDICULAR to the strut, at each step along it?

    Sampling the bare centre line is wrong: a real strut frequently sits a few
    voxels off the registered line, and the line then reads empty even though the
    strut is intact.  Searching a perpendicular disk follows the strut without
    smearing along its axis (a 3-D box would bridge genuine breaks).
    """
    u = pb - pa; u = u / (np.linalg.norm(u) + 1e-9)
    e1, e2 = _perp_axes(u)
    hits = np.zeros(K, bool)
    for i, t in enumerate(np.linspace(0, 1, K)):
        c = pa * (1 - t) + pb * t
        for a, b in _disk_offsets():
            q = np.round(c + a * e1 + b * e2).astype(int)
            if np.all(q >= 0) and np.all(q < shape) and metal[q[0], q[1], q[2]]:
                hits[i] = True
                break
    return hits


def build_asbuilt_graph(mask):
    """cached skeleton -> cleaned graph: node positions (z,y,x) + edges (u,v,bow)."""
    coords = np.load(SKELC)["coords"]
    skel = np.zeros(mask.shape, bool)
    skel[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    K = np.ones((3, 3, 3), int); K[1, 1, 1] = 0
    deg = ndi.convolve(skel.astype(np.uint8), K, mode="constant") * skel
    node_vox = skel & (deg != 2); path_vox = skel & (deg == 2)
    del deg; gc.collect()

    node_lbl, n_nodes = ndi.label(node_vox, structure=CONN)
    nvz, nvy, nvx = np.where(node_vox)
    nvl = node_lbl[nvz, nvy, nvx]
    cnt = np.bincount(nvl, minlength=n_nodes + 1).astype(float); cnt[cnt == 0] = 1
    cz = np.bincount(nvl, weights=nvz.astype(float), minlength=n_nodes + 1) / cnt
    cy = np.bincount(nvl, weights=nvy.astype(float), minlength=n_nodes + 1) / cnt
    cx = np.bincount(nvl, weights=nvx.astype(float), minlength=n_nodes + 1) / cnt
    centroids = np.stack([cz, cy, cx], 1)[1:]

    pv = np.where(path_vox)
    node_lbl_dil = ndi.grey_dilation(node_lbl, footprint=CONN)
    nlab = node_lbl_dil[pv]
    del node_lbl, node_lbl_dil, node_vox; gc.collect()
    edge_lbl, n_seg = ndi.label(path_vox, structure=CONN)
    elab = edge_lbl[pv]
    path_pts = np.stack(pv, 1).astype(np.float32)
    del edge_lbl, path_vox, skel; gc.collect()

    order = np.argsort(elab, kind="stable")
    elab_s = elab[order]; pts_s = path_pts[order]
    sb = np.searchsorted(elab_s, np.arange(n_seg + 2))
    seg_coords = {e: pts_s[sb[e + 1]:sb[e + 2]] for e in range(n_seg)}

    pairs = np.unique(np.stack([elab, nlab], 1)[nlab > 0], axis=0)
    seg_nodes = defaultdict(set)
    for e, nl in pairs:
        seg_nodes[int(e)].add(int(nl) - 1)

    edges = {}; adj = defaultdict(set); eid = 0
    for e, ns in seg_nodes.items():
        if len(ns) != 2:
            continue
        u, v = sorted(ns)
        edges[eid] = {"u": u, "v": v, "vox": [seg_coords.get(e - 1, np.empty((0, 3), np.float32))]}
        adj[u].add(eid); adj[v].add(eid); eid += 1

    # merge fragments
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
    remap = {i: gid[find(i)] for i in range(n_nodes)}
    ne = 0; E = {}; A = defaultdict(set); seen = {}
    for e in edges.values():
        u, v = remap[e["u"]], remap[e["v"]]
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen[key] = ne; E[ne] = {"u": key[0], "v": key[1], "vox": e["vox"]}
        A[key[0]].add(ne); A[key[1]].add(ne); ne += 1

    # dissolve degree-2
    def other(e, n):
        return E[e]["v"] if E[e]["u"] == n else E[e]["u"]
    stack = [n for n in A if len(A[n]) == 2]
    while stack:
        n = stack.pop()
        if len(A[n]) != 2:
            continue
        e1, e2 = list(A[n]); a, b = other(e1, n), other(e2, n)
        if a == b:
            for e in (e1, e2):
                E.pop(e, None)
            A[a].discard(e1); A[a].discard(e2); A[n].clear(); continue
        nv = E[e1]["vox"] + E[e2]["vox"]; nid = ne; ne += 1
        E[nid] = {"u": min(a, b), "v": max(a, b), "vox": nv}
        for e in (e1, e2):
            E.pop(e, None)
        A[a].discard(e1); A[a].add(nid); A[b].discard(e2); A[b].add(nid); A[n].clear()
        if len(A[a]) == 2: stack.append(a)
        if len(A[b]) == 2: stack.append(b)

    # prune spurs + short artifacts
    def clen(e):
        return float(np.linalg.norm(NP[e["u"]] - NP[e["v"]]))
    changed = True
    while changed:
        changed = False
        d = defaultdict(int)
        for e in E.values():
            d[e["u"]] += 1; d[e["v"]] += 1
        for k in list(E):
            if clen(E[k]) < SPUR_LEN and (d[E[k]["u"]] == 1 or d[E[k]["v"]] == 1):
                E.pop(k); changed = True
    d = defaultdict(int)
    for e in E.values():
        d[e["u"]] += 1; d[e["v"]] += 1
    for k in list(E):
        if clen(E[k]) < MIN_STRUT and d[E[k]["u"]] >= 3 and d[E[k]["v"]] >= 3:
            E.pop(k)

    # bow per edge
    edge_of = {}; bow = {}
    for e in E.values():
        p0, p1 = NP[e["u"]], NP[e["v"]]
        Q = np.concatenate(e["vox"]).astype(float) if e["vox"] else np.empty((0, 3))
        b = 0.0
        if len(Q) >= 8:
            cen = Q.mean(0); X = Q - cen
            axis = np.linalg.svd(X, full_matrices=False)[2][0]
            Qs = Q[np.argsort(X @ axis)]
            steps = np.linalg.norm(np.diff(Qs, axis=0), axis=1)
            if len(steps) and steps.max() <= 3.0:
                a0, b0 = Qs[0], Qs[-1]; ab = b0 - a0; span = np.linalg.norm(ab)
                if 40 <= span <= 75:
                    tt = np.clip(((Qs - a0) @ ab) / float(ab @ ab), 0, 1)
                    b = float(np.linalg.norm(Qs - (a0 + tt[:, None] * ab), axis=1).max())
        key = (e["u"], e["v"])
        edge_of[key] = True; bow[key] = b
    return NP, edge_of, bow


def main():
    print("loading mask + raw ...")
    mask = tifffile.imread(MASK) > 0
    metal = ndi.binary_dilation(mask, iterations=1)
    raw = tifffile.imread(RAW)
    shape = np.array(mask.shape)

    print("rebuilding as-built graph (cached skeleton) ...")
    NP, edge_of, bow = build_asbuilt_graph(mask)
    del mask; gc.collect()
    print(f"  as-built: {len(NP)} nodes, {len(edge_of)} struts")

    # per-edge density (sample raw along the node-to-node line)
    dens_of = {}
    for (u, v) in edge_of:
        p0, p1 = NP[u], NP[v]; ts = np.linspace(0, 1, 16)
        pts = np.round(p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]).astype(int)
        pts = np.clip(pts, 0, shape - 1)
        dens_of[(u, v)] = float(raw[pts[:, 0], pts[:, 1], pts[:, 2]].mean())
    dvals = np.array(list(dens_of.values()))
    dmed = np.median(dvals); dsig = 1.4826 * np.median(np.abs(dvals - dmed))
    d_cut = dmed - K_OUT * dsig

    # design + affine registration to as-built nodes
    gt = json.load(open(GT))
    GP = np.array([j["position"] for j in gt["junctions"]], float)[:, ::-1]   # [x,y,z]->(z,y,x)
    struts = [(s["junction0"], s["junction1"]) for s in gt["struts"]]
    tree = cKDTree(NP)
    T = None
    GPc = GP.copy()
    for _ in range(30):
        dd, idx = tree.query(GPc)
        m = dd < 45
        G1 = np.hstack([GP[m], np.ones((m.sum(), 1))])
        T, *_ = np.linalg.lstsq(G1, NP[idx[m]], rcond=None)     # (4,3)
        GPc = np.hstack([GP, np.ones((len(GP), 1))]) @ T
    print("affine registration done")

    # anchor every design node to real metal
    dd, idx = tree.query(GPc)
    anchor = np.full((len(GP), 3), np.nan)
    ab_node = np.full(len(GP), -1, int)
    n_snap = n_metal = n_absent = 0
    for i in range(len(GP)):
        if dd[i] < SNAP_R:
            anchor[i] = NP[idx[i]]; ab_node[i] = idx[i]; n_snap += 1
        else:
            c = np.round(GPc[i]).astype(int)
            lo = np.maximum(c - METAL_R, 0); hi = np.minimum(c + METAL_R + 1, shape)
            sub = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            if sub.any():
                zz, yy, xx = np.where(sub)
                anchor[i] = [lo[0] + zz.mean(), lo[1] + yy.mean(), lo[2] + xx.mean()]
                n_metal += 1
            else:
                n_absent += 1
    print(f"  anchors: {n_snap} snapped to as-built node, {n_metal} to local metal, {n_absent} absent")

    # classify each designed strut
    counts = defaultdict(int)
    out_struts = []

    for a, b in struts:
        na, nb = ab_node[a], ab_node[b]
        pa, pb = anchor[a], anchor[b]
        if np.isnan(pa).any() or np.isnan(pb).any():
            v = "missing"                    # a node region has no metal at all
        elif na >= 0 and nb >= 0 and (min(na, nb), max(na, nb)) in edge_of:
            key = (min(na, nb), max(na, nb))
            if bow[key] > BENT_THR:
                v = "bent"
            elif dens_of[key] < d_cut:
                v = "thin"
            else:
                v = "present"
        else:
            # No as-built edge: follow the strut with a perpendicular disk and judge
            # it by the size of the largest BREAK.  Measuring the break directly
            # needs no node-blob correction - an empty strut leaves a gap spanning
            # nearly the whole span, while its two end blobs stay at the ends.
            hit = disk_profile(pa, pb, metal, shape)
            gap = longest_gap(hit) / len(hit)                 # fraction of the span
            gap_um = gap * float(np.linalg.norm(pb - pa)) * UM
            if gap >= MISSING_GAP:
                v = "missing"           # empty over half the span or more
            elif gap_um >= DISCONNECT_GAP_UM:
                v = "disconnected"      # a real break of >= one strut diameter
            else:
                v = "present"
        counts[v] += 1
        # export in [x,y,z] scan coords (reverse zyx)
        out_struts.append({"p0": [float(pa[2]), float(pa[1]), float(pa[0])] if not np.isnan(pa).any()
                           else [float(GPc[a][2]), float(GPc[a][1]), float(GPc[a][0])],
                           "p1": [float(pb[2]), float(pb[1]), float(pb[0])] if not np.isnan(pb).any()
                           else [float(GPc[b][2]), float(GPc[b][1]), float(GPc[b][0])],
                           "verdict": v})

    n = len(struts)
    print("\n=== ACCURATE UNIFIED DEFECTS (anchored to real metal) ===")
    for k in ("present", "bent", "thin", "disconnected", "missing"):
        print(f"  {k:12s}: {counts[k]:5d}  ({100*counts[k]/n:.2f}%)")
    defects = n - counts["present"]
    print(f"  TOTAL defective: {defects}  ({100*defects/n:.2f}%)")

    OUT.write_text(json.dumps({"struts": out_struts,
                               "meta": {"counts": dict(counts), "n": n,
                                        "volume_shape_zyx": [int(x) for x in shape]}}),
                   encoding="utf-8")
    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
