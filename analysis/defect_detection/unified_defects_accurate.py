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

from config import (
    ROOT, STK, BASE, DESIGN_JSON as GT,
    VOXEL_UM as UM, STRUT_RADIUS_VOX,
    MISSING_FRAC, GAP_FRAC, THIN_OUTLIER_K, BENT_RADIUS_MULT,
    SNAP_R_VOX, METAL_R_VOX,
)
MASK = STK / f"{BASE}_segmented_clean.tif"
RAW = STK / f"{BASE}.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
OUT = STK / f"{BASE}_unified_defects_accurate.json"

CONN = np.ones((3, 3, 3), int)
# Graph-cleanup geometry (node merging / spur pruning / min strut length) shapes
# the as-built graph itself, not the classification decision, and the same
# constants are duplicated in skel_to_json.py, clean_and_compare.py, and
# bent_struts.py. They stay hardcoded here rather than becoming a classification
# "threshold": changing them means rebuilding the graph in all four places
# consistently, which is a different, bigger operation than re-classifying.
D_MERGE = 20.0
SPUR_LEN = 30.0
MIN_STRUT = 30.0
STRUT_RADIUS = STRUT_RADIUS_VOX         # ~3.65 vox (config: LATTICE_STRUT_DIAMETER_UM / LATTICE_VOXEL_UM)
BENT_THR = STRUT_RADIUS * BENT_RADIUS_MULT      # config: LATTICE_BENT_RADIUS_MULT
SNAP_R = SNAP_R_VOX                     # config: LATTICE_SNAP_R_VOX
METAL_R = METAL_R_VOX                   # config: LATTICE_METAL_R_VOX
K_OUT = THIN_OUTLIER_K                  # config: LATTICE_THIN_OUTLIER_K
# MISSING_FRAC, GAP_FRAC imported directly from config (LATTICE_MISSING_FRAC / LATTICE_GAP_FRAC)


def longest_gap(hit):
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1; best = max(best, cur)
    return best


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

    # as-built radius per edge: same technique as skel_to_json.py Stage 3 (a
    # 2x-downsampled distance transform, rescaled by 2.0 back to full-res voxel
    # units) so this figure matches what the rest of the pipeline means by
    # "thickness" rather than introducing a second definition of it.
    m2 = mask[::2, ::2, ::2]
    edt2 = ndi.distance_transform_edt(m2)

    # bow + measured radius per edge
    edge_of = {}; bow = {}; thick = {}
    for e in E.values():
        p0, p1 = NP[e["u"]], NP[e["v"]]
        Q = np.concatenate(e["vox"]).astype(float) if e["vox"] else np.empty((0, 3))
        b = 0.0
        r = None
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
        if len(Q):
            qi = np.round(Q).astype(int)
            qi = np.clip(qi, 0, np.array(mask.shape) - 1)
            r = float(edt2[qi[:, 0] // 2, qi[:, 1] // 2, qi[:, 2] // 2].mean() * 2.0)
        key = (e["u"], e["v"])
        edge_of[key] = True; bow[key] = b; thick[key] = r
    del edt2, m2; gc.collect()
    return NP, edge_of, bow, thick


def main():
    print("loading mask + raw ...")
    mask = tifffile.imread(MASK) > 0
    metal = ndi.binary_dilation(mask, iterations=1)
    raw = tifffile.imread(RAW)
    shape = np.array(mask.shape)

    print("rebuilding as-built graph (cached skeleton) ...")
    NP, edge_of, bow, thick = build_asbuilt_graph(mask)
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
    struts = [(s["junction0"], s["junction1"], s.get("id", i), s.get("thickness"))
              for i, s in enumerate(gt["struts"])]
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
    anchor_kind = ["absent"] * len(GP)
    anchor_snap_dist = [None] * len(GP)
    n_snap = n_metal = n_absent = 0
    for i in range(len(GP)):
        if dd[i] < SNAP_R:
            anchor[i] = NP[idx[i]]; ab_node[i] = idx[i]; n_snap += 1
            anchor_kind[i] = "snapped_to_asbuilt_node"
            anchor_snap_dist[i] = float(dd[i])
        else:
            c = np.round(GPc[i]).astype(int)
            lo = np.maximum(c - METAL_R, 0); hi = np.minimum(c + METAL_R + 1, shape)
            sub = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            if sub.any():
                zz, yy, xx = np.where(sub)
                anchor[i] = [lo[0] + zz.mean(), lo[1] + yy.mean(), lo[2] + xx.mean()]
                n_metal += 1
                anchor_kind[i] = "local_metal_centroid"
            else:
                n_absent += 1
                anchor_kind[i] = "absent"
    print(f"  anchors: {n_snap} snapped to as-built node, {n_metal} to local metal, {n_absent} absent")

    # classify each designed strut
    counts = defaultdict(int)
    out_struts = []
    for a, b, sid, design_thickness in struts:
        na, nb = ab_node[a], ab_node[b]
        pa, pb = anchor[a], anchor[b]
        evidence = {
            "junction_a_anchor": anchor_kind[a],
            "junction_b_anchor": anchor_kind[b],
            "junction_a_snap_dist_vox": anchor_snap_dist[a],
            "junction_b_snap_dist_vox": anchor_snap_dist[b],
        }
        if np.isnan(pa).any() or np.isnan(pb).any():
            v = "missing"                    # a node region has no metal at all
            evidence["reason"] = "no_metal_at_anchor"
        elif na >= 0 and nb >= 0 and (min(na, nb), max(na, nb)) in edge_of:
            key = (min(na, nb), max(na, nb))
            evidence.update({
                "reason": "as_built_edge_matched",
                "as_built_node_a": int(na),
                "as_built_node_b": int(nb),
                "bow_um": float(bow[key] * UM),
                "bow_threshold_um": float(BENT_THR * UM),
                "mean_density": float(dens_of[key]),
                "density_cutoff": float(d_cut),
                "density_median": float(dmed),
                "density_mad_scaled": float(dsig),
                "outlier_k": K_OUT,
                # Measured via a 2x-downsampled distance transform along this
                # edge's own centerline voxels (same method as skel_to_json.py).
                # nominal_radius_um is the physical strut radius the pipeline's
                # own thresholds (e.g. BENT) are computed against
                # (config.STRUT_DIAMETER_UM/2) -- note this may differ from the
                # design file's own nominal "thickness" field on this strut
                # (see design_thickness above); both are reported as-is.
                "measured_radius_um": float(thick[key] * UM) if thick[key] is not None else None,
                "nominal_radius_um": float(STRUT_RADIUS * UM),
            })
            if bow[key] > BENT_THR:
                v = "bent"
            elif dens_of[key] < d_cut:
                v = "thin"
            else:
                v = "present"
        else:
            # no as-built edge: sample real material between the two anchors
            ts = np.linspace(0, 1, 24)
            pts = np.round(pa[None] * (1 - ts[:, None]) + pb[None] * ts[:, None]).astype(int)
            pts = np.clip(pts, 0, shape - 1)
            hit = metal[pts[:, 0], pts[:, 1], pts[:, 2]][2:-2]
            frac = hit.mean(); gap = longest_gap(hit) / len(hit)
            evidence.update({
                "reason": "material_sample_between_anchors",
                "metal_fraction": float(frac),
                "missing_fraction_threshold": MISSING_FRAC,
                "longest_gap_fraction": float(gap),
                "gap_fraction_threshold": GAP_FRAC,
            })
            if frac < MISSING_FRAC:
                v = "missing"
            elif gap >= GAP_FRAC:
                v = "disconnected"
            else:
                v = "present"
        counts[v] += 1
        # export in [x,y,z] scan coords (reverse zyx)
        out_struts.append({"id": int(sid),
                           "p0": [float(pa[2]), float(pa[1]), float(pa[0])] if not np.isnan(pa).any()
                           else [float(GPc[a][2]), float(GPc[a][1]), float(GPc[a][0])],
                           "p1": [float(pb[2]), float(pb[1]), float(pb[0])] if not np.isnan(pb).any()
                           else [float(GPc[b][2]), float(GPc[b][1]), float(GPc[b][0])],
                           "verdict": v,
                           "design_thickness": design_thickness,
                           "evidence": evidence})

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
