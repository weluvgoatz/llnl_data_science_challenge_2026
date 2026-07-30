"""Strut-defect detector v2 — self-contained, measurement-carrying, auditable.

Fully separate from the v1 pipeline (unified_defects_accurate.py): different
folder, different output files.  Nothing in v1 is overwritten.

Design principles, each one fixing a failure mode found during v1 manual
validation:

 1. ONE SOURCE OF TRUTH.  Every strut's JSON record stores the measurements the
    verdict was decided on (metal profile string, gap_um, stub_um, bow_um,
    dia_um, node states).  The validation renderer draws FROM THE RECORD, so a
    panel can never contradict its verdict.

 2. NODE STATES BEFORE STRUT VERDICTS.  A strut whose endpoint node was never
    printed is a VOID (subtype "void"), not a normal missing strut — the panel
    for it has no second node to show, which is correct and now labelled.
    States: "junction" (snapped to an as-built graph node), "partial" (metal in
    the node region but no junction), "absent" (no metal at all).

 3. TOPOLOGY FIRST, MATERIAL SECOND.  If the as-built skeleton graph has a
    direct edge between the two nodes, metal connects them (present family:
    present / bent / thin).  If not, the strut is followed with a perpendicular
    disk (R=6) and judged by its largest BREAK; a break must survive a WIDER
    disk (R=8) before it counts, because a thin+bowed strut can wander outside
    a narrow disk and fake a gap.  Perpendicular disk, never a 3-D box: a box
    smears along the axis and bridges genuine breaks.

 4. MISSING vs DISCONNECTED by SURVIVING MATERIAL, not by gap fraction.  The
    two node blobs always read as metal, so a gap fraction can never reach
    ~90% even for a totally absent strut.  Instead: does any continuous piece
    of strut >= 212 um (half a diameter) exist mid-span (20-80%, node blobs
    excluded)?  No -> missing.  Yes, but broken >= 424 um -> disconnected.

 5. GRAPH CONFIRMATION.  For every no-edge strut the shortest-path hop count
    between the two as-built nodes is recorded (BFS).  A genuine missing strut
    between two intact nodes shows hops >= 2 — an independent, topology-only
    confirmation of the material verdict.

 6. GEOMETRY MEASURED WHERE IT IS SHOWN.  bent = sustained bow of the actual
    metal centreline (3-sample rolling median, mid-span, unsmoothed data —
    max-of-noise cannot fake it); thin = median EDT diameter of the actual
    centreline.  The same numbers appear on the panels.

 7. CONTEXT FLAGS.  Struts within 45 vox of the printed lattice's bounding
    faces, or in the fused end-caps (z<70 or z>690), are flagged "boundary":
    individually real absences, collectively an artifact of comparing a
    perfect infinite design against a finite printed part.  Interior-context
    verdicts are the defensible ones; the JSON carries both, labelled.

Output:  v2/strut_classification_v2.json  +  v2/RESULTS_V2.md
"""

import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import STK, BASE, DESIGN_JSON, VOXEL_UM, STRUT_DIAMETER_UM  # noqa: E402

V2 = Path(__file__).resolve().parent
RAW_P = STK / f"{BASE}.tif"
# Namespaced with a _v2 suffix so this detector never loads a mask/skeleton
# cache left by the v1 pipeline (config.MASK / config.SKELC, written by
# clean_segmentation.py / skel_to_json.py) -- v1 uses per-slice adaptive Otsu,
# v2 uses a single global Otsu (segment.py); the two are not interchangeable.
# A stale v1-named file at the old path is simply never looked at.
MASK_P = STK / f"{BASE}_segmented_clean_v2.tif"
SKELC = STK / f"{BASE}_segmented_clean_v2.skelcoords.npz"
V1_JSON = STK / f"{BASE}_unified_defects_accurate.json"
OUT_JSON = V2 / "strut_classification_v2.json"
OUT_MD = V2 / "RESULTS_V2.md"

UM = VOXEL_UM                          # 58.1 um / voxel
NOM_D = STRUT_DIAMETER_UM              # 424 um nominal strut diameter

# ------------------------------------------------------------------ REFERENCE --
# THE VALIDATED STATIC BASELINE.  Every voxel-unit constant in this file is
# written for the specimen the detector was manually validated on (200/200
# panels reviewed): 424 um struts at 58.1 um/voxel, design pitch 55.85 vox,
# fused end-caps hand-set to z 70..690.  calibrate() rescales those constants
# from the ACTUAL voxel size / strut diameter / design pitch of whatever scan is
# loaded, so a different TIF + design JSON is handled correctly.  For the
# reference specimen every scale factor is exactly 1.0, so the validated numbers
# reproduce unchanged — that is the regression test.
REF_VOXEL_UM = 58.1
REF_STRUT_DIAMETER_UM = 424.0
REF_STRUT_R = REF_STRUT_DIAMETER_UM / 2 / REF_VOXEL_UM     # 3.6489 vox
REF_STRUT_L = 55.84560994969745                            # design pitch, vox (EXACT)
REF_CAP_Z = (70, 690)                                      # hand-set, validated
CAP_MULT = 1.5                     # cap where smoothed slice metal > this x baseline
# Scale factors within this tolerance of 1.0 are snapped to exactly 1.0.  Node
# merging uses a KD-tree radius query, so a scale of 0.999993 instead of 1.0
# moves the merge radius by 1e-4 vox and flips node pairs sitting at exactly
# that distance — churning the graph and a handful of verdicts for no physical
# reason.  A pitch within 0.1% is the same geometry, so treat it as identical.
SCALE_SNAP_TOL = 1e-3

# ---- thresholds (all physically motivated; documented in module docstring) ----
DISK_R1 = 6.0                          # centroid-path perpendicular search radius
CORR_R = 8.0                           # corridor radius for the connectivity test
END_R = 4.0                            # corridor metal this close to an endpoint seeds its component
LOCAL_SNAP = 5.0                       # local-connectivity test: point snaps to metal within this
PROF_K = 60                            # samples along the span for profiles
STUB_MIN_UM = NOM_D / 2                # surviving material must be >= this
MID_LO, MID_HI = 0.20, 0.80            # mid-span window (node blobs excluded)
BENT_BOW_UM = NOM_D / 2                # sustained centreline bow >= 1 radius
CPATH_K = 61                           # centreline samples (full span)
NODE_R = 6.0                           # metal within this radius = node has metal
BOUND_MARGIN = 45                      # vox from printed bbox face -> boundary
CAP_Z = REF_CAP_Z                      # fused end-caps; auto-detected in calibrate()
BOW_SPAN_LO, BOW_SPAN_HI = 30.0, 90.0  # plausible skeleton-segment span, vox
VERT_DZ = 20.0                         # |dz| above this = a z-climbing strut
# graph build (identical to the proven v1 values)
D_MERGE = 20.0
SPUR_LEN = 30.0
MIN_STRUT = 30.0
SNAP_R = 14.0                          # design node -> as-built node snap radius
METAL_R = 11                           # else search this ball for any node metal
CONN = np.ones((3, 3, 3), int)


# -------------------------------------------------------------- calibration ----
def detect_caps(mask, strut_len):
    """Find the open-lattice z-range, excluding fused end-caps.

    A cap slice is nearly solid metal (~70% here) while an open-lattice slice is
    a few percent — but the raw per-slice fraction OSCILLATES, because every node
    layer spikes it (up to 3x baseline).  Thresholding the raw profile therefore
    misfires inside the lattice.  Smoothing over one node-layer period (half a
    strut pitch) removes the oscillation and leaves the two regimes cleanly
    separated, then we walk inward from each end until the profile settles.
    """
    frac = mask.reshape(mask.shape[0], -1).mean(axis=1)
    per = max(int(round(strut_len / 2)), 3)
    sm = ndi.uniform_filter1d(frac, size=per, mode="nearest")
    core = sm[int(0.25 * len(sm)):int(0.75 * len(sm))]
    base = float(np.median(core))
    thr = CAP_MULT * base
    Z = len(sm)
    lo = 0
    while lo < Z - 1 and sm[lo] > thr:
        lo += 1
    hi = Z - 1
    while hi > lo and sm[hi] > thr:
        hi -= 1
    return int(lo), int(hi), base


def calibrate(design_len_med, mask, report=True):
    """Rescale every voxel-unit constant for the scan actually loaded.

    Radius-like quantities scale with the strut radius in voxels; length-like
    quantities scale with the design pitch.  Both factors are exactly 1.0 for
    the reference specimen, so its validated results are reproduced unchanged.
    """
    global CORR_R, END_R, LOCAL_SNAP, NODE_R, DISK_R1, OFF1
    global BOUND_MARGIN, D_MERGE, SPUR_LEN, MIN_STRUT, SNAP_R, METAL_R
    global BOW_SPAN_LO, BOW_SPAN_HI, VERT_DZ, CAP_Z

    r_scale = (NOM_D / 2 / UM) / REF_STRUT_R
    l_scale = design_len_med / REF_STRUT_L
    if abs(r_scale - 1.0) < SCALE_SNAP_TOL:
        r_scale = 1.0
    if abs(l_scale - 1.0) < SCALE_SNAP_TOL:
        l_scale = 1.0

    CORR_R *= r_scale; END_R *= r_scale; LOCAL_SNAP *= r_scale
    NODE_R *= r_scale; DISK_R1 *= r_scale
    OFF1 = _offsets(DISK_R1)                      # depends on DISK_R1

    BOUND_MARGIN *= l_scale; D_MERGE *= l_scale; SPUR_LEN *= l_scale
    MIN_STRUT *= l_scale; SNAP_R *= l_scale; METAL_R = int(round(METAL_R * l_scale))
    BOW_SPAN_LO *= l_scale; BOW_SPAN_HI *= l_scale; VERT_DZ *= l_scale

    env = os.environ.get("LATTICE_CAP_Z")
    if env:                                       # explicit override, e.g. "70,690"
        a, b = (int(v) for v in env.split(","))
        CAP_Z = (a, b); cap_src = "env LATTICE_CAP_Z"
        cap_base = None
    else:
        a, b, cap_base = detect_caps(mask, design_len_med)
        CAP_Z = (a, b); cap_src = "auto-detected"

    if report:
        print(f"  calibration: radius x{r_scale:.4f}, length x{l_scale:.4f} "
              f"(1.0000 = the validated reference specimen)", flush=True)
        print(f"    corridor R {CORR_R:.2f} · disk R {DISK_R1:.2f} · node R {NODE_R:.2f} "
              f"· snap {SNAP_R:.1f} · merge {D_MERGE:.1f} · boundary margin {BOUND_MARGIN:.1f}",
              flush=True)
        msg = f"    open lattice z {CAP_Z[0]}..{CAP_Z[1]} ({cap_src}"
        if cap_base is not None:
            msg += f", baseline metal {cap_base:.3f}"
        msg += f"); validated static was {REF_CAP_Z[0]}..{REF_CAP_Z[1]}"
        print(msg, flush=True)
    return {"radius_scale": round(r_scale, 6), "length_scale": round(l_scale, 6),
            "cap_z": list(CAP_Z), "cap_source": cap_src,
            "static_reference_cap_z": list(REF_CAP_Z)}


# ---------------------------------------------------------------- geometry ----
def _perp(u):
    t = np.array([1., 0, 0]) if abs(u[0]) < 0.9 else np.array([0., 1, 0])
    e1 = np.cross(u, t); e1 /= np.linalg.norm(e1) + 1e-12
    return e1, np.cross(u, e1)


def _offsets(R):
    r = int(np.ceil(R))
    pts = [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1)
           if a * a + b * b <= R * R]
    return np.array(pts, float)


OFF1 = _offsets(DISK_R1)


def corridor_test(pa, pb, metal, shape, R=CORR_R, K=PROF_K):
    """Connectivity + material profile inside a radius-R corridor around the strut.

    PRESENCE is not CONNECTION.  A strut that detached at the joint leaves only a
    hairline crack — a step-by-step presence profile bridges straight past it (and
    junction metal masks it near the node), which is exactly how real disconnected
    struts were being missed.  Here the metal voxels inside the corridor are
    LABELLED into connected components, and the strut counts as connected only if
    ONE component reaches from node A to node B.  Restricting the labelling to the
    corridor also prevents the opposite error: being "connected" the long way
    round through the rest of the lattice.

    Returns: connected, prof[K] (any corridor metal per axial bin — the bar shown
    on validation panels), gap_um (longest empty axial run), stub_um (longest
    continuous material run in the mid-window, where neighbouring struts cannot
    reach the corridor, so it is strut material only).
    """
    d = pb - pa
    L = float(np.linalg.norm(d))
    out = {"connected": False, "prof": np.zeros(K, bool),
           "gap_um": L * UM, "stub_um": 0.0}
    if L < 1e-6:
        return out
    u = d / L
    step_um = L / (K - 1) * UM
    lo = np.maximum(np.floor(np.minimum(pa, pb) - R - 1).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(pa, pb) + R + 2).astype(int), shape)
    sub = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if not sub.any():
        return out
    zz, yy, xx = np.nonzero(sub)
    P = np.stack([zz, yy, xx], 1).astype(np.float64) + lo
    rel = P - pa
    t = rel @ u
    tc = np.clip(t, 0.0, L)
    foot = pa[None, :] + tc[:, None] * u[None, :]
    dseg = np.linalg.norm(P - foot, axis=1)
    inC = dseg <= R
    if not inC.any():
        return out
    Pc = P[inC]
    tin = t[inC]
    idx = (Pc - lo).astype(int)
    volc = np.zeros(sub.shape, bool)
    volc[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, _ = ndi.label(volc, structure=CONN)
    lv = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    dA = np.linalg.norm(Pc - pa, axis=1)
    dB = np.linalg.norm(Pc - pb, axis=1)
    la = set(np.unique(lv[dA <= END_R]).tolist())
    lb = set(np.unique(lv[dB <= END_R]).tolist())
    out["connected"] = len(la & lb) > 0
    prof = np.zeros(K, bool)
    bins = np.clip(np.round(tin / L * (K - 1)).astype(int), 0, K - 1)
    prof[np.unique(bins)] = True
    out["prof"] = prof
    out["gap_um"] = longest_gap(prof) * step_um
    k0, k1 = int(MID_LO * K), int(MID_HI * K)
    out["stub_um"] = longest_run(prof[k0:k1]) * step_um
    return out


def locally_connected(p1, p2, metal, shape, snap=LOCAL_SNAP):
    """Are the metal voxels at p1 and p2 one LOCAL connected component?

    The decisive phantom-edge test.  A genuine strut termination is materially
    fused to its junction core; a broken stub whose tip cluster got union-merged
    into the node (creating a phantom edge) has a real air gap there.  Labelling
    the metal in the small box spanning both points answers it with no tunable
    distance threshold to fool.
    """
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    lo = np.maximum(np.floor(np.minimum(p1, p2) - 6).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(p1, p2) + 7).astype(int), shape)
    sub = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if not sub.any():
        return False
    lab, _ = ndi.label(sub, structure=CONN)
    pts = np.argwhere(sub) + lo

    def lab_of(p):
        d = np.linalg.norm(pts - p, axis=1)
        i = int(np.argmin(d))
        if d[i] > snap:
            return 0
        q = pts[i] - lo
        return int(lab[q[0], q[1], q[2]])

    l1 = lab_of(p1); l2 = lab_of(p2)
    return l1 > 0 and l1 == l2


def longest_gap(h):
    best = cur = 0
    for v in h:
        cur = 0 if v else cur + 1
        best = max(best, cur)
    return best


def longest_run(h):
    best = cur = 0
    for v in h:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def prof_str(h):
    return "".join("#" if v else "." for v in h)


def centroid_path(pa, pb, mask, shape, K=CPATH_K, lo=0.0, hi=1.0):
    """(centres[K,3], ts[K]): metal centroid of the perpendicular disk at each
    step along the full span.  NaN rows where the disk holds no metal."""
    d = pb - pa
    L = float(np.linalg.norm(d))
    u = d / max(L, 1e-9)
    e1, e2 = _perp(u)
    ts = np.linspace(lo, hi, K)
    C = pa[None, :] + ts[:, None] * d[None, :]
    P = (C[:, None, :]
         + OFF1[None, :, 0, None] * e1[None, None, :]
         + OFF1[None, :, 1, None] * e2[None, None, :])
    Q = np.round(P).astype(np.int64)
    np.clip(Q, 0, shape - 1, out=Q)
    m = mask[Q[..., 0], Q[..., 1], Q[..., 2]]
    cnt = m.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore"):
        cent = (P * m[..., None]).sum(axis=1) / np.where(cnt > 0, cnt, np.nan)[:, None]
    return cent, ts


def bow_from_path_um(cent, ts, lo=0.15, hi=0.85):
    """Fallback bow for struts WITHOUT a skeleton edge: sustained deviation of
    the metal-centroid path from the chord between its own mid-window endpoints.

    Mid-window only (0.15-0.85): nearer the junctions, neighbouring struts enter
    the search disk and drag the centroid — that contamination is smooth enough
    to survive a rolling median and was the source of inflated bows.  The
    primary bow measure is the skeleton path (edge_bow), which cannot be
    contaminated; this fallback slightly UNDERSTATES a full arc (it sees only
    the middle of it), which is the safe direction for a defect flag."""
    ok = np.where(~np.isnan(cent[:, 0]) & (ts >= lo) & (ts <= hi))[0]
    if len(ok) < 6:
        return None
    p_start = cent[ok[:2]].mean(axis=0)
    p_end = cent[ok[-2:]].mean(axis=0)
    axis = p_end - p_start
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return None
    axis /= L
    v = cent - p_start[None, :]
    proj = np.nansum(v * axis[None, :], axis=1)
    perp = v - proj[:, None] * axis[None, :]
    dev = np.linalg.norm(perp, axis=1)
    dev[np.isnan(cent[:, 0]) | (ts < lo) | (ts > hi)] = np.nan
    best = 0.0
    for i in range(len(dev)):
        w = dev[max(0, i - 1):i + 2]
        w = w[~np.isnan(w)]
        if len(w):
            best = max(best, float(np.median(w)))
    return best * UM


# ------------------------------------------------------------- graph build ----
def load_or_build_mask():
    """The clean metal mask: loaded if cached, otherwise segmented from the raw TIF.

    This is what makes the detector self-contained — give it a raw CT stack and a
    registered design JSON and nothing else is needed.  The segmentation is the
    exact pipeline the validated mask was built with (see segment.py), and the
    result is cached as a TIF so the slow step runs once.
    """
    if MASK_P.exists():
        print(f"v2: loading cached mask {MASK_P.name} ...", flush=True)
        return tifffile.imread(MASK_P) > 0
    if not RAW_P.exists():
        raise FileNotFoundError(
            f"need either a cached mask ({MASK_P.name}) or the raw scan "
            f"({RAW_P.name}) in {STK}")
    print(f"v2: no mask cached; segmenting {RAW_P.name} ...", flush=True)
    from segment import segment_volume
    vol = tifffile.imread(RAW_P)
    mask, thr = segment_volume(vol)
    del vol
    gc.collect()
    tifffile.imwrite(MASK_P, (mask.astype(np.uint8) * 255), compression="zlib")
    print(f"    cached mask -> {MASK_P.name} (Otsu {thr:.0f})", flush=True)
    return mask


def load_or_build_skeleton(mask):
    """Skeleton voxel coordinates, computed and cached if not present.

    A new scan will not have the .skelcoords.npz cache, so build it rather than
    failing — this is what lets the detector run on a TIF it has never seen.
    """
    if SKELC.exists():
        return np.load(SKELC)["coords"]
    print(f"  no skeleton cache; skeletonising (one-off, slow) -> {SKELC.name}",
          flush=True)
    from skimage.morphology import skeletonize
    skel = skeletonize(mask)
    coords = np.argwhere(skel).astype(np.int16)
    np.savez_compressed(SKELC, coords=coords)
    del skel; gc.collect()
    return coords


def build_asbuilt_graph(mask):
    """cached skeleton -> cleaned graph (proven v1 logic, copied verbatim)."""
    coords = load_or_build_skeleton(mask)
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

    edges = {}; eid = 0
    for e, ns in seg_nodes.items():
        if len(ns) != 2:
            continue
        u, v = sorted(ns)
        edges[eid] = {"u": u, "v": v,
                      "vox": [seg_coords.get(e - 1, np.empty((0, 3), np.float32))]}
        eid += 1

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
    # per merged node: the LARGEST member cluster's centroid = the junction CORE.
    # (The merged-group mean can sit well off the true junction; the core is
    # the physical reference for "does this strut actually reach its node?")
    csize = cnt[1:]
    CORE = np.array([centroids[idxs[int(np.argmax(csize[np.array(idxs)]))]]
                     for idxs in groups.values()])
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
        nv = E[e1]["vox"] + E[e2]["vox"]
        nid = ne; ne += 1
        E[nid] = {"u": min(a, b), "v": max(a, b), "vox": nv}
        for e in (e1, e2):
            E.pop(e, None)
        A[a].discard(e1); A[a].add(nid); A[b].discard(e2); A[b].add(nid); A[n].clear()
        if len(A[a]) == 2:
            stack.append(a)
        if len(A[b]) == 2:
            stack.append(b)

    def clen(e):
        return float(np.linalg.norm(NP[e["u"]] - NP[e["v"]]))

    changed = True
    while changed:
        changed = False
        dcount = defaultdict(int)
        for e in E.values():
            dcount[e["u"]] += 1; dcount[e["v"]] += 1
        for k in list(E):
            if clen(E[k]) < SPUR_LEN and (dcount[E[k]["u"]] == 1 or dcount[E[k]["v"]] == 1):
                E.pop(k); changed = True
    dcount = defaultdict(int)
    for e in E.values():
        dcount[e["u"]] += 1; dcount[e["v"]] += 1
    for k in list(E):
        if clen(E[k]) < MIN_STRUT and dcount[E[k]["u"]] >= 3 and dcount[E[k]["v"]] >= 3:
            E.pop(k)

    # per-edge skeleton bow + polyline.  The skeleton is a 1-voxel centreline
    # BELONGING TO THIS STRUT ONLY, so unlike a metal-centroid path it cannot be
    # dragged sideways by neighbouring struts near the junctions — that
    # contamination is exactly what inflated the centroid-based bow.
    edge_set = set()
    edge_bow = {}
    edge_path = {}
    edge_ends = {}
    for e in E.values():
        key = (e["u"], e["v"])
        edge_set.add(key)
        edge_bow[key] = None
        Q = (np.concatenate(e["vox"]).astype(float)
             if e["vox"] else np.empty((0, 3)))
        if len(Q) < 8:
            continue
        cen = Q.mean(0); X = Q - cen
        axis = np.linalg.svd(X, full_matrices=False)[2][0]
        Qs = Q[np.argsort(X @ axis)]
        a0, b0 = Qs[0], Qs[-1]
        steps = np.linalg.norm(np.diff(Qs, axis=0), axis=1)
        if not len(steps) or steps.max() > 3.0:
            continue                       # fragmented path — ends unreliable
        # a CONTIGUOUS path's endpoints, for the phantom-edge test: each end
        # must be locally fused to its junction core, else the "edge" is a
        # broken stub whose tip cluster got union-merged into the node.
        edge_ends[key] = (a0.copy(), b0.copy())
        ab = b0 - a0
        span = float(np.linalg.norm(ab))
        if span < BOW_SPAN_LO or span > BOW_SPAN_HI:
            continue                       # degenerate / doubly-merged segment
        tt = np.clip(((Qs - a0) @ ab) / float(ab @ ab), 0, 1)
        dev = np.linalg.norm(Qs - (a0 + tt[:, None] * ab), axis=1)
        # sustained: max of 3-sample rolling medians along the ordered path
        best = 0.0
        for i in range(len(dev)):
            w = dev[max(0, i - 1):i + 2]
            best = max(best, float(np.median(w)))
        edge_bow[key] = best * UM
        edge_path[key] = Qs[::2]
    return NP, CORE, edge_set, edge_bow, edge_path, edge_ends


# ---------------------------------------------------------------- EDT dias ----
_NBR = np.array([[dz, dy, dx] for dz in (-1, 0, 1)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)])


def edt_radii(mask, pts):
    """Local strut radius at integer points: the MAXIMUM of the EDT (distance to
    background) over the point's 3x3x3 neighbourhood, via overlapping z-slabs.

    The max, not the single voxel: the sample point is a metal-centroid estimate
    that can ride a voxel or two off the strut's true centre (roughness,
    porosity), where the EDT reads low and fakes a neck.  The neighbourhood max
    recovers the true in-radius of the cross-section regardless of that drift.
    """
    if not len(pts):
        return np.zeros(0, np.float32)
    Z = mask.shape[0]
    PAD, CORE = 14, 110
    order = np.argsort(pts[:, 0], kind="stable")
    ps = pts[order]
    res = np.zeros(len(ps), np.float32)
    z0 = 0
    while z0 < Z:
        z1 = min(z0 + CORE, Z)
        a, b = max(z0 - PAD, 0), min(z1 + PAD, Z)
        sel = (ps[:, 0] >= z0) & (ps[:, 0] < z1)
        if sel.any():
            ed = ndi.distance_transform_edt(mask[a:b])
            q = ps[sel]
            qn = q[:, None, :] + _NBR[None, :, :]              # (n, 27, 3)
            qn[..., 0] -= a
            np.clip(qn[..., 0], 0, ed.shape[0] - 1, out=qn[..., 0])
            np.clip(qn[..., 1], 0, ed.shape[1] - 1, out=qn[..., 1])
            np.clip(qn[..., 2], 0, ed.shape[2] - 1, out=qn[..., 2])
            res[sel] = ed[qn[..., 0], qn[..., 1], qn[..., 2]].max(axis=1)
            del ed; gc.collect()
        z0 = z1
    out = np.zeros(len(ps), np.float32)
    out[order] = res
    return out


# --------------------------------------------------------------------- main ----
def main():
    # load_or_build_mask() -- not a raw tifffile.imread(MASK_P) -- is what
    # makes this self-contained on a scan with no cache yet (see its own
    # docstring); calling imread directly here would make every fresh job
    # fail with FileNotFoundError instead of self-segmenting.
    mask = load_or_build_mask()
    metal = ndi.binary_dilation(mask, iterations=1)
    shape = np.array(mask.shape)
    nz = np.argwhere(mask)
    LOB, HIB = nz.min(0), nz.max(0)
    del nz; gc.collect()

    # design first, so the pitch is known before anything voxel-scaled is used
    gt = json.load(open(DESIGN_JSON))
    GP = np.array([j["position"] for j in gt["junctions"]], float)[:, ::-1]
    struts = [(s["junction0"], s["junction1"]) for s in gt["struts"]]
    _gpx = GP[:, ::-1]
    design_len_med = float(np.median([
        np.linalg.norm(_gpx[a] - _gpx[b]) for a, b in struts]))
    print(f"v2: design {len(GP)} junctions, {len(struts)} struts, "
          f"median pitch {design_len_med:.2f} vox", flush=True)
    calib = calibrate(design_len_med, mask)

    print("v2: building as-built graph ...", flush=True)
    NPn, COREn, edge_set, edge_bow, edge_path, edge_ends = build_asbuilt_graph(mask)
    print(f"  as-built: {len(NPn)} nodes, {len(edge_set)} struts "
          f"({sum(1 for b in edge_bow.values() if b is not None)} with skeleton bow)", flush=True)
    adjn = defaultdict(set)
    for (u, v) in edge_set:
        adjn[u].add(v); adjn[v].add(u)

    def hops(a, b, maxd=4):
        """BFS hop count in the as-built graph (graph shortest-distance check)."""
        if a < 0 or b < 0:
            return -1
        if a == b:
            return 0
        seen = {a}; frontier = {a}
        for dd in range(1, maxd + 1):
            nxt = set()
            for n in frontier:
                nxt |= adjn[n]
            if b in nxt:
                return dd
            nxt -= seen; seen |= nxt; frontier = nxt
            if not frontier:
                break
        return 99

    # ---- affine ICP refinement (proven v1 registration) ----
    tree = cKDTree(NPn)
    GPc = GP.copy()
    for _ in range(30):
        dd, idx = tree.query(GPc)
        m = dd < 0.806 * design_len_med       # ICP inliers (45 vox at the reference pitch)
        G1 = np.hstack([GP[m], np.ones((int(m.sum()), 1))])
        T, *_ = np.linalg.lstsq(G1, NPn[idx[m]], rcond=None)
        GPc = np.hstack([GP, np.ones((len(GP), 1))]) @ T
    print("v2: affine registration done", flush=True)

    # ---- per design node: state + anchor ----
    dd, idx = tree.query(GPc)
    n_nodes = len(GP)
    anchor = GPc.copy()
    ab_node = np.full(n_nodes, -1, int)
    state = np.empty(n_nodes, object)
    for i in range(n_nodes):
        if dd[i] < SNAP_R:
            anchor[i] = NPn[idx[i]]
            ab_node[i] = idx[i]
            state[i] = "junction"
            continue
        c = np.round(GPc[i]).astype(int)
        lo = np.maximum(c - METAL_R, 0)
        hi = np.minimum(c + METAL_R + 1, shape)
        sub = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if sub.any():
            zz, yy, xx = np.where(sub)
            pts = np.stack([zz, yy, xx], 1) + lo
            dist = np.linalg.norm(pts - GPc[i], axis=1)
            # "partial" only if metal is genuinely AT the node position (within
            # NODE_R vox).  A few debris voxels at the far edge of the search
            # ball do not make a node - that is exactly the "where is the
            # second node?" failure a validation panel must never show.
            if dist.min() <= NODE_R:
                near = pts[dist <= NODE_R + 2]
                anchor[i] = near.mean(0)
                state[i] = "partial"
            else:
                state[i] = "absent"
        else:
            state[i] = "absent"
    sc = defaultdict(int)
    for s in state:
        sc[s] += 1
    print(f"  node states: {dict(sc)}", flush=True)

    def boundary(p):
        if np.min(np.minimum(p - LOB, HIB - p)) < BOUND_MARGIN:
            return True
        return p[0] < CAP_Z[0] or p[0] > CAP_Z[1]

    # ---- phase A: topology + material families ----
    print("v2: phase A — topology + break profiles ...", flush=True)
    recs = []
    family = []                      # index lists for phase B
    fam_edge = {}                    # strut index -> as-built edge key
    audit = defaultdict(int)
    for si, (a, b) in enumerate(struts):
        pa, pb = anchor[a], anchor[b]
        L_um = float(np.linalg.norm(pb - pa)) * UM
        ctx = "boundary" if (boundary(pa) or boundary(pb)) else "interior"
        na, nb = ab_node[a], ab_node[b]
        key = (min(na, nb), max(na, nb)) if (na >= 0 and nb >= 0) else None
        has_edge = key in edge_set if key is not None else False
        rec = {"i": si,
               "p0": [round(float(pa[2]), 2), round(float(pa[1]), 2), round(float(pa[0]), 2)],
               "p1": [round(float(pb[2]), 2), round(float(pb[1]), 2), round(float(pb[0]), 2)],
               "node0": state[a], "node1": state[b],
               "context": ctx, "edge": bool(has_edge), "len_um": round(L_um, 0)}

        # ONE test for every strut: is the metal inside the corridor actually
        # CONNECTED from node to node?  (presence-profiles bridge hairline
        # cracks and junction-adjacent breaks — connectivity does not.)
        cor = corridor_test(pa, pb, metal, shape)
        rec["prof"] = prof_str(cor["prof"])
        rec["prof_r"] = CORR_R
        rec["gap_um"] = round(cor["gap_um"], 0)
        rec["stub_um"] = round(cor["stub_um"], 0)

        broken = not cor["connected"]
        if has_edge and broken:
            # The skeleton claims a connection the corridor cannot see (a
            # displaced or bowed strut can exit the corridor).  A contiguous
            # skeleton path is topological proof of connection PROVIDED each of
            # its ends is locally fused to its junction core — a broken stub
            # whose tip cluster got union-merged into the node (phantom edge)
            # fails that local test at the gap end.
            ends = edge_ends.get(key)
            if ends is not None:
                a0, b0 = ends
                cu, cv = COREn[key[0]], COREn[key[1]]
                if (np.linalg.norm(a0 - cu) + np.linalg.norm(b0 - cv)
                        > np.linalg.norm(a0 - cv) + np.linalg.norm(b0 - cu)):
                    a0, b0 = b0, a0
                if (locally_connected(a0, cu, metal, shape)
                        and locally_connected(b0, cv, metal, shape)):
                    broken = False
                    audit["edge_rescues_corridor"] += 1
                else:
                    audit["phantom_edge_broken"] += 1
                    rec["flag_phantom_edge"] = True
            else:
                # fragmented skeleton segment — no proof either way; the
                # corridor's material verdict stands.
                audit["fragmented_edge_corridor_broken"] += 1

        if not broken:
            family.append(si)
            if has_edge:
                fam_edge[si] = key
            recs.append(rec)
            continue

        rec["hops"] = hops(na, nb)
        n_absent = int(state[a] == "absent") + int(state[b] == "absent")
        if rec["stub_um"] <= STUB_MIN_UM:
            v = "missing"
            subtype = ("dropped", "node_lost", "void")[n_absent]
        else:
            # the user's rule: if any real piece of strut was built, it is
            # BROKEN, not missing — regardless of how the nodes fared.
            v = "disconnected"
            subtype = (None, "node_lost", "void")[n_absent]
        rec.update(verdict=v, subtype=subtype)
        recs.append(rec)

    n_broken = sum(1 for r in recs if "verdict" in r)
    print(f"  phase A: {n_broken} missing/disconnected/void, "
          f"{len(family)} present-family to measure", flush=True)

    # ---- phase B: geometry (bow + diameter) for the present family ----
    print("v2: phase B — centrelines, bow, diameters ...", flush=True)
    cent_pts = []
    cent_ptr = {}
    cent_ts = {}
    for si in family:
        r = recs[si]
        pa = np.array(r["p0"][::-1], float)
        pb = np.array(r["p1"][::-1], float)
        cent, ts = centroid_path(pa, pb, mask, shape)
        key = fam_edge.get(si)
        bw = edge_bow.get(key) if key is not None else None
        r["bow_src"] = "skeleton" if bw is not None else "centroid"
        if bw is None:
            bw = bow_from_path_um(cent, ts)
        r["bow_um"] = round(bw, 0) if bw is not None else None
        # diameters sampled mid-span only (node blobs would read fat)
        mid = (ts >= MID_LO) & (ts <= MID_HI) & ~np.isnan(cent[:, 0])
        pts = np.round(cent[mid]).astype(np.int64)
        np.clip(pts, 0, shape - 1, out=pts)
        cent_ptr[si] = (len(cent_pts), len(pts))
        cent_ts[si] = ts[mid]
        cent_pts.extend(pts.tolist())
    cent_pts = np.array(cent_pts, np.int64) if cent_pts else np.zeros((0, 3), np.int64)
    print(f"  sampling EDT at {len(cent_pts)} centreline points ...", flush=True)
    radii = edt_radii(mask, cent_pts)
    NECK_LO, NECK_HI = 0.25, 0.75
    for si in family:
        o, n = cent_ptr[si]
        rr = radii[o:o + n].astype(float)          # ordered along the strut
        rr[rr <= 0] = np.nan
        tss = cent_ts[si]
        r = recs[si]
        if np.isfinite(rr).sum() >= 5:
            r["dia_um"] = round(float(np.nanmedian(rr)) * 2 * UM, 0)
            # thinnest SUSTAINED section (3-sample rolling median): catches a
            # locally necked strut whose median diameter looks normal.  Windowed
            # to 0.25-0.75 of the span: the natural fillet where a strut meets
            # its node is not a defect.
            best = np.inf
            for i in range(len(rr)):
                if not (NECK_LO <= tss[i] <= NECK_HI):
                    continue
                w = rr[max(0, i - 1):i + 2]
                w = w[np.isfinite(w)]
                if len(w) >= 2:
                    best = min(best, float(np.median(w)))
            r["dia_min_um"] = round(best * 2 * UM, 0) if np.isfinite(best) else None
        else:
            r["dia_um"] = None
            r["dia_min_um"] = None

    dias = np.array([recs[si]["dia_um"] for si in family
                     if recs[si]["dia_um"] is not None], float)
    dmed = float(np.median(dias))
    dmad = float(np.median(np.abs(dias - dmed)))
    thin_cut = dmed - 3 * dmad
    print(f"  diameter population: median {dmed:.0f} um, MAD {dmad:.0f} -> thin < {thin_cut:.0f} um", flush=True)

    for si in family:
        r = recs[si]
        flags = []
        # thin takes precedence: an extremely thin strut also wanders, which
        # reads as bow — its thinness is the primary defect, not the wander.
        # A strut is thin if ANY sustained section is below the cutoff: this
        # catches hairline struts (median low) AND locally necked ones.
        if r["dia_min_um"] is not None and r["dia_min_um"] < thin_cut:
            flags.append("thin")
            if r["dia_um"] is not None and r["dia_um"] >= thin_cut:
                audit["thin_necked"] += 1
            else:
                audit["thin_hairline"] += 1
        if r["bow_um"] is not None and r["bow_um"] > BENT_BOW_UM:
            flags.append("bent")
        r["flags"] = flags
        r["verdict"] = flags[0] if flags else "present"
        r["subtype"] = None
        if r["verdict"] == "bent":
            key = fam_edge.get(si)
            if key is not None and key in edge_path:
                # the exact skeleton polyline the bow was measured on ([x,y,z])
                r["path"] = [[round(float(p[2]), 1), round(float(p[1]), 1),
                              round(float(p[0]), 1)] for p in edge_path[key]]

    # ---- summary + audits ----
    counts = defaultdict(int)
    ic = defaultdict(int)
    for r in recs:
        counts[r["verdict"]] += 1
        if r["context"] == "interior":
            ic[r["verdict"]] += 1
    n = len(recs)

    miss_drop = [r for r in recs if r["verdict"] == "missing" and r["subtype"] == "dropped"]
    miss_nl = [r for r in recs if r["verdict"] == "missing" and r["subtype"] == "node_lost"]
    miss_void = [r for r in recs if r["verdict"] == "missing" and r["subtype"] == "void"]
    disc_nl = [r for r in recs if r["verdict"] == "disconnected" and r.get("subtype") == "node_lost"]
    g2 = sum(1 for r in miss_drop if r.get("hops", -1) >= 2)
    gk = sum(1 for r in miss_drop if r.get("hops", -1) >= 0)
    audit["missing_dropped"] = len(miss_drop)
    audit["missing_node_lost"] = len(miss_nl)
    audit["missing_void"] = len(miss_void)
    audit["disconnected_node_lost"] = len(disc_nl)
    audit["missing_dropped_graph_confirmed"] = g2
    audit["missing_dropped_graph_known"] = gk

    bows_p = np.array([r["bow_um"] for r in recs if r["verdict"] == "present" and r.get("bow_um") is not None], float)
    bows_b = np.array([r["bow_um"] for r in recs if r["verdict"] == "bent"], float)
    dias_p = np.array([r["dia_um"] for r in recs if r["verdict"] == "present" and r.get("dia_um") is not None], float)
    dias_t = np.array([r["dia_um"] for r in recs if r["verdict"] == "thin" and r.get("dia_um") is not None], float)

    meta = {"model": "v2", "n": n, "counts": dict(counts),
            "counts_interior": dict(ic),
            "thresholds": {"corridor_r": CORR_R, "end_r": END_R,
                           "local_snap": LOCAL_SNAP, "centroid_disk_r": DISK_R1,
                           "stub_min_um": STUB_MIN_UM,
                           "bent_bow_um": BENT_BOW_UM, "thin_cut_um": round(thin_cut, 1),
                           "dia_median_um": round(dmed, 1), "dia_mad_um": round(dmad, 1),
                           "mid_window": [MID_LO, MID_HI],
                           "boundary_margin_vox": round(BOUND_MARGIN, 2),
                           "cap_z": list(CAP_Z), "calibration": calib,
                           "voxel_um": UM, "nominal_diameter_um": NOM_D},
            "audit": dict(audit),
            "node_states": {k: int(v) for k, v in sc.items()},
            "volume_shape_zyx": [int(x) for x in shape],
            "printed_bbox_zyx": [[int(x) for x in LOB], [int(x) for x in HIB]]}

    OUT_JSON.write_text(json.dumps({"struts": recs, "meta": meta}), encoding="utf-8")

    # ---- report ----
    L = ["# v2 strut classification — results\n",
         f"{n} design struts.  Verdict counts (all / interior-context only):\n",
         "| verdict | all | % | interior |", "|---|---:|---:|---:|"]
    for k in ("present", "missing", "disconnected", "thin", "bent"):
        L.append(f"| {k} | {counts[k]} | {100*counts[k]/n:.2f}% | {ic[k]} |")
    defects = n - counts["present"]
    L += [f"\n**defective: {defects} ({100*defects/n:.2f}%)**\n",
          "## missing, split by cause",
          f"- dropped strut (both nodes printed): **{len(miss_drop)}**"
          f" — graph-confirmed (hops>=2): {g2}/{gk} with both nodes in the graph",
          f"- node_lost (one endpoint node never printed): **{len(miss_nl)}**",
          f"- void (both endpoint nodes never printed): **{len(miss_void)}**\n",
          f"disconnected with a lost node: {len(disc_nl)}\n",
          "## thin, split by cause",
          f"- hairline (whole strut below cutoff): {audit['thin_hairline']}",
          f"- necked (locally below cutoff, median normal): {audit['thin_necked']}\n",
          "## audits",
          f"- phantom edges vetoed (stub tip merged into a node): {audit['phantom_edge_broken']}",
          f"- corridor-blind but skeleton path locally fused to both cores (trusted edge): {audit['edge_rescues_corridor']}",
          f"- fragmented edges where the corridor verdict stood: {audit['fragmented_edge_corridor_broken']}",
          f"- present bow um (med/p90): {np.median(bows_p):.0f}/{np.percentile(bows_p,90):.0f}"
          f" — bent bow um (med): {np.median(bows_b):.0f}" if len(bows_b) else "- no bent",
          f"- present dia um (med): {np.median(dias_p):.0f}"
          f" — thin dia um (med): {np.median(dias_t):.0f}" if len(dias_t) else "- no thin",
          "\nEvery strut record carries its own measurements (profile string, gap, stub,",
          "bow, diameter, node states, graph hops) — validation panels are drawn from",
          "these records, so a panel cannot disagree with its verdict."]
    OUT_MD.write_text("\n".join(str(x) for x in L), encoding="utf-8")

    print("\n=== V2 RESULTS ===")
    for k in ("present", "missing", "disconnected", "thin", "bent"):
        print(f"  {k:12s}: {counts[k]:5d}  ({100*counts[k]/n:.2f}%)   interior: {ic[k]}")
    print(f"  defective: {defects} ({100*defects/n:.2f}%)")
    print(f"  missing = {len(miss_drop)} dropped + {len(miss_nl)} node_lost + "
          f"{len(miss_void)} void; graph-confirmed dropped: {g2}/{gk}")
    print(f"  thin = {audit['thin_hairline']} hairline + {audit['thin_necked']} necked; "
          f"disconnected with lost node: {len(disc_nl)}")
    print(f"  audits: {dict(audit)}")
    if V1_JSON.exists():
        v1 = json.load(open(V1_JSON))["meta"]["counts"]
        print(f"  (v1 for reference: {v1})")
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
