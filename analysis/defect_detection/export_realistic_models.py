"""GEOMETRY-ACCURATE coloured lattice model.

Unlike MODEL_lattice_coloured.ply (every strut a straight recoloured rod), this
draws each DEFECT with its real as-built shape, measured from the scan:

  * BENT         — tube swept along the strut's ACTUAL curved centreline
                   (the bowed skeleton chain), so the rod physically arcs.
  * THIN         — tube radius taken from the LOCAL distance transform along the
                   centreline, so thin struts render genuinely skinnier / tapered.
  * DISCONNECTED — one tube per real metal stub (traced skeleton), with the true
                   empty gap between them.
  * PRESENT      — straight tube at the measured nominal radius (kept simple; they
                   are straight and near-nominal by definition).
  * MISSING      — straight red "ghost" rod marking where the strut should be
                   (kept as-is; there is no metal to show).

Every centreline is retraced from the cached skeleton; every radius is the
Euclidean distance transform of the segmented metal (radius of the maximal
inscribed sphere) sampled on that centreline. Nothing about the shapes is
assumed — it is all read back out of the CT.

Outputs (analysis/defect_detection/):
  MODEL_lattice_realistic.ply            per-vertex RGB, curved geometry
  MODEL_lattice_realistic_coloured.stl   VisCAM per-facet colour extension
  MODEL_realistic_parts/<verdict>.stl    one mesh per verdict (universal colour)
"""

import gc
import json
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d, distance_transform_edt

from config import ROOT, STK, BASE
UNI = STK / f"{BASE}_unified_defects_accurate.json"
MASK = STK / f"{BASE}_segmented_clean.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
OUTDIR = ROOT / "analysis/defect_detection"
PLY = OUTDIR / "MODEL_lattice_realistic.ply"
COLSTL = OUTDIR / "MODEL_lattice_realistic_coloured.stl"
PARTS = OUTDIR / "MODEL_realistic_parts"

# brighter, high-value palette so colours stay vivid under MeshLab's lighting
RGB = {"present": (120, 180, 255), "bent": (245, 70, 240), "thin": (255, 225, 45),
       "disconnected": (255, 140, 30), "missing": (255, 65, 65)}
NSIDE_STRAIGHT = 8
NSIDE_DEFECT = 16
# a real strut radius never exceeds ~the design radius; larger values are the
# distance transform bleeding into a thick JUNCTION, which would balloon a tube
# into a sphere. Cap at design radius (3.65 vox) + a small margin.
MAX_R = 424.0 / 2 / 58.1 * 1.25          # ~4.6 vox
KERN = np.ones((3, 3, 3), int)


# ----------------------------------------------------------------------------- meshing
def _frames(P):
    """Parallel-transport orthonormal frames (T, N, B) along a polyline P (M,3)."""
    M = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]; T[0] = P[1] - P[0]; T[-1] = P[-1] - P[-2]
    Tn = np.linalg.norm(T, axis=1, keepdims=True); Tn[Tn == 0] = 1; T = T / Tn
    a = np.array([1.0, 0, 0])
    if abs(a @ T[0]) > 0.9:
        a = np.array([0, 1.0, 0])
    N = a - (a @ T[0]) * T[0]; N = N / np.linalg.norm(N)
    Ns = [N]
    for i in range(1, M):
        v = np.cross(T[i - 1], T[i]); s = np.linalg.norm(v); n = Ns[-1]
        if s > 1e-9:
            v = v / s; c = float(np.clip(T[i - 1] @ T[i], -1, 1)); ang = np.arccos(c)
            n = (n * np.cos(ang) + np.cross(v, n) * np.sin(ang)
                 + v * (v @ n) * (1 - np.cos(ang)))
        n = n - (n @ T[i]) * T[i]
        nn = np.linalg.norm(n); n = n / nn if nn > 1e-9 else Ns[-1]
        Ns.append(n)
    Ns = np.array(Ns); B = np.cross(T, Ns)
    return Ns, B


def swept_tube(P, radii, nside):
    """Swept tube of varying radius along polyline P. Returns (verts, faces)."""
    P = np.asarray(P, float); radii = np.asarray(radii, float)
    if len(P) < 2:
        return None
    Ns, B = _frames(P)
    ang = np.linspace(0, 2 * np.pi, nside, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    rings = (P[:, None, :] + radii[:, None, None]
             * (ca[None, :, None] * Ns[:, None, :] + sa[None, :, None] * B[:, None, :]))
    M = len(P)
    verts = rings.reshape(-1, 3)
    faces = []
    for i in range(M - 1):
        b0 = i * nside; b1 = (i + 1) * nside
        for j in range(nside):
            k = (j + 1) % nside
            faces.append((b0 + j, b0 + k, b1 + k))
            faces.append((b0 + j, b1 + k, b1 + j))
    c0 = len(verts); c1 = c0 + 1
    verts = np.vstack([verts, P[0], P[-1]])
    for j in range(nside):
        k = (j + 1) % nside
        faces.append((c0, k, j))                              # start cap
        faces.append((c1, (M - 1) * nside + j, (M - 1) * nside + k))   # end cap
    return verts, faces


# ----------------------------------------------------------------------------- tracing
def trace(p0, p1, mask, coords, shape, verdict, pad=12):
    """Return list of (polyline_zyx, radii) for the REAL metal of this strut,
    every radius read from the local distance transform. bent/thin -> one merged
    centreline (curved if the skeleton chain exists, else a straight fallback at
    the true measured radius); disconnected -> one tube per real metal stub, so
    the gap is preserved."""
    u = p1 - p0; L = np.linalg.norm(u); u = u / L
    clo = np.maximum(np.minimum(p0, p1).astype(int) - pad, 0)
    chi = np.minimum(np.maximum(p0, p1).astype(int) + pad + 1, shape)
    edt = distance_transform_edt(mask[clo[0]:chi[0], clo[1]:chi[1], clo[2]:chi[2]])
    eshape = np.array(edt.shape)

    def radii_at(Q):
        li = np.clip(np.round(Q).astype(int) - clo, 0, eshape - 1)
        return edt[li[:, 0], li[:, 1], li[:, 2]]

    def finish(Q, smooth_xyz=True):
        r = radii_at(Q)
        r = np.clip(r, 0.6, MAX_R)                    # kill junction-inflated blobs
        if len(Q) >= 5:
            if smooth_xyz:
                Q = gaussian_filter1d(Q, 1.2, axis=0, mode="nearest")
            r = gaussian_filter1d(r, 1.5, mode="nearest")
        return Q, np.clip(r, 0.6, MAX_R)

    # ---- skeleton components hugging the chord ----
    cm = np.all((coords >= clo) & (coords < chi), axis=1)
    comps = []
    if cm.sum() >= 5:
        loc = np.zeros(chi - clo, bool)
        lc = coords[cm] - clo; loc[lc[:, 0], lc[:, 1], lc[:, 2]] = True
        Kd = np.ones((3, 3, 3), int); Kd[1, 1, 1] = 0
        deg = ndi.convolve(loc.astype(np.uint8), Kd, mode="constant") * loc
        pathv = loc & (deg == 2)
        lbl, n = ndi.label(pathv, structure=KERN)
        pz, py, px = np.where(pathv)
        g = np.stack([pz, py, px], 1) + clo; pl = lbl[pz, py, px]
        span_min = 0.30 * L if verdict in ("bent", "thin") else 4.0
        for c in range(1, n + 1):
            pc = g[pl == c].astype(float)
            if len(pc) < 4:
                continue
            tt = (pc - p0) @ u
            pp = np.linalg.norm((pc - p0) - np.outer(tt, u), axis=1)
            if np.median(pp) < 5.0 and tt.min() > -6 and tt.max() < L + 6 \
                    and (tt.max() - tt.min()) >= span_min:
                comps.append((pc, tt))

    if verdict in ("bent", "thin"):
        if comps:                                        # real curved centreline
            Q = np.concatenate([c[0] for c in comps])
            t = np.concatenate([c[1] for c in comps])
            return [finish(Q[np.argsort(t)])]
        # fallback: straight chord, but radius still measured from EDT (stays thin)
        K = max(int(L), 8)
        ts = np.linspace(0, 1, K)
        Q = p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]
        return [finish(Q, smooth_xyz=False)]

    # ---- disconnected: draw only where metal exists (keep the gap) ----
    if comps:
        return [finish(c[0][np.argsort(c[1])]) for c in comps]   # one tube per stub
    # fallback: contiguous metal runs sampled along the chord
    K = max(int(L), 24)
    ts = np.linspace(0, 1, K)
    pts = p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]
    ip = np.clip(np.round(pts).astype(int), 0, shape - 1)
    hit = mask[ip[:, 0], ip[:, 1], ip[:, 2]]
    segs = []; i = 0
    while i < K:
        if hit[i]:
            j = i
            while j < K and hit[j]:
                j += 1
            if j - i >= 3:
                segs.append(finish(pts[i:j], smooth_xyz=False))
            i = j
        else:
            i += 1
    return segs


# ----------------------------------------------------------------------------- writers
def write_ply(path, V, F, C):
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\n"
           f"property float x\nproperty float y\nproperty float z\n"
           f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
           f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    vv = np.empty(len(V), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vv["x"], vv["y"], vv["z"] = V[:, 0], V[:, 1], V[:, 2]
    vv["r"], vv["g"], vv["b"] = C[:, 0], C[:, 1], C[:, 2]
    ff = np.empty(len(F), dtype=[("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
    ff["n"] = 3; ff["a"], ff["b"], ff["c"] = F[:, 0], F[:, 1], F[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(vv.tobytes()); f.write(ff.tobytes())


def write_stl(path, tris, colors=None):
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nn = np.linalg.norm(nrm, axis=1, keepdims=True); nn[nn == 0] = 1; nrm = nrm / nn
    with open(path, "wb") as f:
        f.write(b"COLOR=\xff\xff\xff\xff" + b"\0" * 70)
        f.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack("<fff", *nrm[i]))
            for vtx in tris[i]:
                f.write(struct.pack("<fff", *vtx))
            if colors is not None:
                r, g, b = (int(x) for x in colors[i])
                attr = (1 << 15) | ((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3)
            else:
                attr = 0
            f.write(struct.pack("<H", attr))


# ----------------------------------------------------------------------------- main
def main():
    S = json.load(open(UNI))["struts"]
    print("loading mask + skeleton ...")
    mask = tifffile.imread(MASK) > 0
    coords = np.load(SKELC)["coords"]
    shape = np.array(mask.shape)

    def pos(s):
        return np.array(s["p0"])[::-1].astype(float), np.array(s["p1"])[::-1].astype(float)

    # nominal radius from the metal itself: EDT sampled on the medial axis
    # (skeleton voxels), where the distance transform IS the local strut radius.
    print("measuring nominal radius ...")
    rng = np.random.RandomState(0)
    samp = coords[rng.choice(len(coords), 3000, replace=False)]
    rads = []
    for c in samp:
        lo = np.maximum(c - 8, 0); hi = np.minimum(c + 9, shape)
        e = distance_transform_edt(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        rads.append(e[tuple(c - lo)])
    nominal = float(np.median([r for r in rads if r > 0]))
    print(f"  nominal strut radius = {nominal:.2f} vox = {nominal*58.1:.0f} um")

    V_parts = defaultdict(list); F_parts = defaultdict(list); base_parts = defaultdict(int)

    def add(verdict, verts, faces):
        b = base_parts[verdict]
        V_parts[verdict].append(verts)
        F_parts[verdict].extend([(a + b, c + b, d + b) for a, c, d in faces])
        base_parts[verdict] += len(verts)

    counts = defaultdict(int); traced = defaultdict(int)
    for k, s in enumerate(S):
        vd = s["verdict"]; p0, p1 = pos(s)
        if vd in ("bent", "thin", "disconnected"):
            segs = trace(p0, p1, mask, coords, shape, vd)
            if segs:
                for Q, r in segs:
                    tb = swept_tube(Q, r, NSIDE_DEFECT)
                    if tb:
                        add(vd, tb[0], tb[1])
                traced[vd] += 1; counts[vd] += 1
            else:                                         # no metal at all (rare)
                tb = swept_tube(np.array([p0, p1]), [nominal, nominal], NSIDE_STRAIGHT)
                add(vd, tb[0], tb[1]); counts[vd] += 1
        else:                                             # present / missing: straight
            tb = swept_tube(np.array([p0, p1]), [nominal, nominal], NSIDE_STRAIGHT)
            add(vd, tb[0], tb[1]); counts[vd] += 1
        if (k + 1) % 4000 == 0:
            print(f"  {k+1}/{len(S)} struts")

    del mask, coords; gc.collect()

    # assemble global mesh (coords -> [x,y,z] to match existing models)
    allV = []; allF = []; allC = []; base = 0
    part_tris = {}; per = {}
    for vd in ("present", "bent", "thin", "disconnected", "missing"):
        if not V_parts[vd]:
            continue
        V = np.vstack(V_parts[vd])[:, ::-1]               # (z,y,x) -> (x,y,z)
        F = np.array(F_parts[vd], np.int32)
        C = np.tile(RGB[vd], (len(V), 1)).astype(np.uint8)
        part_tris[vd] = V[F]; per[vd] = (V, F, C)
        allV.append(V); allF.append(F + base); allC.append(C)
        base += len(V)
    V = np.vstack(allV); F = np.vstack(allF); C = np.vstack(allC)
    print(f"\nmesh: {len(V):,} vertices, {len(F):,} triangles")
    for vd in ("present", "bent", "thin", "disconnected", "missing"):
        print(f"  {vd:13s}: {counts[vd]:5d} struts   ({traced[vd]} with traced real shape)")

    # (1) full model
    write_ply(PLY, V, F, C)
    print(f"saved {PLY.name}  ({PLY.stat().st_size/1e6:.1f} MB)")
    write_stl(COLSTL, V[F], C[F[:, 0]])
    print(f"saved {COLSTL.name}  ({COLSTL.stat().st_size/1e6:.1f} MB)")

    # (2) DEFECTS-ONLY coloured PLY (present removed -> defects finally visible)
    dV = []; dF = []; dC = []; b = 0
    for vd in ("bent", "thin", "disconnected", "missing"):
        if vd not in per:
            continue
        v, f, c = per[vd]; dV.append(v); dF.append(f + b); dC.append(c); b += len(v)
    DV = np.vstack(dV); DF = np.vstack(dF); DC = np.vstack(dC)
    dpath = OUTDIR / "MODEL_defects_only.ply"
    write_ply(dpath, DV, DF, DC)
    print(f"saved {dpath.name}  ({dpath.stat().st_size/1e6:.1f} MB)")

    # (3) one COLOURED PLY per category (open just one at a time)
    PARTS.mkdir(exist_ok=True)
    for vd, (v, f, c) in per.items():
        pp = PARTS / f"{vd}.ply"; write_ply(pp, v, f, c)
        ps = PARTS / f"{vd}.stl"; write_stl(ps, part_tris[vd], None)
        print(f"saved MODEL_realistic_parts/{vd}.ply / .stl  ({pp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
