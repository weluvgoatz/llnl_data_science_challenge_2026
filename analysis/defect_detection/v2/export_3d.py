"""Realistic colour-coded 3-D model of the classified lattice.

Every strut is a swept tube built from MEASURED geometry, so the defects are
visible as geometry and not only as colour:

  present       blue      straight-ish tube at its measured radius
  bent          magenta   swept along the strut's ACTUAL skeleton centreline,
                          so the real curvature is what you see
  thin          yellow    radius read from the distance transform at every step,
                          so a hairline strut is genuinely thin and a necked one
                          genuinely pinches in the middle
  disconnected  orange    ONE TUBE PER SURVIVING PIECE - the break is a real gap
                          in the mesh, placed where the metal actually stops
  missing       red       kept as a full straight tube (deliberately NOT removed)
                          so you can see where the strut should have been

Outputs (v2/model3d/):
  lattice_full.ply          all struts, per-vertex colour  <- open this in MeshLab
  lattice_defects_only.ply  only the 1822 defective struts (small, fast, web-ready)
  parts/<verdict>.ply       one file per category
  color_key.png             legend

Radii come from the distance transform of the segmented mask, capped at 1.25x
the nominal strut radius: at a junction the transform reports the radius of the
whole node blob, which without the cap inflates strut ends into blobs.
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detect_v2 as D  # noqa: E402

OUT = Path(__file__).resolve().parent / "model3d"
NSIDE = 14                     # radial segments per tube (was 8 - visibly faceted)
MAX_R_MULT = 1.25              # radius cap, x nominal strut radius
MIN_R = 0.6                    # never render a degenerate sliver

# Axial samples by verdict.  The polygon budget goes to the defects: a healthy
# strut is straight, so more samples along it buy nothing, while a bent one needs
# them to render its curve smoothly and a necked one to render its pinch.
K_BY = {"present": 6, "missing": 2, "thin": 22, "disconnected": 26, "bent": 30}

RGB = {
    "present": (110, 170, 255),      # blue
    "missing": (240, 60, 60),        # red
    "disconnected": (250, 130, 30),  # orange
    "thin": (245, 205, 40),          # yellow
    "bent": (240, 80, 220),          # magenta
}
ORDER = ("present", "missing", "disconnected", "thin", "bent")


# --------------------------------------------------------------------- meshing
def _frames(P):
    """Parallel-transport orthonormal frames along a polyline (no twist)."""
    M = len(P)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]; T[0] = P[1] - P[0]; T[-1] = P[-1] - P[-2]
    Tn = np.linalg.norm(T, axis=1, keepdims=True); Tn[Tn == 0] = 1; T = T / Tn
    a = np.array([1.0, 0, 0])
    if abs(a @ T[0]) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    N = a - (a @ T[0]) * T[0]; N = N / np.linalg.norm(N)
    Ns = [N]
    for i in range(1, M):
        v = np.cross(T[i - 1], T[i]); s = np.linalg.norm(v); n = Ns[-1]
        if s > 1e-9:
            v = v / s
            c = float(np.clip(T[i - 1] @ T[i], -1, 1)); ang = np.arccos(c)
            n = (n * np.cos(ang) + np.cross(v, n) * np.sin(ang)
                 + v * (v @ n) * (1 - np.cos(ang)))
        n = n - (n @ T[i]) * T[i]
        nn = np.linalg.norm(n)
        Ns.append(n / nn if nn > 1e-9 else Ns[-1])
    Ns = np.array(Ns)
    return Ns, np.cross(T, Ns)


def swept_tube(P, radii, nside=NSIDE):
    """Closed swept tube of varying radius along polyline P -> (verts, faces)."""
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
        b0, b1 = i * nside, (i + 1) * nside
        for j in range(nside):
            k = (j + 1) % nside
            faces.append((b0 + j, b0 + k, b1 + k))
            faces.append((b0 + j, b1 + k, b1 + j))
    c0 = len(verts); c1 = c0 + 1
    verts = np.vstack([verts, P[0], P[-1]])
    for j in range(nside):
        k = (j + 1) % nside
        faces.append((c0, k, j))
        faces.append((c1, (M - 1) * nside + j, (M - 1) * nside + k))
    return verts, np.array(faces, np.int64)


def write_ply(path, V, F, C):
    """Binary little-endian PLY with per-vertex colour."""
    V = np.asarray(V, np.float32); F = np.asarray(F, np.int32)
    C = np.asarray(C, np.uint8)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(b"comment v2 lattice defect model (voxel units)\n")
        f.write(f"element vertex {len(V)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n")
        f.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(F)}\n".encode())
        f.write(b"property list uchar int vertex_indices\nend_header\n")
        vc = np.empty(len(V), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        vc["p"] = V; vc["c"] = C
        f.write(vc.tobytes())
        fc = np.empty(len(F), dtype=[("n", "u1"), ("v", "<i4", 3)])
        fc["n"] = 3; fc["v"] = F
        f.write(fc.tobytes())


# ------------------------------------------------------------------ geometry
def runs_of_true(flags):
    """[(start, stop)] index ranges of consecutive True — one per surviving piece."""
    out = []
    s = None
    for i, v in enumerate(list(flags) + [False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i))
            s = None
    return out


def disconnected_pieces(p0, p1, mask, shape, K):
    """One polyline per CONNECTED PIECE of metal in the strut's corridor.

    A presence-based centreline cannot show a joint detachment: the crack is
    hairline, so metal exists at every axial position and the tube would render
    continuous.  Labelling the corridor metal into connected components and
    rendering each separately reproduces the real severance — a piece that
    detached from its node visibly stops short of it.
    """
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return []
    u = d / L
    R = D.CORR_R
    lo = np.maximum(np.floor(np.minimum(p0, p1) - R - 1).astype(int), 0)
    hi = np.minimum(np.ceil(np.maximum(p0, p1) + R + 2).astype(int), shape)
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if not sub.any():
        return []
    zz, yy, xx = np.nonzero(sub)
    P = np.stack([zz, yy, xx], 1).astype(np.float64) + lo
    rel = P - p0
    t = rel @ u
    foot = p0[None] + np.clip(t, 0, L)[:, None] * u[None]
    inC = np.linalg.norm(P - foot, axis=1) <= R
    if not inC.any():
        return []
    Pc = P[inC]; tc = t[inC]
    idx = (Pc - lo).astype(int)
    volc = np.zeros(sub.shape, bool)
    volc[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    lab, n = ndi.label(volc, structure=D.CONN)
    lv = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
    pieces = []
    for c in range(1, n + 1):
        sel = lv == c
        if sel.sum() < 8:
            continue
        pt, tt = Pc[sel], tc[sel]
        span = float(tt.max() - tt.min())
        if span < 2.0:
            continue
        nb = max(int(round(span / max(L / K, 1.0))), 2)
        edges = np.linspace(tt.min(), tt.max(), nb + 1)
        cen = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (tt >= a) & (tt <= b)
            if m.any():
                cen.append(pt[m].mean(axis=0))
        if len(cen) >= 2:
            pieces.append(np.array(cen))
    return pieces


def strut_paths(rec, mask, shape, nominal_r):
    """Polylines (z,y,x) for one strut: one per surviving piece of metal.

    missing -> the straight design line (kept on purpose, just coloured red).
    bent    -> the stored skeleton polyline, i.e. the exact curve the bow was
               measured on, so the render shows the real curvature.
    others  -> the metal-centroid centreline; samples with no metal are dropped,
               which is what turns a break into a real gap in the mesh.
    """
    p0 = np.array(rec["p0"][::-1], float)          # -> (z,y,x)
    p1 = np.array(rec["p1"][::-1], float)
    vd = rec["verdict"]
    K = K_BY[vd]

    if vd == "missing":
        return [np.stack([p0, p1])]

    if vd == "bent" and "path" in rec:
        Q = np.array(rec["path"], float)[:, ::-1]  # stored xyz -> zyx
        if len(Q) >= 2:
            if len(Q) > K:                         # decimate evenly
                Q = Q[np.linspace(0, len(Q) - 1, K).round().astype(int)]
            return [Q]

    if vd == "disconnected":
        pieces = disconnected_pieces(p0, p1, mask, shape, K)
        if pieces:
            return pieces

    cent, _ts = D.centroid_path(p0, p1, mask, shape, K=K, lo=0.0, hi=1.0)
    ok = ~np.isnan(cent[:, 0])
    pieces = []
    for a, b in runs_of_true(ok):
        seg = cent[a:b]
        if len(seg) >= 2:
            pieces.append(seg)
        elif len(seg) == 1:                        # single sample -> tiny nub
            pieces.append(np.stack([seg[0] - 0.5 * (p1 - p0) / np.linalg.norm(p1 - p0),
                                    seg[0] + 0.5 * (p1 - p0) / np.linalg.norm(p1 - p0)]))
    if not pieces:                                 # nothing measurable; fall back
        return [np.stack([p0, p1])]
    return pieces


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "parts").mkdir(exist_ok=True)

    data = json.load(open(D.OUT_JSON))
    S = data["struts"]
    nominal_r = D.NOM_D / 2 / D.UM
    max_r = nominal_r * MAX_R_MULT
    print(f"loading mask ...", flush=True)
    mask = D.load_or_build_mask()
    shape = np.array(mask.shape)
    # same calibration the detector used, so corridor/disk radii match the scan
    gt = json.load(open(D.DESIGN_JSON))
    _jp = np.array([j["position"] for j in gt["junctions"]], float)
    _pitch = float(np.median([
        np.linalg.norm(_jp[s["junction0"]] - _jp[s["junction1"]])
        for s in gt["struts"]]))
    D.calibrate(_pitch, mask)
    print(f"  nominal strut radius {nominal_r:.2f} vox, radius cap {max_r:.2f}",
          flush=True)

    # ---- pass 1: every polyline, and all their points in one array ----
    print("pass 1/3: tracing measured centrelines ...", flush=True)
    entries = []            # (verdict, [polyline, ...])
    allpts = []
    for n, rec in enumerate(S):
        pieces = strut_paths(rec, mask, shape, nominal_r)
        entries.append((rec["verdict"], pieces))
        for P in pieces:
            allpts.append(P)
        if (n + 1) % 4000 == 0:
            print(f"    {n+1}/{len(S)} struts", flush=True)
    lens = [len(P) for P in allpts]
    pts = np.concatenate(allpts, axis=0)
    print(f"  {len(pts):,} centreline points on {len(allpts):,} pieces", flush=True)

    # ---- pass 2: true local radius at every point (chunked EDT) ----
    print("pass 2/3: measuring radius from the distance transform ...", flush=True)
    qi = np.round(pts).astype(np.int64)
    np.clip(qi, 0, shape - 1, out=qi)
    radii = D.edt_radii(mask, qi)
    del mask
    gc.collect()
    radii = np.clip(radii.astype(float), MIN_R, max_r)

    # ---- pass 3: build the meshes ----
    print("pass 3/3: meshing ...", flush=True)
    buf = {k: {"V": [], "F": [], "n": 0} for k in ORDER}
    off = 0
    counted = {k: 0 for k in ORDER}
    for (vd, pieces) in entries:
        for P in pieces:
            m = len(P)
            r = radii[off:off + m]
            off += m
            if m >= 5:
                P = gaussian_filter1d(P, 1.0, axis=0, mode="nearest")
                r = gaussian_filter1d(r, 1.2, mode="nearest")
            if vd == "missing":                    # no metal to measure
                r = np.full(m, nominal_r)
            out = swept_tube(P, np.clip(r, MIN_R, max_r))
            if out is None:
                continue
            V, F = out
            b = buf[vd]
            b["V"].append(V.astype(np.float32))
            b["F"].append((F + b["n"]).astype(np.int32))
            b["n"] += len(V)
        counted[vd] += 1

    # ---- write ----
    def stack(keys):
        V, F, C, n = [], [], [], 0
        for k in keys:
            b = buf[k]
            if not b["V"]:
                continue
            v = np.concatenate(b["V"]); f = np.concatenate(b["F"])
            V.append(v); F.append(f + n)
            C.append(np.tile(RGB[k], (len(v), 1)).astype(np.uint8))
            n += len(v)
        if not V:
            return None
        return np.concatenate(V), np.concatenate(F), np.concatenate(C)

    for k in ORDER:
        got = stack([k])
        if got:
            write_ply(OUT / "parts" / f"{k}.ply", *got)
            print(f"  parts/{k}.ply  {counted[k]:6d} struts  {len(got[0]):9,} verts",
                  flush=True)

    full = stack(ORDER)
    write_ply(OUT / "lattice_full.ply", *full)
    print(f"  lattice_full.ply          {len(full[0]):,} verts, {len(full[1]):,} faces "
          f"({(OUT/'lattice_full.ply').stat().st_size/1e6:.0f} MB)")

    def_ = stack([k for k in ORDER if k != "present"])
    write_ply(OUT / "lattice_defects_only.ply", *def_)
    print(f"  lattice_defects_only.ply  {len(def_[0]):,} verts, {len(def_[1]):,} faces "
          f"({(OUT/'lattice_defects_only.ply').stat().st_size/1e6:.0f} MB)")

    # ---- colour key ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cnt = {k: sum(1 for r in S if r["verdict"] == k) for k in ORDER}
    note = {
        "present": "healthy — straight tube at its measured radius",
        "missing": "never built — kept as a full tube on purpose, so you can see where it belongs",
        "disconnected": "severed — rendered as separate pieces, the gap is real",
        "thin": "under-thickness — radius measured at every step, so it is genuinely thin",
        "bent": "bowed — swept along the strut's real skeleton curve",
    }
    fig, ax = plt.subplots(figsize=(11, 3.4), facecolor="white")
    for i, k in enumerate(ORDER):
        y = len(ORDER) - 1 - i
        ax.add_patch(Rectangle((0, y - 0.34), 0.55, 0.68,
                               facecolor=np.array(RGB[k]) / 255, edgecolor="black", lw=0.6))
        ax.text(0.7, y, f"{k.upper()}  ({cnt[k]:,})", va="center",
                fontsize=12, fontweight="bold")
        ax.text(4.0, y, note[k], va="center", fontsize=10, color="#333")
    ax.set_xlim(-0.15, 15); ax.set_ylim(-0.7, len(ORDER) - 0.3); ax.axis("off")
    ax.set_title("Lattice defect model — colour key\n"
                 "defects are shown as real geometry, not colour alone",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "color_key.png", dpi=140, bbox_inches="tight", facecolor="white")
    print(f"  color_key.png")
    print(f"\nOpen {OUT/'lattice_full.ply'} in MeshLab "
          f"(set Render > Shading > None to see the colours flat).")


if __name__ == "__main__":
    main()
