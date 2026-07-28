"""Colored STL export of the full lattice (accurate defect classification).

STL has no official colour, so we produce BOTH:
  * MODEL_lattice_coloured.stl  — binary STL with the VisCAM/SolidView per-facet
      colour extension (packed into the 2-byte attribute). MeshLab & several
      slicers show the colours; plain viewers show grey.
  * MODEL_parts/<verdict>.stl   — one STL per defect type. Load all together in
      any viewer and assign each its colour — the universal "coloured STL".

The PLY (MODEL_lattice_coloured.ply) remains the most reliably-coloured model.
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
UNI = STK / f"{BASE}_unified_defects_accurate.json"
OUTDIR = ROOT / "analysis/defect_detection"
PARTS = OUTDIR / "MODEL_parts"
COLSTL = OUTDIR / "MODEL_lattice_coloured.stl"

RGB = {"present": (70, 120, 210), "bent": (200, 40, 210), "thin": (235, 200, 40),
       "disconnected": (250, 130, 40), "missing": (235, 50, 50)}
RAD = {"present": 1.6, "bent": 3.4, "thin": 3.4, "disconnected": 3.4, "missing": 3.4}
NSIDE = 8


def tube(p0, p1, r, nside=NSIDE):
    axis = p1 - p0
    L = np.linalg.norm(axis)
    if L < 1e-6:
        return None
    a = axis / L
    ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, ref); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    ang = np.linspace(0, 2 * np.pi, nside, endpoint=False)
    ring = np.array([np.cos(t) * u + np.sin(t) * v for t in ang]) * r
    p0c = p0 + a * r * 0.5; p1c = p1 - a * r * 0.5           # inset caps
    verts = np.vstack([p0c + ring, p1c + ring, p0c, p1c])    # + 2 cap centres
    faces = []
    for i in range(nside):
        j = (i + 1) % nside
        faces += [(i, j, nside + j), (i, nside + j, nside + i)]      # side
        faces += [(2 * nside, j, i)]                                 # cap p0
        faces += [(2 * nside + 1, nside + i, nside + j)]             # cap p1
    return verts, faces


def write_stl(path, tris, colors=None):
    """tris: (T,3,3) float; colors: (T,3) uint8 or None."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True); nn[nn == 0] = 1
    n = n / nn
    with open(path, "wb") as f:
        f.write(b"COLOR=\xff\xff\xff\xff" + b"\0" * 70)          # header (Materialise-style tag)
        f.write(struct.pack("<I", len(tris)))
        for i in range(len(tris)):
            f.write(struct.pack("<fff", *n[i]))
            for vtx in tris[i]:
                f.write(struct.pack("<fff", *vtx))
            if colors is not None:
                r, g, b = (int(x) for x in colors[i])
                attr = (1 << 15) | ((b >> 3) << 10) | ((g >> 3) << 5) | (r >> 3)  # VisCAM 1-5-5-5
            else:
                attr = 0
            f.write(struct.pack("<H", attr))


def build(struts):
    V, F, Fcol = [], [], []
    base = 0
    for s in struts:
        p0 = np.array(s["p0"], float); p1 = np.array(s["p1"], float)
        tb = tube(p0, p1, RAD[s["verdict"]])
        if tb is None:
            continue
        verts, faces = tb
        V.append(verts)
        F.extend([(a + base, b + base, c + base) for a, b, c in faces])
        Fcol.extend([RGB[s["verdict"]]] * len(faces))
        base += len(verts)
    V = np.vstack(V); F = np.array(F, np.int32); Fcol = np.array(Fcol, np.uint8)
    return V, F, Fcol


def main():
    d = json.load(open(UNI))
    S = d["struts"]
    PARTS.mkdir(exist_ok=True)

    # ---- single coloured STL (colour extension) ----
    V, F, Fcol = build(S)
    write_stl(COLSTL, V[F], Fcol)
    print(f"saved {COLSTL.name}  ({COLSTL.stat().st_size/1e6:.1f} MB, {len(F):,} tris)")

    # ---- one STL per verdict (universal coloured-STL workflow) ----
    for cat in ("present", "bent", "thin", "disconnected", "missing"):
        items = [s for s in S if s["verdict"] == cat]
        if not items:
            continue
        V, F, _ = build(items)
        p = PARTS / f"{cat}.stl"
        write_stl(p, V[F], None)
        print(f"saved MODEL_parts/{cat}.stl  ({len(items)} struts, {p.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
