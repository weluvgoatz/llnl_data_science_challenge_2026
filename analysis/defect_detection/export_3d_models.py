"""Export the coloured as-built lattice as 3-D model files you can spin around.

  * PLY  — every strut a coloured tube (per-vertex RGB). Opens in the Windows
           3-D Viewer (double-click), Blender, MeshLab. Colour = defect type.
  * STL  — same tube geometry, no colour (STL has no colour). Universal CAD mesh.

Struts are drawn as thin hexagonal prisms; defective struts are drawn thicker.
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
UNI = STK / f"{BASE}_unified_defects_accurate.json"
PLY = ROOT / "analysis/defect_detection/MODEL_lattice_coloured.ply"
STL = ROOT / "analysis/defect_detection/MODEL_lattice.stl"

RGB = {"present": (70, 120, 210), "bent": (200, 40, 210), "thin": (235, 200, 40),
       "disconnected": (250, 130, 40), "missing": (235, 50, 50)}
RAD = {"present": 1.6, "bent": 3.2, "thin": 3.2, "disconnected": 3.2, "missing": 3.2}
NSIDE = 6                       # hexagonal cross-section


def tube(p0, p1, r, nside=NSIDE):
    """Vertices (2*nside,3) and triangle faces for a hex prism from p0 to p1."""
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
    verts = np.vstack([p0 + ring, p1 + ring])
    faces = []
    for i in range(nside):
        j = (i + 1) % nside
        faces.append((i, j, nside + j))
        faces.append((i, nside + j, nside + i))
    return verts, faces


def main():
    d = json.load(open(UNI))
    S = d["struts"]
    # missing struts have no material — draw them as "ghost" tubes so the gap is
    # visible in the model (they mark where a strut SHOULD be).
    all_v = []
    all_f = []
    all_c = []
    base = 0
    for s in S:
        p0 = np.array(s["p0"], float); p1 = np.array(s["p1"], float)   # [x,y,z]
        col = RGB[s["verdict"]]; r = RAD[s["verdict"]]
        tb = tube(p0, p1, r)
        if tb is None:
            continue
        verts, faces = tb
        all_v.append(verts)
        all_f.extend([(a + base, b + base, c + base) for a, b, c in faces])
        all_c.extend([col] * len(verts))
        base += len(verts)
    V = np.vstack(all_v)
    F = np.array(all_f, np.int32)
    C = np.array(all_c, np.uint8)
    print(f"mesh: {len(V):,} vertices, {len(F):,} triangles")

    # ---------- PLY (binary little-endian, per-vertex colour) ----------
    with open(PLY, "wb") as f:
        hdr = (f"ply\nformat binary_little_endian 1.0\n"
               f"element vertex {len(V)}\n"
               f"property float x\nproperty float y\nproperty float z\n"
               f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
               f"element face {len(F)}\nproperty list uchar int vertex_indices\n"
               f"end_header\n")
        f.write(hdr.encode())
        for (x, y, z), (r_, g_, b_) in zip(V, C):
            f.write(struct.pack("<fffBBB", x, y, z, r_, g_, b_))
        for a, b, c in F:
            f.write(struct.pack("<Biii", 3, a, b, c))
    print(f"saved {PLY.name}  ({PLY.stat().st_size/1e6:.1f} MB)")

    # ---------- STL (binary, no colour) ----------
    tri = V[F]                                   # (T,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nn = np.linalg.norm(n, axis=1, keepdims=True); nn[nn == 0] = 1
    n = n / nn
    with open(STL, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(F)))
        for i in range(len(F)):
            f.write(struct.pack("<fff", *n[i]))
            for vtx in tri[i]:
                f.write(struct.pack("<fff", *vtx))
            f.write(struct.pack("<H", 0))
    print(f"saved {STL.name}  ({STL.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
