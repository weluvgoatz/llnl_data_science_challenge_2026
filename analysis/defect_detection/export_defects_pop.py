"""'Defects pop' full model: present struts as a THIN, faint grey wireframe you
can see through, with the defects at full size and full brightness on top. Keeps
the whole lattice for context while making every defect stand out.

Fast: reuses the already-traced defect meshes (the per-category .ply files) and
only rebuilds the present struts as thin straight rods — no scan re-processing.
"""
import json
from pathlib import Path

import numpy as np

import export_realistic_models as ER      # swept_tube(), write_ply(), RGB

from config import ROOT, STK, BASE
UNI = json.load(open(STK / f"{BASE}_unified_defects_accurate.json"))["struts"]
PARTS = ROOT / "analysis/defect_detection/MODEL_realistic_parts"
OUT = ROOT / "analysis/defect_detection/MODEL_lattice_defects_pop.ply"

PRESENT_R = 0.9                 # thin present rods
FAINT = (160, 175, 195)         # muted grey-blue so colours pop against it


def read_ply(path):
    with open(path, "rb") as f:
        hdr = b""
        while b"end_header" not in hdr:
            hdr += f.readline()
        lines = hdr.split(b"\n")
        nv = int([l for l in lines if l.startswith(b"element vertex")][0].split()[-1])
        nf = int([l for l in lines if l.startswith(b"element face")][0].split()[-1])
        vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        V = np.frombuffer(f.read(nv * vdt.itemsize), dtype=vdt)
        fdt = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])
        F = np.frombuffer(f.read(nf * fdt.itemsize), dtype=fdt)
    xyz = np.stack([V["x"], V["y"], V["z"]], 1)
    col = np.stack([V["r"], V["g"], V["b"]], 1).astype(np.uint8)
    fac = np.stack([F["a"], F["b"], F["c"]], 1).astype(np.int64)
    return xyz, col, fac


def main():
    Vg = []; Fg = []; Cg = []; base = 0

    def add(v, f, c):
        nonlocal base
        Vg.append(v); Fg.append(f + base); Cg.append(c); base += len(v)

    # thin faint present rods
    pv = []; pf = []; b = 0
    for s in UNI:
        if s["verdict"] != "present":
            continue
        p0 = np.array(s["p0"])[::-1].astype(float); p1 = np.array(s["p1"])[::-1].astype(float)
        tb = ER.swept_tube(np.array([p0, p1]), [PRESENT_R, PRESENT_R], 6)
        if tb:
            v = tb[0][:, ::-1]
            pv.append(v); pf.append(np.array(tb[1], np.int64) + b); b += len(v)
    PV = np.vstack(pv); PF = np.vstack(pf)
    add(PV, PF, np.tile(FAINT, (len(PV), 1)).astype(np.uint8))
    print(f"present (thin): {len(PV):,} verts")

    # defects at full size/brightness (reuse traced part PLYs)
    for vd in ("thin", "disconnected", "bent", "missing"):
        p = PARTS / f"{vd}.ply"
        if not p.exists():
            continue
        v, c, f = read_ply(p)
        add(v, f, c)
        print(f"{vd}: {len(v):,} verts")

    V = np.vstack(Vg); F = np.vstack(Fg); C = np.vstack(Cg)
    print(f"total: {len(V):,} verts, {len(F):,} tris")
    ER.write_ply(OUT, V, F, C)
    print(f"saved {OUT.name}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
