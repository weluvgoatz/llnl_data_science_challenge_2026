"""3D DEFECT GALLERY — the clearest individual defects, isolated + enlarged and
laid out in a grid, so each bend / gap / thinning is unmistakable in MeshLab.

Each strut keeps its TRUE shape (real traced centreline + measured EDT radius);
we only translate it to its own grid cell and scale that cell uniformly (shape
is preserved, just bigger). A thin GREY straight rod is drawn behind each strut
as the "ideal" path, so you can see exactly how far the real strut deviates.

Rows (in +Y, top to bottom):  BENT · THIN · DISCONNECTED · MISSING
Columns (in +X):              6 examples each.

Output: MODEL_defect_gallery.ply  (per-vertex colour).
"""
import json
from pathlib import Path

import numpy as np
import tifffile

import export_realistic_models as ER      # reuse trace(), swept_tube(), write_ply()

from config import ROOT, STK, BASE
UNI = json.load(open(STK / f"{BASE}_unified_defects_accurate.json"))["struts"]
BENT = json.load(open(STK / f"{BASE}_asbuilt_bent.json"))["struts"]
OUT = ROOT / "analysis/defect_detection/MODEL_defect_gallery.ply"

NCOL = 6
SCALE = 3.0            # enlarge each cell (uniform -> shape preserved)
CELL = 240.0           # grid spacing in world units
GREY = (120, 120, 120)
MISSING_R = 8.0        # red ghost rod radius (nominal ~2.7 vox * SCALE)


def main():
    print("loading mask + skeleton ...")
    mask = tifffile.imread(STK / f"{BASE}_segmented_clean.tif") > 0
    coords = np.load(STK / f"{BASE}_segmented_clean.skelcoords.npz")["coords"]
    shape = np.array(mask.shape); margin = 55

    def interior(p):
        return np.all(p > margin) and np.all(p < shape - margin)

    def upos(s):
        return np.array(s["p0"])[::-1].astype(float), np.array(s["p1"])[::-1].astype(float)

    picks = {}

    # BENT: the six largest bows (interior)
    picks["bent"] = []
    for s in sorted(BENT, key=lambda s: -s["max_dev"]):
        p0 = np.array(s["p0"], float); p1 = np.array(s["p1"], float)
        if interior(p0) and interior(p1):
            picks["bent"].append((p0, p1))
        if len(picks["bent"]) >= NCOL:
            break

    # THIN / DISCONNECTED / MISSING: an even spread of interior examples
    for vd in ("thin", "disconnected", "missing"):
        cands = [upos(s) for s in UNI
                 if s["verdict"] == vd and interior(upos(s)[0]) and interior(upos(s)[1])]
        idx = np.linspace(0, len(cands) - 1, min(NCOL, len(cands))).astype(int)
        picks[vd] = [cands[i] for i in idx]

    rows = ["bent", "thin", "disconnected", "missing"]
    Vg = []; Fg = []; Cg = []; base = 0

    def emit(verts_zyx, faces, color):
        nonlocal base
        v = verts_zyx[:, ::-1]                         # (z,y,x) -> (x,y,z)
        Vg.append(v)
        Fg.extend([(a + base, b + base, c + base) for a, b, c in faces])
        Cg.extend([color] * len(v)); base += len(v)

    for r, vd in enumerate(rows):
        col = ER.RGB[vd]
        for c, (p0, p1) in enumerate(picks[vd]):
            centre = (p0 + p1) / 2.0
            offset = np.array([0.0, -r * CELL, c * CELL])       # (z,y,x)

            def place(Q):
                return (np.asarray(Q, float) - centre) * SCALE + offset

            # thin grey "ideal" straight rod behind the strut
            tb = ER.swept_tube(place(np.array([p0, p1])), [2.0, 2.0], 10)
            if tb:
                emit(tb[0], tb[1], GREY)

            if vd == "missing":                                 # no metal -> red ghost rod
                tb = ER.swept_tube(place(np.array([p0, p1])), [MISSING_R, MISSING_R], 14)
                if tb:
                    emit(tb[0], tb[1], col)
                continue

            for Q, rad in ER.trace(p0, p1, mask, coords, shape, vd):   # real geometry
                tb = ER.swept_tube(place(Q), np.asarray(rad) * SCALE, 18)
                if tb:
                    emit(tb[0], tb[1], col)
        print(f"row {vd}: {len(picks[vd])} examples")

    V = np.vstack(Vg); F = np.array(Fg, np.int32); C = np.array(Cg, np.uint8)
    print(f"gallery mesh: {len(V):,} verts, {len(F):,} tris")
    ER.write_ply(OUT, V, F, C)
    print(f"saved {OUT.name}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
