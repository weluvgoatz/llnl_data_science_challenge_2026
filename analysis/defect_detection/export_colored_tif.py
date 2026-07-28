"""Paint the lattice by defect type into an RGB .tif stack you can scroll in Fiji.

Each designed strut is drawn (as a thick coloured line) into a full-size RGB
volume at its scan position, coloured by verdict. Open in Fiji: File > Open,
then scroll — the lattice appears in colour, defects standing out.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
UNI = STK / f"{BASE}_unified_defects_accurate.json"
OUT = STK / f"{BASE}_defects_coloured.tif"

RGB = {"present": (60, 110, 210), "bent": (205, 40, 210), "thin": (235, 205, 40),
       "disconnected": (250, 130, 30), "missing": (235, 40, 40)}
RAD = {"present": 1, "bent": 2, "thin": 2, "disconnected": 2, "missing": 2}


def main():
    d = json.load(open(UNI))
    S = d["struts"]
    Zs, Ys, Xs = d["meta"]["volume_shape_zyx"]
    vol = np.zeros((Zs, Ys, Xs, 3), np.uint8)

    # Draw present first, then defects on top. For each category: rasterise all
    # its strut lines into a boolean mask, dilate it once, then paint the colour.
    cats = ["present", "thin", "disconnected", "bent", "missing"]
    for cat in cats:
        items = [s for s in S if s["verdict"] == cat]
        if not items:
            continue
        m = np.zeros((Zs, Ys, Xs), bool)
        for s in items:
            p0 = np.array(s["p0"]); p1 = np.array(s["p1"])   # [x,y,z]
            L = int(np.linalg.norm(p1 - p0)) + 1
            tvals = np.linspace(0, 1, max(L, 4))
            pts = p0[None] * (1 - tvals[:, None]) + p1[None] * tvals[:, None]
            zi = np.clip(np.round(pts[:, 2]).astype(int), 0, Zs - 1)
            yi = np.clip(np.round(pts[:, 1]).astype(int), 0, Ys - 1)
            xi = np.clip(np.round(pts[:, 0]).astype(int), 0, Xs - 1)
            m[zi, yi, xi] = True
        m = ndi.binary_dilation(m, iterations=RAD[cat])
        vol[m] = np.array(RGB[cat], np.uint8)
        print(f"  painted {cat}: {len(items)} struts")

    # ImageJ-compatible RGB stack (axes Z,Y,X,C)
    tifffile.imwrite(OUT, vol, photometric="rgb", compression="zlib",
                     metadata={"axes": "ZYXS"})
    print(f"saved {OUT.name}  ({OUT.stat().st_size/1e6:.1f} MB)  shape {vol.shape}")


if __name__ == "__main__":
    main()
