"""Reference-BASED defect detection: compare the scan to the no-defect JSON.

Uses the registered reference blueprint (which knows exactly where every strut
should be) and measures the raw CT density + metal along each DESIGNED strut.
A strut the design says exists but that reads empty / low-density is a defect.

This is the "with reference" counterpart to the reference-free pipeline, and
lets us see how close the reference-free result was.

Usage (from repo root):
    python analysis/defect_detection/reference_based_defects.py \
        --tif "data/missing_struts/tif_stacks/....tif" \
        --reference "data/missing_struts/registered_jsons/....json"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def longest_gap(hit):
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1; best = max(best, cur)
    return best


def robust(x):
    med = np.median(x); mad = np.median(np.abs(x - med))
    return med, (1.4826 * mad if mad > 0 else x.std() or 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args()

    ref = json.loads(Path(args.reference).read_text())
    # junction positions [x,y,z] -> full-res voxel (z,y,x)
    P = np.array([j["position"] for j in ref["junctions"]])[:, ::-1]
    struts = [(s["junction0"], s["junction1"]) for s in ref["struts"]]
    print(f"reference: {len(P)} junctions, {len(struts)} designed struts")

    print("loading scan ...")
    raw = tifffile.imread(args.tif)
    t = float(threshold_otsu(raw[::4, ::4, ::4]))
    metal = ndi.binary_dilation(raw >= t, iterations=2)  # tolerance for registration
    shape = np.array(raw.shape)
    print(f"  Otsu {t:.0f}")

    # measure density + gap along every designed strut
    dens = np.zeros(len(struts)); gapf = np.zeros(len(struts))
    for i, (a, b) in enumerate(struts):
        p0, p1 = P[a], P[b]
        ts = np.linspace(0, 1, 24)
        pts = np.round(p0[None]*(1-ts[:, None]) + p1[None]*ts[:, None]).astype(int)
        pts = np.clip(pts, 0, shape-1)
        d = raw[pts[:, 0], pts[:, 1], pts[:, 2]].astype(float)[2:-2]
        h = metal[pts[:, 0], pts[:, 1], pts[:, 2]][2:-2]
        dens[i] = d.mean(); gapf[i] = longest_gap(h) / len(h)

    d_med, d_sig = robust(dens[gapf < 0.15])
    d_cut = d_med - args.k * d_sig

    counts = {"present": 0, "thin": 0, "disconnected": 0, "missing": 0}
    for i in range(len(struts)):
        if gapf[i] >= 0.6: v = "missing"
        elif gapf[i] >= 0.15: v = "disconnected"
        elif dens[i] < d_cut: v = "thin"
        else: v = "present"
        counts[v] += 1
    n = len(struts)
    print("\n=== REFERENCE-BASED DEFECTS (designed struts checked in the scan) ===")
    for k in ("present", "thin", "disconnected", "missing"):
        print(f"  {k:12s}: {counts[k]:6d}  ({100*counts[k]/n:.2f}%)")
    print(f"  total defective: {n-counts['present']} ({100*(n-counts['present'])/n:.2f}%)")


if __name__ == "__main__":
    main()
