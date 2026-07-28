"""Reference-based defects WITH re-registration of the blueprint to the scan.

The provided reference JSON is only approximately aligned (a ~0.6% scale drift),
which produces huge false-defect counts. This first refines a global
scale + translation so the designed struts land on the real metal, THEN
measures density/gap along each designed strut.

Usage (from repo root):
    python analysis/defect_detection/reference_based_registered.py \
        --tif "...tif" --reference "...json"
"""

import argparse
import itertools
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
    P = np.array([j["position"] for j in ref["junctions"]])[:, ::-1]  # (z,y,x)
    struts = [(s["junction0"], s["junction1"]) for s in ref["struts"]]

    print("loading scan ...")
    raw = tifffile.imread(args.tif)
    t = float(threshold_otsu(raw[::4, ::4, ::4]))
    metal = ndi.binary_dilation(raw >= t, iterations=2)
    shape = np.array(raw.shape)
    ctr = P.mean(0)

    mids = np.array([(P[a] + P[b]) / 2 for a, b in struts])
    sub = mids[np.random.RandomState(0).choice(len(mids), 3000, replace=False)]

    def hit_frac(scale, trans):
        q = ctr + scale * (sub - ctr) + trans
        q = np.clip(np.round(q).astype(int), 0, shape - 1)
        return metal[q[:, 0], q[:, 1], q[:, 2]].mean()

    # Step A: best global scale
    best_s, best_v = 1.0, -1
    for s in np.linspace(0.985, 1.006, 22):
        v = hit_frac(s, np.zeros(3))
        if v > best_v: best_v, best_s = v, s
    # Step B: best small translation at that scale
    best_t, best_v = np.zeros(3), -1
    for dz, dy, dx in itertools.product(range(-4, 5, 2), repeat=3):
        v = hit_frac(best_s, np.array([dz, dy, dx]))
        if v > best_v: best_v, best_t = v, np.array([dz, dy, dx])
    print(f"registered: scale={best_s:.4f}, shift={best_t.tolist()}, "
          f"strut-midpoint-on-metal {100*best_v:.1f}%")

    def xform(p): return ctr + best_s * (p - ctr) + best_t

    dens = np.zeros(len(struts)); gapf = np.zeros(len(struts))
    for i, (a, b) in enumerate(struts):
        p0, p1 = xform(P[a]), xform(P[b])
        ts = np.linspace(0, 1, 24)
        pts = np.clip(np.round(p0[None]*(1-ts[:, None]) + p1[None]*ts[:, None]).astype(int), 0, shape-1)
        d = raw[pts[:, 0], pts[:, 1], pts[:, 2]].astype(float)[2:-2]
        h = metal[pts[:, 0], pts[:, 1], pts[:, 2]][2:-2]
        dens[i] = d.mean(); gapf[i] = longest_gap(h) / len(h)

    d_med, d_sig = robust(dens[gapf < 0.15]); d_cut = d_med - args.k * d_sig
    counts = {"present": 0, "thin": 0, "disconnected": 0, "missing": 0}
    for i in range(len(struts)):
        if gapf[i] >= 0.6: v = "missing"
        elif gapf[i] >= 0.15: v = "disconnected"
        elif dens[i] < d_cut: v = "thin"
        else: v = "present"
        counts[v] += 1
    n = len(struts)
    print("\n=== REFERENCE-BASED DEFECTS (after re-registration) ===")
    for k in ("present", "thin", "disconnected", "missing"):
        print(f"  {k:12s}: {counts[k]:6d}  ({100*counts[k]/n:.2f}%)")
    print(f"  total defective: {n-counts['present']} ({100*(n-counts['present'])/n:.2f}%)")


if __name__ == "__main__":
    main()
