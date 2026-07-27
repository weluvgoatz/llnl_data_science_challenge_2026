"""Unified per-strut defect classification (all types in one coherent labeling).

For every DESIGNED strut (registered onto the scan) assign ONE verdict:
  missing  · disconnected · thin · bent · present
by combining the material-sampling analysis (missing/disconnected/thin) with
the bend analysis (matched from the as-built graph).

Output: unified_defects.json  — per strut: endpoints (scan x,y,z), verdict, bow.
This single file drives every downstream visualization / 3-D model.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
GT = ROOT / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
BENT = STK / f"{BASE}_asbuilt_bent.json"
MASK = STK / f"{BASE}_segmented_clean.tif"
RAW = STK / f"{BASE}.tif"
OUT = STK / f"{BASE}_unified_defects.json"

UM = 58.1
K_SAMPLES = 24
MISSING_FRAC = 0.15
GAP_FRAC = 0.25
K_OUT = 3.0


def longest_gap(hit):
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1; best = max(best, cur)
    return best


def robust(x):
    m = np.median(x); s = 1.4826 * np.median(np.abs(x - m))
    return m, (s if s > 0 else x.std() or 1.0)


def main():
    gt = json.load(open(GT))
    GP = np.array([j["position"] for j in gt["junctions"]], float)     # [x,y,z]
    gt_struts = [(s["junction0"], s["junction1"]) for s in gt["struts"]]

    B = json.load(open(BENT))
    bthr = B["meta"]["threshold_vox"]
    ab_p0 = np.array([s["p0"] for s in B["struts"]])[:, ::-1]           # (z,y,x)->[x,y,z]
    ab_p1 = np.array([s["p1"] for s in B["struts"]])[:, ::-1]
    ab_bow = np.array([s["max_dev"] for s in B["struts"]])
    ab_mid = (ab_p0 + ab_p1) / 2
    ab_nodes = np.unique(np.round(np.vstack([ab_p0, ab_p1]), 1), axis=0)

    # ---- register designed lattice -> scan coords (similarity ICP) ----
    tree = cKDTree(ab_nodes)
    s, t = 1.0, np.zeros(3)
    for _ in range(25):
        q = s * GP + t
        dd, idx = tree.query(q)
        m = dd < 40
        X, Y = GP[m], ab_nodes[idx[m]]
        mx, my = X.mean(0), Y.mean(0)
        s = float(((Y - my) * (X - mx)).sum() / ((X - mx) ** 2).sum())
        t = my - s * mx
    GPs = s * GP + t
    print(f"registered ground truth -> scan: scale {s:.4f}")

    print("loading mask + raw ...")
    mask = tifffile.imread(MASK) > 0
    metal = ndi.binary_dilation(mask, iterations=1); del mask
    raw = tifffile.imread(RAW)
    shape = np.array(metal.shape)                                       # (z,y,x)

    # ---- sample material + density along each designed strut ----
    n = len(gt_struts)
    frac = np.zeros(n); gapf = np.zeros(n); dens = np.zeros(n)
    ts = np.linspace(0, 1, K_SAMPLES)
    P0s = GPs[[a for a, b in gt_struts]]
    P1s = GPs[[b for a, b in gt_struts]]
    for i in range(n):
        pts = P0s[i][None] * (1 - ts[:, None]) + P1s[i][None] * ts[:, None]     # [x,y,z]
        vz = np.clip(np.round(pts[:, 2]).astype(int), 0, shape[0] - 1)
        vy = np.clip(np.round(pts[:, 1]).astype(int), 0, shape[1] - 1)
        vx = np.clip(np.round(pts[:, 0]).astype(int), 0, shape[2] - 1)
        hit = metal[vz, vy, vx]
        mid = hit[2:-2]
        frac[i] = mid.mean(); gapf[i] = longest_gap(mid) / len(mid)
        dens[i] = raw[vz, vy, vx][2:-2].mean()

    healthy = (frac > 0.9) & (gapf < 0.15)
    dm, ds = robust(dens[healthy]); d_cut = dm - K_OUT * ds

    # ---- bend: match each designed strut midpoint to the as-built graph ----
    gt_mid = (P0s + P1s) / 2
    dd, mi = cKDTree(ab_mid).query(gt_mid)
    matched_bow = np.where(dd < 18, ab_bow[mi], 0.0)

    # ---- assign one verdict (priority: missing>disc>thin>bent>present) ----
    verdict = np.empty(n, dtype=object)
    for i in range(n):
        if frac[i] < MISSING_FRAC:
            v = "missing"
        elif gapf[i] >= GAP_FRAC:
            v = "disconnected"
        elif dens[i] < d_cut:
            v = "thin"
        elif matched_bow[i] > bthr:
            v = "bent"
        else:
            v = "present"
        verdict[i] = v

    from collections import Counter
    c = Counter(verdict)
    print("\n=== UNIFIED DEFECTS (designed struts) ===")
    for k in ("present", "bent", "thin", "disconnected", "missing"):
        print(f"  {k:12s}: {c[k]:5d}  ({100*c[k]/n:.2f}%)")
    defects = n - c["present"]
    print(f"  TOTAL defective: {defects}  ({100*defects/n:.2f}%)")

    out = {"struts": [{"p0": [float(x) for x in P0s[i]], "p1": [float(x) for x in P1s[i]],
                       "verdict": verdict[i], "bow_um": float(matched_bow[i] * UM)}
                      for i in range(n)],
           "meta": {"scale": s, "n": n, "counts": dict(c),
                    "volume_shape_zyx": [int(x) for x in shape]}}
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"\nSaved: {OUT.name}")


if __name__ == "__main__":
    main()
