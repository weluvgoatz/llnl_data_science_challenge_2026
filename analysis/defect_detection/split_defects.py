"""Split defects into the paper's categories: missing / disconnected / thin.

For every designed strut (registered onto the scan) we sample the actual
material along its path and read the pattern:
  - path essentially empty            -> MISSING (truly absent)
  - partial material with a real gap  -> DISCONNECTED (broken)
  - continuous but low density        -> THIN
  - continuous and full density       -> PRESENT

Registration: align the ground-truth junctions to the as-built skeleton nodes
(which live in true scan-voxel coordinates) with a similarity ICP.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
GT = ROOT / "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
ASBUILT = STK / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_segmented_clean_asbuilt_graph_cleaned.json"
MASK = STK / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices_segmented_clean.tif"
RAW = STK / "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"

K_SAMPLES = 24
MISSING_FRAC = 0.15     # path emptier than this -> truly missing
GAP_FRAC = 0.25         # contiguous empty run this long -> disconnected
K_OUT = 3.0             # density outlier strength for "thin"


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
    GP = np.array([j["position"] for j in gt["junctions"]], float)      # [x,y,z]
    gt_struts = [(s["junction0"], s["junction1"]) for s in gt["struts"]]

    ab = json.load(open(ASBUILT))
    AB = np.array([j["position"] for j in ab["junctions"]], float)      # [x,y,z] scan coords

    # --- register ground truth -> scan coords (similarity ICP) ---
    tree = cKDTree(AB)
    s, t = 1.0, np.zeros(3)
    for _ in range(25):
        q = s * GP + t
        dd, idx = tree.query(q)
        m = dd < 40
        X, Y = GP[m], AB[idx[m]]
        mx, my = X.mean(0), Y.mean(0)
        s = float(((Y - my) * (X - mx)).sum() / ((X - mx) ** 2).sum())
        t = my - s * mx
    print(f"registered ground truth -> scan: scale {s:.4f}, shift {np.round(t,1).tolist()}")
    GPs = s * GP + t                                                    # [x,y,z] scan coords

    print("loading mask + raw ...")
    mask = tifffile.imread(MASK) > 0
    # dilation=1: keep small gaps visible so real disconnections aren't bridged
    metal = ndi.binary_dilation(mask, iterations=1); del mask
    raw = tifffile.imread(RAW)
    shape = np.array(metal.shape)                                       # (z,y,x)

    # --- sample material + density along each designed strut ---
    dens = np.zeros(len(gt_struts)); gapf = np.zeros(len(gt_struts)); frac = np.zeros(len(gt_struts))
    ts = np.linspace(0, 1, K_SAMPLES)
    for i, (a, b) in enumerate(gt_struts):
        p0, p1 = GPs[a], GPs[b]                                          # [x,y,z]
        pts = p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]
        # to voxel index (z,y,x): reverse xyz
        vz = np.clip(np.round(pts[:, 2]).astype(int), 0, shape[0] - 1)
        vy = np.clip(np.round(pts[:, 1]).astype(int), 0, shape[1] - 1)
        vx = np.clip(np.round(pts[:, 0]).astype(int), 0, shape[2] - 1)
        hit = metal[vz, vy, vx]
        d = raw[vz, vy, vx].astype(float)
        mid = hit[2:-2]
        frac[i] = mid.mean()
        gapf[i] = longest_gap(mid) / len(mid)
        dens[i] = d[2:-2].mean()

    # density cutoff for "thin" from the healthy (continuous, filled) population
    healthy = (frac > 0.9) & (gapf < 0.15)
    dm, ds = robust(dens[healthy])
    d_cut = dm - K_OUT * ds

    counts = {"present": 0, "thin": 0, "disconnected": 0, "missing": 0}
    for i in range(len(gt_struts)):
        if frac[i] < MISSING_FRAC:
            v = "missing"
        elif gapf[i] >= GAP_FRAC:
            v = "disconnected"
        elif dens[i] < d_cut:
            v = "thin"
        else:
            v = "present"
        counts[v] += 1

    n = len(gt_struts)
    print("\n" + "=" * 60)
    print("DEFECT BREAKDOWN (designed struts vs the as-built scan)")
    print("=" * 60)
    for k in ("present", "thin", "disconnected", "missing"):
        print(f"  {k:12s}: {counts[k]:5d}  ({100*counts[k]/n:.2f}%)")
    defects = n - counts["present"]
    print(f"  {'TOTAL defects':12s}: {defects:5d}  ({100*defects/n:.2f}%)")
    print("\n  paper (this specimen): 0.57% missing, 5.13% disconnected")


if __name__ == "__main__":
    main()
