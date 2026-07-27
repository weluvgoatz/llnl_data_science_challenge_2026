"""Convert a CT .tif lattice scan into a lattice-graph JSON with measurements.

Extracts the octet lattice as a graph (junctions + struts) reconstructed from
the scan's periodicity, and attaches PHYSICAL MEASUREMENTS to every strut:
mean/min density (raw CT intensity), thickness (distance transform), metal
fraction, and the longest contiguous gap. Defects can then be found by simple
math on this JSON - no need to reprocess the 1 GB volume.

Output JSON schema (mirrors the reference, plus measured fields):
  {
    "junctions": [{"id", "position": [x,y,z], "grid_index": [a,b,c]}, ...],
    "struts":    [{"id", "junction0", "junction1", "length",
                   "mean_density", "min_density", "mean_thickness",
                   "metal_fraction", "gap_frac"}, ...],
    "meta": {...}
  }

Usage (from repo root):
    python analysis/defect_detection/tif_to_lattice_json.py --tif "path/to/scan.tif"
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

DS = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tif_path = Path(args.tif).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else \
        tif_path.parent / (tif_path.stem + "_lattice_graph.json")

    print(f"Loading {tif_path.name} ...")
    raw = tifffile.imread(tif_path)               # full-res, for density
    vol_ds = raw[::DS, ::DS, ::DS]
    t = float(threshold_otsu(vol_ds))
    mask = vol_ds >= t
    metal = ndi.binary_dilation(mask, iterations=1)
    edt = ndi.distance_transform_edt(mask)
    print(f"  {raw.shape}, Otsu={t:.0f}, material={100*mask.mean():.2f}%")

    # --- grid inference (reference-free) ---
    nodes = peak_local_max(edt, min_distance=5, threshold_abs=1.0).astype(float)
    half = np.zeros(3); origin = np.zeros(3)
    for ax in range(3):
        c = nodes[:, ax]; hs = np.linspace(8.5, 12, 400)
        R = [np.abs(np.exp(1j*2*np.pi*c/h).mean()) for h in hs]; h = hs[int(np.argmax(R))]
        ang = np.angle(np.exp(1j*2*np.pi*c/h).mean()); half[ax] = h; origin[ax] = (ang/(2*np.pi))*h % h
    g = np.round((nodes-origin)/half).astype(int)
    best = (0, 0, 0); bs = -1
    for ph in itertools.product((0, 1), repeat=3):
        sc = np.mean(np.isin(((g+np.array(ph)) % 2 != 0).sum(1), [0, 2]))
        if sc > bs: bs, best = sc, ph
    g = g + np.array(best)

    def g2v(G):
        return np.array(G, float) * half + origin - np.array(best) * half

    def has_metal(G):
        v = np.round(g2v(G)).astype(int)
        if np.any(v < 0) or np.any(v >= np.array(mask.shape)): return False
        lo = np.maximum(v-1, 0); hi = np.minimum(v+2, mask.shape)
        return metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any()

    gmin = g.min(0)-1; gmax = g.max(0)+1
    ideal = [(a, b, c) for a in range(gmin[0], gmax[0]+1) for b in range(gmin[1], gmax[1]+1)
             for c in range(gmin[2], gmax[2]+1)
             if ((a % 2 != 0)+(b % 2 != 0)+(c % 2 != 0)) in (0, 2) and has_metal((a, b, c))]
    iset = set(ideal)
    idx_of = {n: i for i, n in enumerate(ideal)}
    base = np.array(ideal).min(0)

    vecs = []
    for i in range(3):
        for j in range(i+1, 3):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0, 0, 0]; v[i] = si; v[j] = sj; vecs.append(tuple(v))
    struts = set()
    for n in ideal:
        for v in vecs:
            m = (n[0]+v[0], n[1]+v[1], n[2]+v[2])
            if m in iset: struts.add(tuple(sorted((n, m))))
    struts = sorted(struts)
    print(f"  junctions {len(ideal)}, struts {len(struts)}")

    # --- junctions JSON ---
    junctions = []
    for n in ideal:
        v = g2v(n) * DS                      # full-res voxel coords (z,y,x)
        junctions.append({"id": idx_of[n],
                          "position": [float(v[2]), float(v[1]), float(v[0])],  # [x,y,z]
                          "grid_index": [int(n[0]-base[0]), int(n[1]-base[1]), int(n[2]-base[2])]})

    # --- per-strut measurements ---
    shape_ds = np.array(mask.shape); shape_full = np.array(raw.shape)

    def longest_gap(hit):
        best_ = cur = 0
        for h in hit:
            cur = 0 if h else cur+1; best_ = max(best_, cur)
        return best_

    strut_json = []
    for sid, (n, m) in enumerate(struts):
        p0d, p1d = g2v(n), g2v(m)
        ts = np.linspace(0, 1, 20)
        # downsampled samples (mask/edt)
        pd = np.round(p0d[None]*(1-ts[:, None]) + p1d[None]*ts[:, None]).astype(int)
        pd = np.clip(pd, 0, shape_ds-1)
        hit = metal[pd[:, 0], pd[:, 1], pd[:, 2]]
        thick = edt[pd[:, 0], pd[:, 1], pd[:, 2]]
        # full-res samples (raw density)
        pf = np.round((p0d*DS)[None]*(1-ts[:, None]) + (p1d*DS)[None]*ts[:, None]).astype(int)
        pf = np.clip(pf, 0, shape_full-1)
        dens = raw[pf[:, 0], pf[:, 1], pf[:, 2]].astype(float)
        mid = slice(2, -2)
        strut_json.append({
            "id": sid,
            "junction0": idx_of[n], "junction1": idx_of[m],
            "length": float(np.linalg.norm((p1d-p0d)*DS)),
            "mean_density": float(dens[mid].mean()),
            "min_density": float(dens[mid].min()),
            "mean_thickness": float(thick[hit].mean()) if hit.any() else 0.0,
            "metal_fraction": float(hit[mid].mean()),
            "gap_frac": float(longest_gap(hit[mid]) / len(hit[mid])),
        })

    out = {"junctions": junctions, "struts": strut_json,
           "meta": {"source_tif": tif_path.name, "otsu": t,
                    "cell_spacing_vox": [float(x) for x in (2*half*DS)],
                    "volume_shape": list(map(int, raw.shape))}}
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Saved lattice graph JSON: {out_path}  ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
