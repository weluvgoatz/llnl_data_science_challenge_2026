"""FIGURE 1 + 2: full visual walkthrough of the extraction pipeline.

Renders every stage on real data from the same slab of the scan:
  1 raw CT -> 2 Otsu mask -> 3 cleaned mask -> 4 skeleton
  5 degree map -> 6 detected nodes -> 7 extracted graph -> 8 bend map
plus a high-zoom "anatomy" figure showing how neighbour-counting turns
centrelines into nodes and edges.
"""

import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
RAW = STK / f"{BASE}.tif"
SEG = STK / f"{BASE}_segmented.tif"
CLEAN = STK / f"{BASE}_segmented_clean.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
BENT = STK / f"{BASE}_asbuilt_bent.json"
OUT = ROOT / "analysis/defect_detection"

Z0, HALF, MARGIN = 462, 6, 3          # slab centred on a "mesh" slice
CONN = np.ones((3, 3, 3), int)


def mip(path, z0, half):
    tif = tifffile.TiffFile(path)
    return np.stack([tif.pages[z].asarray() for z in range(z0 - half, z0 + half + 1)]).max(0)


def main():
    OUT.mkdir(exist_ok=True)
    print("loading slab imagery ...")
    raw = mip(RAW, Z0, HALF).astype(float)
    seg = mip(SEG, Z0, HALF) > 0
    clean = mip(CLEAN, Z0, HALF) > 0
    H, W = raw.shape

    # ---- skeleton sub-volume for this slab (from cached coords) ----
    print("building skeleton slab + degree map ...")
    coords = np.load(SKELC)["coords"]
    zlo, zhi = Z0 - HALF - MARGIN, Z0 + HALF + MARGIN
    sel = (coords[:, 0] >= zlo) & (coords[:, 0] <= zhi)
    sc = coords[sel]
    sub = np.zeros((zhi - zlo + 1, H, W), bool)
    sub[sc[:, 0] - zlo, sc[:, 1], sc[:, 2]] = True
    K = np.ones((3, 3, 3), int); K[1, 1, 1] = 0
    degv = ndi.convolve(sub.astype(np.uint8), K, mode="constant") * sub
    core = slice(MARGIN, MARGIN + 2 * HALF + 1)        # drop the margin slices
    skel2d = sub[core].max(0)
    d_end = ((degv == 1) & sub)[core].max(0)
    d_path = ((degv == 2) & sub)[core].max(0)
    d_junc = ((degv >= 3) & sub)[core].max(0)

    # ---- graph (nodes + struts) in this slab ----
    B = json.load(open(BENT))
    S = B["struts"]
    thr = B["meta"]["threshold_vox"]
    in_slab = []
    for s in S:
        p0, p1 = np.array(s["p0"]), np.array(s["p1"])       # (z,y,x)
        if abs((p0[0] + p1[0]) / 2 - Z0) <= HALF + 4:
            in_slab.append((p0, p1, s["max_dev"]))
    print(f"  {len(in_slab)} struts in this slab")

    # ---- crop everything to the lattice bounding box (drop dead border) ----
    yy, xx = np.where(clean)
    y0, y1 = max(yy.min() - 8, 0), min(yy.max() + 9, H)
    x0, x1 = max(xx.min() - 8, 0), min(xx.max() + 9, W)
    cropper = lambda A: A[y0:y1, x0:x1]
    raw, seg, clean = cropper(raw), cropper(seg), cropper(clean)
    skel2d, d_end, d_path, d_junc = map(cropper, (skel2d, d_end, d_path, d_junc))
    in_slab = [(p0 - np.array([0, y0, x0]), p1 - np.array([0, y0, x0]), dv)
               for p0, p1, dv in in_slab]
    H, W = raw.shape
    print(f"  cropped to lattice: {H}x{W}")

    # =================== FIGURE 1: THE PIPELINE ===================
    fig, ax = plt.subplots(2, 4, figsize=(32, 18))
    ax = ax.ravel()
    titles = [
        "STEP 1 — RAW CT\nfuzzy grey values; no notion of 'metal'",
        "STEP 2 — SEGMENTATION (Otsu = 40127)\nevery voxel brighter than the cutoff = metal",
        "STEP 3 — CLEANED MASK\nspeckle <20 vox removed, pinholes closed",
        "STEP 4 — SKELETON (medial axis)\nfat struts thinned to 1-voxel centrelines",
        "STEP 5 — DEGREE MAP (neighbour count)\nblue = along a strut (2)  ·  gold = junction (3+)  ·  red = loose end (1)",
        "STEP 6 — NODES\njunction clusters collapsed to single points",
        "STEP 7 — GRAPH\nnodes joined by struts = the as-built lattice",
        f"STEP 8 — BEND MAP\ngreen = straight   ·   red = bent (bow > {thr*58.1:.0f} um)",
    ]

    ax[0].imshow(raw, cmap="gray")
    ax[1].imshow(seg, cmap="gray")
    ax[2].imshow(clean, cmap="gray")

    ax[3].imshow(clean, cmap="gray_r", alpha=0.16)          # faint ghost of the metal
    ys, xs = np.where(skel2d)
    ax[3].scatter(xs, ys, s=0.9, c="#0891b2", linewidths=0)

    ax[4].imshow(np.zeros_like(raw), cmap="gray", vmin=0, vmax=1)
    for m, col, sz in ((d_path, "#38bdf8", 0.8), (d_junc, "#fbbf24", 4.0), (d_end, "#ef4444", 14.0)):
        yy2, xx2 = np.where(m)
        ax[4].scatter(xx2, yy2, s=sz, c=col, linewidths=0)

    ax[5].imshow(clean, cmap="gray", alpha=0.30)
    seen = set()
    for p0, p1, _ in in_slab:
        for p in (p0, p1):
            k = (round(p[1]), round(p[2]))
            if k not in seen:
                seen.add(k)
                ax[5].scatter(p[2], p[1], s=26, c="#22d3ee", edgecolors="k", linewidths=0.4, zorder=3)

    ax[6].imshow(clean, cmap="gray", alpha=0.22)
    for p0, p1, _ in in_slab:
        ax[6].plot([p0[2], p1[2]], [p0[1], p1[1]], c="#22c55e", lw=1.5, alpha=0.95, zorder=2)
    for (yy, xx) in seen:
        ax[6].scatter(xx, yy, s=20, c="#f43f5e", zorder=3)

    ax[7].imshow(raw, cmap="gray", alpha=0.55)
    for p0, p1, dv in in_slab:
        c = "#ef4444" if dv > thr else "#22c55e"
        lw = 2.4 if dv > thr else 1.2
        ax[7].plot([p0[2], p1[2]], [p0[1], p1[1]], c=c, lw=lw, alpha=0.95)

    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=14, fontweight="bold", pad=14)
        a.axis("off")
    fig.suptitle(
        "FROM CT SCAN TO LATTICE GRAPH — every stage of the pipeline, same slab of the real specimen\n"
        f"({BASE}   ·   slab z = {Z0}±{HALF}   ·   volume 761×815×837 voxels, 58.1 µm/voxel)",
        fontsize=21, fontweight="bold", y=0.975)
    # explicit spacing so the second-row titles never collide with row-1 images
    fig.subplots_adjust(top=0.885, bottom=0.02, left=0.015, right=0.985,
                        wspace=0.06, hspace=0.20)
    p = OUT / "VIZ_1_pipeline_stages.png"
    fig.savefig(p, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p.name)

    # =================== FIGURE 2: ZOOM ANATOMY ===================
    cy, cx = H // 2, W // 2
    R = 110
    sy, sx = slice(cy - R, cy + R), slice(cx - R, cx + R)
    fig, ax = plt.subplots(1, 4, figsize=(27, 7.4))
    ax[0].imshow(raw[sy, sx], cmap="gray")
    ax[0].set_title("raw CT (zoom)", fontsize=14, fontweight="bold")
    ax[1].imshow(clean[sy, sx], cmap="gray")
    ax[1].set_title("segmented metal", fontsize=14, fontweight="bold")

    ax[2].imshow(clean[sy, sx], cmap="gray", alpha=0.25)
    for m, col, sz in ((d_path, "#3b82f6", 3.0), (d_junc, "#fbbf24", 14), (d_end, "#ef4444", 40)):
        yy, xx = np.where(m[sy, sx])
        ax[2].scatter(xx, yy, s=sz, c=col, linewidths=0)
    ax[2].set_title("centreline coloured by NEIGHBOUR COUNT\n2 = along strut · 3+ = junction · 1 = loose end",
                    fontsize=13, fontweight="bold")
    ax[2].legend(handles=[Line2D([], [], marker="o", ls="", color="#3b82f6", label="degree 2 (strut body)"),
                          Line2D([], [], marker="o", ls="", color="#fbbf24", label="degree 3+ (JUNCTION)"),
                          Line2D([], [], marker="o", ls="", color="#ef4444", label="degree 1 (loose end)")],
                 loc="lower right", fontsize=10, framealpha=0.9)

    ax[3].imshow(raw[sy, sx], cmap="gray", alpha=0.5)
    for p0, p1, dv in in_slab:
        y0, x0, y1, x1 = p0[1] - (cy - R), p0[2] - (cx - R), p1[1] - (cy - R), p1[2] - (cx - R)
        if -R < y0 < 2 * R + R and -R < x0 < 2 * R + R:
            c = "#ef4444" if dv > thr else "#22c55e"
            ax[3].plot([x0, x1], [y0, y1], c=c, lw=2.6, alpha=0.95)
            ax[3].scatter([x0, x1], [y0, y1], s=42, c="#22d3ee", edgecolors="k", linewidths=0.6, zorder=4)
    ax[3].set_xlim(0, 2 * R); ax[3].set_ylim(2 * R, 0)
    ax[3].set_title("extracted GRAPH\ncyan = nodes · green = straight strut · red = bent",
                    fontsize=13, fontweight="bold")
    for a in ax:
        a.axis("off")
    fig.suptitle("GRAPH ANATOMY — how a 3-D image becomes nodes and edges (high zoom on one region)",
                 fontsize=19, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "VIZ_2_graph_anatomy.png"
    fig.savefig(p, dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p.name)


if __name__ == "__main__":
    main()
