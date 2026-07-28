"""THE COMPLETE SKELETON + GRAPH — full-volume, high-resolution visualization.

Figure A: the entire skeleton (all 767,611 centreline voxels) as a 3-D point
          cloud + three orthographic engineering views, coloured by depth.
Figure B: the entire extracted graph (~17,700 struts + ~3,500 nodes) in 3-D +
          three orthographic views, every strut coloured by its bow so the
          whole lattice reads as a bend heat-map.

Large canvases, dark theme, rasterised layers for crisp full-detail output.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
BENT = STK / f"{BASE}_asbuilt_bent.json"
OUT = ROOT / "analysis/defect_detection"
UM = 58.1

plt.rcParams.update({
    "figure.facecolor": "#0a0a0f", "axes.facecolor": "#0a0a0f",
    "savefig.facecolor": "#0a0a0f", "text.color": "#e8e8f0",
    "axes.labelcolor": "#c8c8d4", "xtick.color": "#8a8a9a", "ytick.color": "#8a8a9a",
})


def style3d(ax):
    ax.set_facecolor("#0a0a0f")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.04, 0.04, 0.07, 1.0))
        a.pane.set_edgecolor((0.2, 0.2, 0.3, 0.4))
    ax.grid(False)
    try:
        ax.set_box_aspect((837, 815, 761))
    except Exception:
        pass


# ================= FIGURE A: FULL SKELETON =================
def figure_skeleton():
    print("loading skeleton coords ...")
    C = np.load(SKELC)["coords"].astype(np.float32)     # (N,3) = (z,y,x)
    z, y, x = C[:, 0], C[:, 1], C[:, 2]
    depth = z
    print(f"  {len(C):,} skeleton voxels")

    fig = plt.figure(figsize=(36, 32))
    gs = fig.add_gridspec(2, 2, hspace=0.09, wspace=0.07)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax.scatter(x, y, z, c=depth, cmap="turbo", s=0.7, alpha=0.9,
               linewidths=0, rasterized=True)
    ax.view_init(elev=22, azim=40); style3d(ax)
    ax.set_title("THE COMPLETE SKELETON — 3-D (all 767,611 centreline voxels, coloured by depth)",
                 fontsize=16, fontweight="bold", pad=6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    for pos, (hx, hy, hc, tl, xl, yl) in zip(
            [gs[0, 1], gs[1, 0], gs[1, 1]],
            [(x, y, z, "TOP view  (looking down z)", "x", "y"),
             (x, z, y, "FRONT view  (looking down y)", "x", "z"),
             (y, z, x, "SIDE view  (looking down x)", "y", "z")]):
        a = fig.add_subplot(pos)
        a.scatter(hx, hy, c=hc, cmap="turbo", s=0.9, alpha=0.95, linewidths=0, rasterized=True)
        a.set_aspect("equal"); a.set_facecolor("#050508")
        a.set_title(tl, fontsize=17, fontweight="bold")
        a.set_xlabel(xl); a.set_ylabel(yl)
        a.invert_yaxis()

    fig.suptitle("FULL AS-BUILT SKELETON  ·  the entire 9×9×9 octet lattice after skeletonization  ·  "
                 "761×815×837 voxels (58.1 µm/voxel)",
                 fontsize=24, fontweight="bold", y=0.975, color="#f0f0f8")
    p = OUT / "VIZ_6_full_skeleton.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved", p.name)


# ================= FIGURE B: FULL GRAPH =================
def figure_graph():
    print("loading graph ...")
    B = json.load(open(BENT))
    S = B["struts"]
    P0 = np.array([s["p0"] for s in S])                 # (z,y,x)
    P1 = np.array([s["p1"] for s in S])
    bow = np.array([s["max_dev"] for s in S])
    nodes = np.unique(np.vstack([np.round(P0, 1), np.round(P1, 1)]), axis=0)
    print(f"  {len(S):,} struts, {len(nodes):,} nodes")

    cmap = plt.get_cmap("turbo")
    norm = np.clip(bow / 12.0, 0, 1)          # 0 = straight (blue) -> bent (red)
    colors = cmap(norm)
    colors[:, 3] = np.where(bow > 3.65, 0.98, 0.5)      # bent struts more opaque
    order = np.argsort(bow)                              # draw bent ones last (on top)

    fig = plt.figure(figsize=(34, 30))
    gs = fig.add_gridspec(2, 2, hspace=0.10, wspace=0.08)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    segs = [[(P0[i][2], P0[i][1], P0[i][0]), (P1[i][2], P1[i][1], P1[i][0])] for i in order]
    lc = Line3DCollection(segs, colors=colors[order],
                          linewidths=np.where(bow[order] > 3.65, 1.6, 0.55))
    ax.add_collection3d(lc)
    ax.scatter(nodes[:, 2], nodes[:, 1], nodes[:, 0], c="#e5f2ff", s=1.4,
               alpha=0.55, linewidths=0, rasterized=True)
    ax.set_xlim(0, 837); ax.set_ylim(0, 815); ax.set_zlim(0, 761)
    ax.view_init(elev=24, azim=38); style3d(ax)
    ax.set_title("THE COMPLETE GRAPH — 3-D (every strut coloured by bow: blue = straight → red = bent)",
                 fontsize=16, fontweight="bold", pad=6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    for pos, (ai, bi, tl, xl, yl) in zip(
            [gs[0, 1], gs[1, 0], gs[1, 1]],
            [(2, 1, "TOP view  (looking down z)", "x", "y"),
             (2, 0, "FRONT view  (looking down y)", "x", "z"),
             (1, 0, "SIDE view  (looking down x)", "y", "z")]):
        a = fig.add_subplot(pos)
        seg2 = np.stack([np.stack([P0[order][:, ai], P0[order][:, bi]], 1),
                         np.stack([P1[order][:, ai], P1[order][:, bi]], 1)], 1)
        lc2 = LineCollection(seg2, colors=colors[order],
                             linewidths=np.where(bow[order] > 3.65, 1.4, 0.45), rasterized=True)
        a.add_collection(lc2)
        a.scatter(nodes[:, ai], nodes[:, bi], c="#e5f2ff", s=1.0, alpha=0.4, linewidths=0, rasterized=True)
        a.set_xlim(nodes[:, ai].min() - 10, nodes[:, ai].max() + 10)
        a.set_ylim(nodes[:, bi].min() - 10, nodes[:, bi].max() + 10)
        a.set_aspect("equal"); a.set_facecolor("#0a0a0f")
        a.set_title(tl, fontsize=16, fontweight="bold"); a.set_xlabel(xl); a.set_ylabel(yl)
        a.invert_yaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 12 * UM))
    cb = fig.colorbar(sm, ax=fig.axes, fraction=0.012, pad=0.01)
    cb.set_label("strut bow (µm)", fontsize=13, color="#c8c8d4")

    nbent = int((bow > 3.65).sum())
    fig.suptitle(f"FULL AS-BUILT GRAPH  ·  {len(S):,} struts · {len(nodes):,} nodes  ·  "
                 f"{nbent:,} bent ({100*nbent/len(S):.1f}%) shown in warm colours",
                 fontsize=24, fontweight="bold", y=0.975, color="#f0f0f8")
    p = OUT / "VIZ_7_full_graph.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); print("saved", p.name)


if __name__ == "__main__":
    figure_skeleton()
    figure_graph()
