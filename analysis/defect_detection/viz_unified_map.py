"""FIGURE 8: the full lattice with EVERY defect type colour-coded (3-D + ortho)."""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

from config import ROOT, STK, BASE
UNI = STK / f"{BASE}_unified_defects_accurate.json"
OUT = ROOT / "analysis/defect_detection/VIZ_8_all_defects.png"

COL = {"present": "#2a4a80", "bent": "#c026d3", "thin": "#eab308",
       "disconnected": "#f97316", "missing": "#ef4444"}
LW = {"present": 0.4, "bent": 2.0, "thin": 2.2, "disconnected": 2.2, "missing": 2.4}
ORDER = ["present", "bent", "thin", "disconnected", "missing"]     # defects drawn on top

plt.rcParams.update({"figure.facecolor": "#0a0a0f", "savefig.facecolor": "#0a0a0f",
                     "text.color": "#e8e8f0", "axes.labelcolor": "#c8c8d4",
                     "xtick.color": "#888", "ytick.color": "#888"})


def main():
    d = json.load(open(UNI))
    S = d["struts"]
    counts = d["meta"]["counts"]
    P0 = np.array([s["p0"] for s in S]); P1 = np.array([s["p1"] for s in S])   # [x,y,z]
    ver = np.array([s["verdict"] for s in S])

    fig = plt.figure(figsize=(36, 32))
    gs = fig.add_gridspec(2, 2, hspace=0.09, wspace=0.07)

    ax = fig.add_subplot(gs[0, 0], projection="3d")
    for cat in ORDER:
        m = ver == cat
        segs = [[(P0[i][0], P0[i][1], P0[i][2]), (P1[i][0], P1[i][1], P1[i][2])]
                for i in np.where(m)[0]]
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors=COL[cat], linewidths=LW[cat],
                                                 alpha=0.9 if cat != "present" else 0.35))
    ax.set_xlim(0, 837); ax.set_ylim(0, 815); ax.set_zlim(0, 761)
    ax.view_init(elev=22, azim=40); ax.set_facecolor("#0a0a0f"); ax.grid(False)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.04, 0.04, 0.07, 1)); a.pane.set_edgecolor((0.2, 0.2, 0.3, 0.4))
    try:
        ax.set_box_aspect((837, 815, 761))
    except Exception:
        pass
    ax.set_title("ALL DEFECTS — 3-D", fontsize=17, fontweight="bold", pad=6)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    for pos, (ai, bi, tl, xl, yl) in zip(
            [gs[0, 1], gs[1, 0], gs[1, 1]],
            [(0, 1, "TOP view  (looking down z)", "x", "y"),
             (0, 2, "FRONT view  (looking down y)", "x", "z"),
             (1, 2, "SIDE view  (looking down x)", "y", "z")]):
        a = fig.add_subplot(pos); a.set_facecolor("#0a0a0f")
        for cat in ORDER:
            m = ver == cat
            seg = np.stack([np.stack([P0[m][:, ai], P0[m][:, bi]], 1),
                            np.stack([P1[m][:, ai], P1[m][:, bi]], 1)], 1)
            a.add_collection(LineCollection(seg, colors=COL[cat], linewidths=LW[cat],
                                            alpha=0.9 if cat != "present" else 0.35, rasterized=True))
        a.set_xlim(P0[:, ai].min() - 10, P0[:, ai].max() + 10)
        a.set_ylim(P0[:, bi].min() - 10, P0[:, bi].max() + 10)
        a.set_aspect("equal"); a.invert_yaxis()
        a.set_title(tl, fontsize=17, fontweight="bold"); a.set_xlabel(xl); a.set_ylabel(yl)

    handles = [Line2D([], [], color=COL[c], lw=4, label=f"{c}  ({counts.get(c,0)})") for c in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=17,
               facecolor="#111", edgecolor="#333", labelcolor="#eee")
    tot = sum(counts.values()) - counts.get("present", 0)
    fig.suptitle(f"COMPLETE AS-BUILT DEFECT MAP  ·  {sum(counts.values()):,} designed struts  ·  "
                 f"{tot:,} defective ({100*tot/sum(counts.values()):.1f}%): "
                 f"missing · disconnected · bent · thin",
                 fontsize=24, fontweight="bold", y=0.975, color="#f0f0f8")
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print("saved", OUT.name)


if __name__ == "__main__":
    main()
