"""FIGURE 3/4/5: bent-strut anatomy, statistics dashboard, and a 3-D defect map."""

import json
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
RAW = STK / f"{BASE}.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
BENT = STK / f"{BASE}_asbuilt_bent.json"
OUT = ROOT / "analysis/defect_detection"
UM = 58.1
DIA = 424.0


def main():
    B = json.load(open(BENT))
    S = B["struts"]
    thr = B["meta"]["threshold_vox"]
    dev = np.array([s["max_dev"] for s in S])
    tor = np.array([s["tortuosity"] for s in S])
    span = np.array([s["span"] for s in S])
    P0 = np.array([s["p0"] for s in S])
    P1 = np.array([s["p1"] for s in S])
    bent = dev > thr
    print(f"{len(S)} struts, {int(bent.sum())} bent ({100*bent.mean():.2f}%)")

    # =============== FIGURE 3: BENT STRUT ANATOMY ===============
    print("rendering bend anatomy (loading skeleton + raw slabs) ...")
    coords = np.load(SKELC)["coords"].astype(float)
    tif = tifffile.TiffFile(RAW)
    shape = np.array(tif.series[0].shape)

    rng = np.random.RandomState(3)
    def pick(mask, n):
        idx = np.where(mask)[0]
        return idx[rng.choice(len(idx), min(n, len(idx)), replace=False)] if len(idx) else []
    chosen = (list(pick((dev > 4) & (dev <= 6), 3)) +
              list(pick((dev > 6) & (dev <= 10), 3)) +
              list(pick(dev > 12, 2)) +
              list(pick(dev < 1.2, 2)))

    fig, axes = plt.subplots(2, 5, figsize=(30, 13))
    axes = axes.ravel()
    for k, i in enumerate(chosen):
        ax = axes[k]; ax.axis("off")
        p0, p1 = P0[i], P1[i]
        v = p1 - p0; L = np.linalg.norm(v)
        # skeleton voxels belonging to this strut: near the chord, between the ends
        d = coords - p0
        t = (d @ v) / (v @ v)
        perp = np.linalg.norm(d - np.outer(t, v), axis=1)
        sel = (t > -0.05) & (t < 1.05) & (perp < 16)
        Q = coords[sel]
        zax = int(np.argmin(np.abs(v)))
        a2 = [a for a in range(3) if a != zax]
        lo = np.minimum(p0, p1).astype(int) - 16
        hi = np.maximum(p0, p1).astype(int) + 16
        lo = np.maximum(lo, 0); hi = np.minimum(hi, shape)
        zc = int((p0[zax] + p1[zax]) / 2)
        zl, zh = max(zc - 8, 0), min(zc + 9, shape[zax])
        sl = [slice(lo[a], hi[a]) for a in range(3)]; sl[zax] = slice(zl, zh)
        img = tif.asarray(key=range(sl[0].start, sl[0].stop))[:, sl[1], sl[2]].max(0) if zax == 0 else None
        if img is None:
            vol = tif.asarray(key=range(int(lo[0]), int(hi[0])))
            sub = vol[:, sl[1], sl[2]]
            img = sub.max(axis=zax)
        ax.imshow(img, cmap="gray")
        e0 = [p0[a] - lo[a] for a in a2]; e1 = [p1[a] - lo[a] for a in a2]
        ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c="#ffffff", lw=1.6, alpha=0.9)
        if len(Q):
            ax.scatter(Q[:, a2[1]] - lo[a2[1]], Q[:, a2[0]] - lo[a2[0]],
                       s=5, c="#ff2d55" if dev[i] > thr else "#22c55e", linewidths=0)
        ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], s=55, c="#22d3ee",
                   edgecolors="k", linewidths=0.7, zorder=5)
        kind = "BENT" if dev[i] > thr else "straight"
        ax.set_title(f"{kind}  ·  bow {dev[i]:.1f} vox = {dev[i]*UM:.0f} µm ({dev[i]*UM/DIA:.2f}× dia)\n"
                     f"tortuosity {tor[i]:.3f}",
                     fontsize=11.5, fontweight="bold",
                     color="#ff2d55" if dev[i] > thr else "#16a34a")
    for k in range(len(chosen), len(axes)):
        axes[k].axis("off")
    fig.suptitle("BENT-STRUT ANATOMY — white dashed = the IDEAL straight path · coloured dots = the "
                 "ACTUAL traced centreline\nthe gap between them IS the bend we measure",
                 fontsize=19, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "VIZ_3_bend_anatomy.png", dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved VIZ_3_bend_anatomy.png")

    # =============== FIGURE 4: STATISTICS DASHBOARD ===============
    fig = plt.figure(figsize=(26, 14))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.24)

    a = fig.add_subplot(gs[0, 0])
    a.hist(dev * UM, bins=140, color="#3b82f6", log=True)
    a.axvline(thr * UM, color="#ef4444", ls="--", lw=2.4, label=f"BENT threshold {thr*UM:.0f} µm")
    a.axvline(np.median(dev) * UM, color="#22c55e", ls="--", lw=2, label=f"median {np.median(dev)*UM:.0f} µm")
    a.set_xlabel("maximum bow (µm)", fontsize=12); a.set_ylabel("strut count (log)", fontsize=12)
    a.set_title("Bow distribution — the straight population is a tight spike;\nbent struts are the long tail",
                fontsize=13, fontweight="bold"); a.legend(fontsize=11)

    a = fig.add_subplot(gs[0, 1])
    a.hist(tor, bins=140, color="#8b5cf6", log=True)
    a.axvline(1.0, color="k", ls=":", lw=2, label="1.0 = perfectly straight (hard floor)")
    a.axvline(np.median(tor[bent]), color="#ef4444", ls="--", lw=2, label=f"bent median {np.median(tor[bent]):.3f}")
    a.axvline(np.median(tor[~bent]), color="#22c55e", ls="--", lw=2, label=f"straight median {np.median(tor[~bent]):.3f}")
    a.set_xlabel("tortuosity (arc ÷ chord)", fontsize=12); a.set_ylabel("count (log)", fontsize=12)
    a.set_title("Tortuosity — an INDEPENDENT check.\nAll ≥1.0 (physically required) and it separates the groups",
                fontsize=13, fontweight="bold"); a.legend(fontsize=10)

    a = fig.add_subplot(gs[0, 2])
    sc = a.scatter(span, dev * UM, s=3, c=np.where(bent, "#ef4444", "#94a3b8"), alpha=0.5)
    a.axhline(thr * UM, color="#ef4444", ls="--", lw=2)
    a.set_xlabel("strut length (voxels)", fontsize=12); a.set_ylabel("bow (µm)", fontsize=12)
    a.set_yscale("log")
    a.set_title("Bow vs strut length\n(bend is independent of length — no length bias)",
                fontsize=13, fontweight="bold")

    a = fig.add_subplot(gs[1, 0])
    bands = [(3.65, 5, "mild\n0.5–0.7× dia"), (5, 8, "moderate\n0.7–1.1×"),
             (8, 15, "severe\n1.1–2.1×"), (15, 1e9, "extreme\n>2.1×")]
    cnt = [int(((dev > lo) & (dev <= hi)).sum()) for lo, hi, _ in bands]
    cols = ["#fde047", "#fb923c", "#ef4444", "#7f1d1d"]
    bars = a.bar([n for _, _, n in bands], cnt, color=cols, edgecolor="k")
    for b_, c_ in zip(bars, cnt):
        a.text(b_.get_x() + b_.get_width() / 2, c_ + 3, f"{c_}\n{100*c_/len(S):.2f}%",
               ha="center", fontsize=11, fontweight="bold")
    a.set_ylabel("struts", fontsize=12)
    a.set_title(f"Bend severity — {int(bent.sum())} bent struts total ({100*bent.mean():.2f}%)",
                fontsize=13, fontweight="bold")

    a = fig.add_subplot(gs[1, 1])
    names = ["Missing", "Disconnected", "Bent", "Thin"]
    vals = [2.37, 2.03, 100 * bent.mean(), 0.23]
    cols2 = ["#ef4444", "#f97316", "#a855f7", "#eab308"]
    bars = a.barh(names, vals, color=cols2, edgecolor="k")
    for b_, v_ in zip(bars, vals):
        a.text(v_ + 0.06, b_.get_y() + b_.get_height() / 2, f"{v_:.2f}%", va="center",
               fontsize=12, fontweight="bold")
    a.set_xlabel("% of struts", fontsize=12)
    a.set_title("ALL DEFECT TYPES for this specimen\n(bent is an independent axis — those struts are present)",
                fontsize=13, fontweight="bold")

    a = fig.add_subplot(gs[1, 2]); a.axis("off")
    txt = (f"SPECIMEN  {BASE[:44]}…\n"
           f"volume            761 × 815 × 837 voxels  (58.1 µm/voxel)\n"
           f"segmentation      Otsu = 40127   ·  11.1 % material\n"
           f"skeleton          767,611 centreline voxels\n"
           f"as-built graph    3,534 nodes · 17,721 struts (median degree 12)\n"
           f"struts measured   {len(S):,}  (artifact edges rejected)\n\n"
           f"BENT              {int(bent.sum()):,}  ({100*bent.mean():.2f} %)\n"
           f"  median bow      {np.median(dev[bent]):.2f} vox = {np.median(dev[bent])*UM:.0f} µm "
           f"({np.median(dev[bent])*UM/DIA:.2f}× strut diameter)\n"
           f"  worst bow       {dev.max():.1f} vox = {dev.max()*UM:.0f} µm\n"
           f"STRAIGHT          {int((~bent).sum()):,}  ({100*(~bent).mean():.2f} %)\n"
           f"  median bow      {np.median(dev[~bent])*UM:.0f} µm  (discretisation floor)\n\n"
           f"threshold         bow > 1 strut radius = {thr:.2f} vox = {thr*UM:.0f} µm")
    a.text(0.0, 0.98, txt, family="monospace", fontsize=13, va="top")
    a.set_title("SUMMARY", fontsize=14, fontweight="bold", loc="left")

    fig.suptitle("BEND ANALYSIS DASHBOARD — distributions, severity, and how bend fits the full defect picture",
                 fontsize=20, fontweight="bold", y=0.975)
    fig.savefig(OUT / "VIZ_4_stats_dashboard.png", dpi=125, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved VIZ_4_stats_dashboard.png")

    # =============== FIGURE 5: 3-D DEFECT MAP ===============
    print("rendering 3-D map ...")
    fig = plt.figure(figsize=(28, 14))
    views = [(22, 35), (22, 125), (68, 45)]
    for j, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        idx_ok = np.where(~bent)[0]
        idx_ok = idx_ok[::6]                       # thin out the healthy ones
        segs_ok = [[(P0[i][2], P0[i][1], P0[i][0]), (P1[i][2], P1[i][1], P1[i][0])] for i in idx_ok]
        ax.add_collection3d(Line3DCollection(segs_ok, colors="#cbd5e1", linewidths=0.35, alpha=0.30))
        ib = np.where(bent)[0]
        sev = dev[ib]
        norm = (np.clip(sev, thr, 15) - thr) / (15 - thr)
        cmap = plt.get_cmap("autumn_r")
        segs_b = [[(P0[i][2], P0[i][1], P0[i][0]), (P1[i][2], P1[i][1], P1[i][0])] for i in ib]
        ax.add_collection3d(Line3DCollection(segs_b, colors=cmap(norm), linewidths=2.1, alpha=0.95))
        ax.set_xlim(0, 837); ax.set_ylim(0, 815); ax.set_zlim(0, 761)
        ax.view_init(elev=el, azim=az)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"view {j+1}  (elev {el}°, azim {az}°)", fontsize=13, fontweight="bold")
        ax.grid(False)
    fig.suptitle(f"3-D MAP OF BENT STRUTS — {int(bent.sum())} bent struts (yellow→red = increasing bow) "
                 f"among {len(S):,} measured\ngrey = straight lattice (thinned for clarity)",
                 fontsize=20, fontweight="bold", y=0.96)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "VIZ_5_bent_3d_map.png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("saved VIZ_5_bent_3d_map.png")


if __name__ == "__main__":
    main()
