"""Histograms justifying the thin and bent thresholds.

Each panel shows the measured population, where the threshold sits in it, and
the separation between the healthy bulk and the flagged tail.  The point of the
figure is that neither threshold cuts through the middle of the distribution:
each sits out in a tail, so a strut has to be a genuine outlier to be flagged.

  thin  median - 3*MAD of the scan's own diameters (recomputed per scan)
  bent  one strut radius, a physical scale (the bow at which a strut no longer
        overlaps its own ideal axis)

Writes threshold_justification.png (both panels) plus the two panels separately.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
from detect_v2 import OUT_JSON, UM, NOM_D  # noqa: E402

OUT = V2 / "threshold_plots"
HEALTHY = "#6aa9ff"
FLAG_THIN = "#f0c800"
FLAG_BENT = "#f050dc"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)


def thin_panel(ax, S, thr):
    v = np.array([r["dia_min_um"] for r in S
                  if r.get("dia_min_um") is not None], float)
    med = float(np.median([r["dia_um"] for r in S if r.get("dia_um") is not None]))
    mad = float(np.median(np.abs(
        np.array([r["dia_um"] for r in S if r.get("dia_um") is not None]) - med)))
    cut = thr["thin_cut_um"]

    # Clip the view: a handful of struts inside the fused end-caps sit in nearly
    # solid metal, so the distance transform there reports the blob, not a strut,
    # and stretches the axis to 2 mm.  Bins of ~18 um also matter: the transform
    # is quantised (distances are sqrt of integers), so narrow bins comb the
    # histogram into stripes that look like structure and are not.
    hi = 620.0
    n_clip = int((v > hi).sum())
    bins = np.arange(0, hi + 18, 18)
    n, edges, patches = ax.hist(np.clip(v, None, hi), bins=bins, color=HEALTHY,
                                edgecolor="white", linewidth=0.5)
    for p, left in zip(patches, edges[:-1]):
        if left < cut:
            p.set_facecolor(FLAG_THIN)

    ax.axvline(cut, color="#c8a200", lw=2.6, zorder=6)
    ax.axvline(med, color="#1f4e9c", lw=2.0, ls="--", zorder=6)
    for k, a in ((1, 0.5), (2, 0.35)):
        ax.axvline(med - k * mad, color="#1f4e9c", lw=1.0, ls=":", alpha=a, zorder=4)

    n_flag = int((v < cut).sum())
    ax.set_yscale("log")
    ax.set_xlim(0, hi)
    ax.set_xlabel("thinnest sustained section along the strut  (um)", fontsize=10)
    ax.set_ylabel("struts  (log scale)", fontsize=10)
    ax.set_title("THIN — the threshold is derived from this scan's own struts",
                 fontsize=12.5, fontweight="bold", color="#a88600")
    ax.annotate(f"THIN  <  {cut:.0f} um\n= median - 3 x MAD\n{n_flag:,} struts flagged",
                xy=(cut, 3), xytext=(0.055, 0.60), textcoords="axes fraction",
                fontsize=10, fontweight="bold", color="#8a6f00",
                arrowprops=dict(arrowstyle="->", color="#8a6f00", lw=1.5))
    ax.annotate(f"healthy median\n{med:.0f} um   (MAD {mad:.0f})",
                xy=(med, 1600), xytext=(0.44, 0.80), textcoords="axes fraction",
                fontsize=10, fontweight="bold", color="#1f4e9c",
                arrowprops=dict(arrowstyle="->", color="#1f4e9c", lw=1.5))
    ax.text(0.985, 0.96,
            f"dotted lines: -1 and -2 MAD\n"
            f"3 MAD is deliberately conservative\n\n"
            f"nominal spec {NOM_D:.0f} um  ·  as printed {med:.0f} um\n"
            f"the part is thinner than its own drawing,\n"
            f"which is exactly why the cutoff is measured\n"
            f"from the scan instead of taken from spec"
            + (f"\n\n{n_clip} struts beyond {hi:.0f} um not shown\n"
               f"(inside the fused end-caps)" if n_clip else ""),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            bbox=dict(fc="white", ec="#bbb", lw=0.8, pad=0.55))
    style(ax)
    return med, mad, cut, n_flag


def bent_panel(ax, S, thr):
    v = np.array([r["bow_um"] for r in S if r.get("bow_um") is not None], float)
    pres = np.array([r["bow_um"] for r in S
                     if r["verdict"] == "present" and r.get("bow_um") is not None],
                    float)
    cut = thr["bent_bow_um"]

    hi = 560.0
    n_clip = int((v > hi).sum())
    bins = np.arange(0, hi + 10, 10)
    n, edges, patches = ax.hist(np.clip(v, None, hi), bins=bins, color=HEALTHY,
                                edgecolor="white", linewidth=0.5)
    for p, left in zip(patches, edges[:-1]):
        if left >= cut:
            p.set_facecolor(FLAG_BENT)

    ax.axvline(cut, color="#b0219a", lw=2.6, zorder=6)
    pm = float(np.median(pres))
    ax.axvline(pm, color="#1f4e9c", lw=2.0, ls="--", zorder=6)

    n_flag = int((v >= cut).sum())
    ax.set_yscale("log")
    ax.set_xlim(0, hi)
    ax.set_xlabel("sustained bow of the strut's centreline off its own chord  (um)",
                  fontsize=10)
    ax.set_ylabel("struts  (log scale)", fontsize=10)
    ax.set_title("BENT — the threshold is a physical scale, one strut radius",
                 fontsize=12.5, fontweight="bold", color="#a0208c")
    ax.annotate(f"BENT  >  {cut:.0f} um\n= one strut radius\n{n_flag:,} struts flagged",
                xy=(cut, 40), xytext=(0.46, 0.62), textcoords="axes fraction",
                fontsize=10, fontweight="bold", color="#8c1a7c",
                arrowprops=dict(arrowstyle="->", color="#8c1a7c", lw=1.5))
    ax.annotate(f"healthy struts\npeak at {pm:.0f} um",
                xy=(pm, 1500), xytext=(0.22, 0.83), textcoords="axes fraction",
                fontsize=10, fontweight="bold", color="#1f4e9c",
                arrowprops=dict(arrowstyle="->", color="#1f4e9c", lw=1.5))
    ax.text(0.985, 0.96,
            "why one radius: bowed by more than its own\n"
            "radius, the strut no longer overlaps the straight\n"
            "axis it was meant to follow\n\n"
            "the healthy bulk dies away well before the line —\n"
            "the flagged struts are a separate tail, not the\n"
            "edge of the main population"
            + (f"\n\n{n_clip} struts beyond {hi:.0f} um not shown"
               if n_clip else ""),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
            bbox=dict(fc="white", ec="#bbb", lw=0.8, pad=0.55))
    style(ax)
    return pm, cut, n_flag


def main():
    OUT.mkdir(exist_ok=True)
    d = json.load(open(OUT_JSON))
    S, thr = d["struts"], d["meta"]["thresholds"]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 11), facecolor="white")
    med, mad, tcut, tn = thin_panel(axes[0], S, thr)
    pm, bcut, bn = bent_panel(axes[1], S, thr)
    fig.suptitle("Where the thresholds sit in the measured population\n"
                 "neither cuts through the bulk — a strut must be a genuine "
                 "outlier to be flagged",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(OUT / "threshold_justification.png", dpi=145,
                bbox_inches="tight", facecolor="white")

    for name, fn, col in (("thin", thin_panel, None), ("bent", bent_panel, None)):
        f, a = plt.subplots(figsize=(11, 5.6), facecolor="white")
        fn(a, S, thr)
        f.tight_layout()
        f.savefig(OUT / f"threshold_{name}.png", dpi=145,
                  bbox_inches="tight", facecolor="white")
        plt.close(f)

    print(f"THIN : healthy median {med:.0f} um, MAD {mad:.0f} -> cutoff {tcut:.0f} um"
          f"  ({tn:,} flagged)")
    print(f"BENT : healthy median bow {pm:.0f} um -> cutoff {bcut:.0f} um"
          f"  ({bn:,} flagged)")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
