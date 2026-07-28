"""Find lattice defects by statistical analysis of the lattice-graph JSON.

Loads the JSON produced by tif_to_lattice_json.py and classifies each strut
from its measured density/thickness/gap - using the POPULATION statistics of
the lattice itself, not hard-coded thresholds. A defective strut is one whose
density or thickness is a statistical outlier below the healthy population, or
that contains a real contiguous gap.

Logic:
  - missing:      almost no metal / a gap spanning most of the strut
  - disconnected: a real contiguous gap in the middle
  - thin:         continuous, but density OR thickness is a low outlier
                  (below median - k*MAD of the healthy population)
  - present:      full density and thickness, no gap

Usage:
    python analysis/defect_detection/analyze_lattice_json.py --json "path/to/..._lattice_graph.json"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def robust_stats(x):
    """Median and MAD-based sigma (robust to the defect outliers themselves)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma = 1.4826 * mad if mad > 0 else (x.std() or 1.0)
    return med, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--k", type=float, default=3.0,
                    help="Outlier strength: flag struts below median - k*sigma (default 3).")
    args = ap.parse_args()
    jpath = Path(args.json).expanduser().resolve()
    data = json.loads(jpath.read_text())
    S = data["struts"]
    n = len(S)

    dens = np.array([s["mean_density"] for s in S])
    thick = np.array([s["mean_thickness"] for s in S])
    gap = np.array([s["gap_frac"] for s in S])

    # Healthy population = struts with no meaningful gap; get robust density/thickness.
    healthy = gap < 0.15
    d_med, d_sig = robust_stats(dens[healthy])
    t_med, t_sig = robust_stats(thick[healthy])
    d_cut = d_med - args.k * d_sig
    t_cut = t_med - args.k * t_sig
    print(f"healthy density  median {d_med:.0f} sigma {d_sig:.0f} -> thin if < {d_cut:.0f}")
    print(f"healthy thickness median {t_med:.2f} sigma {t_sig:.2f} -> thin if < {t_cut:.2f}")

    counts = {"present": 0, "thin": 0, "disconnected": 0, "missing": 0}
    for s in S:
        if s["gap_frac"] >= 0.6:
            v = "missing"
        elif s["gap_frac"] >= 0.15:
            v = "disconnected"
        elif s["mean_density"] < d_cut or s["mean_thickness"] < t_cut:
            v = "thin"
        else:
            v = "present"
        s["verdict"] = v
        counts[v] += 1

    print("\n=== DEFECTS (statistical, from density/thickness outliers) ===")
    for k in ("present", "thin", "disconnected", "missing"):
        print(f"  {k:12s}: {counts[k]:6d}  ({100*counts[k]/n:.2f}%)")
    defect_pct = 100*(n-counts['present'])/n
    print(f"  total defective: {n-counts['present']} ({defect_pct:.2f}%)")

    # Save classified JSON + density-distribution figure.
    out_json = jpath.with_name(jpath.stem + "_classified.json")
    out_json.write_text(json.dumps(data), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(dens, bins=60, color="steelblue")
    ax[0].axvline(d_cut, color="red", ls="--", label=f"thin cutoff ({d_cut:.0f})")
    ax[0].axvline(d_med, color="green", ls="--", label=f"healthy median ({d_med:.0f})")
    ax[0].set_xlabel("mean strut density (CT intensity)"); ax[0].set_ylabel("count")
    ax[0].set_title("Per-strut density distribution"); ax[0].legend()

    ax[1].scatter(thick, dens, s=4, c=["red" if s["verdict"] == "missing"
                                       else "orange" if s["verdict"] == "disconnected"
                                       else "gold" if s["verdict"] == "thin" else "lightgray"
                                       for s in S])
    ax[1].axvline(t_cut, color="red", ls=":"); ax[1].axhline(d_cut, color="red", ls=":")
    ax[1].set_xlabel("thickness"); ax[1].set_ylabel("density")
    ax[1].set_title("density vs thickness (defects colored)")
    fig.tight_layout()
    fig_path = jpath.with_name(jpath.stem + "_density_analysis.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_json.name}, {fig_path.name}")


if __name__ == "__main__":
    main()
