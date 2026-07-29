"""Re-examine the 4 disputed struts (#2 #7 #9 #10): drawn with a THICK slab
centred on the actual metal path, so a displaced-but-intact strut cannot hide
outside the display slice the way it could in the thin default view."""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, MASK  # noqa: E402
from detect_v2 import centroid_path  # noqa: E402

OUT = Path(__file__).resolve().parent / "panels/recheck4.png"
CASES = [("#2", [293, 403, 220], [331, 407, 183]),
         ("#7", [500, 207, 531], [531, 248, 540]),
         ("#9", [346, 211, 375], [335, 171, 344]),
         ("#10", [541, 486, 629], [571, 484, 580])]
PAD = 14


def main():
    raw = tifffile.imread(RAW)
    mask = tifffile.imread(MASK) > 0
    shape = np.array(raw.shape)
    fig, axes = plt.subplots(2, len(CASES), figsize=(4.8 * len(CASES), 9.6),
                             facecolor="white")
    for col, (nm, t0, t1) in enumerate(CASES):
        pa = np.array(t0[::-1], float)
        pb = np.array(t1[::-1], float)
        d = pb - pa
        cent, ts = centroid_path(pa, pb, mask, shape)
        ok = ~np.isnan(cent[:, 0])
        idx = np.arange(len(cent))
        for a in range(3):
            cent[:, a] = np.interp(idx, idx[ok], cent[ok, a])
        path = ndi.uniform_filter1d(cent, size=5, axis=0, mode="nearest")
        fax = int(np.argmin(np.abs(d)))
        a2 = [a for a in range(3) if a != fax]
        lo = np.maximum(np.minimum(pa, pb).astype(int) - PAD, 0)
        hi = np.minimum(np.maximum(pa, pb).astype(int) + PAD + 1, shape)
        # slab centred on the METAL path, generous half-width
        fc = int(round(np.nanmedian(path[:, fax])))
        for row, (slab, ttl) in enumerate([(2, "thin slab (like the old sheet)"),
                                           (7, "THICK slab centred on the metal")]):
            ax = axes[row][col]
            sl = [slice(lo[a], hi[a]) for a in range(3)]
            sl[fax] = slice(max(fc - slab, 0), min(fc + slab + 1, shape[fax]))
            ax.imshow(raw[tuple(sl)].max(axis=fax).astype(float),
                      cmap="gray", vmin=18000, vmax=52000)
            px = path[:, a2[1]] - lo[a2[1]]
            py = path[:, a2[0]] - lo[a2[0]]
            ax.plot(px, py, "-", c="#22d3ee", lw=2.0, alpha=0.9)
            ax.scatter([pa[a2[1]] - lo[a2[1]], pb[a2[1]] - lo[a2[1]]],
                       [pa[a2[0]] - lo[a2[0]], pb[a2[0]] - lo[a2[0]]],
                       c="#22d3ee", s=60, edgecolors="white", linewidths=1.1, zorder=5)
            if row == 0:
                ax.set_title(f"{nm}  {t0}->{t1}\n{ttl}", fontsize=10, fontweight="bold")
            else:
                ax.set_title(ttl, fontsize=10, fontweight="bold")
            ax.axis("off")
    fig.suptitle("RECHECK — the 4 disputed struts, now PRESENT\n"
                 "top: the thin display slab the old sheet used · bottom: thick slab "
                 "centred on the metal (a displaced strut cannot hide here)",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight", facecolor="white")
    print("saved", OUT)


if __name__ == "__main__":
    main()
