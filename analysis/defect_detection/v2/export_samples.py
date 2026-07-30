"""Export individual validation samples: 50 per category, one PNG per strut,
in manual_validation/{missing,disconnected,thin,bent}/.

Every panel shows the classifier's OWN stored measurements (title + green/red
profile bar).  Views follow what manual review proved works:
  - missing:      thin slab on the chord (absence shows honestly)
  - disconnected: thin slab AND thick metal-centred slab side by side — a
                  detached-in-place strut is visible in the thick view while
                  the crack evidence stays in the bar/numbers
  - thin:         slab centred on the metal path, path drawn
  - bent:         resampled in the strut's own bending plane, skeleton path
Selection: interior-context first (disconnected tops up from boundary, clearly
labelled), seeded RNG for reproducibility.
"""
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
from detect_v2 import OUT_JSON  # noqa: E402
from validate_v2 import bend_plane_view, draw_path  # noqa: E402

BASE = Path(__file__).resolve().parent / "manual_validation"
N = 50
SEED = 7
PAD = 16
VMIN, VMAX = 18000, 52000
COLOR = {"missing": "#ef4444", "disconnected": "#f97316",
         "thin": "#eab308", "bent": "#ec4899"}


def bar_draw(bar, prof, r_search):
    arr = np.array([1 if ch == "#" else 0 for ch in prof])
    bar.imshow(arr[None, :], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    bar.set_xticks([]); bar.set_yticks([])
    bar.set_xlabel(f"metal within R={r_search:.0f} vox of the line "
                   "(green = found, red = gap)", fontsize=8)


def crop(ax, raw, fax, lo, hi, c, w, shape):
    sl = [slice(lo[a], hi[a]) for a in range(3)]
    sl[fax] = slice(max(c - w, 0), min(c + w + 1, shape[fax]))
    ax.imshow(raw[tuple(sl)].max(axis=fax).astype(float),
              cmap="gray", vmin=VMIN, vmax=VMAX)


def head(r):
    st = f" ({r['subtype']})" if r.get("subtype") else ""
    ctx = r["context"].upper() if r["context"] == "boundary" else r["context"]
    return (f"strut {r['i']}  ·  {r['verdict'].upper()}{st}  ·  {ctx}\n"
            f"nodes {r['node0'][0].upper()}/{r['node1'][0].upper()}  ·  "
            f"{np.round(r['p0']).astype(int)} -> {np.round(r['p1']).astype(int)} [x,y,z]")


def main():
    data = json.load(open(OUT_JSON))
    S = data["struts"]
    thr = data["meta"]["thresholds"]
    raw = tifffile.imread(RAW)
    mask = tifffile.imread(MASK) > 0
    shape = np.array(raw.shape)
    rng = np.random.RandomState(SEED)

    def pick(cat):
        cand = [r for r in S if r["verdict"] == cat]
        if cat == "missing":
            cand = [r for r in cand if r.get("subtype") != "void"]
        inner = [r for r in cand if r["context"] == "interior"]
        outer = [r for r in cand if r["context"] != "interior"]
        rng.shuffle(inner); rng.shuffle(outer)
        sel = inner[:N]
        if len(sel) < N:
            sel += outer[:N - len(sel)]
        return sel

    for cat in ["missing", "disconnected", "thin", "bent"]:
        out = BASE / cat
        out.mkdir(parents=True, exist_ok=True)
        sel = pick(cat)
        col = COLOR[cat]
        for k, r in enumerate(sel, 1):
            pa = np.array(r["p0"][::-1], float)
            pb = np.array(r["p1"][::-1], float)
            d = pb - pa
            fax = int(np.argmin(np.abs(d)))
            a2 = [a for a in range(3) if a != fax]
            lo = np.maximum(np.minimum(pa, pb).astype(int) - PAD, 0)
            hi = np.minimum(np.maximum(pa, pb).astype(int) + PAD + 1, shape)
            e0 = [pa[a] - lo[a] for a in a2]; e1 = [pb[a] - lo[a] for a in a2]

            if cat == "missing":
                fig, (ax, bar) = plt.subplots(
                    2, 1, figsize=(5.4, 6.6), facecolor="white",
                    gridspec_kw={"height_ratios": [11, 1]})
                crop(ax, raw, fax, lo, hi, int(round((pa[fax] + pb[fax]) / 2)), 3, shape)
                ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=col, lw=2.2, alpha=0.9)
                ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=70,
                           edgecolors="white", linewidths=1.2, zorder=5)
                info = (f"gap {r['gap_um']:.0f} um · stub {r.get('stub_um', 0):.0f} um · "
                        f"graph hops {r.get('hops', '?')}")
                fname = f"{k:02d}_i{r['i']}_gap{r['gap_um']:.0f}um.png"
            elif cat == "disconnected":
                fig, axes = plt.subplots(
                    2, 2, figsize=(10.4, 6.9), facecolor="white",
                    gridspec_kw={"height_ratios": [11, 1]})
                ax1, ax2 = axes[0]
                path = draw_path(pa, pb, mask, shape)
                crop(ax1, raw, fax, lo, hi, int(round((pa[fax] + pb[fax]) / 2)), 3, shape)
                ax1.set_title("thin slab", fontsize=9)
                fc = int(round(float(np.nanmedian(path[:, fax]))))
                crop(ax2, raw, fax, lo, hi, fc, 7, shape)
                ax2.set_title("thick slab centred on the metal", fontsize=9)
                for ax in (ax1, ax2):
                    ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=col, lw=2.0, alpha=0.85)
                    ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=65,
                               edgecolors="white", linewidths=1.2, zorder=5)
                bar = axes[1][0]
                axes[1][1].axis("off")
                ph = " · PHANTOM EDGE (detached at a joint)" if r.get("flag_phantom_edge") else ""
                info = f"break {r['gap_um']:.0f} um · surviving piece {r['stub_um']:.0f} um{ph}"
                fname = f"{k:02d}_i{r['i']}_break{r['gap_um']:.0f}um_stub{r['stub_um']:.0f}um.png"
            elif cat == "thin":
                fig, (ax, bar) = plt.subplots(
                    2, 1, figsize=(5.4, 6.6), facecolor="white",
                    gridspec_kw={"height_ratios": [11, 1]})
                path = draw_path(pa, pb, mask, shape)
                fc = int(round(float(np.nanmedian(path[:, fax]))))
                crop(ax, raw, fax, lo, hi, fc, 5, shape)
                ax.plot([e0[1], e1[1]], [e0[0], e1[0]], ":", c="white", lw=1.1, alpha=0.5)
                ax.plot(path[:, a2[1]] - lo[a2[1]], path[:, a2[0]] - lo[a2[0]],
                        "-", c=col, lw=2.2, alpha=0.95)
                ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=70,
                           edgecolors="white", linewidths=1.2, zorder=5)
                kind = ("hairline" if r["dia_um"] is not None
                        and r["dia_um"] < thr["thin_cut_um"] else "necked")
                info = (f"dia median {r['dia_um']:.0f} um · thinnest {r['dia_min_um']:.0f} um "
                        f"({kind}; cut {thr['thin_cut_um']:.0f}, healthy {thr['dia_median_um']:.0f})")
                fname = f"{k:02d}_i{r['i']}_min{r['dia_min_um']:.0f}um_{kind}.png"
            else:  # bent
                fig, (ax, bar) = plt.subplots(
                    2, 1, figsize=(7.6, 5.4), facecolor="white",
                    gridspec_kw={"height_ratios": [8, 1]})
                if "path" in r:
                    path = np.array(r["path"], float)[:, ::-1]
                else:
                    path = draw_path(pa, pb, mask, shape)
                img, px, py = bend_plane_view(raw, pa, pb, path)
                ax.imshow(img, cmap="gray", vmin=VMIN, vmax=VMAX)
                ax.plot([px[0], px[-1]], [py[0], py[-1]], ":", c="white", lw=1.3, alpha=0.6)
                ax.plot(px, py, "-", c=col, lw=2.4, alpha=0.95)
                ax.scatter([px[0], px[-1]], [py[0], py[-1]], c=col, s=60,
                           edgecolors="white", linewidths=1.1, zorder=5)
                info = f"bow {r['bow_um']:.0f} um  (bent > {thr['bent_bow_um']:.0f} um)"
                fname = f"{k:02d}_i{r['i']}_bow{r['bow_um']:.0f}um.png"

            ax_t = fig.axes[0]
            ax_t.set_title(head(r) + "\n" + info, fontsize=9.5,
                           color=col, fontweight="bold")
            for a_ in fig.axes[:-1] if cat != "disconnected" else fig.axes[:2]:
                a_.axis("off")
            bar_draw(bar, r["prof"], r["prof_r"])
            fig.tight_layout()
            fig.savefig(out / fname, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        n_int = sum(1 for r in sel if r["context"] == "interior")
        print(f"{cat}: wrote {len(sel)} panels ({n_int} interior, "
              f"{len(sel) - n_int} boundary) -> {out}", flush=True)


if __name__ == "__main__":
    main()
