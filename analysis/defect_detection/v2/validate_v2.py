"""v2 validation renderer — panels drawn FROM THE CLASSIFIER'S OWN RECORDS.

Every number shown (gap, stub, diameter, bow) and every green/red profile bar
comes straight out of strut_classification_v2.json — the exact evidence the
verdict was decided on.  Geometry overlays (centreline paths) are recomputed
with the SAME functions imported from detect_v2, so code cannot diverge.

Usage:
    python validate_v2.py missing 20
    python validate_v2.py disconnected 5
    python validate_v2.py thin 5
    python validate_v2.py bent 5
    python validate_v2.py missing 10 --subtype void      (look at void cases)
    python validate_v2.py missing 10 --context any       (include boundary)

Defaults: --context interior, and for missing --subtype dropped, because those
are the individually-defensible cases; boundary/void cases are real absences
but need their own reading (design extends past the printed part / node never
printed), so they are opt-in and labelled when shown.
"""

import argparse
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
from config import RAW  # noqa: E402
from detect_v2 import (OUT_JSON, UM, centroid_path, _perp,  # noqa: E402
                       MID_LO, MID_HI)

V2 = Path(__file__).resolve().parent
PANELS = V2 / "panels"
COLS = 5
PAD = 16
SLAB = 3
VMIN, VMAX = 18000, 52000
COLOR = {"missing": "#ef4444", "disconnected": "#f97316",
         "thin": "#eab308", "bent": "#ec4899", "present": "#22c55e"}


def draw_path(pa, pb, mask, shape):
    """display centreline: same centroid_path as the classifier, then NaN-filled
    and lightly smoothed FOR DRAWING ONLY (measurements stay untouched)."""
    cent, _ = centroid_path(pa, pb, mask, shape)
    ok = ~np.isnan(cent[:, 0])
    if ok.sum() < 3:
        return np.stack([pa, pb])
    idx = np.arange(len(cent))
    for a in range(3):
        cent[:, a] = np.interp(idx, idx[ok], cent[ok, a])
    cent = ndi.uniform_filter1d(cent, size=5, axis=0, mode="nearest")
    return np.vstack([pa[None, :], cent, pb[None, :]])


def bend_plane_view(raw, pa, pb, path, pad=10, half=15, thick=3):
    """resample the volume in the strut's own (axis, bow) plane."""
    d = pb - pa
    L = float(np.linalg.norm(d))
    u = d / (L + 1e-9)
    dev = path - (pa[None, :] + ((path - pa) @ u)[:, None] * u[None, :])
    k = int(np.argmax(np.linalg.norm(dev, axis=1)))
    w = dev[k]
    nw = np.linalg.norm(w)
    w = w / nw if nw > 1e-6 else _perp(u)[0]
    v = np.cross(u, w)
    ii = np.arange(-pad, L + pad + 1.0)
    jj = np.arange(-half, half + 1.0)
    kk = np.arange(-thick, thick + 1.0)
    G = (pa[None, None, None, :]
         + ii[:, None, None, None] * u
         + jj[None, :, None, None] * w
         + kk[None, None, :, None] * v)
    vals = ndi.map_coordinates(raw, G.reshape(-1, 3).T, order=1,
                               mode="constant", cval=0.0)
    img = vals.reshape(len(ii), len(jj), len(kk)).max(axis=2).T
    rel = path - pa
    return img, rel @ u + pad, rel @ w + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("category", choices=["missing", "disconnected", "thin", "bent", "present"])
    ap.add_argument("count", type=int, nargs="?", default=5)
    ap.add_argument("--context", default="interior", choices=["interior", "boundary", "any"])
    ap.add_argument("--subtype", default=None,
                    choices=[None, "dropped", "node_lost", "void", "not_void", "any"])
    ap.add_argument("--orient", default="any", choices=["any", "flat", "vertical"],
                    help="octet strut family: flat = in-plane (dz~0), "
                         "vertical = z-climbing")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    data = json.load(open(OUT_JSON))
    S = data["struts"]
    thr = data["meta"]["thresholds"]

    sub = args.subtype
    if args.category == "missing" and sub is None:
        # dropped + node_lost by default: a strut can be missing with only one
        # node present (the node was lost too) — exclude only both-nodes-absent
        # voids, which need their own reading.
        sub = "not_void"

    cand = [r for r in S if r["verdict"] == args.category]
    n_all = len(cand)
    if args.context != "any":
        cand = [r for r in cand if r["context"] == args.context]
    if sub == "not_void":
        cand = [r for r in cand if r.get("subtype") != "void"]
    elif sub not in (None, "any"):
        cand = [r for r in cand if r.get("subtype") == sub]
    if args.orient != "any":
        def dz(r):
            return abs(r["p1"][2] - r["p0"][2])       # z-component of the strut
        cand = [r for r in cand
                if (dz(r) > 20) == (args.orient == "vertical")]
    print(f"{args.category}: {n_all} total -> {len(cand)} after filters "
          f"(context={args.context}, subtype={sub})")
    if not cand:
        print("nothing to draw — relax --context/--subtype")
        return

    rng = np.random.RandomState(args.seed)
    pick = [cand[i] for i in rng.choice(len(cand), min(args.count, len(cand)), replace=False)]

    need_mask = args.category in ("thin", "bent")
    raw = tifffile.imread(RAW)
    mask = None
    if need_mask:
        from config import MASK as MASK_P
        mask = tifffile.imread(MASK_P) > 0
    shape = np.array(raw.shape)

    n = len(pick)
    rows = (n + COLS - 1) // COLS
    ncols = min(COLS, n)
    fig, axes = plt.subplots(rows * 2, ncols, figsize=(3.6 * ncols, 4.6 * rows),
                             facecolor="white", squeeze=False,
                             gridspec_kw={"height_ratios": [10, 1] * rows})
    col = COLOR[args.category]
    for i, r in enumerate(pick):
        rr, cc = divmod(i, ncols)
        ax = axes[rr * 2][cc]; bar = axes[rr * 2 + 1][cc]
        pa = np.array(r["p0"][::-1], float)
        pb = np.array(r["p1"][::-1], float)
        d = pb - pa

        if args.category == "bent":
            if "path" in r:
                # the exact skeleton polyline the bow was measured on
                path = np.array(r["path"], float)[:, ::-1]      # xyz -> zyx
            else:
                path = draw_path(pa, pb, mask, shape)
            img, px, py = bend_plane_view(raw, pa, pb, path)
            ax.imshow(img, cmap="gray", vmin=VMIN, vmax=VMAX)
            ax.plot([px[0], px[-1]], [py[0], py[-1]], ":", c="white", lw=1.3, alpha=0.6)
            ax.plot(px, py, "-", c=col, lw=2.4, alpha=0.95)
            ax.scatter([px[0], px[-1]], [py[0], py[-1]], c=col, s=60,
                       edgecolors="white", linewidths=1.1, zorder=5)
        else:
            fax = int(np.argmin(np.abs(d)))
            a2 = [a for a in range(3) if a != fax]
            fc = int(round((pa[fax] + pb[fax]) / 2))
            lo = np.maximum(np.minimum(pa, pb).astype(int) - PAD, 0)
            hi = np.minimum(np.maximum(pa, pb).astype(int) + PAD + 1, shape)
            sl = [slice(lo[a], hi[a]) for a in range(3)]
            sl[fax] = slice(max(fc - SLAB, 0), min(fc + SLAB + 1, shape[fax]))
            ax.imshow(raw[tuple(sl)].max(axis=fax).astype(float),
                      cmap="gray", vmin=VMIN, vmax=VMAX)
            e0 = [pa[a] - lo[a] for a in a2]; e1 = [pb[a] - lo[a] for a in a2]
            if args.category == "thin":
                path = draw_path(pa, pb, mask, shape)
                ax.plot(path[:, a2[1]] - lo[a2[1]], path[:, a2[0]] - lo[a2[0]],
                        "-", c=col, lw=2.2, alpha=0.95)
                ax.plot([e0[1], e1[1]], [e0[0], e1[0]], ":", c="white", lw=1.1, alpha=0.5)
            else:
                ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=col, lw=2.0, alpha=0.9)
            ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=55,
                       edgecolors="white", linewidths=1.1, zorder=5)

        # ---- title straight from the record ----
        nodes = f"{r['node0'][0].upper()}/{r['node1'][0].upper()}"
        if args.category == "missing":
            info = f"gap {r['gap_um']:.0f}um · stub {r.get('stub_um', 0):.0f}um · hops {r.get('hops', '?')}"
        elif args.category == "disconnected":
            info = f"break {r['gap_um']:.0f}um · stub {r['stub_um']:.0f}um"
        elif args.category == "thin":
            info = (f"dia med {r['dia_um']:.0f} · thinnest {r['dia_min_um']:.0f}um "
                    f"(cut {thr['thin_cut_um']:.0f}, pop {thr['dia_median_um']:.0f})")
        elif args.category == "bent":
            info = f"bow {r['bow_um']:.0f}um (bent > {thr['bent_bow_um']:.0f})"
        else:
            info = f"gap {r.get('gap_um', 0):.0f}um · bow {r.get('bow_um', 0):.0f} · dia {r.get('dia_um', 0)}"
        st = f" ({r['subtype']})" if r.get("subtype") else ""
        ax.set_title(f"#{i+1} {r['verdict'].upper()}{st} · {r['context']}\n{info}\n"
                     f"nodes {nodes} · {np.round(pa[::-1]).astype(int)}->{np.round(pb[::-1]).astype(int)}",
                     fontsize=8, color=col, fontweight="bold")
        ax.axis("off")

        prof = np.array([1 if ch == "#" else 0 for ch in r["prof"]])
        bar.imshow(prof[None, :], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        bar.set_xticks([]); bar.set_yticks([])
        bar.set_xlabel(f"metal profile (search R={r['prof_r']:.0f} vox)", fontsize=7)

    for j in range(n, rows * ncols):
        rr, cc = divmod(j, ncols)
        axes[rr * 2][cc].axis("off"); axes[rr * 2 + 1][cc].axis("off")

    fig.suptitle(f"v2 VALIDATION — {n} {args.category.upper()} struts "
                 f"(context={args.context}"
                 + (f", subtype={sub}" if sub not in (None, 'any') else "") + ")\n"
                 "every number and every green/red bar is the classifier's own stored "
                 "measurement · nodes J=junction P=partial A=absent",
                 fontsize=12, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    PANELS.mkdir(exist_ok=True)
    out = PANELS / f"v2_{args.category}_{n}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
