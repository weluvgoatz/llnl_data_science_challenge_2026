"""Capture up to 5 representative sample images per defect class from v2's
OWN classification output, plus a combined atlas (one row per class) and a
manifest.json mapping class -> images + metrics, for the app's "show me a
sample of each defect class" chat feature.

Reads EXCLUSIVELY from v2 outputs: detect_v2.OUT_JSON
(strut_classification_v2.json) and detect_v2.MASK_P/RAW_P -- deliberately NOT
config.MASK, which is the v1-named, unsuffixed path. detect_v2.py's own
MASK_P is namespaced with a _v2 suffix specifically so nothing here can
silently pick up a stale v1-written mask (see detect_v2.py's comment on
why the two are not interchangeable).

Classes: missing, disconnected, thin, bent -- v2's four defect verdicts.
"present" is intentionally excluded, matching export_samples.py's and
viz_defect_atlas.py's existing convention (a gallery of healthy struts isn't
what "show me the defects" is asking for).

Selection (not just the first N): interior-context struts first (boundary
ones are edge-of-part artifacts per START_HERE.md, not individual defects --
only topped up with boundary struts if fewer than N interior examples
exist), ranked by each class's own decision metric (the number the
classifier itself thresholds on, never an invented severity score), then N
picked at roughly evenly-spaced percentiles of that ranking so the gallery
shows a spread of severity rather than N near-duplicates of the single
worst case.

Rendering reuses export_samples.py's proven per-class view choices (that
file is left untouched -- its own 50-per-category manual_validation workflow
is a separate concern) and validate_v2.py's draw_path/bend_plane_view.

Output (shared scratch; the caller -- web/backend/app/agent_tools.py --
serializes access the same way it already does for detect_v2.py/export_3d.py,
then moves the result into the job's own directory and clears this one, so a
stale run is never shown for a different job):
    v2/defect_samples/<class>/<class>_<strut_id>.png
    v2/defect_samples/atlas.png
    v2/defect_samples/manifest.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # same pattern as export_samples.py
from config import RAW as RAW_P  # noqa: E402 -- same file for v1 and v2, not model-specific
from detect_v2 import MASK_P, OUT_JSON  # noqa: E402 -- the v2-namespaced mask, never config.MASK
from validate_v2 import bend_plane_view, draw_path  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "defect_samples"
N = 5
PAD = 16
VMIN, VMAX = 18000, 52000
COLOR = {"missing": "#ef4444", "disconnected": "#f97316", "thin": "#eab308", "bent": "#ec4899"}
CLASSES = ["missing", "disconnected", "thin", "bent"]


def crop(ax, raw, fax, lo, hi, c, w, shape):
    sl = [slice(lo[a], hi[a]) for a in range(3)]
    sl[fax] = slice(max(c - w, 0), min(c + w + 1, shape[fax]))
    ax.imshow(raw[tuple(sl)].max(axis=fax).astype(float), cmap="gray", vmin=VMIN, vmax=VMAX)


def bar_draw(bar, prof, r_search):
    arr = np.array([1 if ch == "#" else 0 for ch in prof])
    bar.imshow(arr[None, :], aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    bar.set_xticks([])
    bar.set_yticks([])
    bar.set_xlabel(f"metal within R={r_search:.0f} vox of the line (green = found, red = gap)", fontsize=8)


def head(r):
    st = f" ({r['subtype']})" if r.get("subtype") else ""
    ctx = r["context"].upper() if r["context"] == "boundary" else r["context"]
    return (
        f"strut {r['i']}  ·  {r['verdict'].upper()}{st}  ·  {ctx}\n"
        f"nodes {r['node0'][0].upper()}/{r['node1'][0].upper()}  ·  "
        f"{np.round(r['p0']).astype(int)} -> {np.round(r['p1']).astype(int)} [x,y,z]"
    )


def _severity_rank(cat, structs):
    """Descending severity order using each class's own decision metric."""
    if cat == "missing":
        return sorted(structs, key=lambda r: (r.get("hops") or 0, r.get("gap_um") or 0), reverse=True)
    if cat == "disconnected":
        return sorted(structs, key=lambda r: r.get("gap_um") or 0, reverse=True)
    if cat == "thin":
        # ascending: smaller dia_min_um = thinner = more severe
        return sorted(structs, key=lambda r: r["dia_min_um"] if r.get("dia_min_um") is not None else 1e9)
    if cat == "bent":
        return sorted(structs, key=lambda r: r.get("bow_um") or 0, reverse=True)
    raise ValueError(cat)


def pick_spread(cat, all_structs, n=N):
    cand = [r for r in all_structs if r["verdict"] == cat]
    if cat == "missing":
        cand = [r for r in cand if r.get("subtype") != "void"]  # no real geometry to crop
    interior = [r for r in cand if r["context"] == "interior"]
    boundary = [r for r in cand if r["context"] != "interior"]
    pool = _severity_rank(cat, interior) if len(interior) >= n else (
        _severity_rank(cat, interior) + _severity_rank(cat, boundary)
    )
    total = len(cand)
    if len(pool) <= n:
        return pool, total
    positions = np.linspace(0.1, 0.9, n)
    idxs = sorted({int(round(p * (len(pool) - 1))) for p in positions})
    i = 0
    while len(idxs) < n and i < len(pool):
        if i not in idxs:
            idxs.append(i)
        i += 1
    idxs = sorted(idxs)[:n]
    return [pool[i] for i in idxs], total


def render_one(r, raw, mask, shape, thr):
    cat = r["verdict"]
    col = COLOR[cat]
    pa = np.array(r["p0"][::-1], float)
    pb = np.array(r["p1"][::-1], float)
    d = pb - pa
    fax = int(np.argmin(np.abs(d)))
    a2 = [a for a in range(3) if a != fax]
    lo = np.maximum(np.minimum(pa, pb).astype(int) - PAD, 0)
    hi = np.minimum(np.maximum(pa, pb).astype(int) + PAD + 1, shape)
    e0 = [pa[a] - lo[a] for a in a2]
    e1 = [pb[a] - lo[a] for a in a2]

    if cat == "missing":
        fig, (ax, bar) = plt.subplots(2, 1, figsize=(5.4, 6.6), facecolor="white", gridspec_kw={"height_ratios": [11, 1]})
        crop(ax, raw, fax, lo, hi, int(round((pa[fax] + pb[fax]) / 2)), 3, shape)
        ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=col, lw=2.2, alpha=0.9)
        ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=70, edgecolors="white", linewidths=1.2, zorder=5)
        info = f"gap {r['gap_um']:.0f} um · stub {r.get('stub_um', 0):.0f} um · graph hops {r.get('hops', '?')}"
        axes_to_hide = fig.axes[:-1]
    elif cat == "disconnected":
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.9), facecolor="white", gridspec_kw={"height_ratios": [11, 1]})
        ax1, ax2 = axes[0]
        path = draw_path(pa, pb, mask, shape)
        crop(ax1, raw, fax, lo, hi, int(round((pa[fax] + pb[fax]) / 2)), 3, shape)
        ax1.set_title("thin slab", fontsize=9)
        fc = int(round(float(np.nanmedian(path[:, fax]))))
        crop(ax2, raw, fax, lo, hi, fc, 7, shape)
        ax2.set_title("thick slab centred on the metal", fontsize=9)
        for ax in (ax1, ax2):
            ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=col, lw=2.0, alpha=0.85)
            ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=65, edgecolors="white", linewidths=1.2, zorder=5)
        bar = axes[1][0]
        axes[1][1].axis("off")
        ph = " · PHANTOM EDGE (detached at a joint)" if r.get("flag_phantom_edge") else ""
        info = f"break {r['gap_um']:.0f} um · surviving piece {r['stub_um']:.0f} um{ph}"
        axes_to_hide = fig.axes[:2]
    elif cat == "thin":
        fig, (ax, bar) = plt.subplots(2, 1, figsize=(5.4, 6.6), facecolor="white", gridspec_kw={"height_ratios": [11, 1]})
        path = draw_path(pa, pb, mask, shape)
        fc = int(round(float(np.nanmedian(path[:, fax]))))
        crop(ax, raw, fax, lo, hi, fc, 5, shape)
        ax.plot([e0[1], e1[1]], [e0[0], e1[0]], ":", c="white", lw=1.1, alpha=0.5)
        ax.plot(path[:, a2[1]] - lo[a2[1]], path[:, a2[0]] - lo[a2[0]], "-", c=col, lw=2.2, alpha=0.95)
        ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=70, edgecolors="white", linewidths=1.2, zorder=5)
        kind = "hairline" if r["dia_um"] is not None and r["dia_um"] < thr["thin_cut_um"] else "necked"
        info = (
            f"dia median {r['dia_um']:.0f} um · thinnest {r['dia_min_um']:.0f} um "
            f"({kind}; cut {thr['thin_cut_um']:.0f}, healthy {thr['dia_median_um']:.0f})"
        )
        axes_to_hide = fig.axes[:-1]
    else:  # bent
        fig, (ax, bar) = plt.subplots(2, 1, figsize=(7.6, 5.4), facecolor="white", gridspec_kw={"height_ratios": [8, 1]})
        path = np.array(r["path"], float)[:, ::-1] if "path" in r else draw_path(pa, pb, mask, shape)
        img, px, py = bend_plane_view(raw, pa, pb, path)
        ax.imshow(img, cmap="gray", vmin=VMIN, vmax=VMAX)
        ax.plot([px[0], px[-1]], [py[0], py[-1]], ":", c="white", lw=1.3, alpha=0.6)
        ax.plot(px, py, "-", c=col, lw=2.4, alpha=0.95)
        ax.scatter([px[0], px[-1]], [py[0], py[-1]], c=col, s=60, edgecolors="white", linewidths=1.1, zorder=5)
        info = f"bow {r['bow_um']:.0f} um  (bent > {thr['bent_bow_um']:.0f} um)"
        axes_to_hide = fig.axes[:-1]

    fig.axes[0].set_title(head(r) + "\n" + info, fontsize=9.5, color=col, fontweight="bold")
    for a_ in axes_to_hide:
        a_.axis("off")
    bar_draw(bar, r["prof"], r["prof_r"])
    fig.tight_layout()
    return fig


def render_atlas_cell(ax, r, raw, shape, col):
    """Simple single-crop cell for the combined atlas (one row per class) --
    same dashed-line + white-edged-dot convention as viz_defect_atlas.py."""
    pa = np.array(r["p0"][::-1], float)
    pb = np.array(r["p1"][::-1], float)
    d = pb - pa
    fax = int(np.argmin(np.abs(d)))
    a2 = [a for a in range(3) if a != fax]
    lo = np.maximum(np.minimum(pa, pb).astype(int) - PAD, 0)
    hi = np.minimum(np.maximum(pa, pb).astype(int) + PAD + 1, shape)
    crop(ax, raw, fax, lo, hi, int(round((pa[fax] + pb[fax]) / 2)), 5, shape)
    e0 = [pa[a] - lo[a] for a in a2]
    e1 = [pb[a] - lo[a] for a in a2]
    ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=col, lw=2.0, alpha=0.85)
    ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=col, s=55, edgecolors="white", linewidths=1.0, zorder=5)
    ax.set_title(f"strut {r['i']}", fontsize=8, color=col)
    ax.axis("off")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    structs = data["struts"]
    thr = data["meta"]["thresholds"]
    raw = tifffile.imread(RAW_P)
    mask = tifffile.imread(MASK_P) > 0
    shape = np.array(raw.shape)

    manifest = {
        "source": str(OUT_JSON.name),
        "selection_criterion": (
            "interior-context struts preferred (boundary ones are edge-of-part artifacts, "
            "topped up only if fewer than N interior examples exist); ranked by each class's "
            "own decision metric (hops+gap_um missing, gap_um disconnected, dia_min_um "
            "ascending thin, bow_um bent); N picked at ~10/30/50/70/90th percentile of that "
            "ranking for a severity spread, not the single worst-case pileup"
        ),
        "classes": {},
    }

    for cat in CLASSES:
        out_dir = OUTDIR / cat
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob(f"{cat}_*.png"):
            stale.unlink()
        picks, total = pick_spread(cat, structs, N)
        images = []
        for r in picks:
            fig = render_one(r, raw, mask, shape, thr)
            fname = f"{cat}_{r['i']}.png"
            fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            metrics = {
                k: r[k]
                for k in ("gap_um", "stub_um", "hops", "subtype", "dia_um", "dia_min_um", "bow_um", "context", "len_um")
                if k in r and r[k] is not None
            }
            images.append({"strut_id": r["i"], "path": f"{cat}/{fname}", "metrics": metrics})
        manifest["classes"][cat] = {"count_total": total, "count_sampled": len(picks), "images": images}
        n_int = sum(1 for r in picks if r["context"] == "interior")
        print(f"{cat}: captured {len(picks)}/{total} ({n_int} interior, {len(picks) - n_int} boundary)", flush=True)

    # combined atlas: one row per class, up to N columns, reusing the exact
    # same picks (and therefore the exact same struts) as the per-class galleries.
    rows = [cat for cat in CLASSES if manifest["classes"][cat]["count_sampled"] > 0]
    ncol = max(manifest["classes"][cat]["count_sampled"] for cat in rows) if rows else 0
    if rows and ncol:
        fig, axes = plt.subplots(len(rows), ncol, figsize=(3.6 * ncol, 3.8 * len(rows)), facecolor="white", squeeze=False)
        by_id = {r["i"]: r for r in structs}
        for ri, cat in enumerate(rows):
            col = COLOR[cat]
            img_entries = manifest["classes"][cat]["images"]
            for ci in range(ncol):
                ax = axes[ri][ci]
                if ci < len(img_entries):
                    r = by_id[img_entries[ci]["strut_id"]]
                    render_atlas_cell(ax, r, raw, shape, col)
                else:
                    ax.axis("off")
            axes[ri][0].set_ylabel(cat.upper(), fontsize=11, color=col, fontweight="bold")
        fig.suptitle(
            "DEFECT SAMPLE ATLAS (v2) — zoomed raw CT per class · dashed line = the strut, dots = its joints",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUTDIR / "atlas.png", dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        manifest["atlas"] = {"path": "atlas.png"}
    else:
        manifest["atlas"] = None

    (OUTDIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest + images -> {OUTDIR}")


if __name__ == "__main__":
    main()
