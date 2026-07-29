"""DEFECT ATLAS — big, tightly-zoomed views of each defect type in the raw CT.

For each verdict (present / bent / thin / disconnected / missing) we pick the
CLEAREST interior examples (verified by sampling the actual material) and render
a tight, high-zoom crop with the strut marked, so you can see exactly what each
defect looks like.
"""

import json
import os
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT, STK, BASE
UNI = STK / f"{BASE}_unified_defects_accurate.json"
BENT = STK / f"{BASE}_asbuilt_bent.json"
RAW = STK / f"{BASE}.tif"
MASK = STK / f"{BASE}_segmented_clean.tif"
OUT = Path(
    os.environ.get("LATTICE_OUTPUT_DIR", str(ROOT / "analysis/defect_detection"))
) / "VIZ_9_defect_atlas.png"

NCOL = 6
UM = 58.1
COLORLABEL = {"present": ("#3b82f6", "PRESENT (healthy)"),
              "bent": ("#c026d3", "BENT"),
              "thin": ("#eab308", "THIN"),
              "disconnected": ("#f97316", "DISCONNECTED (broken)"),
              "missing": ("#ef4444", "MISSING")}


def longest_gap(hit):
    best = cur = 0
    for h in hit:
        cur = 0 if h else cur + 1; best = max(best, cur)
    return best


def sample(p0, p1, metal, raw, shape, K=26):
    ts = np.linspace(0, 1, K)
    pts = np.round(p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]).astype(int)
    pts = np.clip(pts, 0, shape - 1)
    hit = metal[pts[:, 0], pts[:, 1], pts[:, 2]]
    d = raw[pts[:, 0], pts[:, 1], pts[:, 2]].astype(float)
    mid = hit[2:-2]
    return mid.mean(), longest_gap(mid) / len(mid), d[2:-2].mean()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    uni = json.load(open(UNI))["struts"]
    bent = json.load(open(BENT))["struts"]
    bthr = json.load(open(BENT))["meta"]["threshold_vox"]

    print("loading raw + mask ...")
    raw = tifffile.imread(RAW)
    metal = ndi.binary_dilation(tifffile.imread(MASK) > 0, iterations=1)
    shape = np.array(raw.shape)
    margin = 70

    def interior(pz):  # (z,y,x)
        return np.all(pz > margin) and np.all(pz < shape - margin)

    # ---- gather candidates with measurements ----
    def uni_pos(s):
        return np.array(s["p0"])[::-1], np.array(s["p1"])[::-1]     # [x,y,z]->(z,y,x)

    picks = {}
    # PRESENT: high metal, clear
    cands = []
    for s in uni:
        if s["verdict"] != "present":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        f, g, d = sample(p0, p1, metal, raw, shape)
        if f > 0.95 and g < 0.1:
            cands.append((d, p0, p1, f, g, d))       # sort by density (pick solid)
    cands.sort(key=lambda x: -x[0]); picks["present"] = cands[:NCOL]

    # solid reference density from the present controls (for the thin comparison)
    ref_d = np.median([c[0] for c in picks["present"]]) if picks["present"] else 43000.0

    # THIN: skinny filament — pick the most-continuous ones so the thread is visible
    cands = []
    for s in uni:
        if s["verdict"] != "thin":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        f, g, d = sample(p0, p1, metal, raw, shape)
        cands.append((f, g, p0, p1, f, g, d))
    cands.sort(key=lambda x: (-x[0], x[1]))                 # most metal-visible, least gap
    picks["thin"] = [(c[4], c[2], c[3], c[4], c[5], c[6]) for c in cands[:NCOL]]

    # DISCONNECTED: clear contiguous gap
    cands = []
    for s in uni:
        if s["verdict"] != "disconnected":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        f, g, d = sample(p0, p1, metal, raw, shape)
        if 0.25 <= g <= 0.55:
            cands.append((-g, p0, p1, f, g, d))
    cands.sort(key=lambda x: x[0]); picks["disconnected"] = cands[:NCOL]

    # MISSING: lowest metal fraction (clearest empty)
    cands = []
    for s in uni:
        if s["verdict"] != "missing":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        f, g, d = sample(p0, p1, metal, raw, shape)
        cands.append((f, p0, p1, f, g, d))
    cands.sort(key=lambda x: x[0]); picks["missing"] = cands[:NCOL]

    # BENT: clear bows (7-14 vox), interior
    cands = []
    for s in bent:
        p0 = np.array(s["p0"]); p1 = np.array(s["p1"])       # already (z,y,x)
        if not (interior(p0) and interior(p1)):
            continue
        if 7 <= s["max_dev"] <= 14:
            cands.append((s["max_dev"], p0, p1, s["max_dev"], 0, 0))
    cands.sort(key=lambda x: -x[0]); picks["bent"] = cands[:NCOL]

    # ---- render ----
    rows = ["present", "bent", "thin", "disconnected", "missing"]
    fig, axes = plt.subplots(len(rows), NCOL, figsize=(4.6 * NCOL, 4.7 * len(rows)),
                             facecolor="white")
    for r, cat in enumerate(rows):
        color, label = COLORLABEL[cat]
        for c in range(NCOL):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(picks[cat]):
                continue
            _, p0, p1, f, g, d = picks[cat]  if False else picks[cat][c]
            vec = p1 - p0; zax = int(np.argmin(np.abs(vec)))
            a2 = [a for a in range(3) if a != zax]
            lo = np.minimum(p0, p1).astype(int) - 9
            hi = np.maximum(p0, p1).astype(int) + 10
            lo = np.maximum(lo, 0); hi = np.minimum(hi, shape)
            zc = int((p0[zax] + p1[zax]) / 2); zl, zh = max(zc - 4, 0), min(zc + 5, shape[zax])
            sl = [slice(lo[a], hi[a]) for a in range(3)]; sl[zax] = slice(zl, zh)
            img = raw[tuple(sl)].max(axis=zax).astype(float)
            vmin, vmax = np.percentile(img, 2), np.percentile(img, 99.7)
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            e0 = [p0[a] - lo[a] for a in a2]; e1 = [p1[a] - lo[a] for a in a2]
            ax.plot([e0[1], e1[1]], [e0[0], e1[0]], "--", c=color, lw=2.0, alpha=0.85)
            ax.scatter([e0[1], e1[1]], [e0[0], e1[0]], c=color, s=55,
                       edgecolors="white", linewidths=1.0, zorder=5)
            if cat == "bent":
                sub = f"bow {f:.1f} vox = {f*UM:.0f} µm"
            elif cat == "missing":
                sub = f"metal {100*f:.0f}%  (empty)"
            elif cat == "disconnected":
                sub = f"gap {100*g:.0f}% of strut"
            elif cat == "thin":
                sub = f"density {d:.0f} vs {ref_d:.0f} solid"
            else:
                sub = f"metal {100*f:.0f}%  solid"
            ax.set_title(sub, fontsize=11, color=color, fontweight="bold")
        axes[r, 0].axis("on"); axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_color(color); spine.set_linewidth(3)
        axes[r, 0].set_ylabel(label, fontsize=15, fontweight="bold", color=color)

    fig.suptitle("DEFECT ATLAS — zoomed raw CT of each strut type  ·  dashed line = the strut, dots = its joints\n"
                 "PRESENT = full solid bar  ·  BENT = curves off the line  ·  THIN = skinny/faint  ·  "
                 "DISCONNECTED = gap in the middle  ·  MISSING = empty path",
                 fontsize=17, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(OUT, dpi=135, bbox_inches="tight", facecolor="white")
    print("saved", OUT.name)
    for cat in rows:
        print(f"  {cat}: {len(picks[cat])} examples")


if __name__ == "__main__":
    main()
