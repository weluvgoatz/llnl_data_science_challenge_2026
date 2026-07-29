"""PER-DEFECT PIPELINE VALIDATION — one 4-panel figure for each defect type.

For MISSING, DISCONNECTED, THIN and BENT we take ONE clear representative strut
and walk it through the model exactly as the classifier does:

  STEP 1  SEGMENTATION      raw CT + the isolated metal (Otsu mask)
  STEP 2  SKELETONIZATION   the medial-axis centreline + graph joints
  STEP 3  DETECTION         the real measurement the model thresholds on
  RESULT  VERDICT           the coloured strut with the number that decided it

The decision rules (from unified_defects_accurate.py):
  missing       : metal fraction along the strut < 0.15   (or joint has no metal)
  disconnected  : longest contiguous gap >= 0.25 of length
  thin          : strut density < median - 3*MAD  (robust outlier)
  bent          : max bow from strut axis > 1 radius = 3.65 vox = 212 um
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
from matplotlib.patches import Patch

from config import ROOT, STK, BASE
UNI = STK / f"{BASE}_unified_defects_accurate.json"
BENT = STK / f"{BASE}_asbuilt_bent.json"
GRAPH = STK / f"{BASE}_segmented_clean_asbuilt_graph_cleaned.json"
RAW = STK / f"{BASE}.tif"
MASK = STK / f"{BASE}_segmented_clean.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
OUTDIR = Path(
    os.environ.get("LATTICE_OUTPUT_DIR", str(ROOT / "analysis/defect_detection"))
)

UM = 58.1
MISSING_FRAC = 0.15
GAP_FRAC = 0.25
BENT_THR = 424.0 / 2 / UM               # 3.65 vox
K_OUT = 3.0
COL = {"missing": "#ef4444", "disconnected": "#f97316",
       "thin": "#eab308", "bent": "#c026d3", "present": "#3b82f6"}


def longest_gap_run(hit):
    best = cur = bs = cs = 0
    for i, h in enumerate(hit):
        if h:
            cur = 0
        else:
            cur = cur + 1 if cur else 1
            cs = i - cur + 1
            if cur > best:
                best = cur; bs = cs
    return best, bs                       # length, start-index


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("loading raw, mask, skeleton, graph ...")
    raw = tifffile.imread(RAW)
    mask = tifffile.imread(MASK) > 0
    metal = ndi.binary_dilation(mask, iterations=1)
    shape = np.array(raw.shape)
    coords = np.load(SKELC)["coords"]                       # (N,3) z,y,x
    junctions = np.array([j["position"] for j in json.load(open(GRAPH))["junctions"]], float)  # z,y,x
    uni = json.load(open(UNI))["struts"]
    bent = json.load(open(BENT))["struts"]
    margin = 55

    def interior(p):
        return np.all(p > margin) and np.all(p < shape - margin)

    def uni_pos(s):
        return np.array(s["p0"])[::-1], np.array(s["p1"])[::-1]

    def metal_line(p0, p1, K=24):
        ts = np.linspace(0, 1, K)
        pts = np.round(p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]).astype(int)
        pts = np.clip(pts, 0, shape - 1)
        return metal[pts[:, 0], pts[:, 1], pts[:, 2]][2:-2]

    def raw_line(p0, p1, K=16):
        ts = np.linspace(0, 1, K)
        pts = np.round(p0[None] * (1 - ts[:, None]) + p1[None] * ts[:, None]).astype(int)
        pts = np.clip(pts, 0, shape - 1)
        return float(raw[pts[:, 0], pts[:, 1], pts[:, 2]].mean())

    def ball(p):                                   # metal fraction in a 7^3 box
        c = np.round(p).astype(int)
        return metal[max(c[0]-3, 0):c[0]+4, max(c[1]-3, 0):c[1]+4,
                     max(c[2]-3, 0):c[2]+4].mean()

    def strut_chain(p0, p1):
        """Ordered centreline voxels of the strut between p0,p1 (degree-2 chain),
        traced locally so the polyline lies exactly on the metal."""
        u = p1 - p0; L = np.linalg.norm(u); u = u / L
        pad = 16
        clo = np.maximum(np.minimum(p0, p1).astype(int) - pad, 0)
        chi = np.minimum(np.maximum(p0, p1).astype(int) + pad + 1, shape)
        cm = np.all((coords >= clo) & (coords < chi), axis=1)
        local = np.zeros(chi - clo, bool)
        lc = coords[cm] - clo; local[lc[:, 0], lc[:, 1], lc[:, 2]] = True
        Kd = np.ones((3, 3, 3), int); Kd[1, 1, 1] = 0
        degl = ndi.convolve(local.astype(np.uint8), Kd, mode="constant") * local
        pathv = local & (degl == 2)
        lbl, nlbl = ndi.label(pathv, structure=np.ones((3, 3, 3), int))
        pz, py, px = np.where(pathv)
        g = np.stack([pz, py, px], 1) + clo; pl = lbl[pz, py, px]
        keep = []
        for c in range(1, nlbl + 1):
            pc = g[pl == c].astype(float)
            if len(pc) < 5:
                continue
            tt = (pc - p0) @ u
            pp = np.linalg.norm((pc - p0) - np.outer(tt, u), axis=1)
            if (np.median(pp) < 5.0 and tt.min() > -6 and tt.max() < L + 6
                    and (tt.max() - tt.min()) > 0.30 * L):
                keep.append(pc)
        if not keep:
            return None
        Q = np.concatenate(keep)
        return Q[np.argsort((Q - p0) @ u)]

    # ---- density distribution + robust thin cutoff (as the model computes) ----
    dens_all = []
    for s in uni:
        if s["verdict"] in ("present", "thin", "bent"):
            p0, p1 = uni_pos(s)
            dens_all.append(raw_line(p0, p1))
    dens_all = np.array(dens_all)
    dmed = np.median(dens_all); dsig = 1.4826 * np.median(np.abs(dens_all - dmed))
    d_cut = dmed - K_OUT * dsig
    print(f"  density median {dmed:.0f}, thin cutoff {d_cut:.0f}")

    # ---- pick one clear representative per defect ----
    picks = {}

    # MISSING: textbook case — BOTH joints present on solid metal, empty span
    # between them (not a whole absent region). Pick the most central such strut.
    best = None
    for s in uni:
        if s["verdict"] != "missing":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        if ball(p0) > 0.5 and ball(p1) > 0.5 and metal_line(p0, p1).mean() < 0.15:
            central = min(np.min([p0, p1]), np.min(shape - np.maximum(p0, p1)))
            if best is None or central > best[0]:
                best = (central, p0, p1)
    picks["missing"] = best[1:]

    # DISCONNECTED: two-sided mid-break — metal (and a skeleton stub) on BOTH
    # sides with the gap centred, so both stubs read as distinct lines.
    best = None
    for s in uni:
        if s["verdict"] != "disconnected":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        hit = metal_line(p0, p1, K=40)
        glen, gs = longest_gap_run(hit); ge = gs + glen
        gfrac = glen / len(hit); center = (gs + ge) / 2 / len(hit)
        before = hit[:gs].mean() if gs > 3 else 0.0
        after = hit[ge:].mean() if len(hit) - ge > 3 else 0.0
        stub0 = gs / len(hit); stub1 = (len(hit) - ge) / len(hit)   # length of each stub
        if (0.25 <= gfrac <= 0.5 and before > 0.6 and after > 0.6
                and stub0 >= 0.22 and stub1 >= 0.22):     # BOTH stubs long enough to see
            score = abs(center - 0.5) + abs(gfrac - 0.33)
            if best is None or score < best[0]:
                best = (score, p0, p1)
    picks["disconnected"] = best[1:]

    # THIN: most-continuous, lowest density
    best = None
    for s in uni:
        if s["verdict"] != "thin":
            continue
        p0, p1 = uni_pos(s)
        if not (interior(p0) and interior(p1)):
            continue
        f = metal_line(p0, p1).mean(); d = raw_line(p0, p1)
        score = (-f, d)
        if best is None or score < best[0]:
            best = (score, p0, p1)
    picks["thin"] = best[1:]

    # BENT: pick a clearly-bent, CONTINUOUS strut — the largest bow (up to ~13 vox)
    # that still traces one clean, contiguous centreline chain.
    cand = sorted([s for s in bent
                   if interior(np.array(s["p0"], float)) and interior(np.array(s["p1"], float))
                   and 6.0 <= s["max_dev"] <= 13.0],
                  key=lambda s: -s["max_dev"])
    chosen = None
    for s in cand:
        p0 = np.array(s["p0"], float); p1 = np.array(s["p1"], float)
        Q = strut_chain(p0, p1)
        if Q is None or len(Q) < 14:
            continue
        cen = Q.mean(0); Xc = Q - cen
        ax_ = np.linalg.svd(Xc, full_matrices=False)[2][0]
        Qs = Q[np.argsort(Xc @ ax_)]
        if np.linalg.norm(np.diff(Qs, axis=0), axis=1).max() <= 3.0:   # single clean chain
            chosen = (s["max_dev"], p0, p1); break
    if chosen is None:                                    # fallback: any bent, largest bow
        s = cand[0] if cand else bent[0]
        chosen = (s["max_dev"], np.array(s["p0"], float), np.array(s["p1"], float))
    picks["bent"] = chosen[1:]
    bent_bow = chosen[0]

    # ---- crop / projection helpers ----
    def crop_geom(p0, p1, pad=15, slab=6):
        vec = p1 - p0; zax = int(np.argmin(np.abs(vec)))
        a2 = [a for a in range(3) if a != zax]
        lo = np.maximum(np.minimum(p0, p1).astype(int) - pad, 0)
        hi = np.minimum(np.maximum(p0, p1).astype(int) + pad + 1, shape)
        zc = int((p0[zax] + p1[zax]) / 2)
        zl, zh = max(zc - slab, 0), min(zc + slab + 1, shape[zax])
        return zax, a2, lo, hi, zl, zh

    def project(vol, zax, a2, lo, hi, zl, zh):
        sl = [slice(lo[a], hi[a]) for a in range(3)]; sl[zax] = slice(zl, zh)
        return vol[tuple(sl)].max(axis=zax)

    def to2d(p, a2, lo):
        return (p[a2[1]] - lo[a2[1]], p[a2[0]] - lo[a2[0]])   # (x=col, y=row)

    def oblique_view(p0, p1, slab=8, pad=18):
        """Render the strut in ITS OWN bending plane: rotate a small crop so the
        strut lies horizontal (e1) and its bow points up (e2), then MIP along the
        third axis (e3). This makes a bent strut read as a continuous curved bar
        instead of fading out of an axis-aligned slab."""
        Q = strut_chain(p0, p1)
        if Q is None or len(Q) < 8:
            uu = (p1 - p0) / np.linalg.norm(p1 - p0)
            lo0 = np.maximum(np.minimum(p0, p1).astype(int) - pad, 0)
            hi0 = np.minimum(np.maximum(p0, p1).astype(int) + pad + 1, shape)
            cmf = np.all((coords >= lo0) & (coords < hi0), axis=1)
            P = coords[cmf].astype(float); t0 = (P - p0) @ uu
            Q = P[np.linalg.norm((P - p0) - np.outer(t0, uu), axis=1) < 5]
        cen = Q.mean(0); Xc = Q - cen
        axis = np.linalg.svd(Xc, full_matrices=False)[2][0]
        Qs = Q[np.argsort(Xc @ axis)]; a0, b0 = Qs[0], Qs[-1]
        e1v = b0 - a0; e1v = e1v / np.linalg.norm(e1v)
        tt = np.clip(((Qs - a0) @ (b0 - a0)) / float((b0 - a0) @ (b0 - a0)), 0, 1)
        dvec = Qs - (a0 + np.outer(tt, (b0 - a0)))
        k = int(np.argmax(np.linalg.norm(dvec, axis=1)))
        e2v = dvec[k] - (dvec[k] @ e1v) * e1v
        if np.linalg.norm(e2v) < 1e-6:
            tmp = np.array([1.0, 0, 0]); e2v = tmp - (tmp @ e1v) * e1v
        e2v = e2v / np.linalg.norm(e2v)
        e3v = np.cross(e1v, e2v); e3v = e3v / np.linalg.norm(e3v)
        R = np.stack([e3v, e2v, e1v])                 # rows: (view, row, col) in z,y,x
        c = (p0 + p1) / 2.0; cw = np.round(c).astype(int)
        half = int(np.linalg.norm(p1 - p0) / 2) + pad
        lo = np.maximum(cw - half, 0); hi = np.minimum(cw + half + 1, shape)
        sub_raw = raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        sub_mask = metal[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
        csize = np.array(sub_raw.shape); c_out = (csize - 1) / 2.0
        M = R.T; off = (c - lo) - M @ c_out
        rot_raw = ndi.affine_transform(sub_raw, M, offset=off, output_shape=tuple(csize),
                                       order=1, cval=0.0)
        rot_mask = ndi.affine_transform(sub_mask, M, offset=off, output_shape=tuple(csize),
                                        order=0, cval=0.0)
        mid = int(round(c_out[0]))
        zl, zh = max(mid - slab, 0), min(mid + slab + 1, csize[0])
        img = rot_raw[zl:zh].max(axis=0); mimg = rot_mask[zl:zh].max(axis=0)
        return dict(R=R, c=c, c_out=c_out, mid=mid, slab=slab, lo=lo, hi=hi,
                    img=img, mimg=mimg)

    # ================= render one figure per defect =================
    order = ["missing", "disconnected", "thin", "bent"]
    RULE = {
        "missing": "RULE:  metal along the strut  <  15%   (or a joint sits on no metal)",
        "disconnected": "RULE:  one continuous gap  ≥  25%  of the strut length",
        "thin": "RULE:  strut density  <  median − 3·MAD   (robust low-material outlier)",
        "bent": "RULE:  peak bow off the strut axis  >  1 radius = 212 µm",
    }
    for cat in order:
        p0, p1 = picks[cat]
        color = COL[cat]
        if cat == "bent":                             # view in the strut's bending plane
            ov = oblique_view(p0, p1)
            R, cW, c_out, mid, slab = ov["R"], ov["c"], ov["c_out"], ov["mid"], ov["slab"]
            lo, hi = ov["lo"], ov["hi"]
            img = ov["img"].astype(float); mimg = ov["mimg"]

            def to2d_fn(w):
                o = c_out + R @ (np.asarray(w, float) - cW)
                return (o[2], o[1])                   # (col=e1 length, row=e2 bow)

            cm = np.all((coords >= lo) & (coords < hi), axis=1)
            oc = c_out + (coords[cm].astype(float) - cW) @ R.T
            insl = np.abs(oc[:, 0] - mid) <= slab
            sk_col = oc[insl, 2]; sk_row = oc[insl, 1]
            jm = np.all((junctions >= lo) & (junctions < hi), axis=1)
            joc = c_out + (junctions[jm] - cW) @ R.T
            jvis = np.abs(joc[:, 0] - mid) <= slab + 4
            jj2d = [(q[2], q[1]) for q in joc[jvis]]
        else:
            zax, a2, lo, hi, zl, zh = crop_geom(p0, p1, slab=7)
            img = project(raw, zax, a2, lo, hi, zl, zh).astype(float)
            mimg = project(metal.astype(np.uint8), zax, a2, lo, hi, zl, zh)

            def to2d_fn(w, a2=a2, lo=lo):
                return (w[a2[1]] - lo[a2[1]], w[a2[0]] - lo[a2[0]])

            cm = ((coords[:, 0] >= lo[0]) & (coords[:, 0] < hi[0]) &
                  (coords[:, 1] >= lo[1]) & (coords[:, 1] < hi[1]) &
                  (coords[:, 2] >= lo[2]) & (coords[:, 2] < hi[2]))
            sc = coords[cm]; sc = sc[(sc[:, zax] >= zl) & (sc[:, zax] < zh)]
            sk_col = sc[:, a2[1]] - lo[a2[1]]; sk_row = sc[:, a2[0]] - lo[a2[0]]
            jm = np.all((junctions >= lo) & (junctions < hi), axis=1)
            jj2d = [to2d_fn(p) for p in junctions[jm]]
        vmin, vmax = np.percentile(img, 2), np.percentile(img, 99.6)
        e0 = to2d_fn(p0); e1 = to2d_fn(p1)

        fig, ax = plt.subplots(1, 4, figsize=(23, 6.2), facecolor="white")

        # ---- STEP 1 SEGMENTATION ----
        a = ax[0]; a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ov = np.zeros((*mimg.shape, 4)); ov[mimg > 0] = (0.10, 0.85, 0.45, 0.45)
        a.imshow(ov)
        a.set_title("STEP 1 · SEGMENTATION\nraw CT  +  isolated metal (green)", fontsize=13, fontweight="bold")
        a.axis("off")

        # ---- STEP 2 SKELETONIZATION ----
        a = ax[1]; a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax * 1.25, alpha=0.55)
        a.scatter(sk_col, sk_row, s=6, c="#00e5ff", marker="s", linewidths=0)
        if len(jj2d):
            a.scatter([q[0] for q in jj2d], [q[1] for q in jj2d], s=140, facecolors="none",
                      edgecolors="#ffd400", linewidths=2.2, zorder=6)
        a.set_title("STEP 2 · SKELETONIZATION\ncentreline (cyan)  +  graph joints (yellow)", fontsize=13, fontweight="bold")
        a.axis("off")

        # ---- STEP 3 DETECTION (the actual measurement) ----
        a = ax[2]
        if cat in ("missing", "disconnected"):
            hit = metal_line(p0, p1, K=40).astype(float)
            x = np.linspace(0, 100, len(hit))
            frac = hit.mean()
            a.fill_between(x, 0, hit, step="mid", color=color, alpha=0.30)
            a.step(x, hit, where="mid", color=color, lw=2.2)
            if cat == "missing":
                a.axhline(MISSING_FRAC, ls="--", c="k", lw=1.4)
                a.text(2, MISSING_FRAC + 0.03, "15% threshold", fontsize=10)
                msg = f"metal along strut = {100*frac:.0f}%\n{100*frac:.0f}%  <  15%  →  MISSING"
            else:
                glen, gs = longest_gap_run(hit)
                gfrac = glen / len(hit)
                a.axvspan(x[gs], x[min(gs + glen, len(x) - 1)], color=color, alpha=0.18)
                a.text((x[gs] + x[min(gs + glen, len(x) - 1)]) / 2, 0.5, "GAP",
                       ha="center", fontsize=11, fontweight="bold", color=color)
                msg = f"longest gap = {100*gfrac:.0f}% of strut\n{100*gfrac:.0f}%  ≥  25%  →  DISCONNECTED"
            a.set_ylim(-0.1, 1.25); a.set_yticks([0, 1]); a.set_yticklabels(["empty", "metal"])
            a.set_xlabel("position along strut (%)")
            a.text(0.5, 1.14, msg, transform=a.transAxes, ha="center", fontsize=12,
                   fontweight="bold", color=color)
        elif cat == "thin":
            d = raw_line(p0, p1)
            a.hist(dens_all, bins=60, color="#9ca3af", alpha=0.8)
            a.axvline(d_cut, ls="--", c="k", lw=1.6); a.text(d_cut, a.get_ylim()[1]*0.9,
                     "  thin cutoff\n  (med − 3·MAD)", fontsize=9, va="top")
            a.axvline(d, c=color, lw=3)
            a.annotate("this strut", (d, a.get_ylim()[1]*0.45), (d - 6000, a.get_ylim()[1]*0.6),
                       color=color, fontsize=11, fontweight="bold",
                       arrowprops=dict(arrowstyle="->", color=color, lw=2))
            a.set_xlabel("strut material density  (mean CT grey value, 16-bit)")
            a.set_ylabel("number of struts")
            a.text(0.5, 1.14, f"density = {d:.0f}  <  cutoff {d_cut:.0f}  (CT grey value)  →  THIN",
                   transform=a.transAxes, ha="center", fontsize=12, fontweight="bold", color=color)
        else:  # bent — reproduce the model exactly: degree-2 chain + PCA-axis bow
            Q = strut_chain(p0, p1)
            if Q is None or len(Q) < 8:             # fallback: near-chord skeleton
                uu = (p1 - p0) / np.linalg.norm(p1 - p0)
                cm2 = np.all((coords >= lo - 3) & (coords < hi + 3), axis=1)
                P = coords[cm2].astype(float); tt0 = (P - p0) @ uu
                pp0 = np.linalg.norm((P - p0) - np.outer(tt0, uu), axis=1)
                Q = P[pp0 < 5]
            # model computation: PCA axis of the chain, bow off the end-to-end line
            cen = Q.mean(0); Xc = Q - cen
            axis_ = np.linalg.svd(Xc, full_matrices=False)[2][0]
            Qs = Q[np.argsort(Xc @ axis_)]
            a0, b0 = Qs[0], Qs[-1]; ax_ = b0 - a0
            tt = np.clip(((Qs - a0) @ ax_) / float(ax_ @ ax_), 0, 1)
            dev = np.linalg.norm(Qs - (a0 + np.outer(tt, ax_)), axis=1) * UM
            xarc = tt * 100.0
            a.plot(xarc, dev, "-", color=color, lw=2.6)
            a.axhline(BENT_THR * UM, ls="--", c="k", lw=1.6)
            a.text(3, BENT_THR * UM + 12, "212 µm threshold (1 radius)", fontsize=10)
            a.fill_between(xarc, BENT_THR * UM, dev, where=dev > BENT_THR * UM,
                           color=color, alpha=0.25, interpolate=True)
            a.scatter([xarc[np.argmax(dev)]], [dev.max()], c=color, s=70, zorder=5)
            a.set_ylim(0, max(dev.max(), BENT_THR * UM) * 1.28)
            a.set_xlabel("position along strut (%)"); a.set_ylabel("bow off axis (µm)")
            a.text(0.5, 1.14, f"peak bow = {dev.max():.0f} µm  >  212 µm  →  BENT",
                   transform=a.transAxes, ha="center", fontsize=12, fontweight="bold", color=color)
        a.set_title("STEP 3 · DETECTION\nthe measurement the model thresholds on",
                    fontsize=13, fontweight="bold")

        # ---- RESULT VERDICT ----
        a = ax[3]; a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        drew_chain = False
        if cat in ("thin", "bent"):                 # strut exists -> trace it ON the metal
            ch = strut_chain(p0, p1)
            if ch is not None and len(ch) > 2:
                cc = np.array([to2d_fn(q) for q in ch])
                a.plot(cc[:, 0], cc[:, 1], "-", c=color, lw=2.8, alpha=0.95)
                drew_chain = True
        if not drew_chain:                          # missing/disconnected: intended path
            a.plot([e0[0], e1[0]], [e0[1], e1[1]], "--", c=color, lw=2.4, alpha=0.9)
        a.scatter([e0[0], e1[0]], [e0[1], e1[1]], c=color, s=70,
                  edgecolors="white", linewidths=1.2, zorder=5)
        sub = {"missing": "both joints present — only the bar between them is gone",
               "disconnected": "stub on each side, empty gap in the middle",
               "thin": "line traced along the actual (skinny) centreline",
               "bent": "continuous, curved — shown in the strut's bending plane"}[cat]
        a.set_title(f"RESULT · {cat.upper()}\n{sub}",
                    fontsize=12, fontweight="bold", color=color)
        a.axis("off")

        fig.suptitle(f"HOW THE MODEL VALIDATES A  {cat.upper()}  STRUT\n{RULE[cat]}",
                     fontsize=17, fontweight="bold", y=1.02, color=color)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = OUTDIR / f"VIZ_10_pipeline_{cat}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {out.name}")


if __name__ == "__main__":
    main()
