"""Threshold sweep analyzer for CT segmentation.

Given a raw 3D CT volume (.npy), this script segments it at a range of
threshold values and reports, for each threshold, quantitative quality
metrics plus visual comparisons. It then recommends a threshold using
Otsu's method (a principled statistical separator between the background
and material intensity populations) and confirms that choice against the
connectivity of the resulting mask.

It is data-adaptive: thresholds are chosen from the volume's ACTUAL
intensity range, so it works whether the CT data is raw (e.g. values in
[-0.003, 0.015]) or normalized to [0, 1]. This matters because a fixed
sweep like 0.3/0.5/0.7 only makes sense for normalized data.

Usage (from the repository root):
    python .agents/skills/threshold_optimizer/scripts/analyze_thresholds.py \
        --volume data/unitcell/unitcell.npy \
        --output-dir data/unitcell/threshold_sweep \
        --num 7 --save-masks

    # or with explicit thresholds:
    python .agents/skills/threshold_optimizer/scripts/analyze_thresholds.py \
        --volume data/unitcell/unitcell.npy \
        --output-dir data/unitcell/threshold_sweep \
        --thresholds 0.004,0.005,0.006,0.007,0.008
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# 26-connectivity in 3D: a voxel touches its neighbor if they share a
# face, edge, or corner. This is the standard choice for lattice struts,
# which run diagonally and would otherwise appear "disconnected".
CONNECTIVITY_3D = np.ones((3, 3, 3), dtype=int)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze CT segmentation across a threshold sweep.")
    parser.add_argument("--volume", required=True, help="Path to the raw 3D CT .npy volume.")
    parser.add_argument("--output-dir", required=True, help="Directory for figures, metrics, and (optional) masks.")
    parser.add_argument("--thresholds", default=None, help="Comma-separated explicit thresholds. Overrides --num.")
    parser.add_argument("--num", type=int, default=7, help="Number of thresholds to auto-generate across the data range.")
    parser.add_argument("--slice-index", type=int, default=None, help="Slice index for the visual comparison (default: middle).")
    parser.add_argument("--axis", type=int, default=0, choices=(0, 1, 2), help="Axis for the comparison slice.")
    parser.add_argument("--save-masks", action="store_true", help="Also save each threshold's mask as a .npy file.")
    return parser.parse_args()


def choose_thresholds(volume, num):
    """Auto-generate thresholds spanning the data's actual intensity range.

    The extreme min/max are excluded because they give trivially full or
    empty masks; the interior points sweep from heavy over-segmentation to
    heavy under-segmentation so the trade-off is visible.
    """
    lo = float(volume.min())
    hi = float(volume.max())
    # num interior points, evenly spaced, excluding the exact endpoints.
    return list(np.linspace(lo, hi, num + 2)[1:-1])


def otsu_threshold(volume):
    """Otsu's method: the threshold that maximizes between-class variance.

    Intuitively, it finds the valley between the two peaks of the intensity
    histogram (background vs material). scikit-image is used if available;
    otherwise a compact NumPy fallback computes the same quantity.
    """
    try:
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(volume.ravel()))
    except Exception:
        # NumPy fallback: classic Otsu over a 256-bin histogram.
        counts, edges = np.histogram(volume.ravel(), bins=256)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = counts.sum()
        weight_bg = np.cumsum(counts)
        weight_fg = total - weight_bg
        valid = (weight_bg > 0) & (weight_fg > 0)
        cum_mean = np.cumsum(counts * centers)
        mean_bg = np.zeros_like(centers)
        mean_fg = np.zeros_like(centers)
        mean_bg[valid] = cum_mean[valid] / weight_bg[valid]
        mean_fg[valid] = (cum_mean[-1] - cum_mean[valid]) / weight_fg[valid]
        between = np.zeros_like(centers)
        between[valid] = weight_bg[valid] * weight_fg[valid] * (mean_bg[valid] - mean_fg[valid]) ** 2
        return float(centers[np.argmax(between)])


def analyze_threshold(volume, threshold):
    """Segment at one threshold and compute quality metrics.

    Returns a dict with:
      - fraction: percent of the volume classified as material
      - components: number of 26-connected foreground pieces
      - largest_fraction: share of foreground in the single biggest piece
        (near 1.0 means one clean structure; low means fragmentation)
    """
    mask = volume >= threshold
    foreground = int(mask.sum())
    total = int(mask.size)

    if foreground == 0:
        return mask, {
            "threshold": float(threshold),
            "foreground_voxels": 0,
            "fraction_pct": 0.0,
            "components": 0,
            "largest_fraction": 0.0,
        }

    labels, n_components = ndi.label(mask, structure=CONNECTIVITY_3D)
    # Component sizes; index 0 is background, so ignore it.
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max()) if n_components > 0 else 0

    return mask, {
        "threshold": float(threshold),
        "foreground_voxels": foreground,
        "fraction_pct": 100.0 * foreground / total,
        "components": int(n_components),
        "largest_fraction": largest / foreground,
    }


def recommend(metrics, t_otsu):
    """Pick a recommended threshold.

    Otsu is the principled continuous optimum. We report it, then find the
    swept threshold nearest to it whose mask is essentially a single
    connected structure (largest_fraction >= 0.95) as a confirmation.
    """
    swept = [m for m in metrics if m["foreground_voxels"] > 0]
    nearest = min(swept, key=lambda m: abs(m["threshold"] - t_otsu)) if swept else None
    single_structure = [m for m in swept if m["largest_fraction"] >= 0.95]
    # Among near-single-structure thresholds, prefer the one closest to Otsu.
    confirmed = (
        min(single_structure, key=lambda m: abs(m["threshold"] - t_otsu))
        if single_structure
        else nearest
    )
    return nearest, confirmed


def plot_metric_curves(metrics, t_otsu, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = [m["threshold"] for m in metrics]
    frac = [m["fraction_pct"] for m in metrics]
    ncc = [m["components"] for m in metrics]
    lcc = [m["largest_fraction"] for m in metrics]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(ts, frac, "o-"); ax[0].set_title("Material fraction (%)")
    ax[1].plot(ts, ncc, "o-", color="tab:orange"); ax[1].set_title("Connected components")
    ax[2].plot(ts, lcc, "o-", color="tab:green"); ax[2].set_title("Largest-component fraction")
    for a in ax:
        a.axvline(t_otsu, color="red", ls="--", lw=1, label=f"Otsu={t_otsu:.5f}")
        a.set_xlabel("threshold"); a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_slice_comparison(volume, metrics, thresholds, axis, slice_index, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw_slice = np.take(volume, slice_index, axis=axis)
    n = len(thresholds) + 1
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(raw_slice, cmap="gray")
    axes[0].set_title(f"Raw CT (slice {slice_index})")
    axes[0].axis("off")

    for i, (t, m) in enumerate(zip(thresholds, metrics), start=1):
        mask_slice = np.take(volume >= t, slice_index, axis=axis)
        axes[i].imshow(mask_slice, cmap="gray")
        axes[i].set_title(f"t={t:.5f}\n{m['fraction_pct']:.2f}%  cc={m['components']}")
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    volume_path = Path(args.volume).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    volume = np.load(volume_path, allow_pickle=False)
    if volume.ndim != 3:
        raise SystemExit(f"Expected a 3D volume, got shape {volume.shape}")
    if volume.dtype == bool:
        raise SystemExit("Input is a boolean mask, not a raw intensity volume. Provide the raw CT .npy.")

    if args.thresholds:
        thresholds = [float(x) for x in args.thresholds.split(",")]
    else:
        thresholds = choose_thresholds(volume, args.num)

    slice_index = args.slice_index if args.slice_index is not None else volume.shape[args.axis] // 2

    print(f"Volume: {volume_path}")
    print(f"Shape: {volume.shape}  dtype: {volume.dtype}")
    print(f"Intensity range: {float(volume.min()):.6f} to {float(volume.max()):.6f}")
    print(f"Thresholds: {[round(t, 6) for t in thresholds]}\n")

    t_otsu = otsu_threshold(volume)

    metrics = []
    for t in thresholds:
        mask, m = analyze_threshold(volume, t)
        metrics.append(m)
        if args.save_masks:
            np.save(output_dir / f"mask_thr_{t:.6f}.npy", mask)
        print(
            f"t={t:.6f}  material={m['fraction_pct']:6.2f}%  "
            f"components={m['components']:5d}  largest={m['largest_fraction']:.4f}"
        )

    # Write metrics to CSV for downstream use.
    csv_path = output_dir / "threshold_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    # Figures.
    curves_path = output_dir / "threshold_metric_curves.png"
    comparison_path = output_dir / "threshold_slice_comparison.png"
    plot_metric_curves(metrics, t_otsu, curves_path)
    plot_slice_comparison(volume, metrics, thresholds, args.axis, slice_index, comparison_path)

    nearest, confirmed = recommend(metrics, t_otsu)

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"Otsu threshold (statistical optimum): {t_otsu:.6f}")
    if confirmed:
        print(
            f"Confirmed threshold (single connected structure, near Otsu): "
            f"{confirmed['threshold']:.6f}"
        )
        print(
            f"  -> material {confirmed['fraction_pct']:.2f}%, "
            f"{confirmed['components']} component(s), "
            f"largest-fraction {confirmed['largest_fraction']:.4f}"
        )
    print(f"\nMetrics CSV:        {csv_path}")
    print(f"Metric curves:      {curves_path}")
    print(f"Slice comparison:   {comparison_path}")


if __name__ == "__main__":
    main()
