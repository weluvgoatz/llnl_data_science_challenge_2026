"""Segment a CT .tif stack and save the binary mask as a viewable .tif.

Otsu-thresholds the volume into material (white, 255) vs background (black, 0)
and writes a full-resolution .tif you can open in Fiji/ImageJ or the built-in
viewer, slice by slice, next to the raw scan.

Usage (from repo root):
    python analysis/segment_to_tif.py --tif "path/to/scan.tif"
    python analysis/segment_to_tif.py --tif "path/to/scan.tif" --out "path/to/mask.tif"
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from skimage.filters import threshold_otsu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, help="Input CT .tif stack.")
    ap.add_argument("--out", default=None, help="Output mask .tif (default: <name>_segmented.tif alongside input).")
    args = ap.parse_args()

    tif_path = Path(args.tif).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else \
        tif_path.with_name(tif_path.stem + "_segmented.tif")

    print(f"Loading {tif_path.name} ...")
    vol = tifffile.imread(tif_path)
    print(f"  shape {vol.shape}, dtype {vol.dtype}, range {int(vol.min())}..{int(vol.max())}")

    # Otsu threshold from a downsampled copy (fast) - it's an intensity cutoff,
    # so it applies directly to the full-resolution volume.
    t = float(threshold_otsu(vol[::4, ::4, ::4]))
    mask = (vol >= t).astype(np.uint8) * 255
    fg = int((mask > 0).sum()); tot = mask.size
    print(f"  Otsu threshold {t:.0f}: material {100*fg/tot:.2f}% ({fg:,} voxels)")

    # zlib/Deflate compression -> small file, built into tifffile, and
    # ImageJ/Fiji reads ZIP/Deflate TIFFs natively.
    tifffile.imwrite(out_path, mask, compression="zlib")
    print(f"Saved segmented TIF: {out_path}")
    print(f"  ({out_path.stat().st_size/1e6:.1f} MB, same {vol.shape} dimensions as the raw scan)")


if __name__ == "__main__":
    main()
