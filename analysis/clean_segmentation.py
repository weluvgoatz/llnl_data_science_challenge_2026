"""Clean up a binary segmentation .tif for easier viewing.

Removes scattered speckle (small stray blobs) and smooths/fills the struts, so
the lattice reads clearly in a viewer. Operates on an existing segmented .tif
(fast - no need to reload the raw CT).

Usage (from repo root):
    python analysis/clean_segmentation.py --mask "path/to/..._segmented.tif"
    python analysis/clean_segmentation.py --mask "..." --min-size 25
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask", required=True, help="Input binary segmentation .tif.")
    ap.add_argument("--out", default=None, help="Output .tif (default: <name>_clean.tif).")
    ap.add_argument("--min-size", type=int, default=20,
                    help="Remove connected blobs smaller than this many voxels (default 20).")
    args = ap.parse_args()

    mask_path = Path(args.mask).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else \
        mask_path.with_name(mask_path.stem + "_clean.tif")

    print(f"Loading {mask_path.name} ...")
    mask = tifffile.imread(mask_path) > 0
    before = int(mask.sum())

    # 1) Remove scattered small noise blobs (the floating specks).
    print(f"Removing blobs smaller than {args.min_size} voxels ...")
    mask = remove_small_objects(mask, min_size=args.min_size)

    # 2) Fill tiny pinholes and smooth strut edges (closing = dilate then erode).
    print("Closing pinholes / smoothing ...")
    mask = ndi.binary_closing(mask, iterations=1)

    after = int(mask.sum())
    removed = before - int((tifffile.imread(mask_path) > 0).sum()) if False else None
    print(f"  foreground voxels: {before:,} -> {after:,}")

    out = (mask.astype(np.uint8) * 255)
    tifffile.imwrite(out_path, out, compression="zlib")
    print(f"Saved cleaned TIF: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
