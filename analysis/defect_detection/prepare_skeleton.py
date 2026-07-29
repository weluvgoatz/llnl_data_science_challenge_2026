"""Create the compact skeleton-coordinate cache used by defect analysis."""

import numpy as np
import tifffile
from skimage.morphology import skeletonize

from config import MASK, SKELC


def main() -> None:
    if SKELC.is_file():
        with np.load(SKELC) as cached:
            coords = cached["coords"]
        if coords.ndim == 2 and coords.shape[1] == 3 and len(coords):
            print(f"Reusing {SKELC.name} ({len(coords):,} voxels)")
            return

    mask = tifffile.imread(MASK) > 0
    coords = np.argwhere(skeletonize(mask))
    np.savez_compressed(SKELC, coords=coords)
    print(f"Saved {SKELC.name} ({len(coords):,} voxels)")


if __name__ == "__main__":
    main()
