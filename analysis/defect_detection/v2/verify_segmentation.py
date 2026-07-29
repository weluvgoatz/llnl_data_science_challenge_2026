"""Prove the built-in segmentation reproduces the validated mask exactly.

Segments the raw TIF with v2/segment.py and compares the result voxel-by-voxel
against the existing {BASE}_segmented_clean.tif that the 200-panel manual
validation was performed on.  Any mismatch means the self-contained path would
NOT give the validated results.
"""
import gc
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW, MASK  # noqa: E402
from segment import segment_volume  # noqa: E402


def main():
    print(f"loading raw {Path(RAW).name} ...", flush=True)
    vol = tifffile.imread(RAW)
    print(f"  shape {vol.shape}, dtype {vol.dtype}", flush=True)

    print("segmenting with v2/segment.py ...", flush=True)
    mine, t = segment_volume(vol)
    del vol
    gc.collect()

    print(f"loading validated mask {Path(MASK).name} ...", flush=True)
    ref = tifffile.imread(MASK) > 0

    same = bool(np.array_equal(mine, ref))
    print()
    print("=== COMPARISON ===")
    print(f"  validated mask metal voxels : {int(ref.sum()):,}")
    print(f"  my segmentation metal voxels: {int(mine.sum()):,}")
    if same:
        print("  IDENTICAL — every voxel matches. The self-contained path")
        print("  reproduces the mask the 200 validated panels were built on.")
    else:
        diff = mine ^ ref
        n = int(diff.sum())
        print(f"  MISMATCH: {n:,} voxels differ "
              f"({100*n/ref.size:.5f}% of volume, "
              f"{100*n/max(int(ref.sum()),1):.3f}% of metal)")
        print(f"    extra in mine  : {int((mine & ~ref).sum()):,}")
        print(f"    missing in mine: {int((~mine & ref).sum()):,}")
    print(f"  Otsu threshold used: {t:.0f}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
