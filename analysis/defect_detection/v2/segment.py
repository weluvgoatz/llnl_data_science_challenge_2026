"""Segmentation, self-contained: raw CT volume -> clean binary metal mask.

This reproduces EXACTLY the two-step pipeline the validated mask was made with
(analysis/segment_to_tif.py then analysis/clean_segmentation.py), so that
detect_v2 can start from a raw TIF and still produce the manually-validated
results.  Do not "improve" the steps without re-validating: the defect verdicts
depend on precisely where the metal boundary sits.

  1. Otsu threshold, computed on a 4x-downsampled copy (it is just an intensity
     cutoff, so it transfers to full resolution), keep voxels >= t.
  2. remove_small_objects(min_size=20) - drops scattered speckle.
  3. binary_closing(iterations=1)     - fills pinholes, smooths strut edges.

Both thresholds are data-driven (Otsu) or resolution-level (20 voxels of
speckle), so this works on a different scan without retuning.
"""
import gc

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

OTSU_STRIDE = 4          # downsample factor for the Otsu estimate
MIN_SPECKLE = 20         # connected blobs smaller than this are noise
CLOSE_ITERS = 1          # pinhole fill / edge smoothing


def otsu_threshold(vol, stride=OTSU_STRIDE):
    return float(threshold_otsu(vol[::stride, ::stride, ::stride]))


def drop_small_objects(mask, min_size=MIN_SPECKLE):
    """Remove connected components smaller than min_size voxels.

    Equivalent to skimage.morphology.remove_small_objects(mask, min_size) with
    its default 1-connectivity (face neighbours), but done via an explicit
    int32 labelling: this skimage version tries to bincount over the raveled
    volume and asks for ~4 GB on a scan this size.  Components are counted from
    the label array instead, which is bounded by the component count.
    """
    lab, n = ndi.label(mask)                      # default = face connectivity
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= min_size                      # matches "remove smaller than"
    keep[0] = False                               # label 0 is background
    return keep[lab]


def segment_volume(vol, verbose=True):
    """raw volume -> clean boolean metal mask (identical to the validated mask)."""
    t = otsu_threshold(vol)
    mask = vol >= t
    if verbose:
        print(f"    Otsu threshold {t:.0f}; metal {100*mask.mean():.2f}% of volume",
              flush=True)
    before = int(mask.sum())
    mask = drop_small_objects(mask, MIN_SPECKLE)
    gc.collect()
    mask = ndi.binary_closing(mask, iterations=CLOSE_ITERS)
    if verbose:
        print(f"    cleanup: {before:,} -> {int(mask.sum()):,} metal voxels "
              f"(speckle < {MIN_SPECKLE} vox removed, pinholes closed)", flush=True)
    return mask, t
