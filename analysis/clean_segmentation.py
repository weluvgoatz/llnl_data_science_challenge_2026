"""Clean a binary segmentation TIFF without loading the volume into RAM.

The challenge volume contains more than 500 million voxels.  Labelling that
entire volume with ``remove_small_objects`` needs several gigabytes for the
label image alone and can make WSL swap until it is effectively unresponsive.
This implementation processes overlapping slabs and streams compressed pages
to an atomic output file.

Usage (from repo root):
    python analysis/clean_segmentation.py --mask "path/to/..._segmented.tif"
    python analysis/clean_segmentation.py --mask "..." --min-size 25
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def _remove_small_interior_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove small components unless they touch a slab's Z boundary.

    Boundary components may continue into the adjacent slab, so retaining them
    avoids cutting real struts.  The tradeoff is that a tiny speck crossing a
    slab boundary can survive; this is preferable to corrupting lattice
    topology and still removes essentially all scattered noise.
    """
    if min_size <= 1 or not mask.any():
        return mask

    labels, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    keep[np.unique(labels[0])] = True
    keep[np.unique(labels[-1])] = True
    keep[0] = False
    return keep[labels]


def clean_tiff(
    mask_path: Path,
    out_path: Path,
    *,
    min_size: int = 20,
    chunk_depth: int = 16,
) -> tuple[int, int]:
    """Clean ``mask_path`` into ``out_path`` with bounded peak memory."""
    if chunk_depth < 1:
        raise ValueError("chunk_depth must be at least 1")
    if mask_path.resolve() == out_path.resolve():
        raise ValueError("input and output paths must be different")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    before = 0
    after = 0

    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{out_path.name}.", suffix=".partial",
            dir=out_path.parent, delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        with tifffile.TiffFile(mask_path) as source:
            pages = source.pages
            depth = len(pages)
            if depth == 0:
                raise ValueError("input TIFF contains no pages")

            first_shape = pages[0].shape
            if len(first_shape) != 2:
                raise ValueError("expected a TIFF stack of 2-D pages")
            use_bigtiff = depth * int(np.prod(first_shape)) >= 4_000_000_000

            # One-slice halos make the 3-D closing correct at slab seams.
            with tifffile.TiffWriter(temporary_path, bigtiff=use_bigtiff) as destination:
                for start in range(0, depth, chunk_depth):
                    stop = min(start + chunk_depth, depth)
                    read_start = max(0, start - 1)
                    read_stop = min(depth, stop + 1)
                    slab = np.stack(
                        [pages[index].asarray() > 0
                         for index in range(read_start, read_stop)]
                    )
                    before += int(slab[start - read_start:stop - read_start].sum())

                    slab = _remove_small_interior_components(slab, min_size)
                    # Replicate the physical first/last slice while closing so
                    # erosion does not erase structure merely because it
                    # reaches the top or bottom of the acquired volume.
                    pad_before = int(read_start == 0)
                    pad_after = int(read_stop == depth)
                    if pad_before or pad_after:
                        slab = np.pad(
                            slab,
                            ((pad_before, pad_after), (0, 0), (0, 0)),
                            mode="edge",
                        )
                    slab = ndi.binary_closing(slab, iterations=1)
                    if pad_before or pad_after:
                        slab = slab[pad_before:len(slab) - pad_after or None]
                    core = slab[start - read_start:stop - read_start]
                    after += int(core.sum())

                    for frame in core:
                        output_frame = frame.astype(np.uint8)
                        np.multiply(output_frame, 255, out=output_frame)
                        destination.write(
                            output_frame,
                            photometric="minisblack",
                            compression="zlib",
                            metadata=None,
                        )

        temporary_path.replace(out_path)
        temporary_path = None
        return before, after
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True, help="Input binary segmentation TIFF")
    parser.add_argument("--out", default=None, help="Output TIFF (default: <name>_clean.tif)")
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument(
        "--chunk-depth", type=int, default=16,
        help="Slices per slab; lower this if memory is constrained (default: 16)",
    )
    args = parser.parse_args()

    mask_path = Path(args.mask).expanduser().resolve()
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else mask_path.with_name(mask_path.stem + "_clean.tif")
    )

    print(
        f"Cleaning {mask_path.name} in {args.chunk_depth}-slice slabs "
        f"(components < {args.min_size} voxels) ...",
        flush=True,
    )
    before, after = clean_tiff(
        mask_path, out_path,
        min_size=args.min_size,
        chunk_depth=args.chunk_depth,
    )
    print(f"  foreground voxels: {before:,} -> {after:,}", flush=True)
    print(
        f"Saved cleaned TIFF: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
