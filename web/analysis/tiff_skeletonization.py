"""Streaming, per-slice skeletonization for segmented TIFF stacks."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from skimage.morphology import skeletonize
import tifffile


def skeletonize_tiff_slices(
    input_filepath: str | Path,
    output_filepath: str | Path,
) -> dict:
    """Skeletonize each page of a segmented TIFF and write an 8-bit TIFF.

    Any nonzero input pixel is treated as foreground. Pages are processed
    independently and streamed to disk, bounding memory usage to a few image
    slices. The destination is replaced only after the full stack succeeds.
    """
    input_path = Path(input_filepath).expanduser().resolve()
    output_path = Path(output_filepath).expanduser().resolve()
    temporary_path: Path | None = None

    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("input_filepath must end in .tif or .tiff")
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("output_filepath must end in .tif or .tiff")
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if input_path == output_path:
        raise ValueError("input and output paths must be different")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".partial",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        page_count = 0
        foreground_pixels = 0
        skeleton_pixels = 0

        with tifffile.TiffFile(input_path) as source:
            with tifffile.TiffWriter(temporary_path, bigtiff=True) as destination:
                for page in source.pages:
                    frame = page.asarray()
                    if frame.ndim != 2:
                        raise ValueError(
                            "every TIFF page must be 2D; "
                            f"page {page_count} has shape {frame.shape}"
                        )
                    if frame.size == 0:
                        raise ValueError(f"TIFF page {page_count} is empty")

                    mask = frame != 0
                    skeleton = skeletonize(mask)
                    foreground_pixels += int(np.count_nonzero(mask))
                    skeleton_pixels += int(np.count_nonzero(skeleton))

                    destination.write(
                        skeleton.astype(np.uint8) * 255,
                        photometric="minisblack",
                        metadata=None,
                    )
                    page_count += 1

        if page_count == 0:
            raise ValueError("input TIFF contains no image pages")

        temporary_path.replace(output_path)
        temporary_path = None
        return {
            "status": "success",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "pages": page_count,
            "foreground_pixels": foreground_pixels,
            "skeleton_pixels": skeleton_pixels,
            "output_encoding": "uint8 (background=0, skeleton=255)",
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
