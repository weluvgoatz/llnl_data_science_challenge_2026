from fastmcp import FastMCP
from pathlib import Path
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as skimage_skeletonize
from skimage.filters import threshold_otsu
import tifffile
from skeletonization import skeletonize_mask

# Initialize the MCP server
mcp = FastMCP("CT Segmentation")

@mcp.tool()
def segment_ct_dataset(input_filepath: str, output_filepath: str, threshold: float) -> str:
    """
    Segments a 3D CT dataset based on a given density threshold value.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT scan data.
        output_filepath: Path indicating where the segmented .npy file should be saved.
        threshold: The density value to use as a threshold. Voxels >= threshold will be set to 1, others to 0.
    
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = Path(input_filepath)
        output_path = Path(output_filepath)

        if not input_path.is_file():
            return f"Error: input file not found: {input_path}"

        ct_volume = np.load(input_path)

        if ct_volume.ndim != 3:
            return (
                f"Error: expected a 3D CT volume, but received "
                f"{ct_volume.ndim}D data with shape {ct_volume.shape}."
            )

        # uint8 mask: 1 for foreground, 0 for background
        segmented_volume = (ct_volume >= threshold).astype(np.uint8)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, segmented_volume)

        foreground_voxels = int(segmented_volume.sum())
        return (
            f"Segmentation complete. Saved mask to {output_path}. "
            f"Foreground voxels: {foreground_voxels} / {segmented_volume.size}."
        )

    except (OSError, ValueError) as exc:
        return f"Error segmenting CT dataset: {exc}"

@mcp.tool()
def segment_tiff(
    input_filepath: str,
    output_filepath: str,
    threshold: float = 140,
    adaptive: bool = False,
    smoothing_window: int = 11,
) -> str:
    """
    Segment a TIFF stack one page at a time to keep memory use bounded.

    In fixed mode (the default), every page uses ``threshold``. In adaptive
    mode, an initial Otsu threshold is measured independently for each page,
    then the threshold sequence is median-smoothed through the stack. This
    adapts to gradual brightness changes without allowing noisy slices to make
    abrupt threshold jumps. ``threshold`` is ignored in adaptive mode.

    Adaptive mode makes two streaming passes over the source TIFF. Only the
    per-page thresholds and one image page are held in memory. Output is
    written to a temporary file and moved into place only after every page has
    been processed successfully.

    Args:
        input_filepath: Path to the source TIFF stack.
        output_filepath: Path for the binary uint8 TIFF stack.
        threshold: Fixed density threshold used when adaptive is false.
        adaptive: Dynamically estimate and regularize a threshold per slice.
        smoothing_window: Odd number of neighboring slice thresholds used by
            the adaptive median filter. Use 1 to disable smoothing.
    """
    input_path = Path(input_filepath)
    output_path = Path(output_filepath)
    temporary_path = None

    try:
        if not input_path.is_file():
            return f"Error: input file not found: {input_path}"

        if input_path.resolve() == output_path.resolve():
            return "Error: input and output paths must be different."

        if smoothing_window < 1 or smoothing_window % 2 == 0:
            return "Error: smoothing_window must be a positive odd integer."

        page_thresholds = None
        if adaptive:
            raw_thresholds = []
            with tifffile.TiffFile(input_path) as source:
                for page in source.pages:
                    frame = page.asarray().astype(np.int32) - 32768
                    if frame.size == 0:
                        raise ValueError("input TIFF contains an empty image page")

                    # A constant page has no separable intensity classes; its
                    # sole value is the least surprising threshold to carry
                    # into the through-stack smoothing step.
                    if frame.min() == frame.max():
                        raw_thresholds.append(float(frame.flat[0]))
                    else:
                        raw_thresholds.append(float(threshold_otsu(frame)))
                    del frame

            if not raw_thresholds:
                raise ValueError("input TIFF contains no image pages")

            radius = smoothing_window // 2
            padded_thresholds = np.pad(raw_thresholds, radius, mode="edge")
            page_thresholds = []
            for index in range(len(raw_thresholds)):
                page_thresholds.append(
                    float(np.median(padded_thresholds[index:index + smoothing_window]))
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".partial",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        foreground_pixels = 0
        total_pixels = 0
        page_count = 0

        with tifffile.TiffFile(input_path) as source:
            with tifffile.TiffWriter(temporary_path, bigtiff=False) as destination:
                for page_index, page in enumerate(source.pages):
                    frame = page.asarray().astype(np.int32) - 32768
                    page_threshold = (
                        page_thresholds[page_index] if adaptive else threshold
                    )
                    segmented_frame = (frame > page_threshold).astype(np.uint8)

                    foreground_pixels += int(segmented_frame.sum())
                    total_pixels += int(segmented_frame.size)
                    page_count += 1
                    np.multiply(segmented_frame, 255, out=segmented_frame)

                    destination.write(
                        segmented_frame,
                        photometric="minisblack",
                        metadata=None,
                    )

                    del frame, segmented_frame

        if page_count == 0:
            raise ValueError("input TIFF contains no image pages")

        temporary_path.replace(output_path)
        temporary_path = None

        mode_summary = (
            "Adaptive Otsu thresholds "
            f"(median window {smoothing_window}, range "
            f"{min(page_thresholds):.3f} to {max(page_thresholds):.3f}). "
            if adaptive
            else f"Fixed threshold {threshold}. "
        )
        return (
            f"Segmentation complete. Saved {page_count} pages to {output_path}. "
            f"{mode_summary}"
            f"Foreground pixels: {foreground_pixels} / {total_pixels}."
        )

    except Exception as exc:
        return f"Error segmenting TIFF: {type(exc).__name__}: {exc}"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

@mcp.tool()
def visualize_slice(input_filepath: str, output_filepath: str, slice_index: int, axis: int = 0) -> str:
    """
    Loads a 3D CT dataset from a .npy file and saves a visualization of a specific slice to an image file.
    
    Args:
        input_filepath: Path to the input .npy file containing the 3D CT data.
        output_filepath: Path indicating where the output image should be saved (e.g., .png).
        slice_index: The index of the slice to visualize.
        axis: The axis along which to take the slice (0, 1, or 2). Default is 0.
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = Path(input_filepath)
        output_path = Path(output_filepath)

        if not input_path.is_file():
            return f"Error: input file not found: {input_path}"

        if axis not in (0, 1, 2):
            return "Error: axis must be 0, 1, or 2."

        volume = np.load(input_path)

        if volume.ndim != 3:
            return (
                f"Error: expected a 3D dataset, but received "
                f"{volume.ndim}D data with shape {volume.shape}."
            )

        if not 0 <= slice_index < volume.shape[axis]:
            return (
                f"Error: slice_index {slice_index} is out of bounds for "
                f"axis {axis}, which has {volume.shape[axis]} slices."
            )

        # Extract the requested 2D slice.
        slice_2d = np.take(volume, slice_index, axis=axis)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Render the complete 256-by-256 unit-cell slice at a practical image
        # size.  A 256-inch figure produces a needlessly large raster and can
        # obscure the spatial scale of an individual lattice cell.
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150, constrained_layout=True)
        ax.imshow(
            slice_2d,
            cmap="gray",
            interpolation="nearest",
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="equal",
        )
        ax.set_title(f"CT slice {slice_index} (axis {axis})")
        ax.set_xlabel("Voxel index")
        ax.set_ylabel("Voxel index")
        ax.set_xlim(-0.5, slice_2d.shape[1] - 0.5)
        ax.set_ylim(-0.5, slice_2d.shape[0] - 0.5)
        fig.savefig(output_path)
        plt.close(fig)

        return f"Visualization saved successfully to: {output_path}"

    except (OSError, ValueError) as exc:
        return f"Error visualizing CT slice: {exc}"

@mcp.tool()
def skeletonize(input_filepath: str, output_filepath: str) -> str:   
    """
    Creates a skeleton from a 3D segmentation mask.
    Args:
        input_filepath: Path to the .npy file containing the 3D mask.
        output_filepath: Path to save the extracted skeleton (.npy).
        
    Returns:
        A status message indicating success and the save location, or an error message.
    """
    try:
        input_path = Path(input_filepath)
        output_path = Path(output_filepath)

        if not input_path.is_file():
            return f"Error: input file not found: {input_path}"

        mask = np.load(input_path)

        if mask.ndim != 3:
            return (
                f"Error: expected a 3D segmentation mask, but received "
                f"{mask.ndim}D data with shape {mask.shape}."
            )

        if not np.any(mask):
            return "Error: the input mask contains no foreground voxels."

        skeletonize_mask(input_path, output_path)

    except (OSError, ValueError, TypeError) as exc:
        return f"Error skeletonizing mask: {exc}"

if __name__ == "__main__":
    # Run the FastMCP server, exposing the tools over standard I/O (default)
    mcp.run()
