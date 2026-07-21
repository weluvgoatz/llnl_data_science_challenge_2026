from fastmcp import FastMCP
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize as skimage_skeletonize
from skeletonization.py import skeletonize_mask

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
