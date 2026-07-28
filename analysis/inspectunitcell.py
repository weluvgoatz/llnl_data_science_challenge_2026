from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu



# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "unitcell"
    / "unitcell.npy"
)

OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "analysis"
    / "outputs"
)

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)



# ---------------------------------------------------------
# Load volume
# ---------------------------------------------------------

volume = np.load(INPUT_PATH, mmap_mode="r")

print("Input file:", INPUT_PATH)
print("Shape:", volume.shape)
print("Datatype:", volume.dtype)
print("Number of voxels:", volume.size)
print("Memory size (MiB):", volume.nbytes / 1024**2)



# ---------------------------------------------------------
# Intensity statistics
# ---------------------------------------------------------

minimum = float(volume.min())
maximum = float(volume.max())
mean = float(volume.mean())
standard_deviation = float(volume.std())

percentile_values = [
    0,
    0.1,
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
    99.5,
    99.9,
    100,
]

percentiles = np.percentile(volume, percentile_values)

print("\nIntensity statistics")
print("--------------------")
print(f"Minimum:            {minimum:.8f}")
print(f"Maximum:            {maximum:.8f}")
print(f"Mean:               {mean:.8f}")
print(f"Standard deviation: {standard_deviation:.8f}")

print("\nPercentiles")
print("-----------")

for percentile, value in zip(percentile_values, percentiles):
    print(f"{percentile:6.1f}%: {value:.8f}")



# ---------------------------------------------------------
# Calculate Otsu threshold
# ---------------------------------------------------------

otsu_threshold = float(threshold_otsu(volume))

# Compare Otsu with slightly more permissive and restrictive
# thresholds.
low_threshold = otsu_threshold * 0.80
high_threshold = otsu_threshold * 1.20

thresholds = {
    "Low threshold": low_threshold,
    "Otsu threshold": otsu_threshold,
    "High threshold": high_threshold,
}

print("\nCandidate thresholds")
print("--------------------")

for name, threshold in thresholds.items():
    foreground_fraction = float(np.mean(volume >= threshold))
    foreground_voxels = int(np.count_nonzero(volume >= threshold))

    print(
        f"{name:16s}: "
        f"{threshold:.8f}, "
        f"foreground={foreground_fraction:.4%}, "
        f"voxels={foreground_voxels:,}"
    )



# ---------------------------------------------------------
# Save histogram
# ---------------------------------------------------------

# Ignore extreme outliers when choosing display limits.
histogram_minimum = float(np.percentile(volume, 0.1))
histogram_maximum = float(np.percentile(volume, 99.9))

figure, axis = plt.subplots(figsize=(10, 6))

axis.hist(
    np.asarray(volume).ravel(),
    bins=512,
    range=(histogram_minimum, histogram_maximum),
    color="steelblue",
    alpha=0.85,
)

colors = {
    "Low threshold": "orange",
    "Otsu threshold": "red",
    "High threshold": "green",
}

for name, threshold in thresholds.items():
    axis.axvline(
        threshold,
        color=colors[name],
        linestyle="--",
        linewidth=2,
        label=f"{name}: {threshold:.6f}",
    )

axis.set_title("Unit-cell CT intensity distribution")
axis.set_xlabel("Voxel intensity")
axis.set_ylabel("Voxel count")
axis.legend()
axis.grid(alpha=0.2)

figure.tight_layout()

histogram_path = (
    OUTPUT_DIRECTORY
    / "unitcell_intensity_histogram.png"
)

figure.savefig(histogram_path, dpi=200)
plt.close(figure)

print("\nSaved histogram:", histogram_path)



# ---------------------------------------------------------
# Compare thresholds along all three axes
# ---------------------------------------------------------

center_indices = [
    volume.shape[0] // 2,
    volume.shape[1] // 2,
    volume.shape[2] // 2,
]



def extract_slice(array, axis, index):
    """Extract one 2D slice along a selected axis."""

    if axis == 0:
        return array[index, :, :]

    if axis == 1:
        return array[:, index, :]

    if axis == 2:
        return array[:, :, index]

    raise ValueError("Axis must be 0, 1, or 2.")



figure, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(16, 13),
)

column_titles = [
    "Raw CT",
    f"Low: {low_threshold:.6f}",
    f"Otsu: {otsu_threshold:.6f}",
    f"High: {high_threshold:.6f}",
]

for column, title in enumerate(column_titles):
    axes[0, column].set_title(title)

for axis_number in range(3):
    slice_index = center_indices[axis_number]

    raw_slice = extract_slice(
        volume,
        axis=axis_number,
        index=slice_index,
    )

    low_mask = raw_slice >= low_threshold
    otsu_mask = raw_slice >= otsu_threshold
    high_mask = raw_slice >= high_threshold

    images = [
        raw_slice,
        low_mask,
        otsu_mask,
        high_mask,
    ]

    for column, image in enumerate(images):
        color_map = "gray" if column == 0 else "viridis"

        axes[axis_number, column].imshow(
            image,
            cmap=color_map,
        )

        axes[axis_number, column].axis("off")

    axes[axis_number, 0].set_ylabel(
        f"Axis {axis_number}\nSlice {slice_index}",
        fontsize=12,
    )

figure.suptitle(
    "Unit-cell segmentation threshold comparison",
    fontsize=16,
)

figure.tight_layout()

comparison_path = (
    OUTPUT_DIRECTORY
    / "unitcell_threshold_comparison.png"
)

figure.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close(figure)

print("Saved comparison:", comparison_path)
print("\nInspection completed successfully.")