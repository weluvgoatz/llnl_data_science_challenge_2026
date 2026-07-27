"""Estimate and correct small out-of-plane tilts in binary TIFF stacks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import tifffile


ANGLE_RESOLUTION_DEGREES = 0.01
MAX_ANGLE_DISPERSION_DEGREES = 0.25
MAX_VERIFIED_RESIDUAL_DEGREES = 0.1
MIN_DISTINCT_LINES = 4


@dataclass(frozen=True)
class AxisEstimate:
    """Tilt estimate and supporting Hough-line statistics for one projection."""

    projection: str
    angle_degrees: float
    line_count: int
    angular_mad_degrees: float
    peak_support: float


@dataclass(frozen=True)
class TiltEstimate:
    """Estimated pitch and roll for a TIFF stack."""

    zy: AxisEstimate
    zx: AxisEstimate
    analyzed_slice_start: int
    analyzed_slice_stop: int
    shape: tuple[int, int, int]
    dtype: str
    background_value: int | float | bool
    foreground_value: int | float | bool


@dataclass(frozen=True)
class TiltCorrectionResult:
    """Machine-readable result from a successful correction."""

    input_path: str
    output_path: str
    shape: tuple[int, int, int]
    dtype: str
    estimated_zy_degrees: float
    estimated_zx_degrees: float
    applied_zy_degrees: float
    applied_zx_degrees: float
    residual_zy_degrees: float
    residual_zx_degrees: float
    middle_fraction: float
    voxel_spacing: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _validate_controls(
    max_angle: float,
    middle_fraction: float,
    voxel_spacing: Sequence[float],
) -> tuple[float, float, tuple[float, float, float]]:
    if not math.isfinite(max_angle) or not 0 < max_angle <= 45:
        raise ValueError("max_angle must be finite and in (0, 45]")
    if not math.isfinite(middle_fraction) or not 0 < middle_fraction <= 1:
        raise ValueError("middle_fraction must be finite and in (0, 1]")
    if len(voxel_spacing) != 3:
        raise ValueError("voxel_spacing must contain Z, Y, and X sizes")
    spacing = tuple(float(value) for value in voxel_spacing)
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("voxel spacing values must be finite and positive")
    return float(max_angle), float(middle_fraction), spacing


def _python_scalar(value: np.generic | int | float | bool) -> int | float | bool:
    return value.item() if isinstance(value, np.generic) else value


def _inspect_binary_tiff(
    input_path: Path,
    middle_fraction: float,
) -> tuple[
    tuple[int, int, int],
    np.dtype,
    tuple[int | float | bool, int | float | bool],
    int,
    int,
    dict[str, object],
]:
    values: set[int | float | bool] = set()
    page_shape: tuple[int, int] | None = None
    dtype: np.dtype | None = None

    with tifffile.TiffFile(input_path) as source:
        page_count = len(source.pages)
        if page_count < 2:
            raise ValueError("input must be a multi-page 3D TIFF stack")

        first_page = source.pages[0]
        tags = first_page.tags
        metadata: dict[str, object] = {
            "photometric": first_page.photometric.name,
            "compression": first_page.compression.name,
            "axes": source.series[0].axes,
        }
        for tag_name in ("XResolution", "YResolution", "ResolutionUnit"):
            tag = tags.get(tag_name)
            if tag is not None:
                metadata[tag_name] = tag.value

        for page_index, page in enumerate(source.pages):
            frame = page.asarray()
            if frame.ndim != 2:
                raise ValueError(
                    f"page {page_index} is not two-dimensional: {frame.shape}"
                )
            if not (
                np.issubdtype(frame.dtype, np.number)
                or np.issubdtype(frame.dtype, np.bool_)
            ):
                raise ValueError(f"unsupported TIFF dtype: {frame.dtype}")
            if page_shape is None:
                page_shape = frame.shape
                dtype = frame.dtype
            elif frame.shape != page_shape:
                raise ValueError(
                    f"page {page_index} has shape {frame.shape}; expected {page_shape}"
                )
            elif frame.dtype != dtype:
                raise ValueError(
                    f"page {page_index} has dtype {frame.dtype}; expected {dtype}"
                )

            unique = np.unique(frame)
            values.update(_python_scalar(value) for value in unique)
            if len(values) > 2:
                raise ValueError(
                    "input must be a binary mask with no more than two values"
                )

    if page_shape is None or dtype is None:
        raise ValueError("input TIFF contains no readable pages")
    if len(values) != 2:
        raise ValueError("input mask must contain both background and foreground")

    sorted_values = tuple(sorted(values))
    analyzed_count = max(8, int(round(page_count * middle_fraction)))
    analyzed_count = min(page_count, analyzed_count)
    start = (page_count - analyzed_count) // 2
    stop = start + analyzed_count
    return (
        (page_count, page_shape[0], page_shape[1]),
        np.dtype(dtype),
        (sorted_values[0], sorted_values[1]),
        start,
        stop,
        metadata,
    )


def _make_projections(
    input_path: Path,
    foreground_value: int | float | bool,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    zy_rows: list[np.ndarray] = []
    zx_rows: list[np.ndarray] = []
    with tifffile.TiffFile(input_path) as source:
        for page_index in range(start, stop):
            foreground = source.pages[page_index].asarray() == foreground_value
            zy_rows.append(
                np.count_nonzero(foreground, axis=1).astype(np.uint32)
            )
            zx_rows.append(
                np.count_nonzero(foreground, axis=0).astype(np.uint32)
            )
    return np.asarray(zy_rows), np.asarray(zx_rows)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2, side="left"))
    return float(sorted_values[index])


def _estimate_projection_angle(
    projection: np.ndarray,
    projection_name: str,
    max_angle_degrees: float,
    z_spacing: float,
    lateral_spacing: float,
) -> AxisEstimate:
    image = projection.astype(np.float32, copy=False)
    low, high = np.percentile(image, (1.0, 99.5))
    if not high > low:
        raise ValueError(
            f"{projection_name} projection has no usable intensity contrast"
        )
    normalized = np.clip((image - low) / (high - low), 0, 1)
    edges = canny(
        normalized,
        sigma=1.5,
        low_threshold=0.08,
        high_threshold=0.20,
    )
    if np.count_nonzero(edges) < 100:
        raise ValueError(
            f"{projection_name} projection contains too few line edges"
        )

    max_index_angle = math.degrees(
        math.atan(
            math.tan(math.radians(max_angle_degrees))
            * lateral_spacing
            / z_spacing
        )
    )
    sample_count = max(
        101,
        int(math.ceil(2 * max_index_angle / ANGLE_RESOLUTION_DEGREES)) + 1,
    )
    direction_angles = np.linspace(
        -max_index_angle,
        max_index_angle,
        sample_count,
    )
    normal_angles = np.deg2rad(direction_angles - 90.0)
    accumulator, angles, distances = hough_line(edges, theta=normal_angles)
    peak_values, peak_angles, peak_distances = hough_line_peaks(
        accumulator,
        angles,
        distances,
        threshold=0.45 * float(accumulator.max()),
        min_distance=4,
        min_angle=5,
        num_peaks=64,
    )
    if len(peak_values) < MIN_DISTINCT_LINES:
        raise ValueError(
            f"{projection_name} projection yielded only {len(peak_values)} "
            f"distinct lines; at least {MIN_DISTINCT_LINES} are required"
        )

    index_directions = np.rad2deg(peak_angles) + 90.0
    physical_directions = np.rad2deg(
        np.arctan(
            np.tan(np.deg2rad(index_directions))
            * z_spacing
            / lateral_spacing
        )
    )
    all_weights = peak_values.astype(float)
    # Repeated lattice lines produce a tight angular mode. Choose that mode
    # explicitly so weaker diagonal intersections and clipped edge fragments
    # cannot drag a global median away from the actual repeated family.
    cluster_radius = 0.30
    # Anchor the family on the strongest full-length line. Selecting the
    # largest cluster by count can favor numerous short, clipped fragments.
    mode_center = physical_directions[int(np.argmax(all_weights))]
    selected = np.abs(physical_directions - mode_center) <= cluster_radius
    if np.count_nonzero(selected) < MIN_DISTINCT_LINES:
        raise ValueError(
            f"{projection_name} projection has no repeated angular mode with "
            f"at least {MIN_DISTINCT_LINES} distinct lines"
        )
    directions = physical_directions[selected]
    weights = all_weights[selected]
    angle = _weighted_median(directions, weights)
    deviations = np.abs(directions - angle)
    angular_mad = _weighted_median(deviations, weights)
    if angular_mad > MAX_ANGLE_DISPERSION_DEGREES:
        raise ValueError(
            f"{projection_name} line angles are inconsistent "
            f"(weighted MAD {angular_mad:.3f} degrees)"
        )

    return AxisEstimate(
        projection=projection_name,
        angle_degrees=angle,
        line_count=int(np.count_nonzero(selected)),
        angular_mad_degrees=angular_mad,
        peak_support=float(weights.sum() / all_weights.sum()),
    )


def estimate_tilt(
    input_path: str | Path,
    max_angle: float = 5.0,
    middle_fraction: float = 1 / 3,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> TiltEstimate:
    """Estimate Z-Y and Z-X tilt from internal lines in a binary TIFF stack."""

    max_angle, middle_fraction, spacing = _validate_controls(
        max_angle, middle_fraction, voxel_spacing
    )
    path = Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"input TIFF does not exist: {path}")
    if path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("input path must end in .tif or .tiff")

    shape, dtype, values, start, stop, _ = _inspect_binary_tiff(
        path, middle_fraction
    )
    zy_projection, zx_projection = _make_projections(
        path, values[1], start, stop
    )
    zy = _estimate_projection_angle(
        zy_projection, "Z-Y", max_angle, spacing[0], spacing[1]
    )
    zx = _estimate_projection_angle(
        zx_projection, "Z-X", max_angle, spacing[0], spacing[2]
    )
    return TiltEstimate(
        zy=zy,
        zx=zx,
        analyzed_slice_start=start,
        analyzed_slice_stop=stop,
        shape=shape,
        dtype=str(dtype),
        background_value=values[0],
        foreground_value=values[1],
    )


def _forward_rotation_matrix(
    zy_angle_degrees: float,
    zx_angle_degrees: float,
) -> np.ndarray:
    zy = math.radians(zy_angle_degrees)
    zx = math.radians(zx_angle_degrees)
    c_zy, s_zy = math.cos(zy), math.sin(zy)
    c_zx, s_zx = math.cos(zx), math.sin(zx)
    rotate_zy = np.array(
        [[c_zy, -s_zy, 0], [s_zy, c_zy, 0], [0, 0, 1]],
        dtype=float,
    )
    rotate_zx = np.array(
        [[c_zx, 0, -s_zx], [0, 1, 0], [s_zx, 0, c_zx]],
        dtype=float,
    )
    return rotate_zx @ rotate_zy


def _copy_tiff_to_memmap(
    input_path: Path,
    array_path: Path,
    shape: tuple[int, int, int],
    dtype: np.dtype,
) -> np.memmap:
    try:
        mapped = tifffile.memmap(input_path)
        if mapped.shape == shape and mapped.dtype == dtype:
            return mapped
    except (ValueError, OSError):
        pass

    mapped = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=dtype, shape=shape
    )
    with tifffile.TiffFile(input_path) as source:
        for index, page in enumerate(source.pages):
            mapped[index] = page.asarray()
    mapped.flush()
    return mapped


def _resolution_as_float(value: object) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _write_tiff(
    path: Path,
    volume: np.ndarray,
    metadata: dict[str, object],
) -> None:
    compression_name = str(metadata.get("compression", "NONE"))
    compression = {
        "NONE": None,
        "ADOBE_DEFLATE": "zlib",
        "DEFLATE": "zlib",
        "PACKBITS": "packbits",
        "LZW": "lzw",
    }.get(compression_name)

    kwargs: dict[str, object] = {
        "photometric": "minisblack",
        "metadata": {"axes": "ZYX"},
        "compression": compression,
        "bigtiff": volume.nbytes >= 2**32 - 2**25,
    }
    if "XResolution" in metadata and "YResolution" in metadata:
        kwargs["resolution"] = (
            _resolution_as_float(metadata["XResolution"]),
            _resolution_as_float(metadata["YResolution"]),
        )
    resolution_unit = metadata.get("ResolutionUnit")
    if resolution_unit is not None:
        kwargs["resolutionunit"] = resolution_unit
    tifffile.imwrite(path, volume, **kwargs)


def _verified(initial: float, residual: float) -> bool:
    if abs(residual) > MAX_VERIFIED_RESIDUAL_DEGREES:
        return False
    if abs(initial) <= MAX_VERIFIED_RESIDUAL_DEGREES:
        return abs(residual) <= MAX_VERIFIED_RESIDUAL_DEGREES
    return abs(residual) < abs(initial)


def correct_tiff_tilt(
    input_path: str | Path,
    output_path: str | Path,
    max_angle: float = 5.0,
    middle_fraction: float = 1 / 3,
    voxel_spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> TiltCorrectionResult:
    """Estimate tilt, correct it once in 3D, and verify the saved result."""

    max_angle, middle_fraction, spacing = _validate_controls(
        max_angle, middle_fraction, voxel_spacing
    )
    source_path = Path(input_path).expanduser().resolve()
    destination_path = Path(output_path).expanduser().resolve()
    if source_path == destination_path:
        raise ValueError("input and output paths must be different")
    if destination_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("output path must end in .tif or .tiff")

    initial = estimate_tilt(
        source_path,
        max_angle=max_angle,
        middle_fraction=middle_fraction,
        voxel_spacing=spacing,
    )
    shape, dtype, _, _, _, metadata = _inspect_binary_tiff(
        source_path, middle_fraction
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    partial_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{destination_path.stem}-tilt-",
            dir=destination_path.parent,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            source_volume = _copy_tiff_to_memmap(
                source_path,
                temporary_root / "source.npy",
                shape,
                dtype,
            )
            corrected_volume = np.lib.format.open_memmap(
                temporary_root / "corrected.npy",
                mode="w+",
                dtype=dtype,
                shape=shape,
            )

            physical_rotation = _forward_rotation_matrix(
                initial.zy.angle_degrees,
                initial.zx.angle_degrees,
            )
            scale = np.diag(spacing)
            forward_index = (
                np.linalg.inv(scale) @ physical_rotation @ scale
            )
            inverse_index = np.linalg.inv(forward_index)
            center = (np.asarray(shape, dtype=float) - 1.0) / 2.0
            offset = center - inverse_index @ center
            ndi.affine_transform(
                source_volume,
                inverse_index,
                offset=offset,
                output_shape=shape,
                output=corrected_volume,
                order=0,
                mode="constant",
                cval=initial.background_value,
                prefilter=False,
            )
            corrected_volume.flush()

            partial_path = temporary_root / f"corrected{destination_path.suffix}"
            _write_tiff(partial_path, corrected_volume, metadata)
            residual = estimate_tilt(
                partial_path,
                max_angle=max_angle,
                middle_fraction=middle_fraction,
                voxel_spacing=spacing,
            )
            if not _verified(
                initial.zy.angle_degrees, residual.zy.angle_degrees
            ) or not _verified(
                initial.zx.angle_degrees, residual.zx.angle_degrees
            ):
                raise RuntimeError(
                    "correction verification failed: "
                    f"initial Z-Y={initial.zy.angle_degrees:.3f}, "
                    f"residual Z-Y={residual.zy.angle_degrees:.3f}, "
                    f"initial Z-X={initial.zx.angle_degrees:.3f}, "
                    f"residual Z-X={residual.zx.angle_degrees:.3f} degrees"
                )

            os.replace(partial_path, destination_path)
            partial_path = None

        return TiltCorrectionResult(
            input_path=str(source_path),
            output_path=str(destination_path),
            shape=shape,
            dtype=str(dtype),
            estimated_zy_degrees=initial.zy.angle_degrees,
            estimated_zx_degrees=initial.zx.angle_degrees,
            applied_zy_degrees=initial.zy.angle_degrees,
            applied_zx_degrees=initial.zx.angle_degrees,
            residual_zy_degrees=residual.zy.angle_degrees,
            residual_zx_degrees=residual.zx.angle_degrees,
            middle_fraction=middle_fraction,
            voxel_spacing=spacing,
        )
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
