#!/usr/bin/env python3
"""Command-line interface for automatic binary TIFF stack tilt correction."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiff_tilt import correct_tiff_tilt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate pitch and roll from repeating internal lattice lines "
            "and apply the opposite correction to a binary TIFF stack."
        )
    )
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("output_tiff", type=Path)
    parser.add_argument(
        "--max-angle",
        type=float,
        default=5.0,
        help="maximum absolute pitch or roll to search, in degrees (default: 5)",
    )
    parser.add_argument(
        "--middle-fraction",
        type=float,
        default=1 / 3,
        help="centered fraction of Z slices used for estimation (default: 1/3)",
    )
    parser.add_argument("--voxel-size-z", type=float, default=1.0)
    parser.add_argument("--voxel-size-y", type=float, default=1.0)
    parser.add_argument("--voxel-size-x", type=float, default=1.0)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional path for a JSON correction report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = correct_tiff_tilt(
        args.input_tiff,
        args.output_tiff,
        max_angle=args.max_angle,
        middle_fraction=args.middle_fraction,
        voxel_spacing=(
            args.voxel_size_z,
            args.voxel_size_y,
            args.voxel_size_x,
        ),
    )
    report = result.to_json()
    print(report)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
