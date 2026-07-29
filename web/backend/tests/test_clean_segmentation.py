from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import tifffile


SCRIPT = Path(__file__).resolve().parents[3] / "analysis" / "clean_segmentation.py"
SPEC = importlib.util.spec_from_file_location("clean_segmentation", SCRIPT)
assert SPEC and SPEC.loader
clean_segmentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean_segmentation)


def test_clean_tiff_streams_slabs_and_preserves_crossing_structure(tmp_path):
    source = tmp_path / "segmented.tif"
    output = tmp_path / "clean.tif"
    volume = np.zeros((9, 32, 32), dtype=np.uint8)
    volume[:, 15:18, 15:18] = 255  # real structure crossing every slab boundary
    volume[3, 2, 2] = 255  # isolated noise
    tifffile.imwrite(source, volume, photometric="minisblack")

    before, after = clean_segmentation.clean_tiff(
        source, output, min_size=4, chunk_depth=3
    )

    cleaned = tifffile.imread(output) > 0
    assert before == int((volume > 0).sum())
    assert after == int(cleaned.sum())
    assert cleaned[:, 16, 16].all()
    assert not cleaned[3, 2, 2]
    assert cleaned.shape == volume.shape


def test_clean_tiff_rejects_overwriting_input(tmp_path):
    source = tmp_path / "segmented.tif"
    tifffile.imwrite(source, np.zeros((2, 8, 8), dtype=np.uint8))

    try:
        clean_segmentation.clean_tiff(source, source)
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("expected ValueError")
