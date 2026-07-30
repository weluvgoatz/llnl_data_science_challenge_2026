"""Central, specimen-parameterized configuration for the defect-detection pipeline.

Every stage script imports its paths from here, so the whole pipeline can be
pointed at a DIFFERENT scan without editing code — just set environment
variables before running:

    LATTICE_BASE         base file name of the scan (no extension), e.g.
                         "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
    LATTICE_STK          directory holding the .tif stack and all derived files
                         (default: data/missing_struts/tif_stacks)
    LATTICE_DESIGN_JSON  repo-relative path to the defect-free design graph JSON
                         used as the registration target for classification.

Defaults reproduce the 2026 DSC challenge specimen, so running with no env vars
set behaves exactly as before.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"

BASE = os.environ.get("LATTICE_BASE", _DEFAULT_BASE)
STK = Path(os.environ.get("LATTICE_STK", str(ROOT / "data/missing_struts/tif_stacks")))
DESIGN_JSON = ROOT / os.environ.get(
    "LATTICE_DESIGN_JSON",
    f"data/missing_struts/registered_jsons/{_DEFAULT_BASE}.json",
)

# physical constants (challenge specimen; override via env if your scan differs)
VOXEL_UM = float(os.environ.get("LATTICE_VOXEL_UM", "58.1"))
STRUT_DIAMETER_UM = float(os.environ.get("LATTICE_STRUT_DIAMETER_UM", "424.0"))
STRUT_RADIUS_VOX = STRUT_DIAMETER_UM / 2 / VOXEL_UM      # ~3.65 vox

# classification thresholds (analysis/defect_detection/unified_defects_accurate.py).
# Defaults reproduce the values that were previously hardcoded in that script, so
# running with no env vars set behaves exactly as before. Override any of these to
# run a stricter/looser classification pass without editing code.
MISSING_FRAC = float(os.environ.get("LATTICE_MISSING_FRAC", "0.15"))
GAP_FRAC = float(os.environ.get("LATTICE_GAP_FRAC", "0.25"))
THIN_OUTLIER_K = float(os.environ.get("LATTICE_THIN_OUTLIER_K", "3.0"))
BENT_RADIUS_MULT = float(os.environ.get("LATTICE_BENT_RADIUS_MULT", "1.0"))
SNAP_R_VOX = float(os.environ.get("LATTICE_SNAP_R_VOX", "14.0"))
METAL_R_VOX = int(os.environ.get("LATTICE_METAL_R_VOX", "11"))

# convenient derived paths (all scans keep this naming scheme)
RAW = STK / f"{BASE}.tif"
MASK = STK / f"{BASE}_segmented_clean.tif"
SKELC = STK / f"{BASE}_segmented_clean.skelcoords.npz"
ASBUILT_GRAPH = STK / f"{BASE}_segmented_clean_asbuilt_graph.json"
ASBUILT_GRAPH_CLEAN = STK / f"{BASE}_segmented_clean_asbuilt_graph_cleaned.json"
BENT_JSON = STK / f"{BASE}_asbuilt_bent.json"
CLASSIFIED_JSON = STK / f"{BASE}_unified_defects_accurate.json"
