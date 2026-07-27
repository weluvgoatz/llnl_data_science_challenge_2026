"""End-to-end test of all three MCP tools through the MCP client.

This exercises the SAME code path Codex uses (list tools, then call them by
name over the MCP protocol), but without needing Codex configured. If this
passes, the tools themselves work; any remaining issue is Codex config only.

Run from the repository root:
    python analysis/test_all_tools.py
"""

import asyncio
from pathlib import Path
import sys

import numpy as np
from fastmcp import Client

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.mcp_server import mcp

DATA = REPOSITORY_ROOT / "data" / "unitcell"
OUT = REPOSITORY_ROOT / "analysis" / "outputs"

CT_PATH = DATA / "unitcell.npy"
MASK_PATH = DATA / "segmentation" / "test_mask.npy"
SKELETON_PATH = DATA / "segmentation" / "test_skeleton.npy"
SLICE_IMAGE = OUT / "test_ct_slice.png"

THRESHOLD = 0.005813


async def main():
    client = Client(mcp)

    async with client:
        # --- Level 1: the tools are registered --------------------------
        tools = [t.name for t in await client.list_tools()]
        print("Registered tools:", tools)
        for expected in ("segment_ct_dataset", "visualize_slice", "skeletonize"):
            assert expected in tools, f"MISSING TOOL: {expected}"
        print("PASS: all three tools are registered\n")

        # --- Task 1: segment --------------------------------------------
        r = await client.call_tool("segment_ct_dataset", {
            "input_filepath": str(CT_PATH),
            "output_filepath": str(MASK_PATH),
            "threshold": THRESHOLD,
        })
        print(r.data, "\n")
        assert "successfully" in r.data
        assert MASK_PATH.exists()
        print("PASS: segmentation created a mask\n")

        # --- Task 2: visualize ------------------------------------------
        r = await client.call_tool("visualize_slice", {
            "input_filepath": str(CT_PATH),
            "output_filepath": str(SLICE_IMAGE),
            "slice_index": 128,
            "axis": 0,
        })
        print(r.data, "\n")
        assert "successfully" in r.data
        assert SLICE_IMAGE.exists()
        print("PASS: visualization created an image\n")

        # --- Task 3: skeletonize ----------------------------------------
        r = await client.call_tool("skeletonize", {
            "input_filepath": str(MASK_PATH),
            "output_filepath": str(SKELETON_PATH),
        })
        print(r.data, "\n")
        assert "successfully" in r.data
        assert SKELETON_PATH.exists()
        print("PASS: skeletonization created a skeleton\n")

    # --- Independent correctness checks (outside the MCP layer) ---------
    mask = np.load(MASK_PATH)
    skel = np.load(SKELETON_PATH)
    assert skel.shape == mask.shape, "skeleton shape != mask shape"
    assert skel.dtype == bool, "skeleton is not boolean"
    assert bool(np.all(mask[skel])), "skeleton escapes the material (BUG)"
    print("PASS: skeleton is boolean, same shape, and lies inside the mask")

    print("\nALL TESTS PASSED - the three tools work end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
