import asyncio
from pathlib import Path
import sys

import numpy as np
from fastmcp import Client



# Make the repository root importable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.mcp_server import mcp



INPUT_PATH = (
  REPOSITORY_ROOT
  / "data"
  / "unitcell"
  / "unitcell.npy"
)

OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "unitcell"
    / "segmentation"
    / "unitcell_mask.npy"
)

THRESHOLD = 0.005813



async def main():
    client = Client(mcp)

    async with client:
        # Confirm that FastMCP registered the tools.
        tools = await client.list_tools()

        print("Registered MCP tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # Invoke the segmentation function through MCP.
        result = await client.call_tool(
            "segment_ct_dataset",
            {
                "input_filepath": str(INPUT_PATH),
                "output_filepath": str(OUTPUT_PATH),
                "threshold": THRESHOLD,
            },
        )

        print("\nMCP result:")
        print(result.data)

    # Independently verify the saved output.
    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            f"Expected output was not created: {OUTPUT_PATH}"
        )

    input_volume = np.load(
        INPUT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    output_mask = np.load(
        OUTPUT_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    print("\nIndependent output verification:")
    print("Input shape:", input_volume.shape)
    print("Output shape:", output_mask.shape)
    print("Output datatype:", output_mask.dtype)
    print(
        "Unique output values:",
        np.unique(output_mask),
    )

    assert output_mask.shape == input_volume.shape
    assert output_mask.dtype == np.bool_

    unique_values = np.unique(output_mask)

    assert set(unique_values.tolist()).issubset(
        {False, True}
    )

    print("\nAll segmentation checks passed.")



if __name__ == "__main__":
    asyncio.run(main())