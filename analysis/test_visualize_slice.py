"""Quick test for the visualize_slice MCP tool.

Run from the repository root:
    python analysis/test_visualize_slice.py
"""

import asyncio
from pathlib import Path
import sys

from fastmcp import Client

# Make the repository root importable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.mcp_server import mcp

INPUT_PATH = REPOSITORY_ROOT / "data" / "unitcell" / "unitcell.npy"
OUTPUT_PATH = REPOSITORY_ROOT / "analysis" / "outputs" / "slice_128.png"


async def main():
    client = Client(mcp)

    async with client:
        # Confirm the tool is registered.
        tools = await client.list_tools()
        print("Registered MCP tools:")
        for tool in tools:
            print(f"- {tool.name}")

        # Call visualize_slice on the middle slice.
        result = await client.call_tool(
            "visualize_slice",
            {
                "input_filepath": str(INPUT_PATH),
                "output_filepath": str(OUTPUT_PATH),
                "slice_index": 128,
                "axis": 0,
            },
        )

        print("\nMCP result:")
        print(result.data)

    # Independently confirm the image was created.
    if OUTPUT_PATH.exists():
        print(f"\nImage created: {OUTPUT_PATH}")
        print(f"File size: {OUTPUT_PATH.stat().st_size:,} bytes")
    else:
        raise RuntimeError(f"Expected image was not created: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
