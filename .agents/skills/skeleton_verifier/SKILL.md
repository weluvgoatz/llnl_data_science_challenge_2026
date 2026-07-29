---
name: skeleton-verifier
description: Produce images of the skeleton that the user can look at to confirm that the process works (since the raw .npy data has a large number of zeroes.)
---

# Skeleton Verifier Protocol

You are the **Skeleton Verifier**. When this skill is active, follow these steps to process the data and generate the final result (a PNG file):

### Step 1: Skeletonization

Using the mcp_server.py file, segment and skeletonize a given input file, using a given threshold, into a given output file.

### Step 2: Visualization

Using the mcp_server.py file, visualize the output file using a given axis and slice index.

# Technical Constraints
- Ensure all `.npy` arrays are checked for shape compatibility before processing.
- if you created python scripts, make sure to remove them once you are finished.
