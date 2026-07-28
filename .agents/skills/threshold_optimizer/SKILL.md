---
name: threshold-optimizer
description: Sweeps multiple segmentation thresholds on a CT volume, compares them with quantitative and visual metrics, and recommends the best threshold using Otsu's method confirmed by connectivity analysis.
---

# Threshold Optimization Protocol

You are the **Segmentation Threshold Optimizer**. Choosing the intensity threshold
is the single most important decision in CT segmentation: too low includes
background noise (over-segmentation), too high erases thin struts
(under-segmentation). When this skill is active, follow these steps to find and
justify the best threshold for a given volume.

### Step 1: Inspect the Intensity Range (do this first)
- Load the raw CT `.npy` volume and report its **minimum and maximum intensity**.
- **Critical:** thresholds must lie inside this range. A generic sweep such as
  0.3 / 0.5 / 0.7 only works for volumes normalized to `[0, 1]`. For raw CT data
  (for example intensities in `[-0.003, 0.015]`), those values produce empty masks.
  Always choose thresholds from the volume's actual range.

### Step 2: Run the Threshold Sweep Analysis
Run the helper script `scripts/analyze_thresholds.py`. It segments the volume at a
data-adaptive set of thresholds and computes, for each one:

| Metric | Meaning | What good looks like |
| :--- | :--- | :--- |
| Material fraction (%) | Share of voxels called material | On a stable plateau, not a steep slope |
| Connected components | Number of separate 26-connected pieces | Small (ideally 1 for an intact lattice) |
| Largest-component fraction | Share of foreground in the biggest piece | Close to 1.0 (one clean structure) |

Example invocation (from the repository root):
```
python .agents/skills/threshold_optimizer/scripts/analyze_thresholds.py \
    --volume data/unitcell/unitcell.npy \
    --output-dir data/unitcell/threshold_sweep \
    --num 7
```
The script writes `threshold_metrics.csv`, `threshold_metric_curves.png`, and
`threshold_slice_comparison.png`, and prints the **Otsu threshold** (the statistical
separator between the background and material intensity peaks) plus a confirmed
recommendation.

### Step 3: Produce the Official Masks with the MCP Tool
For the thresholds worth keeping — at minimum the recommended one — call the
`segment_ct_dataset()` MCP tool to save each result as its own `.npy` file, so the
masks are reproducible and comparable:
```
segment_ct_dataset(
    input_filepath = "<volume>.npy",
    output_filepath = "data/.../threshold_sweep/mask_thr_<value>.npy",
    threshold = <value>,
)
```
The masks the MCP tool saves are identical to the masks the analysis script
evaluated, because both apply `volume >= threshold`.

### Step 4: Recommend and Justify
Compile a short report containing:
1. **Sweep table:** threshold, material fraction, connected components, and
   largest-component fraction for every threshold tried.
2. **Figures:** embed `threshold_slice_comparison.png` and
   `threshold_metric_curves.png`.
3. **Recommendation:** state the recommended threshold and justify it with three
   signals that should agree — it sits (a) at or near the Otsu value, (b) in the
   valley of the connected-components curve, and (c) on the plateau of the material
   fraction curve.

# Technical Constraints
- Always derive thresholds from the volume's real intensity range (Step 1).
- Use 26-connectivity for 3D component analysis; lattice struts run diagonally and
  would look broken under 6-connectivity.
- Treat Otsu as the principled continuous optimum, and use the connectivity and
  plateau evidence to confirm it rather than trusting any single metric.
- If you create scratch Python files beyond this skill's own scripts, remove them
  when finished.
