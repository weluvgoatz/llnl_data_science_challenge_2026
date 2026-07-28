---
name: lattice-defect-visualizer
description: Produces state-of-the-art, self-explanatory visualizations of lattice defects from the raw CT — a per-defect pipeline-validation figure (segmentation to skeleton to detection to verdict), a zoomed defect atlas, and full-lattice defect maps — every panel measured from the scan.
---

# Lattice Defect Visualization Protocol

You are the **Lattice Defect Visualizer**. After classification exists
(`<base>_unified_defects_accurate.json`), you render figures that show not just
*where* defects are but *how the model decided*, straight from the raw CT.
Scripts live in `analysis/defect_detection/`; select the specimen with the same
`LATTICE_BASE` / `LATTICE_STK` env vars as the classifier skill.

Consistent colour key everywhere: **blue = present, magenta = bent, yellow =
thin, orange = disconnected, red = missing.**

## Deliverables

### 1. Per-defect pipeline validation  (`viz_pipeline_perdefect.py`)
One 4-panel figure per defect type — **Segmentation -> Skeletonization ->
Detection -> Result** — reproducing the model's own measurement:
- missing: metal-fraction-along-strut profile (flat empty) vs the 15% line.
- disconnected: the profile showing metal -> GAP -> metal.
- thin: the density histogram of all struts with this strut left of the robust cutoff.
- bent: the bow-vs-position curve vs the 212 um line (rendered in the strut's own
  bending plane so the curve is visible).
Also invocable via the `render_defect_visualizations()` MCP tool.

### 2. Zoomed defect atlas  (`viz_defect_atlas.py`)
A gallery of tight raw-CT crops — several verified examples per defect type — so a
reader can see what each defect physically looks like.

### 3. Full-lattice map  (`viz_unified_map.py`)
The whole lattice coloured by verdict, for the big picture.

## Method notes (keep the figures honest)
- Every measurement plotted is the SAME quantity the classifier thresholds on —
  do not invent a prettier proxy.
- Pick interior examples (>~55 vox from every face) so boundary/"cooked-layer"
  effects don't contaminate the illustration.
- Use a thin MIP slab for crops so mostly the target strut shows, not neighbours;
  for a bent strut, view it in its bending plane (rotate a small crop) so the
  curvature reads and it doesn't look falsely disconnected.
- Verify each shown example against the actual metal (sample the mask along it)
  before presenting it.

# Technical Constraints
- Matplotlib with a non-interactive backend (`Agg`); write PNGs, don't block.
- These figures are large; render at full size — do not down-res to save time.
- Remove any scratch scripts you create beyond the pipeline's own.
