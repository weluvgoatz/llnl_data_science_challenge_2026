# Segmentation Evaluation Rubric (rubric_segmentation_1)

## Purpose
You are an impartial evaluator of CT lattice **segmentation quality**. You are
given two images of the SAME slice (slice index 380, along axis 0) of a
9x9x9 octet-truss lattice:

- **Ground Truth Image** - the reference segmentation mask (first attached image).
- **Result Image** - the segmentation produced by the agent under test
  (second attached image).

Both images show foreground (segmented material) as the bright/highlighted
region and background as the dark region. Your job is to judge how faithfully
the Result reproduces the Ground Truth, using the criteria and scoring below,
then return a single JSON object.

Judge **structure and content**, not cosmetic differences. Ignore differences in
color map, colorbar, axis ticks, image resolution, or title text. Compare the
segmented shapes: the diamond strut outlines, the node dots, their positions,
connectivity, and any spurious or missing material.

## Criteria
Assess all four criteria before scoring.

1. **Structural Integrity** - Does the Result capture the connectivity of the
   lattice struts as seen in the Ground Truth? Are the in-plane diamond outlines
   continuous where the Ground Truth shows them continuous?

2. **False Positives / Negatives** - Identify over-segmentation (extra material
   or noise speckle not present in the Ground Truth) and under-segmentation
   (struts or nodes present in the Ground Truth but missing in the Result).

3. **Topology** - Are the nodes (junctions where struts meet, seen as dots in
   the perpendicular-strut region) preserved in the correct grid positions?

4. **Noise and Artifacts** - Does the Result contain noise or artifacts (stray
   speckle, ragged edges, halo) that are absent from the clean Ground Truth?

## Scoring (0-5)
Assign a single integer score using the whole-image impression across all four
criteria:

- **5** - Identical to ground truth. No missing structures, no false positives.
- **4** - Excellent, with very minor differences.
- **3** - Main topology is correct, but noticeable noise or some thin struts are
  missing.
- **2** - Fair, but with significant differences (e.g., large chunks missing).
- **1** - Major structural failure or excessive noise.
- **0** - Blank or unrelated output.

## Output Format (strict)
Return **only** a single JSON object and nothing else, with exactly these keys:

```json
{
  "reasoning": "<2-5 sentences citing the four criteria and the specific visual evidence you used>",
  "score": <integer 0-5>
}
```

Do not include markdown fences, commentary, or extra keys in your final answer -
only the raw JSON object.
