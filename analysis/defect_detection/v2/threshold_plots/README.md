# Threshold justification

Where the `thin` and `bent` thresholds sit inside the measured population.

| file | contents |
|---|---|
| `threshold_justification.png` | both panels together |
| `threshold_thin.png` | thin only |
| `threshold_bent.png` | bent only |

Regenerate with `python plot_thresholds.py`.

## The claim these figures support

**Neither threshold cuts through the bulk of the distribution.** Each sits out in
a tail, so a strut has to be a genuine outlier to be flagged — the counts are not
an artifact of where a line was drawn.

## THIN — derived from the scan, not from the drawing

Healthy struts pile up at **329 µm** with a MAD of **44 µm**. The cutoff is

> **median − 3 × MAD = 197 µm** → 548 struts flagged

The −1 and −2 MAD lines are drawn to show that 3 is deliberately conservative.

The important detail is in the callout: **the part measures 329 µm against a
424 µm nominal spec.** The printed struts are 22% thinner than their own drawing.
Anyone who used the nominal diameter as the healthy baseline would mislabel a
large share of perfectly good struts. That is why the cutoff is recomputed from
each scan's own strut population rather than taken from the design.

MAD rather than standard deviation, because the defects are *in* the data: a
standard deviation is inflated by the very outliers being hunted, so the
threshold drifts toward them. MAD is fixed by the middle of the distribution and
ignores the tails.

## BENT — a physical scale, not a statistical one

Healthy struts form a tight peak at **69 µm** that decays away well before the
line at

> **212 µm = one strut radius** → 380 struts flagged

The flagged struts are a **separate tail**, not the shoulder of the main
population — which is the real justification for the cut.

The threshold is physical rather than statistical: bowed by more than its own
radius, a strut no longer overlaps the straight axis it was meant to follow.

## Reading the plots honestly

- **Log y-axis** on both. With ~16,000 healthy struts the tails are invisible on
  a linear scale.
- **The x-axes are clipped, and the omissions are labelled on the figure.**
  497 thin measurements beyond 620 µm are struts inside the fused end-caps, where
  the distance transform reports the solid slab rather than a strut; 36 bows
  exceed 560 µm.
- **Thin's histogram is blocky on purpose.** The distance transform is quantised
  (distances come out as √integers), so narrow bins comb it into stripes that
  look like structure and are not. Bins are 18 µm.
