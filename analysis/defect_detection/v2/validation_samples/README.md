# Validation samples

Three examples of each defect type, taken from the 200 panels that were manually
reviewed (all 200 were correct). These are the clearest illustrations of what
each verdict means — not a random sample, so read them as *what the defect looks
like*, not as evidence of the detector's rate.

Every panel is drawn **from the classifier's own stored measurements**, so the
numbers in the title are the exact values the verdict was decided on. The bar
underneath reads left(p0) → right(p1): **green = metal found, red = gap**.

Filenames carry the strut id (`i####`) and the measurement, so any panel can be
traced straight back to its record in `defect_list/defective_struts.json`.

## missing — the strut was never built

| file | what it shows |
|---|---|
| `01_i7706_gap2156um.png` | textbook case: two intact junctions, 2156 µm of nothing between them, graph confirms 2 hops |
| `04_i10003_gap2713um.png` | `node_lost` — the far node was never printed either, so the marker sits in empty space |
| `05_i12286_gap2116um.png` | a clean vertical (z-climbing) drop-out |

The red bar spans the middle with green only at the ends — that green is the two
node blobs, not strut material.

## disconnected — built, then severed

| file | what it shows |
|---|---|
| `08_i6245_break1153um_stub878um.png` | unambiguous: 1153 µm of open gap with real strut on both sides |
| `05_i4923_break162um_stub1566um.png` | a narrow break — the strut is nearly whole but not connected |
| `04_i11681_break0um_stub2079um.png` | **detached at a joint**: the break reads 0 µm and the bar is all green, because the crack is hairline and metal exists at every position along the strut. Connectivity still says severed. This is the case a presence-based test cannot see. |

## thin — under-thickness

| file | what it shows |
|---|---|
| `06_i15067_min116um_necked.png` | severe neck, 116 µm at its thinnest against a 329 µm healthy median |
| `07_i14469_min116um_necked.png` | another deep pinch, mid-span |
| `01_i13146_min164um_necked.png` | median 232 µm but necks to 164 µm — normal-looking overall, thin in one place, which is why the thinnest *sustained* section is what triggers the verdict rather than the median |

## bent — bowed off its own axis

| file | what it shows |
|---|---|
| `08_i10969_bow349um.png` | strong, obvious curve |
| `05_i13039_bow329um.png` | clear sag below the chord |
| `01_i9320_bow297um.png` | moderate bow, still well above the 212 µm threshold |

Rendered in each strut's own bending plane, with the white dotted line marking
the straight chord between its own ends — the gap between the pink centreline and
that chord *is* the measured bow.

## Reproduce the full set

```bash
cd analysis/defect_detection/v2
python export_samples.py        # 50 individual panels per category
python make_review_grids.py     # packs them into 2x2 grids for fast review
```
