# Defect detection output — for the frontend / dashboard

This folder has a **ready-to-use sample output** from the Strut Error Detection
subagent, so you can build the UI without running any Python.

**File:** `octet_9x9x9_defects.json` — every strut in the 9×9×9 octet lattice,
labelled with its defect type.

## What each strut looks like
```json
{ "p0": [64.0, 57.0, 33.0],      // one joint, [x, y, z] in scan voxels
  "p1": [108.1, 98.6, 18.7],     // the other joint, [x, y, z]
  "verdict": "bent" }            // present | missing | bent | thin | disconnected
```
To draw a strut: **a line (or tube) from `p0` to `p1`, coloured by `verdict`.**
That's the whole model — 18,468 of these.

## Colour key
| verdict | meaning | colour |
| :--- | :--- | :--- |
| `present` | healthy strut | blue `#78B4FF` |
| `missing` | no strut there | red `#FF4141` |
| `bent` | curved strut | magenta `#F546F0` |
| `thin` | skinny strut | yellow `#FFE12D` |
| `disconnected` | broken (gap in middle) | orange `#FF8C1E` |

## The summary numbers (`meta`)
```json
"meta": {
  "counts": { "present": 17253, "missing": 410, "bent": 365,
              "thin": 287, "disconnected": 153 },
  "n": 18468,                       // total struts
  "volume_shape_zyx": [761, 815, 837]   // scan size, for scaling the view
}
```
= **6.58% defective** (present is 93.42%). Use `counts` for stat tiles/charts;
use `volume_shape_zyx` to fit the 3D view (note: it's z,y,x; strut coords are x,y,z).

## How this file is produced (FYI — you don't need to)
The `strut_error_detection_agent` takes a raw lattice TIFF + the registered design
JSON and writes this JSON. It's deterministic — same inputs give this exact file
every time. To regenerate: run `analysis/defect_detection/unified_defects_accurate.py`
(or call the `classify_lattice_defects` MCP tool).

Full details: see [`../../../INTEGRATION.md`](../../../INTEGRATION.md).
