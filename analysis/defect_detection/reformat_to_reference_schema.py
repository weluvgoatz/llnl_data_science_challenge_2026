"""Reformat a TIF-derived lattice JSON to match the reference (model) schema.

The reference registered JSON uses:
  top-level: junctions, struts, unit_cells
  junction:  {id, position, indices}
  strut:     {id, unit_cell_edge_idx, junction0, junction1, thickness}
  unit_cell: {id, struts, indices}

This rewrites our TIF-derived graph into that exact schema, keeping our
measured fields (mean_density, mean_thickness, metal_fraction, gap_frac,
verdict) as extra keys so no measurement is lost. `thickness` is set to our
measured mean_thickness.

Usage:
    python analysis/defect_detection/reformat_to_reference_schema.py --json "..._lattice_graph_classified.json"
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    jpath = Path(args.json).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else \
        jpath.with_name(jpath.stem + "_refschema.json")

    d = json.loads(jpath.read_text())

    # junction id -> half-grid index (for cell grouping)
    gi = {j["id"]: j.get("grid_index", j.get("indices")) for j in d["junctions"]}

    # --- junctions: grid_index -> indices ---
    junctions = [{"id": j["id"], "position": j["position"],
                  "indices": j.get("grid_index", j.get("indices"))}
                 for j in d["junctions"]]

    # Number of cells per axis = (max corner index)/2. Corners sit on even
    # half-grid indices 0..max, so a 0..18 range means 9 cells (0..8).
    gmax = [max(gi[i][k] for i in gi) for k in range(3)]
    n_cells = [gmax[k] // 2 for k in range(3)]  # e.g. 18//2 = 9 cells

    # --- group struts into unit cells by the cell containing their midpoint ---
    # Clamp to the last real cell so boundary struts (midpoint at the far
    # corner) don't create a phantom extra cell.
    cell_of = {}
    for s in d["struts"]:
        a = gi[s["junction0"]]; b = gi[s["junction1"]]
        mid = [(a[k] + b[k]) / 2 for k in range(3)]
        cell = tuple(min(int(mid[k] // 2), n_cells[k] - 1) for k in range(3))
        cell_of[s["id"]] = cell

    cells = sorted(set(cell_of.values()))
    cell_id = {c: i for i, c in enumerate(cells)}
    cell_base = [min(c[k] for c in cells) for k in range(3)]
    cell_struts = {i: [] for i in range(len(cells))}
    edge_idx = {}
    for s in d["struts"]:
        cid = cell_id[cell_of[s["id"]]]
        edge_idx[s["id"]] = len(cell_struts[cid])  # index within the cell
        cell_struts[cid].append(s["id"])

    # --- struts: reference field order + measured extras ---
    struts = []
    for s in d["struts"]:
        entry = {
            "id": s["id"],
            "unit_cell_edge_idx": edge_idx[s["id"]],
            "junction0": s["junction0"],
            "junction1": s["junction1"],
            "thickness": s.get("mean_thickness", 0.0),
        }
        # keep our measurements as extras
        for k in ("length", "mean_density", "min_density", "mean_thickness",
                  "metal_fraction", "gap_frac", "verdict"):
            if k in s:
                entry[k] = s[k]
        struts.append(entry)

    # --- unit_cells ---
    unit_cells = [{"id": cell_id[c],
                   "struts": cell_struts[cell_id[c]],
                   "indices": [c[0]-cell_base[0], c[1]-cell_base[1], c[2]-cell_base[2]]}
                  for c in cells]

    out = {"junctions": junctions, "struts": struts, "unit_cells": unit_cells}
    if "meta" in d:
        out["meta"] = d["meta"]
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Wrote reference-schema JSON: {out_path.name}")
    print(f"  junctions {len(junctions)}, struts {len(struts)}, unit_cells {len(unit_cells)}")


if __name__ == "__main__":
    main()
