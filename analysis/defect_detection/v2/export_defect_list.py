"""Export every DEFECTIVE strut with its coordinates and evidence.

Writes three files to v2/defect_list/:
  defective_struts.json  - all defective struts, one record each, self-documenting
  defective_struts.csv   - same rows, flat, opens in Excel
  README.md              - what every field means

Only defective struts are included (present ones are omitted); the full 18,468
remain in strut_classification_v2.json.
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BASE  # noqa: E402
from detect_v2 import OUT_JSON, UM  # noqa: E402

OUT = Path(__file__).resolve().parent / "defect_list"
DEFECTS = ("missing", "disconnected", "thin", "bent")

SUBTYPE_MEANING = {
    "dropped": "both endpoint nodes were printed - the strut itself is absent",
    "node_lost": "one endpoint node was never printed, so the strut has nothing to span to",
    "void": "both endpoint nodes were never printed (an unprinted region)",
    None: "not applicable",
}


def main():
    OUT.mkdir(exist_ok=True)
    data = json.load(open(OUT_JSON))
    S = data["struts"]
    meta = data["meta"]
    thr = meta["thresholds"]

    rows = []
    for r in S:
        if r["verdict"] not in DEFECTS:
            continue
        p0 = [round(float(v), 2) for v in r["p0"]]      # [x, y, z] scan voxels
        p1 = [round(float(v), 2) for v in r["p1"]]
        d = np.array(p1) - np.array(p0)
        vertical = abs(float(d[2])) > 20                # z changes ~40 layers
        rec = {
            "strut_id": r["i"],
            "defect": r["verdict"],
            "subtype": r.get("subtype"),
            # --- location ---
            "p0_xyz_voxels": p0,
            "p1_xyz_voxels": p1,
            "p0_xyz_um": [round(v * UM, 1) for v in p0],
            "p1_xyz_um": [round(v * UM, 1) for v in p1],
            "midpoint_xyz_voxels": [round((a + b) / 2, 1) for a, b in zip(p0, p1)],
            "z_slice_range": [int(round(min(p0[2], p1[2]))),
                              int(round(max(p0[2], p1[2])))],
            "length_um": r["len_um"],
            "orientation": "vertical" if vertical else "flat",
            # --- context you must read before quoting a defect ---
            "region": r["context"],
            "node0_state": r["node0"],
            "node1_state": r["node1"],
            # --- the measurements the verdict was decided on ---
            "gap_um": r.get("gap_um"),
            "surviving_piece_um": r.get("stub_um"),
            "diameter_median_um": r.get("dia_um"),
            "diameter_thinnest_um": r.get("dia_min_um"),
            "bow_um": r.get("bow_um"),
            "graph_hops": r.get("hops"),
            "detached_at_joint": bool(r.get("flag_phantom_edge", False)),
            "metal_profile": r["prof"],
        }
        rows.append(rec)

    rows.sort(key=lambda x: (DEFECTS.index(x["defect"]), x["strut_id"]))

    counts = Counter(r["defect"] for r in rows)
    sub = defaultdict(Counter)
    for r in rows:
        sub[r["defect"]][r["subtype"] or "-"] += 1
    region = defaultdict(Counter)
    orient = defaultdict(Counter)
    for r in rows:
        region[r["defect"]][r["region"]] += 1
        orient[r["defect"]][r["orientation"]] += 1

    doc = {
        "what_this_is": (
            f"Every defective strut found in {BASE}. "
            f"{len(rows)} defective of {meta['n']} design struts "
            f"({100*len(rows)/meta['n']:.2f}%). Present struts are not listed here."),
        "coordinates": (
            "p0/p1 are the two ends of the strut in SCAN VOXEL coordinates, ordered "
            "[x, y, z]. z is the TIF slice number, so z_slice_range tells you which "
            "slices to open. *_um fields are the same points in microns "
            f"(1 voxel = {UM} um)."),
        "defect_definitions": {
            "missing": "no strut was built between the two nodes",
            "disconnected": ("a strut was built but is severed - no single connected "
                             "piece of metal reaches from one node to the other"),
            "thin": (f"some sustained section is thinner than {thr['thin_cut_um']:.0f} um "
                     f"(healthy struts measure ~{thr['dia_median_um']:.0f} um)"),
            "bent": (f"the strut's centreline bows more than {thr['bent_bow_um']:.0f} um "
                     "off the straight line between its own ends"),
        },
        "subtype_definitions": {k: v for k, v in SUBTYPE_MEANING.items() if k},
        "node_states": {
            "junction": "a real printed junction is there",
            "partial": "some metal is there but not a clean junction",
            "absent": "nothing was printed at that node position",
        },
        "region_warning": (
            "region=interior means >=35 voxels inside the printed lattice AND clear of "
            "the fused end-caps - these are the individually defensible defects. "
            "region=boundary means near an outer face, where the design extends past "
            "what was actually printed; the metal really is missing/severed there, but "
            "it reflects the under-built shell rather than an isolated defect. Quote "
            "interior counts for verified defects."),
        "reading_the_measurements": {
            "gap_um": "longest stretch with no metal within 8 voxels of the strut",
            "surviving_piece_um": ("longest continuous piece of strut mid-span; "
                                   f">{thr['stub_min_um']:.0f} um means material was "
                                   "built here, so the strut is disconnected not missing"),
            "diameter_thinnest_um": "thinnest sustained section (drives the thin verdict)",
            "bow_um": "sustained deviation of the centreline from its own chord",
            "graph_hops": ("shortest path in hops between the two nodes through the "
                           "as-built skeleton; >=2 independently confirms no direct "
                           "connection, -1 means a node is not in the graph"),
            "detached_at_joint": ("true = the strut body is still lying in place but is "
                                  "cracked off at a joint (break may read ~0 because the "
                                  "crack is short, yet the metal is not connected)"),
            "metal_profile": ("60 samples from p0 to p1; '#' = metal found within 8 "
                              "voxels, '.' = nothing"),
        },
        "totals": dict(counts),
        "by_subtype": {k: dict(v) for k, v in sub.items()},
        "by_region": {k: dict(v) for k, v in region.items()},
        "by_orientation": {k: dict(v) for k, v in orient.items()},
        "specimen": {
            "name": BASE,
            "voxel_um": UM,
            "nominal_strut_diameter_um": thr["nominal_diameter_um"],
            "volume_shape_zyx": meta["volume_shape_zyx"],
            "printed_lattice_bbox_zyx": meta["printed_bbox_zyx"],
        },
        "thresholds": thr,
        "source": "strut_classification_v2.json (detect_v2.py)",
    }

    (OUT / "defective_struts.json").write_text(
        json.dumps({"documentation": doc, "defective_struts": rows}, indent=1),
        encoding="utf-8")

    cols = ["strut_id", "defect", "subtype", "region", "orientation",
            "x0", "y0", "z0", "x1", "y1", "z1", "z_slice_from", "z_slice_to",
            "length_um", "node0_state", "node1_state", "gap_um",
            "surviving_piece_um", "diameter_median_um", "diameter_thinnest_um",
            "bow_um", "graph_hops", "detached_at_joint"]
    with open(OUT / "defective_struts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                "strut_id": r["strut_id"], "defect": r["defect"],
                "subtype": r["subtype"] or "", "region": r["region"],
                "orientation": r["orientation"],
                "x0": r["p0_xyz_voxels"][0], "y0": r["p0_xyz_voxels"][1],
                "z0": r["p0_xyz_voxels"][2],
                "x1": r["p1_xyz_voxels"][0], "y1": r["p1_xyz_voxels"][1],
                "z1": r["p1_xyz_voxels"][2],
                "z_slice_from": r["z_slice_range"][0],
                "z_slice_to": r["z_slice_range"][1],
                "length_um": r["length_um"],
                "node0_state": r["node0_state"], "node1_state": r["node1_state"],
                "gap_um": r["gap_um"], "surviving_piece_um": r["surviving_piece_um"],
                "diameter_median_um": r["diameter_median_um"],
                "diameter_thinnest_um": r["diameter_thinnest_um"],
                "bow_um": r["bow_um"], "graph_hops": r["graph_hops"],
                "detached_at_joint": int(r["detached_at_joint"]),
            })

    L = [f"# Defective struts — {BASE}\n",
         f"**{len(rows)} defective struts** out of {meta['n']} "
         f"({100*len(rows)/meta['n']:.2f}%).\n",
         "## Files",
         "- `defective_struts.json` — full records + inline documentation",
         "- `defective_struts.csv` — same rows, flat, for Excel\n",
         "## Counts", "| defect | total | interior | boundary | vertical | flat |",
         "|---|---:|---:|---:|---:|---:|"]
    for k in DEFECTS:
        L.append(f"| {k} | {counts[k]} | {region[k]['interior']} | "
                 f"{region[k]['boundary']} | {orient[k]['vertical']} | "
                 f"{orient[k]['flat']} |")
    L += ["", "## Subtypes (missing / disconnected)"]
    for k in ("missing", "disconnected"):
        parts = ", ".join(f"{n} {s}" for s, n in sorted(sub[k].items()))
        L.append(f"- **{k}**: {parts}")
    L += ["",
          "## Coordinates",
          "`x0,y0,z0 -> x1,y1,z1` are the strut's two ends in scan voxels. "
          f"**z is the TIF slice number** (1 voxel = {UM} um), so to inspect a strut "
          "open the slices in `z_slice_from`..`z_slice_to`.",
          "",
          "## Read before quoting numbers",
          "`region=boundary` struts sit near an outer face where the design extends "
          "past what was actually printed. The metal really is absent/severed, but it "
          "reflects the under-built shell rather than an isolated defect — "
          f"**{sum(region[k]['interior'] for k in DEFECTS)} of {len(rows)}** defects "
          "are `interior` and individually defensible.",
          "",
          "`detached_at_joint=1` (disconnected) means the strut body is still in place "
          "but cracked off at a joint — `gap_um` can read ~0 because the crack is short, "
          "yet no connected metal path exists. These are real severances, verified by "
          "connected-component labelling.",
          "",
          "Each `strut_id` indexes the design JSON strut list and matches the "
          "`i####` in the manual-validation panel filenames."]
    (OUT / "README.md").write_text("\n".join(L), encoding="utf-8")

    print(f"wrote {len(rows)} defective struts to {OUT}")
    for k in DEFECTS:
        print(f"  {k:13s} {counts[k]:5d}   interior {region[k]['interior']:5d}   "
              f"vertical {orient[k]['vertical']:5d}")


if __name__ == "__main__":
    main()
