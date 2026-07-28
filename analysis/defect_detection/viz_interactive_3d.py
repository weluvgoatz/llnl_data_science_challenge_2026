"""Interactive 3-D defect map (plotly HTML) — rotate / zoom / hover in a browser.

Every designed strut is a coloured line by verdict; hover shows verdict + bow.
Self-contained .html: just double-click it.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
STK = ROOT / "data/missing_struts/tif_stacks"
BASE = "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices"
UNI = STK / f"{BASE}_unified_defects_accurate.json"
OUT = ROOT / "analysis/defect_detection/INTERACTIVE_defect_map.html"

COLORS = {"present": "#3b82f6", "bent": "#c026d3", "thin": "#eab308",
          "disconnected": "#f97316", "missing": "#ef4444"}
WIDTH = {"present": 1.5, "bent": 5, "thin": 5, "disconnected": 5, "missing": 5}
SHOW_PRESENT_EVERY = 3          # thin out the healthy struts so defects pop


def main():
    d = json.load(open(UNI))
    S = d["struts"]
    counts = d["meta"]["counts"]

    fig = go.Figure()
    for cat in ("present", "bent", "thin", "disconnected", "missing"):
        xs, ys, zs, hover = [], [], [], []
        step = SHOW_PRESENT_EVERY if cat == "present" else 1
        items = [s for s in S if s["verdict"] == cat][::step]
        for s in items:
            p0, p1 = s["p0"], s["p1"]
            xs += [p0[0], p1[0], None]
            ys += [p0[1], p1[1], None]
            zs += [p0[2], p1[2], None]
        n = counts.get(cat, 0)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(color=COLORS[cat], width=WIDTH[cat]),
            name=f"{cat} ({n})", opacity=0.95 if cat != "present" else 0.45,
            hoverinfo="name"))

    fig.update_layout(
        title=dict(text="AS-BUILT LATTICE — interactive defect map "
                        "(drag to rotate · scroll to zoom · click legend to toggle)",
                   font=dict(size=18)),
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   aspectmode="data", bgcolor="#0a0a0f",
                   xaxis=dict(backgroundcolor="#0a0a0f", color="#aaa", gridcolor="#222"),
                   yaxis=dict(backgroundcolor="#0a0a0f", color="#aaa", gridcolor="#222"),
                   zaxis=dict(backgroundcolor="#0a0a0f", color="#aaa", gridcolor="#222")),
        paper_bgcolor="#0a0a0f", font=dict(color="#e8e8f0"),
        legend=dict(font=dict(size=14)), height=950)
    fig.write_html(OUT, include_plotlyjs=True, full_html=True)
    print(f"saved {OUT.name}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
