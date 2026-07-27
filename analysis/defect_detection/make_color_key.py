"""Colour key for the 3D lattice models (matches export_realistic_models RGB)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "analysis/defect_detection/MODEL_color_key.png"

ROWS = [
    ("present",      (120, 180, 255), "PRESENT",      "17,253", "93.42%", "healthy strut — full, straight, near-nominal thickness"),
    ("missing",      (255, 65, 65),   "MISSING",      "410",    "2.22%",  "no strut between the two joints (red = where it should be)"),
    ("bent",         (245, 70, 240),  "BENT",         "365",    "1.98%",  "present but curved — bow > 1 radius (212 um)"),
    ("thin",         (255, 225, 45),  "THIN",         "287",    "1.55%",  "present but skinny — density a robust low outlier"),
    ("disconnected", (255, 140, 30),  "DISCONNECTED", "153",    "0.83%",  "broken — a continuous gap >= 25% of the strut"),
]

fig, ax = plt.subplots(figsize=(12, 6.2), facecolor="white")
ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
ax.text(0.3, 6.2, "LATTICE DEFECT MODEL — COLOUR KEY", fontsize=20, fontweight="bold")
ax.text(0.3, 5.75, "MODEL_lattice_realistic.ply  ·  18,468 struts  ·  6.58% defective",
        fontsize=12, color="#555")

y = 5.0
for name, (r, g, b), label, cnt, pct, desc in ROWS:
    c = (r / 255, g / 255, b / 255)
    ax.add_patch(FancyBboxPatch((0.4, y - 0.28), 0.9, 0.56, boxstyle="round,pad=0.02",
                                fc=c, ec="#333", lw=1.2))
    ax.text(1.55, y + 0.06, label, fontsize=16, fontweight="bold", va="center", color=c if name != "thin" else "#c8a000")
    ax.text(1.55, y - 0.24, desc, fontsize=10.5, va="center", color="#333")
    ax.text(10.2, y + 0.06, cnt, fontsize=13, fontweight="bold", va="center", ha="right")
    ax.text(11.7, y + 0.06, pct, fontsize=13, fontweight="bold", va="center", ha="right")
    y -= 0.95

ax.text(10.2, 5.55, "count", fontsize=10, ha="right", color="#777")
ax.text(11.7, 5.55, "share", fontsize=10, ha="right", color="#777")
fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
print("saved", OUT.name)
