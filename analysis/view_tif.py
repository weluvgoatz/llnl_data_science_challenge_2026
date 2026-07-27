"""Interactive TIF-stack slice viewer.

Opens a window with a slider to scroll through every slice of a 3D .tif CT
stack. Reads slices on demand, so it stays fast and low-memory even for a
~1 GB stack.

Usage (from the repo root):
    python analysis/view_tif.py
    python analysis/view_tif.py "path/to/other.tif"

Controls:
    - drag the slider, or
    - press Left / Right arrow keys to step one slice,
    - press Up / Down to jump 10 slices.
"""

import sys
from pathlib import Path

import tifffile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIF = ROOT / "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"

tif_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TIF
print(f"Opening {tif_path.name} ...")

tif = tifffile.TiffFile(tif_path)
n_slices = len(tif.pages)
start = n_slices // 2  # start in the middle (the lattice), not the skin

fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.15)
img = ax.imshow(tif.pages[start].asarray(), cmap="gray")
ax.set_title(f"slice {start} / {n_slices - 1}")
ax.axis("off")

slider_ax = plt.axes([0.15, 0.05, 0.7, 0.03])
slider = Slider(slider_ax, "slice", 0, n_slices - 1, valinit=start, valstep=1)


def update(val):
    i = int(slider.val)
    img.set_data(tif.pages[i].asarray())
    ax.set_title(f"slice {i} / {n_slices - 1}")
    fig.canvas.draw_idle()


slider.on_changed(update)


def on_key(event):
    i = int(slider.val)
    if event.key == "right":
        slider.set_val(min(i + 1, n_slices - 1))
    elif event.key == "left":
        slider.set_val(max(i - 1, 0))
    elif event.key == "up":
        slider.set_val(min(i + 10, n_slices - 1))
    elif event.key == "down":
        slider.set_val(max(i - 10, 0))


fig.canvas.mpl_connect("key_press_event", on_key)
print(f"{n_slices} slices. Drag the slider or use arrow keys. Close the window to exit.")
plt.show()
