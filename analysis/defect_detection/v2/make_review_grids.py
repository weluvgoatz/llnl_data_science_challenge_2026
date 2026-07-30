"""Pack the 200 manual-validation panels into 2x2 review grids (4 per image,
near-native resolution) so every single panel can be eyeballed efficiently.
Output: manual_validation/_review/{category}_grid_NN.png
"""
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent / "manual_validation"
OUT = BASE / "_review"
COLS, ROWS = 2, 2
PAD = 12


def main():
    OUT.mkdir(exist_ok=True)
    for cat in ["missing", "disconnected", "thin", "bent"]:
        files = sorted((BASE / cat).glob("*.png"))
        for g in range(0, len(files), COLS * ROWS):
            chunk = files[g:g + COLS * ROWS]
            ims = [Image.open(f) for f in chunk]
            w = max(im.width for im in ims)
            h = max(im.height for im in ims)
            sheet = Image.new("RGB", (COLS * w + (COLS + 1) * PAD,
                                      ROWS * h + (ROWS + 1) * PAD), "white")
            for i, im in enumerate(ims):
                r, c = divmod(i, COLS)
                sheet.paste(im, (PAD + c * (w + PAD), PAD + r * (h + PAD)))
            n = g // (COLS * ROWS) + 1
            sheet.save(OUT / f"{cat}_grid_{n:02d}.png")
        print(f"{cat}: {len(files)} panels -> {(len(files)+3)//4} grids", flush=True)


if __name__ == "__main__":
    main()
