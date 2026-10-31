"""Light colour grading for concept stills (PIL): saturation + gentle S-curve. Applied identically to every concept render of the world."""
import sys
import numpy as np
from PIL import Image


def grade(path, sat=1.06, contrast=0.10, out=None):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    lum = (a * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
    a = np.clip(lum + (a - lum) * sat, 0, 1)
    a = np.clip(a + contrast * (a - 0.5) * (1 - np.abs(a - 0.5) * 2) * 2, 0, 1)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(out or path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        grade(p)
