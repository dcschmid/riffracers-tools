"""v3 grading (PIL/numpy): saturation, warm highlights / cool shadows, soft bloom on bright areas, gentle vignette. Applied identically to all v3 stills."""
import sys
import numpy as np
from PIL import Image, ImageFilter


def grade(path, out=None, sat=1.20, contrast=0.12, bloom=0.22, vig=0.16, warm=0.045):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, np.float32) / 255.0
    lum = (a * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
    a = np.clip(lum + (a - lum) * sat, 0, 1)
    a = np.clip(a + contrast * (a - 0.5) * (1 - np.abs(a - 0.5) * 2) * 2, 0, 1)
    sh = np.clip(1 - lum * 1.8, 0, 1)
    hi = np.clip(lum * 1.6 - 0.5, 0, 1)
    a = np.clip(a + hi * np.array([warm, warm * 0.45, -warm * 0.8], np.float32) + sh * np.array([-warm * 0.5, 0.0, warm * 0.8], np.float32), 0, 1)
    thr = np.clip((lum - 0.62) / 0.38, 0, 1) * a
    big = Image.fromarray((thr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(im.size) / 90))
    big2 = Image.fromarray((thr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(max(im.size) / 32))
    g = (np.asarray(big, np.float32) * 0.65 + np.asarray(big2, np.float32) * 0.35) / 255.0
    a = 1 - (1 - a) * (1 - g * bloom)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    a = a * (1 - vig * np.clip(r - 0.55, 0, 1.2)[..., None])
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(out or path)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        grade(p)
