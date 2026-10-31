"""MOSS concept art via the local `imgai` CLI (sequential, 8 GB VRAM). MOSS never sees or invents the track geometry: it only paints environment,
architecture, props, crowd and lighting concepts. The moodboard is composed from four MOSS tiles plus the palette."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_create import get_project, gpu_job
PROJECT = get_project()
from PIL import Image, ImageDraw, ImageFont

STYLE = ("premium stylized low-poly 3D game environment concept art, polished indie AA racing game, clearly visible faceted planar geometry, chunky readable silhouettes, "
         "flat-shaded polygon facets with subtle gradients, clean colour blocking, rich materials, cohesive palette, soft cinematic lighting, not realistic, not a photograph")
NEG = ("photorealistic, photograph, realistic skin, 2d flat illustration, comic outline, extremely primitive blocky cubes, noisy micro detail, clutter, text, letters, watermark, logo, "
       "extra limbs, deformed, cropped, blurry, empty landscape")
FONT = "/usr/share/fonts/google-noto/NotoSans-"


def gen(prompt, out, seed, w=1024, h=576, preset="quality", force=False):
    if os.path.exists(out) and not force:
        return True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [PROJECT.tool("image_ai", "imgai"), "generate", "-p", f"{prompt}, {STYLE}", "-n", NEG, "-m", "moss", "--preset", preset, "-W", str(w), "-H", str(h), "-s", str(seed), "-o", out, "--force"]
    t0 = time.time()
    with gpu_job():
        r = subprocess.run(cmd, capture_output=True, text=True)
    ok = os.path.exists(out)
    print(f"  moss {os.path.basename(out)} {w}x{h} seed {seed}: {'ok' if ok else 'FAILED'} {time.time()-t0:.0f}s", flush=True)
    if not ok:
        print(r.stdout[-500:], r.stderr[-500:])
    return ok


def moodboard(tiles, palette, title, subtitle, out, W=2400, H=1500, labels=None):
    img = Image.new("RGB", (W, H), (24, 28, 38))
    d = ImageDraw.Draw(img)
    tw, th = (W - 60) // 2, 560
    pos = [(20, 128), (W // 2 + 10, 128), (20, 128 + th + 20), (W // 2 + 10, 128 + th + 20)]
    labels = labels or ["architecture & materials", "crowd & atmosphere", "light installations", "landscape & landmarks"]
    for (x, y), t, lab in zip(pos, tiles, labels):
        im = Image.open(t).convert("RGB")
        im = im.resize((tw, int(im.height * tw / im.width)), Image.LANCZOS)
        im = im.crop((0, max(0, (im.height - th) // 2), tw, max(0, (im.height - th) // 2) + th))
        img.paste(im, (x, y))
        d.rectangle([x, y + th - 44, x + 420, y + th], fill=(0, 0, 0, 160))
        d.text((x + 14, y + th - 38), lab.upper(), font=ImageFont.truetype(FONT + "SemiBold.ttf", 22), fill=(255, 255, 255))
    d.text((24, 22), title.upper(), font=ImageFont.truetype(FONT + "Bold.ttf", 54), fill=(255, 255, 255))
    d.text((24, 92), subtitle, font=ImageFont.truetype(FONT + "Regular.ttf", 22), fill=(190, 200, 215))
    y0 = 128 + 2 * th + 50
    sw = (W - 40) // len(palette)
    for i, (name, hexc) in enumerate(palette.items()):
        x = 20 + i * sw
        d.rectangle([x, y0, x + sw - 10, y0 + 120], fill=hexc)
        lum = sum(int(hexc[j:j + 2], 16) for j in (1, 3, 5)) / 3
        d.text((x + 12, y0 + 12), name, font=ImageFont.truetype(FONT + "SemiBold.ttf", 22), fill=(0, 0, 0) if lum > 140 else (255, 255, 255))
        d.text((x + 12, y0 + 84), hexc.upper(), font=ImageFont.truetype(FONT + "Regular.ttf", 20), fill=(0, 0, 0) if lum > 140 else (255, 255, 255))
    img.save(out)


def run(track_folder, concepts, palette, title, subtitle, seed=2024):
    base = f"{track_folder}/concepts"
    tmp = f"{base}/_tiles"
    ok = []
    for i, (name, prompt) in enumerate(concepts["tiles"].items()):
        ok.append(gen(prompt, f"{tmp}/{name}.png", seed + i, 1024, 672, "balanced"))
    moodboard([f"{tmp}/{n}.png" for n in concepts["tiles"]], palette, title, subtitle, f"{base}/moodboard.png")
    for k, fn in (("environment", "environment-concept.png"), ("pitlane", "pitlane-concept.png"), ("props", "signature-props.png")):
        ok.append(gen(concepts[k], f"{base}/{fn}", seed + 10 + len(ok), 1024, 576, "quality"))
    return all(ok)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import importlib
    mod = importlib.import_module(f"tracks.{sys.argv[1]}")
    folder = PROJECT.folder(sys.argv[1], definition=mod)
    run(str(PROJECT.root / folder), mod.CONCEPTS, mod.PALETTE, mod.TITLE, mod.SUBTITLE)
