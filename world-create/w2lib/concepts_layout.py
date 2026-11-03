"""Layout-faithful concept art: MOSS img2img repaints of the Blender renders (which are built from the validated layout).
Strength is chosen per image so that the edge structure of the render survives (normalised cross-correlation of blurred gradient maps >= 0.62);
the measurement is a structural check, not a proof of geometric correctness (that stays with validation.json and geometry-overlay.png)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world_create import get_project, gpu_job, contained
PROJECT = get_project()
from w2lib import concepts

W2 = str(PROJECT.root)
STYLE_I2I = ("the road layout, grandstands, buildings and landmarks stay exactly where they are in the reference, vivid saturated colours, rich materials, cinematic lighting, "
             "premium stylized low-poly 3D game environment concept art, faceted planar geometry, not realistic, not a photograph")
NEG = concepts.NEG


def edges(img: Image.Image):
    g = img.convert("L").resize((256, 144)).filter(ImageFilter.GaussianBlur(1.2))
    a = np.asarray(g, float)
    gx = np.abs(np.diff(a, axis=1))[:-1, :]
    gy = np.abs(np.diff(a, axis=0))[:, :-1]
    return np.hypot(gx, gy)


def ncc(a, b):
    a = (a - a.mean()) / max(a.std(), 1e-6)
    b = (b - b.mean()) / max(b.std(), 1e-6)
    return float((a * b).mean())


def prep(src, dst, w=1024, h=576):
    im = Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS)
    a = np.asarray(im, float) / 255.0
    if a.mean() < 0.28:                      # dark night renders: lift so MOSS can see forms (edges stay the same)
        gamma = np.log(0.34) / np.log(max(a.mean(), 0.02))
        a = np.clip(a ** min(gamma, 0.8), 0, 1)
        a = np.clip(a / max(np.percentile(a, 99.5), 0.4), 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(dst)
    return Image.open(dst)


def repaint(init_path, prompt, out, seed, strengths=(0.5, 0.44, 0.38, 0.32)):
    best = None
    init = Image.open(init_path)
    e0 = edges(init)
    attempts = os.path.join(os.path.dirname(out), "_attempts", str(time.time_ns()))
    os.makedirs(attempts, exist_ok=True)
    for st in strengths:
        candidate = os.path.join(attempts, f"{os.path.basename(out)}-strength-{st}.png")
        cmd = [PROJECT.tool("image_ai", "imgai"), "img2img", "-i", init_path, "-p", f"{prompt}, {STYLE_I2I}", "--strength", str(st), "-m", "moss", "--preset", "quality", "-W", "1024", "-H", "576", "-s", str(seed), "-o", candidate, "--force"]
        t0 = time.time()
        with gpu_job():
            process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode or not os.path.exists(candidate):
            print("  FAILED", out, process.stderr[-400:], flush=True)
            continue
        c = ncc(e0, edges(Image.open(candidate)))
        print(f"  {os.path.basename(out)} strength {st}: layout correlation {c:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if best is None or c > best[1]:
            best = (st, c)
            shutil.copy2(candidate, out)
        if c >= 0.62:
            break
    if best is None or best[1] < 0.62:
        raise RuntimeError(f"No verified fresh MOSS repaint for {out}: {best}")
    return best


def run(mod_name, folder):
    import importlib
    m = importlib.import_module(f"tracks.{mod_name}")
    track_dir = contained(PROJECT.root, folder)
    d = str(track_dir)
    pre = getattr(m, "HQ_SOURCES", {})
    required = ("environment", "pitlane", "trap", "crowd")
    missing = [key for key in required if not pre.get(key)]
    if missing:
        raise ValueError(f"Define this track's own HQ_SOURCES in {mod_name}: {', '.join(missing)}")
    sources = {}
    for key in required:
        sources[key] = contained(track_dir, pre[key])
        if not sources[key].is_file():
            raise FileNotFoundError(f"Missing {key} reference for {mod_name}: {sources[key]}")
    flat = getattr(m, "CONCEPT_DIR", "concepts") == "."
    cdir = str(contained(track_dir, getattr(m, "CONCEPT_DIR", "concepts")))
    arch = f"{cdir}/_v1-text-only"  # first (text-only) set, kept as-is
    if not flat and not os.path.exists(arch) and any(os.path.exists(f"{cdir}/{f}") for f in ("moodboard.png", "environment-concept.png")):
        os.makedirs(arch)
        for f in ("moodboard.png", "environment-concept.png", "pitlane-concept.png", "signature-props.png"):
            if os.path.exists(f"{cdir}/{f}"):
                shutil.copy2(f"{cdir}/{f}", f"{arch}/{f}")
        if os.path.exists(f"{cdir}/_tiles"):
            shutil.copytree(f"{cdir}/_tiles", f"{arch}/_tiles")
    tmp = str(contained(PROJECT.cache_dir, f"{mod_name}/moss-work")) if flat else f"{cdir}/_layout-tiles"
    os.makedirs(tmp, exist_ok=True)
    crowd = f"{cdir}/crowd-concept.png" if flat else f"{tmp}/crowd.png"
    C = m.CONCEPTS
    import json as _j
    haz = _j.load(open(f"{d}/planning/hazards.json"))["traps"][0]
    jobs = [("environment", C["environment"], f"{cdir}/environment-concept.png", 101),
            ("pitlane", C["pitlane"], f"{cdir}/pitlane-concept.png", 102),
            ("trap", f"signature trap concept '{haz['name']}' on the racing road, {haz['visual']}, {C['tiles'].get('lights', '')}", f"{cdir}/signature-props.png", 103),
            ("crowd", f"stage with a large cheering crowd of stylized low-poly people: {C['tiles'].get('crowd', '')}", crowd, 104)]
    res = {}
    for key, prompt, out, seed in jobs:
        init = f"{tmp}/init_{key}.png"
        prep(str(sources[key]), init)
        r = repaint(init, prompt, out, seed + (7 if key == "trap" else 0) + int(getattr(m, "MOSS_SEED_OFFSET", 0)))
        res[key] = dict(source=str(sources[key].relative_to(track_dir)), strength=r[0], layout_correlation=round(r[1], 3), output=os.path.relpath(out, d))
    concepts.moodboard([f"{cdir}/environment-concept.png", crowd, f"{cdir}/signature-props.png", f"{cdir}/pitlane-concept.png"], m.PALETTE, m.TITLE,
                       m.SUBTITLE + "  ·  repainted from the validated layout", f"{cdir}/moodboard.png", labels=["track overview (layout-faithful)", "start straight & crowd", "signature trap in place", "pit lane"])
    fidelity = contained(track_dir, getattr(m, "FIDELITY_FILE", "concepts/layout-fidelity.json"))
    fidelity.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(fidelity, "w"), indent=1)
    print("done", folder, flush=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    selected = PROJECT.specs(True) if only else PROJECT.specs()
    mods = [(s["module"], s["folder"]) for s in selected]
    unknown = set(only) - {mod for mod, _ in mods}
    if unknown:
        raise SystemExit(f"Track modules not configured for this world: {sorted(unknown)}")
    for mod, folder in [m_ for m_ in mods if not only or m_[0] in only]:
        print("===", folder, time.strftime("%H:%M:%S"), flush=True)
        run(mod, folder)
    print("ALL LAYOUT CONCEPTS DONE", time.strftime("%H:%M:%S"), flush=True)
