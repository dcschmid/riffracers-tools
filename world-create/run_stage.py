"""Stage runner: python run_stage.py <track-module> <stage...>   stages: layout, blueprints"""
import sys, json, importlib, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from world_create import get_project, gpu_job, contained
PROJECT = get_project()
from w2lib import validate as V, draw
W2 = str(PROJECT.root)
FOLDERS = {s["id"]: s["folder"] for s in PROJECT.specs()}

def build(mod):
    global MOD
    MOD = mod
    module = PROJECT.import_track(mod)
    T = module.make()
    lay = T.build()
    if hasattr(module, "DRIVE_CONTRACT"):
        lay["drivability_contract"] = module.DRIVE_CONTRACT
    val = V.validate(lay)
    FOLDERS[lay["id"]] = PROJECT.folder(mod, lay["id"], module)
    return T, lay, val

def stage_layout(T, lay, val):
    d = f"{W2}/{FOLDERS[lay['id']]}/planning"
    os.makedirs(d, exist_ok=True)
    traps = lay["traps"]
    json.dump({k: v for k, v in lay.items() if k != "traps"}, open(f"{d}/track-layout.json", "w"), indent=1)
    json.dump(dict(schema="riffracers.w2.hazards/1", track=lay["id"], note="design data only - not implemented in the game", traps=traps), open(f"{d}/hazards.json", "w"), indent=1)
    json.dump(val, open(f"{d}/validation.json", "w"), indent=1, default=str)

def stage_blueprints(T, lay, val):
    d = f"{W2}/{FOLDERS[lay['id']]}/blueprints"
    os.makedirs(d, exist_ok=True)
    draw.draw_clean(lay, f"{d}/topdown-clean.png")
    draw.draw_annotated(lay, val, f"{d}/topdown-annotated.png")
    draw.draw_pitlane(lay, f"{d}/pitlane-detail.png")
    draw.draw_shortcuts_jumps(lay, f"{d}/shortcuts-jumps.png")
    draw.draw_traps(lay, val, f"{d}/trap-positions.png")

def stage_pack(T, lay, val, mod=None):
    if val["critical"] or val["fails"]:
        raise RuntimeError("Resolve geometry validation failures before packing a render scene")
    import numpy as np
    from w2lib import envdata, palettes
    mod = mod or MOD
    m = importlib.import_module(f"tracks.{mod}")
    W = envdata.World(lay, seed=lay.get("seed", 7) + 6)
    pal = getattr(palettes, lay["terrain"].get("palette", "coastal"))
    pal = pal if callable(pal) and lay["terrain"].get("palette", "coastal") != "flat_city" else palettes.flat_city(*lay["terrain"].get("ground_cols", ("#272c3c", "#2e3446")))
    V, F, C = W.terrain_mesh(dict(func=pal, drop_below=None))
    wV, wF, wC = W.water_mesh(level=lay["terrain"].get("water_level", 0.0), **lay["terrain"].get("water_cols", {}))
    d = f"{OUT}/{pack_key(lay)}/pack"
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(f"{d}/terrain.npz", V=V.astype(np.float32), F=F, C=C.astype(np.float32), wV=wV.astype(np.float32), wF=wF, wC=wC.astype(np.float32), H=W.H, bbox=np.array(W.bbox), cell=W.cell)
    json.dump({k: v for k, v in lay.items()}, open(f"{d}/layout.json", "w"))
    json.dump(W.barrier_masks(), open(f"{d}/masks.json", "w"))
    props = m.props(W, lay)
    json.dump(props, open(f"{d}/props.json", "w"))
    cams = m.cameras(W, lay)
    xs = [p for e in lay["edges"] for p in np.array(e["centerline"])[:, 0]]
    ys = [p for e in lay["edges"] for p in np.array(e["centerline"])[:, 1]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    ext = max(max(xs) - min(xs), (max(ys) - min(ys)) * 16 / 9) + 140
    cams["topdown"] = dict(cx=cx, cy=cy, scale=ext, res=[1920, int(1920 * 9 / 16)])
    FILEMAP = dict(hero="hero-16x9", overview="overview-2-5d", gameplay="gameplay-camera", pitlane="pitlane-closeup", hazard="hazard-closeup")
    for k, v in FILEMAP.items():
        if k in cams:
            cams[k]["file"] = v
    for k, v in cams.items():
        if isinstance(v, dict) and v.get("rel_out"):
            v["out"] = f"{W2}/{FOLDERS[lay['id']]}/{v['rel_out']}"
    json.dump(cams, open(f"{d}/cams.json", "w"))
    print(" pack:", len(V), "terrain verts,", len(F), "faces,", len(props), "props ->", d)

OUT = str(PROJECT.cache_dir)
MOD = None


def pack_key(lay):
    """Pack folder name: a track module may set PACK_ID so that an alternative pass (v3) does not overwrite the pack of the first one."""
    key = getattr(importlib.import_module(f"tracks.{MOD}"), "PACK_ID", None) or lay["id"]
    contained(PROJECT.cache_dir, key)
    return key
BLENDER = PROJECT.tool("blender", "blender")

def stage_bake(T, lay, val, mode="blend", shots="", samples=96):
    if val["critical"] or val["fails"]:
        raise RuntimeError("Resolve geometry validation failures before Blender output")
    import subprocess
    d = f"{OUT}/{pack_key(lay)}/pack"
    env = getattr(importlib.import_module(f"tracks.{MOD}"), "ENV", None) or (lay["theme_env"] if "theme_env" in lay else lay["id"].split("-")[0])
    outd = f"{W2}/{FOLDERS[lay['id']]}/" + ("blender" if mode == "blend" else "renders")
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [BLENDER, "-b", "-P", f"{here}/w2blend/scene.py", "--", d, env, outd if mode != "mask" else f"{OUT}/{pack_key(lay)}", mode, shots, str(samples)]
    print(" ", " ".join(cmd[-6:]))
    with gpu_job():
        r = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("triangles", "rendered", "done", "Error", "Traceback")) or "Error" in l)[-2500:])
    if r.returncode or "Traceback" in r.stderr or "Traceback" in r.stdout:
        print(r.stderr[-3500:])
        raise RuntimeError(f"Blender stage failed with exit code {r.returncode}")

def stage_overlay(T, lay, val):
    import numpy as np
    from PIL import Image, ImageDraw
    from shapely.geometry import Polygon
    cams = json.load(open(f"{OUT}/{pack_key(lay)}/pack/cams.json"))
    td = cams["topdown"]
    folder = f"{W2}/{FOLDERS[lay['id']]}/renders"
    top = Image.open(f"{folder}/topdown.png").convert("RGB")
    msk = np.array(Image.open(f"{OUT}/{pack_key(lay)}/roadmask.png").convert("L")) > 100
    Wp, Hp = top.size
    k = Wp / td["scale"]
    tf = lambda xy: ((np.asarray(xy)[:, 0] - td["cx"]) * k + Wp / 2, Hp / 2 - (np.asarray(xy)[:, 1] - td["cy"]) * k)
    lay_arr = np.zeros((Hp, Wp), np.uint8)
    from w2lib.envdata import edge_poly
    for e in lay["edges"]:
        if not e.get("polygon", True):
            continue
        pl = edge_poly(e)
        im = Image.new("L", (Wp, Hp), 0)
        d = ImageDraw.Draw(im)
        for g in (pl.geoms if hasattr(pl, "geoms") else [pl]):
            x, y = tf(np.array(g.exterior.coords)); d.polygon(list(zip(x, y)), fill=255)
            for h in g.interiors:
                hx_, hy_ = tf(np.array(h.coords)); d.polygon(list(zip(hx_, hy_)), fill=0)
        lay_arr = np.maximum(lay_arr, np.array(im))
    lay_img = Image.fromarray(lay_arr)
    lm = np.array(lay_img) > 128
    inter, union = (lm & msk).sum(), (lm | msk).sum()
    iou = inter / max(union, 1)
    spill = (msk & ~lm).sum() / max(msk.sum(), 1)
    miss = (lm & ~msk).sum() / max(lm.sum(), 1)
    ov = top.copy()
    dd = ImageDraw.Draw(ov, "RGBA")
    for e in lay["edges"]:
        c = np.array(e["centerline"])
        x, y = tf(c[:, :2])
        col = (255, 60, 60, 255) if e["kind"] == "main" else (255, 160, 0, 255) if e["kind"] == "shortcut" else (60, 220, 255, 255) if e["kind"].startswith("pit") else (230, 60, 220, 255)
        dd.line(list(zip(x, y)), fill=col, width=2)
        if e.get("polygon", True):
            for g in (edge_poly(e).geoms if hasattr(edge_poly(e), "geoms") else [edge_poly(e)]):
                px, py = tf(np.array(g.exterior.coords)); dd.line(list(zip(px, py)), fill=(255, 255, 255, 140), width=1)
    dd.rectangle([10, 10, 640, 112], fill=(255, 255, 255, 220))
    f = ImageFont_load()
    dd.text((20, 16), f"geometry overlay: layout polygons (white), centrelines (red/orange/cyan/magenta)", font=f, fill=(20, 30, 40, 255))
    dd.text((20, 44), f"IoU road render vs layout = {iou:.3f}   render outside layout {spill*100:.1f} %   layout missing in render {miss*100:.1f} %", font=f, fill=(20, 30, 40, 255))
    dd.text((20, 72), "verified by ID-mask of the Blender road surfaces (not by eye)", font=f, fill=(20, 30, 40, 255))
    ov.save(f"{folder}/geometry-overlay.png")
    # Keep the top-down render as evidence; cleanup is explicit via gio trash.
    res = dict(iou=round(float(iou), 4), render_outside_layout=round(float(spill), 4), layout_missing_in_render=round(float(miss), 4), verified=bool(iou > 0.93 and spill < 0.06 and miss < 0.06))
    json.dump(res, open(f"{W2}/{FOLDERS[lay['id']]}/planning/render-verification.json", "w"), indent=1)
    print(" overlay:", res)


def stage_docs(T, lay, val, status="REVIEW_REQUIRED"):
    from w2lib import docs
    m = importlib.import_module(f"tracks.{MOD}")
    d = f"{W2}/{FOLDERS[lay['id']]}"
    ver = None
    vp = f"{d}/planning/render-verification.json"
    if os.path.exists(vp):
        ver = json.load(open(vp))
    fidp = str(contained(PROJECT.track_dir(MOD), getattr(m, "FIDELITY_FILE", "concepts/layout-fidelity.json")))
    fid = json.load(open(fidp)) if os.path.exists(fidp) else None
    open(f"{d}/planning/track-identity.md", "w").write(docs.identity_md(lay, m.DOC, m.PALETTE, status, fid))
    open(f"{d}/planning/gameplay-design.md", "w").write(docs.gameplay_md(lay, m.DOC, val))
    open(f"{d}/planning/validation-report.md", "w").write(docs.validation_md(lay, val, ver))
    if hasattr(m, "REVIEW"):
        txt, total = docs.review_md(lay, m.REVIEW, status, val)
        open(f"{d}/review.md", "w").write(txt)
        json.dump(dict(track=lay["id"], status=status, weighted_score=round(total, 2), scores=m.REVIEW["scores"], geometry_validated=val["geometry_validated"], critical=val["critical"]),
                  open(f"{d}/planning/status.json", "w"), indent=1)
    if getattr(m, "AUDIT_NOTE", None):
        for name in ("planning/gameplay-design.md", "planning/validation-report.md", "review.md"):
            path = f"{d}/{name}"
            with open(path) as source:
                first, rest = source.read().split("\n", 1)
            with open(path, "w") as output:
                note = m.AUDIT_NOTE.replace(") beziehungsweise im planning-Ordner `drivability-audit.md`", ")")
                if name.startswith("planning/"):
                    note = note.replace("](planning/drivability-audit.md)", "](drivability-audit.md)")
                output.write(first + "\n\n" + note + "\n" + rest)
    print(" docs written:", d)


def ImageFont_load():
    from PIL import ImageFont
    return ImageFont.truetype("/usr/share/fonts/google-noto/NotoSans-SemiBold.ttf", 22)


if __name__ == "__main__":
    mod, stages = sys.argv[1], sys.argv[2:] or ["layout", "blueprints"]
    MOD = mod
    T, lay, val = build(mod)
    print(lay["name"], "critical", val["critical"], "warn", val["warnings"])
    for s in stages:
        t0 = time.time()
        if ":" in s:
            name, arg = s.split(":", 1)
            kw = dict(a.split("=") for a in arg.split(";")) if arg else {}
            if "samples" in kw: kw["samples"] = int(kw["samples"])
            globals()[f"stage_{name}"](T, lay, val, **kw)
        else:
            globals()[f"stage_{s}"](T, lay, val)
        print(" stage", s, f"{time.time()-t0:.1f}s")
