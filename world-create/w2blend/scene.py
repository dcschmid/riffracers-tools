"""Blender entry point:  blender -b -P w2blend/scene.py -- <pack_dir> <env_module> <out_dir> <mode> [shots] [samples]
mode: blend (build + save) | render (build + render shots) | mask (road mask + top-down)"""
import json, math, os, sys, time
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import importlib
from world_create import get_project
PROJECT = get_project()
from w2blend import core, roads
from w2blend.core import Batch, hx


def clean():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_hfn(pack):
    H, bb, cell = pack["H"], pack["bbox"], float(pack["cell"])
    ny, nx = H.shape

    def f(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        u, v = (x - bb[0]) / cell, (y - bb[1]) / cell
        i0 = np.clip(np.floor(u).astype(int), 0, nx - 2)
        j0 = np.clip(np.floor(v).astype(int), 0, ny - 2)
        fu, fv = np.clip(u - i0, 0, 1), np.clip(v - j0, 0, 1)
        return (H[j0, i0] * (1 - fu) + H[j0, i0 + 1] * fu) * (1 - fv) + (H[j0 + 1, i0] * (1 - fu) + H[j0 + 1, i0 + 1] * fu) * fv
    return f


def far_sea_grad(B, bb, fz, th, near_vertices=None):
    """Open sea around the terrain box as a grid with cell size growing with distance (log-spaced). Colour = distance gradient (shallow turquoise -> deep blue -> atmospheric haze towards the horizon)
    with large-scale reef / current patches and per-cell jitter, so the sea has depth instead of one flat colour."""
    x0, y0, x1, y1 = [float(v) for v in bb]
    near_x = near_y = near_h = None
    if near_vertices is not None:
        # Match the ACTUAL finite near-water grid, not the terrain bbox. Every
        # boundary vertex is shared in position/height; overlapping reflective
        # sheets produced a black line by reflecting each other's undersides.
        nv = np.asarray(near_vertices)
        near_x, near_y = np.unique(nv[:, 0]), np.unique(nv[:, 1])
        near_h = nv[:, 2].reshape(len(near_y), len(near_x))
        x0, x1, y0, y1 = float(near_x[0]), float(near_x[-1]), float(near_y[0]), float(near_y[-1])
    D = np.concatenate([[0.0], 12.0 * 1.17 ** np.arange(0, 49)]); D = D[D < 40000]; D = np.append(D, 40000.0)      # log-spaced rings: smooth gradient, cell size grows with distance
    xs = np.concatenate([x0 - D[::-1], near_x if near_x is not None else [x0, x1], x1 + D]); xs = np.unique(xs)
    ys = np.concatenate([y0 - D[::-1], near_y if near_y is not None else [y0, y1], y1 + D]); ys = np.unique(ys)
    stops = [(0, "#0a60ae"), (150, "#0a62b0"), (500, "#0e72bc"), (1200, "#1d86c8"), (2500, "#4f9fd2"), (5000, "#8fc0dc"), (9000, "#bcd8e4"), (16000, "#dbe6e6"), (40000, "#ece4d2")]
    sd = np.array([d for d, _ in stops]); sc = np.array([hx(c) for _, c in stops])
    rg = np.random.default_rng(17)

    def col_at(d):
        i = int(np.clip(np.searchsorted(sd, d) - 1, 0, len(sd) - 2))
        t = float(np.clip((d - sd[i]) / (sd[i + 1] - sd[i]), 0, 1))
        return sc[i] * (1 - t) + sc[i + 1] * t
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            ax, bx, ay, by = xs[i], xs[i + 1], ys[j], ys[j + 1]
            if ax >= x0 - 1e-6 and bx <= x1 + 1e-6 and ay >= y0 - 1e-6 and by <= y1 + 1e-6:
                continue
            cx, cy = (ax + bx) / 2, (ay + by) / 2
            dx = max(x0 - cx, 0, cx - x1); dy = max(y0 - cy, 0, cy - y1)
            d = math.hypot(dx, dy)
            c = col_at(d)
            patch = math.sin(cx * 0.0042 + 1.3) * math.cos(cy * 0.0036) + 0.6 * math.sin(cx * 0.0101 + cy * 0.0083)
            if 40 < d < 2200:
                c = c * (1 + 0.07 * patch * min((d - 40) / 200, 1.0))
            c = c * rg.uniform(0.985, 1.015)
            def zz(px, py):                                                   # shared corner heights (no cracks): gentle swell noise that fades with distance, matches the near water mesh
                dxx = max(x0 - px, 0, px - x1); dyy = max(y0 - py, 0, py - y1)
                dd = math.hypot(dxx, dyy)
                if near_h is not None:
                    qx, qy = float(np.clip(px, x0, x1)), float(np.clip(py, y0, y1))
                    if qy <= y0:
                        zh = float(np.interp(qx, near_x, near_h[0]))
                    elif qy >= y1:
                        zh = float(np.interp(qx, near_x, near_h[-1]))
                    elif qx <= x0:
                        zh = float(np.interp(qy, near_y, near_h[:, 0]))
                    else:
                        zh = float(np.interp(qy, near_y, near_h[:, -1]))
                    return fz + (zh-fz) * math.exp(-dd/45.0)
                h_ = math.sin(px * 12.9898 + py * 78.233) * 43758.5453
                # The old per-cell 0.12 m lift stopped abruptly after 30 m,
                # exposing a dark slit at grazing angles. Shared corner heights
                # must include the lift continuously too.
                return fz + 0.12 * math.exp(-dd / 45.0) + 0.14 * math.exp(-dd / 90.0) * ((h_ - math.floor(h_)) - 0.5) * 2
            ov = 0.0 if near_h is not None else (12.0 if d < 30 else 0.0)
            lift = 0.0
            ax2 = ax - (ov if abs(ax - x1) < 1e-6 else 0); bx2 = bx + (ov if abs(bx - x0) < 1e-6 else 0)      # extend INTO the terrain box so that no gap remains
            ay2 = ay - (ov if abs(ay - y1) < 1e-6 else 0); by2 = by + (ov if abs(by - y0) < 1e-6 else 0)
            qv = np.array([[ax2, ay2, zz(ax, ay) + lift], [bx2, ay2, zz(bx, ay) + lift], [bx2, by2, zz(bx, by) + lift], [ax2, by2, zz(ax, by) + lift]])
            flip = (i + j) % 2
            B["water"].add(qv, [[0, 1, 2], [0, 2, 3]] if flip else [[0, 1, 3], [1, 2, 3]], c)


def bird_glow(lay, B, th):
    """Glowing outline ribbons along both road edges. Hidden by default; shots with `glow: true` show them (dark roads on dark ground are unreadable from above)."""
    c_main, c_alt = hx(th["glow"]), hx(th.get("glow2", th["glow"]))
    for e in lay["edges"]:
        if e.get("airborne") or not e.get("polygon", True):
            continue
        c = np.array(e["centerline"], float)
        if len(c) < 3:
            continue
        w = np.array(e["widths"], float)
        t = np.gradient(c[:, :2], axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nrm = np.stack([-t[:, 1], t[:, 0]], 1)
        col = c_main if e["kind"] in ("main", "jump_approach", "jump_ramp", "jump_landing", "jump_exit") else (c_alt if e["kind"] == "shortcut" else c_main * 0.35)
        for sg in (-1, 1):
            o0, o1 = w / 2 + 0.6, w / 2 + 3.6
            L = np.column_stack([c[:, :2] + nrm * (sg * o0)[:, None], c[:, 2] - 0.04])
            R = np.column_stack([c[:, :2] + nrm * (sg * o1)[:, None], c[:, 2] - 0.04])
            V, F = core.strip(L, R)
            B["glow"].add(V, F, col)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    pack_dir, env_name, out_dir, mode = argv[:4]
    shots = argv[4].split(",") if len(argv) > 4 and argv[4] else []
    samples = int(argv[5]) if len(argv) > 5 else 96
    t0 = time.time()
    clean()
    lay = json.load(open(f"{pack_dir}/layout.json"))
    masks = json.load(open(f"{pack_dir}/masks.json"))
    props = json.load(open(f"{pack_dir}/props.json"))
    cams = json.load(open(f"{pack_dir}/cams.json"))
    z = np.load(f"{pack_dir}/terrain.npz")
    pack = dict(H=z["H"], bbox=z["bbox"], cell=z["cell"])
    Hfn = make_hfn(pack)
    env = importlib.import_module(f"w2blend.{env_name}")
    th = env.theme()
    rng = np.random.default_rng(99)
    B = {k: Batch() for k in ("road", "solid", "emit", "glass", "people", "terrain", "water", "mask", "glow", "ghost")}
    B["terrain"].add(z["V"], z["F"], z["C"])
    B["water"].add(z["wV"], z["wF"], z["wC"])
    if "far_sea" in th:
        bb = pack["bbox"]; R = 40000.0
        fz = th.get('far_z', -1.4)
        if th.get("far_ring"):                       # ring around the terrain bbox (a full plane would hide pits and channels)
            x0, y0, x1, y1 = [float(v) for v in bb]
            ins = float(th.get("far_ring_inset", 0.0))   # let the ring slip under the terrain edge (a flush seam lets the sky shine through)
            if ins > 0:
                ins = max(ins, float(pack["cell"]) + 1.0)      # the last terrain column can end up to one cell short of the bbox
            rects = [(-R, -R, R, y0 + ins), (-R, y1 - ins, R, R), (-R, y0 + ins, x0 + ins, y1 - ins), (x1 - ins, y0 + ins, R, y1 - ins)]
            fz += float(th.get("far_ring_dz", 0.0))
            for (a0, b0, a1, b1) in rects:
                qv = np.array([[a0, b0, fz], [a1, b0, fz], [a1, b1, fz], [a0, b1, fz]])
                B["solid"].add(qv, [[0, 1, 2], [0, 2, 3]], hx(th["far_sea"]))
        elif th.get("far_grad"):
            if not os.environ.get("W2_NOFAR"):
                far_sea_grad(B, bb, fz, th, z["wV"])
        else:
            qv = np.array([[-R, -R, fz], [R, -R, fz], [R, R, fz], [-R, R, fz]])
            B["water"].add(qv, [[0, 1, 2], [0, 2, 3]], hx(th["far_sea"]))
    if getattr(env, "OWN_ROADS", False):
        env.build_roads(lay, masks, Hfn, th, B, rng)
    else:
        roads.build_roads(lay, masks, Hfn, th, B, rng)
    if th.get("glow"):
        bird_glow(lay, B, th)
    if not getattr(env, "OWN_ROADS", False):
        roads.start_grid(lay, th, B)
        roads.garage_bays(lay, th, B, rng)
        roads.stands(lay, th, B, Hfn, rng)
    if not getattr(env, "OWN_GANTRY", False):
        roads.sf_gantry(lay, th, B)
    if mode != "mask" or not getattr(env, "OWN_ROADS", False):
        env.build_env(lay, props, Hfn, th, B, rng)
    # karts on the grid
    for k, g in enumerate([] if th.get("no_karts") else lay["start_finish"]["grid"]):
        c = th["team_colors"]
        core.kart(B["people"], g["xy"][0], g["xy"][1], 0.0, 0.0, c[k % len(c)], "#ffffff", c[(k + 4) % len(c)])
    for kk in [k for c in cams.values() if isinstance(c, dict) for k in c.get("karts", [])]:
        c = th["team_colors"]
        core.kart(B["people"], kk[0], kk[1], 0.0, kk[2], c[int(kk[3]) % len(c)], "#ffffff", c[(int(kk[3]) + 5) % len(c)])
    stats = {k: v.count() for k, v in B.items()}
    print("triangles", stats, "build s", round(time.time() - t0, 1))
    if mode == "mask":
        for k in B:
            if k != "mask":
                B[k] = Batch()
        m = core.mat_vcol("mask", emit_strength=1.0)
        B["mask"].build("mask", m)
        core.set_world("night", strength=0.0, night_color="#000000")
    else:
        mats = dict(
            terrain=core.mat_vcol("terrain", rough=0.85, spec=0.15), water=(core.mat_vcol("water", emit_strength=th["water_emit"]) if th.get("water_emit") else core.mat_vcol("water", rough=th.get("water_rough", 0.14), spec=th.get("water_spec", 0.6))), road=core.mat_vcol("road", rough=th.get("road_rough", 0.82), spec=th.get("road_spec", 0.2)),
            solid=core.mat_vcol("solid", rough=0.58, spec=0.3), emit=core.mat_vcol("emit", emit_strength=th.get("emit", 3.0)), glass=core.mat_vcol("glass", glass=True),
            people=core.mat_vcol("people", rough=0.7, spec=0.2), glow=core.mat_vcol("glow", emit_strength=th.get("glow_strength", 2.4)), ghost=core.mat_vcol("ghost", emit_strength=th.get("ghost_strength", 3.0)))
        if hasattr(env, "water_material"):
            mats["water"] = env.water_material(th)
        for k, b in B.items():
            if k == "mask":
                continue
            b.build(k, mats[k])
        for nm_ in ("glow", "ghost"):
            go = bpy.data.objects.get(nm_)
            if go is not None:
                go.hide_render = True
        if th["sky"] in ("day", "gradient", "golden"):
            core.set_world(th["sky"], th["sun_elev"], th["sun_rot"], strength=th.get("sky_strength", 0.5), haze=th.get("haze", 0.0), exposure=th.get("exposure", 0.0), aerosol=th.get("aerosol"), air=th.get("air"), dust=th.get("dust"), cam_strength=th.get("sky_cam"))
            core.add_sun(th["sun_elev"], th["sun_rot"], th["sun_energy"], color=tuple(th.get("sun_color", (1.0, 0.95, 0.88))), angle=th.get("sun_angle", 0.8))
        else:
            core.set_world(th.get("night_sky", "night"), cam_strength=th.get("sky_cam"), strength=th.get("world_strength", 0.6), night_color=th.get("night_color", "#0b1230"), haze=th.get("haze", 0.0), exposure=th.get("exposure", 0.0))
            core.add_sun(th.get("moon_elev", 38.0), 150.0, th.get("moon", 3.6), color=(0.55, 0.66, 1.0), angle=2.0)
    core.setup_render(samples=samples if mode != "mask" else 4, res=(1920, 1080), denoise=mode != "mask", view=th.get("view", "AgX"), look=th.get("look"))
    if mode != "mask" and hasattr(env, "finalize_scene"):
        env.finalize_scene(lay, cams, th)
    os.makedirs(out_dir, exist_ok=True)
    if mode in ("blend", "render") and not os.environ.get("W2_NO_SAVE"):
        bdir = lay.get("master_dir") or (os.path.join(os.path.dirname(out_dir.rstrip("/")), "blender") if out_dir.rstrip("/").endswith("renders") else out_dir)
        os.makedirs(bdir, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=f"{bdir}/track-preview.blend", compress=True)
    if mode == "render":
        for name in [x for x in shots if x != "topdown"]:
            cd = cams[name]
            if hasattr(env, "prepare_shot"):
                env.prepare_shot(name, cd)
            sc = bpy.context.scene
            res = [1920, 1080] if os.environ.get("W2_QUICK") else cd.get("res", [1920, 1080])
            sc.render.resolution_x, sc.render.resolution_y = res
            if cd.get("ortho"):
                cam = core.camera(cd["loc"], cd["target"], cd.get("lens", 50), ortho=cd["ortho"])
                cam.rotation_euler = (0, 0, 0)
            else:
                cam = core.camera(cd["loc"], cd["target"], cd.get("lens", 32), dof=cd.get("dof"))
            for nm_ in ("glow", "ghost"):
                go = bpy.data.objects.get(nm_)
                if go is not None:
                    go.hide_render = not cd.get(nm_, False)
                    if os.environ.get("W2_DEBUG"):
                        import mathutils
                        bb = [go.matrix_world @ mathutils.Vector(c) for c in go.bound_box]
                        print("DBG", nm_, "hide", go.hide_render, "z", round(min(v.z for v in bb), 1), round(max(v.z for v in bb), 1), "x", round(min(v.x for v in bb)), round(max(v.x for v in bb)), "mat", go.data.materials[0].name if go.data.materials else None)
            out_path = cd.get("out") or f"{out_dir}/{cd.get('file', name)}.png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            t1 = time.time()
            core.render(cam, out_path, samples=cd.get("samples", samples))
            print("rendered", name, round(time.time() - t1, 1), "s")
    if mode == "mask" or (mode == "render" and "topdown" in shots):
        td = cams["topdown"]
        sc = bpy.context.scene
        sc.render.resolution_x, sc.render.resolution_y = td["res"]
        cam = core.camera([td["cx"], td["cy"], 1800], [td["cx"], td["cy"], 0], 50, ortho=td["scale"])
        cam.rotation_euler = (0, 0, 0)
        core.render(cam, f"{out_dir}/{'roadmask' if mode == 'mask' else 'topdown'}.png", samples=8 if mode == "mask" else 48)
    print("done", round(time.time() - t0, 1))


main()
