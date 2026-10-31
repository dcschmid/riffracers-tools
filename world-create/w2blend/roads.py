"""Roads, kerbs, barriers, markings, elevated decks and pillars, start grid, pit garages and stands - all generated from layout.json."""
from __future__ import annotations

import math

import numpy as np

from .core import (Batch, box, frustum, hx, people, kart, prism_poly, rotz, sphere, strip, tri_quads, xf, FUN, RNG)


def _sf_w(lay):
    e = next(x for x in lay["edges"] if x["id"] == lay["main_loop"][0])
    return float(e["widths"][0])


def normals(c):
    t = np.gradient(c[:, :2], axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return np.column_stack([-t[:, 1], t[:, 0]]), t


def offset(c, nrm, d, dz=0.0):
    p = c.copy()
    p[:, :2] += nrm * (d[:, None] if np.ndim(d) else d)
    p[:, 2] += dz
    return p


def strip_between(c, nrm, d0, d1, dz, col, B, every=None, alt=None):
    """Band between lateral offsets d0..d1 (arrays or scalars) along centreline c; optional alternating colours (every N quads)."""
    a, b = offset(c, nrm, d0, dz), offset(c, nrm, d1, dz)
    n = len(c)
    V = np.vstack([a, b])
    Q = np.array([[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
    F = tri_quads(Q)
    m = len(Q)
    if alt is not None and every:
        cols = np.array([col if (i // every) % 2 == 0 else alt for i in range(m)])
    else:
        cols = np.tile(col, (m, 1))
    cols2 = np.vstack([cols, cols])  # tri_quads stacks [0,1,2] then [0,2,3]
    B.add(V, F, cols2)


def wall_band(c, nrm, d, z0, z1, col, B, side_out=1.0):
    """Vertical band along centreline at lateral offset d (scalar or array) from height z0 (array or scalar, absolute offset from c.z) to z1."""
    a = offset(c, nrm, d, z0)
    b = offset(c, nrm, d, z1)
    n = len(c)
    V = np.vstack([a, b])
    Q = np.array([[i, i + 1, n + i + 1, n + i] for i in range(n - 1)])
    F = tri_quads(Q)
    if side_out < 0:
        F = F[:, ::-1]
    B.add(V, F, np.tile(col, (len(F), 1)) if np.ndim(col) == 1 else col)


def build_roads(lay, masks, Hfn, th, B, rng):
    E = {e["id"]: e for e in lay["edges"]}
    zones = lay.get("zones", [])
    for e in lay["edges"]:
        if not e.get("polygon", True):
            continue
        c = np.array(e["centerline"])
        w = np.array(e["widths"])
        n = len(c)
        nrm, tang = normals(c)
        kind = e["kind"]
        base = {"main": th["asphalt"], "pit_lane": th["pit_floor"], "pit_entry": th["pit_floor"], "pit_exit": th["pit_floor"], "shortcut": th.get("shortcut", th["asphalt"]),
                "jump_ramp": th["ramp"], "jump_landing": th["landing"], "jump_approach": th["asphalt"], "jump_exit": th["asphalt"]}[kind]
        if e.get("structure") == "pier":
            base = th.get("pier", base)
        alt = {"shortcut": th.get("shortcut_alt"), "jump_ramp": th.get("ramp_alt"), "jump_landing": th.get("landing_alt")}.get(kind) if e.get("structure") != "pier" else th.get("pier_alt")
        patch = base * rng.uniform(0.96, 1.04)
        strip_between(c, nrm, -w / 2, w / 2, 0.0, base, B["road"], every=(2 if kind in ("shortcut", "jump_ramp", "jump_landing") else 8), alt=alt if alt is not None else base * 0.94)
        if "mask" in B:
            strip_between(c, nrm, -w / 2, w / 2, 0.0, np.array([1.0, 1.0, 1.0]), B["mask"])
        # surface-zone overlays (polished / slippery / wet / cobble ...)
        if kind == "main" and th.get("zone_cols"):
            cs = e["s0"] + np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
            for zn in lay.get("zones", []):
                col = th["zone_cols"].get(zn["kind"])
                if col is None:
                    continue
                sel = (cs >= zn["s0"]) & (cs <= zn["s1"]) if zn["s1"] >= zn["s0"] else (cs >= zn["s0"]) | (cs <= zn["s1"])
                ii = np.where(sel)[0]
                if len(ii) > 2:
                    strip_between(c[ii], nrm[ii], -w[ii] / 2, w[ii] / 2, 0.025, col, B["road"], every=th.get("zone_every", {}).get(zn["kind"], 6), alt=th.get("zone_alt", {}).get(zn["kind"], col * 0.93))
                    if th.get("zone_glow") and zn["kind"] in th["zone_glow"]:
                        for sd in (-1, 1):
                            strip_between(c[ii], nrm[ii], sd * (w[ii] / 2 - 1.4), sd * (w[ii] / 2 - 1.0), 0.05, th["zone_glow"][zn["kind"]], B["emit"])
        # lane markings
        line = th["line"]
        if kind in ("main", "jump_approach", "jump_exit"):
            strip_between(c, nrm, -w / 2 + 0.55, -w / 2 + 0.9, 0.035, line, B["road"])
            strip_between(c, nrm, w / 2 - 0.9, w / 2 - 0.55, 0.035, line, B["road"])
            for i in range(0, n - 4, 6):
                sl = slice(i, i + 3)
                if len(c[sl]) > 1:
                    cc, nn = c[sl], nrm[sl]
                    strip_between(cc, nn, -0.15, 0.15, 0.036, th["line_center"], B["road"])
        elif kind in ("pit_lane", "pit_entry", "pit_exit"):
            strip_between(c, nrm, -w / 2 + 0.3, -w / 2 + 0.6, 0.035, hx("#ffd23f"), B["road"])
            strip_between(c, nrm, w / 2 - 0.6, w / 2 - 0.3, 0.035, hx("#ffd23f"), B["road"])
        elif kind in ("jump_ramp", "jump_landing"):
            for i in range(0, n - 1, 3):
                sl = slice(i, i + 2)
                strip_between(c[sl], nrm[sl], -w[sl] / 2 + 1.0, w[sl] / 2 - 1.0, 0.04, th["chevron"], B["road"])
        # kerbs + barriers
        mk = masks.get(e["id"], {})
        ml, mr = np.array(mk.get("l", [True] * n)), np.array(mk.get("r", [True] * n))
        gapz = c[:, 2] - Hfn(c[:, 0], c[:, 1])
        for side, mm in ((1, ml), (-1, mr)):
            d_in = side * (w / 2)
            for i0 in range(0, n - 1, 2):
                sl = slice(i0, i0 + 3)
                if len(c[sl]) < 2 or not mm[sl].all():
                    continue
                cc, nn, ww = c[sl], nrm[sl], w[sl]
                ci = (i0 // 2) % 2
                if kind in ("main", "shortcut", "jump_approach", "jump_exit", "pit_lane", "pit_entry", "pit_exit") and e.get("structure") != "pier":
                    strip_between(cc, nn, side * ww / 2, side * (ww / 2 + 1.1), 0.07, th["kerb_a"] if ci == 0 else th["kerb_b"], B["road"])
                bar_off = ww / 2 + 1.75 if kind not in ("pit_lane",) else ww / 2 + 0.5
                pc = (th["barrier_a"], th["barrier_b"], th["barrier_c"])[(i0 // 6) % 3]
                hh = 0.95 if kind != "pit_lane" else 0.7
                wall_band(cc, nn, side * bar_off, 0.0, hh, pc, B["solid"], side_out=side)
                wall_band(cc, nn, side * (bar_off - 0.45), 0.0, hh, pc * 0.85, B["solid"], side_out=-side)
                strip_between(cc, nn, side * (bar_off - 0.45), side * bar_off, hh, pc * 1.08, B["solid"])
                if th.get("barrier_led") is not None:
                    strip_between(cc, nn, side * (bar_off - 0.38), side * (bar_off - 0.12), hh + 0.02, th["barrier_led"], B["emit"])
        # elevated deck: skirts and pillars, or retaining walls for ramps
        hi = gapz > 0.45
        if hi.any():
            ramp_like = kind in ("jump_ramp", "jump_landing", "jump_approach", "jump_exit")
            for i in range(n - 1):
                if not (hi[i] or hi[i + 1]):
                    continue
                g0, g1 = max(gapz[i], 0.0), max(gapz[i + 1], 0.0)
                for side in (1, -1):
                    p0 = c[i].copy(); p0[:2] += nrm[i] * side * w[i] / 2
                    p1 = c[i + 1].copy(); p1[:2] += nrm[i + 1] * side * w[i + 1] / 2
                    depth0 = (g0 + 0.5) if ramp_like else min(1.1, g0 + 0.3)
                    depth1 = (g1 + 0.5) if ramp_like else min(1.1, g1 + 0.3)
                    V = np.array([p0, p1, p1 - [0, 0, depth1], p0 - [0, 0, depth0]])
                    B["solid"].add(V, tri_quads([[0, 1, 2, 3]]) if side > 0 else tri_quads([[3, 2, 1, 0]]), th["deck_side"] if not ramp_like else th["ramp_side"])
                if not ramp_like and gapz[i] > 1.2:
                    v0 = c[i].copy(); v0[2] -= 1.1
                    a_ = offset(c[i:i + 2], nrm[i:i + 2], -w[i:i + 2] / 2, -1.1)
                    b_ = offset(c[i:i + 2], nrm[i:i + 2], w[i:i + 2] / 2, -1.1)
                    V, F = strip(a_, b_)
                    B["solid"].add(V, F[:, ::-1], th["deck_side"] * 0.7)
            if not ramp_like:
                pil_ok = mk.get("pil", [True] * n)
                step = 8
                for i in range(2, n - 2, step):
                    if gapz[i] > 1.6 and pil_ok[i]:
                        z0 = Hfn(c[i, 0], c[i, 1]) - 0.8
                        hgt = c[i, 2] - 1.1 - z0
                        r1 = 1.5 + 0.03 * hgt
                        for sd in ((-1, 1) if w[i] > 18 else (0,)):
                            pos = c[i].copy()
                            pos[:2] += nrm[i] * sd * w[i] * 0.28
                            pos[2] = z0
                            v, f = frustum(pos, r1 * 1.25, r1, hgt, 8, rot=22.5)
                            B["solid"].add(v, f, th["pillar"])
                            v, f = box((pos[0], pos[1], pos[2] + hgt - 0.25), (r1 * 2.6, w[i] * 0.34 if sd else w[i] * 0.7, 0.3), rot=math.degrees(math.atan2(tang[i][1], tang[i][0])) + 90)
                            B["solid"].add(v, f, th["pillar"] * 1.1)


def start_grid(lay, th, B):
    w = _sf_w(lay)
    line = th["line"]
    n = 12
    for k in range(n):
        for j in range(2):
            yy = -w / 2 + 1.1 + k * (w - 2.2) / n
            col = hx("#101216") if (k + j) % 2 == 0 else hx("#f4f6f8")
            v, f = box((j * 1.5 - 0.75 + 0.0, yy + (w - 2.2) / n / 2, 0.03), (1.5, (w - 2.2) / n, 0.03))
            B["road"].add(v, f, col)
    for g in lay["start_finish"]["grid"]:
        x, y = g["xy"]
        for (dx, dy, sx, sy) in ((0, 1.1, 4.4, 0.18), (0, -1.1, 4.4, 0.18), (-2.2, 0, 0.18, 2.4)):
            v, f = box((x + dx, y + dy, 0.035), (sx, sy, 0.03))
            B["road"].add(v, f, line)


def garage_bays(lay, th, B, rng, karts_in=(2, 5, 8, 11)):
    p = lay["pit"]
    sd = p["side"]
    lane_y = p["lane_y"]
    team = [hx(c) for c in th["team_colors"]]
    for bx in p["boxes"]:
        cx, cy = bx["center"]
        w, d = bx["size"]
        i = bx["index"] - 1
        tc = team[i % len(team)]
        front = -sd  # direction from garage towards the lane (lane is at lower |y| than the bay for side=-1 -> +y)
        # floor
        v, f = box((cx, cy, 0.0), (w, d, 0.14)); B["road"].add(v, f, th["pit_floor"] * 0.8)
        # back wall + side walls
        back_y = cy + sd * (d / 2 - 0.25)
        v, f = box((cx, back_y, 0.1), (w, 0.5, 4.4)); B["solid"].add(v, f, tc * 0.9)
        for sx in (-1, 1):
            v, f = box((cx + sx * (w / 2 - 0.15), cy, 0.1), (0.3, d, 4.4)); B["solid"].add(v, f, th["garage_wall"])
        # roof overhanging toward the lane
        roof_c = cy - sd * 0.9
        v, f = box((cx, roof_c, 4.45), (w + 0.4, d + 1.8, 0.4), taper=1.0); B["solid"].add(v, f, th["roof_a"] if i % 2 == 0 else th["roof_b"])
        # sign (emissive) above the front edge
        sy = cy - sd * (d / 2 + 0.95)
        v, f = box((cx, sy, 4.9), (w - 1.6, 0.35, 1.5)); B["emit"].add(v, f, tc * 0.9 + 0.1)
        v, f = box((cx, sy + sd * 0.05, 4.85), (w - 1.3, 0.3, 1.7)); B["solid"].add(v, f, hx("#1a1f2a"))
        # tyre stacks, tool cart
        for k, sx in enumerate((-1, 1)):
            for h_ in range(3):
                v, f = frustum((cx + sx * (w / 2 - 1.2), cy + sd * (d / 2 - 1.1), 0.14 + h_ * 0.34), 0.42, 0.42, 0.34, 8)
                B["solid"].add(v, f, hx("#23262e") if h_ % 2 == 0 else hx("#32363f"))
        v, f = box((cx + 0.2, cy + sd * 1.2, 0.14), (1.1, 0.7, 0.85)); B["solid"].add(v, f, hx("#d9422f"))
        # crew
        spots = []
        for k in range(3):
            spots.append((cx - 2.2 + k * 1.6 + rng.uniform(-0.2, 0.2), cy - sd * 0.9 + rng.uniform(-0.3, 0.3), 0.14, 90 if front > 0 else 270))
        people(B["people"], spots, rng, palette=[th["team_colors"][i % len(th["team_colors"])], "#ffffff", "#1a1f2a"])
        if bx["index"] in karts_in:
            kart(B["people"], cx, cy - sd * 0.2, 0.14, 90 if front > 0 else 270, th["team_colors"][i % len(th["team_colors"])], "#ffffff", th["team_colors"][(i + 3) % len(th["team_colors"])])
    b = p["building"]
    cx, cy = b["center"]
    sx, sy = b["size"]
    v, f = box((cx, cy, 0.0), (sx, sy, b["height"])); B["solid"].add(v, f, th["pit_building"])
    v, f = box((cx, cy, b["height"]), (sx + 1.0, sy + 1.0, 0.7)); B["solid"].add(v, f, th["roof_a"])
    for k in range(int(sx // 9)):
        v, f = box((cx - sx / 2 + 6 + k * 9, cy - sd * (sy / 2 + 0.06), 3.2), (6.5, 0.1, 2.0)); B["emit"].add(v, f, th.get("window", hx("#7fe8ff")))


def stands(lay, th, B, Hfn, rng, density=0.86):
    for s in lay["stands"]:
        cx, cy = s["center"]
        length, depth = s["size"]
        hd = math.radians(s["heading_deg"])
        u = np.array([math.cos(hd), math.sin(hd)])
        nl = np.array([-u[1], u[0]])
        v_out = nl * s["side"]
        z0 = float(Hfn(cx, cy))
        rows, H = s["rows"], s["height"]
        dd, rh = depth / rows, H / rows
        face_deg = math.degrees(math.atan2(-v_out[1], -v_out[0]))
        sec = [hx(c) for c in th["seat_colors"]]
        for r in range(rows):
            vpos = -depth / 2 + (r + 0.5) * dd
            pc = np.array([cx, cy]) + v_out * vpos
            rot = s["heading_deg"] if True else 0
            v, f = box((pc[0], pc[1], z0), (length, dd, rh * (r + 1)), rot=0)
            # rotate around centre of the stand: build in local frame
            Vl = np.array([[-length / 2, vpos - dd / 2], [length / 2, vpos - dd / 2], [length / 2, vpos + dd / 2], [-length / 2, vpos + dd / 2]])
            for k in range(0, int(length // 6) + 1):
                a, b = -length / 2 + k * 6, min(-length / 2 + (k + 1) * 6, length / 2)
                if b - a < 0.2:
                    continue
                pl = np.array([[a, vpos - dd / 2], [b, vpos - dd / 2], [b, vpos + dd / 2], [a, vpos + dd / 2]])
                w_ = np.array([cx, cy]) + pl[:, :1] * u + pl[:, 1:] * v_out
                col = sec[(k + r // 3) % len(sec)] * (0.92 if r % 2 else 1.0)
                vv, ff = prism_poly(w_, z0, z0 + rh * (r + 1))
                B["solid"].add(vv, ff, col)
        # back wall + side wings
        bw = np.array([cx, cy]) + v_out * (depth / 2 + 0.3)
        v, f = box((bw[0], bw[1], z0), (length + 0.8, 0.6, H + 1.2), rot=s["heading_deg"]); B["solid"].add(v, f, th["stand_back"])
        # canopy
        top = H + 3.4
        for k in range(int(length // 4)):
            a = -length / 2 + k * 4
            pl = np.array([[a, -depth / 2 - 1.0], [a + 4, -depth / 2 - 1.0], [a + 4, depth / 2 + 0.4], [a, depth / 2 + 0.4]])
            w_ = np.array([cx, cy]) + pl[:, :1] * u + pl[:, 1:] * v_out
            vv, ff = prism_poly(w_, z0 + top, z0 + top + 0.35)
            B["solid"].add(vv, ff, th["canopy_a"] if k % 2 == 0 else th["canopy_b"])
        for k in range(int(length // 16) + 2):
            a = -length / 2 + 0.5 + k * min(16.0, (length - 1) / max(int(length // 16) + 1, 1))
            for vposs in (-depth / 2 - 0.6,):
                pp = np.array([cx, cy]) + u * a + v_out * vposs
                v, f = frustum((pp[0], pp[1], z0), 0.22, 0.22, top, 6)
                B["solid"].add(v, f, th["pillar"] * 1.15)
        # spectators
        spots = []
        for r in range(rows):
            vpos = -depth / 2 + (r + 0.5) * dd
            for a in np.arange(-length / 2 + 0.6, length / 2 - 0.4, 0.62):
                if rng.uniform() > density:
                    continue
                p = np.array([cx, cy]) + u * (a + rng.uniform(-0.08, 0.08)) + v_out * (vpos + rng.uniform(-0.1, 0.1))
                spots.append((p[0], p[1], z0 + rh * (r + 1), face_deg + rng.uniform(-14, 14)))
        people(B["people"], spots, rng, palette=th.get("crowd_colors", FUN))


def sf_gantry(lay, th, B):
    w = _sf_w(lay)
    for sy in (-1, 1):
        v, f = box((2.0, sy * (w / 2 + 3.0), 0.0), (1.4, 1.4, 9.5)); B["solid"].add(v, f, th["pillar"] * 1.2)
    v, f = box((2.0, 0, 9.5), (1.6, w + 8.4, 1.3)); B["solid"].add(v, f, th["roof_a"])
    n = 14
    for k in range(n):
        v, f = box((2.0, -w / 2 - 1.5 + k * (w + 3.0) / n + (w + 3.0) / n / 2, 10.85), (1.5, (w + 3.0) / n, 1.0)); B["emit"].add(v, f, hx(FUN[k % len(FUN)]) * 0.9)
