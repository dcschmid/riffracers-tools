"""Shared night-city / industrial kit: neon towers, hologram boards, lamps, tunnels, tubes, billboards. Track modules choose palettes and landmarks."""
from __future__ import annotations

import math

import numpy as np

from .core import Batch, box, frustum, hx, people, prism_poly, rotz, roty, sphere, strip, tri_quads, FUN
from .roads import normals


def _tilt_box(B, a, b, th, col, group="emit", wd=0.6, hd=0.6):
    """Thin beam between two 3D points."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-3:
        return
    v, f = box((0, 0, -hd / 2), (L, wd, hd))
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    v = v @ roty(-pitch).T @ rotz(yaw).T + (a + b) / 2
    B[group].add(v, f, col)


def tower(B, p, th, rng):
    x, y, z, w, d, h, rot = p["x"], p["y"], p["z"], p["w"], p["d"], p["h"], p["rot"]
    r = np.random.default_rng(p.get("seed", 1))
    pal = th["bld_cols"]
    body = hx(pal[r.integers(len(pal))]) * r.uniform(0.85, 1.1)
    neon = [hx(c) for c in th["neon"]]
    R = rotz(rot)
    tiers = 1 + int(h > 55) + int(h > 105)
    z0 = z
    ht = h / tiers
    for t in range(tiers):
        sw, sd = w * (1 - 0.2 * t), d * (1 - 0.2 * t)
        hh = ht * (1.15 if t == 0 else 0.95)
        v, f = box((x, y, z0), (sw, sd, hh), rot=rot, taper=1.0 if t < tiers - 1 else 0.94)
        B["solid"].add(v, f, body * (1 - 0.06 * t))
        nf = int(hh // th.get("floor", 5.0))
        for face in range(4):
            ax = (face % 2 == 0)
            half = (sd if ax else sw) / 2 + 0.06
            span = (sw if ax else sd) * 0.84
            for k in range(1, nf):
                if r.uniform() > th.get("lit", 0.55):
                    continue
                c = neon[r.integers(len(neon))] * r.uniform(0.35, 0.9) if r.uniform() < th.get("neon_ratio", 0.35) else hx(th["win_col"]) * r.uniform(0.5, 1.0)
                off = np.array([0.0, half * (1 if face < 2 else -1), 0.0]) if ax else np.array([half * (1 if face < 2 else -1), 0.0, 0.0])
                loc = (off @ R.T) + [x, y, z0 + k * th.get("floor", 5.0)]
                v, f = box((loc[0], loc[1], loc[2]), (span if ax else 0.14, 0.14 if ax else span, 1.5), rot=rot)
                B["emit"].add(v, f, c)
        z0 += hh * 0.97
    # roof neon + antenna
    nc = neon[r.integers(len(neon))]
    for sx, sy, ww, dd in ((0, 1, w * 0.78, 0.4), (0, -1, w * 0.78, 0.4), (1, 0, 0.4, d * 0.78), (-1, 0, 0.4, d * 0.78)):
        loc = (np.array([sx * w * 0.39 * (1 - 0.2 * (tiers - 1)), sy * d * 0.39 * (1 - 0.2 * (tiers - 1)), 0]) @ R.T) + [x, y, z0 + 0.2]
        v, f = box((loc[0], loc[1], loc[2]), (ww * (1 - 0.2 * (tiers - 1)), dd, 0.5), rot=rot)
        B["emit"].add(v, f, nc)
    if r.uniform() < 0.5:
        v, f = frustum((x, y, z0), 0.35, 0.12, r.uniform(7, 16), 4); B["solid"].add(v, f, hx("#cfd6e6"))
    if th.get("vertical_neon", True) and r.uniform() < 0.45:
        for sx, sy in ((1, 1), (-1, -1)):
            loc = (np.array([sx * w / 2, sy * d / 2, 0]) @ R.T) + [x, y, z]
            v, f = box((loc[0], loc[1], loc[2]), (0.5, 0.5, h * 0.96), rot=rot)
            B["emit"].add(v, f, nc * 0.9)


def lamp(B, p, th):
    sd = p["sd"]
    v, f = frustum((p["x"], p["y"], p["z"]), 0.16, 0.12, 7.0, 5); B["solid"].add(v, f, hx("#2b3140"))
    v, f = box((p["x"], p["y"], p["z"] + 7.0), (1.5, 0.5, 0.35)); B["emit"].add(v, f, hx(th["lamp"]))


def ctree(B, p, rng, th):
    h = rng.uniform(5, 9)
    v, f = frustum((p["x"], p["y"], p["z"]), 0.3, 0.2, h * 0.5, 5); B["solid"].add(v, f, hx("#4a3a30"))
    v, f = sphere((p["x"], p["y"], p["z"] + h * 0.7), h * 0.42, 6, 4); B["solid"].add(v, f, hx(th.get("tree", "#2a7a6a")) * rng.uniform(0.8, 1.2))


def hologram(B, x, y, z, hd, w, h, col_a, col_b, rng):
    R = rotz(hd)
    n = 8
    for i in range(n):
        c = col_a * (1 - i / n) + col_b * (i / n)
        loc = (np.array([(i + 0.5) * w / n - w / 2, 0, 0]) @ R.T) + [x, y, z]
        v, f = box((loc[0], loc[1], loc[2]), (w / n - 0.2, 0.25, h * rng.uniform(0.45, 1.0)), rot=hd + 90)
        B["emit"].add(v, f, c)
    for sx in (-1, 1):
        loc = (np.array([sx * w / 2, 0, 0]) @ R.T) + [x, y, z - 6]
        v, f = box((loc[0], loc[1], loc[2]), (0.8, 0.8, 6.5)); B["solid"].add(v, f, hx("#222836"))
    loc = np.array([x, y, z + h * 0.5])
    v, f = box((x, y, z + h), (w + 1, 0.3, 0.5), rot=hd + 90); B["emit"].add(v, f, col_b)


def tunnel(lay, th, B, Hfn, kinds=("tunnel", "glass_tunnel")):
    """Arch tube + portals over main-road stretches flagged as tunnels."""
    loop = [e for e in lay["edges"] if e["id"] in lay["main_loop"]]
    for zn in lay.get("zones", []):
        if zn["kind"] not in kinds:
            continue
        glass = zn["kind"] == "glass_tunnel"
        for e in loop:
            c = np.array(e["centerline"])
            s = e["s0"] + np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
            m = (s >= zn["s0"]) & (s <= zn["s1"]) if zn["s1"] >= zn["s0"] else (s >= zn["s0"]) | (s <= zn["s1"])
            idx = np.where(m)[0]
            if len(idx) < 3:
                continue
            nrm, tang = normals(c)
            w = np.array(e["widths"])
            for k in range(idx[0], idx[-1]):
                a, b = c[k], c[k + 1]
                if k % 2:
                    continue
                n_arc = 10
                ang = np.linspace(0, math.pi, n_arc)
                for j in range(n_arc - 1):
                    R_ = w[k] / 2 + 2.2
                    def P(aa, pt, nn):
                        return np.array([pt[0] + nn[0] * math.cos(aa) * R_, pt[1] + nn[1] * math.cos(aa) * R_, pt[2] + math.sin(aa) * (R_ * 0.78) + 0.0])
                    pa0, pa1 = P(ang[j], a, nrm[k]), P(ang[j + 1], a, nrm[k])
                    pb0, pb1 = P(ang[j], b, nrm[k + 1]), P(ang[j + 1], b, nrm[k + 1])
                    V = np.array([pa0, pa1, pb1, pb0])
                    col = hx(th.get("tunnel_glass", "#7fe8ff")) if glass else hx(th.get("tunnel_wall", "#3a4252"))
                    B["glass" if glass else "solid"].add(V, tri_quads([[0, 1, 2, 3]]), col * (0.95 + 0.05 * (j % 2)))
                if (k // 2) % 5 == 0:
                    for j in range(n_arc - 1):
                        R_ = w[k] / 2 + 2.3
                        p0 = np.array([a[0] + nrm[k][0] * math.cos(ang[j]) * R_, a[1] + nrm[k][1] * math.cos(ang[j]) * R_, a[2] + math.sin(ang[j]) * R_ * 0.78 + 0.1])
                        p1 = np.array([a[0] + nrm[k][0] * math.cos(ang[j + 1]) * R_, a[1] + nrm[k][1] * math.cos(ang[j + 1]) * R_, a[2] + math.sin(ang[j + 1]) * R_ * 0.78 + 0.1])
                        _tilt_box(B, p0, p1, th, hx(th["neon"][(k // 10) % len(th["neon"])]), wd=0.5, hd=0.4)
            for end in (idx[0], idx[-1]):
                R_ = w[end] / 2 + 3.0
                for sd in (-1, 1):
                    p = c[end][:2] + nrm[end] * sd * R_
                    v, f = box((p[0], p[1], c[end][2] - 0.5), (1.6, 1.6, R_ * 0.8 + 1.2)); B["solid"].add(v, f, hx("#cfd6e6"))
                p = c[end]
                v, f = box((p[0], p[1], p[2] + R_ * 0.78 + 0.6), (2.0, w[end] + 7.0, 1.2), rot=math.degrees(math.atan2(tang[end][1], tang[end][0])) + 90)
                B["emit"].add(v, f, hx(th["neon"][0]))


def gantry_arch(B, x, y, z, hd, w, col, h=9.0):
    R = rotz(hd - 90)
    for sd in (-1, 1):
        p = np.array([sd * (w / 2 + 1.0), 0, 0]) @ R.T + [x, y, z]
        v, f = box((p[0], p[1], p[2]), (1.2, 1.2, h)); B["solid"].add(v, f, hx("#262c3a"))
    p = np.array([0, 0, 0]) @ R.T + [x, y, z]
    v, f = box((p[0], p[1], p[2] + h), (1.4, w + 3.4, 1.1), rot=hd - 90 + 90 if False else hd); B["emit"].add(v, f, col)


def far_skyline(B, lay, Hfn, th, rng, W, ring=2):
    pass


def pit_canopy(B, lay, th, style="flat"):
    p = lay["pit"]
    sd = p["side"]
    lane_y = p["lane_y"]
    n = len(p["boxes"])
    x0, x1 = p["boxes"][0]["center"][0] - 7, p["boxes"][-1]["center"][0] + 7
    cy = lane_y + sd * 3.0
    if style == "scallop":
        for i in range(n):
            cx = p["boxes"][i]["center"][0]
            for j in range(9):
                a0, a1 = math.pi * j / 9, math.pi * (j + 1) / 9
                for dy in (-1, 1):
                    pass
                V = np.array([[cx - 6.2 + 12.4 * (1 - math.cos(a0)) / 2, cy - 7.0, 6.0 + 2.6 * math.sin(a0)], [cx - 6.2 + 12.4 * (1 - math.cos(a1)) / 2, cy - 7.0, 6.0 + 2.6 * math.sin(a1)],
                              [cx - 6.2 + 12.4 * (1 - math.cos(a1)) / 2, cy + 7.0, 6.0 + 2.6 * math.sin(a1)], [cx - 6.2 + 12.4 * (1 - math.cos(a0)) / 2, cy + 7.0, 6.0 + 2.6 * math.sin(a0)]])
                B["solid"].add(V, tri_quads([[0, 1, 2, 3]]), hx(th["team_colors"][i % len(th["team_colors"])]) * 0.9 + 0.1)
            v, f = box((cx, cy - 7.0, 5.9), (12.6, 0.4, 0.4)); B["emit"].add(v, f, hx(th["neon"][i % len(th["neon"])]))
    else:
        v, f = box(((x0 + x1) / 2, cy, 6.3), (x1 - x0, 15.0, 0.5), taper=1.0); B["solid"].add(v, f, hx(th.get("canopy", "#1d2330")))
        for k in range(5):
            v, f = box(((x0 + x1) / 2, cy - 7.4 + k * 3.7, 6.25), (x1 - x0 - 1.0, 0.35, 0.12)); B["emit"].add(v, f, hx(th["neon"][k % len(th["neon"])]))
        for i in range(0, n + 1, 3):
            cx = x0 + 7 + i * (x1 - x0 - 14) / n
            v, f = box((cx, cy + 6.8, 0.0), (0.7, 0.7, 6.3)); B["solid"].add(v, f, hx("#2a3042"))
