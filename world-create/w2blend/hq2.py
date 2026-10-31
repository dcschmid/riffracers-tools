"""Extra detail props for the Coastal HQ set (all generated, flat-shaded). Every function: fn(B, p, rng) with p a dict from props.json.
Motifs belong to the genre (beach festival, harbour, jungle, cliffs): lanterns, bunting, tiki bars, surf racks, balloons, turbines, flamingos ..."""
from __future__ import annotations

import math

import numpy as np

from .core import box, frustum, hx, kart, people, rotx, roty, rotz, sphere, strip, tri_quads, FUN
from .core import _person_proto  # noqa: F401  (ensures the improved figure kit is loaded)
from . import hq
from .hq import beam, face_shade, text3d

CORAL, TURQ, SUN, PINK, WH, NAVY, WOOD, STRAW = (hx(c) for c in ("#ff7a59", "#27c4c9", "#ffd23f", "#f25c9b", "#fbfbfd", "#0b1f3a", "#9a6a43", "#e3b85a"))
PAL = [CORAL, SUN, TURQ, PINK, WH]


def L2W(x, y, z, hd):
    R = rotz(hd)

    def f(lx, ly, lz=0.0):
        q = np.array([lx, ly, 0.0]) @ R.T
        return (x + q[0], y + q[1], z + lz)
    return f


def _rot(v, hd, x, y, z):
    return v @ rotz(hd).T + np.array([x, y, z])


# ------------------------------------------------------------------------------------------------ beach club
def p_lounge(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    n = p.get("n", 4)
    for i in range(n):
        lx = (i - (n - 1) / 2) * 3.4
        c = PAL[int(rng.integers(len(PAL)))]
        v, w = box(f(lx, 0, 0.25), (1.0, 2.1, 0.3), rot=hd); B["solid"].add(v, w, WOOD)
        v, w = box(f(lx, 0, 0.55), (0.95, 2.0, 0.18), rot=hd); B["solid"].add(v, w, c)
        v, w = box(f(lx, 0.9, 0.7), (0.95, 0.35, 0.55), rot=hd); B["solid"].add(v, w, c * 0.9)
    for i in range(n // 2):
        px = (i - (n // 2 - 1) / 2) * 6.8 + 1.7
        px = (i * 2 - (n - 1) / 2 + 0.5) * 3.4
        c = PAL[int(rng.integers(len(PAL)))]
        q = f(px, -1.9, 0.0)
        v, w = frustum(q, 0.07, 0.07, 3.1, 5); B["solid"].add(v, w, hx("#f3efe6"))
        for k in range(8):
            a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
            T = np.array([[q[0] + 2.5 * math.cos(a0), q[1] + 2.5 * math.sin(a0), q[2] + 2.7], [q[0] + 2.5 * math.cos(a1), q[1] + 2.5 * math.sin(a1), q[2] + 2.7], [q[0], q[1], q[2] + 3.6]])
            B["solid"].add(T, [[0, 1, 2]], c if k % 2 == 0 else WH)


def p_towels(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    for _ in range(p.get("n", 5)):
        c = PAL[int(rng.integers(len(PAL)))]
        px, py = x + rng.uniform(-5, 5), y + rng.uniform(-5, 5)
        v, w = box((px, py, z + 0.02), (1.0, 2.0, 0.06), rot=float(rng.uniform(0, 180))); B["solid"].add(v, w, c)
    if rng.uniform() < 0.7:
        v, w = sphere((x + rng.uniform(-3, 3), y + rng.uniform(-3, 3), z + 0.4), 0.38, 6, 4); B["solid"].add(v, w, PAL[int(rng.integers(len(PAL)))])


def p_surf(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    for sx in (-1, 1):
        v, w = box(f(sx * 2.0, 0, 0), (0.18, 0.9, 2.4), rot=hd); B["solid"].add(v, w, WOOD)
    v, w = box(f(0, 0, 0.9), (4.2, 0.2, 0.14), rot=hd); B["solid"].add(v, w, WOOD)
    v, w = box(f(0, 0, 2.3), (4.2, 0.2, 0.14), rot=hd); B["solid"].add(v, w, WOOD)
    for i in range(6):
        V, F = sphere((0, 0, 0), 1.0, 7, 5)
        V = V * np.array([0.34, 0.1, 1.6])
        V = V @ rotx(-9).T
        V = _rot(V, hd, *f((i - 2.5) * 0.66, 0.1, 1.9))
        B["solid"].add(V, F, face_shade(V, F, PAL[i % len(PAL)] * rng.uniform(0.95, 1.05), None, 0.18))


def p_lifeguard(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    for sx in (-1, 1):
        for sy in (-1, 1):
            v, w = frustum(f(sx * 1.0, sy * 1.0, 0), 0.14, 0.12, 2.7, 5); B["solid"].add(v, w, WOOD)
    v, w = box(f(0, 0, 2.6), (2.6, 2.6, 0.2), rot=hd); B["solid"].add(v, w, hx("#f3f1ec"))
    for k in range(3):
        v, w = box(f(0, -1.6 - k * 0.5, 0.0 + k * 0.85), (1.3, 0.5, 0.2), rot=hd); B["solid"].add(v, w, WOOD)
    v, w = box(f(0, 0, 2.8), (2.1, 2.1, 1.9), rot=hd); B["solid"].add(v, w, CORAL)
    v, w = box(f(0, -1.07, 3.5), (1.6, 0.06, 0.8), rot=hd); B["emit"].add(v, w, hx("#aef5ff") * 0.8)
    for k in range(3):
        v, w = box(f(0, 0, 3.0 + k * 0.6), (2.14, 2.14, 0.22), rot=hd); B["solid"].add(v, w, WH)
    v, w = frustum(f(0, 0, 4.7), 1.9, 0.3, 0.9, 4, rot=hd + 45); B["solid"].add(v, w, TURQ)
    v, w = frustum(f(1.2, 1.2, 4.7), 0.05, 0.05, 2.0, 4); B["solid"].add(v, w, hx("#e8edf1"))
    q = f(1.2, 1.2, 6.2)
    B["solid"].add(np.array([q, (q[0] + 0.8 * math.cos(math.radians(hd)), q[1] + 0.8 * math.sin(math.radians(hd)), q[2] - 0.3), (q[0], q[1], q[2] - 0.8)]), [[0, 1, 2]], CORAL)


def p_volley(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    for sx in (-1, 1):
        v, w = frustum(f(sx * 4.6, 0, 0), 0.1, 0.09, 2.6, 5); B["solid"].add(v, w, hx("#e8edf1"))
    v, w = box(f(0, 0, 1.6), (9.2, 0.05, 0.95), rot=hd); B["glass"].add(v, w, hx("#f6f7f9"))
    v, w = box(f(0, 0, 2.55), (9.2, 0.07, 0.1), rot=hd); B["solid"].add(v, w, WH)
    for sy in (-1, 1):
        v, w = box(f(0, sy * 4.5, 0.02), (9.0, 0.1, 0.05), rot=hd); B["solid"].add(v, w, WH)
    for sx in (-1, 1):
        v, w = box(f(sx * 4.5, 0, 0.02), (0.1, 9.0, 0.05), rot=hd); B["solid"].add(v, w, WH)


def p_tiki(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    R = p.get("R", 4.6)
    for k in range(6):
        a = 2 * math.pi * k / 6
        v, w = frustum((x + R * math.cos(a), y + R * math.sin(a), z), 0.2, 0.17, 3.0, 5); B["solid"].add(v, w, WOOD)
    V, F = frustum((x, y, z + 3.0), R + 1.6, 0.25, 3.4, 10)
    B["solid"].add(V, F, face_shade(V, F, STRAW * rng.uniform(0.92, 1.05), None, 0.28))
    V, F = frustum((x, y, z + 2.9), R + 1.75, R + 1.6, 0.35, 10); B["solid"].add(V, F, STRAW * 0.8)
    v, w = frustum((x, y, z), R * 0.45, R * 0.45, 1.1, 10); B["solid"].add(v, w, WOOD)
    v, w = frustum((x, y, z + 1.1), R * 0.5, R * 0.5, 0.12, 10); B["solid"].add(v, w, hx("#f3efe6"))
    for k in range(5):
        a = 2 * math.pi * (k + 0.5) / 5
        v, w = frustum((x + R * 0.8 * math.cos(a), y + R * 0.8 * math.sin(a), z), 0.3, 0.26, 0.85, 6); B["solid"].add(v, w, PAL[k % 4])
    for k in range(10):
        a0 = 2 * math.pi * k / 10
        v, w = sphere((x + (R + 1.2) * math.cos(a0), y + (R + 1.2) * math.sin(a0), z + 3.0), 0.22, 5, 3); B["emit"].add(v, w, SUN)
    people(B["people"], [(x + 1.2 * math.cos(a), y + 1.2 * math.sin(a), z + 1.1, math.degrees(a) + 180) for a in (0.5, 2.2, 4.0)], rng, palette=["#ff7a59", "#27c4c9", "#ffd23f", "#ffffff"], scale=(0.9, 1.0))


def p_stall(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    c1, c2 = (CORAL, TURQ) if rng.uniform() < 0.5 else (SUN, PINK)
    v, w = box(f(0, 1.3, 0), (4.0, 0.25, 3.0), rot=hd); B["solid"].add(v, w, hx("#f3efe6"))
    for sx in (-1, 1):
        v, w = box(f(sx * 1.9, 0.2, 0), (0.25, 2.4, 3.0), rot=hd); B["solid"].add(v, w, hx("#f3efe6"))
    v, w = box(f(0, -0.6, 0), (4.0, 0.9, 1.1), rot=hd); B["solid"].add(v, w, c1)
    v, w = box(f(0, -0.6, 1.1), (4.2, 1.1, 0.1), rot=hd); B["solid"].add(v, w, WOOD)
    for i in range(8):
        lx = (i - 3.5) * 0.55
        q = np.array([f(lx - 0.27, 1.3, 3.0), f(lx + 0.27, 1.3, 3.0), f(lx + 0.27, -1.7, 2.45), f(lx - 0.27, -1.7, 2.45)])
        B["solid"].add(q, tri_quads([[0, 1, 2, 3]]), c1 if i % 2 == 0 else WH)
    v, w = box(f(0, -1.7, 2.3), (4.4, 0.12, 0.35), rot=hd); B["solid"].add(v, w, c2)
    v, w = box(f(0, -1.0, 1.2), (3.0, 0.08, 0.5), rot=hd); B["emit"].add(v, w, SUN)
    for k in range(3):
        v, w = sphere(f((k - 1) * 0.9, -0.5, 1.45), 0.22, 5, 3); B["solid"].add(v, w, PAL[int(rng.integers(len(PAL)))])
    sp = [(*f(0.4, 0.6, 0.0)[:2], z, hd + 180)]
    people(B["people"], [(sp[0][0], sp[0][1], z, hd - 90)], rng, palette=["#ffffff", "#ff7a59"], scale=(0.95, 1.0))
    people(B["people"], [(f(-1.0, -2.6)[0], f(-1.0, -2.6)[1], z, hd + 90), (f(1.2, -3.2)[0], f(1.2, -3.2)[1], z, hd + 90)], rng, scale=(0.9, 1.05))


# ------------------------------------------------------------------------------------------------ nature
def p_balloon(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.0)
    r = 7.0 * s
    V, F = sphere((0, 0, 0), r, 14, 8, 1.12)
    low = V[:, 2] < 0
    V[low, :2] *= (1 - 0.55 * np.clip(-V[low, 2] / (r * 1.12), 0, 1) ** 1.6)[:, None]
    cen = V[F].mean(1)
    az = np.arctan2(cen[:, 1], cen[:, 0])
    idx = (((az + math.pi) / (2 * math.pi)) * 14).astype(int) % 14
    pal = [CORAL, SUN, TURQ, PINK, WH, SUN, TURQ]
    cols = np.array([pal[(i + int(p.get("seed", 0))) % len(pal)] for i in idx]) * rng.uniform(0.96, 1.04)
    V = V + np.array([x, y, z + r * 1.1 + 6.5 * s])
    B["solid"].add(V, F, cols)
    v, w = box((x, y, z + 3.2 * s), (1.5 * s, 1.5 * s, 1.1 * s)); B["solid"].add(v, w, WOOD)
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        beam(B, (x + dx * 0.7 * s, y + dy * 0.7 * s, z + 4.3 * s), (x + dx * 2.2 * s, y + dy * 2.2 * s, z + r * 0.35 + 6.5 * s), hx("#e8edf1"), 0.1)
    v, w = sphere((x, y, z + 4.6 * s), 0.35 * s, 5, 3); B["emit"].add(v, w, SUN)


def p_seagull(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    s = p.get("s", 1.0) * 1.8
    hd = math.radians(p.get("hd", 0.0))
    R = rotz(p.get("hd", 0.0) - 90)
    w = 2.4 * s
    pts = np.array([[0, 1.0 * s, 0], [0, -0.6 * s, 0], [-w, -1.0 * s, 0.55 * s], [w, -1.0 * s, 0.55 * s]])
    P = pts @ R.T + np.array([x, y, z])
    B["solid"].add(P, [[0, 1, 2], [0, 3, 1]], hx("#ffffff"))
    B["solid"].add(P, [[0, 2, 1], [0, 1, 3]], hx("#e3edf6"))


def p_flamingo(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.0)
    hd = p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    pink = hx("#ff7fb0")
    for dx in (-0.12, 0.12):
        beam(B, f(dx, 0, 0.0), f(dx, 0, 1.7 * s), hx("#ff9ec4"), 0.07)
    V, F = sphere((0, 0, 0), 1.0, 7, 5); V = V * np.array([0.5, 0.85, 0.5]) * s; V = _rot(V, hd, *f(0, 0, 1.95 * s)); B["solid"].add(V, F, face_shade(V, F, pink, None, 0.15))
    pts = [f(0, 0.5, 2.0 * s), f(0, 0.85, 2.5 * s), f(0, 0.8, 3.1 * s), f(0, 1.05, 3.45 * s)]
    for a, b in zip(pts[:-1], pts[1:]):
        beam(B, a, b, pink, 0.14)
    v, w = sphere(pts[-1], 0.2 * s, 5, 3); B["solid"].add(v, w, pink)
    beam(B, pts[-1], f(0, 1.5, 3.3 * s), hx("#2a2f3a"), 0.08)


def p_fern(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.0)
    base = hx("#2f9a4a") * rng.uniform(0.85, 1.15)
    for k in range(7):
        a = 2 * math.pi * k / 7 + rng.uniform(-0.25, 0.25)
        d = np.array([math.cos(a), math.sin(a)])
        L = rng.uniform(1.6, 2.4) * s
        tip = np.array([x + d[0] * L, y + d[1] * L, z + rng.uniform(0.3, 0.9) * s])
        mid = np.array([x + d[0] * L * 0.5, y + d[1] * L * 0.5, z + 1.1 * s])
        nrm = np.array([-d[1], d[0], 0]) * 0.38 * s
        P = np.array([[x, y, z + 0.1], mid + nrm, tip, mid - nrm])
        B["solid"].add(P, tri_quads([[0, 1, 2, 3]]), base * (0.9 + 0.1 * (k % 2)))


def p_tuft(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    for k in range(4):
        v, w = frustum((x + rng.uniform(-0.4, 0.4), y + rng.uniform(-0.4, 0.4), z), 0.2, 0.02, rng.uniform(0.7, 1.3), 4, rot=float(rng.uniform(0, 90)))
        B["solid"].add(v, w, hx(["#3fba4a", "#5ccb52", "#2f9a44"][k % 3]))


def p_flowers(B, p, rng):
    x, y, z, R = p["x"], p["y"], p["z"], p.get("R", 3.0)
    for _ in range(p.get("n", 9)):
        fx, fy = x + rng.uniform(-R, R), y + rng.uniform(-R, R)
        beam(B, (fx, fy, z), (fx, fy, z + 0.55), hx("#2f9a44"), 0.06)
        v, w = sphere((fx, fy, z + 0.62), 0.26, 5, 3); B["solid"].add(v, w, PAL[int(rng.integers(len(PAL)))])


def p_reeds(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    for _ in range(7):
        v, w = frustum((x + rng.uniform(-1.2, 1.2), y + rng.uniform(-1.2, 1.2), z - 0.1), 0.07, 0.015, rng.uniform(1.6, 2.8), 4, rot=float(rng.uniform(0, 90)))
        B["solid"].add(v, w, hx("#b9b469") * rng.uniform(0.9, 1.1))
        if rng.uniform() < 0.4:
            q = (x + rng.uniform(-1.2, 1.2), y + rng.uniform(-1.2, 1.2), z + 2.4)
            v, w = frustum(q, 0.14, 0.1, 0.55, 5); B["solid"].add(v, w, hx("#7a5a3c"))


def p_turbine(B, p, rng):
    x, y, z, hd, h = p["x"], p["y"], p["z"], p.get("hd", 0.0), p.get("h", 46.0)
    v, w = frustum((x, y, z), 1.7, 0.9, h, 10); B["solid"].add(v, w, hx("#f4f7fa"))
    v, w = frustum((x, y, z + h * 0.30), 1.46, 1.40, 2.4, 10); B["solid"].add(v, w, CORAL)
    d = np.array([math.cos(math.radians(hd)), math.sin(math.radians(hd)), 0.0])
    v, w = box((x, y, z + h), (4.2, 1.7, 1.7), rot=hd); B["solid"].add(v, w, hx("#eef3f7"))
    hub = np.array([x, y, z + h + 0.85]) + d * 2.3
    v, w = sphere(tuple(hub), 0.95, 6, 4); B["solid"].add(v, w, CORAL)
    a0 = float(rng.uniform(0, 120))
    for k in range(3):
        V, F = box((0, 0, 0.5), (0.34, 1.35, 21.0), taper=0.18)
        V = V @ rotx(a0 + 120 * k).T @ rotz(hd).T + hub
        B["solid"].add(V, F, hx("#ffffff"))


def p_cornersign(B, p, rng):
    x, y, z, face, d = p["x"], p["y"], p["z"], p["face"], p.get("dir", 1)
    R = rotz(face)
    for sy in (-1, 1):
        q = np.array([0.0, sy * 2.7, 0.0]) @ R.T
        v, w = frustum((x + q[0], y + q[1], z), 0.16, 0.14, 2.4, 5); B["solid"].add(v, w, hx("#e8edf1"))
    def put(V, F, col, group="solid"):
        Vw = V @ R.T + np.array([x, y, z + 2.0])
        B[group].add(Vw, F, col)
    V, F = box((0.0, 0.0, 0.0), (0.22, 6.4, 3.5)); put(V, F, WH)
    V, F = box((0.07, 0.0, 0.12), (0.2, 6.0, 3.2)); put(V, F, CORAL)
    for k in range(3):
        yk = (k - 1) * 1.9
        for sgn in (1, -1):
            poly = np.array([[yk - d * 0.55, sgn * 1.15], [yk + d * 0.05, sgn * 1.15], [yk + d * 0.65, 0.0], [yk + d * 0.05, 0.0]])
            if d < 0:
                poly = poly[::-1]
            Vp, Fp = _prism(poly, 0.0, 0.1)
            Vl = Vp[:, [2, 0, 1]] + np.array([0.18, 0.0, 1.75])
            put(Vl, Fp, WH)
    # corner marker lamps on top
    for sy in (-1, 1):
        V, F = sphere((0.0, sy * 2.9, 3.75), 0.2, 5, 3); put(V, F, SUN, "emit")


def _prism(poly, z0, z1):
    poly = np.asarray(poly, float)
    k = len(poly)
    V = np.vstack([np.column_stack([poly, np.full(k, z0)]), np.column_stack([poly, np.full(k, z1)])])
    F = []
    for i in range(1, k - 1):
        F += [[k, k + i, k + i + 1], [0, i + 1, i]]
    for i in range(k):
        j = (i + 1) % k
        F += [[i, j, k + j], [i, k + j, k + i]]
    return V, np.array(F)


def p_brake(B, p, rng):
    x, y, z, face = p["x"], p["y"], p["z"], p["face"]
    R = rotz(face)
    v, w = frustum((x, y, z), 0.12, 0.12, 1.6, 5); B["solid"].add(v, w, hx("#e8edf1"))
    V, F = box((0, 0, 0), (0.2, 2.0, 1.5)); B["solid"].add(V @ R.T + np.array([x, y, z + 1.4]), F, WH)
    V, F = box((0.06, 0, 0.1), (0.14, 1.8, 1.3)); B["solid"].add(V @ R.T + np.array([x, y, z + 1.4]), F, CORAL)
    fx, fy = math.cos(math.radians(face)), math.sin(math.radians(face))
    rz = math.degrees(math.atan2(fx, -fy))
    text3d(B, p["text"], (x + fx * 0.2, y + fy * 0.2, z + 2.15), 0.9, rot_z=rz, col=hx("#ffffff"), group="solid", extrude=0.06)


def p_tyres(B, p, rng):
    a, b = np.array(p["a"], float), np.array(p["b"], float)
    L = float(np.linalg.norm(b - a))
    n = max(int(L // 1.0), 2)
    for i in range(n):
        q = a + (b - a) * i / (n - 1)
        for k in range(3):
            c = [hx("#23262e"), WH, CORAL][(i + k) % 3] if k < 2 else hx("#23262e")
            v, w = frustum((q[0], q[1], q[2] + k * 0.34), 0.5, 0.5, 0.32, 9); B["solid"].add(v, w, c)


def p_crowd(B, p, rng):
    x, y, z, face = p["x"], p["y"], p["z"], p["face"]
    n = p.get("n", 10)
    f = L2W(x, y, z, face - 90)
    spots = []
    for i in range(n):
        lx, ly = rng.uniform(-3.4, 3.4), rng.uniform(-1.2, 1.2) + 0.6 * (i % 2)
        q = f(lx, ly)
        spots.append((q[0], q[1], z, face + rng.uniform(-20, 20)))
    people(B["people"], spots, rng, palette=FUN)
    for sx in (-2.2, 2.6):
        q = f(sx, 0.4)
        v, w = frustum((q[0], q[1], z), 0.06, 0.05, 4.6, 4); B["solid"].add(v, w, hx("#e8edf1"))
        t = np.array([q[0], q[1], z + 4.2])
        fv = np.array([math.cos(math.radians(face - 90)), math.sin(math.radians(face - 90)), 0.0])
        B["solid"].add(np.array([t, t + fv * 1.6 + np.array([0, 0, -0.15]), t + np.array([0, 0, -1.0])]), [[0, 1, 2]], PAL[int(rng.integers(len(PAL)))])


def p_kart(B, p, rng):
    cols = ["#ff7a59", "#27c4c9", "#ffd23f", "#f25c9b", "#7c5cd6", "#34b36b", "#2d6fdc", "#ff9f43", "#e0453a", "#b8e86a"]
    i = int(p.get("c", 0))
    kart(B["people"], p["x"], p["y"], p["z"], p["hd"], cols[i % len(cols)], "#ffffff", cols[(i + 4) % len(cols)], scale=p.get("s", 1.5))


def p_ribbon(B, p, rng):
    pts = np.array(p["pts"], float)
    n = len(pts)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1))])
    zc = pts[:, 2] + 3.0 + 2.6 * np.sin(cum / 14.0)
    for off, hh in ((0.0, 1.5), (2.0, 1.0)):
        zz = zc + off * np.sin(cum / 9.0 + off)
        Lz = np.column_stack([pts[:, :2], zz - hh])
        Rz = np.column_stack([pts[:, :2], zz + hh])
        V, F = strip(Lz, Rz)
        cols = np.array([[CORAL, SUN, TURQ, PINK][(i // 3 + int(off)) % 4] for i in range(n - 1)])
        B["emit"].add(V, F, np.vstack([cols, cols]))
    for i in range(0, n, 6):
        v, w = frustum((pts[i, 0], pts[i, 1], pts[i, 2]), 0.28, 0.2, zc[i] - pts[i, 2] - 1.2, 6); B["solid"].add(v, w, hx("#f3f6f8"))


def p_drone(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    v, w = box((x, y, z), (1.0, 1.0, 0.3)); B["solid"].add(v, w, hx("#2a2f3a"))
    for dx, dy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        beam(B, (x, y, z + 0.15), (x + dx * 0.9, y + dy * 0.9, z + 0.15), hx("#2a2f3a"), 0.12)
        v, w = frustum((x + dx * 0.9, y + dy * 0.9, z + 0.2), 0.55, 0.55, 0.05, 10); B["glass"].add(v, w, hx("#d6f8ff"))
    v, w = sphere((x, y, z - 0.2), 0.3, 5, 3); B["emit"].add(v, w, [TURQ, CORAL, SUN][int(rng.integers(3))])


def p_container(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    for k in range(p.get("n", 3)):
        c = PAL[int(rng.integers(4))]
        for lvl in range(int(rng.integers(1, 3))):
            v, w = box(f(k * 2.6, 0, lvl * 2.6), (2.4, 6.0, 2.5), rot=hd); B["solid"].add(v, w, c * 0.95)
            for r in range(5):
                v, w = box(f(k * 2.6, -3.02 + 0 * r, lvl * 2.6 + 0.25 + r * 0.45), (2.3, 0.06, 0.12), rot=hd); B["solid"].add(v, w, c * 0.75)


def p_crane(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    H = 22.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            beam(B, f(sx * 4, sy * 3, 0), f(sx * 2.4, sy * 2, H), TURQ * 0.9, 0.5)
    for k in range(4):
        zk = 3 + k * 5.5
        beam(B, f(-4 + 0.4 * k, -3, zk), f(4 - 0.4 * k, 3, zk), hx("#f3f6f8"), 0.2)
    v, w = box(f(0, 0, H), (7.0, 5.0, 1.2), rot=hd); B["solid"].add(v, w, CORAL)
    beam(B, f(-12, 0, H + 0.6), f(22, 0, H + 0.6), hx("#f3f6f8"), 0.7)
    beam(B, f(22, 0, H + 0.6), f(22, 0, H - 9.0), hx("#2a2f3a"), 0.12)
    v, w = box(f(22, 0, H - 11.0), (1.2, 1.2, 1.4), rot=hd); B["solid"].add(v, w, SUN)
    v, w = sphere(f(0, 0, H + 1.6), 0.45, 5, 3); B["emit"].add(v, w, SUN)


def p_dock(B, p, rng):
    x, y, z, hd, L = p["x"], p["y"], p["z"], p.get("hd", 0.0), p.get("L", 28.0)
    f = L2W(x, y, z, hd)
    n = int(L / 1.3)
    for i in range(n):
        v, w = box(f(i * 1.3 + 0.65, 0, 0.0), (1.15, 4.2, 0.22), rot=hd); B["solid"].add(v, w, WOOD * rng.uniform(0.9, 1.1))
    for i in range(0, n, 3):
        for sy in (-1, 1):
            v, w = frustum(f(i * 1.3 + 0.65, sy * 1.9, -2.4), 0.22, 0.2, 3.2, 6); B["solid"].add(v, w, WOOD * 0.8)
    for i in range(0, n, 5):
        v, w = frustum(f(i * 1.3 + 0.65, -2.0, 0.2), 0.12, 0.12, 2.6, 5); B["solid"].add(v, w, hx("#2a2f3a"))
        v, w = sphere(f(i * 1.3 + 0.65, -2.0, 2.9), 0.28, 5, 3); B["emit"].add(v, w, hx("#ffe9a8"))
    for i in range(p.get("crates", 3)):
        v, w = box(f(L - 3 - i * 1.6, 1.0, 0.22), (1.0, 1.0, 1.0), rot=hd + float(rng.uniform(-8, 8))); B["solid"].add(v, w, PAL[i % 4])


def p_foam(B, p, rng):
    Lp, Rp = np.array(p["L"], float), np.array(p["R"], float)
    z = p["z"]
    V, F = strip(np.column_stack([Lp, np.full(len(Lp), z)]), np.column_stack([Rp, np.full(len(Rp), z)]))
    B["solid"].add(V, F, hx("#f4fbff"))


PROPS = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("p_") and callable(v)}


# ------------------------------------------------------------------------------------------------ trap visuals
def _band_side(bid):
    return bid.split("_")[1][0]          # 'l', 'c' or 'r'


def jets2(B, tr, th, rng):
    """Bass Wave Jets: per station one band fires (pattern c, l, r, c, l), the next to fire shows an amber warning ring, the rest sit idle."""
    pattern = ["c", "l", "r", "c", "l", "r"]
    for k, st in enumerate(tr["stations"]):
        h = math.radians(st["heading_deg"])
        u = np.array([math.cos(h), math.sin(h)])
        n_ = np.array([-u[1], u[0]])
        active = pattern[k % len(pattern)]
        nxt = {"c": "l", "l": "r", "r": "c"}[active]
        for bd in st["bands"]:
            side = _band_side(bd["id"])
            wid = bd["d1"] - bd["d0"]
            mid = np.array(st["xy"]) + n_ * (bd["d0"] + bd["d1"]) / 2
            v, w = box((mid[0], mid[1], st["z"] + 0.02), (bd.get("len", 6.0), wid, 0.12), rot=math.degrees(h)); B["solid"].add(v, w, hx("#1d2a3a"))
            v, w = box((mid[0], mid[1], st["z"] + 0.14), (bd.get("len", 6.0) - 0.6, wid - 0.6, 0.06), rot=math.degrees(h)); B["emit"].add(v, w, hx("#ff9f43") * (0.9 if side == nxt else 0.35))
            ncol = 3 if side == "c" else 5
            if side == active:
                for j in range(ncol):
                    lat = (j - (ncol - 1) / 2) * (wid / ncol)
                    pp = mid + n_ * lat + u * rng.uniform(-1.2, 1.2)
                    hh = rng.uniform(8.0, 12.5)
                    for layer, (r0, r1, hf, col) in enumerate(((1.0, 0.34, 1.0, "#9fe9f7"), (0.62, 0.18, 0.86, "#d6f8ff"), (1.3, 0.5, 0.7, "#6fd6ec"))):
                        v, w = frustum((pp[0], pp[1], st["z"] + 0.1), r0, r1, hh * hf, 7, rot=layer * 17); B["glass"].add(v, w, hx(col))
                    v, w = sphere((pp[0], pp[1], st["z"] + 0.1 + hh), 0.75, 6, 4); B["solid"].add(v, w, hx("#ffffff"))
                    for q in range(5):
                        a = rng.uniform(0, 6.28); r = rng.uniform(1.0, 2.5)
                        v, w = sphere((pp[0] + math.cos(a) * r, pp[1] + math.sin(a) * r, st["z"] + 0.1 + hh * rng.uniform(0.5, 0.98)), rng.uniform(0.22, 0.45), 5, 3); B["glass"].add(v, w, hx("#e6fbff"))
                for q in range(10):
                    pp = mid + u * rng.uniform(-3, 3) + n_ * rng.uniform(-wid / 2, wid / 2)
                    v, w = sphere((pp[0], pp[1], st["z"] + 0.35), rng.uniform(0.35, 0.8), 5, 3, 0.6); B["solid"].add(v, w, hx("#ffffff"))
            elif side == nxt:
                for q in range(12):
                    a = 2 * math.pi * q / 12
                    pp = mid + u * math.cos(a) * 2.6 + n_ * math.sin(a) * (wid / 2 - 0.4)
                    v, w = box((pp[0], pp[1], st["z"] + 0.12), (0.5, 0.5, 0.22)); B["emit"].add(v, w, SUN)
                for q in range(3):
                    pp = mid + n_ * (q - 1) * wid / 3
                    for t in range(4):
                        v, w = sphere((pp[0] + rng.uniform(-0.6, 0.6), pp[1] + rng.uniform(-0.6, 0.6), st["z"] + 0.3 + t * 0.35), 0.28, 5, 3); B["glass"].add(v, w, hx("#e6fbff"))
        # LED arrow gantry on the idle edge showing the next lane
        for sd in (-1, 1):
            pp = np.array(st["xy"]) + n_ * sd * 9.4
            v, w = frustum((pp[0], pp[1], st["z"]), 0.3, 0.26, 3.4, 6); B["solid"].add(v, w, hx("#e8edf1"))
            col = {"l": 1, "r": -1, "c": 0}[nxt]
            v, w = box((pp[0], pp[1], st["z"] + 3.4), (0.9, 0.9, 0.9), rot=math.degrees(h)); B["emit"].add(v, w, SUN if (sd == col or nxt == "c") else hx("#27c4c9") * 0.5)


def barriers2(B, tr, th, rng):
    """Festival barriers: left/right panels alternate per row (L closed, R closed, ...); open panels sit retracted in their housings."""
    for k, st in enumerate(tr["stations"]):
        h = math.radians(st["heading_deg"])
        u = np.array([math.cos(h), math.sin(h)])
        n_ = np.array([-u[1], u[0]])
        for bd in st["bands"]:
            side = 1 if (bd["d0"] + bd["d1"]) > 0 else -1
            closed = (side == 1 and k % 2 == 0) or (side == -1 and k % 2 == 1)
            wid = bd["d1"] - bd["d0"]
            mid = np.array(st["xy"]) + n_ * (bd["d0"] + bd["d1"]) / 2
            v, w = box((mid[0], mid[1], st["z"] + 0.03), (1.0, wid, 0.08), rot=math.degrees(h)); B["solid"].add(v, w, hx("#334155"))
            edge = np.array(st["xy"]) + n_ * side * 8.2
            v, w = box((edge[0], edge[1], st["z"]), (5.6, 1.6, 2.6), rot=math.degrees(h)); B["solid"].add(v, w, hx("#334155"))
            v, w = box((edge[0], edge[1], st["z"] + 2.6), (5.8, 1.8, 0.2), rot=math.degrees(h)); B["emit"].add(v, w, SUN if closed else hx("#27c4c9") * 0.6)
            if closed:
                for j in range(8):
                    pp = mid + n_ * (j - 3.5) * (wid / 8)
                    v, w = box((pp[0], pp[1], st["z"] + 0.1), (0.7, wid / 8 - 0.06, 2.8), rot=math.degrees(h)); B["solid"].add(v, w, CORAL if j % 2 == 0 else TURQ)
                v, w = box((mid[0], mid[1], st["z"] + 2.9), (0.9, wid, 0.18), rot=math.degrees(h)); B["emit"].add(v, w, SUN)
                for j in range(4):
                    pp = mid + n_ * (j - 1.5) * (wid / 4)
                    v, w = box((pp[0] - u[0] * 0.4, pp[1] - u[1] * 0.4, st["z"] + 1.0), (0.08, 0.9, 0.9), rot=math.degrees(h)); B["emit"].add(v, w, SUN)
            else:
                for j in range(3):
                    pp = np.array(st["xy"]) + n_ * side * (7.4 - 0.5 * j)
                    v, w = box((pp[0], pp[1], st["z"] + 0.1), (0.7, 0.45, 2.8), rot=math.degrees(h)); B["solid"].add(v, w, CORAL if j % 2 == 0 else TURQ)
        top = np.array(st["xy"])
        for sd in (-1, 1):
            pp = top + n_ * sd * 8.9
            v, w = frustum((pp[0], pp[1], st["z"]), 0.32, 0.3, 6.5, 6); B["solid"].add(v, w, hx("#e8edf1"))
        v, w = box((top[0], top[1], st["z"] + 6.4), (0.9, 18.2, 0.5), rot=math.degrees(h) + 90 - 90); B["solid"].add(v, w, hx("#e8edf1"))
        for j in range(8):
            pp = top + n_ * (j - 3.5) * 2.2
            Vt = np.array([[pp[0], pp[1], st["z"] + 6.4], [pp[0] + u[0] * 1.0, pp[1] + u[1] * 1.0, st["z"] + 6.4], [pp[0] + u[0] * 0.5, pp[1] + u[1] * 0.5, st["z"] + 4.9]])
            B["solid"].add(Vt, [[0, 1, 2]], [CORAL, TURQ, SUN, PINK][j % 4])
