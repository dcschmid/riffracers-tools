"""Coastal v3 kit, part 1: high-fidelity low-poly nature, sea life, boats, village houses, crowds and sky.
Every prop function: fn(B, p, rng) with p a dict from props.json (same convention as hq2). All geometry is faceted (flat shaded), per-face colours with light-direction shading,
so it stays 'low poly' but with many more planes, gradients, rings and small details than v2. No karts anywhere."""
from __future__ import annotations

import math

import numpy as np

from .core import box, frustum, hx, people, prism_poly, rotx, roty, rotz, sphere, strip, tri_quads
from .hq import beam, text3d

CORAL, TURQ, SUN, PINK, WH, NAVY = (hx(c) for c in ("#ff6f59", "#22c3c8", "#ffd23f", "#ff5f9e", "#ffffff", "#12335a"))
LIGHT = np.array([-0.5, -0.62, 0.6])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def L2W(x, y, z, hd):
    R = rotz(hd)

    def f(lx, ly, lz=0.0):
        q = np.array([lx, ly, 0.0]) @ R.T
        return (x + q[0], y + q[1], z + lz)
    return f


def _mix(a, b, t):
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def shade(V, F, base, k=0.34, amb=0.70):
    """Per-face colour: base * (amb + k * max(n . L, 0)), with a slightly cooler tint in the shade."""
    V = np.asarray(V, float)
    P = V[np.asarray(F)]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    d = np.abs(n @ LIGHT) if False else np.maximum(n @ LIGHT, -0.4 * n @ LIGHT)
    base = np.asarray(base, float)
    base = base if base.ndim == 2 else np.tile(base, (len(F), 1))
    lit = (amb + k * np.clip(d, 0, 1) + 0.12 * np.abs(n[:, 2]))[:, None]
    cool = (1 - np.clip(d, 0, 1))[:, None] * np.array([-0.02, 0.0, 0.05])
    return np.clip(base * lit + cool * base, 0, 1.5)


def tube(pts, radii, n=7, rot=0.0):
    """Faceted tube along a 3D polyline. Returns V, F, seg (segment index per face, -1 for caps)."""
    pts = np.asarray(pts, float)
    k = len(pts)
    T = np.gradient(pts, axis=0)
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-9)
    rings = []
    for i in range(k):
        t = T[i]
        a = np.cross(t, [0.0, 0.0, 1.0])
        if np.linalg.norm(a) < 1e-6:
            a = np.array([1.0, 0.0, 0.0])
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
        rings.append(pts[i] + radii[i] * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b))
    V = np.vstack(rings)
    F, seg = [], []
    for i in range(k - 1):
        for j in range(n):
            j2 = (j + 1) % n
            a0, a1, b0, b1 = i * n + j, i * n + j2, (i + 1) * n + j, (i + 1) * n + j2
            F += [[a0, a1, b1], [a0, b1, b0]]
            seg += [i, i]
    c0 = len(V)
    V = np.vstack([V, pts[0], pts[-1]])
    for j in range(n):
        j2 = (j + 1) % n
        F.append([c0, j2, j]); seg.append(-1)
        F.append([c0 + 1, (k - 1) * n + j, (k - 1) * n + j2]); seg.append(-1)
    return V, np.array(F), np.array(seg)


def _frond(B, crown, d, L, up, droop, wmax, cols, m=9, jag=True, drop_edge=0.28):
    ts = np.linspace(0, 1, m)
    cen = np.array([crown + np.array([d[0] * L * t, d[1] * L * t, up * math.sin(math.pi * 0.75 * t) - droop * L * t * t]) for t in ts])
    perp = np.array([-d[1], d[0], 0.0])
    wd = wmax * np.sin(np.pi * np.clip(ts, 0, 1) ** 0.72) + 0.04
    zig = np.where(np.arange(m) % 2 == 0, 1.0, 0.52) if jag else np.ones(m)
    Lp, Rp = cen + perp * (wd * zig)[:, None], cen - perp * (wd * zig)[:, None]
    Lp[:, 2] -= drop_edge * wd
    Rp[:, 2] -= drop_edge * wd
    V, F = strip(Lp, Rp)
    rc = np.array([_mix(cols[0], _mix(cols[1], cols[2], max(t - 0.5, 0) * 2), min(t * 2, 1)) for t in ts[:-1]])
    B["solid"].add(V, F, np.vstack([rc, rc]))


# ------------------------------------------------------------------------------------------------ vegetation
def p_palm3(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 2.0)
    H = rng.uniform(6.9, 8.8) * s
    lean = rng.uniform(0, 2 * math.pi)
    ld = np.array([math.cos(lean), math.sin(lean)])
    bend = rng.uniform(0.12, 0.42) * H * 0.4
    ts = np.linspace(0, 1, 10)
    pts = np.array([[x + ld[0] * bend * t * t + 0.04 * s * math.sin(7 * t), y + ld[1] * bend * t * t, z + H * t] for t in ts])
    radii = 0.62 * s * (1 - 0.52 * ts) * (1 + 0.1 * (np.arange(10) % 2))
    V, F, seg = tube(pts, radii, 7, rot=rng.uniform(0, 1))
    c1, c2 = hx("#a98260"), hx("#8a6745")
    cols = np.array([(c1 if (sg % 2 == 0) else c2) * (0.82 + 0.22 * (sg / 9)) if sg >= 0 else c2 for sg in seg])
    B["solid"].add(V, F, shade(V, F, cols, 0.2, 0.84))
    crown = pts[-1] + np.array([0, 0, 0.2 * s])
    v, f = sphere(tuple(crown + np.array([0, 0, 0.05 * s])), 0.62 * s, 7, 4, 0.7)
    B["solid"].add(v, f, shade(v, f, hx("#7b8f3b"), 0.2, 0.8))
    nf = int(rng.integers(12, 15))
    g0, g1, g2 = hx("#2a8043"), hx("#4fb458"), hx("#a6e36e")
    for i in range(nf):
        a = 2 * math.pi * i / nf + rng.uniform(-0.18, 0.18)
        d = np.array([math.cos(a), math.sin(a)])
        _frond(B, crown, d, rng.uniform(4.4, 5.8) * s, rng.uniform(0.9, 2.1) * s, rng.uniform(0.5, 0.85), 1.22 * s,
               (g0 * rng.uniform(0.9, 1.08), g1 * rng.uniform(0.92, 1.08), g2 * rng.uniform(0.9, 1.05)), m=9)
    for i in range(2):                                   # dry hanging fronds
        a = rng.uniform(0, 2 * math.pi)
        d = np.array([math.cos(a), math.sin(a)])
        _frond(B, crown + np.array([0, 0, -0.3 * s]), d, 3.2 * s, -0.2 * s, 1.1, 0.7 * s, (hx("#8a6a3a"), hx("#a98250"), hx("#c8a468")), m=7)
    for i in range(int(rng.integers(3, 6))):
        a = rng.uniform(0, 2 * math.pi)
        v, f = sphere((crown[0] + 0.55 * s * math.cos(a), crown[1] + 0.55 * s * math.sin(a), crown[2] - 0.55 * s), 0.3 * s, 6, 4)
        B["solid"].add(v, f, shade(v, f, hx("#7a5330"), 0.3, 0.75))


def p_bush3(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.6)
    kind = rng.choice(["bush", "bush", "flower", "broad"])
    if kind == "broad":
        g0, g1, g2 = hx("#2c8a45"), hx("#55bb5a"), hx("#9fe071")
        for i in range(8):
            a = 2 * math.pi * i / 8 + rng.uniform(-0.2, 0.2)
            d = np.array([math.cos(a), math.sin(a)])
            _frond(B, np.array([x, y, z + 0.3 * s]), d, rng.uniform(2.6, 3.6) * s, rng.uniform(1.3, 2.2) * s, rng.uniform(0.35, 0.7), 0.75 * s, (g0, g1, g2), m=6, jag=False, drop_edge=0.12)
        return
    greens = [hx(c) for c in ("#3fa85a", "#58c065", "#2f9650", "#79d070", "#4cb560")]
    nl = int(rng.integers(5, 8))
    for k in range(nl):
        r = rng.uniform(0.8, 1.45) * s
        ox, oy = rng.uniform(-1.1, 1.1) * s, rng.uniform(-1.1, 1.1) * s
        oz = (0.3 + 0.35 * rng.uniform()) * s
        v, f = sphere((x + ox, y + oy, z + oz), r, 7, 5, 0.72)
        base = greens[int(rng.integers(len(greens)))] * rng.uniform(0.92, 1.1)
        B["solid"].add(v, f, shade(v, f, base, 0.36, 0.7))
    if kind == "flower":
        fc = [hx(c) for c in ("#ff5f9e", "#ffd23f", "#ffffff", "#ff8c42", "#c77dff")]
        c = fc[int(rng.integers(len(fc)))]
        for k in range(int(rng.integers(8, 14))):
            ox, oy = rng.uniform(-1.4, 1.4) * s, rng.uniform(-1.4, 1.4) * s
            v, f = sphere((x + ox, y + oy, z + (1.0 + 0.45 * rng.uniform()) * s), 0.2 * s, 5, 3)
            B["solid"].add(v, f, c)


def _chunk(c, R, H, n, rot, taper, rng, jit=0.18):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    r0 = R * (1 + rng.uniform(-jit, jit, n))
    r1 = R * taper * (1 + rng.uniform(-jit, jit, n))
    sh = rng.uniform(-0.12, 0.12, 2) * R
    bot = np.column_stack([c[0] + r0 * np.cos(ang), c[1] + r0 * np.sin(ang), np.full(n, c[2])])
    top = np.column_stack([c[0] + sh[0] + r1 * np.cos(ang), c[1] + sh[1] + r1 * np.sin(ang), np.full(n, c[2] + H)])
    V = np.vstack([bot, top, [[c[0] + sh[0], c[1] + sh[1], c[2] + H]]])
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, n + i, n + j]]
    return V, np.array(F), n


def p_rockstack3(B, p, rng):
    """Layered sandstone stack with warm strata, darker crevice faces, a grassy cap with bush and optional palm."""
    x, y, z = p["x"], p["y"], p["z"]
    R, H = p.get("R", 9.0), p.get("H", 20.0)
    tiers = int(p.get("tiers", 4))
    strata = [hx(c) for c in ("#f6d9a8", "#ebc08a", "#dba670", "#cf9560", "#f0cf9c")]
    zc, rr = z, R
    th = H / tiers
    rot = rng.uniform(0, 1)
    for t in range(tiers):
        n = int(rng.integers(8, 11))
        V, F, nn = _chunk((x + rng.uniform(-0.12, 0.12) * R, y + rng.uniform(-0.12, 0.12) * R, zc), rr, th * rng.uniform(0.9, 1.15), n, rot + 0.3 * t, rng.uniform(0.82, 0.96), rng)
        col = strata[(t + int(rng.integers(0, 2))) % len(strata)] * rng.uniform(0.94, 1.05)
        cols = shade(V, F, col, 0.42, 0.66)
        topf = np.arange(len(F)) % 3 == 2
        cols[topf] = col * 0.98
        B["solid"].add(V, F, cols)
        zc += th * 0.92
        rr *= rng.uniform(0.78, 0.9)
    capr = rr * 1.05
    V, F, nn = _chunk((x, y, zc - 0.2), capr, 1.0, 9, rot, 0.85, rng, 0.12)
    cols = shade(V, F, hx("#6fd35a"), 0.3, 0.8)
    cols[np.arange(len(F)) % 3 != 2] = hx("#d9a56a") * 0.92
    B["solid"].add(V, F, cols)
    if rr > 2.4 and rng.uniform() < 0.8:
        p_bush3(B, dict(x=x + rng.uniform(-0.5, 0.5) * rr, y=y + rng.uniform(-0.5, 0.5) * rr, z=zc + 0.8, s=rr * 0.5), rng)
    if rr > 3.2 and rng.uniform() < 0.45:
        p_palm3(B, dict(x=x + rng.uniform(-0.4, 0.4) * rr, y=y + rng.uniform(-0.4, 0.4) * rr, z=zc + 0.8, s=min(rr * 0.35, 1.4)), rng)


def p_cloud3(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.4)
    nb = int(rng.integers(9, 14))
    top, bot = hx("#ffffff"), hx("#bfd2ee")
    for k in range(nb):
        r = rng.uniform(16, 34) * s * (1.2 - 0.5 * abs(k / max(nb - 1, 1) - 0.5))
        ox = (k / max(nb - 1, 1) - 0.5) * 120 * s + rng.uniform(-10, 10)
        oy = rng.uniform(-22, 22) * s
        oz = rng.uniform(-3, 8) * s + 0.25 * r
        V, F = sphere((0, 0, 0), r, 8, 5, 0.55)
        V[:, 2] = np.maximum(V[:, 2], -0.18 * r)
        V = V + np.array([x + ox, y + oy, z + oz])
        P = V[F]
        n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        t = np.clip((n[:, 2] + 0.55) / 1.4, 0, 1)[:, None]
        cols = bot * (1 - t) + top * t
        B["solid"].add(V, F, cols * rng.uniform(0.97, 1.03))


# ------------------------------------------------------------------------------------------------ crowds, flags, confetti
def p_crowd3(B, p, rng):
    x, y, z, face = p["x"], p["y"], p["z"], p["face"]
    n = int(p.get("n", 14))
    f = L2W(x, y, z, face - 90)
    wd, dp = p.get("w", 8.0), p.get("d", 3.4)
    spots = []
    for i in range(n):
        q = f(rng.uniform(-wd / 2, wd / 2), rng.uniform(-dp / 2, dp / 2) + 0.5 * (i % 3))
        spots.append((q[0], q[1], z, face + rng.uniform(-24, 24)))
    people(B["people"], spots, rng, palette=["#ff6f59", "#22c3c8", "#ffd23f", "#ff5f9e", "#8b6cf0", "#3dbb6b", "#ffffff", "#2f7fe0", "#ff9f43", "#f2f2f2"])
    for k in range(int(p.get("flags", 3))):
        q = f(rng.uniform(-wd / 2, wd / 2), rng.uniform(-dp / 2, dp / 2))
        beam(B, (q[0], q[1], z), (q[0], q[1], z + 3.6), hx("#e9eef2"), 0.07)
        c = [CORAL, TURQ, SUN, PINK][int(rng.integers(4))]
        hd = math.radians(face)
        top = np.array([q[0], q[1], z + 3.6])
        B["solid"].add(np.array([top, top + np.array([math.cos(hd), math.sin(hd), 0.0]) * 1.5 + [0, 0, -0.25], top + [0, 0, -0.9]]), [[0, 1, 2]], c)
    for k in range(int(p.get("balloons", 2))):
        q = f(rng.uniform(-wd / 2, wd / 2), rng.uniform(-dp / 2, dp / 2))
        hh = rng.uniform(3.4, 5.0)
        beam(B, (q[0], q[1], z + 1.6), (q[0], q[1], z + hh), hx("#ffffff"), 0.03)
        v, w = sphere((q[0], q[1], z + hh + 0.5), 0.55, 6, 4, 1.15)
        B["solid"].add(v, w, shade(v, w, [CORAL, TURQ, SUN, PINK][int(rng.integers(4))], 0.4, 0.75))


def p_crowdline(B, p, rng):
    """Spectators behind a low fence along a stretch of road. p['spots'] = [(x, y, z, face)], p['fence'] = [(x, y, z)] polyline."""
    spots = [tuple(s_) for s_ in p["spots"]]
    people(B["people"], spots, rng, palette=["#ff6f59", "#22c3c8", "#ffd23f", "#ff5f9e", "#8b6cf0", "#3dbb6b", "#ffffff", "#2f7fe0", "#ff9f43"])
    pts = np.array(p["fence"], float)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 0.5:
            continue
        ang = math.degrees(math.atan2(d[1], d[0]))
        for rz in (0.55, 1.05):
            c = (a + b) / 2
            v, f = box((c[0], c[1], c[2] + rz), (L + 0.1, 0.1, 0.12), rot=ang)
            B["solid"].add(v, f, shade(v, f, hx("#f5efe6"), 0.2, 0.85))
        v, f = frustum((a[0], a[1], a[2]), 0.07, 0.07, 1.15, 4)
        B["solid"].add(v, f, hx("#c9a36a"))
    for k, s_ in enumerate(spots):
        if k % 5 == 2:
            hd = math.radians(s_[3])
            top = np.array([s_[0], s_[1], s_[2] + 3.1])
            beam(B, (s_[0], s_[1], s_[2] + 1.4), tuple(top), hx("#e9eef2"), 0.06)
            c = [CORAL, TURQ, SUN, PINK][k % 4]
            B["solid"].add(np.array([top, top + np.array([math.cos(hd), math.sin(hd), 0.0]) * 1.3 + [0, 0, -0.2], top + [0, 0, -0.8]]), [[0, 1, 2]], c)


def p_confetti(B, p, rng):
    x, y, z, R, n = p["x"], p["y"], p["z"], p.get("R", 6.0), int(p.get("n", 70))
    cols = [hx(c) for c in ("#ff4d6d", "#ffd23f", "#2ec4b6", "#ff9f1c", "#7b5cff", "#ffffff", "#3ddc97")]
    for i in range(n):
        c = np.array([x, y, z]) + rng.normal(0, R * 0.45, 3) * np.array([1, 1, 0.8])
        a = rng.uniform(0, 2 * math.pi)
        w, h = rng.uniform(0.22, 0.42), rng.uniform(0.12, 0.22)
        u = np.array([math.cos(a), math.sin(a), rng.uniform(-0.4, 0.4)])
        u /= np.linalg.norm(u)
        v = np.cross(u, [0, 0, 1.0])
        v /= max(np.linalg.norm(v), 1e-6)
        P = np.array([c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2])
        B["solid"].add(P, tri_quads([[0, 1, 2, 3]]), cols[int(rng.integers(len(cols)))])


def p_kite(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 2.0)
    yaw = rng.uniform(0, 360)
    R = rotz(yaw) @ rotx(rng.uniform(-35, -10))
    w, h = 0.8 * s, 1.1 * s
    pts = np.array([[0, 0, h], [w, 0, 0], [0, 0, -1.25 * h], [-w, 0, 0], [0, 0, 0]]) @ R.T + np.array([x, y, z])
    ca, cb = [CORAL, TURQ, SUN, PINK][int(rng.integers(4))], [WH, NAVY, CORAL, SUN][int(rng.integers(4))]
    B["solid"].add(pts, [[0, 1, 4], [4, 1, 2]], ca)
    B["solid"].add(pts, [[0, 4, 3], [4, 2, 3]], cb)
    tail = [pts[2] + R @ np.array([0.35 * math.sin(i * 1.3) * s, 0.0, -0.55 * s * i]) for i in range(1, 8)]
    for i, q in enumerate(tail):
        B["solid"].add(np.array([q, q + R @ np.array([0.3 * s, 0, 0]), q + R @ np.array([0.15 * s, 0, -0.35 * s])]), [[0, 1, 2]], [CORAL, TURQ, SUN, PINK][i % 4])


def p_paraglider(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.6)
    hd = p.get("hd", 0.0)
    f = L2W(x, y, z, hd)
    n = 12
    L_, R_ = [], []
    for i in range(n + 1):
        t = i / n * 2 - 1
        c = f(t * 4.2 * s, 0.0, 0.0 - 1.0 * s * (t * t) * 0.6)
        L_.append(f(t * 4.2 * s, 1.0 * s, -1.0 * s * t * t * 0.6 + 0.0)); R_.append(f(t * 4.2 * s, -1.0 * s, -1.0 * s * t * t * 0.6 + 0.0))
    V, F = strip(np.array(L_), np.array(R_))
    cols = np.array([[CORAL, WH, TURQ, SUN][(i // 2) % 4] for i in range(n)])
    B["solid"].add(V, F, np.vstack([cols, cols]))
    for t in (-0.8, -0.4, 0.4, 0.8):
        beam(B, f(t * 4.2 * s, 0, -1.0 * s * t * t * 0.6), f(0, 0, -4.2 * s), hx("#e9eef2"), 0.03)
    people(B["people"], [(f(0, 0, -5.9 * s)[0], f(0, 0, -5.9 * s)[1], f(0, 0, -5.9 * s)[2], hd + 90)], rng, scale=(1.0, 1.0))


# ------------------------------------------------------------------------------------------------ village
WALLS = [hx(c) for c in ("#ff9fb2", "#ffd27a", "#8fe0d4", "#9ccbff", "#ffb27a", "#f6f0e2", "#c9a8ff", "#ff7f70")]
ROOFS = [hx(c) for c in ("#c4573c", "#2f7a96", "#8a4566", "#d98a3a")]


def p_house3(B, p, rng):
    x, y, z = p["x"], p["y"], p["z"]
    w, d, h = p.get("w", 9.0), p.get("d", 7.5), p.get("h", 7.5)
    rot = p.get("rot", 0.0)
    wall = WALLS[int(p.get("seed", rng.integers(8))) % len(WALLS)]
    roof = ROOFS[int(p.get("seed", rng.integers(4)) // 2) % len(ROOFS)]
    f = L2W(x, y, z, rot)

    def bx(cx, cy, cz, sx, sy, sz, col, k=0.28, group="solid"):
        c = f(cx, cy, cz)
        v, ff = box(c, (sx, sy, sz), rot=rot)
        B[group].add(v, ff, shade(v, ff, col, k, 0.72) if group == "solid" else col)

    bx(0, 0, -0.2, w + 0.9, d + 0.9, 0.7, hx("#d8cdb8"), 0.2)
    bx(0, 0, 0.5, w, d, h, wall)
    bx(0, 0, h * 0.5 + 0.5, w + 0.16, d + 0.16, 0.22, hx("#fffaf0"), 0.15)
    ov = 0.7
    hw, hd_ = w / 2 + ov, d / 2 + ov
    rh = d * 0.46
    zr = 0.5 + h
    A, Bb, C, D = f(-hw, -hd_, zr), f(hw, -hd_, zr), f(hw, hd_, zr), f(-hw, hd_, zr)
    R1, R2 = f(-hw, 0, zr + rh), f(hw, 0, zr + rh)
    for side, (q0, q1) in enumerate(((A, Bb), (D, C))):
        for band in range(3):
            t0, t1 = band / 3, (band + 1) / 3
            lo0, lo1 = np.array(q0) * (1 - t0) + (np.array(R1) if side == 0 else np.array(R1)) * t0, np.array(q1) * (1 - t0) + np.array(R2) * t0
            hi0, hi1 = np.array(q0) * (1 - t1) + np.array(R1) * t1, np.array(q1) * (1 - t1) + np.array(R2) * t1
            Q = np.array([lo0, lo1, hi1, hi0])
            B["solid"].add(Q, tri_quads([[0, 1, 2, 3]]), shade(Q, tri_quads([[0, 1, 2, 3]]), roof * (0.92 + 0.08 * (band % 2)), 0.4, 0.62))
    for (g0, g1, g2) in ((A, D, R1), (Bb, C, R2)):
        T = np.array([np.array(g0) + [0, 0, 0], g1, g2])
        B["solid"].add(T, [[0, 1, 2]], wall * 0.92)
    bx(w * 0.28, d * 0.12, zr + 0.2, 0.9, 0.9, rh * 0.8, hx("#b9a58d"), 0.3)
    nwin = max(2, int(w // 3.3))
    for fl, zz in enumerate((2.0, 2.0 + h * 0.5)):
        for k in range(nwin):
            lx = -w / 2 + (k + 0.5) * w / nwin
            if fl == 0 and k == nwin // 2:
                continue
            bx(lx, d / 2 + 0.03, zz + 0.4, 1.35, 0.14, 1.75, hx("#fffaf0"), 0.15)
            bx(lx, d / 2 + 0.08, zz + 0.55, 1.05, 0.12, 1.45, hx("#2a6c86"), 0.1)
            bx(lx - 0.95, d / 2 + 0.1, zz + 0.55, 0.45, 0.1, 1.45, [CORAL, TURQ, SUN][k % 3], 0.2)
            bx(lx + 0.95, d / 2 + 0.1, zz + 0.55, 0.45, 0.1, 1.45, [CORAL, TURQ, SUN][k % 3], 0.2)
            if fl == 0 and rng.uniform() < 0.8:
                bx(lx, d / 2 + 0.35, zz + 0.2, 1.4, 0.45, 0.35, hx("#8a5a3a"), 0.2)
                for q in range(4):
                    c = f(lx + (q - 1.5) * 0.3, d / 2 + 0.35, zz + 0.65)
                    v, ff = sphere(c, 0.17, 5, 3)
                    B["solid"].add(v, ff, [PINK, SUN, WH, CORAL][(q + k) % 4])
    dc = nwin // 2
    dx_ = -w / 2 + (dc + 0.5) * w / nwin
    bx(dx_, d / 2 + 0.03, 0.5, 1.5, 0.16, 2.5, hx("#7a4a30"), 0.2)
    for sI in range(5):
        c0 = f(dx_ - 1.25 + sI * 0.5, d / 2 + 0.2, 3.1)
        Q = np.array([f(dx_ - 1.25 + sI * 0.5, d / 2 + 0.1, 3.2), f(dx_ - 0.75 + sI * 0.5, d / 2 + 0.1, 3.2), f(dx_ - 0.75 + sI * 0.5, d / 2 + 1.5, 2.6), f(dx_ - 1.25 + sI * 0.5, d / 2 + 1.5, 2.6)])
        B["solid"].add(Q, tri_quads([[0, 1, 2, 3]]), (CORAL if sI % 2 == 0 else WH))
    c = f(dx_ + 1.5, d / 2 + 0.35, 2.6)
    v, ff = sphere(c, 0.2, 5, 3)
    B["emit"].add(v, ff, hx("#ffd9a0") * 0.9)


# ------------------------------------------------------------------------------------------------ boats and sea life
def _hull(L, Bm, Dp, n=8):
    t = np.linspace(0, 1, n)
    xs = (t - 0.5) * L
    w = np.maximum(Bm / 2 * np.sin(np.pi * np.clip(t, 0, 1) ** 0.72) * (1 - 0.3 * t), 0.05)
    deck = 0.85 + 0.55 * t ** 3
    secs = []
    for i in range(n):
        secs.append(np.array([[xs[i], -w[i], deck[i]], [xs[i], -0.78 * w[i], deck[i] * 0.15 - 0.2], [xs[i], 0.0, -Dp], [xs[i], 0.78 * w[i], deck[i] * 0.15 - 0.2], [xs[i], w[i], deck[i]]]))
    return secs, xs, w, deck


def _boat_hull(B, f, L, Bm, Dp, hull_col, trim):
    secs, xs, w, deck = _hull(L, Bm, Dp)
    n = len(secs)
    for sg in range(4):
        Lp = np.array([f(*secs[i][sg]) for i in range(n)])
        Rp = np.array([f(*secs[i][sg + 1]) for i in range(n)])
        V, F = strip(Lp, Rp)
        col = hull_col if sg in (0, 3) else hx("#1d3552")
        B["solid"].add(V, F, shade(V, F, col, 0.35, 0.72))
    Lp = np.array([f(xs[i], -w[i], deck[i]) for i in range(n)])
    Rp = np.array([f(xs[i], w[i], deck[i]) for i in range(n)])
    V, F = strip(Lp, Rp)
    B["solid"].add(V, F, shade(V, F, hx("#e6c597"), 0.3, 0.8))
    Lp = np.array([f(xs[i], -w[i] * 0.97, deck[i] * 0.55) for i in range(n)])
    Rp = np.array([f(xs[i], -w[i] * 1.01, deck[i] * 0.78) for i in range(n)])
    for sd in (-1, 1):
        Lp = np.array([f(xs[i], sd * w[i] * 0.99, deck[i] * 0.52) for i in range(n)])
        Rp = np.array([f(xs[i], sd * w[i] * 1.0, deck[i] * 0.74) for i in range(n)])
        V, F = strip(Lp, Rp)
        B["solid"].add(V, F, trim)


def _wake(B, f, L, wmax, z):
    m = 9
    Lp, Rp = [], []
    for i in range(m):
        t = i / (m - 1)
        wd = (0.4 + wmax * t) * (1.0 if i % 2 == 0 else 0.7)
        Lp.append(f(-L * 0.5 - L * 1.3 * t, wd, z)); Rp.append(f(-L * 0.5 - L * 1.3 * t, wd + 0.5 * (1 - 0.6 * t), z))
    V, F = strip(np.array(Lp), np.array(Rp))
    B["solid"].add(V, F, hx("#f4fbff"))
    Lp, Rp = [], []
    for i in range(m):
        t = i / (m - 1)
        wd = (0.4 + wmax * t) * (1.0 if i % 2 == 0 else 0.7)
        Lp.append(f(-L * 0.5 - L * 1.3 * t, -wd, z)); Rp.append(f(-L * 0.5 - L * 1.3 * t, -wd - 0.5 * (1 - 0.6 * t), z))
    V, F = strip(np.array(Lp), np.array(Rp))
    B["solid"].add(V, F, hx("#f4fbff"))


def p_sailboat(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    s = p.get("s", 1.0)
    L, Bm = 11.0 * s, 3.4 * s
    f = L2W(x, y, z, rot)
    hull = [WH, hx("#fff3df"), CORAL, TURQ, NAVY][int(p.get("c", rng.integers(5))) % 5]
    _boat_hull(B, f, L, Bm, 0.9 * s, hull, [CORAL, TURQ, SUN][int(rng.integers(3))])
    mast_x = 0.15 * L
    beam(B, f(mast_x, 0, 0.9), f(mast_x, 0, 14.0 * s), hx("#e9eef2"), 0.22)
    beam(B, f(mast_x, 0, 2.3 * s), f(-0.42 * L, 0, 2.6 * s), hx("#e9eef2"), 0.16)
    sc1 = [WH, hx("#fff3df"), CORAL, TURQ, SUN][int(rng.integers(5))]
    bulge = 0.9 * s
    main = np.array([f(mast_x, 0, 13.4 * s), f(mast_x, 0, 2.5 * s), f(-0.42 * L, 0, 2.7 * s), f(-0.12 * L, bulge, 7.6 * s)])
    B["solid"].add(main, [[0, 1, 3], [1, 2, 3], [0, 3, 2]], shade(main, np.array([[0, 1, 3], [1, 2, 3], [0, 3, 2]]), sc1, 0.3, 0.86))
    jib = np.array([f(mast_x + 0.02 * L, 0, 12.4 * s), f(mast_x + 0.02 * L, 0, 1.6 * s), f(0.46 * L, 0, 1.4 * s), f(0.22 * L, -bulge * 0.8, 6.0 * s)])
    sc2 = [WH, hx("#fff3df"), TURQ, SUN, CORAL][int(rng.integers(5))]
    B["solid"].add(jib, [[0, 1, 3], [1, 2, 3], [0, 3, 2]], shade(jib, np.array([[0, 1, 3], [1, 2, 3], [0, 3, 2]]), sc2, 0.3, 0.86))
    top = f(mast_x, 0, 14.0 * s)
    B["solid"].add(np.array([top, f(mast_x - 1.4 * s, 0, 14.3 * s), f(mast_x, 0, 13.6 * s)]), [[0, 1, 2]], [CORAL, SUN, PINK][int(rng.integers(3))])
    _wake(B, f, L, 1.9 * s, z + 0.03)


def p_fishboat(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    s = p.get("s", 1.0)
    L, Bm = 12.0 * s, 4.2 * s
    f = L2W(x, y, z, rot)
    _boat_hull(B, f, L, Bm, 1.0 * s, [hx("#2f7fe0"), hx("#e84a4a"), hx("#ffd23f"), hx("#3dbb6b")][int(rng.integers(4))], WH)

    def bx(cx, cy, cz, sx, sy, sz, col, k=0.3):
        v, ff = box(f(cx, cy, cz), (sx, sy, sz), rot=rot)
        B["solid"].add(v, ff, shade(v, ff, col, k, 0.74))
    bx(-0.12 * L, 0, 0.9, 3.4 * s, 2.5 * s, 2.3 * s, hx("#f7f2e8"))
    bx(-0.12 * L, 0, 3.15 * s, 3.8 * s, 2.8 * s, 0.35 * s, [CORAL, TURQ][int(rng.integers(2))])
    bx(-0.12 * L + 1.4 * s, 0, 1.9 * s, 0.3 * s, 2.0 * s, 1.0 * s, hx("#2a6c86"), 0.1)
    beam(B, f(0.12 * L, 0, 0.9), f(0.12 * L, 0, 8.0 * s), hx("#e9eef2"), 0.18)
    beam(B, f(0.12 * L, 0, 7.2 * s), f(0.34 * L, 0, 2.4 * s), hx("#e9eef2"), 0.08)
    for k in range(3):
        bx(-0.38 * L, (k - 1) * 0.9 * s, 0.9, 0.8 * s, 0.8 * s, 0.7 * s, hx("#c79a5e"))
    for k in range(5):
        c = f(0.2 * L + 0.4 * k, (k % 2 - 0.5) * 0.7, 1.0)
        v, ff = sphere(c, 0.28 * s, 5, 3)
        B["solid"].add(v, ff, [CORAL, WH][k % 2])
    c = f(0.12 * L, 0, 8.2 * s)
    v, ff = sphere(c, 0.3 * s, 5, 3)
    B["emit"].add(v, ff, hx("#ffd9a0") * 0.8)
    _wake(B, f, L, 2.2 * s, z + 0.03)


def p_windsurf(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    s = p.get("s", 1.6)
    f = L2W(x, y, z, rot)
    v, ff = box(f(0, 0, 0.02), (2.9 * s, 0.7 * s, 0.14), rot=rot, taper=0.7)
    B["solid"].add(v, ff, shade(v, ff, [CORAL, TURQ, SUN][int(rng.integers(3))], 0.3, 0.8))
    beam(B, f(0.1, 0, 0.1), f(0.1, 0, 4.4 * s), hx("#e9eef2"), 0.07)
    sc = [CORAL, TURQ, SUN, PINK, WH][int(rng.integers(5))]
    sail = np.array([f(0.1, 0, 4.3 * s), f(0.1, 0, 0.9 * s), f(-1.9 * s, 0.35 * s, 1.0 * s), f(-1.1 * s, 0.45 * s, 3.1 * s)])
    B["solid"].add(sail, [[0, 1, 2], [0, 2, 3]], shade(sail, np.array([[0, 1, 2], [0, 2, 3]]), sc, 0.3, 0.86))
    q = f(0.0, 0.2 * s, 0.1)
    people(B["people"], [(q[0], q[1], q[2], rot + 90)], rng, scale=(0.9, 0.95))
    _wake(B, f, 2.6 * s, 1.1 * s, z + 0.03)


def p_dolphin(B, p, rng):
    x, y, z, hd = p["x"], p["y"], p["z"], p.get("hd", 0.0)
    s = p.get("s", 1.7)
    f = L2W(x, y, z, hd)
    L, Hh = 4.0 * s, 3.4 * s
    ts = np.linspace(0, 1, 13)
    pts = np.array([f((t - 0.5) * L, 0.0, Hh * math.sin(math.pi * t) ** 0.9 - 0.2) for t in ts])
    rad = s * 0.5 * np.interp(ts, [0, 0.15, 0.4, 0.65, 0.82, 0.93, 1.0], [0.06, 0.22, 0.5, 0.55, 0.4, 0.17, 0.1])
    V, F, seg = tube(pts, rad, 8)
    Pf = V[F]
    n = np.cross(Pf[:, 1] - Pf[:, 0], Pf[:, 2] - Pf[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    cols = np.where((n[:, 2] > -0.05)[:, None], np.tile(hx("#5a82a8"), (len(F), 1)), np.tile(hx("#e8f1f7"), (len(F), 1)))
    B["solid"].add(V, F, shade(V, F, cols, 0.3, 0.8))
    T = pts[0] - pts[2]
    T /= np.linalg.norm(T)
    side = np.cross(T, [0, 0, 1.0])
    side /= max(np.linalg.norm(side), 1e-6)
    tip = pts[0] + T * 0.55 * s
    B["solid"].add(np.array([pts[0] - T * 0.1 * s, tip + side * 1.0 * s, tip - side * 1.0 * s]), [[0, 1, 2]], hx("#4a6e92"))
    B["solid"].add(np.array([pts[0] - T * 0.1 * s, tip - side * 1.0 * s, tip + side * 1.0 * s]), [[0, 1, 2]], hx("#4a6e92"))
    k = 6
    B["solid"].add(np.array([pts[k] + [0, 0, rad[k] * 0.9], pts[k - 2] + [0, 0, rad[k - 2] * 0.5], pts[k] + [0, 0, rad[k] + 0.8 * s]]), [[0, 1, 2]], hx("#4a6e92"))
    for sd in (-1, 1):
        B["solid"].add(np.array([pts[7] + side * sd * rad[7] * 0.8, pts[6] + side * sd * (rad[6] + 0.7 * s) - [0, 0, 0.4 * s], pts[5] + side * sd * rad[5]]), [[0, 1, 2]], hx("#4a6e92"))
    for end in (0.02, 0.98):
        c = f((end - 0.5) * L * 1.15, 0, 0.0)
        for q in range(9):
            a = 2 * math.pi * q / 9
            v, w = sphere((c[0] + math.cos(a) * 1.5 * s, c[1] + math.sin(a) * 1.5 * s, z + 0.15), rng.uniform(0.3, 0.55) * s, 5, 3, 0.7)
            B["solid"].add(v, w, hx("#f4fbff"))


def p_waterfall(B, p, rng):
    """Cliff waterfall: translucent blue sheets with a white crest and a foam pool. p: x, y (top edge), z_top, z_bot, hd (facing water), w."""
    x, y, zt, zb, hd, w = p["x"], p["y"], p["z_top"], p["z_bot"], p.get("hd", 0.0), p.get("w", 5.0)
    f = L2W(x, y, 0.0, hd)
    cols = [hx("#bfeeff"), hx("#e8fbff"), hx("#8fdcf2")]
    n = 5
    for i in range(n):
        lx = (i - (n - 1) / 2) * w / n
        m = 6
        Lp = np.array([f(lx - w / n * 0.45, 0.4 * (j / (m - 1)) ** 2 * 3.0 + 0.0, zt - (zt - zb) * j / (m - 1)) for j in range(m)])
        Rp = np.array([f(lx + w / n * 0.45, 0.4 * (j / (m - 1)) ** 2 * 3.0 + 0.0, zt - (zt - zb) * j / (m - 1)) for j in range(m)])
        V, F = strip(Lp, Rp)
        B["glass"].add(V, F, cols[i % 3])
    for q in range(10):
        c = f(rng.uniform(-w / 2, w / 2), 1.6 + rng.uniform(-0.6, 1.4), zb + rng.uniform(0.0, 0.6))
        v, ff = sphere(c, rng.uniform(0.6, 1.4), 6, 4, 0.7)
        B["solid"].add(v, ff, hx("#f4fbff"))
    c = f(0, 0.2, zt + 0.1)
    v, ff = sphere(c, w * 0.35, 6, 3, 0.4)
    B["solid"].add(v, ff, hx("#f4fbff"))


def p_foam3(B, p, rng):
    Lp, Rp = np.array(p["L"], float), np.array(p["R"], float)
    z = p["z"]
    V, F = strip(np.column_stack([Lp, np.full(len(Lp), z)]), np.column_stack([Rp, np.full(len(Rp), z)]))
    B["solid"].add(V, F, hx("#f8fdff"))


PROPS = {k[2:]: v for k, v in dict(globals()).items() if k.startswith("p_") and callable(v)}


# ================================================================================================ v3b: rounder people, nicer boats, far sea, new vegetation
def person_proto3(pose, style=0):
    """Rounder figure than the v1/v2 box person: tapered limbs, elliptical torso, skirt/dress variants, hair buns/ponytails/afro, sun hats, caps, sunglasses, bent arms.
    Local space, feet at z = 0, facing +y; cid 0 shirt, 1 trousers/skirt, 2 skin, 3 hair, 4 accent (hat, bag), 5 shoes, 6 dark,
    7 eye whites, 8 mouth, 9 cheeks. Facial geometry is visible in close shots; no textures or game assets."""
    V, F, cid = [], [], []

    def add(v, f, c):
        off = sum(len(x) for x in V)
        V.append(np.asarray(v, float))
        F.append(np.asarray(f) + off)
        cid.append(np.full(len(f), c))

    def limb(a, b, r0, r1, c, n=5):
        v, f, _ = tube([a, b], [r0, r1], n)
        add(v, f, c)

    def hair_cap(c, radius, segments=10, rings=6, squash=0.8, colour=3):
        # Cut away the front of the old full hair sphere. It covered the eyes.
        v, f = sphere(c, radius, segments, rings, squash)
        centres = v[f].mean(axis=1)
        front = centres[:, 1] > 0.025
        keep = (centres[:, 2] > hz + 0.058) | (~front & (centres[:, 2] > hz - 0.070))
        add(v, f[keep], colour)

    def oval(c, size, colour, segments=8, rings=5):
        v, f = sphere((0, 0, 0), 1, segments, rings)
        add(v * np.array(size) + np.array(c), f, colour)

    dress = style in (2, 5)
    stance = 0.07 if pose == "jump" else 0.0
    for sx in (-1, 1):                                    # legs + shoes
        hip = np.array([sx * 0.085, 0.0, 0.82])
        ank = np.array([sx * (0.10 + stance), 0.02 * sx, 0.09])
        if not dress:
            limb(hip, ank, 0.092, 0.066, 1, 6)
        else:
            limb(np.array([sx * 0.085, 0.0, 0.6]), ank, 0.06, 0.05, 2, 5)
        v, f = box((sx * (0.10 + stance), 0.05, 0.0), (0.15, 0.30, 0.10), taper=0.8); add(v, f, 5)
    if dress:
        v, f = frustum((0, 0, 0.56), 0.30, 0.15, 0.58, 8); add(v, f, 1)
    # torso (elliptical frustum: hips -> shoulders), waist band, neck
    v, f = frustum((0, 0, 0.80), 0.165, 0.215, 0.60, 10)
    v[:, 0] *= 1.28; v[:, 1] *= 0.82
    add(v, f, 0)
    if not dress:
        v, f = frustum((0, 0, 0.78), 0.175, 0.17, 0.09, 8); v[:, 0] *= 1.2; v[:, 1] *= 0.84; add(v, f, 1)
    v, f = frustum((0, 0, 1.40), 0.055, 0.05, 0.10, 6); add(v, f, 2)
    hz = 1.62
    v, f = sphere((0, 0, hz), 0.135, 12, 8, 1.08); add(v, f, 2)
    for sx in (-1, 1):
        oval((sx * .132, -.005, hz - .008), (.024, .026, .038), 2, 6, 4)
        oval((sx * .076, .112, hz - .034), (.026, .007, .016), 9, 6, 4)
        oval((sx * .047, .128, hz + .013), (.025, .013, .016), 7)
        oval((sx * .046, .140, hz + .013), (.0095, .0045, .012), 6, 7, 4)
        oval((sx * .044, .144, hz + .017), (.0030, .0015, .0035), 7, 5, 3)
        brow = np.array([[sx * .071, .121, hz + .042], [sx * .047, .131, hz + .048], [sx * .025, .129, hz + .042]])
        v, f, _ = tube(brow, [.004, .006, .004], 4); add(v, f, 3)
    # A small faceted nose and an upward-curved smile stay legible at close range.
    nose = np.array([[-.016, .129, hz-.006], [.016, .129, hz-.006], [0, .159, hz-.026], [0, .125, hz-.039]])
    add(nose, np.array([[0,1,2],[0,2,3],[1,3,2]]), 2)
    smile = np.array([[x, .121 + .006*(1-(x/.040)**2), hz-.057+.015*(x/.040)**2] for x in np.linspace(-.040,.040,7)])
    v, f, _ = tube(smile, [.0045]*7, 4); add(v, f, 8)
    # hair / hats
    if style == 0:                                                                     # short hair
        hair_cap((0, -0.015, hz + 0.045), 0.145)
    elif style == 1:                                                                   # cap with visor
        hair_cap((0, -0.01, hz + 0.05), 0.145, squash=0.72, colour=4)
        v, f = box((0, 0.17, hz + 0.075), (0.22, 0.17, 0.03), taper=0.8); add(v, f, 4)
    elif style == 2:                                                                   # ponytail
        hair_cap((0, -0.015, hz + 0.045), 0.145)
        limb(np.array([0, -0.15, hz + 0.02]), np.array([0, -0.26, hz - 0.30]), 0.045, 0.02, 3, 5)
    elif style == 3:                                                                   # afro
        hair_cap((0, -0.025, hz + 0.05), 0.19, squash=0.9)
    elif style == 4:                                                                   # sun hat
        v, f = frustum((0, 0, hz + 0.075), 0.30, 0.27, 0.03, 10); add(v, f, 4)
        v, f = frustum((0, 0, hz + 0.09), 0.14, 0.12, 0.10, 8); add(v, f, 4)
        hair_cap((0, -0.02, hz + 0.01), 0.14, squash=0.7)
    else:                                                                              # bun
        hair_cap((0, -0.015, hz + 0.04), 0.145, squash=0.78)
        v, f = sphere((0, -0.04, hz + 0.19), 0.07, 5, 3); add(v, f, 3)
    if style in (1, 4):
        for sx in (-1, 1):
            oval((sx*.05, .145, hz+.014), (.041, .010, .023), 6)
        v, f = box((0, .144, hz+.012), (.036, .012, .008)); add(v, f, 6)
    # arms: (shoulder angle from hanging, bend, forward) per side, left then right
    P = dict(idle=((12, 18, 0.1), (12, 18, 0.1)), up=((158, 12, 0.0), (158, 12, 0.0)), wave=((165, 40, 0.0), (28, 40, 0.3)), cheer=((172, 8, 0.0), (172, 8, 0.0)),
             hat=((14, 20, 0.0), (150, 95, 0.2)), point=((20, 25, 0.0), (88, 5, 1.0)), jump=((148, 20, 0.0), (148, 20, 0.0)))[pose if pose != "cheer" else "cheer"]
    for si, sx in enumerate((-1, 1)):
        ang, bend, fwd = P[si]
        sh = np.array([sx * 0.255, 0.0, 1.33])
        a1 = math.radians(ang)
        d1 = np.array([sx * math.sin(a1) * (1 - fwd * 0.6), math.sin(a1) * fwd * 0.9, -math.cos(a1)])
        d1 /= np.linalg.norm(d1)
        el = sh + d1 * 0.285
        a2 = math.radians(ang + bend)
        d2 = np.array([sx * math.sin(a2) * (1 - fwd * 0.6), math.sin(a2) * fwd * 0.9 + 0.04, -math.cos(a2)])
        d2 /= np.linalg.norm(d2)
        hd_ = el + d2 * 0.265
        limb(sh, el, 0.058, 0.048, 0, 5)
        limb(el, hd_, 0.046, 0.04, 2 if ang > 100 or style in (0, 1, 3) else 2, 5)
        v, f = sphere(tuple(hd_), 0.05, 5, 3); add(v, f, 2)
    return np.vstack(V), np.vstack(F), np.concatenate(cid)


def p_tree3(B, p, rng):
    """Broadleaf tree: forked trunk and three stacked faceted crowns in two greens, optional blossoms."""
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 2.0)
    H = rng.uniform(5.5, 8.0) * s
    lean = rng.uniform(0, 2 * math.pi)
    ld = np.array([math.cos(lean), math.sin(lean)])
    pts = np.array([[x + ld[0] * 0.5 * t * t, y + ld[1] * 0.5 * t * t, z + H * 0.55 * t] for t in np.linspace(0, 1, 5)])
    V, F, seg = tube(pts, 0.5 * s * (1 - 0.45 * np.linspace(0, 1, 5)), 6)
    B["solid"].add(V, F, shade(V, F, hx("#8e6c4a"), 0.2, 0.84))
    top = pts[-1]
    pal = [hx("#2f9650"), hx("#4cb85c"), hx("#79d070"), hx("#a6e36e")]
    bl = rng.uniform() < 0.25
    for k in range(int(rng.integers(4, 7))):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(1.6, 2.6) * s * (1.1 - 0.12 * k)
        c = top + np.array([math.cos(a) * rng.uniform(0.0, 1.8) * s, math.sin(a) * rng.uniform(0.0, 1.8) * s, (0.35 * k - 0.3) * s + rng.uniform(0.0, 0.6) * s])
        v, f = sphere(tuple(c), r, 8, 5, 0.78)
        col = (hx("#ff8fb3") if (bl and rng.uniform() < 0.5) else pal[int(rng.integers(4))]) * rng.uniform(0.94, 1.06)
        B["solid"].add(v, f, shade(v, f, col, 0.4, 0.68))


def p_banana(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.8)
    for k in range(int(rng.integers(2, 4))):
        ox, oy = rng.uniform(-0.7, 0.7) * s, rng.uniform(-0.7, 0.7) * s
        H = rng.uniform(2.4, 3.6) * s
        v, f, _ = tube(np.array([[x + ox, y + oy, z], [x + ox * 1.2, y + oy * 1.2, z + H]]), [0.2 * s, 0.14 * s], 6)
        B["solid"].add(v, f, shade(v, f, hx("#9ac05a"), 0.2, 0.84))
        crown = np.array([x + ox * 1.2, y + oy * 1.2, z + H])
        for i in range(7):
            a = 2 * math.pi * i / 7 + rng.uniform(-0.3, 0.3)
            d = np.array([math.cos(a), math.sin(a)])
            _frond(B, crown, d, rng.uniform(2.6, 3.4) * s, rng.uniform(0.2, 0.9) * s, rng.uniform(0.45, 0.8), 0.7 * s, (hx("#2f9650"), hx("#59c25d"), hx("#a6e36e")), m=6, jag=False, drop_edge=0.1)
        if rng.uniform() < 0.7:
            for q in range(5):
                v, f = sphere((crown[0] + 0.3 * math.cos(q), crown[1] + 0.3 * math.sin(q), crown[2] - 0.7 * s), 0.17 * s, 5, 3)
                B["solid"].add(v, f, hx("#f4d24a"))


def p_bamboo(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.6)
    for k in range(int(rng.integers(7, 12))):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.0, 1.2) * s
        H = rng.uniform(5, 8.5) * s
        base = np.array([x + math.cos(a) * r, y + math.sin(a) * r, z])
        tip = base + np.array([math.cos(a) * 0.5, math.sin(a) * 0.5, H])
        v, f, _ = tube(np.array([base, (base + tip) / 2, tip]), [0.09 * s, 0.075 * s, 0.05 * s], 5)
        B["solid"].add(v, f, shade(v, f, hx("#a8c95a") * rng.uniform(0.9, 1.1), 0.2, 0.86))
        for j in range(3):
            t = 0.55 + 0.15 * j
            c = base + (tip - base) * t
            d = np.array([math.cos(a + 1.5 * j), math.sin(a + 1.5 * j), 0.0])
            B["solid"].add(np.array([c, c + d * 1.4 * s + [0, 0, -0.5 * s], c + d * 0.7 * s + [0, 0.25 * s, 0.1 * s]]), [[0, 1, 2]], hx("#6fbf55"))


def p_hibiscus(B, p, rng):
    x, y, z, s = p["x"], p["y"], p["z"], p.get("s", 1.5)
    p_bush3(B, dict(x=x, y=y, z=z, s=s * 0.9), rng)
    c = [hx("#ff4f6e"), hx("#ff8fb3"), hx("#ffd23f"), hx("#ff9f43")][int(rng.integers(4))]
    for q in range(int(rng.integers(9, 15))):
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(0.2, 1.4) * s
        cc = np.array([x + math.cos(a) * r, y + math.sin(a) * r, z + rng.uniform(0.9, 1.7) * s])
        for k in range(5):
            b = 2 * math.pi * k / 5
            B["solid"].add(np.array([cc, cc + np.array([math.cos(b) * 0.34, math.sin(b) * 0.34, 0.12]) * s, cc + np.array([math.cos(b + 1.2) * 0.34, math.sin(b + 1.2) * 0.34, 0.12]) * s]), [[0, 1, 2]], c)
        v, f = sphere(tuple(cc + np.array([0, 0, 0.1 * s])), 0.07 * s, 4, 3); B["emit"].add(v, f, hx("#ffd23f") * 0.35)


def p_field(B, p, rng):
    """Crop / flower field following the terrain: p['grid'] = rows x cols of (x, y, z); alternating rows in two colours with a darker furrow edge."""
    G_ = np.array(p["grid"], float)
    ca, cb = hx(p["a"]), hx(p["b"])
    for i in range(len(G_) - 1):
        L_, R_ = G_[i].copy(), G_[i + 1].copy()
        L_[:, 2] += 0.07; R_[:, 2] += 0.07
        V, Fq = strip(L_, R_)
        col = ca if i % 2 == 0 else cb
        k = len(Fq) // 2
        B["solid"].add(V, Fq, shade(V, Fq, col * rng.uniform(0.95, 1.05), 0.25, 0.8))
    # hedge border along the long edges
    for edge in (G_[:, 0], G_[:, -1]):
        for j in range(len(edge) - 1):
            a, b = edge[j], edge[j + 1]
            v, f, _ = tube(np.array([a + [0, 0, 0.3], b + [0, 0, 0.3]]), [0.55, 0.55], 5)
            B["solid"].add(v, f, shade(v, f, hx("#2f8f48"), 0.3, 0.72))


def p_fence(B, p, rng):
    pts = np.array(p["pts"], float)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        for rz in (0.45, 0.85):
            beam(B, a + [0, 0, rz], b + [0, 0, rz], hx("#f2e8d4"), 0.09)
        beam(B, a, a + [0, 0, 1.0], hx("#c9a36a"), 0.12)
    beam(B, pts[-1], pts[-1] + [0, 0, 1.0], hx("#c9a36a"), 0.12)


def p_pond(B, p, rng):
    x, y, z, R = p["x"], p["y"], p["z"], p.get("R", 6.0)
    V, F = frustum((x, y, z + 0.05), R, R * 0.96, 0.12, 12)
    B["solid"].add(V, F, shade(V, F, hx("#7fe0dc"), 0.2, 0.9))
    V, F = frustum((x, y, z + 0.0), R * 1.12, R * 1.18, 0.1, 12)
    B["solid"].add(V, F, shade(V, F, hx("#e6d29a"), 0.2, 0.85))
    for k in range(int(rng.integers(4, 8))):
        a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.0, R * 0.7)
        V, F = frustum((x + math.cos(a) * r, y + math.sin(a) * r, z + 0.18), 0.7, 0.6, 0.05, 7)
        B["solid"].add(V, F, hx("#3fb04f"))
        if rng.uniform() < 0.4:
            v, f = sphere((x + math.cos(a) * r, y + math.sin(a) * r, z + 0.3), 0.2, 5, 3); B["solid"].add(v, f, hx("#ff8fb3"))
    for k in range(5):
        a = rng.uniform(0, 2 * math.pi)
        p_reeds(B, dict(x=x + math.cos(a) * R * 1.05, y=y + math.sin(a) * R * 1.05, z=z), rng) if False else None


# ---- boats v3b
def _sail(B, p0, p1, p2, bulge, n, cols):
    """Curved sail: triangle p0 (head), p1 (tack), p2 (clew) with a bulge perpendicular to the sail plane; n horizontal panels with alternating colours."""
    p0, p1, p2 = (np.asarray(q, float) for q in (p0, p1, p2))
    nrm = np.cross(p1 - p0, p2 - p0)
    nrm /= max(np.linalg.norm(nrm), 1e-9)
    rows = []
    for i in range(n + 1):
        t = i / n
        a = p0 * (1 - t) + p1 * t
        b = p0 * (1 - t) + p2 * t
        row = []
        for j in range(5):
            u = j / 4
            q = a * (1 - u) + b * u
            row.append(q + nrm * bulge * math.sin(math.pi * u) * (0.25 + 0.75 * math.sin(math.pi * min(t * 0.95 + 0.04, 1))))
        rows.append(np.array(row))
    for i in range(n):
        V, Fq = strip(rows[i], rows[i + 1])
        B["solid"].add(V, Fq, shade(V, Fq, cols[i % len(cols)], 0.32, 0.8))
        B["solid"].add(V[:, :][:], Fq[:, ::-1], shade(V, Fq[:, ::-1], cols[i % len(cols)] * 0.9, 0.3, 0.7))


def p_sailboat2(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    s = p.get("s", 1.0)
    L, Bm = 11.5 * s, 3.5 * s
    f = L2W(x, y, z, rot)
    hull = [WH, hx("#fff3df"), CORAL, TURQ, NAVY, hx("#ffd23f")][int(p.get("c", rng.integers(6))) % 6]
    trim = [CORAL, TURQ, SUN, NAVY][int(rng.integers(4))]
    _boat_hull(B, f, L, Bm, 0.95 * s, hull, trim)
    secs, xs, w, deck = _hull(L, Bm, 0.95 * s)
    for sd in (-1, 1):                                                   # waterline stripe
        Lp = np.array([f(xs[i], sd * w[i] * 0.93, deck[i] * 0.22) for i in range(len(xs))])
        Rp = np.array([f(xs[i], sd * w[i] * 0.97, deck[i] * 0.34) for i in range(len(xs))])
        V, F = strip(Lp, Rp)
        B["solid"].add(V, F, trim)
    q = f(-0.20 * L, 0, 0.9)
    v, ff = box(q, (3.2 * s, 1.9 * s, 0.9 * s), rot=rot, taper=0.84); B["solid"].add(v, ff, shade(v, ff, hx("#f6efe2"), 0.3, 0.8))
    for sd in (-1, 1):
        c = f(-0.20 * L + 0.2, sd * 0.96 * s, 1.2)
        v, ff = box(c, (1.8 * s, 0.06, 0.34 * s), rot=rot); B["solid"].add(v, ff, hx("#2a6c86"))
    mast_x = 0.14 * L
    beam(B, f(mast_x, 0, 0.9), f(mast_x, 0, 15.0 * s), hx("#e9eef2"), 0.2)
    for hz_ in (6.0, 10.0):
        beam(B, f(mast_x, -1.0 * s, hz_ * s), f(mast_x, 1.0 * s, hz_ * s), hx("#e9eef2"), 0.1)
    beam(B, f(mast_x, 0, 2.3 * s), f(-0.44 * L, 0, 2.6 * s), hx("#e9eef2"), 0.15)
    cols_a = [[WH, hx("#f4efe4")], [WH, CORAL], [hx("#fff3df"), TURQ], [WH, SUN], [CORAL, hx("#ff9a86")]][int(rng.integers(5))]
    _sail(B, f(mast_x, 0, 14.4 * s), f(mast_x, 0, 2.5 * s), f(-0.44 * L, 0, 2.7 * s), 0.9 * s, 6, cols_a)
    cols_b = [[WH, hx("#f4efe4")], [WH, TURQ], [WH, SUN]][int(rng.integers(3))]
    _sail(B, f(mast_x + 0.02 * L, 0, 13.2 * s), f(mast_x + 0.02 * L, 0, 1.7 * s), f(0.47 * L, 0, 1.45 * s), -0.8 * s, 5, cols_b)
    top = f(mast_x, 0, 15.0 * s)
    B["solid"].add(np.array([top, f(mast_x - 1.6 * s, 0, 15.3 * s), f(mast_x, 0, 14.5 * s)]), [[0, 1, 2]], [CORAL, SUN, PINK][int(rng.integers(3))])
    for sd in (-1, 1):                                                   # shrouds
        beam(B, f(mast_x, 0, 11.0 * s), f(mast_x - 0.4, sd * 1.5 * s, 1.0), hx("#dfe6ee"), 0.04)
    q = f(-0.40 * L, 0, 0.95)
    people(B["people"], [(q[0], q[1], q[2], rot + 90)], rng, scale=(0.95, 1.0))
    _wake(B, f, L, 2.0 * s, z + 0.03)
    for k in range(7):                                                   # bow foam
        c = f(0.48 * L + rng.uniform(-0.2, 0.9), rng.uniform(-1.1, 1.1) * s, 0.04)
        v, ff = sphere(c, rng.uniform(0.25, 0.5) * s, 5, 3, 0.5); B["solid"].add(v, ff, hx("#ffffff"))


def p_fishboat2(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    s = p.get("s", 1.0)
    L, Bm = 12.5 * s, 4.4 * s
    f = L2W(x, y, z, rot)
    hull = [hx("#2f7fe0"), hx("#e84a4a"), hx("#ffd23f"), hx("#3dbb6b"), hx("#19b8b0")][int(rng.integers(5))]
    _boat_hull(B, f, L, Bm, 1.05 * s, hull, WH)

    def bx(cx, cy, cz, sx, sy, sz, col, k=0.3, tp=1.0):
        v, ff = box(f(cx, cy, cz), (sx, sy, sz), rot=rot, taper=tp)
        B["solid"].add(v, ff, shade(v, ff, col, k, 0.74))
    bx(-0.12 * L, 0, 0.95, 3.6 * s, 2.7 * s, 2.3 * s, hx("#f7f2e8"))
    bx(-0.12 * L, 0, 3.2 * s, 4.0 * s, 3.0 * s, 0.35 * s, [CORAL, TURQ][int(rng.integers(2))])
    for sd in (-1, 1):
        for k in range(3):
            c = f(-0.12 * L + (k - 1) * 1.0 * s, sd * 1.36 * s, 2.0 * s)
            v, ff = box(c, (0.7 * s, 0.07, 0.8 * s), rot=rot); B["solid"].add(v, ff, hx("#2a6c86"))
    bx(-0.12 * L + 1.5 * s, 0, 2.0 * s, 0.3 * s, 2.2 * s, 1.0 * s, hx("#2a6c86"), 0.1)
    v, ff = frustum(f(-0.20 * L, 0.8 * s, 3.5 * s), 0.22 * s, 0.18 * s, 1.4 * s, 6); B["solid"].add(v, ff, shade(v, ff, hx("#c0392b"), 0.3, 0.8))
    beam(B, f(0.12 * L, 0, 0.95), f(0.12 * L, 0, 8.4 * s), hx("#e9eef2"), 0.18)
    beam(B, f(0.12 * L, 0, 7.4 * s), f(0.36 * L, 0, 2.4 * s), hx("#e9eef2"), 0.08)
    beam(B, f(0.12 * L, 0, 7.4 * s), f(-0.18 * L, 0, 3.4 * s), hx("#e9eef2"), 0.08)
    for k in range(3):                                                   # crates and nets
        bx(-0.40 * L, (k - 1) * 0.95 * s, 0.95, 0.85 * s, 0.85 * s, 0.7 * s, hx("#c79a5e"))
    c = f(0.30 * L, 0, 0.95)
    v, ff = frustum(c, 1.3 * s, 0.3 * s, 0.9 * s, 7); B["solid"].add(v, ff, shade(v, ff, hx("#7fa08f"), 0.3, 0.8))
    for k in range(6):
        cc = f(0.18 * L + 0.45 * k, ((k % 2) - 0.5) * 0.9 * s, 1.0)
        v, ff = sphere(cc, 0.28 * s, 5, 3); B["solid"].add(v, ff, [CORAL, WH, SUN][k % 3])
    c = f(0.12 * L, 0, 8.6 * s)
    v, ff = sphere(c, 0.3 * s, 5, 3); B["emit"].add(v, ff, hx("#ffd9a0") * 0.4)
    q = f(-0.12 * L + 0.2, -1.6 * s, 1.0)
    people(B["people"], [(q[0], q[1], q[2], rot + 180)], rng, scale=(0.95, 1.0))
    _wake(B, f, L, 2.3 * s, z + 0.03)


def p_ferry(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    s = p.get("s", 1.0)
    L, Bm = 46.0 * s, 11.0 * s
    f = L2W(x, y, z, rot)
    _boat_hull(B, f, L, Bm, 2.2 * s, hx("#fffaf0"), CORAL)

    def bx(cx, cy, cz, sx, sy, sz, col, k=0.3, tp=1.0, group="solid"):
        v, ff = box(f(cx, cy, cz), (sx, sy, sz), rot=rot, taper=tp)
        B[group].add(v, ff, shade(v, ff, col, k, 0.76) if group == "solid" else col)
    bx(-0.05 * L, 0, 1.5 * s, 0.62 * L, 0.78 * Bm, 3.6 * s, hx("#fffaf0"))
    bx(-0.05 * L, 0, 5.0 * s, 0.50 * L, 0.70 * Bm, 3.4 * s, hx("#fffaf0"), tp=0.94)
    bx(-0.08 * L, 0, 8.3 * s, 0.22 * L, 0.5 * Bm, 3.0 * s, hx("#f4efe4"))
    bx(-0.08 * L, 0, 11.3 * s, 0.24 * L, 0.54 * Bm, 0.3 * s, CORAL)
    for sd in (-1, 1):
        for lvl, zz in ((0, 2.4), (1, 6.0)):
            for k in range(11):
                cx = -0.05 * L + (k - 5) * 0.045 * L
                bx(cx, sd * (0.78 * Bm / 2 + 0.05), zz * s, 0.032 * L, 0.06, 1.2 * s, hx("#4a96c4"))
    v, ff = frustum(f(-0.22 * L, 0, 11.3 * s), 1.5 * s, 1.2 * s, 4.8 * s, 8); B["solid"].add(v, ff, shade(v, ff, CORAL, 0.3, 0.8))
    v, ff = frustum(f(-0.22 * L, 0, 14.0 * s), 1.52 * s, 1.52 * s, 1.2 * s, 8); B["solid"].add(v, ff, shade(v, ff, hx("#12335a"), 0.3, 0.8))
    for k in range(10):
        cc = f(0.30 * L - k * 0.05 * L, 0, 3.5 * s)
        bx(cc[0] - x if False else 0.30 * L - k * 0.05 * L, 0, 5.1 * s + (0.0 if k % 2 == 0 else 0.0), 0.6 * s, 0.6 * s, 0.9 * s, [CORAL, TURQ, SUN, PINK][k % 4])
    bx(0.0, 0, 11.6 * s, 0.1 * L, 0.22 * Bm, 0.8 * s, hx("#12335a"))
    _wake(B, f, L, 5.0 * s, z + 0.03)


def p_dinghy(B, p, rng):
    x, y, z, rot = p["x"], p["y"], p["z"], p.get("rot", 0.0)
    f = L2W(x, y, z, rot)
    col = [CORAL, TURQ, SUN, WH][int(rng.integers(4))]
    _boat_hull(B, f, 4.2, 1.6, 0.4, col, WH)
    q = f(0, 0, 0.35)
    people(B["people"], [(q[0], q[1], q[2], rot + 90)], rng, scale=(0.9, 0.95))
    for sd in (-1, 1):
        beam(B, f(-0.2, sd * 0.3, 1.0), f(0.4, sd * 1.5, 0.35), hx("#c9a36a"), 0.06)
    _wake(B, f, 4.2, 0.6, z + 0.03)


def p_farisle(B, p, rng):
    """Distant island silhouette in haze colours: layered faceted frusta, optional palm tufts and a lighthouse."""
    x, y, s, hz = p["x"], p["y"], p.get("s", 1.0), np.array(p.get("haze", [0.78, 0.86, 0.9]))
    t = float(p.get("tt", 0.6))
    tiers = int(rng.integers(2, 4))
    R = rng.uniform(110, 210) * s
    zc = -1.2
    base = [hx("#d9b98a"), hx("#5fb05a"), hx("#3f9a50"), hx("#2f8a48")]
    for k in range(tiers):
        n = int(rng.integers(7, 10))
        h = rng.uniform(7, 16) * s
        V, F, nn = _chunk((x + rng.uniform(-0.2, 0.2) * R, y + rng.uniform(-0.2, 0.2) * R, zc), R * (1 - 0.18 * k), h, n, rng.uniform(0, 1), rng.uniform(0.6, 0.85), rng, 0.22)
        col = (base[min(k, 3)] * (1 - t) + hz * t)
        B["solid"].add(V, F, shade(V, F, col, 0.3 * (1 - 0.5 * t), 0.75 + 0.2 * t))
        zc += h * 0.85
    if rng.uniform() < 0.5:
        for k in range(int(rng.integers(3, 7))):
            a = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(0, R * 0.4)
            pts = np.array([[x + math.cos(a) * r, y + math.sin(a) * r, zc], [x + math.cos(a) * r, y + math.sin(a) * r, zc + 14 * s]])
            v, f, _ = tube(pts, [0.9 * s, 0.6 * s], 5)
            B["solid"].add(v, f, hx("#7a6a58") * (1 - t) + hz * t)
            for q in range(6):
                b = 2 * math.pi * q / 6
                B["solid"].add(np.array([pts[1], pts[1] + np.array([math.cos(b) * 7 * s, math.sin(b) * 7 * s, -3 * s]), pts[1] + np.array([math.cos(b + 0.5) * 7 * s, math.sin(b + 0.5) * 7 * s, -3 * s])]), [[0, 1, 2]], hx("#3f9a50") * (1 - t) + hz * t)


def p_whitecaps(B, p, rng):
    """Wave crests on the open sea: thin elongated faces along the swell direction, size growing with distance so they stay visible towards the horizon."""
    cx, cy = p["cx"], p["cy"]
    swell = math.radians(p.get("swell", 60.0))
    u = np.array([math.cos(swell), math.sin(swell)])
    n_ = np.array([-u[1], u[0]])
    z = p["z"]
    for _ in range(int(p.get("n", 600))):
        d = math.exp(rng.uniform(math.log(p["d0"]), math.log(p["d1"])))
        a = rng.uniform(0, 2 * math.pi)
        c = np.array([cx + math.cos(a) * d, cy + math.sin(a) * d])
        L = 3.0 + d * 0.014 * rng.uniform(0.6, 1.4)
        w = 0.55 + d * 0.0030 * rng.uniform(0.6, 1.4)
        ang_j = swell + rng.uniform(-0.45, 0.45)
        u = np.array([math.cos(ang_j), math.sin(ang_j)]); n_ = np.array([-u[1], u[0]])
        t = min(d / 6000.0, 1.0)
        col = (hx("#ffffff") * (0.9 - 0.25 * t)) + hx("#9ec8e6") * (0.1 + 0.25 * t)
        pts = np.array([c - u * L / 2, c + u * L / 2 * 0.3 + n_ * w, c + u * L / 2, c + u * L / 2 * 0.3 - n_ * w])
        P = np.column_stack([pts, np.full(4, z + 0.04)])
        B["solid"].add(P, tri_quads([[0, 1, 2, 3]]), col)


PROPS.update({k[2:]: v for k, v in dict(globals()).items() if k.startswith("p_") and callable(v)})
