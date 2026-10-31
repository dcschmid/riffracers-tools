"""High-quality stylised kit (Blender side): lattice trusses, speaker/equalizer stacks, LED torches, banners, chunky banded rocks with lit tops,
low-poly clouds, 3D text, foam rings, scallop-shell stage and finish gantry. Everything is generated geometry with per-face colour (flat shaded)."""
from __future__ import annotations

import math

import bpy
import numpy as np
from mathutils import Matrix

from .core import Batch, box, frustum, hx, people, rotx, roty, rotz, sphere, strip, tri_quads, FUN
from . import city

FONT = "/usr/share/fonts/google-noto/NotoSans-Bold.ttf"
_font_cache = {}


def face_shade(V, F, base, top=None, k=0.30):
    P = np.asarray(V)[np.asarray(F)]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    nz = np.abs(n[:, 2])
    base = np.asarray(base, float)
    col = (base if base.ndim == 2 else np.tile(base, (len(F), 1))) * (1 - k + k * nz)[:, None]
    if top is not None:
        col = np.where((nz > 0.82)[:, None], np.asarray(top, float), col)
    return col


def chunk(c, R, H, n, rot, taper, rng, jitter=0.16, shift=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    r0 = R * (1 + rng.uniform(-jitter, jitter, n))
    r1 = R * taper * (1 + rng.uniform(-jitter, jitter, n))
    sh = np.array([rng.uniform(-shift, shift), rng.uniform(-shift, shift), 0.0]) * R
    bot = np.column_stack([r0 * np.cos(a), r0 * np.sin(a), np.zeros(n)])
    top = np.column_stack([r1 * np.cos(a), r1 * np.sin(a), np.full(n, H)]) + sh
    V = np.vstack([bot, top, [[sh[0], sh[1], H]]]) + np.asarray(c, float)
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [n + i, n + j, 2 * n]]
    return V, np.array(F)


def rock_stack(B, x, y, z, rng, R=8.0, H=14.0, tiers=3, pal=("#d8a77c", "#e9c29a", "#f3d7b4"), top="#6fd35a", group="solid", n=None):
    zc = z
    cx, cy = x, y
    for t in range(tiers):
        r = R * (1 - 0.26 * t) * rng.uniform(0.85, 1.12)
        h = H / tiers * rng.uniform(0.8, 1.25)
        V, F = chunk((cx, cy, zc), r, h, n or int(rng.integers(5, 8)), rng.uniform(0, 6.28), rng.uniform(0.62, 0.85), rng, jitter=0.2, shift=0.18)
        B[group].add(V, F, face_shade(V, F, hx(pal[min(t, len(pal) - 1)]) * rng.uniform(0.94, 1.06), hx(top) if t == tiers - 1 else hx(pal[min(t + 1, len(pal) - 1)]) * 1.05))
        zc += h * 0.92
        cx += rng.uniform(-0.18, 0.18) * r
        cy += rng.uniform(-0.18, 0.18) * r


def boulder_field(B, x, y, z, rng, n=5, R=3.0, pal=("#d8a77c", "#e9c29a"), top="#8fdc6a"):
    for _ in range(n):
        rock_stack(B, x + rng.uniform(-R * 2.2, R * 2.2), y + rng.uniform(-R * 2.2, R * 2.2), z - 0.4, rng, R=R * rng.uniform(0.5, 1.2), H=R * rng.uniform(0.9, 2.0), tiers=2, pal=pal, top=top)


# ---------------------------------------------------------------------------------------------------- structures
def beam(B, a, b, col, w=0.35, group="solid"):
    city._tilt_box(B, a, b, None, col, group=group, wd=w, hd=w)


def truss(B, a, b, w=1.4, col=None, seg=2.2, vertical=False):
    """Lattice girder between points a and b (square cross-section w x w): 4 chords, rungs and diagonals."""
    col = hx("#d7dde6") if col is None else col
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 0.5:
        return
    u = d / L
    ref = np.array([0, 0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0, 0])
    s = np.cross(u, ref); s /= np.linalg.norm(s)
    t = np.cross(u, s)
    corners = [(+1, +1), (+1, -1), (-1, -1), (-1, +1)]
    off = [s * cx * w / 2 + t * cy * w / 2 for cx, cy in corners]
    for o in off:
        beam(B, a + o, b + o, col * 0.92, 0.16)
    n = max(1, int(L // seg))
    for i in range(n + 1):
        p = a + u * (L * i / n)
        for k in range(4):
            beam(B, p + off[k], p + off[(k + 1) % 4], col, 0.1)
        if i < n:
            q = a + u * (L * (i + 1) / n)
            beam(B, p + off[0], q + off[1], col, 0.09)
            beam(B, p + off[2], q + off[3], col, 0.09)


def spot_lights(B, a, b, n, col=None, size=0.9, drop=0.6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    for i in range(n):
        p = a + (b - a) * (i + 0.5) / n
        v, f = box((p[0], p[1], p[2] - drop), (size, size, size * 0.8))
        B["solid"].add(v, f, hx("#2a2f3a"))
        v, f = box((p[0], p[1], p[2] - drop - 0.1), (size * 0.7, size * 0.7, 0.2))
        B["emit"].add(v, f, hx("#fff3c4") if col is None else col)


def speaker_stack(B, x, y, z, hd, w=5.0, d=4.0, h=16.0, rim="#27c4c9", cones=2, body="#232a38", led=True):
    """Stack of speaker cabinets facing local -y rotated by hd (0 = facing -y). Cones are discs, rim glows."""
    R = rotz(hd)
    seg = h / max(int(h // 3.2), 1)
    n = max(int(h // 3.2), 1)
    for k in range(n):
        zc = z + k * seg
        loc = np.array([0, 0, 0.0]) @ R.T + [x, y, zc]
        v, f = box((loc[0], loc[1], zc), (w, d, seg - 0.12), rot=hd)
        B["solid"].add(v, f, hx(body) * (1.0 + 0.06 * (k % 2)))
        for c in range(cones):
            cx = (c - (cones - 1) / 2) * (w / cones)
            p = np.array([cx, -d / 2 - 0.05, seg / 2]) @ R.T + [x, y, zc]
            vv, ff = frustum((0, 0, 0), w / cones * 0.42, w / cones * 0.2, 0.35, 10)
            vv = vv @ rotx(90).T @ R.T + p
            B["solid"].add(vv, ff, hx("#12151d"))
            vv, ff = frustum((0, 0, 0), w / cones * 0.46, w / cones * 0.46, 0.12, 10)
            vv = vv @ rotx(90).T @ R.T + p + (np.array([0, -0.02, 0]) @ R.T)
            B["emit"].add(vv, ff, hx(rim) * 0.9)
    if led:
        v, f = box((x, y, z + h), (w + 0.6, d + 0.6, 0.5), rot=hd)
        B["emit"].add(v, f, hx(rim))


def eq_stack(B, x, y, z, hd, w=7.0, h=18.0, d=3.0, bars=8, seed=1, pal=("#27c4c9", "#ffd23f", "#ff7a59")):
    r = np.random.default_rng(seed)
    R = rotz(hd)
    v, f = box((x, y, z), (w + 0.6, d + 0.4, h + 0.5), rot=hd)
    B["solid"].add(v, f, hx("#1b2130"))
    bw = w / bars
    for i in range(bars):
        lvl = r.integers(2, int(h // 1.2))
        for k in range(lvl):
            c = hx(pal[0]) if k < lvl * 0.55 else (hx(pal[1]) if k < lvl * 0.85 else hx(pal[2]))
            p = np.array([(i - (bars - 1) / 2) * bw, -d / 2 - 0.1, 0.6 + k * 1.1]) @ R.T + [x, y, z]
            v, f = box((p[0], p[1], p[2]), (bw * 0.82, 0.25, 0.85), rot=hd)
            B["emit"].add(v, f, c)


def torch(B, x, y, z, rng, h=3.6, col="#ff7a59", sc=1.0):
    h = h * sc
    v, f = frustum((x, y, z), 0.2 * sc, 0.13 * sc, h, 6); B["solid"].add(v, f, hx("#2a2f3a"))
    v, f = frustum((x, y, z + h), 0.62 * sc, 0.45 * sc, 0.55 * sc, 7); B["solid"].add(v, f, hx(col))
    for k, (r0, hh, c) in enumerate(((0.46, 1.9, "#ff9a3c"), (0.3, 1.4, "#ffd23f"))):
        V, F = frustum((x, y, z + h + 0.5 * sc), r0 * sc, 0.04, hh * sc * rng.uniform(0.9, 1.15), 6, rot=rng.uniform(0, 60))
        B["emit"].add(V, F, hx(c))


def banner(B, x, y, z, hd, rng, cols=("#ff7a59", "#27c4c9", "#ffd23f"), h=7.5, bw=3.2, bh=4.4, side=1):
    v, f = frustum((x, y, z), 0.16, 0.12, h, 5); B["solid"].add(v, f, hx("#e8edf1"))
    R = rotz(hd)
    n = len(cols)
    for i, c in enumerate(cols):
        p = np.array([0, side * 0.0, 0]) @ R.T
        loc = np.array([(i - (n - 1) / 2) * bw / n, 0.0, 0.0]) @ R.T + [x, y, z + h - bh - 0.3]
        v, f = box((loc[0], loc[1], loc[2]), (bw / n, 0.1, bh), rot=hd)
        B["solid"].add(v, f, hx(c))
    v, f = box((x, y, z + h), (bw + 0.5, 0.25, 0.25), rot=hd); B["solid"].add(v, f, hx("#e8edf1"))



def lantern(B, x, y, z, rng, h=9.0, col=None):
    """Glowing festival lantern on a pole with palm-leaf blades; returns the top point (for bunting)."""
    cols = [hx(c) for c in ("#ff7a59", "#ffd23f", "#27c4c9", "#f25c9b")]
    c = cols[int(rng.integers(len(cols)))] if col is None else hx(col)
    v, f = frustum((x, y, z), 0.3, 0.2, h, 6); B["solid"].add(v, f, hx("#f3efe6"))
    v, f = sphere((x, y, z + h + 1.0), 1.2, 7, 5); B["emit"].add(v, f, c)
    a0 = rng.uniform(0, 6.28)
    for k in range(5):
        a = a0 + k * 2 * np.pi / 5
        beam(B, (x, y, z + h + 0.2), (x + 2.8 * math.cos(a), y + 2.8 * math.sin(a), z + h - 1.0), hx("#2fb55a"), 0.9)
    return np.array([x, y, z + h])


def bunting(B, a, b, rng, flags=None, sag=3.2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    n = flags or max(int(L // 2.6), 3)
    cols = [hx(c) for c in ("#ff7a59", "#ffd23f", "#27c4c9", "#f25c9b", "#ffffff")]
    u = (b - a) / L
    for i in range(n):
        t0, t1 = i / n, (i + 0.78) / n
        def P(t):
            p = a + (b - a) * t
            p[2] -= sag * 4 * t * (1 - t)
            return p
        p0, p1 = P(t0), P(t1)
        tip = (p0 + p1) / 2 - np.array([0, 0, 1.9])
        B["solid"].add(np.array([p0, p1, tip]), [[0, 1, 2]], cols[i % len(cols)])
    pts = [a + (b - a) * (i / 12) for i in range(13)]
    for i in range(12):
        pp, qq = pts[i].copy(), pts[i + 1].copy()
        pp[2] -= sag * 4 * (i / 12) * (1 - i / 12)
        qq[2] -= sag * 4 * ((i + 1) / 12) * (1 - (i + 1) / 12)
        beam(B, pp, qq, hx("#e8edf1"), 0.12)


def eq_pylon(B, x, y, z, hd, w=9.0, h=34.0, bars=7, seed=1, d=4.0):
    """Tall equalizer pylon: rounded-feel bars of different height, glowing turquoise->sun->coral, white base."""
    r = np.random.default_rng(seed)
    R = rotz(hd)
    v, f = box((x, y, z), (w + 1.0, d + 1.0, 2.0), rot=hd); B["solid"].add(v, f, hx("#f3f6f8"))
    bw = w / bars
    for i in range(bars):
        hh = h * (0.35 + 0.65 * abs(math.sin(i * 1.3 + seed)))
        p = np.array([(i - (bars - 1) / 2) * bw, 0, 0]) @ R.T + [x, y, z + 2.0]
        v, f = box((p[0], p[1], p[2]), (bw * 0.86, d, hh), rot=hd); B["solid"].add(v, f, hx("#1b2a44"))
        nseg = int(hh // 2.0)
        for k in range(nseg):
            t = k / max(nseg - 1, 1)
            c = hx("#27c4c9") * (1 - t) + hx("#ffd23f") * t if t < 0.7 else hx("#ffd23f") * (1 - (t - 0.7) / 0.3) + hx("#ff7a59") * ((t - 0.7) / 0.3)
            q = np.array([0, -d / 2 - 0.08, 0]) @ R.T
            v, f = box((p[0] + q[0], p[1] + q[1], p[2] + 0.5 + k * 2.0), (bw * 0.74, 0.16, 1.4), rot=hd); B["emit"].add(v, f, c)


def cloud(B, x, y, z, rng, s=1.0):
    n = int(rng.integers(5, 9))
    for i in range(n):
        r = rng.uniform(9, 17) * s
        dx = (i - n / 2) * r * 0.85 + rng.uniform(-3, 3)
        v, f = sphere((x + dx, y + rng.uniform(-6, 6), z + rng.uniform(-1, 3)), r, 7, 4, 0.62)
        B["solid"].add(v, f, face_shade(v, f, hx("#ffffff"), None, 0.18) * rng.uniform(0.97, 1.0))


def text3d(B, s, loc, size, rot_z=0.0, col=None, group="emit", extrude=0.25, bold=True, rx=90.0):
    if FONT not in _font_cache:
        _font_cache[FONT] = bpy.data.fonts.load(FONT)
    cu = bpy.data.curves.new("txt", "FONT")
    cu.body = s
    cu.size = size
    cu.extrude = extrude
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.font = _font_cache[FONT]
    ob = bpy.data.objects.new("txt", cu)
    bpy.context.scene.collection.objects.link(ob)
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    me.calc_loop_triangles()
    nv = len(me.vertices)
    V = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    F = np.array([t.vertices[:] for t in me.loop_triangles], int) if len(me.loop_triangles) else np.zeros((0, 3), int)
    bpy.context.scene.collection.objects.unlink(ob)
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    bpy.data.meshes.remove(me)
    V = V @ rotx(rx).T @ rotz(rot_z).T + np.asarray(loc, float)
    B[group].add(V, F, hx("#ffffff") if col is None else col)


def foam_strip(B, L, Rr, z, rng):
    L, Rr = np.asarray(L, float), np.asarray(Rr, float)
    n = len(L)
    Lz = np.column_stack([L, np.full(n, z)])
    Rz = np.column_stack([Rr, np.full(n, z)])
    V, F = strip(Lz, Rz)
    B["solid"].add(V, F, hx("#f4fbff"))


def checker_panel(B, x, y, z, w, h, hd, cell=1.4):
    R = rotz(hd)
    nx, nz = int(w // cell), int(h // cell)
    for i in range(nx):
        for k in range(nz):
            col = hx("#fbfbfd") if (i + k) % 2 == 0 else hx("#14161d")
            p = np.array([(i - (nx - 1) / 2) * cell, 0, 0]) @ R.T + [x, y, z + k * cell]
            v, f = box((p[0], p[1], p[2]), (cell * 0.5 + 0.55, 0.3, cell), rot=hd + 90)
            B["solid"].add(v, f, col)


def led_wall(B, x, y, z, w, h, hd, rng, cols=("#27c4c9", "#ffd23f", "#ff7a59", "#f25c9b"), cols_n=30, rows=10, title=None, tcol="#fff6dc"):
    """LED screen facing local -y (y = front face of the frame): dark frame, gradient background, equalizer bars, optional 3D title."""
    R = rotz(hd)
    v, f = box((x, y + 0.35, z - 1.1), (w + 2.6, 0.7, h + 2.6), rot=hd)
    B["solid"].add(v, f, hx("#0f121a"))
    cw, ch = w / cols_n, h / rows
    for i in range(cols_n):
        t = i / (cols_n - 1)
        lvl = int(rows * (0.2 + 0.5 * abs(math.sin(i * 0.65 + rng.uniform(0, 0.5)))))
        for k in range(rows):
            bg = (hx("#0d4ea0") * (1 - t) + hx("#6a2394") * t) * (0.55 + 0.35 * (k / rows))
            on = k < lvl
            c = hx(cols[0]) if k < rows * 0.45 else (hx(cols[1]) if k < rows * 0.75 else hx(cols[2]))
            p = np.array([(i - (cols_n - 1) / 2) * cw, -0.14, 0]) @ R.T + [x, y, z + k * ch]
            v, f = box((p[0], p[1], p[2]), (cw * 0.92, 0.22, ch * 0.9), rot=hd)
            B["emit"].add(v, f, c * 1.15 if on else bg)
    if title:
        p = np.array([0, -0.55, h * 0.70]) @ R.T + [x, y, z]
        text3d(B, title, p, h * 0.26, rot_z=hd, col=hx(tcol), extrude=0.7)


# ---------------------------------------------------------------------------------------------------- hero: scallop shell stage
def shell_stage(B, lm, z, rng, title="COASTAL SPRINT"):
    """Scallop-shell festival stage facing -y. Footprint centre lm['xy'], width W, depth D."""
    x, y = lm["xy"]
    W, D = lm["footprint"]
    CORAL, TURQ, SUN, WH, PK = (hx(c) for c in ("#ff7a59", "#27c4c9", "#ffd23f", "#fbfbfd", "#f25c9b"))
    deck = 2.6
    yb = y + D / 2 - 5.0                 # hinge y
    z0 = z + deck
    # podium + steps + lip
    v, f = box((x, y, z), (W, D, deck)); B["solid"].add(v, f, hx("#f3f6f8"))
    v, f = box((x, y - D / 2 + 0.2, z + deck - 0.6), (W - 2, 0.7, 0.8)); B["emit"].add(v, f, TURQ)
    for k in range(5):
        v, f = box((x, y - D / 2 - 1.2 - k * 1.1, z + deck - 0.55 * (k + 1)), (W * 0.42, 1.2, 0.55)); B["solid"].add(v, f, hx("#e4eaee"))
    # shell fan: radial sectors
    Rr = 60.0
    NS, NR = 21, 10
    a0, a1 = math.radians(6), math.radians(174)
    pal = [CORAL, WH, TURQ, SUN, WH]
    def pt(al, rho):
        lean = 20.0 * (rho / Rr) ** 1.7
        return np.array([x + rho * math.cos(al) * 1.0, yb - lean, z0 + rho * math.sin(al) * 0.74 + 0.0])
    for i in range(NS):
        al0 = a0 + (a1 - a0) * i / NS
        al1 = a0 + (a1 - a0) * (i + 1) / NS
        rows = []
        for j in range(NR + 1):
            rho = 16.0 + (Rr - 16.0) * j / NR
            rows.append((pt(al0, rho), pt(al1, rho)))
        base = pal[i % len(pal)]
        for j in range(NR):
            q = np.array([rows[j][0], rows[j][1], rows[j + 1][1], rows[j + 1][0]])
            shade = 0.72 + 0.28 * (j / NR)
            B["solid"].add(q, tri_quads([[0, 1, 2, 3]]), base * shade)
        # raised rib along the sector boundary
        pts = [pt(al0, 16.0 + (Rr - 16.0) * j / NR) for j in range(NR + 1)]
        for j in range(NR):
            beam(B, pts[j] + [0, -0.15, 0], pts[j + 1] + [0, -0.15, 0], hx("#fff8ea"), 0.55)
    # glowing rim
    rim = []
    for s in range(60):
        al = a0 + (a1 - a0) * s / 59
        rim.append(pt(al, Rr))
    for s in range(59):
        beam(B, rim[s], rim[s + 1], TURQ, 1.3, group="emit")
        beam(B, rim[s] + [0, 0, -1.4], rim[s + 1] + [0, 0, -1.4], hx("#ffffff"), 0.7)
    # back wall under the fan + side towers
    v, f = box((x, yb + 1.5, z), (W * 0.9, 3.0, deck + 14), taper=1.0); B["solid"].add(v, f, hx("#1b2130"))
    # LED screen
    led_wall(B, x, yb - 10.0, z0 + 5.0, 50.0, 19.0, 0.0, rng, title=title)
    # front rig: two lattice towers + cross truss + lights
    tw = W / 2 - 3.0
    ht = 30.0
    for sx in (-1, 1):
        truss(B, (x + sx * tw, y - 12.0, z0), (x + sx * tw, y - 12.0, z0 + ht), w=1.6)
    a, b = np.array([x - tw, y - 12.0, z0 + ht]), np.array([x + tw, y - 12.0, z0 + ht])
    truss(B, a, b, w=1.8, seg=2.4)
    spot_lights(B, a + [0, -1.4, 0], b + [0, -1.4, 0], 18)
    a2, b2 = a + [0, 9.0, -4.5], b + [0, 9.0, -4.5]
    truss(B, a2, b2, w=1.4, seg=2.4)
    spot_lights(B, a2 + [0, 0, -0.2], b2 + [0, 0, -0.2], 14, col=hx("#aef5ff"))
    for sx in (-1, 1):
        truss(B, (x + sx * tw, y - 12.0, z0 + ht), (x + sx * tw, yb - 6, z0 + ht - 4.5), w=1.2)
    # speaker + equalizer stacks flanking the stage
    for sx in (-1, 1):
        eq_pylon(B, x + sx * (tw + 10.0), y - 8.0, z0 - deck, 0.0, w=10.0, h=40.0, seed=7 + (1 if sx > 0 else 3))
        lantern(B, x + sx * (tw - 2.0), y - 18.0, z0, rng, h=7.0)
    # banner poles at the front corners
    tops = [lantern(B, x + sx * (W / 2 + 2.0), y - D / 2 - 3.0, z, rng, h=15.0) for sx in (-1, 1)]
    bunting(B, tops[0], tops[1], rng, sag=4.5)
    # floor LED strips
    for k in range(9):
        v, f = box((x - W / 2 + 8 + k * (W - 16) / 8, y - D / 2 + 7, z0 + 0.02), (2.4, 0.4, 0.06)); B["emit"].add(v, f, [TURQ, CORAL, SUN][k % 3])


def finish_gantry(B, x, y, z, w, rng, title="COASTAL SPRINT"):
    """Finish gantry across a road running along +x at position x. Faces -x (approaching karts)."""
    CORAL, TURQ, SUN = (hx(c) for c in ("#ff7a59", "#27c4c9", "#ffd23f"))
    half = w / 2 + 3.2
    for sy in (-1, 1):
        eq_pylon(B, x, y + sy * half, z, -90.0, w=5.4, h=13.4, bars=4, seed=3 + sy, d=3.2)
        lantern(B, x, y + sy * (half + 4.4), z, rng, h=6.0)
    a, b = np.array([x, y - half, z + 13.4]), np.array([x, y + half, z + 13.4])
    truss(B, a, b, w=2.0, seg=2.4, col=hx("#f0f3f7"))
    spot_lights(B, a + [-1.4, 0, 0], b + [-1.4, 0, 0], 10)
    # checkered banner + coral frame
    checker_panel(B, x - 0.2, y, z + 9.0, w + 2.0, 4.2, 90.0)
    v, f = box((x - 0.35, y, z + 8.7), (0.4, w + 3.0, 0.5)); B["solid"].add(v, f, CORAL)
    v, f = box((x - 0.35, y, z + 13.0 - 0.5), (0.4, w + 3.0, 0.5)); B["solid"].add(v, f, CORAL)
    # sign above
    v, f = box((x - 0.2, y, z + 16.2), (0.8, 24.0, 4.6)); B["solid"].add(v, f, hx("#141822"))
    text3d(B, title, (x - 0.75, y, z + 16.2), 3.4, rot_z=-90.0, col=hx("#ff9a3c"), extrude=0.5)
    # light bar
    for k in range(5):
        v, f = frustum((0, 0, 0), 0.8, 0.8, 0.4, 12); v = v @ roty(90).T + [x - 0.9, y - 6 + k * 3.0, z + 19.3]
        B["emit"].add(v, f, [CORAL, SUN, TURQ, SUN, CORAL][k])
