"""Blender-side kit (runs inside `blender -b -P`): numpy primitives -> flat-shaded, vertex-coloured meshes, materials, roads, people, karts,
stands, pit lane, lighting and rendering. Placement decisions come from the pipeline (layout.json, props.json); this file only gives shape."""
from __future__ import annotations

import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

RNG = np.random.default_rng(2024)


def hx(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# ------------------------------------------------------------------------------------------------ geometry batches
class Batch:
    """Accumulates unshared (flat-shaded) triangles with per-face colour."""

    def __init__(self):
        self.P, self.C = [], []

    def add(self, V, F, col):
        V = np.asarray(V, float)
        F = np.asarray(F, int)
        if len(F) == 0:
            return
        col = np.asarray(col, float)
        if col.ndim == 1:
            col = np.tile(col, (len(F), 1))
        self.P.append(V[F].reshape(-1, 3))
        self.C.append(np.repeat(col, 3, axis=0))

    def extend(self, other):
        self.P += other.P
        self.C += other.C

    def count(self):
        return sum(len(p) for p in self.P) // 3

    def build(self, name, mat):
        if not self.P:
            return None
        P = np.vstack(self.P).astype(np.float32)
        C = np.vstack(self.C)
        n = len(P)
        me = bpy.data.meshes.new(name)
        me.vertices.add(n)
        me.vertices.foreach_set("co", P.ravel())
        me.loops.add(n)
        me.loops.foreach_set("vertex_index", np.arange(n, dtype=np.int32))
        m = n // 3
        me.polygons.add(m)
        me.polygons.foreach_set("loop_start", np.arange(0, n, 3, dtype=np.int32))
        try:
            me.polygons.foreach_set("loop_total", np.full(m, 3, dtype=np.int32))
        except Exception:
            pass
        me.update(calc_edges=True)
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        rgba = np.column_stack([lin(C), np.ones(n)]).astype(np.float32)
        ca.data.foreach_set("color", rgba.ravel())
        try:
            me.shade_flat()
        except Exception:
            me.polygons.foreach_set("use_smooth", np.zeros(m, bool))
        me.materials.append(mat)
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        return ob


def tri_quads(Q):
    Q = np.asarray(Q, int)
    return np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])


def rotz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def rotx(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def roty(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


BOX_V = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0], [-.5, -.5, 1], [.5, -.5, 1], [.5, .5, 1], [-.5, .5, 1]], float)
BOX_F = tri_quads([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])


def box(c, size, rot=0.0, taper=1.0, bottom=True):
    """Box centred in xy at c (x, y, z_bottom), size (sx, sy, sz); `taper` scales the top face (<1 = pyramid-frustum)."""
    V = BOX_V.copy()
    V[4:, :2] *= taper
    V *= np.asarray(size, float)
    V = V @ rotz(rot).T + np.asarray(c, float)
    F = BOX_F if bottom else BOX_F[2:] if False else BOX_F
    return V, F


def frustum(c, r0, r1, h, n=8, rot=0.0, cap_top=True, cap_bottom=True):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + math.radians(rot)
    V = np.vstack([np.column_stack([r0 * np.cos(a), r0 * np.sin(a), np.zeros(n)]), np.column_stack([r1 * np.cos(a), r1 * np.sin(a), np.full(n, h)]),
                   [[0, 0, 0], [0, 0, h]]]) + np.asarray(c, float)
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]
        if cap_top:
            F.append([n + i, n + j, 2 * n + 1])
        if cap_bottom:
            F.append([j, i, 2 * n])
    return V, np.array(F)


def sphere(c, r, n=7, m=5, squash=1.0):
    V = [[0, 0, -r * squash]]
    for k in range(1, m):
        th = math.pi * k / m
        for i in range(n):
            ph = 2 * math.pi * i / n
            V.append([r * math.sin(th) * math.cos(ph), r * math.sin(th) * math.sin(ph), -r * math.cos(th) * squash])
    V.append([0, 0, r * squash])
    V = np.array(V) + np.asarray(c, float)
    F = []
    for i in range(n):
        F.append([0, 1 + (i + 1) % n, 1 + i])
    for k in range(m - 2):
        for i in range(n):
            a, b = 1 + k * n + i, 1 + k * n + (i + 1) % n
            F += [[a, b, b + n], [a, b + n, a + n]]
    top = len(V) - 1
    for i in range(n):
        F.append([top, 1 + (m - 2) * n + i, 1 + (m - 2) * n + (i + 1) % n])
    return V, np.array(F)


def strip(L, R):
    """Quads between two polylines L, R (n,3) -> V, F (triangles)."""
    n = len(L)
    V = np.vstack([L, R])
    Q = [[i, i + 1, n + i + 1, n + i] for i in range(n - 1)]
    return V, tri_quads(Q)


def prism_poly(poly, z0, z1):
    """Convex polygon (k,2) extruded from z0 to z1 (fan triangulation)."""
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


def xf(V, rot=0.0, s=1.0, t=(0, 0, 0)):
    V = np.asarray(V, float) * s
    return V @ rotz(rot).T + np.asarray(t, float)


# ------------------------------------------------------------------------------------------------ materials
def _principled(nt):
    return next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")


def _set(node, name, val):
    for k in (name,):
        if k in node.inputs:
            node.inputs[k].default_value = val
            return True
    return False


def mat_vcol(name, rough=0.6, spec=0.35, metallic=0.0, emit_strength=0.0, glass=False, alpha=1.0, sss=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Col"
    if emit_strength > 0:
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = emit_strength
        nt.links.new(vc.outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        return m
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(vc.outputs["Color"], bs.inputs["Base Color"])
    _set(bs, "Roughness", rough)
    _set(bs, "Metallic", metallic)
    for k in ("Specular IOR Level", "Specular"):
        if _set(bs, k, spec):
            break
    if glass:
        _set(bs, "Transmission Weight", 0.92)
        _set(bs, "Roughness", 0.06)
        _set(bs, "IOR", 1.15)
    if alpha < 1.0:
        _set(bs, "Alpha", alpha)
    nt.links.new(bs.outputs[0], out.inputs["Surface"])
    return m


# ------------------------------------------------------------------------------------------------ people
SKIN = ["#f3c9a5", "#e3ad84", "#c98f66", "#a86f4c", "#7d4e34", "#5a3825", "#f6d6b8"]
HAIR = ["#2a2018", "#4a3020", "#7a5230", "#c9a45c", "#e8d9a8", "#1c1c24", "#8a3a2a", "#d8502a", "#3a6ad0", "#c04a9a"]
FUN = ["#ff7a59", "#27c4c9", "#ffd23f", "#f25c9b", "#7c5cd6", "#34b36b", "#ffffff", "#2d6fdc", "#ff9f43", "#0e5a6e", "#e0453a", "#b8e86a"]


def _person_proto(pose, style=0):
    """Local-space person (feet at z=0, facing +y): cid 0 shirt, 1 pants, 2 skin, 3 hair, 4 accent/cap, 5 shoes, 6 dark (sunglasses)."""
    V, F, cid = [], [], []

    def add(v, f, c):
        off = sum(len(x) for x in V)
        V.append(v)
        F.append(np.asarray(f) + off)
        cid.append(np.full(len(f), c))
    for sx in (-0.1, 0.1):
        v, f = box((sx, 0.03, 0.0), (0.17, 0.3, 0.08)); add(v, f, 5)
        v, f = box((sx, 0, 0.08), (0.16, 0.18, 0.74)); add(v, f, 1)
    v, f = box((0, 0, 0.8), (0.4, 0.24, 0.62), taper=1.18); add(v, f, 0)
    v, f = box((0, 0, 1.4), (0.12, 0.12, 0.1)); add(v, f, 2)
    hz = 1.52
    v, f = sphere((0, 0, hz + 0.13), 0.125, 7, 5); add(v, f, 2)
    if style == 0:
        v, f = sphere((0, -0.012, hz + 0.19), 0.133, 7, 4, 0.7); add(v, f, 3)
    elif style == 1:
        v, f = sphere((0, -0.012, hz + 0.19), 0.133, 7, 4, 0.6); add(v, f, 3)
        v, f = frustum((0, 0, hz + 0.24), 0.15, 0.13, 0.1, 7); add(v, f, 4)
        v, f = box((0, 0.17, hz + 0.25), (0.2, 0.16, 0.03)); add(v, f, 4)
    elif style == 2:
        v, f = sphere((0, -0.012, hz + 0.19), 0.135, 7, 4, 0.7); add(v, f, 3)
        v, f = box((0, -0.12, hz - 0.2), (0.24, 0.07, 0.42)); add(v, f, 3)
    else:
        v, f = sphere((0, -0.01, hz + 0.24), 0.17, 7, 5); add(v, f, 3)
    if style in (1, 3):
        v, f = box((0, 0.118, hz + 0.15), (0.22, 0.04, 0.05)); add(v, f, 6)
    arms = dict(idle=(8, 8), up=(168, 168), wave=(168, 20), flag=(150, 16), point=(95, 12), jump=(150, 150), hat=(10, 160))[pose if pose != "cheer" else "up"]
    for sx, ang in ((-1, arms[0]), (1, arms[1])):
        v, f = box((0, 0, -0.52), (0.11, 0.11, 0.56))
        v = v @ rotx(-ang).T + np.array([sx * 0.25, 0, 1.38])
        add(v, f, 0 if ang < 120 else 2)
        hv, hf = sphere((0, 0, -0.55), 0.06, 5, 3)
        hv = hv @ rotx(-ang).T + np.array([sx * 0.25, 0, 1.38])
        add(hv, hf, 2)
    return np.vstack(V), np.vstack(F), np.concatenate(cid)


PEOPLE_POSES = ["idle", "idle", "up", "up", "wave", "cheer", "hat", "point", "jump"]
_PROTO = {}
PERSON_PROTO = None        # optional replacement; cid 0..6 base, 7 whites, 8 mouth, 9 cheeks (v3)
PERSON_STYLES = 4


def people(batch: Batch, spots, rng=None, palette=None, scale=(0.9, 1.08)):
    """spots: list of (x, y, z, heading_deg); the figure faces +y at heading 90. Colours from the world palette."""
    rng = rng or RNG
    pal = [hx(c) for c in (palette or FUN)]
    skin = [hx(c) for c in SKIN]
    hair = [hx(c) for c in HAIR]
    dark = hx("#161a24")
    for (x, y, z, hd) in spots:
        pose = PEOPLE_POSES[rng.integers(len(PEOPLE_POSES))]
        style = int(rng.integers(PERSON_STYLES))
        key = (pose, style, id(PERSON_PROTO))
        if key not in _PROTO:
            _PROTO[key] = (PERSON_PROTO or _person_proto)(pose, style)
        V, F, cid = _PROTO[key]
        s = rng.uniform(*scale)
        jump = 0.28 if pose == "jump" else 0.0
        P = V * s @ rotz(hd - 90).T + np.array([x, y, z + jump])
        cols = np.array([pal[rng.integers(len(pal))], pal[rng.integers(len(pal))] * 0.55, skin[rng.integers(len(skin))], hair[rng.integers(len(hair))], pal[rng.integers(len(pal))], hx("#2a2f3a"), dark])
        if cid.max() > 6:
            cols = np.vstack([cols, hx("#fff9eb"), cols[2] * 0.30 + hx("#8f3a49") * 0.40,
                              cols[2] * 0.78 + hx("#ee8b80") * 0.22])
        batch.add(P, F, cols[cid] * rng.uniform(0.9, 1.08))


# ------------------------------------------------------------------------------------------------ karts
def kart(batch: Batch, x, y, z, hd, body, accent, helmet, scale=1.0, emit: Batch | None = None):
    """Stylised kart proxy (own geometry, NOT the game's kart): wedge body, nose with bumper and lights, side pods, wheels with hubs, spoiler, exhaust, driver."""
    b, a, h = hx(body), hx(accent), hx(helmet)
    dark = hx("#23262e")
    R = rotz(hd - 90)
    T = np.array([x, y, z])

    def put(v, f, c):
        batch.add((v * scale) @ R.T + T, f, c)
    v, f = box((0, -0.1, 0.18), (1.25, 2.3, 0.35), taper=0.8); put(v, f, b)
    v, f = box((0, 1.15, 0.2), (0.62, 0.9, 0.28), taper=0.55); put(v, f, a)
    v, f = box((0, 1.62, 0.14), (0.95, 0.14, 0.2)); put(v, f, hx("#f4f6f8"))
    for sx in (-0.26, 0.26):
        v, f = box((sx, 1.57, 0.3), (0.2, 0.1, 0.12)); put(v, f, hx("#fff2b0"))
    for sx in (-0.8, 0.8):
        v, f = box((sx, -0.1, 0.15), (0.26, 1.15, 0.3), taper=0.85); put(v, f, a)
    v, f = box((0, -1.25, 0.34), (1.5, 0.35, 0.12)); put(v, f, a)
    v, f = box((-0.65, -1.25, 0.34), (0.1, 0.35, 0.55)); put(v, f, a)
    v, f = box((0.65, -1.25, 0.34), (0.1, 0.35, 0.55)); put(v, f, a)
    for sx in (-0.2, 0.2):
        v, f = frustum((0, 0, 0), 0.08, 0.07, 0.42, 6)
        v = v @ rotx(90).T + np.array([sx, -1.05, 0.3]); put(v, f, hx("#8a93a3"))
    for sx, sy, rr in ((-0.72, 0.9, 0.3), (0.72, 0.9, 0.3), (-0.78, -0.85, 0.36), (0.78, -0.85, 0.36)):
        v, f = frustum((0, 0, 0), rr, rr, 0.28, 9)
        v = v @ roty(90).T + np.array([sx, sy, rr])
        put(v, f, dark)
        v2, f2 = frustum((0, 0, 0), rr * 0.5, rr * 0.5, 0.3, 6)
        v2 = v2 @ roty(90).T + np.array([sx + (0.015 if sx > 0 else -0.015), sy, rr])
        put(v2, f2, a)
    v, f = box((0, -0.3, 0.45), (0.5, 0.45, 0.5), taper=0.8); put(v, f, hx("#f2f2f2") * 0.6 + b * 0.4)
    v, f = box((0, 0.2, 0.52), (0.36, 0.06, 0.06)); put(v, f, dark)
    for sx in (-0.2, 0.2):
        v, f = box((sx, -0.05, 0.62), (0.1, 0.5, 0.1)); put(v, f, b)
    v, f = box((0, 0.98, 0.52), (0.32, 0.04, 0.2)); put(v, f, hx("#ffffff"))
    v, f = sphere((0, -0.3, 1.12), 0.24, 7, 5); put(v, f, h)
    v, f = sphere((0, -0.2, 1.13), 0.15, 6, 4, 0.8); put(v, f, hx("#1d2330"))
    v, f = box((0, -0.12, 1.2), (0.3, 0.1, 0.08)); put(v, f, hx("#1d2330"))


# ------------------------------------------------------------------------------------------------ world / lights / render
def set_world(sky="day", sun_elev=38.0, sun_rot=130.0, strength=1.0, night_color="#0b1230", haze=0.0, exposure=0.0, cam_strength=None, aerosol=None, air=None, dust=None):
    sc = bpy.context.scene
    w = bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if sky == "gradient":
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        cr = ramp.color_ramp
        cr.elements[0].position, cr.elements[0].color = 0.42, (*lin(hx("#bfe3fa")), 1)
        cr.elements[1].position, cr.elements[1].color = 0.86, (*lin(hx("#1d78e0")), 1)
        e = cr.elements.new(0.60)
        e.color = (*lin(hx("#74bdf2")), 1)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bgc = nt.nodes.new("ShaderNodeBackground")
        bgc.inputs["Strength"].default_value = 1.15          # what the camera sees: bright, saturated sky
        nt.links.new(ramp.outputs["Color"], bgc.inputs["Color"])
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs[0], mix.inputs[1])
        nt.links.new(bgc.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    elif sky == "dusk":
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        cr = ramp.color_ramp                              # Fac = sin(elevation): 0 at the horizon, 1 at the zenith
        cr.elements[0].position, cr.elements[0].color = 0.0, (*lin(hx("#ffb35a")), 1)
        cr.elements[1].position, cr.elements[1].color = 1.0, (*lin(hx("#120a30")), 1)
        for pos_, col_ in ((0.05, "#ff5fa8"), (0.14, "#b03fd6"), (0.32, "#5a2aa0"), (0.62, "#2a1560")):
            e = cr.elements.new(pos_)
            e.color = (*lin(hx(col_)), 1)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bgc = nt.nodes.new("ShaderNodeBackground")
        bgc.inputs["Strength"].default_value = 1.0 if cam_strength is None else cam_strength
        nt.links.new(ramp.outputs["Color"], bgc.inputs["Color"])
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs[0], mix.inputs[1])
        nt.links.new(bgc.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    elif sky == "golden":
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        cr = ramp.color_ramp                              # Fac = sin(elevation): warm haze at the horizon, saturated blue towards the zenith
        cr.elements[0].position, cr.elements[0].color = 0.0, (*lin(hx("#ffd9a6")), 1)
        cr.elements[1].position, cr.elements[1].color = 1.0, (*lin(hx("#1f6fd0")), 1)
        for pos_, col_ in ((0.05, "#ffe4c2"), (0.13, "#bfe4f7"), (0.30, "#72bdf0"), (0.60, "#3c8fe0")):
            e = cr.elements.new(pos_)
            e.color = (*lin(hx(col_)), 1)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bgc = nt.nodes.new("ShaderNodeBackground")
        bgc.inputs["Strength"].default_value = 1.0 if cam_strength is None else cam_strength
        nt.links.new(ramp.outputs["Color"], bgc.inputs["Color"])
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs[0], mix.inputs[1])
        nt.links.new(bgc.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    elif sky == "midnight":
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        cr = ramp.color_ramp                              # Fac = sin(elevation): city glow at the horizon, deep navy at the zenith
        cr.elements[0].position, cr.elements[0].color = 0.0, (*lin(hx("#ff9a4a")), 1)
        cr.elements[1].position, cr.elements[1].color = 1.0, (*lin(hx("#050a22")), 1)
        for pos_, col_ in ((0.035, "#d83a8a"), (0.11, "#6a2ab0"), (0.26, "#232a74"), (0.55, "#101a52")):
            e = cr.elements.new(pos_)
            e.color = (*lin(hx(col_)), 1)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bgc = nt.nodes.new("ShaderNodeBackground")
        bgc.inputs["Strength"].default_value = 1.0 if cam_strength is None else cam_strength
        nt.links.new(ramp.outputs["Color"], bgc.inputs["Color"])
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs[0], mix.inputs[1])
        nt.links.new(bgc.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    elif sky == "synth":
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        cr = ramp.color_ramp                              # Fac = sin(elevation): 0 at the horizon, 1 at the zenith (the first version used 0.44..0.97 and never showed a gradient)
        cr.elements[0].position, cr.elements[0].color = 0.0, (*lin(hx("#ff4fb0")), 1)
        cr.elements[1].position, cr.elements[1].color = 1.0, (*lin(hx("#090528")), 1)
        for pos_, col_ in ((0.06, "#b02fd0"), (0.2, "#4a1c9a"), (0.5, "#1c0f58")):
            e = cr.elements.new(pos_)
            e.color = (*lin(hx(col_)), 1)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        bgc = nt.nodes.new("ShaderNodeBackground")
        bgc.inputs["Strength"].default_value = 1.0
        nt.links.new(ramp.outputs["Color"], bgc.inputs["Color"])
        lp = nt.nodes.new("ShaderNodeLightPath")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
        nt.links.new(bg.outputs[0], mix.inputs[1])
        nt.links.new(bgc.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])
    elif sky == "day":
        try:
            s = nt.nodes.new("ShaderNodeTexSky")
            items = [i.identifier for i in s.bl_rna.properties["sky_type"].enum_items]
            s.sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in items else ("NISHITA" if "NISHITA" in items else items[0])
            s.sun_elevation = math.radians(sun_elev)
            s.sun_rotation = math.radians(sun_rot)
            s.sun_size = math.radians(0.6)
            try:
                s.sun_disc = False
            except Exception:
                pass
            s.air_density = 0.75 if air is None else air
            s.aerosol_density = 0.5 if aerosol is None else aerosol
            if dust is not None:
                try:
                    s.dust_density = dust
                except Exception:
                    pass
            bg = nt.nodes.new("ShaderNodeBackground")
            bg.inputs["Strength"].default_value = strength
            nt.links.new(s.outputs[0], bg.inputs["Color"])
            nt.links.new(bg.outputs[0], out.inputs["Surface"])
        except Exception as e:
            print("sky fallback", e)
            bg = nt.nodes.new("ShaderNodeBackground")
            bg.inputs["Color"].default_value = (0.55, 0.72, 0.95, 1)
            bg.inputs["Strength"].default_value = strength
            nt.links.new(bg.outputs[0], out.inputs["Surface"])
    else:
        grad = nt.nodes.new("ShaderNodeTexGradient")
        coord = nt.nodes.new("ShaderNodeTexCoord")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        nt.links.new(coord.outputs["Generated"], sep.inputs[0])
        nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
        c0, c1, c2 = hx(night_color), hx(night_color) * 0.5 + np.array([0.02, 0.0, 0.05]), hx(night_color) * 1.8 + np.array([0.02, 0.01, 0.06])
        ramp.color_ramp.elements[0].position = 0.35
        ramp.color_ramp.elements[0].color = (*c0, 1)
        ramp.color_ramp.elements[1].position = 0.62
        ramp.color_ramp.elements[1].color = (*lin(c2), 1)
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.inputs["Strength"].default_value = strength
        nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
        if cam_strength is None:
            nt.links.new(bg.outputs[0], out.inputs["Surface"])
        else:                                            # lighting from `strength`, what the camera sees from `cam_strength`
            bgc = nt.nodes.new("ShaderNodeBackground")
            bgc.inputs["Strength"].default_value = cam_strength
            nt.links.new(ramp.outputs["Color"], bgc.inputs["Color"])
            lp = nt.nodes.new("ShaderNodeLightPath")
            mix = nt.nodes.new("ShaderNodeMixShader")
            nt.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
            nt.links.new(bg.outputs[0], mix.inputs[1])
            nt.links.new(bgc.outputs[0], mix.inputs[2])
            nt.links.new(mix.outputs[0], out.inputs["Surface"])
    if haze > 0:
        try:
            vs = nt.nodes.new("ShaderNodeVolumeScatter")
            vs.inputs["Density"].default_value = haze
            vs.inputs["Color"].default_value = (0.85, 0.92, 1.0, 1) if sky == "day" else (0.4, 0.45, 0.8, 1)
            nt.links.new(vs.outputs[0], out.inputs["Volume"])
        except Exception as e:
            print("haze skipped", e)
    sc.view_settings.exposure = exposure


def add_sun(elev=38.0, rot=130.0, energy=4.0, color=(1.0, 0.95, 0.88), angle=0.8):
    ld = bpy.data.lights.new("Sun", "SUN")
    ld.energy = energy
    ld.color = color
    ld.angle = math.radians(angle)
    ob = bpy.data.objects.new("Sun", ld)
    ob.rotation_euler = (math.radians(90 - elev), 0, math.radians(rot + 180))
    bpy.context.scene.collection.objects.link(ob)
    return ob


def add_area(loc, size, energy, color=(1, 1, 1), look_at=None):
    ld = bpy.data.lights.new("Area", "AREA")
    ld.energy = energy
    ld.size = size
    ld.color = color
    ob = bpy.data.objects.new("Area", ld)
    ob.location = loc
    if look_at is not None:
        d = Vector(look_at) - Vector(loc)
        ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(ob)
    return ob


def setup_render(samples=96, res=(1920, 1080), device_gpu=True, denoise=True, view="AgX", look=None):
    sc = bpy.context.scene
    try:
        sc.render.engine = "CYCLES"
    except TypeError as e:
        print("engine", e)
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = denoise
    cy.max_bounces = 6
    cy.diffuse_bounces = 3
    cy.glossy_bounces = 3
    cy.transmission_bounces = 6
    try:
        pref = bpy.context.preferences.addons["cycles"].preferences
        for t in ("OPTIX", "CUDA"):
            try:
                pref.compute_device_type = t
                pref.get_devices()
                if any(d.type == t for d in pref.devices):
                    for d in pref.devices:
                        d.use = d.type == t
                    break
            except Exception:
                continue
        cy.device = "GPU" if device_gpu else "CPU"
    except Exception as e:
        print("gpu setup", e)
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    try:
        items = [i.identifier for i in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items]
        sc.view_settings.view_transform = view if view in items else ("AgX" if "AgX" in items else items[0])
        if look:
            try:
                lk = [i.identifier for i in bpy.types.ColorManagedViewSettings.bl_rna.properties["look"].enum_items]
                match = next((i for i in lk if look.lower() in i.lower()), None)
                if match:
                    sc.view_settings.look = match
                else:
                    print("look not found", look, lk[:12])
            except Exception as e:
                print("look", e)
    except Exception as e:
        print("look", e)


def camera(loc, target, lens=35.0, ortho=None, name="Cam", dof=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_end = 60000
    cd.clip_start = 0.5
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
    ob = bpy.data.objects.new(name, cd)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    if dof:
        cd.dof.use_dof = True
        cd.dof.focus_distance = (Vector(target) - Vector(loc)).length
        cd.dof.aperture_fstop = dof
    bpy.context.scene.collection.objects.link(ob)
    return ob


def render(cam, path, samples=None):
    sc = bpy.context.scene
    sc.camera = cam
    if samples:
        sc.cycles.samples = samples
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path
