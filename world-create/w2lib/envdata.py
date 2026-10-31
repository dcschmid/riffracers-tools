"""Environment data for the Blender build, computed with shapely/numpy so that every placement is checked against the road geometry:
terrain height field + water mesh, barrier masks per road edge, prop scatter (positions, never meshes)."""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import geom as G


def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def vnoise(x, y, scale, seed, octaves=3):
    """Bilinear value noise in [-1, 1], fBm."""
    rng = np.random.default_rng(seed)
    out, amp, tot = 0.0, 1.0, 0.0
    for o in range(octaves):
        s = scale / (2 ** o)
        n = 64
        grid = rng.uniform(-1, 1, (n, n))
        u, v = x / s, y / s
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        fu, fv = smoothstep(u - i0), smoothstep(v - j0)
        g = lambda a, b: grid[a % n, b % n]
        val = (g(i0, j0) * (1 - fu) + g(i0 + 1, j0) * fu) * (1 - fv) + (g(i0, j0 + 1) * (1 - fu) + g(i0 + 1, j0 + 1) * fu) * fv
        out = out + val * amp
        tot += amp
        amp *= 0.5
    return out / tot


def edge_poly(e):
    c = np.array(e["centerline"])
    return G.var_strip(c[:, :2], np.array(e["widths"]))


class World:
    """Everything the environment builders need to know about the ground."""

    def __init__(self, layout: dict, seed=7):
        self.t = layout
        self.seed = seed
        ter = layout.get("terrain", {})
        self.style = ter.get("style", "flat")
        self.cfg = ter
        self.cell = ter.get("cell", 7.0)
        self.roads = {e["id"]: edge_poly(e) for e in layout["edges"] if e.get("polygon", True)}
        self.road_union = unary_union(list(self.roads.values()))
        self.water = [Polygon(a["poly"]) for a in layout.get("areas", []) if a["kind"] == "water"]
        self.sunken = [Polygon(a["poly"]) for a in layout.get("areas", []) if a["kind"] in ("sunken", "trench", "pit")]
        self.raised = [(a.get("h", 6.0), Polygon(a["poly"]), a.get("ramp", 14.0)) for a in layout.get("areas", []) if a["kind"] in ("rocks", "hill")]
        self.water_union = unary_union(self.water) if self.water else Polygon()
        self.land = None
        if self.style == "island":
            loop = [e for e in layout["edges"] if e["id"] in layout["main_loop"]]
            loop.sort(key=lambda e: layout["main_loop"].index(e["id"]))
            ring = Polygon(np.vstack([np.array(e["centerline"])[:-1, :2] for e in loop]))
            ring = ring.buffer(0)
            land = unary_union([ring, self.road_union]).buffer(ter.get("margin", 90.0))
            land = land.buffer(-24).buffer(14)
            self.land = land.difference(self.water_union)
        xs, ys = [], []
        for e in layout["edges"]:
            c = np.array(e["centerline"])
            xs += [c[:, 0].min(), c[:, 0].max()]
            ys += [c[:, 1].min(), c[:, 1].max()]
        pad = ter.get("pad", 170.0)
        self.bbox = (min(xs) - pad, min(ys) - pad - 40, max(xs) + pad, max(ys) + pad)
        self.build_grid()

    # ---- road samples that shape the ground
    def _carve_samples(self):
        pts, zs, hw = [], [], []
        struct = []
        for z in self.t.get("zones", []):
            if z["kind"] in ("viaduct", "bridge", "flyover", "pier", "elevated", "tunnel", "glass_tunnel"):
                struct.append((z["s0"], z["s1"]))
        for e in self.t["edges"]:
            if e["kind"] == "jump_flight" or e.get("structure") in ("pier", "bridge", "viaduct", "elevated"):
                continue
            c = np.array(e["centerline"])
            w = np.array(e["widths"])
            if e["kind"] == "main":
                s = e["s0"] + np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
                keep = np.ones(len(c), bool)
                for a, b in struct:
                    keep &= ~((s >= a) & (s <= b)) if b >= a else ~((s >= a) | (s <= b))
                c, w = c[keep], w[keep]
            if len(c):
                pts.append(c[:, :2]); zs.append(c[:, 2]); hw.append(w / 2)
        return np.vstack(pts), np.concatenate(zs), np.concatenate(hw)

    def build_grid(self):
        x0, y0, x1, y1 = self.bbox
        cs = self.cell
        nx, ny = int((x1 - x0) / cs) + 1, int((y1 - y0) / cs) + 1
        gx, gy = np.meshgrid(x0 + np.arange(nx) * cs, y0 + np.arange(ny) * cs)
        self.gx, self.gy, self.nx, self.ny = gx, gy, nx, ny
        P = np.column_stack([gx.ravel(), gy.ravel()])
        sp, sz, shw = self._carve_samples()
        d2 = np.empty(len(P)); idx = np.empty(len(P), int)
        for i in range(0, len(P), 1500):
            dd = ((P[i:i + 1500, None, :] - sp[None, :, :]) ** 2).sum(-1)
            k = dd.argmin(1)
            idx[i:i + 1500] = k
            d2[i:i + 1500] = dd[np.arange(len(k)), k]
        d = np.sqrt(d2)
        zr, hw = sz[idx], shw[idx]
        amp = self.cfg.get("hill_amp", 5.0)
        base_n = vnoise(P[:, 0], P[:, 1], self.cfg.get("hill_scale", 160.0), self.seed, 4)
        base_n2 = vnoise(P[:, 0], P[:, 1], 40.0, self.seed + 3, 2)
        if self.style == "island":
            pts = shapely.points(P[:, 0], P[:, 1])
            inland0 = shapely.contains(self.land, pts)
            dist0 = shapely.distance(self.land.boundary, pts)
            signed = np.where(inland0, dist0, -dist0) + 20.0 * vnoise(P[:, 0], P[:, 1], 95.0, self.seed + 9, 2) + 8.0 * vnoise(P[:, 0], P[:, 1], 34.0, self.seed + 4, 2)
            inland = signed > 0
            dist_coast = np.abs(signed)
            beach = smoothstep(dist_coast / 26.0)
            h_land = 1.2 + beach * (amp * (0.5 + 0.5 * base_n) + 1.5 * base_n2)
            h_sea = -np.minimum(dist_coast * 0.16, 12.0) - 0.6
            base = np.where(inland, h_land, h_sea)
        else:
            base = self.cfg.get("ground_z", 0.0) + amp * 0.12 * base_n2 * self.cfg.get("rough", 0.0)
        # carve road corridor: flat shoulder, then falloff
        shoulder = hw + self.cfg.get("shoulder", 5.0)
        fall = self.cfg.get("falloff", 38.0)
        t = smoothstep((d - shoulder) / fall)
        target = zr - 0.35
        h = np.where(d <= shoulder, target, (1 - t) * target + t * base)
        # raised rocks / hills and water / sunken carving
        pts_s = shapely.points(P[:, 0], P[:, 1])
        for hh, poly, ramp in self.raised:
            inside = shapely.contains(poly, pts_s)
            dist = shapely.distance(poly.boundary, pts_s)
            lift = hh * smoothstep(dist / ramp) * (0.85 + 0.3 * base_n2)
            h = np.where(inside & (d > shoulder + 6.0), h + lift, h)
        for w in self.water:
            inside = shapely.contains(w, pts_s)
            dist = shapely.distance(w.boundary, pts_s)
            h = np.where(inside, np.minimum(h, -3.5 - 0.15 * np.minimum(dist, 10.0)), h)
        for s in self.sunken:
            inside = shapely.contains(s, pts_s)
            h = np.where(inside, np.minimum(h, self.cfg.get("ground_z", 0.0) - 3.5), h)
        self.H = h.reshape(ny, nx)
        self.dist_road = d.reshape(ny, nx)

    def height_at(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        u, v = (x - self.bbox[0]) / self.cell, (y - self.bbox[1]) / self.cell
        i0 = np.clip(np.floor(u).astype(int), 0, self.nx - 2)
        j0 = np.clip(np.floor(v).astype(int), 0, self.ny - 2)
        fu, fv = np.clip(u - i0, 0, 1), np.clip(v - j0, 0, 1)
        H = self.H
        return (H[j0, i0] * (1 - fu) + H[j0, i0 + 1] * fu) * (1 - fv) + (H[j0 + 1, i0] * (1 - fu) + H[j0 + 1, i0 + 1] * fu) * fv

    # ---- meshes
    def terrain_mesh(self, pal: dict, seed=11):
        rng = np.random.default_rng(seed)
        nx, ny, cs = self.nx, self.ny, self.cell
        jx = rng.uniform(-0.28, 0.28, (ny, nx)) * cs
        jy = rng.uniform(-0.28, 0.28, (ny, nx)) * cs
        jx[[0, -1], :] = 0; jx[:, [0, -1]] = 0; jy[[0, -1], :] = 0; jy[:, [0, -1]] = 0
        X, Y, Z = self.gx + jx, self.gy + jy, self.H
        V = np.stack([X, Y, Z], -1).reshape(-1, 3)
        ids = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, 1:].ravel(), ids[1:, :-1].ravel()
        flip = ((np.arange(len(a)) // (nx - 1) + np.arange(len(a))) % 2).astype(bool)
        t1 = np.where(flip[:, None], np.column_stack([a, b, c]), np.column_stack([a, b, d]))
        t2 = np.where(flip[:, None], np.column_stack([a, c, d]), np.column_stack([b, c, d]))
        F = np.vstack([t1, t2])
        P0, P1, P2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        nrm = np.cross(P1 - P0, P2 - P0)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        nrm[nrm[:, 2] < 0] *= -1
        cen = (P0 + P1 + P2) / 3
        ctr = V[F].mean(1)
        col = pal["func"](cen, nrm, self, rng)
        keep = np.ones(len(F), bool)
        if pal.get("drop_below") is not None:
            keep = cen[:, 2] > pal["drop_below"]
        return V, F[keep], col[keep]

    def water_mesh(self, level=0.0, cs=9.0, shallow="#3fe6d6", deep="#1479c4", seed=5, foam=True, amp=0.12, mid=None):
        x0, y0, x1, y1 = self.bbox
        nx, ny = int((x1 - x0) / cs) + 1, int((y1 - y0) / cs) + 1
        rng = np.random.default_rng(seed)
        gx, gy = np.meshgrid(x0 + np.arange(nx) * cs, y0 + np.arange(ny) * cs)
        jit = rng.uniform(-amp, amp, gx.shape)
        if amp > 0.2:                                                  # strong swell only inside; calm towards the box edge so that it meets the far sea without a seam
            ed = np.minimum(np.minimum(gx - x0, x1 - gx), np.minimum(gy - y0, y1 - gy))
            jit = jit * (0.22 + 0.78 * np.clip(ed / 140.0, 0, 1))
        V = np.stack([gx, gy, level + jit], -1).reshape(-1, 3)
        ids = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = ids[:-1, :-1].ravel(), ids[:-1, 1:].ravel(), ids[1:, 1:].ravel(), ids[1:, :-1].ravel()
        F = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
        cen = V[F].mean(1)
        gnd = self.height_at(cen[:, 0], cen[:, 1])
        depth = np.maximum(level - gnd, 0.0)
        keep = depth > -0.6
        if self.style == "city" and len(self.water):
            import shapely
            keep &= shapely.contains(self.water_union.buffer(cs * 1.6), shapely.points(cen[:, 0], cen[:, 1]))    # water only inside canal polygons (not in road trenches / tunnel ramps)
        hx = lambda h: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
        t = np.clip(depth / 8.0, 0, 1)[:, None]
        if mid:                                                       # three-stop gradient: shallow -> mid -> deep
            t1 = np.clip(t / 0.45, 0, 1); t2 = np.clip((t - 0.45) / 0.55, 0, 1)
            col = np.where(t < 0.45, hx(shallow) * (1 - t1) + hx(mid) * t1, hx(mid) * (1 - t2) + hx(deep) * t2)
            col = col * (1 + rng.uniform(-0.05, 0.05, (len(col), 1)))
        else:
            col = hx(shallow) * (1 - t) + hx(deep) * t
        if foam:
            col = np.where((depth < 1.1)[:, None], col * 0.18 + 0.82, col)
        return V, F[keep], col[keep]

    # ---- barrier masks (where a road edge gets a guard wall)
    def barrier_masks(self):
        out = {}
        nodes = {n["id"]: n for n in self.t["nodes"]}
        for e in self.t["edges"]:
            if not e.get("polygon", True):
                continue
            c = np.array(e["centerline"])
            w = np.array(e["widths"])
            h = G.headings(c[:, :2])
            nrm = np.column_stack([-h[:, 1], h[:, 0]])
            others = unary_union([p for k, p in self.roads.items() if k != e["id"]]).buffer(3.2)
            res = {}
            for side, nm in ((1, "l"), (-1, "r")):
                pp = c[:, :2] + nrm * (w / 2 + 1.0)[:, None] * side
                pts = shapely.points(pp[:, 0], pp[:, 1])
                inside_other = shapely.contains(others, pts)
                m = ~inside_other
                for nid in (e["from"], e["to"]):
                    nd = nodes[nid]
                    if nd["type"] in ("jump_ramp_start", "jump_takeoff", "jump_landing_start", "jump_landing_end"):
                        continue
                    near = np.linalg.norm(pp - np.array(nd["xy"]), axis=1) < 16.0 if nd["type"] in ("shortcut_fork", "shortcut_merge", "jump_branch", "jump_merge", "pit_entry", "pit_exit") else np.zeros(len(pp), bool)
                    m &= ~near
                if e["kind"] in ("jump_ramp", "jump_landing", "jump_approach", "jump_exit"):
                    m &= True
                res[nm] = m.tolist()
            cp = shapely.points(c[:, 0], c[:, 1])
            res["pil"] = (~shapely.contains(others.buffer(2.0), cp)).tolist()
            out[e["id"]] = res
        return out

    # ---- scatter
    def scatter(self, n, *, region=None, avoid_roads=14.0, h_range=(-1e9, 1e9), seed=1, min_dist=6.0, avoid=None, slope_max=None, tries=40):
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = self.bbox
        reg = region or Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        rx0, ry0, rx1, ry1 = reg.bounds
        road_b = self.road_union.buffer(avoid_roads)
        av = unary_union([road_b] + ([avoid] if avoid is not None else []) + self.water)
        pts, grid = [], {}
        cell = min_dist
        for _ in range(n * tries):
            if len(pts) >= n:
                break
            x, y = rng.uniform(rx0, rx1), rng.uniform(ry0, ry1)
            gi = (int(x // cell), int(y // cell))
            if any((gi[0] + a, gi[1] + b) in grid and np.hypot(grid[(gi[0] + a, gi[1] + b)][0] - x, grid[(gi[0] + a, gi[1] + b)][1] - y) < min_dist
                   for a in (-1, 0, 1) for b in (-1, 0, 1)):
                continue
            p = Point(x, y)
            if not reg.contains(p) or av.contains(p):
                continue
            if self.land is not None and not self.land.contains(p):
                continue
            z = float(self.height_at(x, y))
            if not (h_range[0] <= z <= h_range[1]):
                continue
            if slope_max is not None:
                dz = abs(float(self.height_at(x + 3, y)) - z) + abs(float(self.height_at(x, y + 3)) - z)
                if dz / 6.0 > slope_max:
                    continue
            grid[gi] = (x, y)
            pts.append((x, y, z))
        return pts

    def along(self, edge_id, s_list, offset=0.0):
        """Points at arclength s along an edge, shifted laterally by `offset` (left positive): returns (x, y, z, heading_deg)."""
        e = next(e for e in self.t["edges"] if e["id"] == edge_id)
        c = np.array(e["centerline"])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
        out = []
        for s in s_list:
            i = int(np.clip(np.searchsorted(cum, s), 1, len(c) - 1))
            f = (s - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9)
            p = c[i - 1] + (c[i] - c[i - 1]) * f
            h = c[i][:2] - c[i - 1][:2]
            h = h / max(np.linalg.norm(h), 1e-9)
            n = np.array([-h[1], h[0]])
            out.append((p[0] + n[0] * offset, p[1] + n[1] * offset, p[2], math.degrees(math.atan2(h[1], h[0]))))
        return out

    def ring_at(self, s, offset=0.0):
        """Point on the main loop at lap arclength s."""
        s = s % self.t["length_m"]
        for eid in self.t["main_loop"]:
            e = next(x for x in self.t["edges"] if x["id"] == eid)
            if e["s0"] <= s <= e["s1"] + 1e-6:
                return self.along(eid, [s - e["s0"]], offset)[0]
        return self.along(self.t["main_loop"][-1], [0.0], offset)[0]


def city_lots(W, lay, *, cell=46.0, road_clear=17.0, avoid=None, seed=3, h_fn=None, w_range=(18, 34), keep=0.78, region=None, extra_clear=None):
    """Building lots on a jittered grid, never inside the road buffer; heights from h_fn(x, y, rng)."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = W.bbox
    road_b = W.road_union.buffer(road_clear)
    bad = [road_b] + list(W.water) + list(W.sunken) + ([avoid] if avoid is not None else [])
    for lm in lay.get("landmarks", []):
        if lm.get("footprint") and lm.get("check", True) and lm["footprint"][0] > 6:
            bad.append(G.rect(*lm["xy"], *lm["footprint"], lm.get("rot", 0.0)).buffer(10))
    for st in lay.get("stands", []):
        bad.append(G.rect(*st["center"], *st["size"], st["heading_deg"]).buffer(14))
    pit = lay["pit"]["building"]
    bad.append(G.rect(*pit["center"], *pit["size"], 0).buffer(22))
    bad = unary_union(bad)
    out = []
    for gx in np.arange(x0 + cell / 2, x1, cell):
        for gy in np.arange(y0 + cell / 2, y1, cell):
            if rng.uniform() > keep:
                continue
            x, y = gx + rng.uniform(-6, 6), gy + rng.uniform(-6, 6)
            w, d = rng.uniform(*w_range), rng.uniform(*w_range)
            rot = float(rng.choice([0, 0, 0, 90])) + rng.uniform(-3, 3)
            fp = G.rect(x, y, w + 4, d + 4, rot)
            if fp.intersects(bad) or (region is not None and not region.contains(Point(x, y))):
                continue
            h = float(h_fn(x, y, rng)) if h_fn else float(rng.uniform(25, 120))
            dist_r = float(W.road_union.distance(Point(x, y)))
            h *= float(np.clip(0.28 + dist_r / 120.0, 0.28, 1.0))
            out.append(dict(t="tower", x=float(x), y=float(y), z=float(W.height_at(x, y)), w=float(w), d=float(d), h=h, rot=float(rot), seed=int(rng.integers(1 << 30))))
    return out


def lamps_along(W, lay, spacing=64.0, off=4.5, kinds=("main",)):
    out = []
    for e in lay["edges"]:
        if e["kind"] not in kinds or not e.get("polygon", True):
            continue
        c = np.array(e["centerline"])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
        for s in np.arange(spacing / 2, cum[-1], spacing):
            i = int(np.clip(np.searchsorted(cum, s), 1, len(c) - 1))
            h = c[i][:2] - c[i - 1][:2]
            h /= max(np.linalg.norm(h), 1e-9)
            n = np.array([-h[1], h[0]])
            for sd in (1, -1):
                w = e["widths"][i]
                p = c[i][:2] + n * sd * (w / 2 + off)
                if Point(p).distance(W.road_union.difference(Point(c[i][:2]).buffer(w / 2 + 0.5))) < 2.0 and False:
                    continue
                out.append(dict(t="lamp", x=float(p[0]), y=float(p[1]), z=float(c[i][2]), sd=sd))
    return out
