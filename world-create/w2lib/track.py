"""Reusable track builder (the historical w2 schema name is kept for compatibility).

An authored circuit (straights, exact arcs, clamped splines with continuous heading) becomes a directed road graph with
3D centrelines: main loop, pit lane with 12 boxes, shortcuts with real fork/merge nodes, jumps with ramp / flight /
landing, start grid, stands, sections ("musical" dramaturgy). All geometry is metres, +x east, +y north, z up.
The result is plain JSON-able data (`Track.build()`), validated in `validate.py`, drawn in `draw.py`, built in Blender.

Driving model assumptions (documented in every validation report): see `drive.py` (38 m/s top speed, 13 m/s² lateral
grip, 9 m/s² engine, 22 m/s² brakes). This is a lap-time model for design, not a physics simulation of the game kart."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import drive as D
from . import geom as G
from .pathkit import Path, _spline_c2, hdeg, hvec, smooth_profile, smooth_width

DS = 2.0
ZONE_R = 30.0      # junction zone: road overlap allowed within this radius of a fork/merge node
CLEAR_R = 44.0     # beyond this radius from a node, other roads must keep the safety distance
MIN_CLEAR = 8.0    # road edge to road edge at the same level (barrier + runoff)
LEVEL_CLEAR = 6.0  # vertical clearance deck to deck for bridges / underpasses


class AuthorError(Exception):
    pass


def _r(a, n=2):
    return np.round(np.asarray(a, float), n).tolist()


def _node(nid, typ, xy, z=0.0, heading=None, zone=ZONE_R):
    d = dict(id=nid, type=typ, xy=[round(float(xy[0]), 2), round(float(xy[1]), 2)], z=round(float(z), 2), zone_radius=zone)
    if heading is not None:
        d["heading_deg"] = round(float(heading), 1)
    return d


def _edge(eid, kind, xyz, width, frm, to, grip=1.0, polygon=True, **extra):
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    w = np.broadcast_to(np.asarray(width, float), (n,)).copy()
    g = np.broadcast_to(np.asarray(grip, float), (n,)).copy()
    e = dict(id=eid, kind=kind, **{"from": frm, "to": to}, centerline=_r(xyz, 2), widths=_r(w, 2), grip=_r(g, 3), polygon=polygon,
             length=round(float(G.seglen(xyz[:, :2]).sum()), 1))
    e.update(extra)
    return e


def edge_poly(e: dict) -> Polygon:
    c = np.array(e["centerline"])
    return G.var_strip(c[:, :2], np.array(e["widths"]))


class Track:
    def __init__(self, tid: str, name: str, genre: str, difficulty: int, direction: str = "CCW", theme: str = "coastal", seed: int = 1,
                 half: float = 220.0, boxes: int = 12, pit_side: int = -1, w_pit: float = 9.0, w_short: float = 10.0, w_jump: float = 12.0,
                 min_r_main: float = 22.0, min_r_short: float = 16.0, pit_gap: float = 8.0, lap_target=(60, 95), bpm: int = 120):
        self.id, self.name, self.genre, self.difficulty, self.direction, self.theme, self.seed = tid, name, genre, difficulty, direction, theme, seed
        self.half, self.boxes, self.pit_side, self.w_pit, self.w_short, self.w_jump = half, boxes, pit_side, w_pit, w_short, w_jump
        self.min_r_main, self.min_r_short, self.pit_gap, self.lap_target, self.bpm = min_r_main, min_r_short, pit_gap, lap_target, bpm
        self.P = Path((0.0, 0.0), 0.0, ds=DS)
        self.z_st: list = []
        self.w_st: list = []
        self.g_st: list = []          # grip stations (mark, frac, grip)
        self.zones: list[dict] = []   # surface / structure zones: dict(id, kind, a=(mark,frac), b=(mark,frac), grip=?, label)
        self.sections: list = []      # (key, (mark, frac)) in lap order, first one starts the lap after S/F wrap rules
        self.shortcut_defs: list[dict] = []
        self.jump_defs: list[dict] = []
        self.stand_defs: list[dict] = []
        self.landmarks: list[dict] = []
        self.traps: list[dict] = []
        self.keep_out: list[Polygon] = []   # authored infield areas (stage, arena) that roads must not enter
        self.areas: list[dict] = []         # terrain areas: dict(id, kind water|sunken|plaza|park|rocks|industrial|building, poly=[[x,y],...])
        self.terrain: dict = {}             # style (island|city|...), margin, ground colour, notes
        self.notes: list[str] = []
        self.pit_building_style = "garage"

    # ------------------------------------------------------------------ authoring helpers
    # ------------------------------------------------------------------ authoring macros (exact arcs => radii by construction)
    def lobe(self, name, R=45.0, theta=50.0, s1=30.0, side=1, R2=None):
        """Detour bump of the main road: arc(theta), straight, arc(-2 theta), straight, arc(theta); net heading unchanged.
        side=+1 bulges to the left. Returns (chord, depth, extra_length) so the author can place a shortcut across it."""
        import math as m
        R2 = R2 or R
        th = m.radians(theta)
        a = side * theta
        self.P.arc(R, a, f"{name}_in").line(s1, f"{name}_s1").arc(R2, -2 * a, f"{name}_mid").line(s1, f"{name}_s2").arc(R, a, f"{name}_out")
        chord = 2 * R * m.sin(th) + 2 * s1 * m.cos(th) + 2 * R2 * m.sin(th)
        depth = R * (1 - m.cos(th)) + s1 * m.sin(th) + R2 * (1 - m.cos(th))
        arc_len = 2 * R * th + 2 * R2 * th + 2 * s1
        return dict(chord=round(chord, 1), depth=round(depth, 1), length=round(arc_len, 1))

    def chicane(self, name, R=28.0, theta=35.0, side=1, mid=0.0):
        a = side * theta
        self.P.arc(R, a, f"{name}_1").line(mid, f"{name}_m") if mid else self.P.arc(R, a, f"{name}_1")
        self.P.arc(R, -2 * a, f"{name}_2").arc(R, a, f"{name}_3")
        return self

    def sweep(self, name, R, deg):
        self.P.arc(R, deg, name)
        return self

    def shortcut(self, sid, A, B, via=None, side=1, angA=26, angB=26, width=None, grip=0.94, name="", z_via=None, surface="shortcut", bez=None, **extra):
        self.shortcut_defs.append(dict(id=sid, A=A, B=B, via=via or [], side=side, angA=angA, angB=angB, width=width or self.w_short, grip=grip, bez=bez,
                                       name=name or sid, z_via=z_via or [], surface=surface, **extra))

    def jump(self, jid, A, B, gap, name="", gap_center=0.5, lip_at=None, ramp_len=24.0, ramp_deg=10.0, land_len=48.0, land_deg=7.0, h0=1.6, width=None, **extra):
        self.jump_defs.append(dict(id=jid, A=A, B=B, gap=gap, name=name or jid, gap_center=gap_center, lip_at=lip_at, ramp_len=ramp_len, ramp_deg=ramp_deg,
                                   land_len=land_len, land_deg=land_deg, h0=h0, width=width or self.w_jump, **extra))

    def zone(self, zid, kind, a, b, grip=None, label="", **extra):
        self.zones.append(dict(id=zid, kind=kind, a=a, b=b, grip=grip, label=label or zid, **extra))

    # ------------------------------------------------------------------ build
    def _marks_s(self):
        a = self.P.array()
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(a, axis=0), axis=1))])
        return {m["name"]: (float(cum[m["i0"]]), float(cum[m["i1"]])) for m in self.P.marks if m["name"]}

    def S(self, mark_frac) -> float:
        """Arclength (m, along the resampled ring) of an authored station (mark, fraction)."""
        if isinstance(mark_frac, (int, float)):
            return float(mark_frac) % self.L
        name, frac = mark_frac
        s0, s1 = self.marks[name]
        return s0 + (s1 - s0) * frac

    def _profile(self, table, s, fn, default):
        if not table:
            return np.full_like(s, default, dtype=float)
        xs, ys = [], []
        for name, frac, val in table:
            x = self.S((name, frac))
            if xs and x <= xs[-1] + 1e-6:
                raise AuthorError(f"profile stations must increase along the lap: {(name, frac)} at s={x:.1f} after {xs[-1]:.1f}")
            xs.append(x)
            ys.append(val)
        return fn(xs, ys, s)

    def idx(self, s) -> int:
        return int(round((self.S(s) % self.L) / self.ds_ring)) % self.n

    def _slice(self, i0: int, i1: int) -> np.ndarray:
        """Ring indices from i0 to i1 inclusive, wrapping (i1 == i0 means a full loop is not requested; use i1 = i0 + n)."""
        return np.arange(i0, i1 + 1) % self.n

    def prepare(self):
        P = self.P
        if np.linalg.norm(P.pos) > 0.06:
            raise AuthorError(f"main circuit does not close: end {P.pos.round(3).tolist()}")
        if abs(((hdeg(P.h) + 180) % 360) - 180) > 0.6:
            raise AuthorError(f"closing heading {hdeg(P.h):.2f} deg != 0 (S/F straight must be continuous)")
        self.marks = self._marks_s()
        raw = P.array()[:-1]
        ring = G.resample(raw, DS, closed=True)
        ring[0] = (0.0, 0.0)
        self.ring = ring
        self.n = len(ring)
        self.L = float(G.seglen(ring, True).sum())
        self.ds_ring = self.L / self.n
        k = self.L / self.marks[list(self.marks)[-1]][1]
        self.marks = {m: (a * k, b * k) for m, (a, b) in self.marks.items()}
        self.s_ring = np.arange(self.n) * self.ds_ring
        self.hh = G.headings(ring, True)
        self.zr = self._profile(self.z_st, self.s_ring, smooth_profile, 0.0)
        self.wr = self._profile(self.w_st, self.s_ring, smooth_width, 14.0)
        self.gr = self._profile(self.g_st, self.s_ring, smooth_width, 1.0)
        self.zone_s = []
        for zn in self.zones:
            a, b = self.S(zn["a"]), self.S(zn["b"])
            zn["s0"], zn["s1"] = round(a, 1), round(b, 1)
            if zn.get("grip") is not None:
                m = (self.s_ring >= a) & (self.s_ring <= b)
                self.gr = np.where(m, np.minimum(self.gr, zn["grip"]), self.gr)
        ring_z = np.column_stack([ring, self.zr])
        self.main_poly = unary_union([G.var_strip(ring[[(i0 + k_) % self.n for k_ in range(0, 82)]], self.wr[[(i0 + k_) % self.n for k_ in range(0, 82)]])
                                       for i0 in range(0, self.n, 80)]).buffer(0.02)
        self.prof = D.speed_profile(np.column_stack([ring, self.zr]), self.wr, True, grip=self.gr)
        self._prepared = True
        return self

    def build(self) -> dict:
        if not getattr(self, "_prepared", False):
            self.prepare()
        ring = self.ring
        nodes: dict[str, dict] = {}
        junctions: list[tuple[float, dict]] = [(0.0, _node("n_sf", "start_finish", ring[0], self.zr[0], 0.0, 1.0))]
        edges: list[dict] = []
        self.others: list[tuple[Polygon, set]] = []

        pit = self._build_pit(nodes, junctions, edges)
        short = [self._build_shortcut(d, nodes, junctions, edges) for d in self.shortcut_defs]
        jumps = [self._build_jump(d, nodes, junctions, edges) for d in self.jump_defs]

        # main edges split at the junctions along the ring
        junctions.sort(key=lambda t: t[0])
        for (s, nd) in junctions:
            nodes[nd["id"]] = nd
        main_ids = []
        for j, (s, nd) in enumerate(junctions):
            nxt_s, nxt = junctions[(j + 1) % len(junctions)]
            i0 = self.idx(s)
            i1 = self.idx(nxt_s) if nxt_s > s else self.n
            sl = np.arange(i0, i1 + 1) % self.n
            eid = f"main_{j + 1}"
            xyz = np.column_stack([ring[sl], self.zr[sl]])
            xyz[0, :2], xyz[-1, :2] = nd["xy"], nxt["xy"]
            e = _edge(eid, "main", xyz, self.wr[sl], nd["id"], nxt["id"], grip=self.gr[sl], s0=round(s, 1), s1=round(nxt_s if nxt_s > s else self.L, 1))
            main_ids.append(eid)
            edges.append(e)
        E = {e["id"]: e for e in edges}
        self.nodes, self.edges, self.E = nodes, edges, E
        self.main_ids = main_ids

        routes, laps = self._routes(pit, short, jumps)
        grid = self._grid()
        sections = self._sections()
        over = self._overtaking()
        stands = self._stands()
        traps = [self._resolve_trap(t) for t in self.traps]
        zones = [dict(z, a=list(z["a"]), b=list(z["b"])) for z in self.zones]

        return dict(
            schema="riffracers.w2.track/1", id=self.id, name=self.name, genre=self.genre, difficulty=self.difficulty, direction=self.direction,
            theme=self.theme, bpm=self.bpm, units="m", seed=self.seed, length_m=round(self.L, 1),
            assumptions=dict(kart=D.Kart().__dict__, note="lap-time model for design, no physics simulation; flight = ballistic with ramp angle and height difference"),
            nodes=list(nodes.values()), edges=edges, main_loop=main_ids, start_finish=dict(node="n_sf", xy=[0.0, 0.0], heading_deg=0.0, grid=grid),
            pit=pit, shortcuts=short, jumps=jumps, routes=routes, lap_times=laps, sections=sections, overtaking=over, stands=stands,
            landmarks=self.landmarks, zones=zones, traps=traps, notes=self.notes,
            profile=dict(s=_r(self.s_ring[::5], 1), z=_r(self.zr[::5], 2), w=_r(self.wr[::5], 2), v=_r(self.prof["v"][::5], 1), grip=_r(self.gr[::5], 2)),
            keep_out=[_r(np.array(p.exterior.coords), 1) for p in self.keep_out], areas=self.areas, terrain=self.terrain,
        )

    # ------------------------------------------------------------------ pit lane
    def _build_pit(self, nodes, junctions, edges):
        ring, L = self.ring, self.L
        sd = self.pit_side
        wst = float(self.wr[self.idx(10.0)])
        yl = sd * (wst / 2 + self.pit_gap + self.w_pit / 2)
        s_pe, s_px = L - (self.half - 15), self.half - 15
        pe, px = ring[self.idx(s_pe)], ring[self.idx(s_px)]
        sp, bw, bd = 13.0, 11.0, 7.5
        lane_half = (self.boxes - 1) * sp / 2 + 34.0
        l0, l1 = np.array([-lane_half, yl]), np.array([lane_half, yl])
        e_ = np.array([1.0, 0.0])
        a_ = math.radians(20.0 * sd)
        fork = np.array([math.cos(a_), math.sin(a_)])
        join = np.array([math.cos(-a_), math.sin(-a_)])
        ent = G.resample(G.bezier(pe, pe + fork * 38, l0 - e_ * 42, l0, 90), DS)
        ext = G.resample(G.bezier(l1, l1 + e_ * 42, px - join * 38, px, 90), DS)
        ent[0], ent[-1], ext[0], ext[-1] = pe, l0, l1, px
        lane = np.linspace(l0, l1, int(2 * lane_half / DS) + 1)
        zp = lambda i: float(self.zr[i])
        nodes["n_pit_in"] = _node("n_pit_in", "pit_lane_start", l0, zp(self.idx(s_pe)), 0.0, 1.0)
        nodes["n_pit_out"] = _node("n_pit_out", "pit_lane_end", l1, zp(self.idx(s_px)), 0.0, 1.0)
        junctions += [(s_pe, _node("n_pit_entry", "pit_entry", pe, self.zr[self.idx(s_pe)], 0.0, 56.0)),
                      (s_px, _node("n_pit_exit", "pit_exit", px, self.zr[self.idx(s_px)], 0.0, 56.0))]
        z0, z1 = zp(self.idx(s_pe)), zp(self.idx(s_px))
        g = 0.92
        pe_e = _edge("pit_entry", "pit_entry", np.column_stack([ent, np.full(len(ent), z0)]), self.w_pit, "n_pit_entry", "n_pit_in", grip=g)
        pl_e = _edge("pit_lane", "pit_lane", np.column_stack([lane, np.linspace(z0, z1, len(lane))]), self.w_pit, "n_pit_in", "n_pit_out", grip=g)
        px_e = _edge("pit_exit", "pit_exit", np.column_stack([ext, np.full(len(ext), z1)]), self.w_pit, "n_pit_out", "n_pit_exit", grip=g)
        edges += [pe_e, pl_e, px_e]
        for e in (pe_e, pl_e, px_e):
            self.others.append((edge_poly(e), {e["from"], e["to"]}))
        boxes = []
        for i in range(self.boxes):
            cx = (i - (self.boxes - 1) / 2) * sp
            cy = yl + sd * (self.w_pit / 2 + bd / 2 + 0.5)
            boxes.append(dict(id=f"box_{i + 1}", index=i + 1, center=[round(cx, 2), round(cy, 2)], size=[bw, bd], heading_deg=0.0,
                              lane_point=[round(cx, 2), round(yl, 2)]))
        span = (self.boxes - 1) * sp + bw
        bld_c = yl + sd * (self.w_pit / 2 + bd + 0.5 + 8.0)
        self.keep_out.append(G.rect(0.0, bld_c, span + 10, 17.0))
        return dict(lane_y=round(yl, 2), side=sd, boxes=boxes, lane_half=round(lane_half, 1),
                    building=dict(id="pit_building", center=[0.0, round(bld_c, 2)], size=[round(span + 8, 1), 15.0], height=7.0, heading_deg=0.0,
                                  style=self.pit_building_style),
                    speed_limit_kph=60, one_way=True, entry_node="n_pit_entry", exit_node="n_pit_exit", box_spacing=sp,
                    entry_s=round(s_pe, 1), exit_s=round(s_px, 1))

    # ------------------------------------------------------------------ shortcuts
    def _build_shortcut(self, d, nodes, junctions, edges):
        ring = self.ring
        sA, sB = self.S(d["A"]), self.S(d["B"])
        iA, iB = self.idx(sA), self.idx(sB)
        if not sB > sA:
            raise AuthorError(f"{d['id']}: merge must lie after fork along the lap")
        A, B, hA, hB = ring[iA], ring[iB], self.hh[iA], self.hh[iB]
        side = d["side"]
        if d.get("bez"):
            dA, dB = hvec(hdeg(hA) + side * d["angA"]), hvec(hdeg(hB) - side * d["angB"])
            ch = float(np.linalg.norm(B - A))
            pth = G.resample(G.bezier(A, A + dA * ch * d["bez"][0], B - dB * ch * d["bez"][1], B, 240), DS)
            pth[0], pth[-1] = A, B
        else:
            via = d["via"]
            if d.get("weave"):
                amp, nw = d["weave"]
                vv = B - A
                u_ = vv / float(np.linalg.norm(vv))
                nn = np.array([-u_[1], u_[0]])
                ts = np.linspace(0, 1, 6 * nw + 3)[1:-1]
                via = [(A + vv * t_ + nn * amp * math.sin(math.pi * t_) * math.sin(nw * math.pi * t_)).tolist() for t_ in ts]
            c = Path(A, hdeg(hA) + side * d["angA"], ds=DS)
            c.spline(via, B, hdeg(hB) - side * d["angB"])
            pth = c.array()
        n = len(pth)
        zA, zB = float(self.zr[iA]), float(self.zr[iB])
        t = np.linspace(0, 1, n)
        zz = zA + (zB - zA) * (3 * t ** 2 - 2 * t ** 3)
        for f, dz in d["z_via"]:
            zz = zz + dz * np.exp(-((t - f) / 0.12) ** 2)
        zz[0], zz[-1] = zA, zB
        nA = _node(f"n_{d['id']}_a", "shortcut_fork", A, zA, hdeg(hA), 52.0)
        nB = _node(f"n_{d['id']}_b", "shortcut_merge", B, zB, hdeg(hB), 52.0)
        junctions += [(sA, nA), (sB, nB)]
        e = _edge(d["id"], "shortcut", np.column_stack([pth, zz]), d["width"], nA["id"], nB["id"], grip=d["grip"], surface=d["surface"], label=d["name"], structure=d.get("structure"))
        edges.append(e)
        self.others.append((edge_poly(e), {nA["id"], nB["id"]}))
        main_len = sB - sA
        return dict(id=d["id"], name=d["name"], edge=d["id"], fork_node=nA["id"], merge_node=nB["id"], main_span_s=[round(sA, 1), round(sB, 1)],
                    main_span_len=round(main_len, 1), length=e["length"], saving_m=round(main_len - e["length"], 1), fork_angle_deg=d["angA"], merge_angle_deg=d["angB"],
                    width=d["width"], grip=d["grip"], surface=d["surface"], side=side, entrance=d.get("entrance", ""), exit=d.get("exit", ""),
                    risk=d.get("risk", ""), reward=d.get("reward", ""))

    # ------------------------------------------------------------------ jumps
    def _build_jump(self, d, nodes, junctions, edges):
        ring = self.ring
        sA, sB = self.S(d["A"]), self.S(d["B"])
        iA, iB = self.idx(sA), self.idx(sB)
        A, B = ring[iA].copy(), ring[iB].copy()
        v = B - A
        chord = float(np.linalg.norm(v))
        u = v / chord
        hu = hdeg(u)
        dA = abs(((hdeg(self.hh[iA]) - hu + 180) % 360) - 180)
        dB = abs(((hdeg(self.hh[iB]) - hu + 180) % 360) - 180)
        if dA > 8 or dB > 8:
            raise AuthorError(f"{d['id']}: chord heading differs from road heading at A/B by {dA:.1f}/{dB:.1f} deg (must be <= 8): author A and B on straights collinear with the chord")
        gap, rl, ll = d["gap"], d["ramp_len"], d["land_len"]
        lip_s = d["lip_at"] if d.get("lip_at") is not None else chord * d["gap_center"] - gap / 2
        ap_len = lip_s - rl
        ex_len = chord - (lip_s + gap + ll)
        if ap_len < 12 or ex_len < 30:
            raise AuthorError(f"{d['id']}: chord {chord:.0f} m: approach {ap_len:.0f} m (>=12), exit {ex_len:.0f} m (>=30) - lengthen the chord or shorten ramp/landing")
        pt = lambda s_: A + u * s_
        R0, R1, L0, L1 = pt(ap_len), pt(lip_s), pt(lip_s + gap), pt(lip_s + gap + ll)
        zA, zB = float(self.zr[iA]), float(self.zr[iB])
        rise = rl * math.tan(math.radians(d["ramp_deg"]))
        z_R0 = zA
        z_lip = z_R0 + rise
        z_L0 = z_lip - d["h0"]
        z_L1 = z_L0 - ll * math.tan(math.radians(d["land_deg"]))
        w = d["width"]
        nsm = lambda a, b: max(3, int(round(np.linalg.norm(b - a) / DS)) + 1)
        n_ap, n_rm, n_ld, n_ex = nsm(A, R0), 14, nsm(L0, L1), nsm(L1, B)
        ap = np.column_stack([np.linspace(A[0], R0[0], n_ap), np.linspace(A[1], R0[1], n_ap), np.full(n_ap, zA)])
        tt = np.linspace(0, 1, n_rm)
        rm = np.column_stack([R0[0] + (R1[0] - R0[0]) * tt, R0[1] + (R1[1] - R0[1]) * tt, z_R0 + rise * tt ** 1.35])
        nfl = 22
        ft = np.linspace(0, 1, nfl)
        tan_a = math.tan(math.radians(d["ramp_deg"]))
        drop_total = z_lip - z_L0
        # parabola through lip (z_lip) and landing start (z_L0) leaving the lip at the ramp angle
        fl_z = z_lip + tan_a * (gap * ft) - (tan_a * gap + drop_total) * ft ** 2
        fl = np.column_stack([R1[0] + (L0[0] - R1[0]) * ft, R1[1] + (L0[1] - R1[1]) * ft, fl_z])
        lt = np.linspace(0, 1, n_ld)
        ld = np.column_stack([L0[0] + (L1[0] - L0[0]) * lt, L0[1] + (L1[1] - L0[1]) * lt, z_L0 - (z_L0 - z_L1) * (1 - (1 - lt) ** 1.3)])
        et = np.linspace(0, 1, n_ex)
        ex = np.column_stack([L1[0] + (B[0] - L1[0]) * et, L1[1] + (B[1] - L1[1]) * et, z_L1 + (zB - z_L1) * (3 * et ** 2 - 2 * et ** 3)])
        jid = d["id"]
        nA = _node(f"n_{jid}_a", "jump_branch", A, zA, hu, float(d.get("zone_a", ZONE_R)))     # zone_a / zone_b: author-set connection zones (default 30 / 46 m)
        nB = _node(f"n_{jid}_b", "jump_merge", B, zB, hu, float(d.get("zone_b", 46.0)))
        nn = {k: _node(f"{jid}_{k}", t_, p, z, hu, 1.0) for k, t_, p, z in (("ramp0", "jump_ramp_start", R0, z_R0), ("lip", "jump_takeoff", R1, z_lip),
                                                                             ("land0", "jump_landing_start", L0, z_L0), ("land1", "jump_landing_end", L1, z_L1))}
        nodes.update({n_["id"]: n_ for n_ in nn.values()})
        junctions += [(sA, nA), (sB, nB)]
        zn = [nA["id"], nB["id"]]
        e_ap = _edge(f"{jid}_approach", "jump_approach", ap, w, nA["id"], nn["ramp0"]["id"], jump=jid)
        e_rm = _edge(f"{jid}_ramp", "jump_ramp", rm, w, nn["ramp0"]["id"], nn["lip"]["id"], jump=jid, zone_nodes=zn)
        e_fl = _edge(f"{jid}_flight", "jump_flight", fl, w, nn["lip"]["id"], nn["land0"]["id"], polygon=False, airborne=True, jump=jid)
        e_ld = _edge(f"{jid}_landing", "jump_landing", ld, w + 4, nn["land0"]["id"], nn["land1"]["id"], jump=jid, zone_nodes=zn)
        e_ex = _edge(f"{jid}_exit", "jump_exit", ex, w, nn["land1"]["id"], nB["id"], jump=jid)
        edges += [e_ap, e_rm, e_fl, e_ld, e_ex]
        for e in (e_ap, e_rm, e_ld, e_ex):
            self.others.append((edge_poly(e), {e["from"], e["to"]}))
        # lip speed: main-lap speed at A, then accelerate along approach + ramp
        vA = float(self.prof["v"][iA])
        xy_in = np.vstack([ap, rm[1:]])
        pr = D.speed_profile(xy_in, w, False, grip=1.0, v0=vA)
        v_lip = float(pr["v"][-1])
        fl_ref = D.flight(v_lip, d["ramp_deg"], d["h0"])
        lo, hi = 0.88 * v_lip, 1.05 * v_lip
        d_lo, d_hi = D.flight(lo, d["ramp_deg"], d["h0"])["range"], D.flight(hi, d["ramp_deg"], d["h0"])["range"]
        v_req = next((f for f in np.arange(0.4, 1.3, 0.005) if D.flight(f * v_lip, d["ramp_deg"], d["h0"])["range"] >= gap + 1.5), 9.9)
        main_len = sB - sA
        jlen = sum(e["length"] for e in (e_ap, e_rm, e_fl, e_ld, e_ex))
        return dict(id=jid, name=d["name"], fork_node=nA["id"], merge_node=nB["id"], edges=[e_ap["id"], e_rm["id"], e_fl["id"], e_ld["id"], e_ex["id"]],
                    main_span_s=[round(sA, 1), round(sB, 1)], main_span_len=round(main_len, 1), length=round(jlen, 1), saving_m=round(main_len - jlen, 1),
                    takeoff=_r(R1, 2), landing_start=_r(L0, 2), landing_end=_r(L1, 2), direction_deg=round(hu, 1), gap=gap, ramp_len=rl, ramp_deg=d["ramp_deg"],
                    ramp_rise=round(rise, 2), takeoff_height_above_landing=d["h0"], landing_len=ll, landing_deg=d["land_deg"], width=w,
                    approach_len=round(ap_len, 1), exit_len=round(ex_len, 1), ref_speed_mps=round(v_lip, 1), entry_speed_mps=round(vA, 1),
                    nominal_flight_distance=round(fl_ref["range"], 1), flight_distance_range=[round(d_lo, 1), round(d_hi, 1)], flight_time_s=round(fl_ref["time"], 2),
                    impact_angle_deg=round(fl_ref["impact_deg"], 1), apex_above_takeoff=round(fl_ref["apex"], 2), required_speed_fraction=round(float(v_req), 3),
                    z=dict(approach=round(zA, 2), lip=round(z_lip, 2), land0=round(z_L0, 2), land1=round(z_L1, 2), merge=round(zB, 2)), risk=d.get("risk", ""),
                    reward=d.get("reward", ""), safe_route=d.get("safe", "main loop around the obstacle"), gap_kind=d.get("gap_kind", "gap"))

    # ------------------------------------------------------------------ routes
    def _cycle_replace(self, cycle: list[str], frm: str, to: str, via: list[str]) -> list[str]:
        E = self.E
        starts = [i for i, e in enumerate(cycle) if E[e]["from"] == frm]
        if not starts:
            raise AuthorError(f"route: no edge leaves {frm}")
        i0 = starts[0]
        rot = cycle[i0:] + cycle[:i0]
        for k, e in enumerate(rot):
            if E[e]["to"] == to:
                return via + rot[k + 1:]
        raise AuthorError(f"route: {to} not reachable from {frm} along the cycle")

    def _route_poly(self, ids: list[str]):
        pts, ws, gs = [], [], []
        for eid in ids:
            e = self.E[eid]
            c = np.array(e["centerline"])
            pts.append(c[:-1])
            ws.append(np.array(e["widths"])[:-1])
            gs.append(np.array(e["grip"])[:-1])
        return np.vstack(pts), np.concatenate(ws), np.concatenate(gs)

    def _routes(self, pit, short, jumps):
        main = list(self.main_ids)
        opts = {"main_lap": main}
        opts["pit_stop_lap"] = self._cycle_replace(main, pit["entry_node"], pit["exit_node"], ["pit_entry", "pit_lane", "pit_exit"])
        sc = {}
        for s in short:
            sc[s["id"]] = (s["fork_node"], s["merge_node"], [s["edge"]])
            opts[f"{s['id']}_lap"] = self._cycle_replace(main, s["fork_node"], s["merge_node"], [s["edge"]])
        for j in jumps:
            sc[j["id"]] = (j["fork_node"], j["merge_node"], j["edges"])
            opts[f"{j['id']}_lap"] = self._cycle_replace(main, j["fork_node"], j["merge_node"], j["edges"])
        allr = list(main)
        for k, (a, b, via) in sc.items():
            allr = self._cycle_replace(allr, a, b, via)
        opts["all_options_lap"] = allr
        laps = {}
        for name, ids in opts.items():
            xyz, w, g = self._route_poly(ids)
            pr = D.speed_profile(xyz, w, True, grip=g)
            laps[name] = dict(time_s=round(pr["lap_time"], 2), length_m=round(pr["length"], 1), vmax=round(float(pr["v"].max()), 1), vmin=round(float(pr["v"].min()), 1))
        self._last_routes = opts
        return {k: v for k, v in opts.items()}, laps

    def route_xyz(self, name: str):
        return self._route_poly(self._last_routes[name])

    # ------------------------------------------------------------------ grid / sections / overtaking / stands / traps
    def _grid(self):
        slots, k = [], 0
        for row in range(self.boxes // 2):
            for col in (0, 1):
                x = -14.0 - 9.0 * row - (4.5 if col else 0.0)
                wst = float(self.wr[self.idx(self.L - 20)])
                y = (wst / 4.0) * (1 if col == 0 else -1)
                k += 1
                slots.append(dict(pos=k, xy=[round(x, 2), round(y, 2)], heading_deg=0.0))
        return slots

    def _sections(self):
        keys = [(k, self.S(a)) for k, a in self.sections]
        out = []
        for i, (k, s0) in enumerate(keys):
            s1 = keys[(i + 1) % len(keys)][1]
            if s1 <= s0:
                s1 += self.L
            a, b = int(s0 / self.ds_ring), int(s1 / self.ds_ring)
            idxs = np.arange(a, b + 1) % self.n
            v = self.prof["v"][idxs]
            z = self.zr[idxs]
            k_ = np.abs(self.prof["kappa"][idxs])
            r = 1.0 / np.maximum(k_, 1e-6)
            corner_share = float(np.mean(r < 90))
            drift_share = float(np.mean(self.prof["vmax"][idxs] < 0.78 * 38.0))
            out.append(dict(key=k, s0=round(s0 % self.L, 1), s1=round(s1 % self.L, 1), length=round(s1 - s0, 1), time_s=round(float(np.sum(self.ds_ring / np.maximum(v, 1))), 1),
                            v_min=round(float(v.min()), 1), v_max=round(float(v.max()), 1), v_mean=round(float(v.mean()), 1), z_min=round(float(z.min()), 1),
                            z_max=round(float(z.max()), 1), corner_share=round(corner_share, 2), drift_share=round(drift_share, 2),
                            start_xy=_r(self.ring[idxs[0]], 1)))
        return out

    def _overtaking(self):
        v, s = self.prof["v"], self.s_ring
        dv = np.diff(np.concatenate([v, v[:1]]))
        brk = dv < -0.04
        zones, i = [], 0
        n = self.n
        while i < n:
            if brk[i]:
                j = i
                while j + 1 < n and brk[j + 1]:
                    j += 1
                drop = float(v[i] - v[(j + 1) % n])
                zones.append(dict(i0=i, i1=j, drop=drop, v_in=float(v[i]), v_out=float(v[(j + 1) % n]), length=float((j - i + 1) * self.ds_ring), w=float(self.wr[i])))
                i = j + 1
            else:
                i += 1
        forbid = []
        for z in self.zones:
            if z["kind"] in ("tunnel", "glass_tunnel"):
                forbid.append((z["s0"], z["s1"]))
        good = [z for z in zones if z["drop"] >= 5.0 and z["v_in"] >= 26 and z["w"] >= 13.0]
        good.sort(key=lambda z: -z["drop"] * (1 + 0.02 * z["w"]))
        picked = []
        for z in good:
            si = z["i0"] * self.ds_ring
            if all(min(abs(si - p["s"]), self.L - abs(si - p["s"])) > 160 for p in picked):
                picked.append(dict(s=round(si, 1), xy=_r(self.ring[z["i0"]], 1), v_in=round(z["v_in"], 1), v_out=round(z["v_out"], 1), braking_len=round(z["length"], 1),
                                   width=round(z["w"], 1), reason="long fast approach, wide road, hard braking"))
        return sorted(picked[:6], key=lambda p: p["s"])

    def _stands(self):
        out = []
        for d in self.stand_defs:
            s_mid = self.S(d["a"]) if "b" not in d else 0.5 * (self.S(d["a"]) + self.S(d["b"]))
            i = self.idx(s_mid)
            p, h = self.ring[i], self.hh[i]
            nrm = np.array([-h[1], h[0]]) * d["side"]
            length = d.get("length") or (self.S(d["b"]) - self.S(d["a"]))
            off = self.wr[i] / 2 + d["offset"] + d["depth"] / 2
            c = p + nrm * off
            out.append(dict(id=d["id"], center=_r(c, 2), size=[round(length, 1), d["depth"]], heading_deg=round(hdeg(h), 1), height=d.get("height", 9.0),
                            rows=d.get("rows", 8), side=d["side"], capacity=d.get("capacity", int(length * d.get("rows", 8) / 0.55)), kind=d.get("kind", "grandstand"),
                            facing=d.get("facing", "road"), s=round(s_mid, 1), label=d.get("label", d["id"])))
        return out

    # ---- helper for traps: offset polylines along the ring
    def band(self, a, b, d0, d1, step=1):
        """Polygon of the road band between lateral offsets d0..d1 (left positive) from station a to b."""
        s0, s1 = self.S(a), self.S(b)
        if s1 < s0:
            s1 += self.L
        ii = np.arange(int(s0 / self.ds_ring), int(s1 / self.ds_ring) + 1, step) % self.n
        p, h = self.ring[ii], self.hh[ii]
        nrm = np.column_stack([-h[:, 1], h[:, 0]])
        left = p + nrm * d0
        right = p + nrm * d1
        poly = Polygon(np.vstack([left, right[::-1]]))
        return poly if poly.is_valid else poly.buffer(0)

    def _resolve_trap(self, t):
        s0, s1 = self.S(t["a"]), self.S(t["b"])
        if s1 < s0:
            s1 += self.L
        i0 = int(s0 / self.ds_ring) % self.n
        i1 = int(s1 / self.ds_ring) % self.n
        pre = np.arange(int(s0 / self.ds_ring) - int(40 / self.ds_ring), int(s0 / self.ds_ring) + 1) % self.n
        v_here = float(np.min(self.prof["v"][pre]))
        out = dict(t)
        out["a"], out["b"] = list(t["a"]), list(t["b"])
        out["s0"], out["s1"] = round(s0 % self.L, 1), round(s1 % self.L, 1)
        out["xy_start"], out["xy_end"] = _r(self.ring[i0], 1), _r(self.ring[i1], 1)
        out["width_at"] = round(float(self.wr[i0]) if t["on"] == "main" else float(self.w_short), 1)
        out["approach_speed_mps"] = round(v_here, 1)
        out["z"] = round(float(self.zr[i0]), 1)
        out["stations"] = []
        for st in t["stations"]:
            sm = (s0 + st["off"]) % self.L
            i = self.idx(sm)
            h = self.hh[i]
            ps = []
            for bd in st["bands"]:
                ln = bd.get("len", 6.0)
                ia = np.arange(int((sm - ln / 2) / self.ds_ring), int((sm + ln / 2) / self.ds_ring) + 2) % self.n
                p_, h_ = self.ring[ia], self.hh[ia]
                nrm = np.column_stack([-h_[:, 1], h_[:, 0]])
                poly = Polygon(np.vstack([p_ + nrm * bd["d0"], (p_ + nrm * bd["d1"])[::-1]]))
                poly = poly if poly.is_valid else poly.buffer(0)
                ps.append(dict(bd, polygon=_r(np.array(poly.exterior.coords), 1)))
            out["stations"].append(dict(off=st["off"], s=round(sm, 1), xy=_r(self.ring[i], 2), z=round(float(self.zr[i]), 2), heading_deg=round(hdeg(h), 1), bands=ps))
        return out

    # ------------------------------------------------------------------ misc helpers for authors / blender
    def xy_at(self, mark_frac, d=0.0):
        i = self.idx(mark_frac)
        h = self.hh[i]
        return self.ring[i] + np.array([-h[1], h[0]]) * d
