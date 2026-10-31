"""Blueprint renderers (PIL): every picture is drawn from the exported layout data, never from a render."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from . import geom as G
from .mapkit import ROAD, SECTION_COL, SECTION_LABEL, Map, bbox_of, font, hexrgb

BG = dict(coastal="#bfe3f0", city="#dfe3e8", industrial="#d9dcdf", canal="#d5dbe6", disco="#e6dfe9")
STAND = "#b4766a"
INK = "#1b2430"


def _sf_w(layout):
    e = next(x for x in layout["edges"] if x["id"] == layout["main_loop"][0])
    return float(e["widths"][0])


def _edge_poly(e):
    c = np.array(e["centerline"])
    return G.var_strip(c[:, :2], np.array(e["widths"]))


def terrain_polys(layout):
    """land (island styles) and water bodies from the authored areas; `None` land means the ground fills the whole plan."""
    t = layout.get("terrain", {})
    water = [Polygon(a["poly"]) for a in layout.get("areas", []) if a["kind"] == "water"]
    sunk = [Polygon(a["poly"]) for a in layout.get("areas", []) if a["kind"] in ("sunken", "pit", "trench", "stage_pit")]
    other = [(a["kind"], Polygon(a["poly"])) for a in layout.get("areas", []) if a["kind"] not in ("water", "sunken", "pit", "trench", "stage_pit")]
    land = None
    if t.get("style") == "island":
        roads = unary_union([_edge_poly(e) for e in layout["edges"] if e.get("polygon", True)])
        ring = Polygon(np.vstack([np.array(layout["edges"][i]["centerline"])[:, :2] for i in range(len(layout["edges"])) if layout["edges"][i]["kind"] == "main"]))
        ring = ring.buffer(0) if not ring.is_valid else ring
        base = unary_union([ring, roads]).buffer(t.get("margin", 90.0))
        land = base.buffer(-24).buffer(14)
        if water:
            land = land.difference(unary_union(water))
    return land, water, sunk, other


def paint_world(m: Map, layout, detail=True, roads=True, dim=False, infield=True):
    land, water, sunk, other = terrain_polys(layout)
    style = layout.get("terrain", {}).get("style", "city")
    if land is not None:
        m.d.rectangle([0, 0, m.W * m.ss, m.H * m.ss], fill=hexrgb("#b9dff0"))
        m.polygon(land.buffer(10), fill=hexrgb("#f1e5c4"))        # beach rim
        m.polygon(land, fill=hexrgb("#bfe0a8"))
    else:
        m.d.rectangle([0, 0, m.W * m.ss, m.H * m.ss], fill=hexrgb("#e6eaee"))
    for w in water:
        m.polygon(w, fill=hexrgb("#b9dff0" if land is not None else "#9fd3ea"))
    for s in sunk:
        m.polygon(s, fill=hexrgb("#b9b0c9"), outline=hexrgb("#8e84a4"), width=1.2)
    for kind, p in other:
        col = dict(plaza="#efe6d0", park="#c9e2b5", rocks="#b7aa98", hill="#a8cf8e", industrial="#cfd2d6", building="#c4c9d1").get(kind, "#d8d8d8")
        m.polygon(p, fill=hexrgb(col), outline=hexrgb("#9aa0a8"), width=0.8)
    for k in (layout.get("keep_out", []) if infield else []):
        m.polygon(Polygon(k), fill=hexrgb("#d9c9e8", 55))
    if detail:
        for s in layout["stands"]:
            m.polygon(G.rect(*s["center"], *s["size"], s["heading_deg"]), fill=hexrgb(STAND), outline=hexrgb("#7c4c44"), width=1.0)
        for lm in layout["landmarks"]:
            if infield and lm.get("footprint") and lm.get("check", True) and lm["footprint"][0] > 6:
                col = dict(festival_stage="#7c5cd6", crowd_field="#e6c3da", vip_tower="#5b6d84").get(lm["type"], "#8aa0b5")
                m.polygon(G.rect(*lm["xy"], *lm["footprint"], lm.get("rot", 0.0)), fill=hexrgb(col, 230 if lm["type"] != "crowd_field" else 150), outline=hexrgb("#403060"), width=1.0)
        p = layout["pit"]
        b = p["building"]
        m.polygon(G.rect(*b["center"], *b["size"], b["heading_deg"]), fill=hexrgb("#6e6a8a"), outline=hexrgb("#403d5c"), width=1.0)
        for bx in p["boxes"]:
            m.polygon(G.rect(*bx["center"], *bx["size"], bx["heading_deg"]), fill=hexrgb("#d9d4ec"), outline=hexrgb("#403d5c"), width=0.8)
    if roads:
        E = sorted(layout["edges"], key=lambda e: float(np.mean(np.array(e["centerline"])[:, 2])))
        for e in E:
            if not e.get("polygon", True):
                continue
            poly = _edge_poly(e)
            if dim:
                m.polygon(poly, fill=hexrgb("#9aa3b2"))
                continue
            m.polygon(poly.buffer(1.4), fill=hexrgb("#ffffff", 235))
            m.polygon(poly, fill=hexrgb(ROAD.get(e["kind"], "#556")))


def start_line(m, layout):
    h = 0.0
    m.polygon(G.rect(0.0, 0.0, 3.0, _sf_w(layout)), fill=hexrgb("#ffffff"), outline=hexrgb("#000000"), width=1.0)
    for k in range(6):
        yy = -9 + 3.0 * k + 1.5
        m.polygon(G.rect(-0.75, yy, 1.5, 3.0), fill=hexrgb("#000000"))


def corners(layout):
    """Numbered corners of the main loop: contiguous stretches with |curvature| > 1/110 m and a turn of at least 35 degrees."""
    ring = np.vstack([np.array(layout["edges"][i]["centerline"])[:-1, :2] for i in range(len(layout["edges"])) if False]) if False else None
    loop = [e for e in layout["edges"] if e["id"] in layout["main_loop"]]
    loop.sort(key=lambda e: layout["main_loop"].index(e["id"]))
    P = np.vstack([np.array(e["centerline"])[:-1] for e in loop])
    xy = P[:, :2]
    k = G.signed_curvature(xy, True)
    k = np.convolve(np.pad(k, 5, mode="wrap"), np.ones(7) / 7, mode="valid")
    hot = np.abs(k) > 1 / 110.0
    out, i, n = [], 0, len(k)
    start = int(np.argmin(hot)) if not hot.all() else 0
    order = [(start + j) % n for j in range(n)]
    cur = []
    for idx in order + [None]:
        if idx is not None and hot[idx] and (not cur or np.sign(k[idx]) == np.sign(k[cur[-1]])):
            cur.append(idx)
            continue
        if len(cur) > 3:
            turn = abs(float(np.sum(k[cur]) * 2.0)) * 180 / math.pi
            if turn >= 35:
                mid = cur[len(cur) // 2]
                r = 1 / max(float(np.mean(np.abs(k[cur]))), 1e-6)
                out.append(dict(idx=mid, xy=xy[mid].tolist(), z=float(P[mid, 2]), turn=turn, radius=r, dir="L" if k[mid] > 0 else "R", s=mid * 2.0))
        cur = [idx] if idx is not None and hot[idx] else []
    out.sort(key=lambda c: c["s"])
    # renumber from S/F: ring index 0 = S/F
    return out


def _stars(n):
    return "★" * n + "☆" * (5 - n)


def _centerline(layout, ids):
    E = {e["id"]: e for e in layout["edges"]}
    return np.vstack([np.array(E[i]["centerline"])[:, :2] for i in ids])


def _main_xyz(layout):
    E = {e["id"]: e for e in layout["edges"]}
    return np.vstack([np.array(E[i]["centerline"])[:-1] for i in layout["main_loop"]])


def draw_clean(layout, path, W=2400, H=1600):
    m = Map(W, H, bbox_of(layout, 70), margin=(60, 60, 60, 60))
    paint_world(m, layout, detail=True)
    start_line(m, layout)
    m.img = m.img  # noqa
    m.done().save(path)


def _panel(m, x, y, w, h, fill="#ffffff", outline="#c9d1db"):
    m.d.rounded_rectangle([x * m.ss, y * m.ss, (x + w) * m.ss, (y + h) * m.ss], radius=10 * m.ss, fill=hexrgb(fill, 245), outline=hexrgb(outline), width=2 * m.ss)


def _profile_strip(m, layout, x, y, w, h):
    _panel(m, x, y, w, h)
    P = _main_xyz(layout)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])
    z = P[:, 2]
    zmin, zmax = min(0.0, float(z.min())), float(z.max()) + 2
    L = layout["length_m"]
    px = lambda ss: x + 48 + (w - 70) * ss / L
    py = lambda zz: y + h - 34 - (h - 56) * (zz - zmin) / max(zmax - zmin, 1)
    for sec in layout["sections"]:
        a, b = sec["s0"], sec["s0"] + sec["length"]
        for (aa, bb) in ((a, min(b, L)), (0, b - L)) if b > L else ((a, b),):
            m.d.rectangle([px(aa) * m.ss, (y + 28) * m.ss, px(bb) * m.ss, (y + h - 34) * m.ss], fill=hexrgb(SECTION_COL[sec["key"]], 38))
            if bb - aa > 90:
                m.text((px(aa) + 4, y + 8), SECTION_LABEL[sec["key"]], 12, "SemiBold", SECTION_COL[sec["key"]])
    pts = [(px(a_), py(b_)) for a_, b_ in zip(s, z)]
    m.d.polygon([(p[0] * m.ss, p[1] * m.ss) for p in pts] + [(px(s[-1]) * m.ss, py(zmin) * m.ss), (px(0) * m.ss, py(zmin) * m.ss)], fill=hexrgb("#7f8da3", 90))
    m.d.line([(p[0] * m.ss, p[1] * m.ss) for p in pts], fill=hexrgb("#2d3647"), width=3 * m.ss)
    for zz in np.arange(0, zmax, 5 if zmax < 40 else 10):
        m.d.line([(x + 44) * m.ss, py(zz) * m.ss, (x + w - 20) * m.ss, py(zz) * m.ss], fill=hexrgb("#c9d1db"), width=1 * m.ss) if False else None
        m.text((x + 40, py(zz) - 8), f"{zz:.0f}", 11, "Regular", "#6b778a", anchor="ra")
    m.text((x + 8, y + h - 22), "elevation z (m) along the main lap  –  s = 0 at start/finish", 12, "Regular", "#6b778a")
    for sc in layout["shortcuts"]:
        a, b = sc["main_span_s"]
        m.d.rectangle([px(a) * m.ss, (y + h - 30) * m.ss, px(b) * m.ss, (y + h - 24) * m.ss], fill=hexrgb(ROAD["shortcut"]))
    for j in layout["jumps"]:
        a, b = j["main_span_s"]
        m.d.rectangle([px(a) * m.ss, (y + h - 22) * m.ss, px(b) * m.ss, (y + h - 16) * m.ss], fill=hexrgb(ROAD["jump_approach"]))
    for tr in layout["traps"]:
        a, b = tr["s0"], tr["s1"] if tr["s1"] > tr["s0"] else tr["s1"] + L
        m.d.rectangle([px(a) * m.ss, (y + h - 14) * m.ss, px(min(b, L)) * m.ss, (y + h - 8) * m.ss], fill=hexrgb("#e0453a"))
    for v in range(0, int(L) + 1, 500):
        m.text((px(v), y + h - 6), f"{v}", 11, "Regular", "#6b778a", anchor="ma")


def draw_annotated(layout, val, path, W=2800, H=1800):
    pw = 700
    m = Map(W, H, bbox_of(layout, 60), margin=(30, 130, pw + 50, 330))
    paint_world(m, layout, detail=True)
    E = {e["id"]: e for e in layout["edges"]}
    # section stripes on the main loop
    P = _main_xyz(layout)
    sec = layout["sections"]
    L = layout["length_m"]
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))])
    for sc_ in sec:
        a, b = sc_["s0"], sc_["s0"] + sc_["length"]
        msk = ((cum >= a) & (cum < min(b, L + 1e9))) | ((cum < b - L) if b > L else np.zeros_like(cum, bool))
        idx = np.where(msk)[0]
        if len(idx):
            runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            for run in runs:
                if len(run) > 1:
                    m.line(P[run][:, :2], hexrgb(SECTION_COL[sc_["key"]]), 5.0)
    for e in layout["edges"]:
        if e["kind"] == "main":
            m.chevrons(np.array(e["centerline"]), (255, 255, 255, 235), 90, 5.0)
        elif e["kind"] in ("pit_lane", "shortcut"):
            m.chevrons(np.array(e["centerline"]), (255, 255, 255, 235), 60, 4.0)
        elif e["kind"].startswith("jump") and e["kind"] != "jump_flight":
            pass
        if e["kind"] == "jump_flight":
            m.line(np.array(e["centerline"])[:, :2], hexrgb("#c23ba3"), 3.0, dash=(8, 5))
    start_line(m, layout)
    # numbered corners
    cs = corners(layout)
    for k, c in enumerate(cs, 1):
        m.circle(c["xy"], 11, fill=hexrgb("#ffffff"), outline=hexrgb(INK), width=1.6)
        x, y = m.P(np.array(c["xy"]))
        m.text((x / m.ss, y / m.ss - 8), str(k), 14, "Bold", INK, anchor="ma")
    # junction labels
    nodes = {n["id"]: n for n in layout["nodes"]}
    avoid = unary_union([_edge_poly(e) for e in layout["edges"] if e.get("polygon", True)])
    def pix_avoid():
        xs = []
        for e in layout["edges"]:
            if e.get("polygon", True):
                p = _edge_poly(e)
                xs.append(Polygon(m.pts(np.array(p.exterior.coords))) if not p.is_empty else None)
        xs = [q.buffer(7) for q in xs if q is not None and not q.is_empty]
        return unary_union([Polygon([(a / m.ss, b / m.ss) for a, b in q.exterior.coords]) for q in xs if q.geom_type == "Polygon"]) if xs else None
    av = pix_avoid()
    pitn = layout["pit"]
    m.label(nodes["n_pit_entry"]["xy"], "PIT ENTRY", "#3d78c4", avoid=av, pref=(-70, 40))
    m.label(nodes["n_pit_exit"]["xy"], "PIT EXIT", "#3d78c4", avoid=av, pref=(70, 40))
    m.label([0.0, 0.0], "START / FINISH", INK, avoid=av, pref=(0, -70))
    for sc_ in layout["shortcuts"]:
        a, b = nodes[sc_["fork_node"]], nodes[sc_["merge_node"]]
        m.label(a["xy"], f"{sc_['name'].upper()} IN  (−{sc_['saving_s']:.1f} s)", "#d9730d", avoid=av, size=14)
        m.label(b["xy"], f"{sc_['id'].split('_')[-1].upper()} OUT", "#d9730d", avoid=av, size=13)
    for j in layout["jumps"]:
        m.label(j["takeoff"], f"{j['name'].upper()} – take-off, gap {j['gap']:.0f} m", "#c23ba3", avoid=av, size=14)
        m.label(j["landing_start"], "landing zone", "#1f9d68", avoid=av, size=13)
    for o in layout["overtaking"]:
        x, y = m.P(np.array(o["xy"]))
        tri = [(x, y - 10 * m.ss), (x - 9 * m.ss, y + 7 * m.ss), (x + 9 * m.ss, y + 7 * m.ss)]
        m.d.polygon(tri, fill=hexrgb("#1c6fe0"), outline=hexrgb("#ffffff"))
    for tr in layout["traps"]:
        for st in tr["stations"]:
            for bd in st["bands"]:
                pp = Polygon(bd["polygon"])
                m.polygon(pp, fill=hexrgb("#e0453a", 190), outline=hexrgb("#8f1d14"), width=1.0)
        if tr.get("static"):
            Pm = _main_xyz(layout)
            Ls = layout["length_m"]
            n_ = len(Pm)
            s1 = tr["s1"] if tr["s1"] > tr["s0"] else tr["s1"] + Ls
            ii = [int(((tr["s0"] + t_) % Ls) / (Ls / n_)) % n_ for t_ in np.linspace(0, s1 - tr["s0"], 30)]
            m.line(Pm[ii][:, :2], hexrgb("#5aa9e6", 170), 12.0)
        m.label(tr["stations"][0]["xy"] if tr["stations"] else tr["xy_start"], tr["name"].upper() + (" *" if tr["role"] == "signature" else ""), "#c0392b", avoid=av, size=14)
    for s in layout["stands"]:
        m.label(s["center"], f"{s['label']} ({s['capacity']})", "#7c4c44", avoid=av, size=12, weight="Medium")
    for lm in layout["landmarks"]:
        if lm["type"] in ("festival_stage", "vip_tower", "lighthouse", "arena", "tower", "reactor", "disco_arena", "landmark") or lm.get("show"):
            m.label(lm["xy"], lm["label"], "#403060", avoid=av, size=13, weight="Medium")
    # structural zones
    for z in layout.get("zones", []):
        if z["kind"] in ("tunnel", "glass_tunnel", "bridge", "viaduct", "flyover", "underpass", "pier") and z.get("label"):
            i = int(((z["s0"] + z["s1"]) / 2) / (L / len(P))) % len(P)
            m.label(P[i][:2], f"{z['label']} (z {P[i][2]:.0f} m)", "#2d3647", avoid=av, size=12, weight="Medium")
    m.scale_bar(40, H - 30 - 330 + 300 - 18 if False else 1450, 100)
    m.north(2020, 175)
    # title
    m.text((34, 18), layout["name"].upper(), 46, "Bold", INK)
    m.text((34, 74), f"{layout['genre']}   ·   difficulty {layout['difficulty']}/5   ·   {layout['direction']}   ·   {layout['length_m']:.0f} m   ·   {layout['bpm']} BPM",
           21, "Regular", "#44505f")
    ok = val["geometry_validated"]
    _panel(m, W - pw - 20, 18, pw, 100, fill="#e8f6ee" if ok else "#fdecea", outline="#2e9e63" if ok else "#d64541")
    m.text((W - pw - 4, 30), ("GEOMETRY VALIDATED" if ok else "GEOMETRY NOT VALID"), 26, "Bold", "#1c7c4a" if ok else "#b3261e")
    npass = sum(c["status"] == "PASS" for c in val["checks"])
    m.text((W - pw - 4, 66), f"{npass}/{val['total']} checks PASS · {val['critical']} critical · {val['warnings']} warnings", 17, "Regular", "#2d3647")
    m.text((W - pw - 4, 90), "static plan – model-based, not drive-tested", 14, "Regular", "#6b778a")
    # right panel: legend + movements + times
    x0 = W - pw - 20
    y0 = 138
    _panel(m, x0, y0, pw, 1700 - y0 - 5 + 0 if False else 1480 - y0)
    yy = y0 + 14
    m.text((x0 + 16, yy), "ROADS", 17, "Bold", INK); yy += 28
    for lab, col in (("Main circuit", ROAD["main"]), ("Pit lane / entry / exit", ROAD["pit_lane"]), ("Shortcut", ROAD["shortcut"]), ("Jump approach / exit", ROAD["jump_approach"]),
                     ("Jump ramp", ROAD["jump_ramp"]), ("Landing zone", ROAD["jump_landing"])):
        m.d.rectangle([(x0 + 18) * m.ss, yy * m.ss, (x0 + 52) * m.ss, (yy + 14) * m.ss], fill=hexrgb(col))
        m.text((x0 + 64, yy - 3), lab, 14, "Regular", INK); yy += 22
    m.d.line([(x0 + 18) * m.ss, (yy + 7) * m.ss, (x0 + 52) * m.ss, (yy + 7) * m.ss], fill=hexrgb("#c23ba3"), width=3 * m.ss)
    m.text((x0 + 64, yy - 3), "Jump flight (gap, no road)", 14, "Regular", INK); yy += 22
    m.d.rectangle([(x0 + 18) * m.ss, yy * m.ss, (x0 + 52) * m.ss, (yy + 14) * m.ss], fill=hexrgb("#e0453a", 190))
    m.text((x0 + 64, yy - 3), "Trap hazard band", 14, "Regular", INK); yy += 22
    tri = [((x0 + 35) * m.ss, yy * m.ss), ((x0 + 27) * m.ss, (yy + 15) * m.ss), ((x0 + 43) * m.ss, (yy + 15) * m.ss)]
    m.d.polygon(tri, fill=hexrgb("#1c6fe0"))
    m.text((x0 + 64, yy - 3), "Overtaking zone (braking)", 14, "Regular", INK); yy += 22
    m.d.ellipse([(x0 + 27) * m.ss, yy * m.ss, (x0 + 43) * m.ss, (yy + 16) * m.ss], fill=hexrgb("#ffffff"), outline=hexrgb(INK), width=2 * m.ss)
    m.text((x0 + 64, yy - 3), "Numbered corner (turn >= 35°)", 14, "Regular", INK); yy += 34
    m.text((x0 + 16, yy), "MOVEMENTS (lap order)", 17, "Bold", INK); yy += 28
    for s_ in sorted(layout["sections"], key=lambda s: ["intro", "verse", "build_up", "drop", "breakdown", "finale"].index(s["key"])):
        m.d.rectangle([(x0 + 18) * m.ss, yy * m.ss, (x0 + 34) * m.ss, (yy + 34) * m.ss], fill=hexrgb(SECTION_COL[s_["key"]]))
        m.text((x0 + 46, yy - 3), f"{SECTION_LABEL[s_['key']]}", 15, "Bold", INK)
        m.text((x0 + 46, yy + 15), f"{s_['length']:.0f} m · {s_['time_s']:.0f} s · {s_['v_min']:.0f}–{s_['v_max']:.0f} m/s · z {s_['z_min']:.0f}–{s_['z_max']:.0f} m", 12, "Regular", "#44505f"); yy += 44
    yy += 6
    m.text((x0 + 16, yy), "LAP TIME MODEL", 17, "Bold", INK); yy += 26
    lt = layout["lap_times"]
    for k, lab in (("main_lap", "Main route"), *[(f"{s['id']}_lap", s["name"]) for s in layout["shortcuts"]], *[(f"{j['id']}_lap", j["name"]) for j in layout["jumps"]], ("all_options_lap", "All options")):
        if k in lt:
            m.text((x0 + 18, yy), lab, 14, "Regular", INK)
            m.text((x0 + pw - 18, yy), f"{lt[k]['time_s']:.1f} s", 14, "Bold", INK, anchor="ra"); yy += 21
    yy += 8
    m.text((x0 + 16, yy), "assumed kart: 38 m/s top, 13 m/s² lateral, 9 m/s² engine,", 12, "Regular", "#6b778a"); yy += 17
    m.text((x0 + 16, yy), "22 m/s² brake. Design model, not a physics simulation.", 12, "Regular", "#6b778a")
    # bottom elevation profile
    _profile_strip(m, layout, 30, 1490 + 0, W - 60, 290)
    m.done().save(path)


def zoom_bbox(c, r, aspect):
    return (c[0] - r * aspect, c[1] - r, c[0] + r * aspect, c[1] + r)


def draw_pitlane(layout, path, W=2600, H=1000):
    p = layout["pit"]
    half = 245
    bb = (-half, p["lane_y"] - 62 if p["side"] < 0 else -40, half, 48 if p["side"] < 0 else p["lane_y"] + 62)
    m = Map(W, H, bb, margin=(30, 120, 30, 60))
    paint_world(m, layout, detail=True, infield=False)
    start_line(m, layout)
    E = {e["id"]: e for e in layout["edges"]}
    for k in ("pit_lane", "pit_entry", "pit_exit"):
        m.chevrons(np.array(E[k]["centerline"]), (255, 255, 255, 240), 36, 4.5, 2.2)
    m.chevrons(np.array(E["main_1"]["centerline"]), (255, 255, 255, 120), 60, 4.0)
    for g in layout["start_finish"]["grid"]:
        m.polygon(G.rect(*g["xy"], 4.2, 2.0), fill=hexrgb("#ffffff", 220), outline=hexrgb("#1b2430"), width=1.0)
        x, y = m.P(np.array(g["xy"]))
        m.text((x / m.ss, y / m.ss - 8), str(g["pos"]), 12, "Bold", INK, anchor="ma")
    for b in p["boxes"]:
        x, y = m.P(np.array(b["center"]))
        m.text((x / m.ss, y / m.ss - 12), str(b["index"]), 20, "Bold", "#403d5c", anchor="ma")
    for s in layout["stands"]:
        m.label(s["center"], s["label"], "#7c4c44", size=14, weight="Medium")
    nodes = {n["id"]: n for n in layout["nodes"]}
    m.label(nodes["n_pit_entry"]["xy"], "PIT ENTRY  (fork 20°)", "#3d78c4", size=17, pref=(0, -80))
    m.label(nodes["n_pit_exit"]["xy"], "PIT EXIT  (merge 20°)", "#3d78c4", size=17, pref=(0, -80))
    m.label([0, 0], "START / FINISH LINE", INK, size=17, pref=(0, -90))
    m.label([-45, 0], "12-slot grid (2 abreast, 9 m rows)", INK, size=15, weight="Medium", pref=(-120, -90))
    m.label(p["building"]["center"], f"Pit garage building – {p['building'].get('style', 'garage')}", "#403d5c", size=17, pref=(0, 60))
    # dimension annotations
    ly = p["lane_y"]
    def dim(x, y0, y1, text):
        a, b = m.P(np.array([x, y0])), m.P(np.array([x, y1]))
        m.d.line([a, b], fill=hexrgb("#ffb347"), width=4 * m.ss)
        m.text((a[0] / m.ss + 8, (a[1] + b[1]) / 2 / m.ss - 9), text, 14, "SemiBold", "#ffffff", stroke="#7a3d00", stroke_w=2)
    wm = _sf_w(layout)
    sd = p["side"]
    dim(-150, 0, sd * (wm / 2), f"{wm:.0f} m")
    dim(-150, sd * wm / 2, sd * (wm / 2 + 8), "8 m pit wall gap")
    dim(-150, sd * (wm / 2 + 8), sd * (wm / 2 + 8 + 9), "9 m lane")
    m.label([p["boxes"][0]["center"][0], p["boxes"][0]["center"][1]], "box pitch 13 m · garage 11 × 7.5 m", "#403d5c", size=14, pref=(-60, 80))
    m.label([p["lane_half"] - 30, ly], "pit lane limit 60 km/h · one way", "#3d78c4", size=15, pref=(60, 50))
    m.scale_bar(60, H - 60, 50)
    m.text((40, 22), f"{layout['name'].upper()} – PIT LANE DETAIL", 38, "Bold", INK)
    m.text((40, 74), f"12 pit boxes along the start straight · entry s = {p['entry_s']:.0f} m (before the line) · exit s = {p['exit_s']:.0f} m (after the line) · lane {2 * p['lane_half']:.0f} m · return route to the main circuit closed (pit-stop lap {layout['lap_times']['pit_stop_lap']['time_s']:.1f} s)",
           17, "Regular", "#44505f")
    m.done().save(path)


def _zoom_panel(layout, c, r, W, H, highlight=None, title="", lines=None):
    m = Map(W, H, zoom_bbox(c, r, W / H), margin=(10, 10, 10, 10))
    paint_world(m, layout, detail=True)
    for e in layout["edges"]:
        if e["kind"] in ("shortcut",) or e["kind"].startswith("jump"):
            if e["kind"] == "jump_flight":
                m.line(np.array(e["centerline"])[:, :2], hexrgb("#c23ba3"), 3.0, dash=(8, 5))
            else:
                m.chevrons(np.array(e["centerline"]), (255, 255, 255, 235), 45, 4.0)
        elif e["kind"] == "main":
            m.chevrons(np.array(e["centerline"]), (255, 255, 255, 170), 70, 4.0)
    return m


def draw_shortcuts_jumps(layout, path, W=2800, H=1900):
    nodes = {n["id"]: n for n in layout["nodes"]}
    feats = [("shortcut", s) for s in layout["shortcuts"]] + [("jump", j) for j in layout["jumps"]]
    TH = 900
    top = Map(1500, TH, bbox_of(layout, 40), margin=(20, 100, 20, 10))
    paint_world(top, layout, detail=False, infield=False)
    for e in layout["edges"]:
        if e["kind"] == "main":
            top.chevrons(np.array(e["centerline"]), (255, 255, 255, 190), 100, 5.0)
    for e in layout["edges"]:
        if e["kind"] == "jump_flight":
            top.line(np.array(e["centerline"])[:, :2], hexrgb("#c23ba3"), 3.5, dash=(8, 5))
    start_line(top, layout)
    for k, (kind, f) in enumerate(feats, 1):
        a, b = nodes[f["fork_node"]], nodes[f["merge_node"]]
        col = "#d9730d" if kind == "shortcut" else "#c23ba3"
        top.label(a["xy"], f"{k}  {f['name'].upper()}", col, size=17, avoid=None)
    top.text((30, 16), f"{layout['name'].upper()} \u2013 SHORTCUTS & JUMPS", 38, "Bold", INK)
    top.text((30, 66), "own geometry, marked fork and merge, safe main-circuit route for every jump", 17, "Regular", "#44505f")
    base = top.done()
    img = Image.new("RGB", (W, H), (245, 247, 250))
    img.paste(base, (0, 0))
    from PIL import ImageDraw as _ID
    dd = _ID.Draw(img)
    tx, ty0 = 1560, 40
    dd.text((tx, ty0), "ROUTE OPTIONS AND LAP-TIME MODEL", font=font(26, "Bold"), fill=(27, 36, 48)); ty0 += 52
    lt = layout["lap_times"]
    for k_, lab in (("main_lap", "Main route (reference)"), *[(f"{x['id']}_lap", x["name"]) for x in layout["shortcuts"]], *[(f"{x['id']}_lap", x["name"]) for x in layout["jumps"]], ("all_options_lap", "All options combined"), ("pit_stop_lap", "Pit-stop lap (transit at limit not modelled)")):
        if k_ in lt:
            dd.text((tx, ty0), lab, font=font(20, "Regular"), fill=(27, 36, 48))
            dd.text((W - 60, ty0), f"{lt[k_]['time_s']:.1f} s   ({lt[k_]['length_m']:.0f} m)", font=font(20, "SemiBold"), fill=(27, 36, 48), anchor="ra"); ty0 += 32
    ty0 += 22
    dd.text((tx, ty0), "WHY THE OPTIONS ARE FAIR", font=font(22, "Bold"), fill=(27, 36, 48)); ty0 += 40
    for ln in ("Every shortcut is a separate road with its own centreline: visible fork on a straight stretch, merge at an angle, no isolated pieces.",
               "Every jump is optional: the main circuit around the obstacle stays open and is only slower, never blocked.",
               "Landing zones end >= 30 m before the merge so a landing never feeds directly into traffic.",
               "Time gain is a lap-time model (38 m/s kart); risk is expressed as speed requirement, narrower width and lower grip.",
               "Flight check: ballistic model with ramp angle and height difference, 88-105 % of the reference take-off speed."):
        for l2 in _wrap(ln, 74):
            dd.text((tx, ty0), l2, font=font(18, "Regular"), fill=(60, 70, 85)); ty0 += 25
        ty0 += 6
    n = len(feats)
    cw = (W - 30 * (n + 1)) // max(n, 1)
    for k, (kind, f) in enumerate(feats):
        x0 = 30 + k * (cw + 30)
        a, b = np.array(nodes[f["fork_node"]]["xy"]), np.array(nodes[f["merge_node"]]["xy"])
        c = (a + b) / 2
        r = max(np.linalg.norm(b - a) * 0.62 + 40, 110)
        mh = 520
        zm = _zoom_panel(layout, c, r, cw, mh)
        col = "#d9730d" if kind == "shortcut" else "#c23ba3"
        zm.label(a.tolist(), "FORK", col, size=15, pref=(0, -50))
        zm.label(b.tolist(), "MERGE", col, size=15, pref=(0, -50))
        zi = zm.done()
        img.paste(zi, (x0, 940))
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.rectangle([x0, 940, x0 + cw, 940 + mh], outline=(120, 130, 145), width=3)
        d.text((x0 + 10, 946), f"{k + 1}  {f['name']}", font=font(26, "Bold"), fill=hexrgb(col), stroke_width=3, stroke_fill=(255, 255, 255))
        ty = 940 + mh + 10
        if kind == "shortcut":
            rows = [f"length {f['length']:.0f} m instead of {f['main_span_len']:.0f} m  (−{f['saving_m']:.0f} m)", f"lap time  −{f['saving_s']:.1f} s  (model)",
                    f"fork {f['fork_angle_deg']}° / merge {f['merge_angle_deg']}° · width {f['width']:.0f} m · grip {f['grip']:.2f} ({f['surface']})",
                    f"entrance: {f['entrance']}", f"exit: {f['exit']}", f"risk: {f['risk']}"]
            for ln in rows:
                for l2 in _wrap(ln, 62):
                    d.text((x0 + 8, ty), l2, font=font(17, "Regular"), fill=(27, 36, 48)); ty += 23
        else:
            rows = [f"gap {f['gap']:.0f} m · ramp {f['ramp_len']:.0f} m @ {f['ramp_deg']}° (+{f['ramp_rise']:.1f} m) · landing {f['landing_len']:.0f} m × {f['width'] + 4:.0f} m",
                    f"reference take-off speed {f['ref_speed_mps']:.1f} m/s · needs {f['required_speed_fraction'] * 100:.0f} % of it",
                    f"flight {f['flight_distance_range'][0]:.0f}–{f['flight_distance_range'][1]:.0f} m (88–105 % speed) · impact {f['impact_angle_deg']:.0f}°",
                    f"lap time −{f['saving_s']:.1f} s · safe route: {f['safe_route']} ({f['main_span_len']:.0f} m vs {f['length']:.0f} m)",
                    f"risk: {f['risk']}"]
            for ln in rows:
                for l2 in _wrap(ln, 62):
                    d.text((x0 + 8, ty), l2, font=font(17, "Regular"), fill=(27, 36, 48)); ty += 23
            _jump_profile(img, layout, f, x0, ty + 6, cw, H - ty - 16)
    img.save(path)


def _wrap(s, n):
    words, out, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            out.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        out.append(cur)
    return out or [""]


def _jump_profile(img, layout, j, x, y, w, h):
    """Side view along the jump: road z, ramp, ballistic flight for 88/100/105 % speed, landing zone."""
    from PIL import ImageDraw
    from . import drive as D
    E = {e["id"]: e for e in layout["edges"]}
    ids = j["edges"]
    pts = []
    s0 = 0.0
    for eid in ids:
        c = np.array(E[eid]["centerline"])
        seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))]) + s0
        pts.append(np.column_stack([seg, c[:, 2]]))
        s0 = seg[-1]
    P = np.vstack(pts)
    d = ImageDraw.Draw(img, "RGBA")
    h = max(h, 150)
    zmin, zmax = P[:, 1].min() - 5, P[:, 1].max() + 4
    sx = lambda s: x + 12 + (w - 24) * s / P[-1, 0]
    sy = lambda z: y + h - 14 - (h - 28) * (z - zmin) / (zmax - zmin)
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255, 255), outline=(120, 130, 145, 255), width=2)
    # ground under the gap
    lip = P[[i for i, p in enumerate(P)][0:0] or 0]
    d.line([(sx(a), sy(b)) for a, b in P], fill=(194, 59, 163, 255), width=5)
    lipi = next(i for i, eid in enumerate(ids) if eid.endswith("_ramp"))
    s_lip = sum(E[e]["length"] for e in ids[:lipi + 1])
    z_lip = float(np.array(E[ids[lipi]]["centerline"])[-1][2])
    for f_, col in ((0.88, (230, 120, 20, 230)), (1.0, (30, 140, 90, 255)), (1.05, (200, 50, 50, 230))):
        v = f_ * j["ref_speed_mps"]
        a = math.radians(j["ramp_deg"])
        xs = np.linspace(0, 70, 60)
        zs = z_lip + xs * math.tan(a) - 9.81 * xs ** 2 / (2 * (v * math.cos(a)) ** 2)
        pl = [(sx(s_lip + xx), sy(zz)) for xx, zz in zip(xs, zs) if zz > zmin - 3]
        d.line(pl, fill=col, width=3)
    d.text((x + 8, y + 4), "side view: flight at 88 % / 100 % / 105 % of reference speed", font=font(15, "Regular"), fill=(60, 70, 85))
    ls = sum(E[e]["length"] for e in ids[:next(i for i, eid in enumerate(ids) if eid.endswith("_landing"))])
    d.rectangle([sx(ls), sy(zmin + 2), sx(ls + j["landing_len"]), sy(zmin) + 2], fill=(47, 179, 122, 200))
    d.text((sx(ls) + 4, sy(zmin) - 18), "landing zone", font=font(14, "SemiBold"), fill=(20, 100, 70))


def draw_traps(layout, val, path, W=2800, H=1800):
    m = Map(W, 1200, bbox_of(layout, 50), margin=(30, 120, 30, 20))
    paint_world(m, layout, detail=False)
    for e in layout["edges"]:
        if e["kind"] == "main":
            m.chevrons(np.array(e["centerline"]), (255, 255, 255, 170), 100, 5.0)
    start_line(m, layout)
    E = {e["id"]: e for e in layout["edges"]}
    L = layout["length_m"]
    P = _main_xyz(layout)
    for k, tr in enumerate(layout["traps"], 1):
        col = "#c0392b" if tr["role"] == "signature" else "#d98a1d"
        s0, s1 = tr["s0"], tr["s1"] if tr["s1"] > tr["s0"] else tr["s1"] + L
        sw = s0 - tr["warn_lead_m"]
        n = len(P)
        idxw = [int(((sw + t) % L) / (L / n)) % n for t in np.linspace(0, tr["warn_lead_m"], 12)]
        m.line(P[idxw][:, :2], hexrgb("#f2b01e"), 7.0)
        idxz = [int(((s0 + t) % L) / (L / n)) % n for t in np.linspace(0, s1 - s0, 30)]
        m.line(P[idxz][:, :2], hexrgb(col, 120), 22.0)
        for st in tr["stations"]:
            for bd in st["bands"]:
                m.polygon(Polygon(bd["polygon"]), fill=hexrgb("#e0453a", 230), outline=hexrgb("#7f1810"), width=1.0)
        if tr.get("static"):
            m.line(P[idxz][:, :2], hexrgb("#4f9fe0", 200), 26.0)
        m.label(tr["stations"][0]["xy"] if tr["stations"] else tr["xy_start"], f"{k}  {tr['name'].upper()}  ({'signature' if tr['role'] == 'signature' else 'secondary'})", col, size=18)
        m.label(P[idxw[0]][:2], f"warning visible from {tr['warn_lead_m']:.0f} m", "#a8730b", size=14, pref=(0, 70))
    m.text((34, 18), f"{layout['name'].upper()} – TRAP POSITIONS", 38, "Bold", INK)
    m.text((34, 70), "amber = visual warning lead distance · red bands = hazard zones (active phases) · the opposite carriageway side is always free", 18, "Regular", "#44505f")
    base = m.done()
    img = Image.new("RGB", (W, H), (245, 247, 250))
    img.paste(base, (0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img, "RGBA")
    nt = len(layout["traps"])
    cw = (W - 30 * (nt + 1)) // nt
    for k, tr in enumerate(layout["traps"]):
        x0 = 30 + k * (cw + 30)
        y0 = 1220
        d.rectangle([x0, y0, x0 + cw, H - 20], fill=(255, 255, 255, 255), outline=(120, 130, 145, 255), width=2)
        d.text((x0 + 12, y0 + 8), f"{k + 1}  {tr['name']}", font=font(26, "Bold"), fill=(27, 36, 48))
        # schedule chart for station 1 (+ phase of all bands)
        cyc = tr["cycle"] or 1.0
        rows = [(st["off"], b) for st in tr["stations"] for b in st["bands"]]
        if tr.get("static"):
            d.text((x0 + 12, y0 + 70), f"static surface hazard: grip {tr['grip']:.2f} (hard {tr['grip_hard']:.2f}), no timing", font=font(20, "SemiBold"), fill=(27, 36, 48))
            rows = []
        gx, gy, gw = x0 + 140, y0 + 56, cw - 170
        rh = min(26, 230 // max(len(rows), 1))
        for r_i, (off, b) in enumerate(rows):
            yy = gy + r_i * (rh + 4)
            d.text((x0 + 12, yy + 2), f"{b['id']}  d {b['d0']:+.1f}..{b['d1']:+.1f}", font=font(13, "Regular"), fill=(60, 70, 85))
            d.rectangle([gx, yy, gx + gw, yy + rh], fill=(235, 238, 243, 255))
            for (a_, b_, col) in ((b["warn"][0], b["warn"][1], (242, 176, 30, 255)), (b["on"][0], b["on"][1], (224, 69, 58, 255))):
                for sh in (0, -cyc, cyc):
                    aa, bb = a_ + b.get("phase", 0) + sh, b_ + b.get("phase", 0) + sh
                    aa, bb = max(aa, 0), min(bb, cyc)
                    if bb > aa:
                        d.rectangle([gx + gw * aa / cyc, yy, gx + gw * bb / cyc, yy + rh], fill=col)
        yy = gy + len(rows) * (rh + 4) + 4
        d.text((gx, yy), f"cycle {cyc:.1f} s   amber = warning   red = active", font=font(14, "Regular"), fill=(60, 70, 85))
        fr = tr["fairness"]["normal"]
        if tr.get("static"):
            yy = y0 + 110
        ty = yy + 28
        for ln in (f"fair corridor >= {fr['min_corridor_m']:.1f} m (easy {tr['fairness']['easy']['min_corridor_m']:.1f} / hard {tr['fairness']['hard']['min_corridor_m']:.1f}); road {tr['width_at']:.0f} m wide",
                   f"{fr['paths_tested']} pass-through tests over phase × speed: {fr['paths_failed']} failed · duty {fr['max_duty'] * 100:.0f} %",
                   f"warning {fr['min_warn_s']:.1f} s phase · visible {tr['warn_lead_m']:.0f} m ({tr['warn_lead_s']:.1f} s at {tr['approach_speed_mps']:.0f} m/s)",
                   "safe line: " + tr["safe_evasion"]):
            for l2 in _wrap(ln, 76):
                d.text((x0 + 12, ty), l2, font=font(16, "Regular"), fill=(27, 36, 48)); ty += 21
    img.save(path)


def thumb(layout, W, H, margin=24, dim=False):
    m = Map(W, H, bbox_of(layout, 30), margin=(margin,) * 4, bg="#eef2f5")
    paint_world(m, layout, detail=False)
    start_line(m, layout)
    return m.done()
