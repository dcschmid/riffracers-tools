"""Geometric / topological / kinematic validation of an exported track layout dict (never of an image).

Every check returns dict(id, name, status PASS|WARN|FAIL, critical, detail, data). `critical` = a geometry error that blocks release.
Heuristic checks (jump ballistics, lap times, trap schedules) are labelled `model` and never claim to be driving tests."""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import geom as G

MIN_WIDTH = dict(main=12.0, pit_lane=8.0, pit_entry=8.0, pit_exit=8.0, shortcut=9.0, jump_approach=10.0, jump_ramp=10.0, jump_landing=12.0,
                 jump_exit=10.0, jump_flight=10.0)
MIN_RADIUS = dict(main=22.0, shortcut=16.0, pit_lane=40.0, pit_entry=18.0, pit_exit=18.0, jump_approach=50.0, jump_ramp=80.0, jump_landing=80.0, jump_exit=40.0)
MAX_SLOPE = dict(main=0.13, shortcut=0.15, pit_lane=0.06, pit_entry=0.08, pit_exit=0.08, jump_approach=0.14, jump_ramp=0.26, jump_landing=0.16,
                 jump_exit=0.14, jump_flight=0.6)
MIN_CLEAR, LEVEL_CLEAR, PIT_CLEAR, EXEMPT_PATH = 8.0, 6.0, 2.0, 64.0
ROAD_KINDS_POLY = None


def _chk(cid, name, status, detail, critical=False, data=None, model=False):
    return dict(id=cid, name=name, status=status, critical=bool(critical and status == "FAIL"), detail=detail, data=data or {}, model=model)


def _head(c, i0, i1):
    v = np.array(c[i1][:2]) - np.array(c[i0][:2])
    return math.degrees(math.atan2(v[1], v[0]))


def _dang(a, b):
    return abs(((a - b + 180) % 360) - 180)


class Layout:
    def __init__(self, t: dict):
        self.t = t
        self.E = {e["id"]: e for e in t["edges"]}
        self.N = {n["id"]: n for n in t["nodes"]}
        self.C = {k: np.array(e["centerline"]) for k, e in self.E.items()}
        self.W = {k: np.array(e["widths"]) for k, e in self.E.items()}
        self.out, self.inn = {k: [] for k in self.N}, {k: [] for k in self.N}
        for e in t["edges"]:
            self.out[e["from"]].append(e["id"])
            self.inn[e["to"]].append(e["id"])

    def poly(self, eid):
        e = self.E[eid]
        if not e.get("polygon", True):
            return None
        return G.var_strip(self.C[eid][:, :2], self.W[eid])


def validate(t: dict) -> dict:
    L = Layout(t)
    checks = []
    add = checks.append
    E, N, C = L.E, L.N, L.C

    # 1 closed directed main cycle ------------------------------------------------------------------------------------
    loop = t["main_loop"]
    ok = all(E[a]["to"] == E[b]["from"] for a, b in zip(loop, loop[1:] + loop[:1])) and E[loop[0]]["from"] == "n_sf"
    gapxy = float(np.linalg.norm(C[loop[-1]][-1][:2] - C[loop[0]][0][:2]))
    tot = sum(E[e]["length"] for e in loop)
    add(_chk(1, "Hauptstrecke geschlossen (gerichteter Zyklus ueber S/Z)", "PASS" if ok and gapxy < 0.1 else "FAIL",
             f"{len(loop)} Kanten bilden einen Zyklus ab n_sf, Laenge {tot:.0f} m, Schlussfuge {gapxy:.3f} m", True))

    # 2 direction -----------------------------------------------------------------------------------------------------
    ring = np.vstack([C[e][:-1, :2] for e in loop])
    area = 0.5 * float(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1]))
    dirn = "CCW" if area > 0 else "CW"
    hd = np.degrees(np.arctan2(*(np.roll(ring, -1, axis=0) - ring).T[::-1]))
    turn = np.abs(((np.roll(hd, -1) - hd + 180) % 360) - 180)
    sf_h = _head(C[loop[0]], 0, 3)
    pit_dot = float(np.dot(C["pit_lane"][-1][:2] - C["pit_lane"][0][:2], [math.cos(math.radians(sf_h)), math.sin(math.radians(sf_h))]))
    okd = dirn == t["direction"] and turn.max() < 12 and pit_dot > 0 and _dang(sf_h, 0.0) < 1.0
    add(_chk(2, "Fahrtrichtung eindeutig (Umlaufsinn, Kursglaettung, Pitlane gleichgerichtet)", "PASS" if okd else "FAIL",
             f"Umlauf {dirn} (soll {t['direction']}), S/Z-Kurs {sf_h:.1f} deg, max. Kursaenderung {turn.max():.1f} deg je 2 m, Pitlane-Richtung {'gleich' if pit_dot > 0 else 'GEGEN'} Hauptrichtung", True))

    # 3 segment connections -------------------------------------------------------------------------------------------
    bad, notes = [], []
    for e in t["edges"]:
        c = C[e["id"]]
        a, b = N[e["from"]], N[e["to"]]
        if np.linalg.norm(c[0][:2] - a["xy"]) > 0.3 or np.linalg.norm(c[-1][:2] - b["xy"]) > 0.3 or abs(c[0][2] - a["z"]) > 0.2 or abs(c[-1][2] - b["z"]) > 0.2:
            bad.append(f"{e['id']}: Endpunkt weicht vom Knoten ab")
    for nid, n in N.items():
        i_, o_ = L.inn[nid], L.out[nid]
        if len(i_) == 1 and len(o_) == 1:
            hi = _head(C[i_[0]], -4, -1)
            ho = _head(C[o_[0]], 0, 3)
            if _dang(hi, ho) > 9 and not E[i_[0]].get("airborne") and not E[o_[0]].get("airborne"):
                bad.append(f"{nid}: Knick {_dang(hi, ho):.1f} deg zwischen {i_[0]} und {o_[0]}")
        elif len(o_) == 2:
            hs = [_head(C[x], 0, 3) for x in o_]
            ang = _dang(hs[0], hs[1])
            notes.append(f"{nid}: Abzweig {ang:.0f} deg")
            if ang > 42:
                bad.append(f"{nid}: Abzweigwinkel {ang:.0f} deg > 42")
        elif len(i_) == 2:
            hs = [_head(C[x], -4, -1) for x in i_]
            ang = _dang(hs[0], hs[1])
            notes.append(f"{nid}: Einfaedelwinkel {ang:.0f} deg")
            if ang > 42:
                bad.append(f"{nid}: Einfaedelwinkel {ang:.0f} deg > 42")
        elif len(i_) == 0 or len(o_) == 0:
            bad.append(f"{nid}: Sackgasse (in {len(i_)}, out {len(o_)})")
    add(_chk(3, "Segmentverbindungen: Endpunkte = Knoten, glatte Anschluesse, Abzweig-/Einfaedelwinkel <= 42 deg", "PASS" if not bad else "FAIL",
             "; ".join(bad) if bad else f"{len(t['edges'])} Kanten / {len(N)} Knoten konsistent ({'; '.join(notes)})", True, dict(issues=bad)))

    # 4 reachability / strong connectivity ----------------------------------------------------------------------------
    def reach(start, nxt):
        seen, st = {start}, [start]
        while st:
            u = st.pop()
            for v in nxt(u):
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        return seen
    fwd = reach("n_sf", lambda u: [E[e]["to"] for e in L.out[u]])
    bwd = reach("n_sf", lambda u: [E[e]["from"] for e in L.inn[u]])
    missing = [n for n in N if n not in fwd or n not in bwd]
    add(_chk(4, "Erreichbarkeit: jeder Knoten von S/Z erreichbar und fuehrt nach S/Z zurueck (stark zusammenhaengend)", "PASS" if not missing else "FAIL",
             f"{len(N)} Knoten, {len(t['edges'])} Kanten stark zusammenhaengend" if not missing else f"nicht erreichbar / ohne Rueckweg: {missing}", True))

    # 5 routes --------------------------------------------------------------------------------------------------------
    rbad = []
    for name, ids in t["routes"].items():
        for a, b in zip(ids, ids[1:] + ids[:1]):
            if E[a]["to"] != E[b]["from"]:
                rbad.append(f"{name}: {a} -> {b} nicht verbunden")
        if len(set(ids)) != len(ids):
            rbad.append(f"{name}: Kante doppelt befahren")
    laps = t["lap_times"]
    add(_chk(5, "Jede Route (Hauptrunde, Boxenrunde, jede Abkuerzung, jeder Sprung, alle Optionen) ist eine geschlossene Runde", "PASS" if not rbad else "FAIL",
             "; ".join(f"{k} {v['length_m']:.0f} m / {v['time_s']:.1f} s" for k, v in laps.items()) if not rbad else "; ".join(rbad), True))

    # 6 widths --------------------------------------------------------------------------------------------------------
    wbad, wsum = [], {}
    for e in t["edges"]:
        w = np.array(e["widths"])
        wsum.setdefault(e["kind"], []).append((float(w.min()), float(w.max())))
        if e["kind"] in MIN_WIDTH and w.min() < MIN_WIDTH[e["kind"]] - 1e-6:
            wbad.append(f"{e['id']}: {w.min():.1f} m < {MIN_WIDTH[e['kind']]}")
    grid_w = float(np.array(E[loop[-1]]["widths"])[-5:].min())
    if grid_w < 16:
        wbad.append(f"Startgerade {grid_w:.1f} m < 16 m (12 Karts, 2 nebeneinander)")
    add(_chk(6, "Fahrbahnbreiten ausreichend (Mindestbreite je Strassentyp, Startgerade >= 16 m)", "PASS" if not wbad else "FAIL",
             "; ".join(f"{k} {min(a for a, _ in v):.0f}-{max(b for _, b in v):.0f} m" for k, v in wsum.items()) if not wbad else "; ".join(wbad), True))

    # 7 radii ---------------------------------------------------------------------------------------------------------
    rbad, rinfo, tight = [], {}, []
    for e in t["edges"]:
        if e.get("airborne") or e["kind"] not in MIN_RADIUS:
            continue
        c = C[e["id"]][:, :2]
        if len(c) < 9:
            continue
        r = G.radii(c, win=3)[3:-3]
        mr = float(np.nanmin(r))
        k = e["kind"]
        rinfo[k] = min(rinfo.get(k, 1e9), mr)
        if mr < MIN_RADIUS[k] - 0.25:
            rbad.append(f"{e['id']}: min {mr:.1f} m < {MIN_RADIUS[k]}")
        j = int(np.argmin(r)) + 3
        tight.append((mr, e["id"], [round(float(c[j][0]), 1), round(float(c[j][1]), 1)]))
    tight.sort()
    add(_chk(7, "Kurvenradien plausibel (Mindestradius je Strassentyp)", "PASS" if not rbad else "FAIL",
             ("engster Radius je Typ: " + ", ".join(f"{k} {v:.1f} m" for k, v in rinfo.items())) if not rbad else "; ".join(rbad), True,
             dict(tightest=tight[:6])))

    # 8/9 planar overlap, grade separation, clearance (sample based, network-distance exemption) ----------------------
    X = _samples(L)
    ov, crossings, clr = _pair_checks(L, X)
    add(_chk(8, "Keine Strassenueberlappung auf gleichem Niveau; Brücken/Unterfuehrungen mit >= 6 m Hoehenabstand", "PASS" if not ov else "FAIL",
             (f"{len(crossings)} Kreuzung(en) mit Hoehentrennung: " + "; ".join(f"{c['over']} ueber {c['under']} (dz {c['dz']:.1f} m)" for c in crossings) if crossings else "keine Kreuzungen")
             if not ov else "; ".join(ov[:6]), True, dict(crossings=crossings, overlaps=ov)))
    add(_chk(9, "Sicherheitsabstaende: Strasse-Strasse >= 8 m (Pit >= 2.5 m) auf gleichem Niveau ausserhalb der Anschlusszonen", "PASS" if not clr["bad"] else "FAIL",
             f"kleinster Abstand Kante-Kante {clr['min']:.1f} m ({clr['where']})" if not clr["bad"] else "; ".join(clr["bad"][:6]), True, clr))

    # 10 height transitions -------------------------------------------------------------------------------------------
    sbad, smax = [], {}
    for e in t["edges"]:
        c = C[e["id"]]
        if len(c) < 3:
            continue
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
        sl = np.abs(np.gradient(c[:, 2], s))
        lim = MAX_SLOPE.get(e["kind"], 0.15)
        m = float(np.max(sl[1:-1]) if len(sl) > 3 else sl.max())
        smax[e["id"]] = m
        if m > lim:
            sbad.append(f"{e['id']}: {m * 100:.1f} % > {lim * 100:.0f} %")
    zr = [E[e]["centerline"] for e in loop]
    zall = np.concatenate([np.array(z)[:, 2] for z in zr])
    add(_chk(10, "Hoehenprofil: Steigung je Typ begrenzt, keine Hoehenspruenge an Knoten", "PASS" if not sbad else "FAIL",
             f"Hoehe {zall.min():.1f}-{zall.max():.1f} m, max. Steigung Hauptstrecke {max((v for k, v in smax.items() if E[k]['kind'] == 'main'), default=0) * 100:.1f} %" if not sbad else "; ".join(sbad), True,
             dict(z_min=float(zall.min()), z_max=float(zall.max()), slopes={k: round(v, 3) for k, v in smax.items()})))

    # 11 pit lane -----------------------------------------------------------------------------------------------------
    add(_pit_check(L, t))

    # 12 shortcuts ----------------------------------------------------------------------------------------------------
    add(_shortcut_check(L, t))

    # 13 jumps --------------------------------------------------------------------------------------------------------
    add(_jump_check(L, t))

    # 14 start grid ---------------------------------------------------------------------------------------------------
    mp = unary_union([L.poly(e) for e in loop])
    g = t["start_finish"]["grid"]
    inside = [Point(s["xy"]).buffer(1.6).within(mp) for s in g]
    rows = sorted({round(s["xy"][0], 1) for s in g})
    sp = min(np.diff(sorted({round(s["xy"][0] // 4.5 * 4.5, 1) for s in g}))) if len(rows) > 1 else 99
    gokay = len(g) == 12 and all(inside) and -g[0]["xy"][0] > 6
    add(_chk(14, "Start/Ziel: 12 Startplaetze auf der Fahrbahn der Geraden, vor der Linie, Pit-Ein-/Ausfahrt vor/hinter S/Z", "PASS" if gokay else "FAIL",
             f"{len(g)} Plaetze, {sum(inside)} im Strassenpolygon, {len({round(s['xy'][0], 1) for s in g})} Reihen/Staffeln, Pit-Einfahrt s={t['pit']['entry_s']:.0f} (vor S/Z), Ausfahrt s={t['pit']['exit_s']:.0f}", True))

    # 15 stands / landmarks / keep-out --------------------------------------------------------------------------------
    allroads = unary_union([L.poly(e["id"]) for e in t["edges"] if L.poly(e["id"]) is not None])
    sbad = []
    from .track import G as _G  # noqa
    for s in t["stands"]:
        r = _G.rect(*s["center"], *s["size"], s["heading_deg"])
        d = r.distance(allroads)
        if d < 8.0:
            sbad.append(f"{s['id']}: {d:.1f} m von der Strasse")
    for k in t.get("keep_out", []):
        if Polygon(k).intersects(allroads):
            sbad.append("Infield-Sperrflaeche schneidet eine Strasse")
    for lm in t["landmarks"]:
        if lm.get("footprint") and lm.get("check", True):
            fp = _G.rect(*lm["xy"], *lm["footprint"], lm.get("rot", 0.0))
            if fp.distance(allroads) < 6.0 and lm.get("z_ground", True):
                sbad.append(f"{lm['id']}: {fp.distance(allroads):.1f} m von der Strasse")
    cap = sum(s["capacity"] for s in t["stands"])
    add(_chk(15, "Tribuenen/Wahrzeichen >= 8 m (Landmarks >= 6 m) von jeder Strasse, Infield-Flaechen frei", "PASS" if not sbad else "FAIL",
             f"{len(t['stands'])} Tribuenen, Kapazitaet {cap}, {len(t['landmarks'])} Wahrzeichen/Bauwerke" if not sbad else "; ".join(sbad), True))

    # 16 lap time / dramaturgy / overtaking (model) -------------------------------------------------------------------
    lt = laps["main_lap"]["time_s"]
    secs = t["sections"]
    order = ["intro", "verse", "build_up", "drop", "breakdown", "finale"]
    keys = [s["key"] for s in secs]
    cover = sum(s["length"] for s in secs)
    okd = sorted(keys, key=order.index) == sorted(order, key=order.index) and abs(cover - t["length_m"]) < 6 and all(s["length"] > 40 for s in secs)
    add(_chk(16, "Dramaturgie: Intro/Verse/Build-up/Drop/Breakdown/Finale lueckenlos, jeder Abschnitt >= 40 m", "PASS" if okd else "FAIL",
             "; ".join(f"{s['key']} {s['length']:.0f} m {s['time_s']:.0f} s" for s in secs) + f"; Summe {cover:.0f} m", False))
    lo, hi = t.get("lap_target", (60, 95))
    st = "PASS" if lo <= lt <= hi else "WARN"
    add(_chk(17, "Rundenzeit im Zielbereich (Modell: Fahrmodell, KEIN Fahrtest)", st, f"Hauptroute {lt:.1f} s (Ziel {lo}-{hi} s), Optionen: " + ", ".join(
        f"{k.replace('_lap', '')} {v['time_s']:.1f} s" for k, v in laps.items() if k not in ("main_lap", "pit_stop_lap")), False, model=True))
    ot = t["overtaking"]
    add(_chk(18, "Ueberholmoeglichkeiten markiert (Bremszonen nach schnellen, breiten Passagen)", "PASS" if len(ot) >= 3 else "WARN",
             f"{len(ot)} Zonen: " + "; ".join(f"s={o['s']:.0f} ({o['v_in']:.0f}->{o['v_out']:.0f} m/s, {o['width']:.0f} m breit)" for o in ot), False, model=True))

    # 19 traps (fairness) ---------------------------------------------------------------------------------------------
    from .traps import check_traps
    add(check_traps(L, t))

    crit = [c for c in checks if c["critical"]]
    return dict(track=t["id"], checks=checks, critical=len(crit), warnings=sum(c["status"] == "WARN" for c in checks), fails=sum(c["status"] == "FAIL" for c in checks),
                total=len(checks), geometry_validated=not crit)


# ---------------------------------------------------------------------------------------------------------------------
def _samples(L: Layout, step=4.0):
    xy, z, w, ei, s, el, air = [], [], [], [], [], [], []
    names = list(L.E)
    for k, eid in enumerate(names):
        c = L.C[eid]
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1))])
        n = max(2, int(cum[-1] / step) + 1)
        tgt = np.linspace(0, cum[-1], n)
        for t_ in tgt:
            j = int(np.searchsorted(cum, t_))
            j = min(max(j, 1), len(c) - 1)
            f = (t_ - cum[j - 1]) / max(cum[j] - cum[j - 1], 1e-9)
            p = c[j - 1] + (c[j] - c[j - 1]) * f
            xy.append(p[:2]); z.append(p[2]); w.append(float(np.interp(t_, cum, L.W[eid])))
            ei.append(k); s.append(t_); el.append(cum[-1]); air.append(bool(L.E[eid].get("airborne")))
    return dict(xy=np.array(xy), z=np.array(z), w=np.array(w), ei=np.array(ei), s=np.array(s), el=np.array(el), air=np.array(air), names=names)


def _pair_checks(L: Layout, X):
    names = X["names"]
    nid = list(L.N)
    ni = {n: i for i, n in enumerate(nid)}
    D = np.full((len(nid), len(nid)), 1e9)
    np.fill_diagonal(D, 0)
    for e in L.E.values():
        a, b = ni[e["from"]], ni[e["to"]]
        D[a, b] = min(D[a, b], e["length"])
        D[b, a] = min(D[b, a], e["length"])
    for k in range(len(nid)):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    frm = np.array([ni[L.E[names[k]]["from"]] for k in X["ei"]])
    to = np.array([ni[L.E[names[k]]["to"]] for k in X["ei"]])
    s, el = X["s"], X["el"]
    N_ = len(s)
    pd = np.full((N_, N_), 1e9)
    for (ai, a_d) in ((frm, s), (to, el - s)):
        for (bi, b_d) in ((frm, s), (to, el - s)):
            pd = np.minimum(pd, a_d[:, None] + D[ai[:, None], bi[None, :]] + b_d[None, :])
    same = X["ei"][:, None] == X["ei"][None, :]
    pd = np.where(same, np.abs(s[:, None] - s[None, :]), pd)
    xy, z, w = X["xy"], X["z"], X["w"]
    dist = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    need = (w[:, None] + w[None, :]) / 2
    dz = np.abs(z[:, None] - z[None, :])
    kinds = np.array([L.E[names[k]]["kind"] for k in X["ei"]])
    pit = np.isin(kinds, ["pit_entry", "pit_exit", "pit_lane"])
    pitpair = pit[:, None] | pit[None, :]
    air = X["air"][:, None] | X["air"][None, :]
    # samples of the same jump chain are connected by construction
    jmp = np.array([L.E[names[k]].get("jump", "") for k in X["ei"]])
    samejump = (jmp[:, None] == jmp[None, :]) & (jmp[:, None] != "")
    # only pairs whose network path distance is large are checked
    zr_e = np.array([max(L.N[L.E[n_]["from"]].get("zone_radius", 30.0), L.N[L.E[n_]["to"]].get("zone_radius", 30.0)) for n_ in names])
    ex = 2.0 * np.maximum(zr_e[X["ei"]][:, None], zr_e[X["ei"]][None, :])
    ex = np.where(same, np.maximum(ex, 90.0), ex)   # same edge: samples closer than 90 m along the road are one carriageway
    chk = (pd > ex) & ~samejump
    iu = np.triu(np.ones((N_, N_), bool), 1)
    chk &= iu
    overlap = chk & (dist < need * 0.95) & (dz < LEVEL_CLEAR) & ~air
    air_hit = chk & (dist < need * 0.95) & (dz < 4.0) & air
    cross = chk & (dist < need * 0.95) & (dz >= LEVEL_CLEAR)
    mc = np.where(pitpair, PIT_CLEAR, MIN_CLEAR)
    gap = dist - need
    near = chk & (gap < mc) & (dz < LEVEL_CLEAR) & ~overlap & ~air
    ov = []
    for i, j in zip(*np.where(overlap | air_hit)):
        ov.append(f"{names[X['ei'][i]]}@{xy[i][0]:.0f},{xy[i][1]:.0f} x {names[X['ei'][j]]} (Abstand {dist[i, j]:.1f} m, dz {dz[i, j]:.1f} m)")
    ov = _dedupe(ov)
    crossings = {}
    for i, j in zip(*np.where(cross)):
        a, b = names[X["ei"][i]], names[X["ei"][j]]
        over, under = (a, b) if z[i] > z[j] else (b, a)
        key = (over, under)
        d_ = crossings.setdefault(key, dict(over=over, under=under, dz=float(dz[i, j]), xy=[round(float(xy[i][0]), 1), round(float(xy[i][1]), 1)]))
        d_["dz"] = min(d_["dz"], float(dz[i, j]))
    bad = []
    for i, j in zip(*np.where(near)):
        bad.append(f"{names[X['ei'][i]]} x {names[X['ei'][j]]}: Abstand Kante-Kante {gap[i, j]:.1f} m < {mc[i, j]:.1f} bei {xy[i][0]:.0f},{xy[i][1]:.0f} (dz {dz[i, j]:.1f})")
    bad = _dedupe(bad)
    g2 = np.where(chk & (dz < LEVEL_CLEAR) & ~air, gap, 1e9)
    mi = np.unravel_index(np.argmin(g2), g2.shape)
    clr = dict(bad=bad, min=float(g2[mi]) if g2[mi] < 1e8 else 99.0,
               where=f"{names[X['ei'][mi[0]]]} x {names[X['ei'][mi[1]]]}" if g2[mi] < 1e8 else "-")
    return ov, list(crossings.values()), clr


def _dedupe(lst):
    seen, out = set(), []
    for s in lst:
        k = s.split("@")[0].split(":")[0] + s.split(" x ")[1].split(" ")[0] if " x " in s else s
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ---------------------------------------------------------------------------------------------------------------------
def _pit_check(L: Layout, t: dict):
    p, E, C = t["pit"], L.E, L.C
    bad = []
    lane = C["pit_lane"][:, :2]
    lane_ls = LineString(lane)
    if len(p["boxes"]) != 12:
        bad.append(f"{len(p['boxes'])} Boxen statt 12")
    ydev = float(np.ptp(lane[:, 1]))
    if ydev > 0.05:
        bad.append(f"Boxengasse nicht parallel zur Geraden (y-Abweichung {ydev:.2f} m)")
    mainpoly = unary_union([L.poly(e) for e in t["main_loop"]])
    lanepoly = L.poly("pit_lane")
    d_main = lanepoly.distance(mainpoly)
    if d_main < 6.0:
        bad.append(f"Boxengasse nur {d_main:.1f} m von der Strecke (Boxenmauer fehlt)")
    far = 0.0
    for b in p["boxes"]:
        r = G.rect(*b["center"], *b["size"], b["heading_deg"])
        dl = r.distance(lanepoly)
        if dl > 1.2:
            bad.append(f"{b['id']}: {dl:.1f} m von der Boxengasse")
        if r.intersects(mainpoly):
            bad.append(f"{b['id']} ragt in die Strecke")
        far = max(far, dl)
        if lane_ls.distance(Point(b["lane_point"])) > 0.3:
            bad.append(f"{b['id']}: Gassenpunkt nicht auf der Gasse")
    bld = p["building"]
    br = G.rect(*bld["center"], *bld["size"], bld["heading_deg"])
    boxes_u = unary_union([G.rect(*b["center"], *b["size"]) for b in p["boxes"]])
    if br.distance(boxes_u) > 2.5:
        bad.append(f"Boxengebaeude {br.distance(boxes_u):.1f} m von den Boxen entfernt")
    ent = _head(C["pit_entry"], 0, 6)
    mn = _head(C[E["pit_entry"]["from"] and [e for e in t["main_loop"] if E[e]["to"] == "n_pit_entry"][0]], -4, -1)
    ang_in = _dang(ent, mn)
    ex = _head(C["pit_exit"], -6, -1)
    mx = _head(C[[e for e in t["main_loop"] if E[e]["from"] == "n_pit_exit"][0]], 0, 3)
    ang_out = _dang(ex, mx)
    if ang_in > 35 or ang_out > 35:
        bad.append(f"Pit-Anschlusswinkel {ang_in:.0f}/{ang_out:.0f} deg > 35")
    ok_ret = "pit_stop_lap" in t["routes"] and "n_sf" not in [E[x]["from"] for x in t["routes"]["pit_stop_lap"]]
    # entry before S/F (late in the lap), exit after S/F (early)
    if not (p["entry_s"] > t["length_m"] / 2 > p["exit_s"]):
        bad.append("Pit-Einfahrt muss vor, Pit-Ausfahrt hinter S/Z liegen")
    return _chk(11, "Pitlane: Einfahrt vor / Ausfahrt hinter S/Z, parallel, Einbahn, 12 Boxen anliegend, Gebaeude, Rueckweg zur Strecke", "PASS" if not bad else "FAIL",
                f"12 Boxen (Raster {p['box_spacing']} m), Gasse {lanepoly.area / 9:.0f} m lang, Abstand zur Strecke {d_main:.1f} m, Anschlusswinkel {ang_in:.0f}/{ang_out:.0f} deg, Rundenroute {'geschlossen' if ok_ret else 'FEHLT'}" if not bad else "; ".join(bad),
                True, dict(angle_in=ang_in, angle_out=ang_out, dist_to_main=d_main))


def _shortcut_check(L: Layout, t: dict):
    bad, rows = [], []
    sc = t["shortcuts"]
    if len(sc) != 2:
        bad.append(f"{len(sc)} Abkuerzungen statt 2")
    mainpoly = unary_union([L.poly(e) for e in t["main_loop"]])
    lap0 = t["lap_times"]["main_lap"]["time_s"]
    for s in sc:
        e = L.E[s["edge"]]
        c = L.C[s["edge"]][:, :2]
        poly = L.poly(s["edge"])
        zones = unary_union([Point(L.N[s["fork_node"]]["xy"]).buffer(L.N[s["fork_node"]]["zone_radius"] + 4), Point(L.N[s["merge_node"]]["xy"]).buffer(L.N[s["merge_node"]]["zone_radius"] + 4)])
        far = poly.difference(zones)
        own = (1 - poly.intersection(mainpoly.buffer(2.0)).area / poly.area)
        dt = lap0 - t["lap_times"][f"{s['id']}_lap"]["time_s"]
        s["saving_s"] = round(dt, 2)
        if not s["main_span_s"][0] < s["main_span_s"][1]:
            bad.append(f"{s['id']}: Reihenfolge Abzweig/Einfaedelung")
        if s["saving_m"] < 8:
            bad.append(f"{s['id']}: nur {s['saving_m']:.0f} m kuerzer")
        if dt < 0.8:
            bad.append(f"{s['id']}: Zeitgewinn {dt:.2f} s < 0.8 s (kein Risiko-Nutzen-Verhaeltnis)")
        if dt > 7.0:
            bad.append(f"{s['id']}: Zeitgewinn {dt:.1f} s > 7 s (zu stark, entwertet Hauptstrecke)")
        if not far.is_empty and far.distance(mainpoly) < MIN_CLEAR:
            bad.append(f"{s['id']}: Abstand zur Hauptstrecke {far.distance(mainpoly):.1f} m < {MIN_CLEAR}")
        if own < 0.4:
            bad.append(f"{s['id']}: nur {own * 100:.0f} % eigene Geometrie")
        # visible fork: straight-ish main road 40 m before the fork and after the merge
        rA = G.radii(L.C[[x for x in t["main_loop"] if L.E[x]["to"] == s["fork_node"]][0]][:, :2], win=3)[-25:-3]
        rB = G.radii(L.C[[x for x in t["main_loop"] if L.E[x]["from"] == s["merge_node"]][0]][:, :2], win=3)[3:25]
        if len(rA) and np.nanmin(rA) < 40:
            bad.append(f"{s['id']}: Abzweig in einer Kurve (R {np.nanmin(rA):.0f} m < 40 m vor dem Abzweig)")
        if len(rB) and np.nanmin(rB) < 40:
            bad.append(f"{s['id']}: Einfaedelung in einer Kurve (R {np.nanmin(rB):.0f} m < 40 m hinter dem Knoten)")
        rows.append(f"{s['id']}: {s['length']:.0f} m statt {s['main_span_len']:.0f} m, -{dt:.1f} s, Abzweig {s['fork_angle_deg']} deg / Einfaedeln {s['merge_angle_deg']} deg, {own * 100:.0f} % eigene Geometrie")
    return _chk(12, "2 Abkuerzungen: echte eigene Fahrwege mit Abzweig/Einfaedelung, kuerzer und schneller, kein Sprung in den Verkehr", "PASS" if not bad else "FAIL",
                "; ".join(rows) if not bad else "; ".join(bad), True, dict(rows=rows), model=True)


def _jump_check(L: Layout, t: dict):
    bad, rows = [], []
    js = t["jumps"]
    if not js:
        bad.append("kein optionaler Risikosprung")
    lap0 = t["lap_times"]["main_lap"]["time_s"]
    for j in js:
        dt = lap0 - t["lap_times"][f"{j['id']}_lap"]["time_s"]
        j["saving_s"] = round(dt, 2)
        hi_ok = j["flight_distance_range"][1] <= j["gap"] + j["landing_len"] - 10
        lo_req = j["required_speed_fraction"]
        surf = j["impact_angle_deg"] - j["landing_deg"]
        if j["ramp_deg"] > 14:
            bad.append(f"{j['id']}: Rampe {j['ramp_deg']} deg > 14")
        if not 0.60 <= lo_req <= 0.93:
            bad.append(f"{j['id']}: benoetigte Absprunggeschwindigkeit {lo_req * 100:.0f} % der Referenz (soll 60-93 %: schaffbar, aber nicht trivial)")
        if not hi_ok:
            bad.append(f"{j['id']}: bei 105 % Tempo {j['flight_distance_range'][1]:.0f} m Flugweite > Landezone ({j['gap'] + j['landing_len'] - 10:.0f} m)")
        if surf > 16:
            bad.append(f"{j['id']}: Aufprallwinkel zur Landefl. {surf:.0f} deg > 16")
        if j["landing_len"] < 36 or j["width"] + 4 < 15:
            bad.append(f"{j['id']}: Landezone {j['landing_len']:.0f} m x {j['width'] + 4:.0f} m zu klein")
        if j["exit_len"] < 30:
            bad.append(f"{j['id']}: Auslauf {j['exit_len']:.0f} m < 30 (Landung zu nah am Verkehr)")
        if j["apex_above_takeoff"] > 4.5:
            bad.append(f"{j['id']}: Flughoehe {j['apex_above_takeoff']:.1f} m zu hoch")
        if dt < 1.5:
            bad.append(f"{j['id']}: Zeitgewinn {dt:.1f} s < 1.5 s (Risiko nicht gerechtfertigt)")
        # safe alternative: the main loop between the nodes
        if f"{j['id']}_lap" not in t["routes"]:
            bad.append(f"{j['id']}: keine Alternativroute")
        rows.append(f"{j['id']} '{j['name']}': Luecke {j['gap']:.0f} m, v_ref {j['ref_speed_mps']:.1f} m/s, Rampe {j['ramp_deg']} deg, benoetigt {lo_req * 100:.0f} % Tempo, "
                    f"Flugweite {j['flight_distance_range'][0]:.0f}-{j['flight_distance_range'][1]:.0f} m (nominal {j['nominal_flight_distance']:.0f}), Landezone {j['landing_len']:.0f} m, -{dt:.1f} s, sichere Route +{-j['saving_m']:.0f} m" if j["saving_m"] < 0 else
                    f"{j['id']} '{j['name']}': Luecke {j['gap']:.0f} m, v_ref {j['ref_speed_mps']:.1f} m/s, benoetigt {lo_req * 100:.0f} % Tempo, Flugweite {j['flight_distance_range'][0]:.0f}-{j['flight_distance_range'][1]:.0f} m, Landezone {j['landing_len']:.0f} m, -{dt:.1f} s, Hauptstrecke {j['main_span_len']:.0f} m vs Sprung {j['length']:.0f} m")
    return _chk(13, "Risikosprung(e): kinematische Pruefung Absprung/Flug/Landezone, sichere Alternativroute (Modell, keine Physiksimulation)", "PASS" if not bad else "FAIL",
                "; ".join(rows) if not bad else "; ".join(bad), True, dict(rows=rows), model=True)
