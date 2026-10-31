"""Trap fairness model (data-level, not a game simulation).

A trap is a set of stations along the road; each station blocks lateral bands (d0..d1 from the centre line, left positive) in a
periodic schedule. For every arrival phase and a spread of arrival speeds a kart (width 2.2 m, lateral speed limit 6 m/s) must be
able to pass all stations. The same test runs for the easy / hard variants. Warnings must be visible long enough, no permanent
full blocking, traps keep away from fork/merge/landing/pit nodes."""
from __future__ import annotations

import math

import numpy as np

KART_W = 2.2
MARGIN = 0.6          # clearance to barrier / active band
V_LAT = 6.0           # m/s lateral speed limit used for reachability
GRID = 0.25
MIN_CORRIDOR = 3.6    # one kart with margins
GOOD_CORRIDOR = 6.4   # two karts abreast
MIN_WARN_S = 1.2


def _active(b, tau, cycle):
    a0, a1 = b["on"]
    x = (tau - b.get("phase", 0.0)) % cycle
    return a0 <= x < a1


def _warning(b, tau, cycle):
    w0, w1 = b.get("warn", (0.0, 0.0))
    x = (tau - b.get("phase", 0.0)) % cycle
    return w0 <= x < w1


def _free_centres(width, st_bands, tau, cycle, scale=1.0):
    n = int(round(width / GRID)) + 1
    x = np.linspace(-width / 2, width / 2, n)
    ok = (np.abs(x) <= width / 2 - MARGIN - KART_W / 2)
    for b in st_bands:
        if _active(b, tau, cycle):
            c, h = (b["d0"] + b["d1"]) / 2, (b["d1"] - b["d0"]) / 2 * scale
            ok &= np.abs(x - c) > h + KART_W / 2 + 0.2
    return x, ok


def _variant(trap, key):
    v = trap.get("variants", {}).get(key, {})
    return v.get("cycle_scale", 1.0), v.get("band_scale", 1.0), v.get("warn_scale", 1.0)


def analyse(trap: dict, width: float, v_app: float, variant="normal") -> dict:
    cyc_sc, band_sc, warn_sc = _variant(trap, variant) if variant != "normal" else (1.0, 1.0, 1.0)
    cycle = trap["cycle"] * cyc_sc
    sts = trap["stations"]
    dts = np.arange(0.0, cycle, 0.1)
    worst_corr, worst_two, fail_paths, total = 1e9, 1e9, 0, 0
    min_warn = 1e9
    duty = []
    for st in sts:
        for b in st["bands"]:
            on = (b["on"][1] - b["on"][0]) * cyc_sc
            duty.append(on / cycle)
            warn = (b.get("warn", (0, 0))[1] - b.get("warn", (0, 0))[0]) * warn_sc * cyc_sc
            min_warn = min(min_warn, warn)
    # single-station corridors (full time scan)
    scaled = []
    for st in sts:
        bb = []
        for b in st["bands"]:
            b2 = dict(b)
            b2["on"] = (b["on"][0] * cyc_sc, b["on"][1] * cyc_sc)
            b2["phase"] = b.get("phase", 0.0) * cyc_sc
            if "warn" in b:
                b2["warn"] = (b["warn"][0] * cyc_sc, b["warn"][0] * cyc_sc + (b["warn"][1] - b["warn"][0]) * warn_sc * cyc_sc)
            bb.append(b2)
        scaled.append(dict(st, bands=bb))
    for st in scaled:
        for tau in dts:
            x, ok = _free_centres(width, st["bands"], tau, cycle, band_sc)
            # longest run of feasible centres -> corridor width = run length + kart width
            best, run = 0.0, 0.0
            for k in range(len(ok)):
                run = run + GRID if ok[k] else 0.0
                best = max(best, run if ok[k] else 0.0)
            corr = (best + KART_W) if ok.any() else 0.0
            worst_corr = min(worst_corr, corr)
            worst_two = min(worst_two, corr)
    # path feasibility over all stations, arrival phase and speed spread
    for vf in (0.75, 0.9, 1.0, 1.08):
        v = max(v_app * vf, 8.0)
        for t0 in dts:
            reach = None
            prev_tau = None
            for st in scaled:
                tau = t0 + st["off"] / v
                x, ok = _free_centres(width, st["bands"], tau, cycle, band_sc)
                if reach is None:
                    reach = ok.copy()
                else:
                    dt = tau - prev_tau
                    r = int(math.ceil(V_LAT * dt / GRID))
                    idx = np.where(reach)[0]
                    d = np.zeros_like(reach)
                    for i in idx:
                        d[max(0, i - r):i + r + 1] = True
                    reach = d & ok
                prev_tau = tau
                if not reach.any():
                    break
            total += 1
            if reach is None or not reach.any():
                fail_paths += 1
    return dict(variant=variant, cycle_s=round(cycle, 2), min_corridor_m=round(worst_corr, 2), min_warn_s=round(min_warn, 2) if min_warn < 1e8 else 0.0,
                max_duty=round(max(duty), 2) if duty else 0.0, paths_tested=total, paths_failed=fail_paths)


def check_traps(L, t: dict) -> dict:
    from .validate import _chk
    traps = t["traps"]
    if not traps:
        return _chk(19, "Fallen (Fairness-Modell)", "FAIL", "keine Fallen definiert", True)
    bad, warn, rows = [], [], []
    node_pts = [(n["id"], np.array(n["xy"])) for n in t["nodes"] if n["type"] in ("shortcut_fork", "shortcut_merge", "jump_branch", "jump_merge", "jump_landing_start", "jump_landing_end",
                                                                                 "pit_entry", "pit_exit", "start_finish")]
    sig = [x for x in traps if x["role"] == "signature"]
    if len(sig) != 1:
        bad.append(f"{len(sig)} Signature-Fallen statt 1")
    if len(traps) < 2:
        bad.append("zweite Gefahr fehlt")
    for tr in traps:
        w = tr["width_at"]
        if tr.get("static"):
            L_z = tr["s1"] - tr["s0"] if tr["s1"] >= tr["s0"] else tr["s1"] + t["length_m"] - tr["s0"]
            grip = tr["grip"]
            lead_s = tr["warn_lead_m"] / max(tr["approach_speed_mps"], 1.0)
            tr["warn_lead_s"] = round(lead_s, 2)
            mid = np.array(tr["xy_start"])
            fake = dict(min_corridor_m=round(w - 1.2, 2), min_warn_s=round(lead_s, 2), max_duty=1.0, paths_tested=0, paths_failed=0, cycle_s=0.0, variant="static")
            tr["fairness"] = dict(normal=fake, easy=dict(fake, variant="easy"), hard=dict(fake, variant="hard"))
            if grip < 0.6:
                bad.append(f"{tr['id']}: Grip {grip} < 0.6 (unfahrbar glatt)")
            if L_z > 170:
                bad.append(f"{tr['id']}: Zone {L_z:.0f} m > 170 m")
            if lead_s < 1.4:
                bad.append(f"{tr['id']}: Warnung nur {lead_s:.1f} s")
            for nid, p in node_pts:
                if np.linalg.norm(p - mid) < 45.0 and tr["on"] == "main":
                    bad.append(f"{tr['id']}: nur {np.linalg.norm(p - mid):.0f} m vom Knoten {nid}")
            rows.append(f"{tr['name']} ({tr['role']}, statisch): Zone {L_z:.0f} m, Grip {grip:.2f} (hard {tr['grip_hard']:.2f}), Warnung ab {tr['warn_lead_m']:.0f} m ({lead_s:.1f} s), Fahrbahn durchgehend befahrbar ({w:.0f} m)")
            continue
        res = {k: analyse(tr, w, tr["approach_speed_mps"], k) for k in ("normal", "easy", "hard")}
        tr["fairness"] = res
        n = res["normal"]
        lead = tr["warn_lead_m"]
        t_lead = lead / max(tr["approach_speed_mps"], 1.0)
        tr["warn_lead_s"] = round(t_lead, 2)
        if n["min_corridor_m"] < MIN_CORRIDOR:
            bad.append(f"{tr['id']}: freier Korridor nur {n['min_corridor_m']:.1f} m < {MIN_CORRIDOR}")
        elif n["min_corridor_m"] < GOOD_CORRIDOR:
            warn.append(f"{tr['id']}: Korridor {n['min_corridor_m']:.1f} m < {GOOD_CORRIDOR} (keine zwei Karts nebeneinander)")
        if n["paths_failed"]:
            bad.append(f"{tr['id']}: {n['paths_failed']}/{n['paths_tested']} Durchfahrten unmoeglich (Phase/Tempo)")
        if res["hard"]["paths_failed"] or res["hard"]["min_corridor_m"] < MIN_CORRIDOR:
            bad.append(f"{tr['id']}: Variante hard nicht mehr fair (Korridor {res['hard']['min_corridor_m']:.1f} m, {res['hard']['paths_failed']} Fehlpfade)")
        if n["min_warn_s"] < MIN_WARN_S:
            bad.append(f"{tr['id']}: Warnphase {n['min_warn_s']:.1f} s < {MIN_WARN_S} s")
        if t_lead < 1.4:
            bad.append(f"{tr['id']}: Warnung nur {lead:.0f} m ({t_lead:.1f} s) vor der Zone sichtbar")
        if n["max_duty"] > 0.55:
            bad.append(f"{tr['id']}: Aktivzeit {n['max_duty'] * 100:.0f} % > 55 % (quasi Dauersperre)")
        if tr["on"] == "main" and not tr.get("visual_warning"):
            bad.append(f"{tr['id']}: kein rein visuelles Warnsignal definiert")
        mid = np.array(tr["xy_start"])
        for nid, p in node_pts:
            if np.linalg.norm(p - mid) < 45.0 and tr["on"] == "main":
                bad.append(f"{tr['id']}: nur {np.linalg.norm(p - mid):.0f} m vom Knoten {nid} (Verkehrsverdichtung)")
        rows.append(f"{tr['name']} ({tr['role']}): Korridor min {n['min_corridor_m']:.1f} m (easy {res['easy']['min_corridor_m']:.1f} / hard {res['hard']['min_corridor_m']:.1f}), "
                    f"Warnphase {n['min_warn_s']:.1f} s, Sicht {lead:.0f} m ({t_lead:.1f} s), {n['paths_tested']} Durchfahrtstests, {n['paths_failed']} Fehlpfade")
    status = "FAIL" if bad else ("WARN" if warn else "PASS")
    return _chk(19, "Fallen-Fairness (Modell): immer ein freier Korridor, sichtbare Warnphase, Durchfahrt fuer jede Phase/Geschwindigkeit, Varianten easy/hard", status,
                "; ".join(bad + warn) if (bad or warn) else "; ".join(rows), True, dict(rows=rows), model=True)
