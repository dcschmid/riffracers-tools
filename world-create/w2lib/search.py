"""Authoring aid: search shortcut candidates on a prepared Track (fork/merge angles, Bezier body) and report lap-time savings.
The chosen parameters are then written into the track definition; the validator re-checks the exported geometry independently."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

from . import drive as D
from . import geom as G
from .pathkit import hdeg, hvec
from .track import DS, MIN_CLEAR, ZONE_R


def find_shortcuts(T, s_lo, s_hi, span=(110, 330), saving_s=(1.8, 3.6), angles=((26, 26), (30, 30), (34, 28), (28, 34)), ks=((0.34, 0.34), (0.42, 0.42), (0.5, 0.5)),
                   sides=(1, -1), step=10.0, min_r=17.0, others=None, top=10, straight_r=46.0, width=None):
    if not getattr(T, "_prepared", False):
        T.prepare()
    ring, n, L = T.ring, T.n, T.L
    w = width or T.w_short
    others = others or []
    keep = unary_union(T.keep_out + others) if (T.keep_out or others) else None
    main = T.main_poly
    base = T.prof["lap_time"]
    zr, wr, gr = T.zr, T.wr, T.gr
    out = []
    from .pathkit import smooth_profile  # noqa
    for sA in np.arange(s_lo, s_hi - span[0], step):
        iA = T.idx(sA)
        rA = G.radii(ring[[(iA - k) % n for k in range(0, 24)]], win=2)
        if np.nanmin(rA[2:-2]) < straight_r:
            continue
        for sB in np.arange(sA + span[0], min(sA + span[1], s_hi), step):
            iB = T.idx(sB)
            rB = G.radii(ring[[(iB + k) % n for k in range(0, 24)]], win=2)
            if np.nanmin(rB[2:-2]) < straight_r:
                continue
            A, B = ring[iA], ring[iB]
            hA, hB = T.hh[iA], T.hh[iB]
            chord = float(np.linalg.norm(B - A))
            if chord < 70:
                continue
            for side in sides:
                for angA, angB in angles:
                    dA, dB = hvec(hdeg(hA) + side * angA), hvec(hdeg(hB) - side * angB)
                    for k1, k2 in ks:
                        pth = G.resample(G.bezier(A, A + dA * chord * k1, B - dB * chord * k2, B, 200), DS)
                        r = G.radii(pth, win=3)[3:-3]
                        if np.nanmin(r) < min_r:
                            continue
                        Ls = float(G.seglen(pth).sum())
                        main_len = sB - sA
                        if main_len - Ls < 25:
                            continue
                        poly = G.strip(pth, w)
                        zone = unary_union([Point(A).buffer(56.0), Point(B).buffer(56.0)])
                        far = poly.difference(zone)
                        if far.is_empty:
                            continue
                        if far.intersection(main).area > 0.5 or far.distance(main) < MIN_CLEAR + 1.0:
                            continue
                        if keep is not None and poly.distance(keep) < 6.0:
                            continue
                        # lap time with the shortcut spliced in
                        zA, zB = zr[iA], zr[iB]
                        t = np.linspace(0, 1, len(pth))
                        zz = zA + (zB - zA) * (3 * t ** 2 - 2 * t ** 3)
                        if iB > iA:
                            xyz = np.vstack([np.column_stack([ring[:iA], zr[:iA]]), np.column_stack([pth, zz]), np.column_stack([ring[iB + 1:], zr[iB + 1:]])])
                            ww = np.concatenate([wr[:iA], np.full(len(pth), w), wr[iB + 1:]])
                            gg = np.concatenate([gr[:iA], np.full(len(pth), 0.94), gr[iB + 1:]])
                        else:
                            continue
                        dt = base - D.speed_profile(xyz, ww, True, grip=gg)["lap_time"]
                        if not saving_s[0] <= dt <= saving_s[1]:
                            continue
                        out.append(dict(sA=round(float(sA), 1), sB=round(float(sB), 1), side=side, angA=angA, angB=angB, bez=(k1, k2), saving_s=round(dt, 2),
                                        saving_m=round(main_len - Ls, 1), length=round(Ls, 1), min_r=round(float(np.nanmin(r)), 1),
                                        A=np.round(A, 1).tolist(), B=np.round(B, 1).tolist(), score=-abs(dt - 0.5 * sum(saving_s)) + 0.002 * np.nanmin(r)))
    out.sort(key=lambda c: -c["score"])
    return out[:top]
