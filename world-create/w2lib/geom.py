"""2D geometry helpers (numpy + shapely). Units: metres, +x east, +y north, counter-clockwise positive."""
from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point, Polygon


def seglen(p: np.ndarray, closed: bool = False) -> np.ndarray:
    q = np.roll(p, -1, axis=0) if closed else p[1:]
    a = p if closed else p[:-1]
    return np.linalg.norm(q - a, axis=1)


def arclen(p: np.ndarray, closed: bool = False) -> np.ndarray:
    """Cumulative arclength per point (starts at 0)."""
    return np.concatenate([[0.0], np.cumsum(seglen(p, closed))[: (len(p) if closed else len(p) - 1)]])[: len(p)]


def resample(p: np.ndarray, ds: float, closed: bool = False) -> np.ndarray:
    pts = np.vstack([p, p[:1]]) if closed else p
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = cum[-1]
    n = max(2, int(round(total / ds)) + (0 if closed else 1))
    s = np.linspace(0, total, n, endpoint=not closed)
    return np.column_stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])])


def headings(p: np.ndarray, closed: bool = False) -> np.ndarray:
    """Unit tangent per point."""
    if closed:
        t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    else:
        t = np.gradient(p, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(n == 0, 1, n)


def radii(p: np.ndarray, closed: bool = False, win: int = 3) -> np.ndarray:
    """Local radius of curvature (m) from circumradius of points i-win, i, i+win; inf for straights."""
    n = len(p)
    out = np.full(n, np.inf)
    idx = np.arange(n)
    for i in idx:
        if closed:
            a, b, c = p[(i - win) % n], p[i], p[(i + win) % n]
        else:
            if i - win < 0 or i + win >= n:
                continue
            a, b, c = p[i - win], p[i], p[i + win]
        ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area2 > 1e-9:
            out[i] = ab * bc * ca / (2 * area2)
    return out


def signed_curvature(p: np.ndarray, closed: bool = False) -> np.ndarray:
    h = headings(p, closed)
    ang = np.unwrap(np.arctan2(h[:, 1], h[:, 0]))
    ds = np.maximum(np.gradient(arclen(p, closed)) if not closed else np.full(len(p), seglen(p, True).mean()), 1e-6)
    return np.gradient(ang) / ds


def bezier(p0, p1, p2, p3, n: int = 64) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(x, float) for x in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def rounded_polygon(verts: np.ndarray, rad: np.ndarray, ds: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Closed polyline through straight edges between `verts`, each corner replaced by a fillet arc of radius rad[i].
    Returns (points, corners). rad is reduced where adjacent tangent lengths would not fit."""
    n = len(verts)
    rad = rad.copy().astype(float)
    tl = np.zeros(n)

    def tangent_len(i, r):
        v, a, b = verts[i], verts[i - 1], verts[(i + 1) % n]
        u1, u2 = (a - v) / np.linalg.norm(a - v), (b - v) / np.linalg.norm(b - v)
        th = np.arccos(np.clip(u1 @ u2, -1, 1))
        return r / np.tan(th / 2), th, u1, u2

    for _ in range(4):  # fit radii to edge lengths
        for i in range(n):
            tl[i] = tangent_len(i, rad[i])[0]
        for i in range(n):
            j = (i + 1) % n
            L = np.linalg.norm(verts[j] - verts[i])
            if tl[i] + tl[j] > 0.92 * L:
                k = 0.92 * L / (tl[i] + tl[j])
                rad[i] *= k; rad[j] *= k
    pts, corners = [], []
    for i in range(n):
        t, th, u1, u2 = tangent_len(i, rad[i])
        v = verts[i]
        t1, t2 = v + u1 * t, v + u2 * t
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        c = v + bis * rad[i] / np.sin(th / 2)
        a1, a2 = np.arctan2(*(t1 - c)[::-1]), np.arctan2(*(t2 - c)[::-1])
        cross = u1[0] * u2[1] - u1[1] * u2[0]  # >0: travelling along -u1 .. then u2 turns left? (see sweep sign below)
        d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        steps = max(3, int(abs(d) * rad[i] / ds))
        arc = np.array([c + rad[i] * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(a1, a1 + d, steps)])
        pts.append(arc)
        corners.append(dict(vertex=v.tolist(), radius=float(rad[i]), turn_deg=float(np.degrees(abs(d))),
                            direction="left" if d > 0 else "right", tangent_len=float(t), center=c.tolist()))
    poly = np.vstack(pts)
    return poly, corners


def ring_rotate(p: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    """Re-start a closed polyline at the location nearest to `point` (inserted exactly)."""
    ring = LineString(np.vstack([p, p[:1]]))
    s0 = ring.project(Point(point))
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1))])
    pp = np.vstack([p, p[:1]])
    ins = np.array(ring.interpolate(s0).coords[0])
    k = int(np.searchsorted(cum, s0))
    out = np.vstack([ins, pp[k:-1], pp[:k]]) if k < len(pp) else p
    return out


def rect(cx: float, cy: float, w: float, d: float, rot_deg: float = 0.0) -> Polygon:
    """Rectangle w (along heading) x d (across), centred, rotated CCW."""
    a = np.radians(rot_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]]) @ R.T + [cx, cy]
    return Polygon(c)


def strip(p: np.ndarray, width: float, cap: str = "flat") -> Polygon:
    return LineString(p).buffer(width / 2, cap_style=cap, join_style="round", resolution=8)


def var_strip(p: np.ndarray, widths) -> Polygon:
    """Road polygon around a polyline with per-point widths. Long or strongly curved polylines are built piecewise and unioned
    (a single left/right ring self-overlaps on loops > 180 deg and buffer(0) would then fill the loop interior)."""
    from shapely.ops import unary_union
    p = np.asarray(p, float)
    w = np.broadcast_to(np.asarray(widths, float), (len(p),))
    if len(p) > 40:
        parts = []
        for i0 in range(0, len(p) - 1, 30):
            i1 = min(i0 + 34, len(p))
            if i1 - i0 >= 2:
                parts.append(_var_strip_simple(p[i0:i1], w[i0:i1]))
        return unary_union(parts).buffer(0.02).buffer(-0.02)
    return _var_strip_simple(p, w)


def _var_strip_simple(p, widths) -> Polygon:
    w = np.asarray(widths, float) / 2
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.column_stack([-t[:, 1], t[:, 0]])
    left, right = p + n * w[:, None], p - n * w[:, None]
    poly = Polygon(np.vstack([left, right[::-1]]))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly
