"""Centreline authoring kit: straights, exact arcs and C2 clamped splines chained with continuous position and heading,
plus smooth 1-D profiles (elevation, width) along the arclength. Pure numpy, units metres, +x east, +y north."""
from __future__ import annotations

import math

import numpy as np

from . import geom as G


def hvec(deg: float) -> np.ndarray:
    return np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])


def hdeg(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def _spline_c2(t: np.ndarray, p: np.ndarray, d0: np.ndarray, d1: np.ndarray, n_out: int) -> np.ndarray:
    """Clamped C2 cubic interpolating spline through p (N,2) at parameters t with end derivatives d0, d1."""
    n = len(t)
    h = np.diff(t)
    A = np.zeros((n, n))
    B = np.zeros((n, 2))
    A[0, 0], A[0, 1] = 2 * h[0], h[0]
    B[0] = 6 * ((p[1] - p[0]) / h[0] - d0)
    A[-1, -2], A[-1, -1] = h[-1], 2 * h[-1]
    B[-1] = 6 * (d1 - (p[-1] - p[-2]) / h[-1])
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        B[i] = 6 * ((p[i + 1] - p[i]) / h[i] - (p[i] - p[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, B)
    tt = np.linspace(t[0], t[-1], n_out)
    idx = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, n - 2)
    hh = h[idx][:, None]
    a = (t[idx + 1] - tt)[:, None]
    b = (tt - t[idx])[:, None]
    S = (M[idx] * a ** 3 + M[idx + 1] * b ** 3) / (6 * hh) + (p[idx] / hh - M[idx] * hh / 6) * a + (p[idx + 1] / hh - M[idx + 1] * hh / 6) * b
    return S


class Path:
    """Chain of sections. Every section records its name and index range so stations (arclength) can be named."""

    def __init__(self, start=(0.0, 0.0), heading_deg: float = 0.0, ds: float = 2.0):
        self.ds = ds
        self.pts = [np.array(start, float)]
        self.h = hvec(heading_deg)
        self.marks: list[dict] = []

    @property
    def pos(self) -> np.ndarray:
        return self.pts[-1]

    def _add(self, name: str, seg: np.ndarray, kind: str, **info):
        i0 = len(self.pts) - 1
        for p in seg[1:]:
            self.pts.append(np.array(p, float))
        self.marks.append(dict(name=name, kind=kind, i0=i0, i1=len(self.pts) - 1, **info))
        d = self.pts[-1] - self.pts[-2]
        self.h = d / np.linalg.norm(d)

    def line(self, L: float, name: str = "") -> "Path":
        n = max(2, int(round(L / self.ds)) + 1)
        s = np.linspace(0, L, n)[:, None]
        self._add(name, self.pos + self.h * s, "line")
        return self

    def arc(self, R: float, deg: float, name: str = "") -> "Path":
        """deg > 0 turns left (counter-clockwise), < 0 right."""
        sg = 1 if deg > 0 else -1
        n = max(3, int(round(abs(math.radians(deg)) * R / self.ds)) + 1)
        nrm = np.array([-self.h[1], self.h[0]]) * sg
        c = self.pos + nrm * R
        a0 = math.atan2(*(self.pos - c)[::-1])
        a = a0 + sg * np.linspace(0, math.radians(abs(deg)), n)
        h_exact = hvec(hdeg(self.h) + deg)
        self._add(name, np.column_stack([c[0] + R * np.cos(a), c[1] + R * np.sin(a)]), "arc", radius=R, turn=deg)
        self.h = h_exact  # exact tangent (the chord of the last discretised step would drift the heading)
        return self

    def spline(self, through, end, end_heading_deg: float, name: str = "", tension: float = 1.0) -> "Path":
        """C2 curve from the current pose through `through` points to `end` with exact end heading."""
        P = np.vstack([self.pos, np.asarray(through, float).reshape(-1, 2), np.asarray(end, float)])
        t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        seg = _spline_c2(t, P, self.h * tension, hvec(end_heading_deg) * tension, int(t[-1] / 0.5) + 2)
        seg = G.resample(seg, self.ds)
        seg[0], seg[-1] = P[0], P[-1]
        self._add(name, seg, "spline")
        self.h = hvec(end_heading_deg)
        return self

    def array(self) -> np.ndarray:
        return np.array(self.pts)


def smooth_profile(stations, values, s: np.ndarray) -> np.ndarray:
    """Monotone-friendly (Fritsch-Carlson PCHIP) interpolation of values at stations, evaluated at s."""
    x, y = np.asarray(stations, float), np.asarray(values, float)
    h, d = np.diff(x), np.diff(y) / np.diff(x)
    m = np.zeros(len(x))
    for i in range(1, len(x) - 1):
        if d[i - 1] * d[i] > 0:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
    m[0], m[-1] = 0.0, 0.0
    idx = np.clip(np.searchsorted(x, s, side="right") - 1, 0, len(x) - 2)
    t = (s - x[idx]) / h[idx]
    h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    out = h00 * y[idx] + h10 * h[idx] * m[idx] + h01 * y[idx + 1] + h11 * h[idx] * m[idx + 1]
    return np.where(s <= x[0], y[0], np.where(s >= x[-1], y[-1], out))


def smooth_width(stations, values, s: np.ndarray) -> np.ndarray:
    """Cosine-eased piecewise width profile (no overshoot)."""
    x, y = np.asarray(stations, float), np.asarray(values, float)
    idx = np.clip(np.searchsorted(x, s, side="right") - 1, 0, len(x) - 2)
    t = np.clip((s - x[idx]) / (x[idx + 1] - x[idx]), 0, 1)
    t = 0.5 - 0.5 * np.cos(math.pi * t)
    return y[idx] * (1 - t) + y[idx + 1] * t
