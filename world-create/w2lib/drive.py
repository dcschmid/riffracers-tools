"""Simplified arcade-kart driving model: curvature-limited speed profile with forward/backward acceleration passes.

Assumptions (all documented in docs/musical-track-design.md, none measured on a real vehicle model):
  top speed 38 m/s, lateral grip 13 m/s², engine 9 m/s² falling to 0 at top speed, braking 22 m/s²,
  racing-line gain: effective radius = r + 0.35·width (the kart may use the full width), dirt/boardwalk grip factor on shortcuts.
The profile gives lap times, corner speeds, braking zones and take-off speeds. It is not a physics simulation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geom as G

G_ACC = 9.81


@dataclass
class Kart:
    v_top: float = 38.0
    a_lat: float = 13.0
    a_eng: float = 9.0
    a_brake: float = 22.0
    line_gain: float = 0.35
    hairpin_floor: float = 9.0  # m/s, slowest corner speed (arcade drifting)


def speed_profile(xyz: np.ndarray, width, closed: bool, kart: Kart = Kart(), grip=1.0, v0: float | None = None) -> dict:
    """Return dict(s, v, t, lap_time, vmax, kappa, length) for a polyline [N,3]. `grip` may be a scalar or a per-sample array
    (surface factor on lateral grip and braking). Closed laps iterate to a periodic solution."""
    p = xyz[:, :2]
    z = xyz[:, 2]
    n = len(p)
    ds = G.seglen(p, closed)
    s = np.concatenate([[0.0], np.cumsum(ds)])[:n]
    w = np.broadcast_to(np.asarray(width, float), (n,))
    gr = np.broadcast_to(np.asarray(grip, float), (n,))
    k = np.abs(G.signed_curvature(p, closed))
    k = np.convolve(np.pad(k, 4, mode="wrap" if closed else "edge"), np.ones(9) / 9, mode="valid")  # smooth numeric noise
    r_eff = 1.0 / np.maximum(k, 1e-6) + kart.line_gain * w
    v_c = np.maximum(np.minimum(kart.v_top, np.sqrt(kart.a_lat * gr * r_eff)), kart.hairpin_floor)
    slope = np.gradient(z, s) if n > 2 else np.zeros(n)
    v = v_c.copy()
    if v0 is not None:
        v[0] = min(v[0], v0)
    for _ in range(3 if closed else 1):
        for i in range(n - 1, -1, -1):  # braking pass
            if not closed and i == n - 1:
                continue
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * kart.a_brake * gr[i] * ds[i]))
        for i in range(n if closed else n - 1):  # acceleration pass
            j = (i + 1) % n
            a = max(kart.a_eng * (1 - (v[i] / (kart.v_top * 1.03)) ** 2) - G_ACC * float(slope[i]), 0.5)
            v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * a * ds[i]))
    vm = 0.5 * (v + np.roll(v, -1))[: len(ds)]
    dt = ds / np.maximum(vm, 1.0)
    t = np.concatenate([[0.0], np.cumsum(dt)])[:n]
    return dict(s=s, v=v, t=t, lap_time=float(dt.sum()), vmax=v_c, kappa=k, length=float(ds.sum()))


def flight(v: float, ang_deg: float, h0: float, g: float = G_ACC) -> dict:
    """Ballistic flight from take-off speed v (m/s) at ramp angle ang_deg, h0 metres above the landing surface (flat model).
    Returns horizontal range, flight time, apex height above take-off and the descent angle at impact."""
    a = math.radians(ang_deg)
    vx, vy = v * math.cos(a), v * math.sin(a)
    t = (vy + math.sqrt(vy * vy + 2 * g * h0)) / g
    return dict(range=vx * t, time=t, apex=vy * vy / (2 * g), impact_deg=math.degrees(math.atan2(g * t - vy, vx)))
