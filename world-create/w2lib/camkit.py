"""Generic camera set derived from the layout (hero / overview / gameplay / pit lane / hazard)."""
import numpy as np


def default(W, lay, hero_mul=(-0.62, -0.88, 0.40), tgt=(0.0, 0.08, 0.0), overview_mul=(0.04, -0.62, 1.0)):
    xs = np.concatenate([np.array(e["centerline"])[:, 0] for e in lay["edges"]])
    ys = np.concatenate([np.array(e["centerline"])[:, 1] for e in lay["edges"]])
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    ex = max(xs.max() - xs.min(), ys.max() - ys.min())
    sd = lay["pit"]["side"]
    # hazard camera: behind the first trap, a little off the centre line
    tr = lay["traps"][0]
    if tr.get("stations"):
        st = tr["stations"][1 if len(tr["stations"]) > 1 else 0]
        h = np.radians(st["heading_deg"]); p = np.array(st["xy"]); z = st["z"]
    else:
        i0 = np.array(tr["xy_start"]); i1 = np.array(tr["xy_end"]); p = (i0 + i1) / 2; d = i1 - i0; h = np.arctan2(d[1], d[0]); z = tr["z"]
    u = np.array([np.cos(h), np.sin(h)]); n = np.array([-u[1], u[0]])
    cam_h = p - u * 56 - n * 1.0
    karts = [[2, 3.2, 0, 0], [14, -3.6, 0, 3], [30, 2.4, 0, 1], [44, -2.2, 0, 6], [62, 3.4, 0, 2], [78, -3.2, 0, 8]]
    return dict(
        hero=dict(loc=[cx + hero_mul[0] * ex, cy + hero_mul[1] * ex, hero_mul[2] * ex], target=[cx + tgt[0] * ex, cy + tgt[1] * ex, 0], lens=32, samples=192),
        overview=dict(loc=[cx + overview_mul[0] * ex, cy + overview_mul[1] * ex, overview_mul[2] * ex], target=[cx, cy + 0.04 * ex, 0], lens=36),
        gameplay=dict(loc=[-24, 4.0, 3.3], target=[70, 0.5, 2.0], lens=34, karts=karts),
        pitlane=dict(loc=[-75, -sd * 6.0, 7.0], target=[-6, sd * 33, 2.2], lens=30),
        hazard=dict(loc=[float(cam_h[0]), float(cam_h[1]), float(z) + 3.4], target=[float(p[0] + u[0] * 16), float(p[1] + u[1] * 16), float(z) + 3.0], lens=28),
    )
