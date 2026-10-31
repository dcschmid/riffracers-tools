"""Small PIL drawing kit for technical top-down plans: world->pixel transform with supersampling, polygons, lines, arrows, labels.
Equal scale in x and y (plans are never distorted)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box

FONT_DIR = "/usr/share/fonts/google-noto/"


def font(size, weight="Regular"):
    return ImageFont.truetype(f"{FONT_DIR}NotoSans-{weight}.ttf", int(size))


def mono(size):
    return ImageFont.truetype(f"{FONT_DIR}NotoSansMono-Regular.ttf", int(size))


def hexrgb(h, a=255):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


class Map:
    def __init__(self, W, H, bbox, margin=(40, 40, 40, 40), ss=2, bg="#e9eef2"):
        self.W, self.H, self.ss = W, H, ss
        self.img = Image.new("RGBA", (W * ss, H * ss), hexrgb(bg))
        self.d = ImageDraw.Draw(self.img, "RGBA")
        x0, y0, x1, y1 = bbox
        l, t, r, b = margin
        aw, ah = W - l - r, H - t - b
        self.k = min(aw / (x1 - x0), ah / (y1 - y0))
        self.ox = l + (aw - (x1 - x0) * self.k) / 2 - x0 * self.k
        self.oy = t + (ah - (y1 - y0) * self.k) / 2 + y1 * self.k
        self.placed: list[Polygon] = []

    def P(self, xy):
        a = np.asarray(xy, float)
        return ((a[..., 0] * self.k + self.ox) * self.ss, (self.oy - a[..., 1] * self.k) * self.ss)

    def pts(self, xy):
        x, y = self.P(np.asarray(xy, float).reshape(-1, 2))
        return list(zip(x.tolist(), y.tolist()))

    def m2px(self, m):
        return m * self.k * self.ss

    def polygon(self, poly, fill=None, outline=None, width=1.0):
        if poly is None or poly.is_empty:
            return
        geoms = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        for g in geoms:
            if g.geom_type != "Polygon":
                continue
            self.d.polygon(self.pts(np.array(g.exterior.coords)), fill=fill, outline=None)
            for h in g.interiors:
                self.d.polygon(self.pts(np.array(h.coords)), fill=hexrgb("#e9eef2"))
            if outline:
                self.d.line(self.pts(np.array(g.exterior.coords)), fill=outline, width=int(width * self.ss), joint="curve")

    def line(self, xy, color, width=2.0, dash=None, joint="curve"):
        p = self.pts(xy)
        w = max(1, int(width * self.ss))
        if not dash:
            self.d.line(p, fill=color, width=w, joint=joint)
            return
        # dashed polyline in pixel space
        on, off = dash[0] * self.ss, dash[1] * self.ss
        acc, draw_on, cur = 0.0, True, [p[0]]
        for a, b in zip(p[:-1], p[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if seg == 0:
                continue
            t = 0.0
            while t < seg:
                lim = (on if draw_on else off) - acc
                step = min(lim, seg - t)
                pa = (a[0] + (b[0] - a[0]) * t / seg, a[1] + (b[1] - a[1]) * t / seg)
                pb = (a[0] + (b[0] - a[0]) * (t + step) / seg, a[1] + (b[1] - a[1]) * (t + step) / seg)
                if draw_on:
                    self.d.line([pa, pb], fill=color, width=w)
                t += step
                acc += step
                if acc >= (on if draw_on else off) - 1e-6:
                    acc, draw_on = 0.0, not draw_on

    def circle(self, xy, r_px, fill=None, outline=None, width=1.5):
        x, y = self.P(np.asarray(xy, float))
        r = r_px * self.ss
        self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline, width=int(width * self.ss))

    def text_size(self, s, f):
        b = self.d.textbbox((0, 0), s, font=f)
        return b[2] - b[0], b[3] - b[1]

    def text(self, xy_px, s, size=16, weight="Regular", fill="#1b2430", anchor="la", stroke=None, stroke_w=3):
        f = font(size * self.ss, weight)
        x, y = xy_px
        self.d.text((x * self.ss, y * self.ss), s, font=f, fill=fill if not isinstance(fill, str) else hexrgb(fill), anchor=anchor,
                    stroke_width=(stroke_w * self.ss if stroke else 0), stroke_fill=stroke)

    def label(self, xy, s, color="#1b2430", bg="#ffffffee", size=15, weight="SemiBold", dx=0, dy=0, border=None, leader=True, avoid=None, pad=5, pref=None):
        """Place a box label near world point xy; greedy search of 16 directions for a free spot (not over road and other labels)."""
        f = font(size * self.ss, weight)
        tw, th = self.text_size(s, f)
        tw, th = tw / self.ss + 2 * pad, th / self.ss + 2 * pad
        px, py = self.P(np.asarray(xy, float))
        px, py = px / self.ss, py / self.ss
        cands = []
        base = [(dx, dy)] if (dx or dy) else []
        for rad in (0, 30, 52, 78, 110, 150):
            for a in range(0, 360, 22):
                if rad == 0 and a:
                    continue
                cands.append((rad * math.cos(math.radians(a)) + 0, -rad * math.sin(math.radians(a))))
        if pref:
            cands.sort(key=lambda c: (abs(c[0] - pref[0]) + abs(c[1] - pref[1])) * 0.01 + math.hypot(*c) * 0.0)
        best = None
        for ddx, ddy in base + cands:
            x0, y0 = px + ddx - tw / 2, py + ddy - th / 2
            if x0 < 4 or y0 < 4 or x0 + tw > self.W - 4 or y0 + th > self.H - 4:
                continue
            r = box(x0, y0, x0 + tw, y0 + th)
            if any(r.intersects(p) for p in self.placed):
                continue
            if avoid is not None and r.intersects(avoid):
                continue
            best = (x0, y0, ddx, ddy)
            break
        if best is None:
            x0, y0, ddx, ddy = px + dx - tw / 2, py + dy - th / 2, dx, dy
        else:
            x0, y0, ddx, ddy = best
        self.placed.append(box(x0 - 2, y0 - 2, x0 + tw + 2, y0 + th + 2))
        if leader and (abs(ddx) > 1 or abs(ddy) > 1):
            cx, cy = x0 + tw / 2, y0 + th / 2
            self.d.line([(px * self.ss, py * self.ss), (cx * self.ss, cy * self.ss)], fill=hexrgb(color if isinstance(color, str) else "#334"), width=int(1.6 * self.ss))
            self.d.ellipse([px * self.ss - 3 * self.ss, py * self.ss - 3 * self.ss, px * self.ss + 3 * self.ss, py * self.ss + 3 * self.ss], fill=hexrgb(color))
        self.d.rounded_rectangle([x0 * self.ss, y0 * self.ss, (x0 + tw) * self.ss, (y0 + th) * self.ss], radius=5 * self.ss, fill=hexrgb(bg[:7], 235) if len(bg) >= 7 else hexrgb(bg),
                                 outline=hexrgb(border or color), width=int(1.6 * self.ss))
        self.d.text(((x0 + pad) * self.ss, (y0 + pad - 1) * self.ss), s, font=f, fill=hexrgb(color) if isinstance(color, str) else color)
        return (x0, y0, tw, th)

    def chevrons(self, xy, color, every=70.0, size=5.5, width=2.0):
        c = np.asarray(xy, float)
        seg = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        for s in np.arange(every / 2, cum[-1], every):
            i = int(np.searchsorted(cum, s))
            i = min(max(i, 1), len(c) - 1)
            p = c[i - 1][:2] + (c[i][:2] - c[i - 1][:2]) * ((s - cum[i - 1]) / max(cum[i] - cum[i - 1], 1e-9))
            h = (c[i][:2] - c[i - 1][:2])
            h = h / max(np.linalg.norm(h), 1e-9)
            n = np.array([-h[1], h[0]])
            sz = size / self.k
            a = p - h * sz + n * sz * 0.9
            b = p + h * sz * 0.4
            cc = p - h * sz - n * sz * 0.9
            self.d.line(self.pts([a, b, cc]), fill=color, width=int(width * self.ss), joint="curve")

    def scale_bar(self, x_px, y_px, meters=100, color="#1b2430"):
        L = meters * self.k
        self.d.line([(x_px * self.ss, y_px * self.ss), ((x_px + L) * self.ss, y_px * self.ss)], fill=hexrgb(color), width=4 * self.ss)
        for xx in (x_px, x_px + L):
            self.d.line([(xx * self.ss, (y_px - 6) * self.ss), (xx * self.ss, (y_px + 6) * self.ss)], fill=hexrgb(color), width=3 * self.ss)
        self.text((x_px, y_px - 28), f"{meters} m", 15, "SemiBold", color)

    def north(self, x_px, y_px, color="#1b2430"):
        self.d.polygon([(x_px * self.ss, (y_px - 22) * self.ss), ((x_px - 8) * self.ss, (y_px + 6) * self.ss), (x_px * self.ss, (y_px + 1) * self.ss),
                        ((x_px + 8) * self.ss, (y_px + 6) * self.ss)], fill=hexrgb(color))
        self.text((x_px, y_px + 10), "N", 14, "Bold", color, anchor="ma")

    def done(self):
        return self.img.resize((self.W, self.H), Image.LANCZOS).convert("RGB")


ROAD = dict(main="#2d3647", pit_lane="#3d78c4", pit_entry="#3d78c4", pit_exit="#3d78c4", shortcut="#f08a24", jump_approach="#c23ba3", jump_ramp="#f2c418",
            jump_landing="#2fb37a", jump_exit="#c23ba3", jump_flight="#c23ba3")
SECTION_COL = dict(intro="#ff7a59", verse="#f2c418", build_up="#ec4899", drop="#22b8c4", breakdown="#7c5cd6", finale="#34b36b")
SECTION_LABEL = dict(intro="INTRO", verse="VERSE", build_up="BUILD-UP", drop="DROP", breakdown="BREAKDOWN", finale="FINALE")


def bbox_of(layout, pad=60):
    xs, ys = [], []
    for e in layout["edges"]:
        c = np.array(e["centerline"])
        xs += [c[:, 0].min() - 20, c[:, 0].max() + 20]
        ys += [c[:, 1].min() - 20, c[:, 1].max() + 20]
    for b in layout["pit"]["boxes"]:
        xs.append(b["center"][0]); ys.append(b["center"][1] - 20)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def quick_preview(layout, path, W=1800, H=1200):
    from .track import edge_poly
    m = Map(W, H, bbox_of(layout, 40))
    E = sorted(layout["edges"], key=lambda e: float(np.mean(np.array(e["centerline"])[:, 2])))
    for e in E:
        if not e.get("polygon", True):
            m.line(np.array(e["centerline"])[:, :2], "#c23ba3", 3, dash=(8, 6))
            continue
        m.polygon(edge_poly(e), fill=hexrgb(ROAD.get(e["kind"], "#555")))
    for s in layout["stands"]:
        from .geom import rect
        m.polygon(rect(*s["center"], *s["size"], s["heading_deg"]), fill=hexrgb("#b9776b"))
    for k in layout["keep_out"]:
        m.polygon(Polygon(k), fill=hexrgb("#d8c7e6", 160))
    for e in layout["edges"]:
        if e["kind"] == "main":
            m.chevrons(np.array(e["centerline"]), (255, 255, 255, 200), 80)
    for nd in layout["nodes"]:
        m.circle(nd["xy"], 3.5, fill=hexrgb("#ffffff"), outline=hexrgb("#000000"))
    m.scale_bar(40, H - 40)
    m.north(W - 50, 60)
    m.text((20, 12), f"{layout['name']}  {layout['length_m']:.0f} m", 22, "Bold")
    m.done().save(path)
