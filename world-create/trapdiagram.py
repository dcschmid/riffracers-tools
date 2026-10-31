"""Trap phase diagram from the layout data: per trap, three snapshot times; rows along the road (driving direction up), lateral offset left-to-right (left = +d).
Red = band active, amber = warning phase, green = widest free corridor. Shows what a driver actually has to do; numbers come from hazards/layout, not from the render."""
import json, sys
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/google-noto/NotoSans-SemiBold.ttf"
FONTR = "/usr/share/fonts/google-noto/NotoSans-Regular.ttf"


def state(b, cyc, t):
    x = (t - b.get("phase", 0.0)) % cyc
    if b["on"][0] <= x < b["on"][1]:
        return "on"
    w = b.get("warn", (0, 0))
    return "warn" if w[0] <= x < w[1] else "off"


def gaps(st, cyc, t, half):
    act = sorted((b["d0"], b["d1"]) for b in st["bands"] if state(b, cyc, t) == "on")
    m = []
    for a, b in act:
        if m and a <= m[-1][1]:
            m[-1][1] = max(m[-1][1], b)
        else:
            m.append([a, b])
    e = [-half]
    for a, b in m:
        e += [a, b]
    e.append(half)
    return [(e[i], e[i + 1]) for i in range(0, len(e), 2) if e[i + 1] - e[i] > 0.3]


def draw_trap(tr, half, times, out, title, note):
    f1, f2, f3 = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONTR, 17)
    cyc = tr["cycle"]
    sts = tr["stations"]
    n = len(sts)
    W_panel, row_h, pad, top = 360, 54, 40, 150
    scale = (W_panel - 40) / (2 * half)
    H = top + n * row_h + 150
    W = pad + len(times) * (W_panel + pad)
    im = Image.new("RGB", (W, H), (22, 26, 38))
    d = ImageDraw.Draw(im)
    d.text((pad, 22), title, font=f1, fill=(255, 255, 255))
    d.text((pad, 66), note, font=f3, fill=(190, 200, 215))
    for pi, t in enumerate(times):
        x0 = pad + pi * (W_panel + pad)
        d.text((x0, top - 36), f"Phase t = {t:.1f} s (Zyklus {cyc:.1f} s)", font=f2, fill=(255, 255, 255))
        d.rectangle([x0, top, x0 + W_panel, top + n * row_h], fill=(52, 60, 82))
        cx = x0 + W_panel / 2
        d.line([cx, top, cx, top + n * row_h], fill=(120, 130, 150), width=1)
        worst = 99
        for k, st in enumerate(sts):
            y1 = top + (n - k) * row_h
            y0 = y1 - row_h + 8
            for b in st["bands"]:
                s_ = state(b, cyc, t)
                xa, xb = cx - b["d1"] * scale, cx - b["d0"] * scale
                if s_ == "on":
                    d.rectangle([xa, y0, xb, y1 - 8], fill=(235, 70, 70))
                elif s_ == "warn":
                    d.rectangle([xa, y0, xb, y1 - 8], outline=(255, 176, 46), width=3)
            g = gaps(st, cyc, t, half)
            g0, g1 = max(g, key=lambda q: q[1] - q[0])
            worst = min(worst, g1 - g0)
            xa, xb = cx - g1 * scale + 3, cx - g0 * scale - 3
            d.rectangle([xa, y0 + 6, xb, y1 - 14], fill=(61, 255, 154))
            d.text(((xa + xb) / 2 - 24, y0 + 8), f"{g1 - g0:.1f} m", font=f3, fill=(8, 40, 24))
        d.text((x0, top + n * row_h + 8), f"kleinste freie Luecke in dieser Phase: {worst:.1f} m", font=f3, fill=(190, 200, 215))
        d.text((x0, top + n * row_h + 34), "<- links          Fahrtrichtung: oben          rechts ->", font=f3, fill=(150, 160, 180))
    lx = pad
    ly = H - 70
    for col, txt, fill in (((235, 70, 70), "aktiv (gesperrt)", True), ((255, 176, 46), "Warnphase (noch frei)", False), ((61, 255, 154), "breiteste freie Spur (Kart 2.2 m)", True)):
        if fill:
            d.rectangle([lx, ly, lx + 26, ly + 18], fill=col)
        else:
            d.rectangle([lx, ly, lx + 26, ly + 18], outline=col, width=3)
        d.text((lx + 36, ly - 2), txt, font=f3, fill=(235, 240, 250))
        lx += 340
    im.save(out)


if __name__ == "__main__":
    lay = json.load(open(sys.argv[1]))
    outdir = sys.argv[2]
    t0, t1 = lay["traps"][0], lay["traps"][1]
    times = (2.2, 4.9, 7.6) if lay.get("drivability_contract") else (0.8, 2.2, 3.6)
    draw_trap(t0, 8.5, times, f"{outdir}/trap-phases-jets.png", "Bass Wave Jets - freie Flaechen je Phase (keine zeitliche Ideallinie)",
              "5 Reihen im Abstand von 24 m, Strasse 17 m. Pro Reihe feuert immer nur EIN Band (links, Mitte oder rechts); in jeder Phase bleibt eine freie Spur.")
    draw_trap(t1, 7.0, times, f"{outdir}/trap-phases-pontons.png", "Strand-Pontons - freie Flaechen je Phase (keine zeitliche Ideallinie)",
              "4 Reihen im Abstand von 26 m, Strasse 14 m. Links und rechts fahren abwechselnd ein, nie beide gleichzeitig; mindestens eine Haelfte bleibt frei.")
    print("ok")
