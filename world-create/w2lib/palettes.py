"""Face-colour functions for terrain meshes (low-poly look: per-face jitter, slope and height driven)."""
import numpy as np


def hx(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _mix(a, b, t):
    return a * (1 - t[:, None]) + b * t[:, None]


def coastal(cen, nrm, world, rng):
    z, slope = cen[:, 2], 1 - nrm[:, 2]
    n = len(z)
    jit = rng.uniform(-0.06, 0.06, (n, 1))
    sand, grass1, grass2, grassD, rock, rock2 = (hx(c) for c in ("#f6d88a", "#62d04f", "#3fba4a", "#2a9a44", "#e3b88e", "#c99a6e"))
    t_g = np.clip((z - 0.8) / 2.2, 0, 1)
    col = _mix(np.tile(sand, (n, 1)), grass1, t_g)
    pat = (np.sin(cen[:, 0] * 0.05) * np.cos(cen[:, 1] * 0.04) + rng.uniform(-0.4, 0.4, n)) > 0.15
    col = np.where(pat[:, None] & (z > 2.5)[:, None], _mix(col, grass2, np.full(n, 0.9)), col)
    col = np.where((z > 11)[:, None], _mix(col, grassD, np.clip((z - 11) / 9, 0, 1) * 0.8), col)
    nz = np.sin(cen[:, 0] * 0.021 + 1.3) * np.cos(cen[:, 1] * 0.017) + 0.6 * np.sin(cen[:, 0] * 0.047 + cen[:, 1] * 0.033)
    jungle = (nz > 0.5) & (z > 2.0) & (slope < 0.25)
    meadow = (nz < -0.55) & (z > 1.5) & (slope < 0.2)
    col = np.where(jungle[:, None], _mix(col, np.tile(hx("#1f8a40"), (n, 1)), np.full(n, 0.85)), col)
    col = np.where(meadow[:, None], _mix(col, np.tile(hx("#a6e36a"), (n, 1)), np.full(n, 0.8)), col)
    t_r = np.clip((slope - 0.16) / 0.2, 0, 1)
    band = (np.floor(z / 2.4).astype(int) % 2).astype(float)
    pick = np.where(rng.uniform(size=n) < 0.1, 1 - band, band)
    rock_c = _mix(np.tile(rock, (n, 1)), rock2, pick)
    col = _mix(col, rock_c, t_r)
    under = z < -0.4
    col = np.where(under[:, None], col * 0.55 + hx("#1fb7c9") * 0.45, col)
    return np.clip(col * (1 + jit), 0, 1)


def flat_city(base="#272c3c", alt="#2e3446"):
    b, a = hx(base), hx(alt)

    def f(cen, nrm, world, rng):
        n = len(cen)
        col = _mix(np.tile(b, (n, 1)), a, (rng.uniform(0, 1, n) > 0.5).astype(float))
        return np.clip(col * (1 + rng.uniform(-0.04, 0.04, (n, 1))), 0, 1)
    return f


def coastal3(cen, nrm, world, rng):
    """v3 coast: ripple-streaked sand with a wet band; inland five biomes (golden meadow, bright grass, forest, jungle, dry savanna) in large patches, mown stripes near the arena,
    pink/yellow/violet flower fields, winding sand paths (iso-lines of a smooth noise field), warm sandstone strata on slopes, turquoise shallows."""
    z, slope = cen[:, 2], 1 - nrm[:, 2]
    n = len(z)
    x, y = cen[:, 0], cen[:, 1]
    H = lambda c: np.tile(hx(c), (n, 1))
    sand_l, sand_d, wet = H("#fbe6a6"), H("#efca7c"), H("#d9b676")
    rip = (np.sin(x * 0.23 + y * 0.17 + 2.0 * np.sin(y * 0.03)) > 0.25)
    col = _mix(sand_l, sand_d, rip.astype(float) * 0.7)
    col = _mix(col, wet, np.clip(1 - np.abs(z - 0.05) / 0.5, 0, 1) * 0.8)
    # biome field: three smooth noises pick one of five greens per large patch
    b1 = np.sin(x * 0.0125 + 0.7) * np.cos(y * 0.011) + 0.7 * np.sin(x * 0.027 + y * 0.019 + 1.0)
    b2 = np.sin(x * 0.0083 - 1.1) * np.cos(y * 0.0097 + 0.4) + 0.6 * np.cos(x * 0.021 - y * 0.017)
    meadow, grass, forest, jungle, savanna = H("#c4ee72"), H("#69d655"), H("#3fb94d"), H("#1f8442"), H("#d6d878")
    g = _mix(grass, forest, np.clip(b1 * 0.8 + 0.2, 0, 1))
    g = _mix(g, meadow, np.clip(-b1 * 1.1 - 0.15, 0, 1))
    g = _mix(g, jungle, np.clip((b2 - 0.45) * 2.2, 0, 1))
    g = _mix(g, savanna, np.clip((-b2 - 0.5) * 2.4, 0, 1) * (z > 4))
    g = _mix(g, H("#1d7a40"), np.clip((z - 10) / 10, 0, 1) * 0.7)
    # per-face grass speckle (low poly sparkle)
    g = g * (1 + rng.uniform(-0.06, 0.06, (n, 1)))
    # mown stripes in the festival area (arena around x -10, y 128)
    near = np.exp(-(((x + 10) / 150.0) ** 2 + ((y - 128) / 130.0) ** 2))
    stripe = (np.floor((x * 0.7 + y * 0.45) / 7.0).astype(int) % 2).astype(float)
    g = _mix(g, _mix(g, meadow, np.full(n, 0.45)), stripe * np.clip(near * 1.6, 0, 1) * 0.9)
    # flower fields: patches of pink / yellow / violet / white on the lighter biomes
    f1 = np.sin(x * 0.047 + 2.0) * np.cos(y * 0.041) + 0.5 * np.sin(x * 0.11 - y * 0.09)
    fl = (f1 > 0.78) & (z > 1.4)
    fcol = np.array([hx(c) for c in ("#ff6f91", "#ffd23f", "#ff9f43", "#ffd23f", "#ff8fb3")])
    fid = (np.floor((x * 0.05 + y * 0.037) * 2.0).astype(int)) % 5
    g = np.where((fl & (rng.uniform(size=n) < 0.6))[:, None], _mix(g, fcol[fid], np.full(n, 0.85)), g)
    # winding sand paths: iso-line of a smooth noise field
    pn = np.sin(x * 0.0165 + 0.5 * np.sin(y * 0.011) + 1.7) + np.sin(y * 0.0145 + 0.6 * np.sin(x * 0.013)) * 0.9
    path = (np.abs(pn - 0.35) < 0.07) & (z > 1.2) & (slope < 0.22) & (np.abs(x + 10) + np.abs(y - 128) > 150)
    g = _mix(g, H("#ecd49a") * (1 + 0.04 * rng.uniform(-1, 1, (n, 1))), path.astype(float) * 0.95)
    t_g = np.clip((z - 0.7) / 1.5, 0, 1)
    col = _mix(col, g, t_g)
    strata = [H(c) for c in ("#f6d9a8", "#ebc08a", "#dba670", "#c98e5a")]
    band = (np.floor(z / 1.5).astype(int) % 4)
    rock = np.where((band == 0)[:, None], strata[0], np.where((band == 1)[:, None], strata[1], np.where((band == 2)[:, None], strata[2], strata[3])))
    rock = rock * (1 - 0.10 * (np.sin(x * 0.9 + y * 0.7) > 0.6))[:, None] * (1 - 0.18 * np.clip(slope - 0.3, 0, 1))[:, None]
    col = _mix(col, rock, np.clip((slope - 0.14) / 0.16, 0, 1))
    under = np.clip((-z - 0.2) / 3.0, 0, 1)
    col = _mix(col, _mix(H("#3fd2c4"), H("#168ab0"), under), np.clip((-z + 0.2) / 0.6, 0, 1) * 0.65)
    return np.clip(col * (1 + rng.uniform(-0.04, 0.04, (n, 1))), 0, 1)


def iron_canyon(cen, nrm, world, rng):
    """Metal world 'Iron Chorus': rust-red canyon. Packed rust-brown dirt on the flat arena floor, stacked rust/orange strata on slopes (terraced look by height bands),
    dark basalt caps on the rim, near-black concert pit. Per-face jitter keeps the low-poly sparkle; roads stay the darkest, most uniform surface."""
    z, slope = cen[:, 2], 1 - nrm[:, 2]
    x, y = cen[:, 0], cen[:, 1]
    n = len(z)
    H = lambda c: np.tile(hx(c), (n, 1))
    dirt_a, dirt_b = H("#946048"), H("#82533f")
    pat = (np.sin(x * 0.043 + 1.1) * np.cos(y * 0.037) + rng.uniform(-0.5, 0.5, n)) > 0.1
    col = _mix(dirt_a, dirt_b, pat.astype(float))
    col = _mix(col, H("#b07a58"), np.clip(1 - np.abs(z - 0.2) / 1.2, 0, 1) * 0.30)               # lighter dust on the flat floor
    trample = np.exp(-(((x) / 190.0) ** 2 + ((y - 150.0) / 120.0) ** 2))                           # trampled, lighter ground around the stage
    col = _mix(col, H("#c28f68"), np.clip(trample * 1.2, 0, 1) * (z < 1.5) * 0.55)
    strata = [H(c) for c in ("#a8472b", "#c9653a", "#8c3a28", "#d7864a", "#6e2e25")]
    band = (np.floor(z / 3.2).astype(int) + (np.floor(x * 0.012 + y * 0.009)).astype(int)) % 5
    rock = strata[0].copy()
    for k in range(1, 5):
        rock = np.where((band == k)[:, None], strata[k], rock)
    rock = rock * (1 - 0.16 * np.clip(slope - 0.25, 0, 1))[:, None]
    col = _mix(col, rock, np.clip((slope - 0.07) / 0.16, 0, 1))
    col = _mix(col, _mix(col, H("#3a2c32"), np.clip((z - 30) / 14, 0, 1)), np.clip((z - 26) / 6, 0, 1) * 0.9)   # basalt caps on the rim
    col = _mix(col, H("#2a1f26"), np.clip((-z - 0.5) / 1.5, 0, 1))                                     # concert pit
    return np.clip(col * (1 + rng.uniform(-0.05, 0.05, (n, 1))), 0, 1)


def razor_yard(cen, nrm, world, rng):
    """Metal world 'Razorline Riot': a working scrapyard floor. Packed rust-brown dirt and mud as the base, large worn concrete pads only in patches, black oil stains, rust-orange
    streaks, gravel speckle, dark grey rock on the ridges. Per-face jitter keeps the low-poly sparkle; roads stay the darkest, most uniform surface."""
    z, slope = cen[:, 2], 1 - nrm[:, 2]
    x, y = cen[:, 0], cen[:, 1]
    n = len(z)
    H = lambda c: np.tile(hx(c), (n, 1))
    pat = (np.sin(x * 0.041 + 0.7) * np.cos(y * 0.037) + rng.uniform(-0.55, 0.55, n)) > 0.05
    col = _mix(H("#76624f"), H("#685645"), pat.astype(float))                                          # packed dirt
    mud = np.sin(x * 0.011 + 1.0) * np.cos(y * 0.009) + 0.5 * np.sin(x * 0.027 + y * 0.021)
    col = _mix(col, H("#54443a"), np.clip(mud * 0.9, 0, 1) * (slope < 0.3))                              # mud
    slab = ((np.floor(x / 48.0).astype(int) + np.floor(y / 36.0).astype(int)) % 3 == 0) & (np.sin(x * 0.006 + y * 0.0045) > -0.1)
    col = np.where(slab[:, None] & (slope < 0.2)[:, None], _mix(H("#74767a"), H("#6c6e72"), pat.astype(float)), col)   # worn concrete pads
    rust = np.sin(x * 0.019 - y * 0.015 + 2.0) + 0.6 * np.sin(x * 0.041 + y * 0.037)
    col = _mix(col, H("#7a4a32"), np.clip((rust - 0.55) * 1.2, 0, 1) * 0.6 * (slope < 0.3))              # rust streaks
    oil = (np.sin(x * 0.13 + y * 0.09) * np.cos(y * 0.11 + 0.5) > 0.74)
    col = np.where(oil[:, None] & (slope < 0.2)[:, None], col * 0.62, col)                                 # oil stains
    col = _mix(col, H("#6a5a4c"), (rng.uniform(0, 1, n) > 0.93).astype(float) * 0.5)                      # gravel speckle
    strata = [H(c) for c in ("#6b6460", "#57514e", "#7a6a60", "#4a4644", "#6f6258")]
    band = (np.floor(z / 3.0).astype(int) + (np.floor(x * 0.013 + y * 0.01)).astype(int)) % 5
    rock = strata[0].copy()
    for k in range(1, 5):
        rock = np.where((band == k)[:, None], strata[k], rock)
    rock = rock * (1 - 0.14 * np.clip(slope - 0.25, 0, 1))[:, None]
    col = _mix(col, rock, np.clip((slope - 0.07) / 0.16, 0, 1))
    col = _mix(col, H("#46403c"), np.clip((z - 26.0) / 10.0, 0, 1))                                      # dark plateau on the rim
    col = _mix(col, H("#34302c"), np.clip((slope - 0.55) / 0.25, 0, 1))                                   # (near-)vertical faces stay dark
    col = _mix(col, H("#2a2826"), np.clip((-z - 0.5) / 1.5, 0, 1))                                        # scrap press / container gap
    return np.clip(col * (1 + rng.uniform(-0.05, 0.05, (n, 1))), 0, 1)
