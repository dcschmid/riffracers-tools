"""Closure helper: find the two pad lengths so that the west side lands on x_target and has enough vertical room (authoring aid, exact arcs everywhere)."""


def _pad_s_for(build, pad_e, x_target, hi=500.0):
    lo = 0.0
    T, xw = build(pad_e, lo)
    T2, xw2 = build(pad_e, hi)
    if not (xw2 <= x_target <= xw):
        raise RuntimeError(f"closure not bracketed: x at pad 0 = {xw:.1f}, at {hi} = {xw2:.1f}, target {x_target}")
    for _ in range(60):
        mid = (lo + hi) / 2
        T, xw = build(pad_e, mid)
        if xw > x_target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def solve_closure(build, x_target, west_need, y_end=-60.0, extra_e=0.0, south=False):
    for pe in range(0, 900, 6):
        pad_e = pe + extra_e
        try:
            ps = _pad_s_for(build, pad_e, x_target)
        except RuntimeError:
            continue
        T, xw = build(pad_e, ps)
        room = (T.P.pos[1] - y_end) if south else (y_end - T.P.pos[1])
        if room >= west_need:
            return T, pad_e, ps
    raise RuntimeError("no closing pad combination found")
