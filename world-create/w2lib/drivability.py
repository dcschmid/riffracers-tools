"""Additional preproduction checks. Never writes to, builds or runs the game.

The trap search is deliberately conservative: swept rectangular collision envelopes,
finite kart length, bounded lateral velocity/acceleration and a reaction delay. A
failed search means no witness in this discretisation, not mathematical impossibility.
The Lua probe uses the actual unmodified controller with synthetic ground queries;
it is not a Defold/Bullet driving test.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import math
from pathlib import Path

import numpy as np


def lua_result(source: str) -> str:
    lib = ctypes.CDLL(ctypes.util.find_library("lua-5.4") or "liblua-5.4.so")
    lib.luaL_newstate.restype = ctypes.c_void_p
    lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
    lib.luaL_loadstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lua_pcallk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t, ctypes.c_void_p]
    lib.lua_tolstring.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.lua_tolstring.restype = ctypes.c_char_p
    lib.lua_close.argtypes = [ctypes.c_void_p]
    state = lib.luaL_newstate()
    try:
        lib.luaL_openlibs(state)
        status = lib.luaL_loadstring(state, source.encode())
        if not status:
            status = lib.lua_pcallk(state, 0, 1, 0, 0, None)
        value = lib.lua_tolstring(state, -1, None)
        text = value.decode() if value else ""
        if status:
            raise RuntimeError(text)
        return text
    finally:
        lib.lua_close(state)


def trap_search(tr, speed, variant="normal", phase_step=0.1, reaction=0.5,
                vmax=2.8, dt=0.1, store_witness=False):
    """Frenet-coordinate reachability for a constant, explicitly stated forward speed.

    Starts in the road centre at the warning sign, with zero lateral velocity.
    Velocity lattice: 9 values; acceleration <= 7 m/s² at the defaults. Exact
    trapezoidal integration on the position lattice. The entire transition is
    tested against every active band; activation intervals include 0.1 s padding.
    Curvature and controller tracking are separate checks, not claimed here.
    """
    var = tr.get("variants", {}).get(variant, {}) if variant != "normal" else {}
    cs, bs = var.get("cycle_scale", 1), var.get("band_scale", 1)
    cycle = tr["cycle"] * cs
    phases = np.arange(0, cycle, phase_step)
    dv = vmax / 4
    dx = dv * dt / 2
    half = tr["width_at"] / 2 - 1.1 - 0.6
    count = int(math.floor(half / dx))
    offsets = np.arange(-count, count + 1) * dx
    n = len(offsets)
    reach = np.zeros((len(phases), 9, n), bool)
    reach[:, 4, count] = True
    cost = np.where(reach, 0.0, np.inf) if store_witness else None
    lead = tr["warn_lead_m"]
    end = max(st["off"] + max(b.get("len", 6) for b in st["bands"]) / 2 + 1.4 for st in tr["stations"]) + 2
    steps = int(math.ceil((lead + end) / (speed * dt)))
    history = []
    for step in range(steps):
        t0, t1 = step * dt, (step + 1) * dt
        s0, s1 = -lead + speed * t0, -lead + speed * t1
        obstacles = []
        for station in tr["stations"]:
            for b in station["bands"]:
                long_half = b.get("len", 6) / 2 + 1.4
                if s1 < station["off"] - long_half or s0 > station["off"] + long_half:
                    continue
                on0, on1 = b["on"][0] * cs - 0.1, b["on"][1] * cs + 0.1
                tau = phases + t0 - b.get("phase", 0) * cs
                # Exact intersection of [tau, tau+dt] and a periodic ON interval.
                elapsed = (tau - on0) % cycle
                active = (elapsed <= on1 - on0) | (elapsed + dt >= cycle)
                centre = (b["d0"] + b["d1"]) / 2
                radius = (b["d1"] - b["d0"]) * bs / 2 + 1.1 + 0.3
                obstacles.append((active, centre - radius, centre + radius))
        next_reach = np.zeros_like(reach)
        parent = np.full(reach.shape, -1, np.int8) if store_witness else None
        next_cost = np.full(reach.shape, np.inf) if store_witness else None
        for old in range(9):
            for new in range(max(0, old - 1), min(9, old + 2)):
                if t0 < reaction and new != 4:
                    continue
                shift = old + new - 8
                lo, hi = max(0, -shift), min(n, n - shift)
                if hi <= lo:
                    continue
                valid = reach[:, old, lo:hi].copy()
                if not valid.any():
                    continue
                d0, d1 = offsets[lo:hi], offsets[lo + shift:hi + shift]
                for active, a, b in obstacles:
                    lateral = (np.maximum(d0, d1) >= a) & (np.minimum(d0, d1) <= b)
                    valid &= ~(active[:, None] & lateral[None, :])
                target = next_reach[:, new, lo + shift:hi + shift]
                if parent is not None:
                    pv = parent[:, new, lo + shift:hi + shift]
                    candidate = cost[:, old, lo:hi] + dt * (.02*d1**2 + ((new-4)*dv)**2 + .08*((new-old)*dv/dt)**2)
                    dest_cost = next_cost[:, new, lo + shift:hi + shift]
                    improve = valid & (candidate < dest_cost)
                    pv[improve] = old
                    dest_cost[improve] = candidate[improve]
                target |= valid
        reach = next_reach
        cost = next_cost
        if parent is not None:
            history.append(parent)
        if not reach.any():
            break
    success = reach.any(axis=(1, 2))
    result = dict(variant=variant, speed_mps=speed, phases=len(phases), failed=int((~success).sum()),
                  max_lateral_speed_mps=vmax, max_lateral_accel_mps2=dv/dt,
                  dt=dt, position_grid_m=dx, reaction_s=reaction,
                  failed_phases=[round(float(p), 3) for p in phases[~success]])
    if store_witness and success.any():
        # Pick a phase with actual hazards at arrival, not an empty-road example.
        def challenge(pi):
            sides, active_count = set(), 0
            for st in tr["stations"]:
                arrival=phases[pi]+(lead+st["off"])/speed
                for b in st["bands"]:
                    tau=(arrival-b.get("phase",0)*cs)%cycle
                    if b["on"][0]*cs<=tau<b["on"][1]*cs:
                        active_count+=1
                        sides.add(int(np.sign(b["d0"]+b["d1"])))
            return len(sides)*10+active_count
        pi = int(max(np.flatnonzero(success), key=challenge))
        terminal = cost[pi] + .2*offsets[None,:]**2 + (np.arange(-4,5)[:,None]*dv)**2
        vi, xi = np.unravel_index(np.argmin(terminal), terminal.shape)
        positions = [float(offsets[xi])]
        for parent in reversed(history):
            old = int(parent[pi, vi, xi])
            if old < 0:
                raise AssertionError("broken witness")
            xi -= old + vi - 8
            vi = old
            positions.append(float(offsets[xi]))
        positions.reverse()
        result["witness"] = dict(phase=float(phases[pi]), t=[round(i * dt, 4) for i in range(len(positions))],
                                 s=[round(-lead + speed * i * dt, 4) for i in range(len(positions))], d=positions)
    return result


def flight_landing(jump, horizontal_speed, vertical_speed, gravity=9.81*1.65):
    """First descending intersection with the actual curved landing centreline.

    Horizontal and vertical speed are independent, as in runtime.lua's ramp launch.
    Checks the full descending landing curve, not a flat plane at its front edge.
    """
    gap, length = jump["gap"], jump["landing_len"]
    z = jump["z"]
    def ground(x):
        f = np.clip((x-gap)/length, 0, 1)
        return z["land0"] - (z["land0"]-z["land1"]) * (1-(1-f)**1.3)
    def height(x):
        t = x / horizontal_speed
        return z["lip"] + vertical_speed*t - 0.5*gravity*t*t
    if height(gap) < ground(gap):
        return dict(success=False, reason="below landing front", front_clearance_m=float(height(gap)-ground(gap)))
    xs = np.linspace(gap, gap+length, 1001)
    hits = np.flatnonzero((height(xs) <= ground(xs)) & (vertical_speed-gravity*xs/horizontal_speed < 0))
    if not len(hits):
        return dict(success=False, reason="overshoots landing", front_clearance_m=float(height(gap)-ground(gap)))
    k = int(hits[0]); a, b = xs[max(0, k-1)], xs[k]
    for _ in range(35):
        mid = (a+b)/2
        if height(mid) > ground(mid): a = mid
        else: b = mid
    x = float((a+b)/2)
    f = (x-gap)/length
    slope = -(z["land0"]-z["land1"])/length*1.3*(1-f)**0.3
    impact = math.atan2(gravity*x/horizontal_speed-vertical_speed, horizontal_speed)
    relative = math.degrees(impact - math.atan(-slope))
    return dict(success=True, contact_from_lip_m=round(x, 3), landing_inset_m=round(x-gap, 3),
                remaining_landing_m=round(gap+length-x, 3), time_s=round(x/horizontal_speed, 4),
                relative_impact_deg=round(relative, 3), apex_m=round(vertical_speed**2/(2*gravity), 3),
                front_clearance_m=round(float(height(gap)-ground(gap)), 3))


def controller_routes(layout, game_root):
    """Use the exact Lua module on four surface routes, with a synthetic road.

    Our simple autopilot is a witness generator, not the game's AI. No barriers,
    dynamic hazards, weather, opponents or Bullet collision are loaded.
    """
    import json
    from .drive import Kart, speed_profile
    rows = []
    edges = {e["id"]: e for e in layout["edges"]}
    for rid in ("main_lap", "pit_stop_lap", "sc_boardwalk_lap", "sc_arch_lap"):
        ee = [edges[k] for k in layout["routes"][rid]]
        xyz = np.vstack([np.asarray(e["centerline"])[:-1] for e in ee])
        widths = np.concatenate([np.asarray(e["widths"])[:-1] for e in ee])
        grips = np.concatenate([np.asarray(e["grip"])[:-1] for e in ee])
        prof = speed_profile(xyz, widths, True, Kart(v_top=25, a_eng=13.26, a_brake=26), grip=grips)
        points = ["{%s}" % ",".join(f"{v:.8f}" for v in p) for p in np.column_stack([xyz[:, :2], xyz[:, 2], widths, grips, prof["v"]])]
        rows.append("{id=%s,pts={%s}}" % (json.dumps(rid), ",".join(points)))
    harness = Path(__file__).with_name("controller_probe.lua").read_text()
    source = "package.path=" + json.dumps(str(game_root) + "/?.lua;") + "..package.path\n"
    source += "local routes={" + ",".join(rows) + "}\n" + harness
    return json.loads(lua_result(source))


def controller_jump(jump, contract, game_root, entry_speed=25, trigger_x=-1.0):
    """Exact 60-Hz controller flight with the PROPOSED pad/impulse, synthetic road.

    Also follows the landing and exit through the merge; no imaginary continuous
    ground is supplied across the gap. This does not load game track triggers.
    """
    import json
    cfg = dict(contract, gap=jump["gap"], landing_len=jump["landing_len"],
               approach_len=jump["approach_len"], ramp_len=jump["ramp_len"],
               exit_len=jump["exit_len"], entry_speed=entry_speed, trigger_x=trigger_x, **jump["z"])
    source = "package.path=" + json.dumps(str(game_root) + "/?.lua;") + "..package.path\n"
    source += "local c={" + ",".join(k+"="+str(v) for k,v in cfg.items() if isinstance(v, (float,int))) + "}\n"
    source += '''
local ctrl=require "game.core.kart.controller"
local tune=require "game.core.kart.tuning"
local function ground(x)
 if x< -c.ramp_len then return c.approach end
 if x<0 then return c.approach+(c.lip-c.approach)*((x+c.ramp_len)/c.ramp_len)^1.35 end
 if x<c.gap then return nil end
 if x<c.gap+c.landing_len then
  local f=(x-c.gap)/c.landing_len
  return c.land0-(c.land0-c.land1)*(1-(1-f)^1.3)
 end
 local f=math.min(1,(x-c.gap-c.landing_len)/c.exit_len)
 return c.land1+(c.merge-c.land1)*(3*f*f-2*f*f*f)
end
local k=ctrl.new({x=-c.boost_start_before_lip,y=c.approach+tune.ride_height,z=0,yaw=math.pi/2})
k.vx=c.entry_speed;k.speed=c.entry_speed
ctrl.boost(k,c.boost_strength,c.boost_duration)
local env={surface=function() return "road" end,ground=function(x,y,z) return ground(x),0 end}
local launched,landed=false,false
local launch_speed,contact,airtime,impact=0,0,0,0
local min_height=math.huge
for i=1,1200 do
 local oldx=k.x
 ctrl.step(k,{steer=0,throttle=1,brake=0,drift=false,jump=false},tune,
  {speed_multiplier=1,grip_multiplier=1},env,1/60,function(event,data)
   if event=="land" and launched then
    if not landed then contact=k.x;airtime=data.airtime;impact=data.impact_speed end
    landed=true
   end
  end)
 if not launched and oldx<c.trigger_x and k.x>=c.trigger_x then
  -- runtime.lua sets the explicit ramp impulse AFTER controller.step.
  -- Runtime's end_zone begins 1 m before the lip; test its spatial tolerance.
  k.vy=c.vertical_impulse;k.grounded=false;launched=true;launch_speed=k.speed
 end
 if launched then min_height=math.min(min_height,k.y) end
 if k.x>c.gap+c.landing_len+c.exit_len then
  return string.format('{"success":%s,"launch_speed_mps":%.4f,"contact_from_lip_m":%.4f,"landing_inset_m":%.4f,"airtime_s":%.4f,"impact_vertical_mps":%.4f,"grounded_at_merge":%s}',
   tostring(landed and contact>=c.gap and contact<c.gap+c.landing_len and k.grounded),launch_speed,contact,contact-c.gap,airtime,impact,tostring(k.grounded))
 end
 if k.y<c.land1-10 then break end
end
return '{"success":false}'
'''
    return json.loads(lua_result(source))
