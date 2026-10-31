"""Helpers to author trap definitions with the standard field set used by the track pipeline (name, mechanic, position, activation, warning, reaction window,
hazard zone, safe evasion, kart reaction, weather, visual, animation, collision, difficulty variants, AI requirements)."""
from __future__ import annotations

DEFAULT_VARIANTS = dict(easy=dict(cycle_scale=1.25, band_scale=0.85, warn_scale=1.2, note="langsamerer Zyklus, schmalere Gefahrenzone, 20 % laengere Warnphase"),
                        normal=dict(cycle_scale=1.0, band_scale=1.0, warn_scale=1.0, note="Referenzwerte"),
                        hard=dict(cycle_scale=0.85, band_scale=1.1, warn_scale=1.0, note="schnellerer Zyklus, 10 % breitere Gefahrenzone, Warnphase unveraendert"))


def trap(id, name, role, a, b, cycle, stations, *, kind, mechanic, activation, warning, hazard_zone, safe_evasion, kart_reaction, weather, visual, animation, collision,
         ai, warn_lead_m, on="main", variants=None, reaction_note="", extra=None):
    t = dict(id=id, name=name, role=role, kind=kind, on=on, a=a, b=b, cycle=cycle, stations=stations, mechanic=mechanic, activation=activation, warning=warning,
             visual_warning=True, hazard_zone=hazard_zone, safe_evasion=safe_evasion, kart_reaction=kart_reaction, weather=weather, visual=visual, animation=animation,
             collision=collision, ai=ai, warn_lead_m=warn_lead_m, variants=variants or DEFAULT_VARIANTS, reaction_note=reaction_note)
    if extra:
        t.update(extra)
    return t


def band(id, d0, d1, on, warn, phase=0.0, length=8.0):
    return dict(id=id, d0=d0, d1=d1, on=on, warn=warn, phase=phase, len=length)


def static_trap(id, name, role, a, b, *, grip, grip_hard, warn_lead_m, kind, mechanic, activation, warning, hazard_zone, safe_evasion, kart_reaction, weather, visual, animation, collision, ai, reaction_note="", on="main"):
    return dict(id=id, name=name, role=role, kind=kind, on=on, a=a, b=b, static=True, grip=grip, grip_hard=grip_hard, cycle=0.0, stations=[], mechanic=mechanic, activation=activation, warning=warning,
                visual_warning=True, hazard_zone=hazard_zone, safe_evasion=safe_evasion, kart_reaction=kart_reaction, weather=weather, visual=visual, animation=animation, collision=collision, ai=ai,
                warn_lead_m=warn_lead_m, variants=dict(easy=dict(cycle_scale=1.0, band_scale=1.0, note=f"Grip {min(grip + 0.12, 0.95):.2f}"), normal=dict(cycle_scale=1.0, band_scale=1.0, note=f"Grip {grip}"),
                                                       hard=dict(cycle_scale=1.0, band_scale=1.0, note=f"Grip {grip_hard}")), reaction_note=reaction_note)
