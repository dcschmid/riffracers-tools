"""Markdown generation: track identity, gameplay design, validation report, review. Numbers come from the validated layout, texts from the track DOC."""
from __future__ import annotations

import json

ORDER = ["intro", "verse", "build_up", "drop", "breakdown", "finale"]
NAMES = dict(intro="Intro", verse="Verse", build_up="Build-Up", drop="Drop", breakdown="Breakdown", finale="Finale")


def stars(n):
    return "*" * n + "-" * (5 - n)


def identity_md(lay, doc, palette, status, fid=None):
    m = doc["music"]
    L = [f"# {lay['name']} – Track Identity", "", f"**Status:** `{status}` · **Genre:** {lay['genre']} · **Schwierigkeit:** {lay['difficulty']}/5 · **Richtung:** {lay['direction']} · **Laenge:** {lay['length_m']:.0f} m · **BPM-Richtung:** {m['bpm']}", "",
         f"> {doc['tagline']}", "", "## Identitaet", ""]
    L += [f"{p}\n" for p in doc["identity"]]
    L += ["## Umgebung", ""] + [f"- {x}" for x in doc["environment"]] + ["", "## Architektur und Formensprache", ""] + [f"- {x}" for x in doc["architecture"]]
    L += ["", "## Farbpalette", "", "| Name | Hex |", "|---|---|"] + [f"| {k} | `{v}` |" for k, v in palette.items()]
    L += ["", f"**Licht:** {doc['lighting']}", "", f"**Zuschauer:** {doc['spectators']}", "", "## Wahrzeichen und Bauwerke", ""]
    for lm in lay["landmarks"]:
        L.append(f"- **{lm['label']}** (`{lm['type']}`)")
    L += ["", "## Pitlane-Konzept", "", doc["pitlane"], "", "## Streckenrhythmus", "", doc["rhythm"], "", "## Musikalische Dramaturgie (Intro / Verse / Build-Up / Drop / Breakdown / Finale)", "",
          "| Abschnitt | Laenge | Zeit (Modell) | Tempo | Hoehe | Erlebnis |", "|---|---:|---:|---|---|---|"]
    secs = {s["key"]: s for s in lay["sections"]}
    for k in ORDER:
        s = secs[k]
        L.append(f"| {NAMES[k]} | {s['length']:.0f} m | {s['time_s']:.0f} s | {s['v_min']:.0f}–{s['v_max']:.0f} m/s | {s['z_min']:.0f}–{s['z_max']:.0f} m | {doc['dramaturgy'][k]} |")
    L += ["", "## Soundtrack-Brief (keine Musikdateien in dieser Phase)", "", f"- **Genre:** {m['genre']}", f"- **BPM-Richtung:** {m['bpm']}", f"- **Stimmung:** {m['mood']}",
          f"- **Instrumentierung:** {', '.join(m['instruments'])}", "- **Energieverlauf (0-10):**"]
    for sec, lvl, txt in m["energy"]:
        L.append(f"  - {sec}: **{lvl}/10** – {txt}")
    L += ["", "## Signature-Elemente", ""] + [f"- {x}" for x in doc["signature"]]
    L += ["", "## Concept-Art (MOSS, layout-treu)", "",
          "Die Concept-Bilder entstehen per MOSS-`img2img` aus den Blender-Renders, die ihrerseits aus dem validierten `track-layout.json` gebaut sind. Dadurch zeigen sie die tatsaechliche Streckenform, Boxengasse, Tribuenen und Fallenposition. "
          "MOSS malt Material, Licht, Architekturdetails und Publikum neu, die Komposition der Vorlage bleibt erhalten. Die Staerke wird je Bild so gewaehlt, dass die Kantenstruktur der Vorlage ueberlebt (Kantenkorrelation >= 0.62, sonst automatische Wiederholung mit geringerer Staerke).", ""]
    if fid:
        L += ["| Bild | Vorlage (Render) | Staerke | Layout-Korrelation |", "|---|---|---:|---:|"]
        names = dict(environment="environment-concept.png", pitlane="pitlane-concept.png", trap="signature-props.png (Signature-Falle am echten Ort)", crowd="Startgerade/Publikum (Moodboard-Kachel)")
        for k, v in fid.items():
            L.append(f"| {names.get(k, k)} | `{v['source']}` | {v['strength']} | {v['layout_correlation']} |")
        L += ["", "Die Korrelation ist eine Strukturpruefung (blurred Gradient-Maps), **kein Geometriebeweis**: massgeblich bleiben `validation-report.md` und `renders/geometry-overlay.png`. Einzelne Details (Gebaeudeformen, Zusatzobjekte) darf MOSS erfinden und sind kein Teil des Layouts."]
    L += ["", "Reine Text-Concepts zeigen Stimmung ohne gesicherten Layoutbezug und ersetzen diese Strukturpruefung nicht."]
    return "\n".join(L) + "\n"


def gameplay_md(lay, doc, val):
    laps = lay["lap_times"]
    L = [f"# {lay['name']} – Gameplay Design", "", f"Alle Zahlen stammen aus `track-layout.json` und sind **Modellwerte** (Kart 38 m/s, 13 m/s² quer, 9 m/s² Motor, 22 m/s² Bremse). Nicht fahrgetestet.", "",
         "## Eckdaten", "", f"- Laenge Hauptrunde: **{lay['length_m']:.0f} m**, Modell-Rundenzeit **{laps['main_lap']['time_s']:.1f} s** (Alle Optionen: {laps['all_options_lap']['time_s']:.1f} s)",
         f"- Streckenbreite: {min(lay['profile']['w']):.0f}–{max(lay['profile']['w']):.0f} m (Start/Ziel {[e for e in lay['edges'] if e['id'] == lay['main_loop'][0]][0]['widths'][0]:.0f} m), Hoehe {min(lay['profile']['z']):.0f}–{max(lay['profile']['z']):.0f} m",
         f"- Hoechsttempo (Modell) {max(lay['profile']['v']):.0f} m/s, Mindesttempo {min(lay['profile']['v']):.0f} m/s", f"- Charakter: {doc['character']}", "",
         "## Abschnitte", "", "| Abschnitt | s von–bis | Laenge | Eckenanteil | Drift-Anteil |", "|---|---|---:|---:|---:|"]
    secs = {s["key"]: s for s in lay["sections"]}
    for k in ORDER:
        s = secs[k]
        L.append(f"| {NAMES[k]} | {s['s0']:.0f}–{s['s1']:.0f} | {s['length']:.0f} m | {s['corner_share'] * 100:.0f} % | {s['drift_share'] * 100:.0f} % |")
    L += ["", "## Abkuerzungen (je echter, getrennter Fahrweg)", "", "| Name | Abzweig | Einfaedelung | Laenge | Zeitgewinn | Breite / Grip | Risiko |", "|---|---|---|---:|---:|---|---|"]
    for s in lay["shortcuts"]:
        L.append(f"| {s['name']} | {s['entrance']} | {s['exit']} | {s['length']:.0f} m statt {s['main_span_len']:.0f} m | −{s['saving_s']:.1f} s | {s['width']:.0f} m / {s['grip']:.2f} | {s['risk']} |")
    L += ["", "## Risikosprung / Sprunge", ""]
    for j in lay["jumps"]:
        L += [f"### {j['name']}", "", f"- Luecke **{j['gap']:.0f} m**, Rampe {j['ramp_len']:.0f} m · {j['ramp_deg']}° (+{j['ramp_rise']:.1f} m), Landezone {j['landing_len']:.0f} × {j['width'] + 4:.0f} m, Auslauf {j['exit_len']:.0f} m",
              f"- Referenz-Absprungtempo {j['ref_speed_mps']:.1f} m/s (Modell), benoetigt **{j['required_speed_fraction'] * 100:.0f} %** davon; Flugweite {j['flight_distance_range'][0]:.0f}–{j['flight_distance_range'][1]:.0f} m (88–105 % Tempo), Aufprallwinkel {j['impact_angle_deg']:.0f}°",
              f"- Zeitgewinn **−{j['saving_s']:.1f} s**; **sichere Alternative:** {j['safe_route']} ({j['main_span_len']:.0f} m statt {j['length']:.0f} m)", f"- Risiko: {j['risk']}", f"- Annahmen: ballistisches Modell mit Rampenwinkel und Hoehenunterschied, keine Physiksimulation.", ""]
    L += ["## Fallen", "", "Details siehe `hazards.json` und `blueprints/trap-positions.png`. Nicht im Spiel implementiert.", ""]
    for tr in lay["traps"]:
        n, e, h = tr["fairness"]["normal"], tr["fairness"]["easy"], tr["fairness"]["hard"]
        L += [f"### {tr['name']} ({'Signature' if tr['role'] == 'signature' else 'zweite Gefahr'})", "", f"- **Mechanik:** {tr['mechanic']}", f"- **Position:** s = {tr['s0']:.0f}–{tr['s1']:.0f} m, Strassenbreite {tr['width_at']:.0f} m, Anfahrtstempo (Modell) {tr['approach_speed_mps']:.0f} m/s",
              f"- **Aktivierung:** {tr['activation']}", f"- **Warnsequenz:** {tr['warning']}", f"- **Reaktionsfenster:** {tr['reaction_note']} Sichtbar ab {tr['warn_lead_m']:.0f} m ({tr['warn_lead_s']:.1f} s).",
              f"- **Gefahrenzone:** {tr['hazard_zone']}", f"- **Sichere Ausweichmoeglichkeit:** {tr['safe_evasion']}", f"- **Kart-Reaktion:** {tr['kart_reaction']}", f"- **Wetter:** {tr['weather']}",
              f"- **Visuell:** {tr['visual']}", f"- **Animation:** {tr['animation']}", f"- **Kollision:** {tr['collision']}",
              f"- **Varianten:** easy (Zyklus ×{tr['variants']['easy']['cycle_scale']}, Baender ×{tr['variants']['easy']['band_scale']}) Korridor {e['min_corridor_m']:.1f} m · normal {n['min_corridor_m']:.1f} m · hard (Zyklus ×{tr['variants']['hard']['cycle_scale']}, Baender ×{tr['variants']['hard']['band_scale']}) {h['min_corridor_m']:.1f} m",
              f"- **Fairness-Modell:** {n['paths_tested']} Durchfahrtstests ueber Phase × Tempo, {n['paths_failed']} Fehlpfade, Aktivzeit {n['max_duty'] * 100:.0f} %; bei 12 Karts bleibt der Korridor dauerhaft frei (>= {n['min_corridor_m']:.1f} m).",
              f"- **KI-Navigation:** {tr['ai']}", ""]
        if tr.get("states"):
            stt = tr["states"]
            L += [f"- **Zustandsautomat:** {stt['machine']}" + (f" (Zyklus {stt['cycle_s']} s, Beruhigung {stt['recover_s']} s)" if "cycle_s" in stt else ""),
                  f"- **Warnung:** {stt.get('warning_s', 0)} s bei {stt.get('approach_speed_mps', 0)} m/s = {stt.get('warning_distance_needed_m', 0)} m noetig, sichtbar ab {stt['warning_visible_from_m']:.0f} m ({'ok' if stt.get('warning_distance_ok', True) else 'zu kurz'}); rein zeitgesteuert, kein Zufall, kein Audio",
                  f"- **Sichere Linie in jeder Phase:** {stt['safe_line_every_phase']}"]
            if "bands" in stt:
                b0 = next(iter(stt["bands"].values()))
                L += ["- **Zeitplan eines Bandes:** " + " -> ".join(f"{r['state']} {r['duration_s']} s" for r in b0)]
        if tr.get("weather_variants"):
            L += ["- **Wettervarianten:** " + "; ".join(f"{w['weather']}: {w['visual']} ({w['gameplay']})" for w in tr["weather_variants"]), ""]
    dp = lay.get("difficulty_plan")
    if dp:
        L += ["## Schwierigkeitsplanung (12 Karts, Plan - nicht im Spiel)", "", "| Modus | Runden | Variante | Warnphase | Zyklus | Baender | aktive Stationen | gleichzeitig heisse Zonen | KI-Mix (vorsichtig/ausgewogen/aggressiv) |", "|---|---:|---|---:|---:|---:|---|---:|---|"]
        for m_ in dp["modes"]:
            h0 = next(iter(m_["hazards"].values()))
            L.append(f"| {m_['name']} | {m_['laps']} | {m_['variant_ref']} | x{h0['warning_scale']} | x{h0['cycle_scale']} | x{h0['band_scale']} | {m_['hazard_hits_per_lap_model']} | {m_['max_zones_hot_at_once']} | {m_['ai_mix']['cautious']}/{m_['ai_mix']['balanced']}/{m_['ai_mix']['aggressive']} |")
        L += ["", "Notiz je Modus: " + "; ".join(f"**{m_['name']}**: {m_['note']}" for m_ in dp["modes"]), "", "**KI-Risikoprofile:** " + "; ".join(f"**{k}** (Abstand {v['margin_m']} m, Reaktion {v['reaction_s']} s, Abkuerzung {int(v['shortcut'] * 100)} %, Sprung {int(v['jump'] * 100)} %): {v['note']}" for k, v in dp["ai_profiles"].items()), "",
              f"Abstand der Fallenzonen entlang der Runde (m): {dp['trap_zone_gaps_m']}.", ""] + [f"- {r}" for r in dp["rules"]] + [""]
    L += ["## Ueberholmoeglichkeiten", ""] + [f"- s = {o['s']:.0f} m: {o['v_in']:.0f}→{o['v_out']:.0f} m/s, {o['width']:.0f} m breit, Bremsweg {o['braking_len']:.0f} m" for o in lay["overtaking"]]
    p = lay["pit"]
    L += ["", "## Pitlane", "", f"- 12 Boxen, Raster {p['box_spacing']} m, Gasse {2 * p['lane_half']:.0f} m, Einfahrt s = {p['entry_s']:.0f} m (vor S/Z), Ausfahrt s = {p['exit_s']:.0f} m (hinter S/Z), 60 km/h, Einbahn",
          f"- Pit-Stop-Runde (Modell, Limit nicht modelliert): {laps['pit_stop_lap']['time_s']:.1f} s", f"- {doc['pit_design']}", "", "## KI-Hinweise fuer die spaetere Integration", ""] + [f"- {x}" for x in doc["ai_notes"]]
    return "\n".join(L) + "\n"


def validation_md(lay, val, verification=None):
    L = [f"# {lay['name']} – Geometrie- und Validierungsbericht", "",
         f"**Ergebnis:** {'GEOMETRY_VALIDATED' if val['geometry_validated'] else 'NICHT VALIDIERT'} – {val['total']} Pruefungen, {val['critical']} kritisch, {val['warnings']} Warnungen, {val['fails']} FAIL.", "",
         "Gepruefte Daten: `planning/track-layout.json` (gerichteter Graph, 3D-Mittellinien). Die Pruefung arbeitet auf den Layoutdaten, **nicht** auf Bildern. Modellpruefungen sind als *Modell* markiert.", "",
         "| # | Status | Pruefung | Ergebnis |", "|---:|---|---|---|"]
    for c in val["checks"]:
        d = c["detail"].replace("|", "/").replace("\n", " ")
        L.append(f"| {c['id']} | **{c['status']}**{' (Modell)' if c['model'] else ''} | {c['name']} | {d} |")
    L += ["", "## Annahmen und Grenzen", "", "- Fahrmodell: Hoechsttempo 38 m/s, Querbeschleunigung 13 m/s², Motor 9 m/s², Bremse 22 m/s², Ideallinienbonus 0.35 × Breite; Grip-Faktoren je Belag. Das ist ein Auslegungsmodell, **keine** Physiksimulation des Spiel-Karts.",
          "- Sprungkinematik: ballistisch mit Rampenwinkel und Hoehenunterschied zwischen Absprung und Landezone; Reibung, Kart-Federung und Landungsimpuls sind nicht modelliert.",
          "- Fallen: datenbasiertes Zeitplanmodell (Baender × Phasen × Tempo, Kart 2.2 m breit, 6 m/s Querbewegung); keine Simulation mit zwoelf Karts.",
          "- **Die Strecke ist nicht fahrgetestet.** Ein Sichtvergleich ersetzt keine Fahrprobe im Spiel."]
    if verification:
        L += ["", "## Render gegen Geometrie", "", f"- IoU Strassenflaeche Blender-Render vs. Layout: **{verification['iou']:.3f}**; Render ausserhalb Layout {verification['render_outside_layout'] * 100:.1f} %, Layout fehlt im Render {verification['layout_missing_in_render'] * 100:.1f} % → {'VERIFIZIERT' if verification['verified'] else 'ABWEICHUNG'}",
              "- Verfahren: Top-down-Orthorender nur der Strassenflaechen (ID-Maske) gegen die aus dem Layout gerasterten Strassenpolygone (`renders/geometry-overlay.png`)."]
    return "\n".join(L) + "\n"


def review_md(lay, rv, status, val):
    W = dict(logic=0.25, tension=0.20, lowpoly=0.20, music=0.15, features=0.10, pit=0.10)
    lab = dict(logic="Logischer Streckenverlauf", tension="Gameplay-Spannung", lowpoly="Premium-Low-Poly-Qualitaet", music="Musikalische Identitaet", features="Fallen, Spruenge, Abkuerzungen", pit="Pitlane und Umgebung")
    total = sum(rv["scores"][k] * W[k] for k in W)
    L = [f"# {lay['name']} – Review", "", f"**Status:** `{status}`  (CONCEPT_APPROVED wird nicht automatisch vergeben)", "", "| Kriterium | Gewicht | Note (0–10) | Begruendung |", "|---|---:|---:|---|"]
    for k in W:
        L.append(f"| {lab[k]} | {int(W[k] * 100)} % | {rv['scores'][k]:.1f} | {rv['why'][k]} |")
    L += ["", f"**Gewichtete Gesamtnote: {total:.1f} / 10**", f"**Kritische geometrische Fehler:** {val['critical']} (blockieren die Freigabe unabhaengig vom Durchschnitt)", "",
          "## Kritische Selbstpruefung", ""] + [f"- {x}" for x in rv["critique"]]
    L += ["", "## Offene Punkte", ""] + [f"- {x}" for x in rv["open"]]
    L += ["", "## Nicht belegt", "", "- keine Fahrtests mit dem Spiel-Kart, keine Kollisionstests, keine KI-Rennen mit 12 Karts", "- Zuschauer und Props sind prozedurale Low-Poly-Formen, keine finalen Spiel-Assets"]
    return "\n".join(L) + "\n", total
