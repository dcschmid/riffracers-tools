"""Document the completed external source-backed preparation; never approve a concept."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from world_create import get_project


def write(track,rel,text):
    path=track/rel;path.parent.mkdir(exist_ok=True,parents=True);path.write_text(text.strip()+'\n')


def main():
    p=get_project();mod=sys.argv[1];m=p.import_track(mod);track=p.track_dir(mod);planning=track/'planning';pack=p.cache_dir/m.ID/'pack'
    s=json.loads((planning/'source-track.json').read_text());val=json.loads((planning/'validation.json').read_text());rv=json.loads((planning/'render-verification.json').read_text());fidelity=json.loads((planning/'layout-fidelity.json').read_text());metrics=json.loads((planning/'scene-metrics.json').read_text());cams=json.loads((pack/'cams.json').read_text());theory=json.loads((planning/'drivability-audit.json').read_text());routes=json.loads((planning/'shortcut-witnesses.json').read_text())
    if not rv['all_hq_images_valid'] or not rv['all_originals_unchanged'] or not rv['decoration_pass'] or not rv['mask_pass']:raise SystemExit('Verification incomplete')
    if not json.loads((planning/'blend-inspection.json').read_text())['passed']:raise SystemExit('Saved Blender inspection incomplete')
    if not theory['timing_proposal_passed'] or not theory['all_hazard_centerlines_clear'] or not theory['all_sampled_landings_on_road'] or not routes['all_routes_have_witness']:raise SystemExit('Theoretical witnesses incomplete')
    if len(fidelity)!=4 or any(r['layout_correlation']<.62 for r in fidelity.values()):raise SystemExit('MOSS verification incomplete')
    c=np.array([[v['x'],v['z']] for v in s['points']]);ds=np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1);seconds=float(sum(ds/np.array([v['speed'] for v in s['points']])))
    source_refs=json.loads((planning/'preproduction-source-baseline.json').read_text())['references']
    references='\n'.join(f'- [{r["file"]}]({r["file"]})' for r in source_refs)
    lookdev_header=''
    lookdev_short=''
    if (planning/'current-art-review.json').is_file():
        current=json.loads((planning/'current-art-review.json').read_text())
        current_links='\n'.join(f'- [{label}]({file})' for label,file in current['highlights'])
        lookdev_header=f'''## Aktuell: {current['title']}

{current['summary']}

{current_links}

[Aktuelles Review]({current['review']}) · [Aktueller Master]({current['master']}). Die nachfolgende 27er-Serie ist als technische Basis gesondert dokumentiert.
'''
        lookdev_short=f"**Aktuelle Weiterarbeit:** {current['summary']} Einzelheiten: {current['review']}. Die nachfolgende 27er-Serie ist die technische Basis.\n"
    elif (planning/'festival-lookdev-review.md').is_file():
        manifest_path=planning/'festival-lookdev-manifest.json'
        lookdev_manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        lookdev_count=len(lookdev_manifest.get('images',[])) or len(list(track.glob('iron-chorus-lookdev-*.png')))
        lookdev_header=f'''## Aktuell: Festivalfassung und Detailtests

Die eigene Overview- und Abschnittsvorlage wird in einer separaten Blender-Umgebung ausgearbeitet. Aktuell liegen {lookdev_count} native 4K-Testbilder als echte PNGs direkt im Hauptordner. Die jeweilige Revision, Fallenansichten, Prüfungen und Grenzen sind im aktuellen Review beschrieben.

- [Neue Gesamtansicht](iron-chorus-lookdev-overview.png)
- [Festivalmarkt und Besucher](iron-chorus-lookdev-festival.png)
- [Canyon und Aufenthaltsbereiche](iron-chorus-lookdev-canyon.png)

[Aktuelle Gestaltung, Fallen und Grenzen](planning/festival-lookdev-review.md) · [Festival-Master](blender/festival-lookdev.blend) · [Dateimanifest](planning/festival-lookdev-manifest.json).

Die nachfolgende 27er-Bildserie gehört zur technischen Basis. Insgesamt liegen hier 39 Basis-/Referenzbilder plus {lookdev_count} gesonderte Festivaltests. Der vollständige erneuerte Bildsatz entsteht nach Rückmeldung zum Teststand.
'''
        lookdev_short=f'**Aktuelle Weiterarbeit:** Der separate Festival-Master und {lookdev_count} neue `iron-chorus-lookdev-…`-Bilder dokumentieren den aktuellen Teststand. Die hier beschriebene 27er-Serie gehört zur technischen Basis; Einzelheiten stehen in planning/festival-lookdev-review.md.\n'
    write(track,'README.md',f'''
# Iron Chorus – vorbereiteter visueller Prototyp

Stand 25.10.2026 · **REVIEW_REQUIRED** · externe Phase 1.

{lookdev_header}

Unsere neue [Overview-Vorlage](iron-chorus-looktest-overview-v1.png) und die sechs eigenen Abschnittsbilder geben die Gestaltung vor. Der neue Blender-Prototyp übernimmt den bestehenden Streckenverlauf und konkretisiert die Festivalrichtung: Sandstein-Canyon, gehörnte Schädelbühne, Gitarrenbrücke, Verstärkerwände, Schmiede und Backstage. Seine Ausstattung ist noch einfacher und weniger dicht als im KI-Zielbild.

## Vollständige Bildserie der technischen Basis

**27 echte 4K-Blender-PNGs (3840 × 2160)**, vier MOSS-Repaints (1024 × 576) und ein Moodboard (2400 × 1500). Zusätzlich bleiben die sieben bisherigen KI-Vorlagen unverändert erhalten: insgesamt 39 Ergebnis- und Referenzbilder im Hauptordner. Keine Bildverknüpfungen.

- [Gesamtansicht](iron-chorus-hero.png) · [Gegenseite](iron-chorus-isometric.png) · [Draufsicht](iron-chorus-birdseye.png)
- [Bühne](iron-chorus-stage.png) · [Start](iron-chorus-grid-hero.png) · [Boxengasse](iron-chorus-pit-lane-high.png)
- [Publikum](iron-chorus-crowd.png) · [Gesichter](iron-chorus-faces.png)
- [Gitarrenbrücke](iron-chorus-guitar-bridge.png) · [Ladespirale](iron-chorus-loading-spiral.png) · [Amp Canyon](iron-chorus-amp-canyon.png) · [Amp Cut](iron-chorus-amp-cut.png)
- Canyon-Lifts: [A Warnung](iron-chorus-canyon-lift-a.png) · [B aktiv](iron-chorus-canyon-lift-b.png) · [C Gegenphase](iron-chorus-canyon-lift-c.png)
- [Schmiede](iron-chorus-forge.png), Pressen: [A Warnung](iron-chorus-forge-press-a.png) · [B aktiv](iron-chorus-forge-press-b.png) · [C Gegenphase](iron-chorus-forge-press-c.png)
- [Backstage](iron-chorus-backstage.png) · [Backstage Cut](iron-chorus-backstage-cut.png)
- Speaker-Rampen: [Anfahrt](iron-chorus-leap-front.png) · [Seite](iron-chorus-leap-side.png) · [Landung](iron-chorus-leap-landing.png) · [Modell-Flugbahn](iron-chorus-leap-ghost.png)
- [Finale](iron-chorus-finale.png) · [Pyro aktiv](iron-chorus-pyro-active.png)
- MOSS: [Umgebung](environment-concept.png) · [Boxen](pitlane-concept.png) · [Falle](signature-props.png) · [Publikum](crowd-concept.png) · [Moodboard](moodboard.png)

## Technische Vorbereitung

- [Blender-Master mit 27 benannten Kameras](blender/track-preview.blend), [Hinweise zur Datei](blender/README.md); ungegradete 4K-Originale in `blender/raw-renders/`.
- [Layoutplan](blueprints/topdown-annotated.png), [saubere Draufsicht](blueprints/topdown-clean.png), [Boxengasse](blueprints/pitlane-detail.png), [Abkürzungen/Sprünge](blueprints/shortcuts-jumps.png), [Fallenpositionen](blueprints/trap-positions.png).
- [Fallenphasen und freie Spur](blueprints/trap-phases.png) · [Rampenquerschnitte](blueprints/jump-sections.png).
- [19 Quellprüfungen](planning/validation-report.md) · [Fahrbarkeitsmodell](planning/drivability-audit.md) · [Quelle gegenüber neuem Plan](planning/source-vs-plan.md) · [Straßenabgleich](planning/geometry-overlay.png) · [Review](review.md).

1948,6 m Hauptstrecke, 368 Quellpunkte, zwölf Startplätze, **zehn bestehende Serviceboxen**, zwei echte Abkürzungen, zwei Rampenvarianten auf durchgehender Straße. Die Boxenzahl und Strecke werden aus Iron Chorus übernommen. Rundenzeit aus Quell-Tempolimits: ca. {seconds:.1f} s, eine reine Integration dieser Richtwerte ohne Beschleunigung, Verkehr oder Fahrfehler.

## Ehrliche Grenzen

Die neue Dekoration überlappt im konservativen Flächenmodell keine Straße; Masken-IoU **{rv['mask_iou']:.4f}**. Beide Abkürzungen besitzen einen statischen Korridorzeugen. Beide Rampen landen im vereinfachten Flugmodell bei 18/22/25/30/34/38 m/s auf der Quellstraße; die Bonuszone wird bei manchen Tempi verfehlt. Der neue [Timingplan](planning/hazard-timing-proposal.json) hebt die fünf Pressenwarnungen von 0,9 auf 1,2 s an; die Spielquelle ist noch unverändert.

Kein neuer Fahrtest oder Defold-Lauf, keine Spielintegration, keine fertigen Spielassets. Rund {metrics['expanded_visible_instance_triangles']/1e6:.2f} Mio. sichtbare Dreiecke mit aufgefalteten Instanzen: diese Szene dient der Visualisierung, ein Engine-Budget ist noch offen. MOSS kann Schrift und Geometrie verändern; die Blender-Bilder und Prüfberichte sind maßgeblich. Fallenbilder zeigen repräsentative Zustände, keine getestete Animation.

Die bisherigen 16 Spielaufnahmen liegen unverändert in `01-Overview/`, `02-Isometric/` und `03-Highlights/`. Im Spielprojekt wurde nichts geschrieben, exportiert, registriert oder committet. Eine spätere Integration braucht Daniels ausdrückliche Freigabe.

## Erhaltene Gestaltungsvorlagen

{references}
''')
    write(track,'review.md',f'''
# Iron Chorus – Review des vorbereiteten Prototyps

{lookdev_short}

Status **REVIEW_REQUIRED**. Daniels positive Rückmeldungen bestätigen die Gestaltungsrichtung der bisherigen sieben KI-Bilder. Die neu aufgebaute Blender-Fassung und ihre Modellnachweise sind ein eigener, noch zu betrachtender Stand. CONCEPT_APPROVED wird nicht automatisch vergeben.

## Ergebnis

Der vorhandene Streckenverlauf ist als externe Szene mit 27 Kameras und vollständigem 4K-Bildsatz aufgebaut. Verwendet werden nur Iron-Chorus-Motive und eigene generische Festivalfiguren. Unsere Overview-Vorlage ist als Referenz dokumentiert, nicht als Pixel- oder Geometrienachweis. Bühne, erhöhte Gitarrenstraße, Amp Canyon, Schmiede, Backstage, Rampen und Boxengasse sind unterscheidbar ausgestattet. Die Landschaft und Besucherdichte bleiben einfacher als im KI-Zielbild. Die Figuren haben einfache geometrische Gesichter; es sind keine Fahrerassets. Die Quellelemente der bisherigen Umgebung sind als Vorschau übernommen, nicht zu fertigen neuen Spielassets umgebaut.

## Prüfungen

- 19/19 Prüfungen des ausdrücklich beschriebenen bestehenden Quellprofils bestanden. Das sind nicht automatisch dieselben 19 Regeln wie bei einer neu geplanten Welt-2-Strecke.
- Gerenderte Straßenprojektion gegen denselben Quellplan: IoU {rv['mask_iou']:.4f}, Mindestziel 0,93.
- Konservative Kreis-Fußabdrücke der neuen Zuschauer- und Dekoflächen: keine Überlappung mit Hauptstraße, Boxengasse oder Abkürzungen; Details in decoration-validation.json.
- Freie Mittellinie gegen alle neun maximalen Fallen-Fußabdrücke: 3,10–3,60 m Reserve nach 0,85 m Kartradius plus 0,30 m Abstand. Es wird kein erzwungener Spurwechsel angenommen.
- Abkürzungen: beide im statischen Modell mit 22 m/s vorwärts und höchstens 6 m/s seitlich erreichbar; Quell-Kollisionsboxen berücksichtigt. Kein Fahrzeug-, Lenk- oder Verkehrsmodell.
- Zwei Rampen × sechs Tempi: Landung auf der Quellstraße. Normalbonus bei 38 m/s und Risikobonus bei 18/34/38 m/s verfehlt. Eine sichere Straßenlandung und ein Bonus sind getrennte Aussagen.
- Der lesende Quell-Kollisionsaudit meldet 0 Barrierelücken, 0 Löcher in der erhöhten Straße bei 37.941 Proben und 0 ungeschützte feste Props. Drei Mesh-Eingriffe in Abkürzungen wurden nur in dieser externen Szene versetzt (1,5 / 4,5 / 6,0 m; scenery-overrides.json). Die 103 Pitlane-Informationsmeldungen schließen eigene Pitwände und dünne Schilder ein und sind nicht als neue Fehlermeldungen umgedeutet worden; keine vollständige Neuprüfung sämtlicher Quellrequisiten.
- Vier MOSS-Repaints bestehen den strukturellen Kantenvergleich ≥ 0,62; dieser Vergleich beweist keine Geometrie.
- Hashvergleich: drei Spielquellen, sieben Zielbilder und 16 bisherige Spielaufnahmen unverändert.

## Vor einer Integration offen

Der neue Timingplan sieht 1,2 s Pressenwarnung bei unveränderter 3,24-s-Periode vor; er verkürzt dafür die Ruhephase auf 0,86 s. Die alte Spielquelle hat weiterhin 0,9 s und wurde nicht verändert. Umsetzung des Plans, reale Warnlesbarkeit und tatsächlicher Fallentakt bleiben vor Integration zu prüfen. Quell-Rampenauslösung in einem 2-m-Fenster um die Lippe, reale Kollisionsformen, Boost- und Landeverhalten, Verkehr, Wetter und Rückkehr nach Fahrfehlern sind nicht getestet. Die Landungsrechnung startet idealisiert an der Lippe bei voller Rampenhöhe. Die Korridorprüfung ersetzt keinen Fahrtest.

Die Blender-Datei speichert den Warnzustand A. Wechsel der Kamera allein schaltet die Falle nicht in B/C; der Renderer setzt diese Posen gezielt. Linien der Flugbahn sind nur in der dafür vorgesehenen Ansicht sichtbar. MOSS-Schriften können verzerrt sein. Es gibt kein festgelegtes Engine-Polybudget; die Vorschau ist nicht für den direkten Export optimiert.

Nächster fachlicher Schritt: Daniels Sichtprüfung des Blender-Bildsatzes, danach gegebenenfalls Feinschliff. Spielintegration erst auf gesonderten ausdrücklichen Auftrag.
''')
    write(track,'planning/game-design.md',f'''
# Iron Chorus – Fahr- und Abschnittsplan

Heavy Metal · 148 BPM · bestehender 1948,6-m-Ring · 368 Quellpunkte · 18 geordnete Gates · zwölf Startplätze · zehn Serviceboxen.

| Quellknoten | Abschnitt | Eigene Umgebung und Funktion |
|---|---|---|
| 1–53 | Main Stage Arena | Gehörnte Schädelbühne, Startgerade, Konzertpublikum, Pyro mit freier Mitte. |
| 54–124 | Amp Canyon | Verstärkerformationen und Sandstein, seitliche Lifts, Amp Cut als eigene Straße. |
| 125–223 | Guitar Bridge | Erhöhter Gitarrenkörper, Bundstraße, Stahlstützen und freigehaltene Unterführung. |
| 224–259 | Forge | Pressenrahmen, geschützte Fußabdrücke, Signale und durchgehende Leitspur. |
| 260–304 | Backstage | Tourbusse, Cases, Crewbereiche, Backstage Cut mit vier Quellkisten. |
| 305–345 | Speaker Jump | Breite niedrige / schmale höhere Rampe; freier Streifen dazwischen; durchgehende Straße darunter. |
| 346–362 | Finale | Rückführung zur Arena, Zuschauer und sichtbarer Pit-Abzweig. |
| 363–368 | Grid/Pit Return | Zusammenführung und Startgerade; einseitige Quell-Boxengasse. |

Die Hauptstraße besitzt 13–24 m Kernbreite plus je 4 m Quellschulter; genaue Werte stehen im Layout. Zwei eigene Straßen: Amp Cut 6,4 m und Backstage Cut 6,0 m. Die Pit-Verbindungen bleiben mit ihrem absichtlichen 4-m-Mittellinienversatz erhalten. Die Kreuzung besitzt getrennte Höhenebenen.

Runden-Richtwert {seconds:.2f} s = Summe der horizontalen Quellsegmente geteilt durch ihre jeweiligen Tempo-Hinweise (16–25 m/s). Kein Fahrtest, keine Beschleunigungs- oder KI-Simulation. Keine Abkürzungsgewinne erfunden.

Fallenmodell und Rampenwerte: [drivability-audit.md](drivability-audit.md). Abkürzungs-Korridore: [shortcut-witnesses.json](shortcut-witnesses.json). Quell-Kollisionen: [source-collision-audit.txt](source-collision-audit.txt). Externe Requisitenkorrekturen: [scenery-overrides.json](scenery-overrides.json).
''')
    write(track,'planning/visual-identity.md','''
# Iron Chorus – visuelle Identität

Verbindliche eigene Gestaltungsreferenz: iron-chorus-looktest-overview-v1.png plus sechs Iron-Chorus-Abschnittsbilder, alle im Strecken-Hauptordner erhalten. Keine Motive aus Coastal oder den anderen Metal-Strecken übernehmen.

Warmer roter Sandstein, sonniger blauer Himmel, schwarze Verstärker mit roten und silbernen Flächen, warme Bühnenlichter. Schichten im Fels und Saguaro-Silhouetten rahmen das Festival. Zuschauer stehen auf Terrassen oder außerhalb der Straßenreserven. Eigene generische Figuren besitzen Augen, Nase, Mund, Haare und variierende Posen. Marktstände, Tourbusse, Racks und bestehende Festivalrequisiten verbinden die einzelnen Abschnitte. Keine Karts in neuen Bildern.

Die Vorlage zeigt das ambitioniertere Ziel. Die Blender-Fassung konkretisiert Maßstab und Anschlüsse mit vereinfachter Landschaft und Ausstattung. Dieser Unterschied wird nicht als gleichwertige Endproduktion dargestellt. MOSS ist ein zusätzlicher Look-Repaint auf den neuen Blender-Bildern und darf für technische Entscheidungen keine Geometriequelle werden.
''')
    write(track,'blender/README.md',f'''
# Iron Chorus – externer Blender-Master

{lookdev_short}

`track-preview.blend` ist eine eigenständige Vorschau, die vorhandene Spielassets lesend importiert und um neue externe Festivalgestaltung ergänzt. Keine Fahrer-/Kart-Assets. Eingebettete Geometrie, Vertexfarben und Schrift; 27 benannte Kameras `VIEW::…`, aktive Kamera `VIEW::hero`. Achsen: Spiel (x,y,z) → Blender (x,−z,y), Einheiten Meter. Kein Defold-Export.

Die gespeicherte Szene zeigt Warnzustand A. Die Kameras B/C haben ihren Zustand als Eigenschaft; Kamerawechsel allein verändert keine Objekte. Das gemeinsame Renderrezept setzt den Zustand vor jedem Bild. `ghost` ist standardmäßig unsichtbar und nur für leap-ghost eingeblendet. `FAN::` sind generische Vorschaufiguren; `SRC::` sind Instanzen gelesener Umgebungsassets; `VIEW::` sind Ausgabekameras.

{metrics['fan_instances']} Zuschauerinstanzen, {metrics['source_instances']} Quellinstanzen. {metrics['unique_mesh_triangles']:,} Dreiecke in den eindeutigen Meshdaten und ca. {metrics['expanded_visible_instance_triangles']:,} bei aufgefalteten sichtbaren Instanzen. Das ist eine Visualisierung ohne definiertes Engine-Budget; vor Produktion sind LOD, Besucherbudget, Kollisionsformen und Materialkosten festzulegen.

Ungegradete 4K-Originale liegen in `raw-renders/`; die freigehaltenen Ergebnisdateien liegen direkt im Streckenordner. Das Grading erzeugt echte PNG-Dateien, keine Links. Keine .blend-Sicherung enthält die laufende Arbeit aus der anderen Blender-Sitzung.

Reproduktion aus dem gemeinsamen `_pipeline`-Ordner:

```bash
PY=~/.local/share/local-image-ai/venv/bin/python
$PY prepare_source.py --world ../World-01-Metal t01_iron_chorus
$PY run_source.py --world ../World-01-Metal t01_iron_chorus mask
$PY run_source.py --world ../World-01-Metal t01_iron_chorus quick hero,stage
$PY run_source.py --world ../World-01-Metal t01_iron_chorus hq
$PY verify_source.py --world ../World-01-Metal t01_iron_chorus
$PY source_diagrams.py --world ../World-01-Metal t01_iron_chorus
$PY w2lib/concepts_layout.py --world ../World-01-Metal t01_iron_chorus
$PY source_docs.py --world ../World-01-Metal t01_iron_chorus
```

GPU-Aufrufe laufen über die weltübergreifende Sperre nacheinander. Quick-Ausgaben bleiben im Weltcache. `prepare_source.py` schreibt den Quellenprüfstatus neu; die abschließende Doku setzt nach verifizierten Bildern wieder REVIEW_REQUIRED, niemals CONCEPT_APPROVED.
''')
    (planning/'cameras.json').write_text(json.dumps(cams,indent=2))
    inspection=json.loads((planning/'blend-inspection.json').read_text());assets=[]
    for name in inspection['source_asset_files']:
        file=Path(name);assets.append(dict(path=name,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (planning/'source-assets.json').write_text(json.dumps(dict(scope='read-only source asset provenance for the external preview; no export or asset modification',assets=assets),indent=2))
    state=json.loads((planning/'status.json').read_text()) if (planning/'status.json').exists() else {}
    state.update(status='REVIEW_REQUIRED',geometry_validated=True,visualized=True,source_profile=val['profile'],decoration_validated=True,blender_master=True,hq_images=27,moss_images=4,moodboards=1,reference_images=7,theoretical_witnesses=True,human_driving_test=False,engine_simulation=False,integration_authorized=False,concept_approved_by_owner=False)
    (planning/'status.json').write_text(json.dumps(state,indent=2))
    rows=[]
    for file in sorted(track.glob('*.png')):
        if file.is_symlink():raise ValueError('Unexpected image link: '+str(file))
        with Image.open(file) as im:im.verify()
        with Image.open(file) as im:size=list(im.size)
        rows.append(dict(file=file.name,resolution=size,bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),regular_file=True,png_verified=True))
    base_rows=[r for r in rows if not any(tag in r['file'] for tag in ('-lookdev-','-concept3-','-stage4-','-polish5-'))];lookdev_rows=[r for r in rows if '-lookdev-' in r['file']]
    if len(base_rows)!=39:raise ValueError('Expected 27+4+1+7 base/root PNGs, plus explicitly named lookdev tests')
    (planning/'preproduction-manifest.json').write_text(json.dumps(dict(status='REVIEW_REQUIRED',images=base_rows,lookdev_tests=lookdev_rows,concept_environment_tests=[r for r in rows if '-concept3-' in r['file']],stage_road_tests=[r for r in rows if '-stage4-' in r['file']],polish_tests=[r for r in rows if '-polish5-' in r['file']],master='blender/track-preview.blend',human_driving_test=False,game_written=False),indent=2))
    print('Wrote complete preparation docs and manifest:',len(base_rows),'base PNGs,',len(lookdev_rows),'lookdev tests; REVIEW_REQUIRED')


if __name__=='__main__':main()
