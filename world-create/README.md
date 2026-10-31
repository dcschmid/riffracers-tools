# world-create – Riff-Racers-Vorproduktionswerkzeuge

Wiederverwendbare Programme für die Vorproduktion mehrerer Welten und Strecken. Bilder, Streckendefinitionen, spezielle Szenen und Zwischendaten liegen bei der jeweiligen Welt; das Spielprojekt wird von world-create nicht geändert. Der optionale Controller-Audit liest seine Quellen nur.

world-create ist ein eigenständiges Projekt unter `dev/github/riffracers-tools/`. Die Welten liegen separat (z. B. `~/Pictures/AI-Generated/Riff-Racers/World-0X-…`) und wählen world-create über `--world`, `RIFF_WORLD_ROOT` oder die Standardkonfiguration aus.

## Ablage

```text
riffracers-tools/
  world-create/                      gemeinsame Programme und Bibliotheken
    world_create.py                  Welt auswählen, Pfade prüfen, GPU-Sperre
    run_stage.py / dev.py            Layout, Prüfung, Daten, Render und Dokumente
    run_hq.sh / run_hq.py            4K-Render und Grading
    w2lib/                           Geometrie, Validierung, Diagramme, MOSS
    w2blend/                         Szenenbau und wiederverwendbare Bausteine
    templates/riff-world.json        Vorlage für eine weitere Welt
    world-create-defaults.json       lokale Standardwelt, Spiel- und Werkzeugpfade
  World-02-Electro-Groove/           (Beispielwelt, liegt außerhalb von world-create)
    riff-world.json                  Konfiguration und Streckenverzeichnis
    _project/tracks/                 Definitionen dieser Welt
    _project/w2blend/                ihre speziellen Umgebungen und Motive
    _project/scripts/                ihre Zusatzprüfungen und Welt-Dokumente
    _project/out/                    nur ihre Packs, Masken und Vorschauen
    01-coastal-sprint/               aktuelle Bilder und Prüfberichte
```

Die Namen `w2lib` und `w2blend` sind aus Kompatibilitätsgründen erhalten; ihre gemeinsamen Funktionen benötigen keinen festen Welt-2-Pfad. Weltlokale `w2blend`-Module ergänzen die gemeinsamen Module als Python-Namespace. Dort kein `__init__.py` anlegen. `tracks/` enthält dagegen ein leeres `__init__.py` und gehört ausschließlich zur ausgewählten Welt.

## Ausgabe pro Welt und Strecke

Jede Welt hat einen eigenen Cache-Ordner (`cache_dir`, Standard `_project/out`), jede Strecke einen eigenen Ausgabeordner (`folder` in `riff-world.json`, Standard `OUT_FOLDER` im Streckenmodul). Packs, Masken, Vorschauen und Zwischendaten bleiben dadurch pro Welt und pro Strecke getrennt; es gibt keinen gemeinsamen Ausgabeordner.

## Eine Welt auswählen

Reihenfolge: `--world PATH`, danach `RIFF_WORLD_ROOT`, danach `riff-world.json` im aktuellen Ordner oder seinen Eltern, zuletzt `world-create-defaults.json`. Ein erkennbarer, aber unkonfigurierter `World-*`-Ordner fällt nicht still auf eine andere Welt zurück. `default_world` in `world-create-defaults.json` darf ein relativer Pfad (relativ zum Werkzeugordner), ein `~`-Pfad oder ein absoluter Pfad sein.

```bash
cd ~/dev/github/riffracers-tools/world-create
PY=~/.local/share/local-image-ai/venv/bin/python
$PY world_create.py --world ~/Pictures/AI-Generated/Riff-Racers/World-02-Electro-Groove info
$PY dev.py --world ~/Pictures/AI-Generated/Riff-Racers/World-02-Electro-Groove t01_coastal_v3
$PY run_stage.py --world ~/Pictures/AI-Generated/Riff-Racers/World-02-Electro-Groove t01_coastal_v3 layout blueprints pack docs
./run_hq.sh --world ~/Pictures/AI-Generated/Riff-Racers/World-02-Electro-Groove t01_coastal_v3 hq_faces,hq_boats
$PY w2lib/concepts_layout.py --world ~/Pictures/AI-Generated/Riff-Racers/World-02-Electro-Groove t01_coastal_v3
$PY world_docs.py --world ~/Pictures/AI-Generated/Riff-Racers/World-02-Electro-Groove
```

Ohne `--world` wird die Standardwelt aus `world-create-defaults.json` verwendet. `run_hq.sh MODULE FOLDER SHOTS` funktioniert weiterhin; der Ordner muss zur Welt-Konfiguration passen. `GRADE=grade3.py` und `RIFF_PYTHON=/pfad/python` bleiben möglich. Programme, Blender und imgai können über `world-create-defaults.json` oder `tools` in der Welt-Konfiguration angegeben werden. Diese Datei enthält lokale Einstellungen, keine festen Benutzerpfade im Programmcode.

`./run_final.sh --world PATH --all --list` zeigt die geplanten Strecken ohne GPU-Arbeit; ohne `--list` startet die ausdrücklich ausgewählte Batch-Verarbeitung. `./run_moss_all.sh --world PATH` arbeitet die aktiven Strecken nacheinander ab. Ohne konkrete Auswahl startet `run_final.sh` keine Render-Jobs.

## Weitere Welt anlegen

1. Eigenen Weltordner anlegen und `templates/riff-world.json` dorthin kopieren; `id`, `number` und `name` ändern.
2. `_project/tracks/__init__.py` und die eigenen Streckenmodule anlegen. Ihre `make()`-Funktion nutzt denselben `w2lib.track.Track`-Builder. Optionale `props()`/`cameras()` bleiben bei der Streckendefinition.
3. Jede aktive Strecke unter `tracks` eintragen: `{"module":"t01_name","id":"track-id","folder":"01-name"}`. Optionale ältere Fassungen stehen in `legacy_tracks` und sind nicht Teil von Standard-Batches.
4. Eigene Umgebungen unter `_project/w2blend/` anlegen; sie dürfen gemeinsame `core`, `roads`, `city` und Props-Bausteine verwenden. `ENV` in der Streckendatei wählt die Umgebung.
5. Eigene zusätzliche Rezepte bei Bedarf unter `_project/scripts/` ablegen und unter `scripts` konfigurieren, z. B. `{"world_docs":"world_docs.py"}`. Welt-2-spezifische Texte oder Coastal-Prüfungen werden nicht als allgemeine Regeln auf neue Welten kopiert.
6. Für MOSS-Repaints `HQ_SOURCES` im Streckenmodul angeben: `environment`, `pitlane`, `trap`, `crowd` mit eigenen Referenzbildern relativ zum Streckenordner, z. B. `concepts/hq/track-hero.png`. Es gibt keine Ersatzbilder aus einer anderen Strecke. Fehlende Referenzen stoppen vor GPU-Arbeit.
7. Mit `--world` prüfen und erst nach erfolgreicher Geometrieprüfung rendern. Ordner müssen innerhalb der gewählten Welt bleiben. Ausgaben und Cache werden pro Welt getrennt.

Es wird keine neue Welt automatisch angelegt oder ins Spiel registriert. Die Konfigurationsvorlage hat bewusst noch keine Strecken.

## Kompatibilität

Die bisherigen Weltordner verweisen über einen relativen Symlink `World-0X-…/_pipeline` auf diesen Werkzeugordner, damit alte Aufrufe über `../_pipeline/` weiter funktionieren. Für neue Welten direkt `--world PATH` oder `RIFF_WORLD_ROOT` verwenden und die von `world_create.py info` gemeldeten Cache-Pfade nutzen.

Die Eintritte `lookdev_coastal_v3.py` und `render_coastal_followup.py` sind Welt-2-spezifische Kompatibilitäts-Eintritte; sie leiten an die in der Welt konfigurierten Rezepte weiter.

## Prüfung und Grenzen

```bash
$PY -m unittest test_world_create test_drivability -v
```

Die Projektprüfungen benutzen zwei unabhängige Testwelten mit gleicher Modul- und Strecken-ID und prüfen getrennte Ausgaben, Modulauflösung, Auswahlpriorität sowie Pfadgrenzen. Temporäre Daten werden vor dem Aufräumen aufgelistet und per `gio trash` entfernt. Der Render- und MOSS-Aufruf verwenden eine gemeinsame kooperative GPU-Sperre für alle Welten. Direkte manuelle Blender-/imgai-Aufrufe müssen weiterhin nacheinander laufen.

Fahrbarkeitsmodelle bleiben Modelle. Statusfreigaben und eine spätere Spielintegration erfolgen ausschließlich auf ausdrücklichen Auftrag.

### Direkte Bildablage pro Strecke

Eine Strecke kann `CONCEPT_DIR="."` setzen, um ihre Ergebnisbilder als echte Dateien im Hauptordner abzulegen. `HQ_SOURCES` enthält immer Pfade relativ zum Streckenordner. `FIDELITY_FILE` wählt den Ort des Strukturberichts, `RAW_RENDER_DIR` den Ort ungegradeter Originale. Coastal nutzt `planning/layout-fidelity.json` und `blender/raw-renders/`; seine Kameras geben direkte PNG-Dateinamen aus. Ohne diese Optionen gelten die bisherigen Unterordner.

### Bestehende Spielstrecke als externe Vorproduktion

Für Iron Chorus wird die vorhandene Straße gelesen, statt ein neues Welt-2-Layout mit `make()` zu erfinden. Der Datenadapter `prepare_source.py` erzeugt ein eigenes `existing-game-source-v1`-Prüfprofil, einen gerichteten Layoutgraphen, Quellkopien, theoretische Korridor-/Flugnachweise und einen Renderpack. `SOURCE_CORRECTIONS` in der eigenen Streckendefinition benennt dokumentierte externe Requisitenkorrekturen. Spielquellen bleiben unverändert. Die Kamera-/Dokuvorlagen dieses Adapters sind derzeit für Iron Chorus ausgearbeitet; neue Strecken benötigen ihre eigenen Definitionen und Referenzen.

```bash
$PY prepare_source.py --world ~/Pictures/AI-Generated/Riff-Racers/World-01-Metal t01_iron_chorus
$PY run_source.py --world ~/Pictures/AI-Generated/Riff-Racers/World-01-Metal t01_iron_chorus mask
$PY run_source.py --world ~/Pictures/AI-Generated/Riff-Racers/World-01-Metal t01_iron_chorus quick hero,stage
$PY run_source.py --world ~/Pictures/AI-Generated/Riff-Racers/World-01-Metal t01_iron_chorus hq
$PY verify_source.py --world ~/Pictures/AI-Generated/Riff-Racers/World-01-Metal t01_iron_chorus
```

`run_source.py` nutzt den gemeinsamen Blender-Szenenbau und die gemeinsame GPU-Sperre. Schnellbilder landen im Cache, 4K-Originale unter `blender/raw-renders/`, gegradete echte PNGs direkt im Streckenordner. `verify_source.py` prüft Straßenprojektion, neue Dekoreserven, Original-Prüfsummen und Bildinhalte. Benannte Kameras werden in einem eigenen Blender-Master gespeichert. `source_diagrams.py` ergänzt die technischen Ansichten; MOSS folgt mit `w2lib/concepts_layout.py`; `source_docs.py` schreibt nach abgeschlossenen Nachweisen den Status REVIEW_REQUIRED. Keine formale Freigabe oder Integration.

Die reservierten Landschaftssichtachsen stehen in `planning/scenery-sightlines.json`. Sie bleiben beim Nachjustieren von Kameras fest, damit sich die Dekogeometrie zwischen Bilddurchgängen nicht verändert. Dieses Quellenprofil nicht über `dev.py` oder die Welt-2-`make()`-Rezepte aufrufen. Ausführliches Rezept: `World-01-Metal/_project/WORKFLOW.md`.
