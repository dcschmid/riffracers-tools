# Gemeinsamer Ablauf für alle Welten

Zuerst die Welt mit `--world` oder `RIFF_WORLD_ROOT` auswählen und `world_create.py info` prüfen. Danach das Arbeitsrezept der gewählten Welt (`_project/WORKFLOW.md`) lesen.

1. Streckenverlauf zuerst planen, bevor für eine Strecke ein einziges Bild entsteht: geschlossene Hauptrunde, Pitlane mit **12 Boxen** (Boxenstopps für 12 Carts), **genau 2 Abkürzungen** und **genau 2 Risikosprünge** (Absprung, Flug, Landezone, sichere Alternativroute). Erst wenn `dev.py` die Geometrie fehlerfrei validiert hat, sind Blueprints, Pack und Bilder erlaubt.
2. Blueprints und Pack aus denselben Daten bauen (`run_stage.py`). Fehler in der Geometrie blockieren Pack und Blender-Ausgabe.
3. Initiale Bilder (Vogelperspektive, Umgebung, Boxen, Fallen, Publikum, Moodboard) mit dem **MOSS Low-Poly**-Modell des lokalen `local-image-ai-service` erzeugen: Text-Concepts über die JSON-CLI `local-image-ai-service render` (`w2lib/concepts.py`), layout-treue img2img-Repaints der Blender-Renders über `imgai img2img` (`w2lib/concepts_layout.py`, da moss im Service nur Text-zu-Bild kann); Stil und spezielle Umgebung stammen aus der gewählten Welt.
4. GPU-Jobs nacheinander ausführen; die CLI koordiniert Blender und MOSS über eine gemeinsame Sperre.
5. Render und Layout per Straßenmaske vergleichen; MOSS-Concepts sind kein Geometrienachweis.
6. Dokumente, Modellannahmen und Grenzen aktualisieren. Nach jeder Strecke Bilder einzeln ansehen und Rückmeldung abwarten.
7. Vor dem Aufräumen prüfen, was betroffen ist; Löschungen ausschließlich per `gio trash`. Rohbilder, freigegebene Stände und Archive erhalten.

Werkzeuge verändern oder registrieren nichts im Spiel. Eine positive Bildrückmeldung ist keine automatische formale Concept-Freigabe oder Integrationsfreigabe. Welt-spezifische Motive, Regeln und Reihenfolgen nicht ungefragt auf andere Welten übertragen.

Aufrufbeispiele und Konfiguration: `README.md`.
