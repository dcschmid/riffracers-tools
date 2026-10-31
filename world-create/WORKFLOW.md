# Gemeinsamer Ablauf für alle Welten

Zuerst die Welt mit `--world` oder `RIFF_WORLD_ROOT` auswählen und `world_create.py info` prüfen. Danach das Arbeitsrezept der gewählten Welt (`_project/WORKFLOW.md`) lesen.

1. Layout entwerfen und datenbasiert validieren (`dev.py`).
2. Blueprints und Pack aus denselben Daten bauen (`run_stage.py`). Fehler in der Geometrie blockieren Pack und Blender-Ausgabe.
3. Bilder über die gemeinsame Render-Pipeline erzeugen; Stil und spezielle Umgebung stammen aus der gewählten Welt.
4. GPU-Jobs nacheinander ausführen; die CLI koordiniert Blender und MOSS über eine gemeinsame Sperre.
5. Render und Layout per Straßenmaske vergleichen; MOSS-Concepts sind kein Geometrienachweis.
6. Dokumente, Modellannahmen und Grenzen aktualisieren. Nach jeder Strecke Bilder einzeln ansehen und Rückmeldung abwarten.
7. Vor dem Aufräumen prüfen, was betroffen ist; Löschungen ausschließlich per `gio trash`. Rohbilder, freigegebene Stände und Archive erhalten.

Werkzeuge verändern oder registrieren nichts im Spiel. Eine positive Bildrückmeldung ist keine automatische formale Concept-Freigabe oder Integrationsfreigabe. Welt-spezifische Motive, Regeln und Reihenfolgen nicht ungefragt auf andere Welten übertragen.

Aufrufbeispiele und Konfiguration: `README.md`.
