"""Shared HQ renderer; old MODULE FOLDER SHOTS syntax is still accepted."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from world_create import get_project, WORLD_CREATE_ROOT, contained


def main():
    project = get_project()
    args = sys.argv[1:]
    if len(args) not in (2, 3):
        raise SystemExit("Usage: run_hq.sh [--world PATH] MODULE [FOLDER] hq_shot,hq_shot")
    module, shots = args[0], args[-1]
    folder = project.folder(module)
    if len(args) == 3 and args[1] != folder:
        raise SystemExit(f"Configured folder for {module} is {folder}, not {args[1]}")
    if not shots or any(not s.startswith("hq_") for s in shots.split(",")):
        raise SystemExit("HQ rendering expects named hq_* cameras")
    started = time.time_ns()
    subprocess.run([sys.executable, str(WORLD_CREATE_ROOT / "run_stage.py"), module, "pack", f"bake:mode=render;shots={shots};samples=160"], check=True)
    definition = project.import_track(module)
    key = getattr(definition, "PACK_ID", None) or next(s["id"] for s in project.specs(True) if s["module"] == module)
    import json
    cameras = json.loads((contained(project.cache_dir, key) / "pack/cams.json").read_text())
    grade = contained(WORLD_CREATE_ROOT, os.environ.get("GRADE") or project.data.get("grading", "grade.py"))
    from PIL import Image
    for shot in shots.split(","):
        path = Path(cameras[shot]["out"])
        if not path.is_file() or path.stat().st_mtime_ns < started:
            raise RuntimeError(f"No fresh render for {shot}: {path}")
        with Image.open(path) as image:
            if image.size != (3840, 2160):
                raise RuntimeError(f"Expected 4K image: {path}")
        raw = contained(project.track_dir(module), definition.RAW_RENDER_DIR) if hasattr(definition, "RAW_RENDER_DIR") else path.parent / "_raw"
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, raw / path.name)
        subprocess.run([sys.executable, str(grade), str(path)], check=True)
    print("HQ DONE", project.data["id"], folder)


if __name__ == "__main__": main()
