"""Sequential batch actions for the configured world's registered tracks."""
import subprocess
import sys
from world_create import get_project, WORLD_CREATE_ROOT

project = get_project()
args = sys.argv[1:]
if not args or args[0] not in ("final", "moss"):
    raise SystemExit("Usage: run_batch.py [--world PATH] final|moss [--all|MODULE ...] [--list]")
mode = args.pop(0)
listing = "--list" in args
args = [a for a in args if a != "--list"]
if args == ["--all"] or (mode == "moss" and not args):
    modules = [s["module"] for s in project.specs()]
else:
    modules = args
if not modules:
    raise SystemExit("Select track modules or --all (no jobs started)")
unknown = set(modules) - {s["module"] for s in project.specs(True)}
if unknown:
    raise SystemExit(f"Track modules are not registered for this world: {sorted(unknown)}")
for module in modules:
    print(mode, project.data["id"], module, flush=True)
    if listing: continue
    if mode == "moss":
        command = [str(WORLD_CREATE_ROOT / "w2lib/concepts_layout.py"), module]
    else:
        command = [str(WORLD_CREATE_ROOT / "run_stage.py"), module, "layout", "blueprints", "pack", "bake:mode=mask", "bake:mode=render;shots=hero,overview,gameplay,pitlane,hazard,topdown;samples=192", "overlay", "docs"]
    subprocess.run([sys.executable, *command], check=True)
