"""World selection and output isolation for the world-create preproduction tools.

Select once per process: --world PATH, RIFF_WORLD_ROOT, current world directory,
then world-create-defaults.json. World-local tracks and Blender themes are namespace
packages under code_dir; common geometry/rendering modules remain shared.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

WORLD_CREATE_ROOT = Path(__file__).resolve().parent
MANIFEST = "riff-world.json"


def contained(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path must stay inside the world directory: {value}")
    return path


def cli_world() -> str | None:
    result = None
    args = sys.argv
    i = 1
    while i < len(args):
        if args[i] == "--world":
            if i + 1 >= len(args):
                raise ValueError("--world needs a world directory or manifest path")
            result = args[i + 1]
            del args[i:i + 2]
        elif args[i].startswith("--world="):
            result = args.pop(i).split("=", 1)[1]
        else:
            i += 1
    return result


def defaults() -> dict:
    path = WORLD_CREATE_ROOT / "world-create-defaults.json"
    return json.loads(path.read_text()) if path.exists() else {}


def select_world(explicit: str | Path | None = None) -> Path:
    selected = explicit or os.environ.get("RIFF_WORLD_ROOT")
    if selected:
        path = Path(selected).expanduser().resolve()
        return path.parent if path.name == MANIFEST else path
    for start in (Path(os.environ.get("PWD", str(Path.cwd()))), Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / MANIFEST).is_file():
                return candidate.resolve()
        if any(p.name.startswith("World-") for p in (start, *start.parents)):
            raise ValueError(f"This world has no {MANIFEST}; configure it or select --world PATH explicitly")
    configured = defaults().get("default_world")
    if configured:
        # Relative paths resolve against the tool directory; ~ and absolute
        # paths resolve against the home directory, so the default world can
        # live anywhere on the machine.
        return Path(configured).expanduser().resolve()
    raise ValueError("Select a world with --world PATH or RIFF_WORLD_ROOT")


@dataclass(frozen=True)
class Project:
    root: Path
    data: dict

    @classmethod
    def load(cls, root: str | Path) -> "Project":
        root = Path(root).expanduser().resolve()
        data = json.loads((root / MANIFEST).read_text())
        if data.get("schema") != "riff-racers.world-project/1":
            raise ValueError(f"Unsupported world manifest: {root / MANIFEST}")
        if not all(isinstance(data.get(k), str) and data[k].strip() for k in ("id", "name")):
            raise ValueError("World manifest needs non-empty id and name")
        project = cls(root, data)
        for key, fallback in (("code_dir", "_project"), ("cache_dir", "_project/out")):
            contained(root, data.get(key, fallback))
        seen = set()
        active_ids = set()
        for spec in project.specs(True):
            module = spec["module"]
            if not all(p.isidentifier() for p in module.split(".")) or module in seen:
                raise ValueError(f"Invalid or duplicate track module: {module}")
            seen.add(module)
            contained(root, spec["folder"])
        for spec in project.specs():
            if spec["id"] in active_ids:
                raise ValueError(f"Duplicate active track id: {spec['id']}")
            active_ids.add(spec["id"])
        return project

    @property
    def code_dir(self) -> Path:
        return contained(self.root, self.data.get("code_dir", "_project"))

    @property
    def cache_dir(self) -> Path:
        return contained(self.root, self.data.get("cache_dir", "_project/out"))

    @property
    def game_root(self) -> Path:
        value = self.data.get("game_root") or defaults().get("game_root")
        if not value:
            raise ValueError("This read-only controller audit needs game_root in configuration")
        return Path(value).expanduser().resolve()

    def tool(self, name: str, fallback: str) -> str:
        value = self.data.get("tools", {}).get(name) or defaults().get("tools", {}).get(name) or fallback
        return str(Path(value).expanduser())

    def specs(self, legacy: bool = False) -> list[dict]:
        return list(self.data.get("tracks", [])) + (list(self.data.get("legacy_tracks", [])) if legacy else [])

    def import_track(self, module: str):
        if not all(p.isidentifier() for p in module.split(".")):
            raise ValueError(f"Invalid track module: {module}")
        return importlib.import_module("tracks." + module)

    def folder(self, module: str, track_id: str | None = None, definition=None) -> str:
        spec = next((s for s in self.specs(True) if s["module"] == module), None)
        value = spec["folder"] if spec else getattr(definition, "OUT_FOLDER", None)
        if value is None:
            spec = next((s for s in self.specs() if s["id"] == track_id), None)
            value = spec["folder"] if spec else None
        if not value:
            raise ValueError(f"Register track {module} and its folder in {self.root / MANIFEST}")
        contained(self.root, value)
        return value

    def track_dir(self, module: str) -> Path:
        return contained(self.root, self.folder(module))

    def script(self, name: str) -> Path:
        value = self.data.get("scripts", {}).get(name)
        if not value:
            raise ValueError(f"World {self.data['id']} has no configured script '{name}'")
        path = contained(self.code_dir, "scripts/" + value)
        if not path.is_file():
            raise ValueError(f"Project script missing: {path}")
        return path

    def activate(self):
        code = str(self.code_dir)
        if code not in sys.path:
            sys.path.insert(0, code)
        os.environ["RIFF_WORLD_ROOT"] = str(self.root)
        return self


_project = None


def get_project() -> Project:
    global _project
    if _project is None:
        _project = Project.load(select_world(cli_world())).activate()
    return _project


@contextmanager
def gpu_job():
    """One cooperative GPU job at a time, shared across every world."""
    with (WORLD_CREATE_ROOT / ".gpu.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def dispatch(name: str):
    project = get_project()
    runpy.run_path(str(project.script(name)), run_name="__main__")


if __name__ == "__main__":
    try:
        p = get_project()
        command = sys.argv[1] if len(sys.argv) > 1 else "info"
        if command == "root":
            print(p.root)
        elif command == "tracks":
            for spec in p.specs(): print(spec["module"])
        elif command == "info":
            print(json.dumps(dict(world=p.data["id"], name=p.data["name"], root=str(p.root), code=str(p.code_dir), cache=str(p.cache_dir), tools=str(WORLD_CREATE_ROOT), tracks=p.specs()), indent=2, ensure_ascii=False))
        else:
            raise ValueError("Use info, root, or tracks; optionally --world PATH")
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error))
