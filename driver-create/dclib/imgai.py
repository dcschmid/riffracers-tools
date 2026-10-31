"""Thin wrapper around the local-image-ai-service `imgai` CLI (the new image pipeline).

driver-create does not reimplement generation or QA: it drives the `imgai`
character/vehicle/asset/roster sub-commands and adds the convergence loop on top.
Every call that may touch the GPU runs under a shared lock so that driver-create,
world-create and manual imgai/Blender jobs never fight over the GPU.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import unicodedata
from contextlib import contextmanager
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS_FILE = TOOL_ROOT / "driver-create-defaults.json"

# Shared GPU lock: one cooperative GPU job at a time across every tool that uses
# the local image service (driver-create, world-create, manual imgai/Blender).
GPU_LOCK = Path(os.environ.get("LOCAL_IMAGE_AI_HOME", Path.home() / ".local/share/local-image-ai")) / ".gpu.lock"

CHAR_VIEWS = ("front", "side", "back", "three-quarter")
KART_VIEWS = ("front", "side", "back", "top")
PASS, REVIEW, FAIL = "PASS", "REVIEW_REQUIRED", "FAIL"
RANK = {PASS: 0, REVIEW: 1, FAIL: 2}


def defaults() -> dict:
    return json.loads(DEFAULTS_FILE.read_text()) if DEFAULTS_FILE.exists() else {}


def slugify(name: str) -> str:
    """Same slug as imgai.common.slugify, so driver-create and imgai agree on the set directory."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    if not s:
        raise ValueError(f"cannot derive an identifier from name '{name}'")
    return s


@contextmanager
def gpu_job():
    """One cooperative GPU job at a time, shared across every tool."""
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def imgai_bin() -> str:
    value = defaults().get("tools", {}).get("image_ai") or "~/.local/bin/imgai"
    return str(Path(value).expanduser())


def run_imgai(args: list[str], *, check: bool = True, gpu: bool = True, timeout: int = 7200) -> subprocess.CompletedProcess:
    """Run one imgai sub-command. `gpu=True` holds the shared GPU lock for the whole call."""
    cmd = [imgai_bin(), *args]
    if gpu:
        with gpu_job():
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    else:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode not in (0, 1, 2):
        raise RuntimeError(f"imgai {' '.join(args)} failed (exit {proc.returncode}):\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    return proc


def set_dir(name: str, root: Path) -> Path:
    return Path(root) / slugify(name)


def load_spec(d: Path) -> dict:
    p = d / "design-spec.json"
    if not p.is_file():
        raise FileNotFoundError(f"no design-spec.json in {d} (run the create stage first)")
    return json.loads(p.read_text())


def load_qa(d: Path) -> dict:
    p = d / "qa.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def status_of(d: Path) -> str:
    """Overall set status from qa.json (PASS / REVIEW_REQUIRED / FAIL / NOT_BUILT)."""
    qa = load_qa(d)
    if not qa:
        return "NOT_BUILT"
    return qa.get("status", "NOT_BUILT")


def failing_views(d: Path) -> dict[str, list[str]]:
    """Views that are present in qa.json but not PASS, per target. A view that was never
    generated (absent from qa.json) is not 'failing' here; the build stage creates it."""
    qa = load_qa(d)
    out: dict[str, list[str]] = {"character": [], "vehicle": []}
    for target, views in (("character", CHAR_VIEWS), ("vehicle", KART_VIEWS)):
        section = qa.get(target) or {}
        out[target] = [v for v in views if v in section and section[v] != PASS]
    return out


def roster_entries(root: Path) -> list[dict]:
    p = Path(root) / "roster.json"
    if not p.is_file():
        raise FileNotFoundError(f"roster file not found: {p}")
    data = json.loads(p.read_text())
    return data["drivers"] if isinstance(data, dict) else data


def find_entry(root: Path, name: str) -> dict | None:
    for e in roster_entries(root):
        if e.get("name", "").lower() == name.lower():
            return e
    return None


def write_status(root: Path, name: str, status: str) -> None:
    p = Path(root) / "roster-status.json"
    cur = json.loads(p.read_text()) if p.is_file() else {}
    cur[name] = status
    p.write_text(json.dumps(cur, indent=1) + "\n")


def print_status(name: str, status: str) -> None:
    color = {PASS: "\033[32m", REVIEW: "\033[33m", FAIL: "\033[31m"}.get(status, "\033[0m")
    print(f"{name:16} {color}{status}\033[0m", flush=True)
