"""How much disk Lectern uses, broken down, against what the disk has.

Everything is measured, never estimated: directories are walked (symlinks are
not followed, so nothing is counted twice or outside its folder), model files
are stat()ed, and the notes model's size comes from Ollama. A walk over a big
Python environment takes a fraction of a second, so the result is cached
briefly; the Settings panel is the only caller.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

from . import config, models

REPO_DIR = Path(__file__).resolve().parent.parent
CACHE_S = 20
_cache: tuple[float, dict] | None = None
_lock = threading.Lock()


def dir_size(path: Path, *, skip: tuple[Path, ...] = ()) -> int:
    """Bytes on disk under `path`, not following symlinks, minus `skip` dirs."""
    total = 0
    skip_s = {str(p) for p in skip}
    for root, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in skip_s]
        for f in files:
            try:
                st = os.lstat(os.path.join(root, f))
            except OSError:
                continue
            # Allocated blocks where the OS reports them: sparse or compressed
            # files take less than their length.
            total += getattr(st, "st_blocks", 0) * 512 or st.st_size
    return total


def _speech_models() -> tuple[int, int, list[dict]]:
    """(bytes in Lectern's own model folder, bytes in shared folders, details)."""
    own = shared = 0
    found = []
    for m in models.CATALOG:
        p = models.installed_path(m)
        if p is None:
            continue
        size = p.stat().st_size
        is_own = models.MODELS_DIR in p.parents
        if is_own:
            own += size
        else:
            shared += size
        found.append({"id": m.id, "label": m.label, "bytes": size, "shared": not is_own})
    return own, shared, found


def _notes_model() -> dict | None:
    """Size of the notes model as Ollama stores it (in ~/.ollama), if installed."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as r:
            tags = json.loads(r.read())
    except Exception:
        return None
    want = config.LLM_MODEL if ":" in config.LLM_MODEL else config.LLM_MODEL + ":latest"
    for m in tags.get("models", []):
        if m.get("name") == want:
            return {"name": want, "bytes": int(m.get("size") or 0)}
    return None


def report(force: bool = False) -> dict:
    global _cache
    with _lock:
        if not force and _cache and time.monotonic() - _cache[0] < CACHE_S:
            return _cache[1]
        config.ensure_dirs()
        lectures = dir_size(config.LECTURES_ROOT, skip=(config.TRASH_DIR,))
        trash = dir_size(config.TRASH_DIR)
        own_models, shared_models, model_list = _speech_models()
        notes = _notes_model()
        support = dir_size(config.SUPPORT_DIR, skip=(models.MODELS_DIR,))
        # The app folder, minus things counted elsewhere or that are not part
        # of the installed app (a developer's benchmark downloads).
        app = dir_size(REPO_DIR, skip=(REPO_DIR / "bench",))
        disk = shutil.disk_usage(config.LECTURES_ROOT)
        parts = [
            {"key": "lectures", "label": "Lectures", "bytes": lectures,
             "detail": "audio, transcripts and notes in ~/Lectures"},
            {"key": "trash", "label": "Recently deleted", "bytes": trash,
             "detail": "kept 7 days so a delete can be undone, then removed"},
            {"key": "speech", "label": "Speech models", "bytes": own_models,
             "detail": ", ".join(f"{m['label']} {m['bytes'] / 1e6:.0f} MB"
                                 for m in model_list if not m["shared"]) or "none downloaded"},
            {"key": "notes", "label": "Notes model", "bytes": notes["bytes"] if notes else 0,
             "detail": f"{notes['name']}, stored by Ollama" if notes else "not installed"},
            {"key": "app", "label": "App", "bytes": app + support,
             "detail": "program, Python environment, logs and settings"},
        ]
        result = {
            "total_bytes": sum(p["bytes"] for p in parts),
            "parts": parts,
            "shared_bytes": shared_models,
            "shared_detail": ", ".join(f"{m['label']} {m['bytes'] / 1e6:.0f} MB"
                                       for m in model_list if m["shared"]),
            "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        }
        _cache = (time.monotonic(), result)
        return result
