"""Builds and locates the native capture helper (lectern_capture.swift).

ffmpeg 5.0's avfoundation input silently drops audio: it holds one pending
buffer and overwrites it whenever ffmpeg has not polled in time. Measured on
this machine it kept ~88% of the audio when run from a shell and ~32% when run
detached from the backend, so every lecture recorded since mid-September lost
about two thirds of its speech. The helper writes every buffer and cuts
segments by sample count instead. ffmpeg stays as the fallback only if the
helper cannot be built.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from pathlib import Path

from . import config
from . import platform_compat as pc

log = logging.getLogger(__name__)
_lock = threading.Lock()
_failed_mtime: float | None = None      # don't retry a broken build on every start


def portable() -> bool:
    """True when recording should use the portable Python helper
    (capture_py.py) instead of the macOS-only Swift one. Windows and Linux
    always do. macOS keeps the Swift helper unless LECTERN_CAPTURE=python."""
    mode = os.environ.get("LECTERN_CAPTURE", "auto").lower()
    if mode == "python":
        return True
    if mode == "swift":
        return False
    return (not pc.IS_MAC) or os.environ.get("LECTERN_CAPTURE_SOURCE") == "synthetic"


def _stale() -> bool:
    try:
        return config.CAPTURE_BIN.stat().st_mtime < config.CAPTURE_SRC.stat().st_mtime
    except FileNotFoundError:
        return True


def ensure_built(timeout: float = 600) -> Path | None:
    """Return the helper binary, compiling it if missing or out of date.
    None means "use ffmpeg": no compiler, or the build failed."""
    global _failed_mtime
    if portable():
        return None
    with _lock:
        if not _stale():
            return config.CAPTURE_BIN
        try:
            src_mtime = config.CAPTURE_SRC.stat().st_mtime
        except FileNotFoundError:
            return None
        if _failed_mtime == src_mtime:
            return None
        config.CAPTURE_BIN.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.CAPTURE_BIN.with_suffix(".building")
        try:
            proc = subprocess.run(
                ["/usr/bin/xcrun", "swiftc", "-O", "-swift-version", "5",
                 "-o", str(tmp), str(config.CAPTURE_SRC)],
                capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            log.error("could not build capture helper: %s", e)
            _failed_mtime = src_mtime
            return None
        if proc.returncode != 0 or not tmp.is_file():
            log.error("capture helper build failed:\n%s", proc.stderr[-2000:])
            _failed_mtime = src_mtime
            return None
        os.replace(tmp, config.CAPTURE_BIN)
        log.info("built capture helper at %s", config.CAPTURE_BIN)
        return config.CAPTURE_BIN


def available() -> Path | None:
    """Non-blocking: the helper if it is already built and current."""
    if portable():
        return None
    return None if _stale() else config.CAPTURE_BIN
