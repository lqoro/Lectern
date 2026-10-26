"""What this machine can do, so Lectern can pick sensible defaults.

Everything here is best-effort: any probe that fails returns None rather than
raising, and callers treat None as "unknown, assume modest".
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys

from . import platform_compat as pc


def _run(argv: list[str], timeout: float = 4.0) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def total_ram_bytes() -> int | None:
    ps = pc._psutil()
    if ps is not None:
        try:
            return int(ps.virtual_memory().total)
        except Exception:
            pass
    if pc.IS_MAC:
        out = _run(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out) if out.isdigit() else None
    if pc.IS_LINUX:
        m = re.search(r"MemTotal:\s+(\d+) kB", _read("/proc/meminfo"))
        return int(m.group(1)) * 1024 if m else None
    if pc.IS_WIN:
        return _win_mem("ullTotalPhys")
    return None


def available_ram_bytes() -> int | None:
    """RAM that could be used right now without swapping. On macOS this
    UNDERSTATES what is usable (compressed/cached pages are reclaimed on
    demand), so do not gate features on it there."""
    ps = pc._psutil()
    if ps is not None:
        try:
            return int(ps.virtual_memory().available)
        except Exception:
            pass
    if pc.IS_LINUX:
        m = re.search(r"MemAvailable:\s+(\d+) kB", _read("/proc/meminfo"))
        return int(m.group(1)) * 1024 if m else None
    if pc.IS_MAC:
        out = _run(["vm_stat"])
        page = re.search(r"page size of (\d+) bytes", out)
        if not page:
            return None
        size = int(page.group(1))
        pages = 0
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            m = re.search(rf"{key}:\s+(\d+)", out)
            if m:
                pages += int(m.group(1))
        return pages * size or None
    if pc.IS_WIN:
        return _win_mem("ullAvailPhys")
    return None


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _win_mem(field: str) -> int | None:
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        s = MEMSTATUS()
        s.dwLength = ctypes.sizeof(MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))  # type: ignore[attr-defined]
        return int(getattr(s, field))
    except Exception:
        return None


def cpu_threads() -> int:
    return os.cpu_count() or 2


def cpu_cores() -> int:
    """Physical cores when known; logical threads otherwise."""
    ps = pc._psutil()
    if ps is not None:
        try:
            n = ps.cpu_count(logical=False)
            if n:
                return int(n)
        except Exception:
            pass
    if pc.IS_MAC:
        out = _run(["sysctl", "-n", "hw.physicalcpu"]).strip()
        if out.isdigit():
            return int(out)
    return cpu_threads()


def accelerator() -> str:
    """"metal" (Apple Silicon), "cuda" (an NVIDIA GPU is present) or "cpu".

    "cuda" only says a GPU exists. Whether the installed whisper-cli was built
    to use it is not something we can see from here, so the recommender treats
    it as a hint, not a promise.
    """
    if pc.IS_MAC and platform.machine() == "arm64":
        return "metal"
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "cpu"


def on_battery() -> bool | None:
    ps = pc._psutil()
    if ps is None:
        return None
    try:
        b = ps.sensors_battery()
        return None if b is None else (not b.power_plugged)
    except Exception:
        return None


def whisper_threads() -> int:
    """Threads for whisper-cli.

    LECTERN_THREADS overrides. On Apple Silicon the GPU does the work and 4 was
    measured to be enough, so that stays 4. On CPU-only machines whisper scales
    with physical cores, but the OS, browser and the recorder need one too:
    use cores - 1 (at least 1, at most 8; beyond 8 threads whisper.cpp gains
    little and mostly burns power).
    """
    env = os.environ.get("LECTERN_THREADS", "").strip()
    if env.isdigit() and int(env) >= 1:
        return int(env)
    if accelerator() == "metal":
        return 4
    return max(1, min(8, cpu_cores() - 1))


def snapshot() -> dict:
    total, avail = total_ram_bytes(), available_ram_bytes()
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "ram_total_mb": round(total / 2**20) if total else None,
        "ram_available_mb": round(avail / 2**20) if avail else None,
        "cpu_cores": cpu_cores(),
        "cpu_threads": cpu_threads(),
        "accelerator": accelerator(),
        "on_battery": on_battery(),
        "whisper_threads": whisper_threads(),
    }
