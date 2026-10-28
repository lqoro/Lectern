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
import functools
import json
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


# How much of a GB counts, in Mac terms, when deciding what fits. macOS
# compresses memory and Apple chips share it with the GPU, so an 8 GB Mac runs
# what a Windows PC needs about 16 GB for: Windows itself, its services and the
# browser take far more of the same RAM. Every RAM tier in Lectern (model
# recommendation, what "fits", whether AI notes run) was measured on an 8 GB
# M1, so other systems are scaled into those terms instead of each tier
# carrying per-OS numbers.
RAM_FACTOR = {"Darwin": 1.0, "Windows": 0.5}


def effective_ram_bytes(total: int | None = None, os_name: str | None = None) -> int | None:
    """Total RAM expressed as the Mac memory it is worth to Lectern."""
    total = total_ram_bytes() if total is None else total
    if not total:
        return None
    return int(total * RAM_FACTOR.get(os_name or platform.system(), 1.0))


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


@functools.lru_cache(maxsize=1)
def device() -> dict:
    """What this computer is, in words a person recognises: e.g.
    {"name": "MacBook Air", "chip": "Apple M1"}. Only the model name and
    processor are read -- never serial numbers or hardware IDs. Cached: it
    cannot change while the app runs, and system_profiler takes ~0.5 s."""
    name = chip = None
    if pc.IS_MAC:
        try:
            info = json.loads(_run(["system_profiler", "SPHardwareDataType", "-json"],
                                   timeout=8) or "{}")
            hw = (info.get("SPHardwareDataType") or [{}])[0]
            name = hw.get("machine_name")
            chip = hw.get("chip_type") or hw.get("cpu_type")
        except (ValueError, AttributeError, IndexError):
            pass
        chip = chip or _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or None
    elif pc.IS_LINUX:
        name = _read("/sys/devices/virtual/dmi/id/product_name").strip() or None
        m = re.search(r"^model name\s*:\s*(.+)$", _read("/proc/cpuinfo"), re.M)
        chip = m.group(1).strip() if m else None
    elif pc.IS_WIN:
        chip = platform.processor() or None
    if name and name.lower() in ("to be filled by o.e.m.", "system product name", "default string"):
        name = None                    # placeholder text some PC makers leave in
    return {"name": name, "chip": chip}


def snapshot() -> dict:
    total, avail = total_ram_bytes(), available_ram_bytes()
    eff = effective_ram_bytes(total)
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "ram_total_mb": round(total / 2**20) if total else None,
        "ram_effective_mb": round(eff / 2**20) if eff else None,
        "ram_available_mb": round(avail / 2**20) if avail else None,
        "cpu_cores": cpu_cores(),
        "cpu_threads": cpu_threads(),
        "accelerator": accelerator(),
        "device": device(),
        "on_battery": on_battery(),
        "whisper_threads": whisper_threads(),
    }
