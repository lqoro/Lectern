"""Everything that differs between macOS, Windows and Linux, in one place.

Rule for this module: on macOS every function behaves exactly as the code it
replaced did, so the Mac build (the one that has been tested on real lectures)
does not change. Windows and Linux get the portable branches.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")


# ------------------------------------------------------------------ paths
def support_dir(app: str = "Lectern") -> Path:
    """Where the app keeps its rebuildable index, logs and helper binaries."""
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / app
    if IS_WIN:
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / app
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / app


def find_binary(name: str, env_var: str | None = None,
                mac_candidates: tuple[str, ...] = ()) -> str:
    """Locate a helper program (ffmpeg, whisper-cli...).

    Order: explicit env var -> bundled with the app (LECTERN_BIN_DIR) ->
    Homebrew locations (macOS) -> PATH -> the bare name (so the error message
    names what is missing instead of a made-up path).
    """
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]
    exe = name + (".exe" if IS_WIN else "")
    for d in filter(None, os.environ.get("LECTERN_BIN_DIR", "").split(os.pathsep)):
        p = Path(d) / exe
        if p.is_file():
            return str(p)
    if IS_MAC:
        for c in mac_candidates:
            if Path(c).is_file():
                return c
    return shutil.which(name) or (mac_candidates[0] if (IS_MAC and mac_candidates) else name)


# ------------------------------------------------------------- processes
def detached_popen_kwargs() -> dict:
    """Popen options so the recorder outlives the backend, the tab and the app."""
    if IS_WIN:
        DETACHED_PROCESS, NEW_GROUP, NO_WINDOW = 0x00000008, 0x00000200, 0x08000000
        return {"creationflags": DETACHED_PROCESS | NEW_GROUP | NO_WINDOW,
                "close_fds": True}
    return {"start_new_session": True}


def low_priority(argv: list[str]) -> tuple[list[str], dict]:
    """(argv, Popen kwargs) that run a heavy child (whisper, ffmpeg) below
    normal priority, so the UI and the live recorder keep getting CPU.

    POSIX uses the `nice` program rather than preexec_fn, which the Python docs
    warn can deadlock in a multithreaded process such as this backend.
    """
    if IS_WIN:
        return argv, {"creationflags": 0x00004000}      # BELOW_NORMAL_PRIORITY_CLASS
    nice = shutil.which("nice")
    return ([nice, "-n", "10", *argv] if nice else argv), {}


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def pid_alive(pid: int) -> bool:
    """Is this PID a live (non-zombie) process? Never uses os.kill on Windows,
    where os.kill(pid, 0) would TERMINATE the process."""
    ps = _psutil()
    if ps is not None:
        try:
            return ps.Process(pid).status() != ps.STATUS_ZOMBIE
        except ps.NoSuchProcess:
            return False
        except ps.Error:
            return True
    if IS_WIN:
        return False                      # cannot tell without psutil; assume gone
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def command_line(pid: int) -> str:
    ps = _psutil()
    if ps is not None:
        try:
            return " ".join(ps.Process(pid).cmdline())
        except ps.Error:
            return ""
    if IS_WIN:
        return ""
    try:
        return subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                              capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def terminate_pid(pid: int, *, force: bool = False) -> None:
    """Interrupt (default) or kill a process we did not spawn."""
    import signal
    if IS_WIN:
        ps = _psutil()
        if ps is None:
            return
        try:
            p = ps.Process(pid)
            p.kill() if force else p.terminate()
        except ps.Error:
            pass
        return
    os.kill(pid, signal.SIGKILL if force else signal.SIGINT)


def boot_seconds() -> int:
    """Boot time as a PID-reuse-proof liveness token (0 if unknown)."""
    if IS_MAC:
        try:
            out = subprocess.run(["sysctl", "-n", "kern.boottime"],
                                 capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"sec\s*=\s*(\d+)", out)
            return int(m.group(1)) if m else 0
        except Exception:
            return 0
    ps = _psutil()
    if ps is not None:
        try:
            return int(ps.boot_time())
        except Exception:
            pass
    if IS_LINUX:
        try:
            for line in Path("/proc/stat").read_text().splitlines():
                if line.startswith("btime"):
                    return int(line.split()[1])
        except OSError:
            pass
    return 0


def keep_awake(pid: int):
    """Stop the computer idling to sleep while process `pid` records.
    Returns a handle with .terminate(), or None (Windows does this inside the
    capture helper itself)."""
    try:
        if IS_MAC:
            return subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(pid)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
        if IS_LINUX and shutil.which("systemd-inhibit") and shutil.which("tail"):
            return subprocess.Popen(
                ["systemd-inhibit", "--what=sleep:idle", "--who=Lectern",
                 "--why=Recording a lecture", "--mode=block",
                 "tail", f"--pid={pid}", "-f", "/dev/null"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        pass
    return None


# ------------------------------------------------------------- durability
def fsync_fd(fd: int) -> None:
    """Flush an open file all the way to the drive.
    macOS needs F_FULLFSYNC: os.fsync() leaves data in the drive's own cache."""
    if IS_MAC:
        import fcntl
        fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
    else:
        os.fsync(fd)


def fsync_path(path) -> None:
    """Durably flush a file by path (open mode differs per OS)."""
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0) if IS_WIN else os.O_RDONLY
    fd = os.open(str(path), flags)
    try:
        fsync_fd(fd)
    finally:
        os.close(fd)


def fsync_dir(path) -> None:
    """Persist a directory entry (rename). Windows cannot open directories and
    NTFS journals the metadata anyway, so it is a no-op there."""
    if IS_WIN:
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
