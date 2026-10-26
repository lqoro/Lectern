"""Capture-device enumeration and preflight probing.

Two verified hazards drive this module:
  1. avfoundation device indices are volatile. A Continuity iPhone mic occupies
     index 1 on this machine and shifts every index below it when it comes and
     goes. We therefore resolve BY NAME immediately before every spawn.
  2. A recording that produces pure silence (TCC denial, muted input, wrong
     device) looks exactly like a successful recording until you play it back
     50 minutes later. We probe before committing to a lecture.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from . import capture, config, pcm
from . import platform_compat as pc

_AUDIO_HEADER = re.compile(r"AVFoundation audio devices:")
_DEVICE_LINE = re.compile(r"^\[AVFoundation indev @ [^\]]+\] \[(\d+)\] (.+)$")

# libavdevice reports device-layer failures with these strings; treat them as
# hard open errors rather than as silence.
_DEVICE_ERROR_MARKERS = (
    "audio format not available",
    "audio format is not supported",
    "Input/output error",
    "Operation not permitted",
    "Cannot open audio device",
    "Failed to open",
)


@dataclass(frozen=True)
class Device:
    index: int
    name: str


class PreflightError(RuntimeError):
    """Raised when the microphone cannot be used. `kind` is machine-readable."""

    def __init__(self, kind: str, message: str, detail: str = ""):
        super().__init__(message)
        self.kind = kind          # no_device | device_error | silent
        self.message = message
        self.detail = detail


def _ffmpeg_missing() -> PreflightError:
    return PreflightError(
        "ffmpeg_missing", "ffmpeg is not installed.",
        "On a Mac: brew install ffmpeg (or run ./install.sh), then restart Lectern.")


# Enumeration is slow and variable: measured 4.96 s on the first call in a fresh
# process and ~1 s warm, because avfoundation probes Continuity devices. It is
# also called on both preflight and record-start, so results are cached briefly.
_ENUM_TIMEOUT = 30.0
_CACHE_TTL = 15.0
_cache: tuple[float, list["Device"]] | None = None


def list_audio_devices(timeout: float = _ENUM_TIMEOUT,
                       use_cache: bool = True) -> list[Device]:
    """Enumerate audio inputs. On macOS this asks avfoundation (ffmpeg exits 1
    here by design); everywhere else it asks the portable helper."""
    global _cache
    if use_cache and _cache is not None:
        age = time.monotonic() - _cache[0]
        if age < _CACHE_TTL:
            return _cache[1]
    if capture.portable():
        out = _list_portable(timeout)
        if out:
            _cache = (time.monotonic(), out)
        return out
    try:
        proc = subprocess.run(
            [config.FFMPEG, "-hide_banner", "-f", "avfoundation",
             "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        raise _ffmpeg_missing() from None
    except subprocess.TimeoutExpired as e:
        raise PreflightError(
            "device_error",
            "Listing audio devices timed out.",
            "avfoundation did not respond within "
            f"{timeout:.0f}s. Disconnecting an iPhone or Continuity Camera "
            "usually fixes this.",
        ) from e
    out: list[Device] = []
    in_audio = False
    for line in proc.stderr.splitlines():
        if _AUDIO_HEADER.search(line):
            in_audio = True
            continue
        m = _DEVICE_LINE.match(line.strip())
        if not m:
            continue
        if in_audio:
            out.append(Device(int(m.group(1)), m.group(2).strip()))
    if out:
        _cache = (time.monotonic(), out)
    return out


_CAPTURE_PY = str(Path(__file__).with_name("capture_py.py"))


def _list_portable(timeout: float) -> list[Device]:
    """Device list from capture_py (PortAudio). Run as a child process so a
    misbehaving audio driver can never take the backend down with it. The
    system default input comes first, which is what resolve_device falls back
    to when no preferred name matches."""
    import os
    if os.environ.get("LECTERN_CAPTURE_SOURCE") == "synthetic":
        return [Device(0, "Synthetic tone")]
    try:
        proc = subprocess.run([sys.executable, _CAPTURE_PY, "--list-devices"],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise PreflightError("device_error", "Listing audio devices timed out.",
                             "the audio system did not respond") from e
    if proc.returncode == capture_py_exit_no_lib():
        raise PreflightError(
            "device_error", "The audio library is not installed.",
            "Run:  pip install sounddevice numpy   (then restart Lectern)")
    if proc.returncode != 0:
        raise PreflightError("device_error", "Could not list audio devices.",
                             proc.stderr.strip()[-800:])
    try:
        rows = json.loads(proc.stdout or "[]")
    except ValueError:
        rows = []
    rows.sort(key=lambda r: not r.get("default"))
    return [Device(int(r["index"]), r["name"]) for r in rows]


def capture_py_exit_no_lib() -> int:
    return 4


def resolve_device(preferred: list[str] | None = None,
                   fresh: bool = False) -> Device:
    """Pick a capture device by NAME. Never trust a cached index."""
    devices = list_audio_devices(use_cache=not fresh)
    if not devices:
        raise PreflightError(
            "no_device",
            "No microphone was found.",
            "ffmpeg -f avfoundation -list_devices true returned no audio inputs.",
        )
    wanted = preferred if preferred is not None else config.PREFERRED_DEVICES
    for name in wanted:
        for d in devices:
            if d.name == name:
                return d
    for name in wanted:                       # relaxed substring match
        for d in devices:
            if name.lower() in d.name.lower():
                return d
    return devices[0]


def probe(device: Device, seconds: float = 1.8) -> dict:
    """Record a short burst and classify the result.

    Returns {rms, peak, frames} on success. Raises PreflightError with
    kind='device_error' or 'silent' otherwise. This is the difference between
    discovering a problem now and discovering it after a 50-minute lecture.

    avfoundation devices take real time to start streaming after open —
    measured ~0.3s of warm-up on this hardware — so the window must clear
    that latency by a comfortable margin or a working mic fails the "almost
    no audio" check below before it ever gets going.
    """
    if capture.portable():
        return _probe_portable(device, seconds)
    with tempfile.TemporaryDirectory(prefix="lectern-probe-") as td:
        wav = Path(td) / "probe.wav"
        try:
            proc = subprocess.run(
                [config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "warning",
                 "-f", "avfoundation", "-i", f":{device.index}",
                 "-ac", str(config.CHANNELS), "-ar", str(config.SAMPLE_RATE),
                 "-c:a", "pcm_s16le", "-fflags", "+bitexact",
                 "-t", str(seconds), "-y", str(wav)],
                capture_output=True, text=True, errors="replace",
                timeout=seconds + 25,
            )
        except FileNotFoundError:
            raise _ffmpeg_missing() from None
        except subprocess.TimeoutExpired as e:
            raise PreflightError(
                "device_error", "The microphone did not respond.",
                "avfoundation hung while opening the device.") from e
        stderr = proc.stderr or ""
        if proc.returncode != 0 or not wav.exists() or wav.stat().st_size <= 44:
            kind, msg = _classify_open_failure(stderr)
            raise PreflightError(kind, msg, stderr.strip()[-800:])

        with wave.open(str(wav)) as w:
            frames = w.getnframes()
            raw = w.readframes(frames)

    if frames < int(config.SAMPLE_RATE * 0.3):
        raise PreflightError(
            "device_error",
            "The microphone opened but produced almost no audio.",
            f"only {frames} frames captured",
        )

    return _classify_audio(raw, frames)


def _privacy_hint() -> str:
    if pc.IS_MAC:
        return ("Check System Settings > Privacy & Security > Microphone, and "
                "that the input is not muted.")
    if pc.IS_WIN:
        return ("Check Settings > Privacy & security > Microphone (allow desktop "
                "apps), and that the input is not muted.")
    return "Check that the microphone is not muted and that your user can access it."


def _classify_audio(raw: bytes, frames: int) -> dict:
    rms = pcm.rms(raw)
    peak = pcm.peak(raw)
    if peak == 0:
        raise PreflightError(
            "silent",
            "No audio is reaching the recorder — every sample was exactly zero.",
            _privacy_hint(),
        )
    return {"rms": rms, "peak": peak, "frames": frames,
            "seconds": frames / config.SAMPLE_RATE}


def _probe_portable(device: Device, seconds: float) -> dict:
    """Same idea as the ffmpeg probe, but it runs the REAL capture helper, so a
    passing probe proves the exact code path a lecture will use."""
    with tempfile.TemporaryDirectory(prefix="lectern-probe-") as td:
        proc = subprocess.run(
            [sys.executable, _CAPTURE_PY, "--device", device.name,
             "--out", str(Path(td) / "probe-%03d.wav"),
             "--segment-seconds", "10", "--duration", str(seconds)],
            capture_output=True, text=True, timeout=seconds + 45)
        stderr = (proc.stderr or "").strip()
        wav = Path(td) / "probe-000.wav"
        if proc.returncode == capture_py_exit_no_lib():
            raise PreflightError(
                "device_error", "The audio library is not installed.",
                "Run:  pip install sounddevice numpy   (then restart Lectern)")
        if proc.returncode != 0 or not wav.exists() or wav.stat().st_size <= 44:
            msg = ("The microphone could not be opened."
                   if proc.returncode == 3 else "The microphone test failed.")
            raise PreflightError("device_error", msg + " " + _privacy_hint(),
                                 stderr[-800:])
        with wave.open(str(wav)) as w:
            frames = w.getnframes()
            raw = w.readframes(frames)
    if frames < int(config.SAMPLE_RATE * 0.3):
        raise PreflightError("device_error",
                             "The microphone opened but produced almost no audio.",
                             f"only {frames} frames captured")
    return _classify_audio(raw, frames)


def _classify_open_failure(stderr: str) -> tuple[str, str]:
    low = stderr.lower()
    for marker in _DEVICE_ERROR_MARKERS:
        if marker.lower() in low:
            if "permitted" in low or "denied" in low:
                return ("device_error",
                        "macOS blocked microphone access for this process.")
            return ("device_error", "The microphone could not be opened.")
    return ("device_error", "The microphone could not be opened.")
