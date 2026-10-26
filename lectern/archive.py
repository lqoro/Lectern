"""AAC archiving and the verification gate.

Deleting the WAVs is the single most dangerous operation in this app: it is the
only step that can destroy the sole copy of a lecture. Every check below runs
BEFORE a single WAV byte is touched, and even then the WAVs are only moved to
.trash -- the actual unlink happens on a LATER startup, in a different process,
after the archive verifies a second time.

AAC rather than Opus: macOS 26's AVFoundation over-reports Ogg Opus duration by
7.3% on a 50-minute file (and returns nonsense for Opus-in-CAF), which breaks the
seek bar and therefore click-to-jump. AAC is reported exactly, and costs about
1.5 GB more across a four-year degree.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import concat, config, storage
from . import platform_compat as pc
from .storage import Lecture


class GateError(RuntimeError):
    """A verification check failed. The WAVs are always left untouched."""

    def __init__(self, gate: str, message: str):
        super().__init__(f"[{gate}] {message}")
        self.gate = gate
        self.message = message


def _ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [config.FFPROBE, "-v", "error", "-show_entries",
         "format=duration:stream=codec_name,channels", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise GateError("G2", f"ffprobe failed: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise GateError("G2", f"ffprobe returned unparseable JSON: {e}") from e


def encode(lec: Lecture, *, progress_cb=None) -> Path:
    """Encode master.wav to AAC. Writes to .part; does not install it."""
    src = lec.master_wav
    if not src.is_file():
        raise GateError("G0", "master.wav is missing; nothing to archive")
    part = lec.archive_path.with_suffix(lec.archive_path.suffix + ".part")
    part.unlink(missing_ok=True)

    for codec in (config.ARCHIVE_CODEC, config.ARCHIVE_CODEC_FALLBACK):
        argv, extra = pc.low_priority(
            # -f is explicit: the .part suffix stops ffmpeg inferring a muxer
            # from the extension, and it fails with "Unable to find a suitable
            # output format" instead of writing anything.
            [config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-y", "-i", str(src), "-c:a", codec, "-b:a", config.ARCHIVE_BITRATE,
             "-ac", str(config.CHANNELS), "-movflags", "+faststart",
             "-f", config.ARCHIVE_MUXER, str(part)])
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=3600, **extra)
        if proc.returncode == 0 and part.is_file() and part.stat().st_size > 0:
            return part
        part.unlink(missing_ok=True)
    raise GateError("G1", f"AAC encoding failed: {proc.stderr.strip()[:300]}")


def run_gate(lec: Lecture, part: Path) -> dict:
    """G1-G8. Raises GateError on any failure; returns the recorded results."""
    results: dict = {}

    # G1 - the encode produced something
    if not part.is_file() or part.stat().st_size == 0:
        raise GateError("G1", "encoded file is missing or empty")
    results["g1"] = True

    # G2 - it is really AAC, really mono, and readable
    probe = _ffprobe(part)
    streams = probe.get("streams") or []
    if not streams:
        raise GateError("G2", "encoded file has no audio stream")
    codec = streams[0].get("codec_name")
    channels = int(streams[0].get("channels") or 0)
    if codec != "aac":
        raise GateError("G2", f"expected aac, got {codec!r}")
    if channels != config.CHANNELS:
        raise GateError("G2", f"expected {config.CHANNELS} channel(s), got {channels}")
    results["g2"] = True

    # G3 - duration matches the TRUE frame count, never a header
    man_path = lec.work / "concat.manifest.json"
    try:
        wav_ms = int(json.loads(man_path.read_text())["duration_ms"])
    except Exception:
        wav_ms = concat.wav_duration_ms(lec.master_wav)
    try:
        aac_ms = int(float(probe["format"]["duration"]) * 1000)
    except (KeyError, TypeError, ValueError) as e:
        raise GateError("G3", f"could not read encoded duration: {e}") from e
    tol = max(1000.0, 0.005 * wav_ms)
    delta = abs(aac_ms - wav_ms)
    if delta > tol:
        raise GateError(
            "G3", f"duration mismatch: source {wav_ms} ms vs archive {aac_ms} ms "
                  f"(off by {delta} ms, tolerance {tol:.0f} ms)")
    results["g3_delta_ms"] = delta

    # G4 - full decode of every packet; a header probe would miss a corrupt tail
    argv, extra = pc.low_priority(
        [config.FFMPEG, "-nostdin", "-v", "error", "-i", str(part), "-f", "null", "-"])
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=3600, **extra)
    if proc.returncode != 0 or proc.stderr.strip():
        raise GateError("G4", f"full decode reported problems: "
                              f"{(proc.stderr or 'exit ' + str(proc.returncode)).strip()[:300]}")
    results["g4"] = True

    # G5 - not a silent or degenerate encode
    expected = (int(config.ARCHIVE_BITRATE.rstrip("k")) * 1000 / 8) * (wav_ms / 1000)
    if part.stat().st_size < 0.5 * expected:
        raise GateError("G5", f"archive is implausibly small "
                              f"({part.stat().st_size} bytes vs ~{int(expected)} expected)")
    results["g5"] = True
    return results


def install(lec: Lecture, part: Path, gate: dict) -> dict:
    """G6-G8: durable rename into place, re-stat, checksum."""
    import os

    from . import platform_compat as pc

    pc.fsync_path(part)                          # G6: survive a power cut
    size_before = part.stat().st_size
    os.replace(part, lec.archive_path)
    pc.fsync_dir(lec.path)

    # G7 - it really landed, at the expected size
    if not lec.archive_path.is_file():
        raise GateError("G7", "archive is not at its final path after rename")
    size_after = lec.archive_path.stat().st_size
    if size_after != size_before:
        raise GateError("G7", f"size changed across rename: {size_before} -> {size_after}")

    # G8 - checksum, streamed
    h = hashlib.sha256()
    with open(lec.archive_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    gate = dict(gate)
    gate.update({"g6": True, "g7": True, "passed_utc": storage.utcnow()})
    return {"sha256": h.hexdigest(), "bytes": size_after, "gate": gate}


def release_work(lec: Lecture) -> Path | None:
    """G9: move .work to .trash. Nothing is unlinked here, deliberately."""
    if not lec.work.is_dir():
        return None
    config.ensure_dirs()
    dest = config.TRASH_DIR / f"{lec.id}__work"
    n = 2
    while dest.exists():
        dest = config.TRASH_DIR / f"{lec.id}__work-{n}"
        n += 1
    lec.work.rename(dest)                        # same filesystem: atomic, instant
    return dest


def archive_lecture(lec: Lecture, *, progress_cb=None) -> dict:
    """Encode, verify, install, and release the working directory."""
    part = encode(lec, progress_cb=progress_cb)
    try:
        gate = run_gate(lec, part)
    except Exception:
        part.unlink(missing_ok=True)             # never leave a suspect .part around
        raise
    info = install(lec, part, gate)

    m = lec.manifest()
    audio = dict(m.get("audio") or {})
    audio.update({
        "file": lec.archive_path.name,
        "bytes": info["bytes"],
        "sha256": info["sha256"],
        "codec": "aac",
        "bitrate": int(config.ARCHIVE_BITRATE.rstrip("k")) * 1000,
        "gate": info["gate"],
        "wav_released_utc": None,
    })
    lec.update(audio=audio)

    trashed = release_work(lec)
    audio["wav_released_utc"] = storage.utcnow()
    audio["work_trashed_to"] = trashed.name if trashed else None
    lec.update(audio=audio)
    return info
