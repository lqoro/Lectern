"""Gapless concatenation of recorded WAV segments.

Pure stdlib `wave`, streamed in 1 MB blocks (peak RSS ~2 MB). Deliberately not
ffmpeg's concat demuxer: the final segment of an interrupted recording has a
RIFF header claiming more frames than exist (it stays 0xffffffff until a clean
close), and stdlib `wave` was verified to read exactly the frames that are
actually present in that case.

Duration therefore comes from decoded PCM frame counts and never from a header.
"""
from __future__ import annotations

import wave
from pathlib import Path

from . import config, storage

_BLOCK = 1 << 20


_WAV_HEADER_BYTES = 44   # every writer here (Swift, Python, ffmpeg bitexact, wave) emits this


def _stream_segment(seg: Path, out) -> tuple[int, str | None]:
    """Copy one segment's PCM into `out` (None: only count it). Returns
    (frames, error). This is the single definition of how many frames a
    segment contributes to master.wav, so anything that needs a position on
    the master timeline (live.py) counts with it too."""
    frames = 0
    try:
        with wave.open(str(seg)) as w:
            if (w.getnchannels() != config.CHANNELS
                    or w.getsampwidth() != config.SAMPLE_WIDTH):
                return 0, "format mismatch"
            while True:
                # readframes() returns only frames that truly exist,
                # even when the header lies about the count.
                chunk = w.readframes(_BLOCK // config.SAMPLE_WIDTH)
                if not chunk:
                    break
                if out is not None:
                    out.writeframes(chunk)
                frames += len(chunk) // config.SAMPLE_WIDTH
    except (wave.Error, EOFError, OSError) as e:
        return frames, str(e)
    return frames, None


def copy_segment(seg: Path, dest: Path) -> int:
    """One segment as a standalone WAV with a correct header; returns the
    frames it holds on the master timeline (exactly what concat counts)."""
    with wave.open(str(dest), "wb") as out:
        out.setnchannels(config.CHANNELS)
        out.setsampwidth(config.SAMPLE_WIDTH)
        out.setframerate(config.SAMPLE_RATE)
        frames, _err = _stream_segment(seg, out)
    return frames


def concat_segments(segments: list[Path], dest: Path) -> dict:
    """Write `dest` from `segments`; return a manifest with true frame counts."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    per_segment: list[dict] = []
    total = 0

    with wave.open(str(part), "wb") as out:
        out.setnchannels(config.CHANNELS)
        out.setsampwidth(config.SAMPLE_WIDTH)
        out.setframerate(config.SAMPLE_RATE)
        for seg in segments:
            frames, err = _stream_segment(seg, out)
            entry = {"file": seg.name, "frames": frames}
            if err:
                entry["error"] = err
            per_segment.append(entry)
            total += frames

    part.replace(dest)
    manifest = {
        "segments": per_segment,
        "total_frames": total,
        "sample_rate": config.SAMPLE_RATE,
        "duration_ms": int(total * 1000 / config.SAMPLE_RATE),
    }
    storage.atomic_write_json(dest.parent / "concat.manifest.json", manifest,
                              indent=2)
    return manifest


def wav_duration_ms(path: Path) -> int:
    """Duration from the bytes actually on disk, never from the RIFF header:
    a writer killed mid-segment leaves 0xFFFFFFFF there (about 37 hours)."""
    data = max(0, path.stat().st_size - _WAV_HEADER_BYTES)
    return data * 1000 // config.BYTES_PER_SEC
