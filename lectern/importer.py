"""Importing an existing recording (a downloaded lecture video or audio file).

The import becomes one seed segment, `.work/seg-000000.wav`, exactly the shape
the recorder leaves behind, so concat -> transcribe -> notes -> archive run
unchanged. This is the same trick Recorder._seed_from_archive_if_needed uses
to make "resume an archived lecture" just "more segments".

The uploaded file is kept at a fixed name, `.work/import-source`, until the
lecture is archived: a crash mid-extraction can then simply start over, and
the trash release (recovery._DELETABLE_NAMES) knows exactly what to remove.
No extension is needed; ffmpeg identifies the container from its content.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config
from . import platform_compat as pc
from .storage import Lecture

SOURCE_NAME = "import-source"
PART_NAME = "import.wav.part"


class ImportError_(RuntimeError):
    """A file that cannot become a lecture. The message is shown to the user."""


def source_path(lec: Lecture) -> Path:
    return lec.work / SOURCE_NAME


def probe(path: Path) -> dict:
    """Duration and audio-stream check. Raises ImportError_ with a plain message."""
    proc = subprocess.run(
        [config.FFPROBE, "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_name:format=duration,format_name",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise ImportError_("this file could not be read as audio or video")
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        raise ImportError_("this file could not be read as audio or video") from None
    if not info.get("streams"):
        raise ImportError_("this file has no sound track")
    try:
        dur = float((info.get("format") or {}).get("duration") or 0)
    except ValueError:
        dur = 0.0
    return {"duration_s": dur, "codec": info["streams"][0].get("codec_name"),
            "container": (info.get("format") or {}).get("format_name")}


def extract(lec: Lecture, *, duration_s: float = 0.0, progress_cb=None) -> Path:
    """Decode the uploaded source to the 16 kHz mono seed segment."""
    src = source_path(lec)
    if not src.is_file():
        raise ImportError_("the uploaded file is missing")
    part = lec.work / PART_NAME
    dest = lec.work / "seg-000000.wav"
    part.unlink(missing_ok=True)
    argv, extra = pc.low_priority(
        [config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src), "-vn", "-ac", str(config.CHANNELS),
         "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le",
         # +bitexact at the OUTPUT position, or the header is 78 bytes, not 44.
         "-fflags", "+bitexact", "-f", "wav",
         "-progress", "pipe:1", "-nostats", str(part)])
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, **extra)
    for line in proc.stdout:                       # type: ignore[union-attr]
        if progress_cb and duration_s > 0 and line.startswith("out_time_us="):
            try:
                done = int(line.split("=", 1)[1]) / 1e6
            except ValueError:
                continue
            try:
                progress_cb(min(100, int(100 * done / duration_s)))
            except Exception:
                pass
    err = proc.stderr.read() if proc.stderr else ""     # type: ignore[union-attr]
    rc = proc.wait()
    if rc != 0 or not part.is_file() or part.stat().st_size <= 44:
        part.unlink(missing_ok=True)
        raise ImportError_(f"could not extract the audio: {err.strip()[-300:] or rc}")
    part.replace(dest)
    return dest
