"""Central configuration. Every path and tunable lives here."""
from __future__ import annotations
import os
from pathlib import Path

from . import platform_compat as _pc

APP_NAME = "Lectern"
VERSION = "0.1.0"

# --- paths -------------------------------------------------------------
# TRUTH: plain files, visible in Finder, outlive the app.
# Deliberately NOT ~/Documents: iCloud Desktop & Documents sync is enabled on
# this machine, which would evict lectures to network placeholders and corrupt
# SQLite WAL. See plan "Findings that changed the design" #5.
LECTURES_ROOT = Path(os.environ.get("LECTERN_ROOT", Path.home() / "Lectures"))
TRASH_DIR = LECTURES_ROOT / ".trash"

# DERIVED: rebuildable index. Safe to delete at any time.
# LECTERN_SUPPORT points a test copy at its own index/logs/helper binary.
SUPPORT_DIR = (Path(os.environ["LECTERN_SUPPORT"]) if os.environ.get("LECTERN_SUPPORT")
               else _pc.support_dir(APP_NAME))
INDEX_DB = SUPPORT_DIR / "index.sqlite3"
LOG_DIR = SUPPORT_DIR / "logs"

# --- binaries ----------------------------------------------------------
FFMPEG = _pc.find_binary("ffmpeg", "LECTERN_FFMPEG", ("/opt/homebrew/bin/ffmpeg",))
FFPROBE = _pc.find_binary("ffprobe", "LECTERN_FFPROBE", ("/opt/homebrew/bin/ffprobe",))
WHISPER_CLI = _pc.find_binary("whisper-cli", "LECTERN_WHISPER", ("/opt/homebrew/bin/whisper-cli",))
# The speech model is chosen at run time by models.active(); see models.py.

# Native capture helper (see lectern_capture.swift for why ffmpeg's
# avfoundation input is no longer used to record). Built from source on first
# use; the binary is derived and safe to delete.
CAPTURE_SRC = Path(__file__).with_name("lectern_capture.swift")
CAPTURE_BIN = SUPPORT_DIR / "bin" / "lectern-capture"

# --- audio -------------------------------------------------------------
SAMPLE_RATE = 16000          # whisper's native rate; no resample anywhere
CHANNELS = 1
SAMPLE_WIDTH = 2             # pcm_s16le
BYTES_PER_SEC = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH   # 32000
SEGMENT_SECONDS = 60

# Preferred capture device, matched BY NAME. Indices are volatile on this
# machine: a Continuity "iPhone Microphone" sits at index 1 and shifts
# every index below it when it appears/disappears.
PREFERRED_DEVICES = ["MacBook Air Microphone", "MacBook Pro Microphone", "Built-in Microphone"]

# --- archive -----------------------------------------------------------
# AAC, not Opus: macOS 26 AVFoundation over-reports Ogg Opus duration by 7.3%,
# which breaks the seek bar (and therefore click-to-jump) in Safari.
ARCHIVE_CODEC = "aac_at" if _pc.IS_MAC else "aac"   # AudioToolbox hardware encoder is macOS-only
ARCHIVE_CODEC_FALLBACK = "aac"
ARCHIVE_BITRATE = "32k"
ARCHIVE_EXT = "m4a"
ARCHIVE_MUXER = "ipod"        # what ffmpeg picks for .m4a; must be explicit for .part

# --- transcription -----------------------------------------------------
# No -dtw: whisper.cpp 1.9.1 defaults flash_attn=true, which silently disables
# DTW (every t_dtw comes back -1). No -ml 1 -sow: -ojf alone already emits
# per-token ms offsets inside readable sentence segments. No --vad: output_json
# reads raw token structs that are never remapped to the original timeline.
# Thread count is chosen at run time by hardware.whisper_threads().
WHISPER_BEAM = 1             # greedy; beam search is not worth its cost here
WHISPER_LANG = "en"

# --- local LLM for study notes ----------------------------------------
# qwen2.5:3b has 36 layers with only 2 KV heads, so its KV cache costs ~36 KiB
# per token -- 3x cheaper than llama3.2:3b at the same context. At num_ctx 8192
# that is ~288 MiB of KV on top of 1.9 GB of weights.
# Note: Qwen Research License, fine for personal coursework.
LLM_MODEL = os.environ.get("LECTERN_LLM", "qwen2.5:3b")
LLM_NUM_CTX = int(os.environ.get("LECTERN_NUM_CTX", "8192"))
# Unload immediately: 2.3 GB must not sit resident on an 8 GB machine.
LLM_KEEP_ALIVE = 0


# --- server ------------------------------------------------------------
# 8465 is taken by the user's existing LocalFlow app.
# Default is loopback only. Set LECTERN_HOST=0.0.0.0 to reach it from a phone on
# the same wifi -- that also switches on token auth (see require_token below),
# because campus networks are shared and lectures are personal.
HOST = os.environ.get("LECTERN_HOST", "127.0.0.1")
PORT = int(os.environ.get("LECTERN_PORT", "8477"))
TOKEN_FILE = SUPPORT_DIR / "access-token"

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def is_loopback_only() -> bool:
    return HOST in _LOOPBACK


def access_token() -> str:
    """Stable per-install token, created on first non-loopback start."""
    import secrets
    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        tok = TOKEN_FILE.read_text().strip()
        if tok:
            return tok
    except OSError:
        pass
    tok = secrets.token_urlsafe(18)
    TOKEN_FILE.write_text(tok)
    TOKEN_FILE.chmod(0o600)
    return tok


def lan_ip() -> str | None:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))        # TEST-NET-1; no packet is sent
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()

# --- guards ------------------------------------------------------------
MIN_FREE_BYTES_START = 2 * 1024**3    # refuse to start recording below 2 GB
MIN_FREE_BYTES_STOP = 500 * 1024**2   # stop gracefully below 500 MB


def ensure_dirs() -> None:
    for d in (LECTURES_ROOT, TRASH_DIR, SUPPORT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    marker = TRASH_DIR / ".metadata_never_index"
    if not marker.exists():
        marker.touch()
    readme = LECTURES_ROOT / "README.txt"
    if not readme.exists():
        readme.write_text(_README)


_README = """\
This folder holds your lecture recordings. It belongs to you, not to the app.

Each lecture is one folder named YYYY-MM-DD_HHMM_title. Inside:

  audio.m4a        the recording (AAC, about 15 MB per hour)
  transcript.json  every word with its exact millisecond timestamp.
                   This is what makes clicking text jump the audio.
  transcript.txt   the same transcript as plain readable paragraphs
  notes.md         study notes, in Markdown. Yours to edit -- open it in
                   Obsidian or any text editor and the app will pick up
                   your changes.
  lecture.json     title, course, duration, and processing history
  .work/           temporary files, removed once the recording is safely
                   archived. Safe to ignore.

Nothing here needs the app to be readable. If you delete Lectern entirely,
every recording, transcript and note stays exactly where it is, in formats
that any Mac can open.

The app's database lives elsewhere, in
~/Library/Application Support/Lectern/. That is only a search index and can
be deleted at any time -- it rebuilds itself from these folders on startup.

Deleted lectures are moved to .trash/ and kept for 7 days.
"""
