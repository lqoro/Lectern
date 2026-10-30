"""Startup crash recovery and the deferred WAV release.

Two jobs run on every backend start:

1. Find lectures whose recording was interrupted, salvage what exists, and say
   plainly what was lost. A lecture is never silently dropped and a failure
   never deletes anything.
2. Sweep .trash: for each released working directory, re-verify the archive a
   SECOND time, in this new process, before finally unlinking the WAVs. Losing
   a lecture therefore requires two independent full decodes of a good file to
   both be wrong.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from . import archive, config, storage
from .storage import Lecture

# An import whose upload file was written this recently is still streaming in.
_UPLOAD_GRACE_S = 120

log = logging.getLogger(__name__)

TRASH_MAX_AGE_S = 7 * 24 * 3600
MIN_SALVAGE_MS = 5000


# ------------------------------------------------------------ recovery
def scan_interrupted() -> list[Lecture]:
    """Lectures in a non-terminal state whose writer is gone.

    "paused" is deliberately excluded: it is a clean, intentional stop with no
    writer at all by design, and must not be mistaken for a crash. Only the
    user's own Resume action should move a paused lecture forward.
    """
    from .recorder import _read_recording_json, _writer_alive
    out = []
    for lec in Lecture.all():
        m = lec.manifest()
        if m.get("state") in storage.TERMINAL or m.get("state") == "paused":
            continue
        if not lec.work.is_dir():
            continue
        info = _read_recording_json(lec)
        if info and _writer_alive(info):
            continue                      # still running: adopt, do not recover
        out.append(lec)
    return out


def recover(lec: Lecture) -> dict:
    """Get an interrupted lecture moving again. Never deletes anything.

    Two different situations reach this function, and they deserve different
    messages. If the backend died while ffmpeg was still the active writer
    (state was "recording"), the tail of the last segment may genuinely be
    lost. If capture had ALREADY finished cleanly -- state was already past
    "recording" (concatenating/transcribing/noting/archiving) -- then every
    segment is complete and nothing was lost; only the pipeline needs to run
    again. Conflating the two produces a false "the last few seconds were
    lost" alarm on a recording that lost nothing.
    """
    from . import concat

    was_still_recording = lec.manifest().get("state") == "recording"

    lec.set_state("interrupted")
    segs = lec.segments()
    logp = lec.path / "recovery.log"
    lines = [f"[{storage.utcnow()}] recovery started",
             f"was still recording when interrupted: {was_still_recording}",
             f"segments found: {len(segs)}"]

    if not segs:
        lines.append("no usable audio; leaving .work untouched")
        storage.atomic_write_text(logp, "\n".join(lines) + "\n")
        lec.set_state("failed", recovery_note=(
            "This recording could not be recovered — no audio was captured. "
            "Nothing has been deleted."))
        return {"recovered": False, "reason": "no segments"}

    man = concat.concat_segments(segs, lec.master_wav)
    salvaged_ms = man["duration_ms"]
    lines.append(f"salvaged: {salvaged_ms} ms from {len(segs)} segment(s)")

    if salvaged_ms < MIN_SALVAGE_MS:
        lines.append("below the minimum useful length; leaving .work untouched")
        storage.atomic_write_text(logp, "\n".join(lines) + "\n")
        lec.set_state("failed", recovery_note=(
            f"Only {salvaged_ms/1000:.1f} seconds of audio survived, which is too "
            "little to transcribe. The raw files have been left in place."))
        return {"recovered": False, "reason": "too short"}

    audio = dict(lec.manifest().get("audio") or {})
    audio.update({"duration_ms": salvaged_ms,
                  "pcm_frames": man["total_frames"],
                  "segments_used": len(segs)})

    if was_still_recording:
        # The tail of the last segment is the part a crash can eat.
        note = (f"Recording ended unexpectedly. Salvaged "
                f"{salvaged_ms//60000}m{salvaged_ms//1000%60:02d}s; "
                f"about the last few seconds were lost.")
        lec.update(audio=audio, recovered=True, recovery_note=note)
    else:
        # Capture had already finished cleanly before the backend died;
        # only transcribing/notes/archiving was interrupted. Every segment is
        # complete, so say plainly that nothing was lost.
        note = None
        lec.update(audio=audio, recovered=False, recovery_note=None)

    lines.append(f"note: {note or '(none — recording had already finished cleanly)'}")
    storage.atomic_write_text(logp, "\n".join(lines) + "\n")
    return {"recovered": was_still_recording, "salvaged_ms": salvaged_ms, "note": note}


def finish_released(lec: Lecture) -> bool:
    """The last pipeline step installs the archive, moves .work to the trash and
    then marks "complete". A crash between those leaves "archiving" with no
    .work, which scan_interrupted never looks at; a failed move (a file held
    open on Windows) leaves a complete lecture still holding .work. Finish either."""
    if not lec.archive_path.is_file():
        return False
    state = lec.manifest().get("state")
    if state == "archiving" and not lec.work.is_dir():
        lec.set_state("complete")
        return True
    if state == "complete" and lec.work.is_dir() and not (lec.work / "recording.json").exists():
        from .archive import release_work
        release_work(lec)
        return True
    return False


def run_startup_recovery(submit, submit_import=None,
                         is_running=lambda _id: False) -> list[str]:
    """Recover every interrupted lecture and requeue it. Returns lecture ids.
    is_running: a job a user started meanwhile owns its lecture; leave it."""
    from .recorder import self_pause_reason
    for lec in Lecture.all():
        try:
            if finish_released(lec):
                log.warning("finished the interrupted last step of %s", lec.id)
        except Exception:
            log.exception("could not finish %s", lec.id)
    done = []
    for lec in scan_interrupted():
        if is_running(lec.id):
            continue
        try:
            if lec.manifest().get("state") == "importing":
                # An import has no live writer to lose: the uploaded file is
                # still in .work, so start the extraction over from scratch.
                from . import importer
                part = lec.work / (importer.SOURCE_NAME + ".part")
                try:
                    uploading = time.time() - part.stat().st_mtime < _UPLOAD_GRACE_S
                except OSError:
                    uploading = False
                if uploading:
                    # Recovery runs while the server already accepts requests,
                    # so this may be an upload that started seconds ago.
                    continue
                if importer.source_path(lec).is_file() and submit_import:
                    # extract() writes a .part and renames, so re-running it
                    # simply replaces any seed segment from the first attempt.
                    submit_import(lec.id)
                    done.append(lec.id)
                    log.warning("restarting interrupted import %s", lec.id)
                else:
                    lec.set_state("failed", failure="the upload did not finish; "
                                  "import the file again")
                continue
            # The Swift helper pauses itself for the lid or sleep even when no
            # backend is running to hear it (the app was quit, or this backend
            # restarted mid-lecture). That is a clean pause, not a crash: keep
            # it "paused" for the user to continue or finish.
            why = (self_pause_reason(lec)
                   if lec.manifest().get("state") == "recording" else None)
            if why:
                lec.set_state("paused", pause_reason=why, paused_utc=storage.utcnow())
                log.warning("%s paused itself while no backend was running (%s)",
                            lec.id, why)
                continue
            res = recover(lec)
            log.warning("recovered %s: %s", lec.id, res)
            # "recovered" only says whether the tail of the audio may be lost.
            # A lecture whose capture had finished cleanly (recovered=False,
            # no "reason") still needs its pipeline run; requeueing only
            # recovered=True left one stuck in "interrupted" across restarts.
            if "reason" not in res:
                submit(lec.id)
                done.append(lec.id)
        except Exception:
            log.exception("recovery failed for %s", lec.id)
    return done


# ------------------------------------------------- deferred WAV release
def _concat_had_errors(work_dir: Path) -> bool:
    try:
        man = json.loads((work_dir / "concat.manifest.json").read_text())
    except (OSError, ValueError):
        return False
    return any(seg.get("error") for seg in man.get("segments") or [])


def _reverify(lec: Lecture) -> bool:
    """Re-run the codec and full-decode checks against the installed archive."""
    p = lec.archive_path
    if not p.is_file() or p.stat().st_size == 0:
        return False
    try:
        probe = archive._ffprobe(p)
        streams = probe.get("streams") or []
        if not streams or streams[0].get("codec_name") != "aac":
            return False
        proc = subprocess.run(
            [config.FFMPEG, "-nostdin", "-v", "error", "-i", str(p), "-f", "null", "-"],
            capture_output=True, text=True, timeout=3600)
        return proc.returncode == 0 and not proc.stderr.strip()
    except Exception:
        log.exception("re-verification errored for %s", lec.id)
        return False


# Deletion is an explicit allowlist, non-recursive, symlink-refusing.
# There is deliberately no shutil.rmtree anywhere in this codebase: a leftover
# directory is harmless, a wrong rmtree is a semester.
_DELETABLE_GLOBS = ("seg-*.wav",)
_DELETABLE_NAMES = ("master.wav", "master.wav.part", "concat.manifest.json",
                    "import-source", "import-source.part", "import.wav.part",
                    "import.stderr.log",
                    "whisper.raw.json", "whisper.stderr.log", "record.stderr.log",
                    "recording.json", "stage", ".metadata_never_index",
                    "STOP")             # the portable writer's stop request


def _purge_work_dir(work: Path) -> int:
    assert work.name.endswith("__work") or "__work-" in work.name, work
    assert work.parent == config.TRASH_DIR, work
    assert config.TRASH_DIR.parent == config.LECTURES_ROOT
    n = 0
    targets: list[Path] = []
    for g in _DELETABLE_GLOBS:
        targets.extend(sorted(work.glob(g)))
    targets.extend(work / name for name in _DELETABLE_NAMES)
    for p in targets:
        if p.is_file() and not p.is_symlink():
            p.unlink(missing_ok=True)
            n += 1
    try:
        work.rmdir()          # ENOTEMPTY if anything unexpected is still there
    except OSError as e:
        log.warning("leaving %s in place: %s", work, e)
    return n


def sweep_trash() -> dict:
    """Second-phase release, plus the 7-day hard sweep."""
    config.ensure_dirs()
    freed = 0
    purged = 0
    kept = 0
    for entry in sorted(config.TRASH_DIR.iterdir()):
        if entry.name.startswith("."):
            continue
        age = time.time() - entry.stat().st_mtime

        if "__work" in entry.name and entry.is_dir():
            lecture_id = entry.name.split("__work")[0]
            try:
                lec = Lecture.load(lecture_id)
            except (FileNotFoundError, ValueError):
                lec = None
            try:
                if _concat_had_errors(entry):
                    # A segment could not be read into the master, so the
                    # archive is missing audio that may only exist in these
                    # WAVs. Keep them for a person to look at.
                    kept += 1
                    log.warning("keeping %s: a segment failed to concatenate", entry.name)
                elif lec is not None and _reverify(lec):
                    size = sum(p.stat().st_size for p in entry.glob("*") if p.is_file())
                    purged += _purge_work_dir(entry)
                    freed += size
                    # Re-load: the lecture may have been filed into a class
                    # while its archive was being decoded (that takes a while).
                    lec = Lecture.load(lecture_id)
                    m = lec.manifest()
                    audio = dict(m.get("audio") or {})
                    audio["wav_released_utc"] = storage.utcnow()
                    lec.update(audio=audio)
                    log.info("released working files for %s (%.1f MB)", lecture_id, size / 1e6)
                else:
                    kept += 1
                    log.warning("keeping %s: archive did not re-verify", entry.name)
            except Exception:
                # One lecture moving or failing must not stop the rest of the
                # sweep; whatever is left is retried on the next start.
                log.exception("could not release %s", entry.name)
            continue

        if age > TRASH_MAX_AGE_S and entry.is_dir():
            try:
                size = _dir_size(entry)
                n = _purge_trashed_lecture(entry)
                purged += n
                freed += size
                log.info("swept %s from trash after 7 days (%d files, %.1f MB)",
                         entry.name, n, size / 1e6)
            except Exception:
                log.exception("could not sweep %s", entry.name)
    return {"files_removed": purged, "bytes_freed": freed, "dirs_kept": kept}


def _dir_size(d: Path) -> int:
    return sum(p.stat().st_size for p in d.rglob("*")
               if p.is_file() and not p.is_symlink())


# Files a lecture directory is allowed to contain. Anything else is left behind
# rather than guessed at, which is why this is a suffix allowlist and not a
# recursive delete. A leftover file is harmless; a wrong delete is a semester.
_LECTURE_SUFFIXES = {".json", ".md", ".txt", ".wav", ".m4a", ".opus", ".log", ""}
_MAX_DEPTH = 2


def _purge_trashed_lecture(root: Path) -> int:
    """Remove one expired directory from .trash. Never used outside .trash."""
    assert root.parent == config.TRASH_DIR, root
    assert config.TRASH_DIR.parent == config.LECTURES_ROOT, config.TRASH_DIR
    assert not root.is_symlink(), root

    removed = 0
    dirs: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel_depth = len(path.relative_to(root).parts)
        if rel_depth > _MAX_DEPTH or path.is_symlink():
            continue
        if path.is_dir():
            dirs.append(path)
        elif path.is_file() and path.suffix.lower() in _LECTURE_SUFFIXES:
            path.unlink(missing_ok=True)
            removed += 1
    for d in dirs:                       # deepest first; ENOTEMPTY is fine
        try:
            d.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError as e:
        log.warning("leaving %s in place: %s", root, e)
    return removed
