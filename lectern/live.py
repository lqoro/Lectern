"""Transcribe while recording: a draft transcript, one finished segment at a time.

The recorder writes 60-second segments into .work. Once a segment is finished
(the helper has moved on to the next one, or capture has ended) this module
transcribes it with the active model at low priority, shifts its word times
by the segment's start on the master.wav timeline, and rebuilds the draft:

  .work/draft.raw.json  every finished segment's engine tokens, already
                        shifted, in whisper's -ojf shape, plus which segments
                        they came from (file, frames, bytes) and "off"
  .work/draft.json      transcribe.normalize() of the above, i.e. exactly the
                        transcript contract, plus "draft": true, "complete"
                        and "live" -- what the page shows until the real one

The draft never touches transcript.json: recovery and has_transcript keep
meaning "the real transcript exists", and the full-file pass after Stop still
runs and replaces the draft (pipeline deletes it). A draft left by a crash is
disposable: recovery ignores it and the trash purge deletes it.

Recording always wins:
  * the recorder only calls begin() and capture_ended(), which touch memory
    under a lock that is never held across file IO or an engine run;
  * one worker thread, so at most one live engine process at a time, run
    below normal priority (transcribe.run_whisper) on fewer threads;
  * it never takes STAGE_LOCK. While any audio stage holds or waits for it
    (another lecture's transcription; on small machines also notes) it does
    not start a segment, and the transcription stage, before it starts its
    engine, waits at most IDLE_TIMEOUT_S for an in-flight live job and then
    kills it (wait_idle);
  * Stop does not wait for it. The pipeline waits at most FINISH_TIMEOUT_S
    for the last segment (finish), then kills the live job and carries on;
  * a failed segment, or an engine that cannot keep up -- transcribing a
    segment took longer than SLOW_RATIO times that segment's audio, on
    SLOW_RUNS_TO_SWITCH_OFF segments in a row, while recording -- switches it
    off for that recording with a log line. The draft keeps what it had; the
    full pass covers everything.
  * waiting is not falling behind: segments that queue up while another
    lecture's pipeline holds STAGE_LOCK (back-to-back classes) are caught up
    one at a time once it is free. Only engine time is ever judged, never
    time spent deferred. A live job killed to make way for a full
    transcription (wait_idle) is simply retried later.

Resume: a draft is continued when every segment it was built from is still on
disk, in the same order, with the same size; otherwise it is started afresh.
"off" is stored in the draft, so a recording it switched off for stays off
when continued.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import concat, config, hardware, models, storage, transcribe
from .storage import Lecture

log = logging.getLogger(__name__)

POLL_S = 2.0                  # how often a recording's .work is looked at
SLOW_RATIO = 1.0              # engine seconds per audio second above which a run counts as slow
SLOW_RUNS_TO_SWITCH_OFF = 2   # consecutive slow runs (while recording) that switch it off
MIN_JUDGED_AUDIO_S = 10.0     # shorter segments are not judged: fixed start-up cost dominates
FINISH_TIMEOUT_S = 30.0       # after Stop, the pipeline waits at most this for the last segment
IDLE_TIMEOUT_S = 30.0         # the transcription stage waits at most this for a live job
THREAD_SHARE = 2              # live engine runs on 1/THREAD_SHARE of the usual threads
SETTING = "live_transcribe"   # settings.json: true / false / absent (= automatic)

RAW_NAME = "draft.raw.json"
SEG_COPY_NAME = "live-seg.wav"
ENGINE_PREFIX = "live.raw"


# ------------------------------------------------------------- settings
def default_enabled() -> bool:
    """On unless the machine is small (the same RAM rule as notes)."""
    from .pipeline import notes_share_ram
    return notes_share_ram()


def explicit_setting() -> bool | None:
    v = models.load_settings().get(SETTING)
    return v if isinstance(v, bool) else None


def enabled() -> bool:
    v = explicit_setting()
    return default_enabled() if v is None else v


def set_enabled(value: bool | None) -> None:
    """True/False is the user's choice; None goes back to automatic."""
    models.save_settings(**{SETTING: value})


def describe() -> dict:
    return {"enabled": enabled(), "default": default_enabled(),
            "explicit": explicit_setting()}


# ---------------------------------------------------------------- files
def discard(lec: Lecture) -> None:
    """Remove the draft and any live leftovers (the real transcript exists)."""
    for name in (lec.draft_path.name, RAW_NAME, SEG_COPY_NAME,
                 ENGINE_PREFIX + ".json", ENGINE_PREFIX + ".fluidaudio.json"):
        try:
            (lec.work / name).unlink(missing_ok=True)
        except OSError:
            log.warning("could not remove %s of %s", name, lec.id, exc_info=True)


def _stage_busy() -> bool:
    from .pipeline import STAGE_LOCK
    return STAGE_LOCK.busy()


def _threads() -> int:
    return max(1, hardware.whisper_threads() // THREAD_SHARE)


def _kill(proc) -> None:
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:
        log.warning("could not stop a live transcription", exc_info=True)


def _shift(chunk: dict, offset_ms: int) -> list[dict]:
    """One engine segment's tokens on the master timeline. The first word of
    a segment always starts a new word (normalize() glues a token without a
    leading space onto the previous word, which here is another segment's)."""
    out = []
    for tok in chunk.get("tokens") or []:
        off = tok.get("offsets") or {}
        t0 = int(off.get("from", 0))
        out.append({"text": tok.get("text", ""),
                    "offsets": {"from": t0 + offset_ms,
                                "to": int(off.get("to", t0)) + offset_ms},
                    "p": tok.get("p", 1.0)})
    return out


def _first_word_spaced(tokens: list[dict]) -> None:
    for tok in tokens:
        text = tok["text"]
        if not text.strip() or transcribe._SPECIAL.match(text.strip()):
            continue
        if not text.startswith(" "):
            tok["text"] = " " + text
        return


@dataclass(eq=False)
class _Session:
    lecture_id: str
    capturing: bool = True
    finishing: bool = False
    closed: bool = False
    loaded: bool = False
    preempted: bool = False
    slow_runs: int = 0              # consecutive runs slower than real time x SLOW_RATIO
    off: str | None = None
    segments: list = field(default_factory=list)    # [{"file", "frames", "bytes"}]
    chunks: list = field(default_factory=list)      # [{"tokens": [...]}] per segment
    proc: object = None
    done: threading.Event = field(default_factory=threading.Event)


class LiveTranscriber:
    """One worker thread for all recordings. Its lock is held only for
    in-memory bookkeeping, never across IO or an engine run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._current: _Session | None = None
        self._thread: threading.Thread | None = None

    # ---------------------------------------------- called by the recorder
    def begin(self, lecture_id: str) -> None:
        """A writer started (or was adopted) for this lecture. Memory only."""
        with self._lock:
            s = self._sessions.get(lecture_id)
            if s is None:
                s = self._sessions[lecture_id] = _Session(lecture_id)
            s.capturing, s.finishing = True, False
            if not s.off:
                s.done.clear()
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="live-transcribe",
                                                daemon=True)
                self._thread.start()
        self._wake.set()

    def capture_ended(self, lecture_id: str) -> None:
        """Pause, Stop, or the writer exiting: every segment is now final."""
        with self._lock:
            s = self._sessions.get(lecture_id)
            if s is None:
                return
            s.capturing = False
        self._wake.set()

    # ---------------------------------------------- called by the pipeline
    def busy(self, lecture_id: str) -> bool:
        """Live work may still write into this lecture's .work."""
        with self._lock:
            s = self._sessions.get(lecture_id)
            return s is not None and not s.done.is_set()

    def finish(self, lecture_id: str, timeout: float | None = None) -> None:
        """Before the full pass: let the last segment land in the draft, for at
        most `timeout` seconds, then close the session (killing a live job
        still running). Never raises, never waits longer."""
        timeout = FINISH_TIMEOUT_S if timeout is None else timeout
        with self._lock:
            s = self._sessions.get(lecture_id)
            if s is None:
                return
            s.capturing, s.finishing = False, True
        self._wake.set()
        if not s.done.wait(timeout):
            log.warning("live draft of %s not finished after %.0f s; going on with "
                        "the full transcription", lecture_id, timeout)
        with self._lock:
            s.closed = True
            proc = s.proc
            if self._sessions.get(lecture_id) is s:
                del self._sessions[lecture_id]
        _kill(proc)
        s.done.set()
        self._wake.set()          # let an idle worker notice and exit now

    def wait_idle(self, timeout: float | None = None) -> None:
        """Before a full transcription starts its engine: no live engine may
        run beside it. Waits at most `timeout`, then kills the live job (its
        recording's live draft is switched off; the full pass covers it)."""
        timeout = IDLE_TIMEOUT_S if timeout is None else timeout
        if self._idle.wait(timeout):
            return
        with self._lock:
            s = self._current
            proc = s.proc if s is not None else None
            if s is not None:
                s.preempted = True
        log.warning("stopping a live transcription that is still running after "
                    "%.0f s, to make way for a full transcription", timeout)
        _kill(proc)

    # ------------------------------------------------------------ worker
    def _run(self) -> None:
        while True:
            self._wake.clear()
            with self._lock:
                todo = [s for s in self._sessions.values() if not s.done.is_set()]
                if not todo:
                    self._thread = None
                    return
            worked = False
            for s in todo:
                try:
                    worked = self._step(s) or worked
                except Exception as e:
                    log.exception("live transcription error for %s", s.lecture_id)
                    self._switch_off(s, None, f"unexpected error: {e}")
            if not worked:
                self._wake.wait(POLL_S)

    def _step(self, s: _Session) -> bool:
        """At most one segment for one session. True if an engine ran."""
        if s.closed:
            s.done.set()
            return False
        try:
            lec = Lecture.load(s.lecture_id)
        except (FileNotFoundError, ValueError):
            s.off = "the lecture is gone"
            s.done.set()
            return False
        if not s.loaded:
            s.loaded = True
            if not enabled():
                log.info("live transcription is off; %s is transcribed after Stop",
                         s.lecture_id)
                s.off = "off in Settings"
                s.done.set()
                return False
            self._load(s, lec)
            if s.off:
                s.done.set()
                return False
        with self._lock:
            capturing, finishing = s.capturing, s.finishing

        segs = lec.segments()
        if capturing:
            # The newest file (any size) is the one the helper is writing.
            names = sorted(p.name for p in lec.work.glob("seg-*.wav"))
            segs = [p for p in segs if names and p.name < names[-1]]
        have = [x["file"] for x in s.segments]
        if [p.name for p in segs[:len(have)]] != have:
            self._switch_off(s, lec, "the recorded segments changed")
            return False
        todo = segs[len(have):]
        if not todo:
            if not capturing:
                if s.segments:
                    self._write(s, lec, complete=True)
                s.done.set()
            return False
        if _stage_busy():
            if finishing:
                s.done.set()     # another transcription has the engine; the full pass covers it
            return False
        self._transcribe_one(s, lec, todo[0], judge=capturing)
        return True

    def _transcribe_one(self, s: _Session, lec: Lecture, seg: Path, *,
                        judge: bool = True) -> None:
        """judge: count this run toward the too-slow rule (only while
        recording; after Stop, finish() bounds the wait instead)."""
        copy = lec.work / SEG_COPY_NAME
        prefix = lec.work / ENGINE_PREFIX
        offset_frames = sum(x["frames"] for x in s.segments)
        offset_ms = round(offset_frames * 1000 / config.SAMPLE_RATE)

        def keep(proc) -> None:
            with self._lock:
                s.proc = proc
                closed = s.closed
            if closed:
                _kill(proc)

        t0 = time.monotonic()
        with self._lock:
            self._current = s
        self._idle.clear()
        try:
            size = seg.stat().st_size
            # A clean copy: the exact frames concat will count, with a correct
            # header even if the writer died mid-segment.
            frames = concat.copy_segment(seg, copy)
            raw = transcribe.run_whisper(copy, prefix, threads=_threads(), proc_cb=keep)
            data = json.loads(Path(raw).read_text())
        except Exception as e:
            if s.closed:
                return
            if s.preempted:
                # Made way for another lecture's transcription: not a failure
                # of ours. The segment is retried once the stage lock is free.
                s.preempted = False
                log.info("live draft of %s: %s will be retried", lec.id, seg.name)
                return
            why = str(e).splitlines()[0] if str(e) else type(e).__name__
            self._switch_off(s, lec, f"{seg.name} failed: {why}")
            return
        finally:
            with self._lock:
                s.proc = None
                self._current = None
            self._idle.set()
            for p in (copy, Path(str(prefix) + ".json")):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not remove %s", p, exc_info=True)
        if s.closed:
            return
        tokens: list[dict] = []
        for chunk in data.get("transcription") or []:
            tokens.extend(_shift(chunk, offset_ms))
        _first_word_spaced(tokens)
        s.segments.append({"file": seg.name, "frames": frames, "bytes": size})
        s.chunks.append({"tokens": tokens})
        self._write(s, lec, complete=False)
        took = time.monotonic() - t0          # engine time only: deferral never counts
        log.info("live draft of %s: %s in %.1f s", lec.id, seg.name, took)
        self._judge_speed(s, lec, seg, frames / config.SAMPLE_RATE, took, judge)

    def _judge_speed(self, s: _Session, lec: Lecture, seg: Path, audio_s: float,
                     took: float, judge: bool) -> None:
        """Switch off only when the engine itself cannot keep up with the
        recording. Two slow runs in a row, not one: a single run can be slow
        for reasons that pass (the first CoreML/Metal load, a Spotlight burst,
        another app), and a recording loses nothing while the next segment is
        tried; a sustained shortfall still stops it within two minutes."""
        if not judge or audio_s < MIN_JUDGED_AUDIO_S:
            return
        if took <= audio_s * SLOW_RATIO:
            s.slow_runs = 0
            return
        s.slow_runs += 1
        if s.slow_runs >= SLOW_RUNS_TO_SWITCH_OFF:
            self._switch_off(s, lec, f"too slow to keep up ({seg.name}: {took:.1f} s "
                                     f"for {audio_s:.0f} s of audio)")

    # ------------------------------------------------------------- draft
    def _load(self, s: _Session, lec: Lecture) -> None:
        """Continue the draft on disk if it still matches the segments."""
        p = lec.work / RAW_NAME
        try:
            raw = json.loads(p.read_text())
        except FileNotFoundError:
            discard(lec)                 # a draft.json without its source is stale
            return
        except (OSError, ValueError) as e:
            log.info("starting the live draft of %s afresh: %s", lec.id, e)
            discard(lec)
            return
        segs = raw.get("segments") or []
        chunks = raw.get("transcription") or []
        on_disk = lec.segments()
        ok = (len(segs) == len(chunks) and len(segs) <= len(on_disk)
              and all(isinstance(x, dict) and x.get("file") == d.name
                      and x.get("bytes") == d.stat().st_size
                      for x, d in zip(segs, on_disk)))
        if not ok:
            log.info("starting the live draft of %s afresh: its segments changed", lec.id)
            discard(lec)
            return
        s.segments, s.chunks, s.off = list(segs), list(chunks), raw.get("off")

    def _write(self, s: _Session, lec: Lecture, *, complete: bool) -> None:
        """draft.raw.json, then draft.json = normalize() of it. Never creates
        directories: a lecture moved away must not reappear as a ghost."""
        raw_path = lec.work / RAW_NAME
        storage.atomic_write_json(raw_path, {
            "schema": 1, "segments": s.segments, "off": s.off,
            "transcription": s.chunks}, mkdirs=False)
        total = sum(x["frames"] for x in s.segments)
        doc = transcribe.normalize(raw_path, int(total * 1000 / config.SAMPLE_RATE))
        doc.update({"draft": True, "complete": complete,
                    "live": {"segments": len(s.segments), "off": s.off}})
        storage.atomic_write_json(lec.draft_path, doc, mkdirs=False)

    def _switch_off(self, s: _Session, lec: Lecture | None, reason: str) -> None:
        s.off = reason
        log.warning("live transcription switched off for %s: %s (the full "
                    "transcription after Stop covers everything)", s.lecture_id, reason)
        if lec is not None and s.segments and not s.closed:
            try:
                self._write(s, lec, complete=False)
            except Exception:
                log.exception("could not save the live draft of %s", s.lecture_id)
        s.done.set()


LIVE = LiveTranscriber()
