"""Supervised microphone capture.

The recording is deliberately NOT owned by the browser or even by this backend:
the writer is spawned with start_new_session=True, so it survives the tab
closing, the browser quitting, and this process dying. A restarted backend
re-adopts it.

The writer is the native helper in lectern_capture.swift. ffmpeg's avfoundation
input silently dropped ~2/3 of the audio when spawned from here (see
capture.py), so ffmpeg is only the fallback if the helper cannot be built.

Verified details for the ffmpeg fallback:
  * `-flush_packets 1` is a NO-OP with the segment muxer. Measured write
    granularity was 262144 bytes = 8.192 s, meaning the in-progress WAV sits at
    0 bytes for the first ~8 s of every segment. The option must be passed
    through to the child WAV muxer as `-segment_format_options flush_packets=1`.
  * `-fflags +bitexact` must sit at the OUTPUT position or the WAV header is
    78 bytes instead of 44.
  * ffmpeg does not report a vanished capture device, so `-progress pipe:1`
    provides the only reliable stall signal.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

from . import audio, capture, concat, config, storage
from . import pcm
from . import platform_compat as pc
from .storage import Lecture

log = logging.getLogger(__name__)

_STALL_SECONDS = 12.0

# --- laptop closed / asleep ---------------------------------------------
# With the lid shut the built-in mic is physically off, yet the recording used
# to carry on: macOS froze the writer during sleep and it simply resumed on
# wake, mic indicator on, battery draining, recording nothing useful. Three
# layers now turn that into a clean pause, mic released:
#   1. the Swift helper hears the system's will-sleep / lid-closed messages
#      itself and exits cleanly BEFORE sleep (exit code 75, logged reason);
#   2. the watcher polls the lid every few seconds (lid closed on an external
#      display does not sleep, and the ffmpeg fallback hears nothing);
#   3. the watcher notices that the machine slept -- CLOCK_MONOTONIC keeps
#      counting through sleep, CLOCK_UPTIME_RAW does not -- and pauses on wake
#      if anything is somehow still capturing.
# Reopening the laptop never restarts the mic; the lecture waits as "paused".
_LID_POLL_SECONDS = 4.0
_SLEEP_GAP_SECONDS = 5.0
# After a detected sleep, the helper may itself be mid-way through its own
# lid/sleep exit (it is frozen wherever it was when the Mac slept). Give it this
# long to finish before pausing it ourselves, so the lecture is labelled with
# the real cause ("lid closed") rather than the generic "system sleep".
_WAKE_GRACE_SECONDS = 3.0
SYSTEM_PAUSE_EXIT = 75          # lectern_capture.swift: paused by lid/sleep
REASON_LID = "lid closed"
REASON_SLEEP = "system sleep"
REASON_CRASH = "recorder stopped unexpectedly"
REASON_DISK = "disk full"


def lid_closed() -> bool:
    """True only when a MacBook lid is definitely shut."""
    try:
        out = subprocess.run(
            ["/usr/sbin/ioreg", "-r", "-k", "AppleClamshellState", "-d", "1"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return '"AppleClamshellState" = Yes' in out


def _clocks() -> tuple[float, float]:
    """(time including sleep, time excluding sleep)."""
    try:
        return (time.clock_gettime(time.CLOCK_MONOTONIC),
                time.clock_gettime(time.CLOCK_UPTIME_RAW))
    except (AttributeError, OSError):
        t = time.monotonic()
        return t, t                       # no sleep detection; layers 1-2 still work


def _builtin_mic(device_name: str) -> bool:
    """Lid-closed pausing only makes sense for the laptop's own mic: an
    external USB mic keeps working in clamshell mode with a display. MacBooks
    only: the lid check reads the macOS power registry."""
    if not pc.IS_MAC:
        return False
    n = (device_name or "").lower()
    return "macbook" in n or "built-in" in n


@dataclass
class RecordingState:
    lecture_id: str
    pid: int
    device_name: str
    device_index: int
    started_monotonic: float
    started_utc: str
    last_progress: float = field(default_factory=time.monotonic)
    out_time_us: int = 0
    stalled: bool = False
    stop_reason: str = ""     # set BEFORE we signal the writer: the exit is ours
    base_ms: int = 0          # audio recorded in EARLIER sessions, before a resume
    event: str = ""           # last "event=" line from the helper (lid/sleep)
    adopted_bytes: int = -1   # adopted writer: progress is file growth
    log_offset: int = 0       # record.stderr.log size when THIS writer started:
                              # the log spans every session of the lecture



def _require_capture_helper() -> None:
    """On macOS, refuse to record without the sample-exact Swift helper.

    The ffmpeg avfoundation fallback is measured to drop about two thirds of
    the audio, so a first-run user who presses Record while the helper is
    still compiling must wait for it, not silently lose the lecture.
    LECTERN_ALLOW_FFMPEG_CAPTURE=1 keeps the old fallback for debugging.
    """
    if capture.portable() or os.environ.get("LECTERN_ALLOW_FFMPEG_CAPTURE") == "1":
        return
    if capture.available() is not None or capture.ensure_built() is not None:
        return
    raise audio.PreflightError(
        "helper_unavailable",
        "The microphone helper could not be built. Install Apple's command line "
        "tools (run: xcode-select --install), then restart Lectern.")

class RecorderBusy(RuntimeError):
    pass


class Recorder:
    """At most one active recording per backend process."""

    def __init__(self):
        self._lock = threading.RLock()
        self._state: RecordingState | None = None
        self._proc: subprocess.Popen | None = None
        self._caffeinate: subprocess.Popen | None = None
        self._pump: threading.Thread | None = None
        self._guard: threading.Thread | None = None
        # Why the last recording ended, so the UI can say "paused because the
        # lid was closed" after the laptop is reopened.
        self._last_stop: dict | None = None

    # ------------------------------------------------------------ status
    @property
    def active(self) -> bool:
        with self._lock:
            return self._state is not None

    def current_lecture_id(self) -> str | None:
        """Cheap check used before moving files: which lecture has a live writer."""
        with self._lock:
            return self._state.lecture_id if self._state is not None else None

    def status(self) -> dict:
        with self._lock:
            st = self._state
            if st is None:
                return {"recording": False, "last_stop": self._last_stop}
            elapsed = time.monotonic() - st.started_monotonic
            try:
                lec = Lecture.load(st.lecture_id)
                title = lec.manifest().get("title") or st.lecture_id
                nseg = len(lec.segments())
                disk = sum(p.stat().st_size for p in lec.work.glob("seg-*.wav"))
            except Exception:
                title, nseg, disk = st.lecture_id, 0, 0
            return {
                "recording": True,
                "lecture_id": st.lecture_id,
                "title": title,
                "device": st.device_name,
                "elapsed_s": round(elapsed, 1),
                "audio_s": round(st.base_ms / 1000 + st.out_time_us / 1e6, 1),
                "started_utc": st.started_utc,
                "segments": nseg,
                "bytes": disk,
                "stalled": st.stalled,
                "level": self.level(),
            }

    def level(self) -> float | None:
        """RMS 0..1 from the tail of the growing segment. Requires the flush fix."""
        with self._lock:
            st = self._state
            if st is None:
                return None
            try:
                lec = Lecture.load(st.lecture_id)
            except Exception:
                return None
        segs = sorted(lec.work.glob("seg-*.wav"))
        if not segs:
            return None
        p = segs[-1]
        try:
            size = p.stat().st_size
            if size <= 44:
                return 0.0
            n = min(config.BYTES_PER_SEC // 5, size - 44)   # last ~200 ms
            n -= n % 2
            if n <= 0:
                return 0.0
            with open(p, "rb") as f:
                f.seek(size - n)
                raw = f.read(n)
            return round(min(1.0, pcm.rms(raw) / 12000.0), 3)
        except OSError:
            return None

    # ------------------------------------------------------------- start
    def start(self, title: str, course: str = "", folder: str = "",
              *, skip_probe: bool = False) -> dict:
        with self._lock:
            if self._state is not None:
                raise RecorderBusy("a recording is already in progress")

            if storage.free_bytes() < config.MIN_FREE_BYTES_START:
                raise RuntimeError(
                    f"Not enough free disk space to start "
                    f"({storage.free_bytes()/1e9:.1f} GB free).")

            device = audio.resolve_device(fresh=True)   # indices shift; never cache this
            if not skip_probe:
                audio.probe(device)          # raises PreflightError with a kind

            _require_capture_helper()        # before anything is created

            lec = Lecture.create(title=title, course=course, folder=folder)
            log_offset = _size(lec.work / "record.stderr.log")
            try:
                argv = self._argv(lec, device)
                with open(lec.work / "record.stderr.log", "ab", buffering=0) as errlog:
                    proc = subprocess.Popen(
                        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                        stderr=errlog, text=True, bufsize=1, **pc.detached_popen_kwargs(),
                    )
            except Exception:
                # No writer ever ran: never leave it "recording" (that would
                # also keep it busy, so it could not be moved, until a restart).
                lec.set_state("failed", recovery_note=(
                    "The recorder could not be started, so nothing was recorded."))
                raise

            # Written durably BEFORE we depend on it: an orphaned ffmpeg that is
            # not attributable cannot be adopted or cleaned up.
            storage.atomic_write_json(lec.work / "recording.json", {
                "pid": proc.pid,
                "boot_sec": storage.boot_seconds(),
                "argv_token": lec.id,
                "started_utc": storage.utcnow(),
                "device_name": device.name,
                "device_index": device.index,
                "log_offset": log_offset,
            }, durable=True)

            self._proc = proc
            self._state = RecordingState(
                lecture_id=lec.id, pid=proc.pid, device_name=device.name,
                device_index=device.index, started_monotonic=time.monotonic(),
                started_utc=storage.utcnow(), log_offset=log_offset,
            )
            self._caffeinate = self._start_caffeinate(proc.pid)
            self._pump = threading.Thread(target=self._pump_progress,
                                          args=(proc,), daemon=True)
            self._pump.start()
            self._guard = threading.Thread(target=self._watch, daemon=True)
            self._guard.start()
            return self.status()

    def _argv(self, lec: Lecture, device: audio.Device,
              start_number: int = 0) -> list[str]:
        # Never build here: a compile can take minutes and the lecture is
        # starting now. The server builds the helper at startup.
        if capture.portable():
            stop_file = lec.work / "STOP"
            try:
                stop_file.unlink()
            except FileNotFoundError:
                pass
            return [
                sys.executable, str(Path(__file__).with_name("capture_py.py")),
                "--device", device.name,
                "--out", str(lec.work / "seg-%06d.wav"),
                "--segment-seconds", str(config.SEGMENT_SECONDS),
                "--start-number", str(start_number),
                "--stop-file", str(stop_file),
            ]
        helper = capture.available()
        if helper is not None:
            return [
                str(helper), "--device", device.name,
                "--out", str(lec.work / "seg-%06d.wav"),
                "--segment-seconds", str(config.SEGMENT_SECONDS),
                "--start-number", str(start_number),
                *(["--pause-on-lid"] if _builtin_mic(device.name) else []),
            ]
        log.warning("capture helper unavailable; falling back to ffmpeg, "
                    "which is known to drop audio")
        return [
            config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "warning",
            "-f", "avfoundation", "-i", f":{device.index}",
            "-ac", str(config.CHANNELS), "-ar", str(config.SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-fflags", "+bitexact",                     # OUTPUT position: 44-byte header
            "-f", "segment",
            "-segment_time", str(config.SEGMENT_SECONDS),
            "-segment_start_number", str(start_number),  # continue numbering after a pause
            "-reset_timestamps", "1",
            "-segment_format_options", "flush_packets=1",   # the real flush fix
            "-progress", "pipe:1", "-nostats",
            str(lec.work / "seg-%06d.wav"),
        ]

    def _start_caffeinate(self, pid: int) -> subprocess.Popen | None:
        """Keep the machine awake while recording (caffeinate on macOS,
        systemd-inhibit on Linux; Windows does it inside the helper)."""
        return pc.keep_awake(pid)

    # -------------------------------------------------------- monitoring
    def _pump_progress(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stdout:                       # type: ignore[union-attr]
                line = line.strip()
                if not line:
                    continue
                k, _, v = line.partition("=")
                with self._lock:
                    if self._state is None:
                        return
                    if k == "out_time_us" and v.strip().isdigit():
                        self._state.out_time_us = int(v)
                        self._state.last_progress = time.monotonic()
                        self._state.stalled = False
                    elif k == "progress":
                        self._state.last_progress = time.monotonic()
                    elif k == "event":
                        self._state.event = v.strip()
        except Exception:
            pass

    def _watch(self) -> None:
        self._supervise(adopted=False)

    def _supervise(self, *, adopted: bool) -> None:
        """One loop for spawned and adopted writers: exit, stall, lid, sleep,
        disk. Any action that signals the writer runs OUTSIDE the lock.
        Bound to the session it was started for: after a pause and a quick
        Continue, the old watcher must not act on the new writer (its stale
        "slept" flag would pause the fresh recording as "system sleep")."""
        with self._lock:
            mine = self._state
        wall0, awake0 = _clocks()
        next_lid = 0.0
        sleep_seen_at: float | None = None
        while True:
            time.sleep(2.0)
            wall, awake = _clocks()
            slept = (wall - wall0) - (awake - awake0) > _SLEEP_GAP_SECONDS
            wall0, awake0 = wall, awake

            with self._lock:
                st, proc = self._state, self._proc
                if st is None or proc is None or st is not mine:
                    return
                if proc.poll() is not None:
                    if st.stop_reason:
                        return                 # stop()/pause() owns this exit
                    reason = self._exit_reason(st, proc)
                    lecture_id = st.lecture_id
                    # Final state on disk BEFORE we let go: until _state is
                    # cleared, current_lecture_id() keeps the lecture "busy",
                    # so no class move can rename it under this write.
                    self._mark_paused_locked(lecture_id, reason)
                    self._finish_locked(reason)
                    break
                if adopted:
                    self._adopted_progress(st)
                if time.monotonic() - st.last_progress > _STALL_SECONDS:
                    st.stalled = True
                builtin = _builtin_mic(st.device_name)
                owned = bool(st.stop_reason)

            if owned:
                continue
            why = None
            if slept and sleep_seen_at is None:
                sleep_seen_at = time.monotonic()
            if sleep_seen_at is not None:
                if time.monotonic() - sleep_seen_at >= _WAKE_GRACE_SECONDS:
                    why = REASON_SLEEP         # still capturing after the grace
            elif builtin and time.monotonic() >= next_lid:
                next_lid = time.monotonic() + _LID_POLL_SECONDS
                if lid_closed():
                    why = REASON_LID
            if why:
                log.warning("pausing recording: %s", why)
                try:
                    self.pause(reason=why)
                except RuntimeError:
                    pass                       # already stopped by someone else
                return
            if storage.free_bytes() < config.MIN_FREE_BYTES_STOP:
                # A pause, not a stop: stop() leaves lecture.json "recording"
                # with nothing queued. Paused offers Continue / Finish.
                log.warning("pausing recording: %s", REASON_DISK)
                try:
                    self.pause(reason=REASON_DISK)
                except RuntimeError:
                    pass
                return

        # The writer exited on its own: the helper pausing for lid/sleep, or a
        # crash. Either way the lecture is now "paused" -- never stuck in
        # "recording" with no writer -- and the user decides: continue or finish.
        log.warning("recording %s ended by itself: %s", lecture_id, reason)

    @staticmethod
    def _self_stopped(st: RecordingState, proc) -> str | None:
        """REASON_LID / REASON_SLEEP if this writer stopped itself for the lid
        or sleep (its log line, its event= line, or exit code 75), else None."""
        try:
            why = self_pause_reason(Lecture.load(st.lecture_id), st.log_offset)
        except Exception:
            why = None
        if why:
            return why
        if st.event == "lid-closed":
            return REASON_LID
        if st.event == "sleep" or getattr(proc, "returncode", None) == SYSTEM_PAUSE_EXIT:
            return REASON_SLEEP
        return None

    @classmethod
    def _exit_reason(cls, st: RecordingState, proc) -> str:
        """Why did the writer exit when we did not ask it to?"""
        return cls._self_stopped(st, proc) or REASON_CRASH

    def _mark_paused_locked(self, lecture_id: str, reason: str) -> None:
        """Persist 'paused' and publish last_stop. Call with the lock held and
        BEFORE _finish_locked: status() must never report "not recording"
        while lecture.json still says "recording" or last_stop is stale (the
        UI would miss the lid toast), and a class move must never see the
        lecture idle before this write lands."""
        try:
            Lecture.load(lecture_id).set_state(
                "paused", pause_reason=reason, paused_utc=storage.utcnow())
        except Exception:
            log.exception("could not mark %s paused", lecture_id)
        self._last_stop = {"lecture_id": lecture_id, "reason": reason,
                           "paused": True, "at": storage.utcnow()}

    # -------------------------------------------------------------- stop
    @staticmethod
    def _stop_file_for(lecture_id: str):
        """The portable helper stops when this file appears (works for a detached
        Windows process, which cannot receive signals). None for the Swift and
        ffmpeg writers, which are stopped by signal exactly as before."""
        if not capture.portable():
            return None
        try:
            return Lecture.load(lecture_id).work / "STOP"
        except Exception:
            return None

    @staticmethod
    def _signal_stop(proc, stop_file=None) -> None:
        """SIGINT then SIGTERM then SIGKILL. ffmpeg routes all three through the
        same clean-shutdown handler (patches the RIFF sizes either way), so
        this ladder is belt-and-braces rather than load-bearing on which
        signal wins."""
        if stop_file is not None:
            try:
                Path(stop_file).write_text("stop")
            except OSError:
                pass
            end = time.monotonic() + 8.0
            while proc.poll() is None and time.monotonic() < end:
                time.sleep(0.1)
        if pc.IS_WIN:
            for fn in (getattr(proc, "terminate", None) or proc.kill, proc.kill):
                if proc.poll() is not None:
                    break
                try:
                    fn()
                except Exception:
                    pass
                end = time.monotonic() + 5.0
                while proc.poll() is None and time.monotonic() < end:
                    time.sleep(0.1)
            return
        for sig, wait in ((signal.SIGINT, 8.0), (signal.SIGTERM, 5.0)):
            if proc.poll() is not None:
                break
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                break
            deadline = time.monotonic() + wait
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Unkillable (e.g. stuck in the audio driver). Don't raise:
                # the caller must still finish, or the recorder stays "active"
                # until a restart. Recovery adopts or cleans it up later.
                log.error("capture process %s did not exit after SIGKILL", proc.pid)

    def stop(self, reason: str = "user") -> dict:
        with self._lock:
            st, proc = self._state, self._proc
            if st is None or proc is None:
                raise RuntimeError("no recording in progress")
            lecture_id = st.lecture_id
            st.stop_reason = reason

        self._signal_stop(proc, self._stop_file_for(lecture_id))

        with self._lock:
            self._finish_locked(reason)
            self._last_stop = {"lecture_id": lecture_id, "reason": reason,
                               "paused": False, "at": storage.utcnow()}
        return {"lecture_id": lecture_id, "stopped": True, "reason": reason}

    # ------------------------------------------------------- pause / resume
    def pause(self, reason: str = "user") -> dict:
        """Stop capturing without finalizing. The lecture stays 'paused' and
        its segments stay exactly where they are; resume() continues the
        segment numbering from here. Unlike stop(), this never touches the
        pipeline -- there is nothing to transcribe yet. `reason` is shown to
        the user ("lid closed", "system sleep", or their own Pause)."""
        with self._lock:
            st, proc = self._state, self._proc
            if st is None or proc is None:
                raise RuntimeError("no recording in progress")
            if st.stop_reason:
                raise RuntimeError("this recording is already stopping")
            lecture_id = st.lecture_id
            st.stop_reason = f"pause:{reason}"

        self._signal_stop(proc, self._stop_file_for(lecture_id))

        with self._lock:
            superseded = st.stop_reason != f"pause:{reason}"   # Stop won the race
            if not superseded:
                # Pause tapped just after waking: the helper may already have
                # stopped itself because the lid closed or the Mac slept (or
                # did so while our signal was on its way). Its reason wins.
                reason = self._self_stopped(st, proc) or reason
                self._mark_paused_locked(lecture_id, reason)
            self._finish_locked("paused")
        return {"lecture_id": lecture_id, "paused": True, "reason": reason}

    def resume(self, lecture_id: str, *, skip_probe: bool = False) -> dict:
        """Continue recording into an existing lecture.

        Two cases:
          * paused mid-recording -- its .work segments are still on disk;
            new segments simply continue the numbering.
          * a finished (possibly archived) lecture -- its WAVs may be long
            gone, so the archive is decoded back into one seed segment first.
            Everything is re-transcribed and re-archived on the next stop(),
            which is why the old archive is never deleted until then.
        """
        with self._lock:
            if self._state is not None:
                if self._state.lecture_id == lecture_id:
                    return self.status()          # already the active one
                raise RecorderBusy("a different recording is already in progress")

            if storage.free_bytes() < config.MIN_FREE_BYTES_START:
                raise RuntimeError(
                    f"Not enough free disk space to resume "
                    f"({storage.free_bytes()/1e9:.1f} GB free).")

            lec = Lecture.load(lecture_id)
            state = lec.manifest().get("state")
            if state not in ("paused", "complete", "failed"):
                raise RuntimeError(
                    f"this lecture cannot be resumed from its current state ({state!r})")

            _require_capture_helper()

            lec.work.mkdir(parents=True, exist_ok=True)
            next_index = self._seed_from_archive_if_needed(lec)
            base_ms = sum(concat.wav_duration_ms(p) for p in lec.segments())

            t0 = time.monotonic()
            device = self._resume_device(lec, state)
            if device is None:
                # A different mic than the one this lecture was recorded on
                # (or no record of it): enumerate afresh and test it first.
                device = audio.resolve_device(fresh=True)
                if not skip_probe:
                    audio.probe(device)

            argv = self._argv(lec, device, start_number=next_index)
            log_offset = _size(lec.work / "record.stderr.log")
            with open(lec.work / "record.stderr.log", "ab", buffering=0) as errlog:
                proc = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=errlog, text=True, bufsize=1, **pc.detached_popen_kwargs(),
                )

            storage.atomic_write_json(lec.work / "recording.json", {
                "pid": proc.pid,
                "boot_sec": storage.boot_seconds(),
                "argv_token": lec.id,
                "started_utc": storage.utcnow(),
                "device_name": device.name,
                "device_index": device.index,
                "log_offset": log_offset,
            }, durable=True)

            lec.set_state("recording", pause_reason=None, paused_utc=None)
            log.info("resumed %s in %.2f s", lec.id, time.monotonic() - t0)
            self._proc = proc
            self._state = RecordingState(
                lecture_id=lec.id, pid=proc.pid, device_name=device.name,
                device_index=device.index, started_monotonic=time.monotonic(),
                started_utc=storage.utcnow(), base_ms=base_ms,
                log_offset=log_offset,
            )
            self._caffeinate = self._start_caffeinate(proc.pid)
            self._pump = threading.Thread(target=self._pump_progress,
                                          args=(proc,), daemon=True)
            self._pump.start()
            self._guard = threading.Thread(target=self._watch, daemon=True)
            self._guard.start()
            return self.status()

    @staticmethod
    def _resume_device(lec: Lecture, state: str | None):
        """Fast path for Continue after a pause: the mic this lecture was just
        recording from, if it is still connected. It worked seconds or minutes
        ago, so the ~2-3 s test recording is skipped -- that test is what made
        resuming slow. The helper still exits at once if the mic cannot open,
        and a mic that stops sending audio is flagged as stalled."""
        if state != "paused":
            return None
        name = (_read_recording_json(lec) or {}).get("device_name")
        if not name:
            return None
        try:
            devices = audio.list_audio_devices(use_cache=True)
        except audio.PreflightError:
            return None
        return next((d for d in devices if d.name == name), None)

    @staticmethod
    def _seed_from_archive_if_needed(lec: Lecture) -> int:
        """If there are no live segments, decode the existing archive into one
        seed segment first, so appending new audio is just 'more segments'
        rather than a special case downstream in concat/transcribe/archive."""
        existing = lec.segments()
        if existing:
            return len(existing)
        if not lec.archive_path.is_file():
            return 0                      # nothing recorded yet either way
        seed = lec.work / "seg-000000.wav"
        # Decode to a side file and rename only on success: a truncated seed
        # left by a failed decode would be re-archived over the full audio.
        part = seed.with_name(seed.name + ".part")
        try:
            subprocess.run(
                [config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-y", "-i", str(lec.archive_path), "-ac", str(config.CHANNELS),
                 "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le",
                 "-fflags", "+bitexact", "-f", "wav", str(part)],
                check=True, timeout=3600)
            os.replace(part, seed)
        finally:
            part.unlink(missing_ok=True)
        return 1

    def _finish_locked(self, reason: str) -> None:
        if self._caffeinate is not None:
            try:
                self._caffeinate.terminate()
            except Exception:
                pass
            self._caffeinate = None
        self._state = None
        self._proc = None

    # ------------------------------------------------------------- adopt
    def adopt(self) -> str | None:
        """Re-attach to a recording that outlived a previous backend."""
        with self._lock:
            if self._state is not None:
                return self._state.lecture_id
        for lec in Lecture.all():
            info = _read_recording_json(lec)
            if not info or not _writer_alive(info):
                continue
            with self._lock:
                if self._state is not None:
                    return self._state.lecture_id
                self._proc = _AdoptedProc(info["pid"])
                self._state = RecordingState(
                    lecture_id=lec.id, pid=info["pid"],
                    device_name=info.get("device_name", "?"),
                    device_index=int(info.get("device_index", -1)),
                    started_monotonic=time.monotonic(),
                    started_utc=info.get("started_utc", storage.utcnow()),
                    log_offset=int(info.get("log_offset") or 0),
                )
                # No -progress pipe on an adopted process; fall back to file growth.
                self._state.last_progress = time.monotonic()
                self._guard = threading.Thread(target=self._watch_adopted, daemon=True)
                self._guard.start()
                return lec.id
        return None

    def _watch_adopted(self) -> None:
        self._supervise(adopted=True)

    @staticmethod
    def _adopted_progress(st: RecordingState) -> None:
        """No -progress pipe on a process we did not spawn: progress is the
        segment files growing. (It used to count as progress unconditionally,
        which hid a stalled adopted writer.) Called with the lock held."""
        try:
            lec = Lecture.load(st.lecture_id)
            total = sum(p.stat().st_size for p in lec.work.glob("seg-*.wav"))
        except Exception:
            return
        st.out_time_us = int(total / config.BYTES_PER_SEC * 1e6)
        if total != st.adopted_bytes:
            st.adopted_bytes = total
            st.last_progress = time.monotonic()
            st.stalled = False


# ------------------------------------------------------------- helpers
def self_pause_reason(lec: Lecture, log_offset: int | None = None) -> str | None:
    """REASON_LID / REASON_SLEEP if the lecture's most recent writer stopped
    itself cleanly for the lid or sleep, else None. Reads only what that
    writer logged (from log_offset): the log spans every session, and an
    earlier session's "stopped (lid closed)" must not explain a later crash.
    Also used by startup recovery, for a writer that paused itself while no
    backend was running to hear it."""
    if log_offset is None:
        info = _read_recording_json(lec) or {}
        log_offset = int(info.get("log_offset") or 0)
    try:
        with open(lec.work / "record.stderr.log", "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(log_offset, f.tell() - 4096))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    if "stopped (lid closed)" in tail:
        return REASON_LID
    if "stopped (system sleep)" in tail:
        return REASON_SLEEP
    return None


def _size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _read_recording_json(lec: Lecture) -> dict | None:
    p = lec.work / "recording.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _pid_alive(pid: int) -> bool:
    return pc.pid_alive(pid)


def _writer_alive(info: dict) -> bool:
    """PID alone is not enough: PIDs are reused across a reboot."""
    # Exact match on macOS (sysctl); a few seconds of slack elsewhere because
    # some platforms derive boot time from a clock that jitters by a second.
    if abs(int(info.get("boot_sec") or 0) - storage.boot_seconds()) > 5:
        return False                     # machine rebooted -> definitively dead
    pid = int(info.get("pid") or 0)
    if pid <= 0 or not _pid_alive(pid):
        return False
    token = info.get("argv_token") or ""
    return token in pc.command_line(pid)


class _AdoptedProc:
    """Minimal Popen stand-in for a process we did not spawn."""

    def __init__(self, pid: int):
        self.pid = pid
        self.stdout = None

    def poll(self):
        return None if _pid_alive(self.pid) else 0

    def send_signal(self, sig):
        if pc.IS_WIN:
            pc.terminate_pid(self.pid)
        else:
            os.kill(self.pid, sig)

    def terminate(self):
        pc.terminate_pid(self.pid)

    def kill(self):
        pc.terminate_pid(self.pid, force=True)

    def wait(self, timeout=None):
        end = time.monotonic() + (timeout or 10)
        while time.monotonic() < end and _pid_alive(self.pid):
            time.sleep(0.1)
        return 0


RECORDER = Recorder()
