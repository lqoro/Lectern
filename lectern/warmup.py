"""Prepare a Core ML speech engine once, before a lecture needs it.

Core ML compiles a model for this Mac's Neural Engine the first time it is
loaded and caches the result. Measured on an M-series Mac with a 10 s clip:
28.9 s (whisper.cpp with its Core ML encoder) and 10.7 s (Parakeet) the first
time, about 1 s and 0.2 s afterwards; slower Macs pay more. Without this the
first lecture -- or, with live transcription, its first segment -- pays it.

So the engine is run once, in the background, on a short silent clip (loading
the model is what compiles it; the audio does not matter):
  * at app start (after START_DELAY_S), after a model download, and when the
    user picks a model;
  * only for Core ML models: a Parakeet folder, or a ggml file with its
    "-encoder.mlmodelc" beside it;
  * once per key(): app version, this Mac (chip and macOS version), the
    model files and the engine binary. A new app version, macOS update, model
    or engine build warms again. Done keys are kept in RECORD_FILE.

Real work always wins:
  * it does not start while a lecture is recording, while any audio stage
    holds or waits for pipeline.STAGE_LOCK, or while the live draft has work;
  * once running, a watcher kills it within WATCH_S if any of those appear;
  * pipeline._transcribe (holding STAGE_LOCK) and live._transcribe_one (after
    marking its engine busy) call preempt() before they start their engine,
    which kills it and waits for it to exit. Both mark themselves busy BEFORE
    preempt() and the warm-up checks for them under the same lock it starts
    under, so there is no window in which both engines start;
  * a killed warm-up is not a failure: it is retried once the engine is free.
    A failure only logs, and that key is not retried until the app restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
import shutil
import threading
import time
import wave
from pathlib import Path

from . import config, hardware, models, storage, transcribe

log = logging.getLogger(__name__)

CLIP_S = 2                  # seconds of silence; loading the model is what compiles it
START_DELAY_S = 10.0        # at app start, let startup recovery claim the engine first
BUSY_RETRY_S = 15.0         # while real work has the engine, look again this often
WATCH_S = 1.0               # while warming, check this often whether real work arrived
PREEMPT_WAIT_S = 5.0        # preempt() waits at most this for the killed engine to exit
THREAD_SHARE = 2            # like the live draft: half the usual whisper threads
MAX_RECORDS = 16            # warm-up records kept (old app versions fall off)

RECORD_FILE = config.SUPPORT_DIR / "warmup.json"
WORK_DIR = config.SUPPORT_DIR / "warmup"

# whisper.cpp looks for "<model minus -qX_Y>-encoder.mlmodelc" beside the model.
_QUANT_SUFFIX = re.compile(r"-q\d_\d$")


# ------------------------------------------------------------ which model
def coreml_encoder(model: Path) -> Path | None:
    if model.suffix != ".bin":
        return None
    enc = model.with_name(_QUANT_SUFFIX.sub("", model.stem) + "-encoder.mlmodelc")
    return enc if enc.is_dir() else None


def uses_coreml(model: Path) -> bool:
    """A Parakeet folder (fluidaudiocli), or whisper.cpp with a Core ML encoder."""
    return model.is_dir() or coreml_encoder(model) is not None


def engine_for(model: Path) -> str:
    exe = config.FLUIDAUDIO_CLI if model.is_dir() else config.WHISPER_CLI
    return shutil.which(exe) or exe


def _machine_id() -> str:
    """The Core ML cache belongs to this Mac and this macOS (an OS update
    recompiles), so a support folder migrated to another Mac warms again."""
    return "|".join((platform.machine(), platform.mac_ver()[0],
                     hardware.device().get("chip") or ""))


def _stat_id(p: Path | str) -> str:
    p = Path(p)
    try:
        st = p.stat()
    except OSError:
        return f"{p}:missing"
    return f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}"


def key(model: Path) -> str:
    enc = coreml_encoder(model)
    parts = [config.VERSION, _machine_id(), _stat_id(model),
             _stat_id(enc) if enc else "", _stat_id(engine_for(model))]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


# --------------------------------------------------------------- records
def _records() -> dict:
    try:
        data = json.loads(RECORD_FILE.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable %s: %s", RECORD_FILE.name, e)
        return {}
    return data if isinstance(data, dict) else {}


def is_warmed(model: Path) -> bool:
    return key(model) in _records()


def _record(k: str, model: Path, took_s: float) -> None:
    recs = _records()
    recs[k] = {"model": model.name, "engine": Path(engine_for(model)).name,
               "version": config.VERSION, "utc": storage.utcnow(),
               "took_s": round(took_s, 1)}
    newest = sorted(recs.items(), key=lambda kv: str(kv[1].get("utc", "")))[-MAX_RECORDS:]
    storage.atomic_write_json(RECORD_FILE, dict(newest), indent=2)


def _write_clip(path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(config.CHANNELS)
        w.setsampwidth(config.SAMPLE_WIDTH)
        w.setframerate(config.SAMPLE_RATE)
        w.writeframes(b"\0" * (config.BYTES_PER_SEC * CLIP_S))


# ------------------------------------------------------------- real work
def _engine_busy() -> bool:
    """The callers of preempt(): a pipeline audio stage, or the live draft."""
    from .live import LIVE
    from .pipeline import STAGE_LOCK
    return STAGE_LOCK.busy() or LIVE.busy_any()


def _recording() -> bool:
    from .recorder import RECORDER
    return RECORDER.current_lecture_id() is not None


def _kill(proc) -> None:
    try:
        proc.kill()
    except Exception:
        log.warning("could not stop the engine warm-up", exc_info=True)


class Warmer:
    """One background thread at a time. Its lock guards only memory, plus the
    "is real work running" check made just before an engine starts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._dirty = False                  # a request arrived since the worker last looked
        self._wake = threading.Event()
        self._wanted: list[Path] = []        # downloaded models, besides the active one
        self._failed: set[str] = set()       # not retried until the app restarts
        self._running = False
        self._stop = False
        self._proc = None
        self._free = threading.Event()
        self._free.set()
        self._state: dict = {"state": "idle", "model": None, "error": ""}

    # ------------------------------------------------------------ public
    def status(self) -> dict:
        """state: idle (nothing to do) | waiting (for real work to finish) |
        warming | ready (the active model is prepared) | failed."""
        with self._lock:
            return dict(self._state)

    def idle(self) -> bool:
        """No worker thread (tests and diagnostics)."""
        with self._lock:
            return self._thread is None

    def engine_running(self) -> bool:
        with self._lock:
            return self._running

    def request(self, model: Path | None = None, delay: float = 0.0) -> None:
        """Warm `model` (default: the active model) when the engine is free.
        Never blocks."""
        if not config.WARMUP:
            return
        with self._lock:
            if model is not None and model not in self._wanted:
                self._wanted.append(model)
            self._dirty = True
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, args=(delay,),
                                                name="engine-warmup", daemon=True)
                self._thread.start()
        self._wake.set()

    def preempt(self, timeout: float = PREEMPT_WAIT_S) -> None:
        """Real work wants the engine: kill a running warm-up and wait (at most
        `timeout`) for it to exit. Returns at once when none is running."""
        if not self._interrupt():
            return
        if not self._free.wait(timeout):
            log.warning("the engine warm-up did not exit within %.0f s", timeout)

    # ------------------------------------------------------------ worker
    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def _interrupt(self) -> bool:
        with self._lock:
            if not self._running:
                return False
            self._stop = True
            proc = self._proc
        if proc is not None:
            _kill(proc)
        return True

    def _run(self, delay: float) -> None:
        if delay:
            time.sleep(delay)
        while True:
            with self._lock:
                self._dirty = False
            try:
                target = self._next()
            except Exception:
                log.exception("engine warm-up: could not choose a model")
                target = None
            if target is None:
                with self._lock:
                    if self._dirty:
                        continue
                    self._thread = None
                    return
            model, k = target
            if _recording() or not self._begin():
                self._set(state="waiting", model=model.name)
                self._wake.wait(BUSY_RETRY_S)
                self._wake.clear()
                continue
            self._warm(model, k)

    def _next(self) -> tuple[Path, str] | None:
        """The first model that needs warming; updates the idle/ready state."""
        with self._lock:
            wanted = list(self._wanted)
            failed = set(self._failed)
        active = models.active_path()
        done = set(_records())
        for p in [*wanted, active]:
            if p is None or not p.exists() or not uses_coreml(p):
                continue
            k = key(p)
            if k in done or k in failed:
                continue
            return p, k
        with self._lock:
            self._wanted = [p for p in self._wanted if p not in wanted]
            if self._state["state"] in ("waiting", "warming", "idle", "ready"):
                ready = active is not None and active.exists() and uses_coreml(active)
                self._state.update(state="ready" if ready else "idle",
                                   model=active.name if ready else None)
        return None

    def _begin(self) -> bool:
        with self._lock:
            if _engine_busy():
                return False
            self._running, self._stop, self._proc = True, False, None
            self._free.clear()
            return True

    def _keep(self, proc) -> None:
        with self._lock:
            self._proc = proc
            stop = self._stop
        if stop:
            _kill(proc)

    def _watch(self, done: threading.Event) -> None:
        while not done.wait(WATCH_S):
            try:
                busy = _recording() or _engine_busy()
            except Exception:
                log.exception("engine warm-up: could not check for real work")
                busy = False
            if busy:
                log.info("stopping the engine warm-up: real work needs the engine")
                self._interrupt()
                return

    def _warm(self, model: Path, k: str) -> None:
        self._set(state="warming", model=model.name, error="")
        log.info("preparing the speech engine for %s (Core ML compiles it once)", model.name)
        done = threading.Event()
        threading.Thread(target=self._watch, args=(done,), daemon=True).start()
        t0 = time.monotonic()
        error = None
        try:
            WORK_DIR.mkdir(parents=True, exist_ok=True)
            clip = WORK_DIR / "clip.wav"
            _write_clip(clip)
            transcribe.run_whisper(clip, WORK_DIR / "out", model=model,
                                   threads=max(1, hardware.whisper_threads() // THREAD_SHARE),
                                   proc_cb=self._keep)
        except Exception as e:
            error = e
        finally:
            done.set()
            with self._lock:
                stopped = self._stop
                self._running, self._proc = False, None
            self._free.set()
            shutil.rmtree(WORK_DIR, ignore_errors=True)
        took = time.monotonic() - t0
        if error is not None and stopped:
            log.info("engine warm-up for %s stopped after %.1f s; it will be retried",
                     model.name, took)
            self._set(state="waiting")
            return
        if error is not None:
            with self._lock:
                self._failed.add(k)
            log.warning("engine warm-up for %s failed (the first lecture will be slower): %s",
                        model.name, error)
            self._set(state="failed", error=str(error).splitlines()[0][:300])
            return
        try:
            _record(k, model, took)
        except OSError:
            log.exception("could not save the engine warm-up record")
        log.info("speech engine ready for %s (%.1f s)", model.name, took)
        self._set(state="ready")


WARMER = Warmer()
