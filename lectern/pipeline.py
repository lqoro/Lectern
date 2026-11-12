"""Post-recording processing: extract -> concat -> transcribe -> archive -> notes.

Two process-wide FIFO locks:
  * STAGE_LOCK serializes the audio stages (extract, concat, transcribe,
    archive), so at most one whisper runs at a time.
  * NOTES_LOCK serializes the notes stage (the Ollama LLM, up to ~30 min).

Archive runs before notes so the player switches from the huge master.wav to
the compressed archive as soon as the audio is safe, and a lecture's notes no
longer hold up the next lecture's transcription (or a notes rewrite on another
lecture).

The RAM invariant on small machines: whisper peaks around 1.05 GB on a
50-minute lecture and qwen2.5:3b needs ~2.3 GB. Below SHARED_RAM_MIN_BYTES
(hardware.effective_ram_bytes, i.e. in 8 GB-Mac terms; unknown RAM counts as
small) they must NEVER be resident together, so the notes stage also holds
STAGE_LOCK while the LLM runs -- the old behaviour, for both the pipeline's
notes and "Rewrite notes". At or above it, notes hold only NOTES_LOCK and may
run beside another lecture's transcription. Lock order is always NOTES_LOCK
then STAGE_LOCK; the audio path takes only STAGE_LOCK and releases it before
it queues for notes, so the two cannot deadlock.

Recovery (recovery.py) relies on "noting" being set only AFTER the archive
stage has returned, i.e. once the archive is installed and verified.

Where notes may share the machine (notes_share_ram), the notes are COMPUTED
beside the archive: a thread starts as soon as transcript.json is written and
holds only the notes locks. Nothing it produces is persisted, and the lecture
is not marked "noting", until the archive stage has returned; then the
pipeline waits for it and saves. A crash mid-archive therefore leaves exactly
what the serial order left, and recovery is unchanged. An archive failure
cancels the notes run between parts and fails the lecture as before; a notes
error is raised after the archive, as before. Smaller machines keep the
serial order.

Per-stage wall seconds go to the manifest ("timing", plus per-part map and
reduce seconds under "notes") and one summary line to the log.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import (archive, concat, config, hardware, importer, live, models, notes, premap,
               storage, transcribe, warmup)
from .storage import Lecture

class FifoLock:
    """A mutex that hands out turns in arrival order.

    threading.Lock wakes an arbitrary waiter, so after back-to-back classes the
    third lecture could be processed before the first. Students open the oldest
    one first.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._next = 0          # next ticket to hand out
        self._serving = 0       # ticket currently allowed to run

    def __enter__(self) -> "FifoLock":
        with self._cv:
            ticket = self._next
            self._next += 1
            while ticket != self._serving:
                self._cv.wait()
        return self

    def __exit__(self, *exc) -> None:
        with self._cv:
            self._serving += 1
            self._cv.notify_all()

    def busy(self) -> bool:
        """Held, or waited for. Never blocks (live.py polls it)."""
        with self._cv:
            return self._next != self._serving


# Audio stages (at most one transcription at a time).
STAGE_LOCK = FifoLock()
# The notes stage, on its own so a long LLM run never blocks transcription
# (on machines with room for both; see SHARED_RAM_MIN_BYTES).
NOTES_LOCK = FifoLock()
# Below this much RAM, in Mac terms, notes also take STAGE_LOCK (module doc).
SHARED_RAM_MIN_BYTES = 12 * 1024**3
log = logging.getLogger(__name__)


def notes_share_ram() -> bool:
    """May the notes model run while whisper does? False when RAM is unknown."""
    try:
        eff = hardware.effective_ram_bytes()
    except Exception:
        log.exception("could not read total RAM; keeping notes and whisper apart")
        return False
    return bool(eff) and eff >= SHARED_RAM_MIN_BYTES


@contextmanager
def notes_slot():
    """The locks the notes stage holds: NOTES_LOCK, then, on a small machine,
    STAGE_LOCK as well. Never acquire these in the other order."""
    with NOTES_LOCK:
        if notes_share_ram():
            yield
        else:
            with STAGE_LOCK:
                yield


class _NotesRun:
    """Notes computed in the background while the archive runs. Nothing is
    persisted from here: the result waits in memory until the pipeline saves
    it after the archive (module doc)."""

    def __init__(self) -> None:
        self.go = threading.Event()        # the archive is done: show progress
        self.cancel = threading.Event()    # the archive failed: stop between parts
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: BaseException | None = None
        self.last: tuple[int, int] | None = None     # latest (part, of) progress
        self.started = time.monotonic()
        self.finished: float | None = None
        self._lock = threading.Lock()
        self._consumed = False       # the pipeline took the result (or its error)
        self._abandoned = False      # the pipeline will never take it
        self._reported = False

    def wait(self) -> dict | None:
        self.done.wait()
        with self._lock:
            self._consumed = True
        if self.error is not None:
            raise self.error
        return self.result

    def abandon(self, lecture_id: str) -> None:
        """The run ended without taking the result (any failure before the
        save): stop the notes between parts, and log an error of theirs that
        nothing else will report."""
        self.cancel.set()
        with self._lock:
            if self._consumed:
                return
            self._abandoned = True
        self.report_lost(lecture_id)

    def report_lost(self, lecture_id: str) -> None:
        with self._lock:
            e = self.error
            if not (self._abandoned and e is not None and not self._reported):
                return
            self._reported = True
        log.error("notes of %s failed while the run failed for another reason",
                  lecture_id, exc_info=(type(e), e, e.__traceback__))


@dataclass
class Job:
    lecture_id: str
    stage: str = "queued"
    percent: int = 0
    message: str = ""
    error: str = ""
    done: bool = False
    # True while the running engine gives no percentage (Parakeet): the page
    # shows a moving "working" bar instead of a number that never moves.
    indeterminate: bool = False
    started: float = field(default_factory=time.time)
    # Run-private context, never shown (as_dict lists what the page sees).
    premapped: dict = field(default_factory=dict, repr=False)
    notes_run: _NotesRun | None = field(default=None, repr=False)
    timing: dict = field(default_factory=dict, repr=False)
    notes_stats: dict = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict:
        return {"lecture_id": self.lecture_id, "stage": self.stage,
                "percent": self.percent, "message": self.message,
                "indeterminate": self.indeterminate,
                "error": self.error, "done": self.done,
                "elapsed_s": round(time.time() - self.started, 1)}


class Pipeline:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def job(self, lecture_id: str) -> dict | None:
        with self._lock:
            j = self._jobs.get(lecture_id)
            return j.as_dict() if j else None

    def jobs(self) -> list[dict]:
        with self._lock:
            return [j.as_dict() for j in self._jobs.values() if not j.done]

    def is_running(self, lecture_id: str) -> bool:
        with self._lock:
            j = self._jobs.get(lecture_id)
            return bool(j and not j.done)

    def submit(self, lecture_id: str) -> dict:
        with self._lock:
            j = self._jobs.get(lecture_id)
            if j and not j.done:
                return j.as_dict()
            job = Job(lecture_id)
            self._jobs[lecture_id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job.as_dict()

    def submit_import(self, lecture_id: str) -> dict:
        """Extract the audio of an uploaded file, then the normal stages."""
        with self._lock:
            j = self._jobs.get(lecture_id)
            if j and not j.done:
                return j.as_dict()
            job = Job(lecture_id)
            self._jobs[lecture_id] = job
        threading.Thread(target=self._run, args=(job,), kwargs={"extract": True},
                         daemon=True).start()
        return job.as_dict()

    def submit_notes(self, lecture_id: str) -> dict:
        """Regenerate notes only, without re-transcribing."""
        with self._lock:
            j = self._jobs.get(lecture_id)
            if j and not j.done:
                return j.as_dict()
            job = Job(lecture_id)
            self._jobs[lecture_id] = job
        threading.Thread(target=self._run_notes, args=(job,), daemon=True).start()
        return job.as_dict()

    def _run_notes(self, job: Job) -> None:
        try:
            lec = Lecture.load(job.lecture_id)
            if not lec.transcript_path.is_file():
                raise RuntimeError("there is no transcript yet")
            prior = lec.manifest().get("state") or "complete"
            self._set(job, "waiting", 0, "waiting for a free processing slot")
            with notes_slot():
                try:
                    self._notes(job, lec)
                finally:
                    # _notes marks the lecture "noting"; a regenerate must put
                    # it back, or the UI (and startup recovery) treat it as
                    # mid-processing forever. Re-load: it may have been moved.
                    lec = Lecture.load(job.lecture_id)
                    if lec.manifest().get("state") == "noting":
                        lec.set_state(prior if prior != "noting" else "complete")
            self._set(job, "complete", 100, "notes updated")
        except Exception as e:
            with self._lock:
                job.error = str(e)
            self._set(job, "failed", job.percent, str(e)[:300])
        finally:
            with self._lock:
                job.done = True

    # ------------------------------------------------------------- stages
    def _set(self, job: Job, stage: str, percent: int, message: str = "",
             indeterminate: bool = False) -> None:
        with self._lock:
            job.stage, job.percent, job.message = stage, percent, message
            job.indeterminate = indeterminate

    def _run(self, job: Job, extract: bool = False) -> None:
        try:
            lec = Lecture.load(job.lecture_id)
        except Exception as e:
            with self._lock:
                job.error, job.done = f"cannot load lecture: {e}", True
            return

        try:
            if not extract:
                # Let the live draft take in the last segment (bounded; it
                # never takes STAGE_LOCK, so it must run before we queue).
                self._set(job, "waiting", 0, "finishing the live transcript")
                live.LIVE.finish(job.lecture_id)
                # Pre-mapping stops now (never waits); its cache is read before
                # the transcription stage discards it with the draft.
                premap.PREMAP.stop(job.lecture_id)
                job.premapped = premap.load(lec)
            self._set(job, "waiting", 0, "waiting for a free processing slot")
            with STAGE_LOCK:
                if extract:
                    self._extract(job, lec)
                self._timed(job, "concat", self._concat, lec)
                self._timed(job, "transcribe", self._transcribe, lec)
                if notes_share_ram():
                    job.notes_run = self._try_start_notes(job, lec)
                self._timed(job, "archive", self._archive, lec)
            # The audio is archived and verified from here on. "noting" is set
            # only now, which is what tells recovery that only notes are left.
            archived = time.monotonic()
            lec.set_state("noting")
            self._set(job, "noting", 70, "waiting to write study notes")
            if job.notes_run is None:
                # STAGE_LOCK was released above; notes_slot may take it again
                # (small machines), always after NOTES_LOCK.
                start = time.monotonic()
                with notes_slot():
                    self._notes(job, lec)
                job.timing["notes_s"] = round(time.monotonic() - start, 1)
            else:
                self._notes(job, lec)       # waits for the run, then saves
                job.timing["notes_s"] = round(time.monotonic() - job.notes_run.started, 1)
            job.timing["notes_after_archive_s"] = round(time.monotonic() - archived, 1)
            lec.set_state("complete", failure=None)   # a re-run clears an old failure
            self._set(job, "complete", 100, "done")
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            with self._lock:
                job.error = f"{e}"
            try:
                lec.set_state("failed", failure=str(e)[:500])
            except Exception:
                log.exception("could not mark %s failed", job.lecture_id)
            self._set(job, "failed", job.percent, str(e)[:300])
            try:
                with (config.LOG_DIR / "pipeline.log").open("a") as lf:
                    lf.write(f"\n[{storage.utcnow()}] {job.lecture_id}\n{tb}\n")
            except OSError:
                log.exception("could not write pipeline.log")
        finally:
            # Every way out that did not take the notes result (any failure
            # after the run started, including after the archive) stops it.
            run = job.notes_run
            if run is not None:
                run.abandon(job.lecture_id)
            self._record_timing(job, lec)
            job.premapped = {}
            if run is not None:
                run.result = None
            with self._lock:
                job.done = True

    def _try_start_notes(self, job: Job, lec: Lecture) -> _NotesRun | None:
        """The notes beside the archive, or None (serial order) if their thread
        cannot start. Never fails the lecture."""
        try:
            return self._start_notes(job, lec)
        except Exception:
            log.exception("could not start the notes of %s beside the archive; "
                          "writing them after it instead", job.lecture_id)
            return None

    def _timed(self, job: Job, name: str, stage, lec: Lecture) -> None:
        t0 = time.monotonic()
        try:
            stage(job, lec)
        finally:
            job.timing[f"{name}_s"] = round(time.monotonic() - t0, 2)

    def _record_timing(self, job: Job, lec: Lecture) -> None:
        """The run's wall seconds into the manifest, and one log line. Ids and
        numbers only, never lecture text. Never raises."""
        timing = {**job.timing, "overlap": job.notes_run is not None,
                  "total_s": round(time.time() - job.started, 1)}
        try:
            lec.update(timing=timing)
        except Exception:
            log.warning("could not record the timing of %s", job.lecture_id, exc_info=True)
        st = job.notes_stats
        maps = st.get("map_s") or []
        pm = st.get("premap") or {}

        def sec(key: str) -> str:
            v = timing.get(key)
            return "-" if v is None else f"{v:.1f}s"
        log.info("timing %s: concat %s, transcribe %s, archive %s, notes %s "
                 "(%s after the archive, %s); notes: %d parts by the model "
                 "(mean %.1fs), %d reused (%d exact, %d fuzzy), reduce %s",
                 job.lecture_id, sec("concat_s"), sec("transcribe_s"),
                 sec("archive_s"), sec("notes_s"), sec("notes_after_archive_s"),
                 "beside the archive" if timing["overlap"] else "serial",
                 len(maps), (sum(maps) / len(maps)) if maps else 0.0,
                 pm.get("exact", 0) + pm.get("fuzzy", 0), pm.get("exact", 0),
                 pm.get("fuzzy", 0),
                 "-" if st.get("reduce_s") is None else f"{st['reduce_s']:.1f}s")

    def _start_notes(self, job: Job, lec: Lecture) -> _NotesRun:
        """Compute this lecture's notes in the background (module doc). The
        thread holds only notes_slot(), never the STAGE_LOCK we hold."""
        run = _NotesRun()

        def work() -> None:
            try:
                with notes_slot():
                    run.result = self._compute_notes(job, lec, run)
            except notes.Cancelled:
                log.info("notes of %s were not finished: the archive failed",
                         job.lecture_id)
            except BaseException as e:
                run.error = e
            finally:
                run.finished = time.monotonic()
                run.done.set()
                run.report_lost(job.lecture_id)

        threading.Thread(target=work, name="notes-beside-archive", daemon=True).start()
        return run

    def _extract(self, job: Job, lec: Lecture) -> None:
        """Imported file -> the seed segment the recorder would have written."""
        self._set(job, "importing", 1, "reading the file")
        info = importer.probe(importer.source_path(lec))

        def on_pct(p: int) -> None:
            self._set(job, "importing", 1 + p * 8 // 100, f"extracting audio… {p}%")

        importer.extract(lec, duration_s=info["duration_s"], progress_cb=on_pct)

    def _concat(self, job: Job, lec: Lecture) -> None:
        self._set(job, "concatenating", 2, "assembling audio")
        lec.set_state("concatenating")
        segs = lec.segments()
        if not segs:
            # Already archived: the WAVs are long gone, so decode the AAC back
            # to 16 kHz mono so a re-transcribe is still possible.
            if lec.archive_path.is_file():
                self._set(job, "concatenating", 4, "decoding archived audio")
                lec.work.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
                     "-y", "-i", str(lec.archive_path), "-ac", str(config.CHANNELS),
                     "-ar", str(config.SAMPLE_RATE), "-c:a", "pcm_s16le",
                     "-fflags", "+bitexact", str(lec.master_wav)],
                    check=True, timeout=3600)
                man = {"duration_ms": concat.wav_duration_ms(lec.master_wav),
                       "total_frames": 0, "segments": [], "from_archive": True}
                storage.atomic_write_json(lec.work / "concat.manifest.json", man)
                audio = dict(lec.manifest().get("audio") or {})
                audio["duration_ms"] = man["duration_ms"]
                lec.update(audio=audio)
                self._set(job, "concatenating", 8, "decoded from the archive")
                return
            raise RuntimeError("no audio segments were recorded")
        man = concat.concat_segments(segs, lec.master_wav)
        if man["total_frames"] < config.SAMPLE_RATE * 1:
            raise RuntimeError("recording was too short to process")
        m = lec.manifest()
        audio = dict(m.get("audio") or {})
        audio.update({
            "duration_ms": man["duration_ms"],
            "pcm_frames": man["total_frames"],
            "segments_used": len(segs),
            "sample_rate": config.SAMPLE_RATE,
            "channels": config.CHANNELS,
        })
        lec.update(audio=audio)
        self._set(job, "concatenating", 8,
                  f"{man['duration_ms']//60000}m{man['duration_ms']//1000%60}s of audio")

    def _transcribe(self, job: Job, lec: Lecture) -> None:
        lec.set_state("transcribing")
        dur = lec.duration_ms()
        self._set(job, "transcribing", 10, "transcribing")

        def on_pct(p: int | None) -> None:
            if p is None:          # the engine reports no progress (Parakeet): say so, no number
                self._set(job, "transcribing", 10, "transcribing…", indeterminate=True)
                return
            self._set(job, "transcribing", 10 + int(p * 0.55),
                      f"transcribing… {p}%")

        live.LIVE.wait_idle()      # never two engines at once (bounded)
        warmup.WARMER.preempt()    # nor beside the engine warm-up (we hold STAGE_LOCK)
        t0 = time.time()
        raw, model_id = transcribe.run_with_fallback(lec.master_wav, lec.work / "whisper.raw",
                                                     progress_cb=on_pct)
        doc = transcribe.normalize(raw, dur)
        if not doc["words"]["w"]:
            raise RuntimeError("no speech was found in the recording")

        storage.atomic_write_json(lec.transcript_path, doc)
        storage.atomic_write_text(lec.transcript_txt_path,
                                  transcribe.to_plain_text(doc))
        live.discard(lec)          # the real transcript replaces the draft
        lec.update(transcript={
            "file": "transcript.json",
            "words": len(doc["words"]["w"]),
            "sentences": len(doc["sents"]),
            "language": doc["language"],
            "model": model_id,
            "compute_s": round(time.time() - t0, 1),
        })
        self._set(job, "transcribing", 65,
                  f"{len(doc['words']['w'])} words transcribed")

    def _notes(self, job: Job, lec: Lecture) -> None:
        """Study notes, after the archive. Runs under notes_slot(): beside
        another lecture's whisper only on a machine with room for both.
        Notes never fail a recording: the extractive fallback always produces
        something, and notes are regenerable from transcript.json at any time.
        When they were computed beside the archive (job.notes_run), this only
        waits for that result and saves it.
        """
        if lec.manifest().get("state") != "noting":
            lec.set_state("noting")
        self._set(job, "noting", 72, "writing study notes")
        run = job.notes_run
        if run is not None:
            run.go.set()
            if run.last is not None:
                self._notes_progress(job, *run.last)
            result = run.wait()
        else:
            result = self._compute_notes(job, lec)
        if result is not None:
            self._save_notes(job, lec, result)

    def _notes_progress(self, job: Job, i: int, n: int) -> None:
        # Progress, never an ETA: generation speed on a machine that may be
        # paging has not been measured and a wrong ETA is worse than none.
        self._set(job, "noting", 72 + int(26 * i / max(1, n)),
                  f"writing study notes… part {i+1} of {n}")

    def _compute_notes(self, job: Job, lec: Lecture,
                       run: _NotesRun | None = None) -> dict | None:
        """The notes, in memory only. None when there is no transcript."""
        try:
            doc = json.loads(lec.transcript_path.read_text())
        except Exception as e:
            log.warning("no transcript to make notes from: %s", e)
            return None

        def on_chunk(i: int, n: int) -> None:
            if run is not None:
                run.last = (i, n)
                if not run.go.is_set():
                    return              # the archive's progress is on screen
            self._notes_progress(job, i, n)

        if run is not None and run.cancel.is_set():
            raise notes.Cancelled()
        stats: dict = {}
        t0 = time.time()
        nd = notes.generate(doc, progress_cb=on_chunk, premapped=job.premapped,
                            stats=stats, cancel=run.cancel if run else None)
        return {"nd": nd, "stats": stats, "compute_s": round(time.time() - t0, 1)}

    def _save_notes(self, job: Job, lec: Lecture, result: dict) -> None:
        nd, stats = result["nd"], result["stats"]
        m = lec.manifest()
        md = notes.to_markdown(nd, m.get("title") or lec.id,
                               m.get("recovery_note"))
        storage.atomic_write_json(lec.path / "notes.json", nd)
        storage.atomic_write_text(lec.notes_path, md)
        if not (lec.path / "notes.raw.md").exists():
            storage.atomic_write_text(lec.path / "notes.raw.md", md)
        n_bul = sum(len(s["bullets"]) for s in nd.get("sections") or [])
        job.notes_stats = stats
        lec.update(notes={"file": "notes.md", "engine": nd.get("engine"),
                          "fallback": nd.get("fallback"),
                          "bullets": n_bul, "terms": len(nd.get("terms") or []),
                          "quiz": len(nd.get("quiz") or []),
                          "generated_utc": storage.utcnow(),
                          "compute_s": result["compute_s"],
                          "map_s": stats.get("map_s") or [],
                          "reduce_s": stats.get("reduce_s"),
                          "premap": stats.get("premap") or {}})
        self._set(job, "noting", 98, f"{n_bul} notes written")

    def _archive(self, job: Job, lec: Lecture) -> None:
        """Compress to AAC and verify before any WAV is released."""
        lec.set_state("archiving")
        man = {}
        try:
            man = json.loads((lec.work / "concat.manifest.json").read_text())
        except Exception:
            pass
        if man.get("from_archive"):
            # This run decoded an existing archive; re-encoding it would be a
            # second lossy generation for no benefit.
            self._set(job, "archiving", 69, "archive already in place")
            archive.release_work(lec)
            return
        self._set(job, "archiving", 66, "compressing audio for long-term storage")
        before = lec.master_wav.stat().st_size if lec.master_wav.is_file() else 0
        info = archive.archive_lecture(lec)
        after = info["bytes"]
        ratio = (before / after) if after else 0
        self._set(job, "archiving", 69,
                  f"archived to {after/1e6:.1f} MB ({ratio:.1f}x smaller)")


PIPELINE = Pipeline()
