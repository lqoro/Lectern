"""Post-recording processing, serialized by a global stage lock.

The RAM invariant this module enforces: whisper peaks around 1.05 GB on a
50-minute lecture and qwen2.5:3b needs ~2.3 GB. On an 8 GB machine that is
already swapping with a browser open, they must NEVER be resident together.
One process-wide lock guarantees at most one heavy stage runs at a time.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from . import archive, concat, config, importer, models, notes, storage, transcribe
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


# The RAM interlock. Every heavy stage acquires this.
STAGE_LOCK = FifoLock()
log = logging.getLogger(__name__)


@dataclass
class Job:
    lecture_id: str
    stage: str = "queued"
    percent: int = 0
    message: str = ""
    error: str = ""
    done: bool = False
    started: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {"lecture_id": self.lecture_id, "stage": self.stage,
                "percent": self.percent, "message": self.message,
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
            prior = lec.manifest().get("state") or "complete"
            self._set(job, "waiting", 0, "waiting for a free processing slot")
            with STAGE_LOCK:
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
    def _set(self, job: Job, stage: str, percent: int, message: str = "") -> None:
        with self._lock:
            job.stage, job.percent, job.message = stage, percent, message

    def _run(self, job: Job, extract: bool = False) -> None:
        try:
            lec = Lecture.load(job.lecture_id)
        except Exception as e:
            with self._lock:
                job.error, job.done = f"cannot load lecture: {e}", True
            return

        try:
            self._set(job, "waiting", 0, "waiting for a free processing slot")
            with STAGE_LOCK:
                if extract:
                    self._extract(job, lec)
                self._concat(job, lec)
                self._transcribe(job, lec)
                self._notes(job, lec)
                self._archive(job, lec)
            lec.set_state("complete", failure=None)   # a re-run clears an old failure
            self._set(job, "complete", 100, "done")
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            with self._lock:
                job.error = f"{e}"
            try:
                lec.set_state("failed", failure=str(e)[:500])
            except Exception:
                pass
            self._set(job, "failed", job.percent, str(e)[:300])
            try:
                with (config.LOG_DIR / "pipeline.log").open("a") as lf:
                    lf.write(f"\n[{storage.utcnow()}] {job.lecture_id}\n{tb}\n")
            except OSError:
                log.exception("could not write pipeline.log")
        finally:
            with self._lock:
                job.done = True

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

        def on_pct(p: int) -> None:
            self._set(job, "transcribing", 10 + int(p * 0.85),
                      f"transcribing… {p}%")

        t0 = time.time()
        raw = transcribe.run_whisper(lec.master_wav, lec.work / "whisper.raw",
                                     progress_cb=on_pct)
        doc = transcribe.normalize(raw, dur)
        if not doc["words"]["w"]:
            raise RuntimeError("no speech was found in the recording")

        storage.atomic_write_json(lec.transcript_path, doc)
        storage.atomic_write_text(lec.transcript_txt_path,
                                  transcribe.to_plain_text(doc))
        m = lec.manifest()
        lec.update(transcript={
            "file": "transcript.json",
            "words": len(doc["words"]["w"]),
            "sentences": len(doc["sents"]),
            "language": doc["language"],
            "model": models.active_name(),
            "compute_s": round(time.time() - t0, 1),
        })
        self._set(job, "transcribing", 58,
                  f"{len(doc['words']['w'])} words transcribed")

    def _notes(self, job: Job, lec: Lecture) -> None:
        """Study notes. whisper has already exited, so the ~2.3 GB model never
        shares memory with it -- that is the whole point of the stage lock.
        Notes never fail a recording: the extractive fallback always produces
        something, and notes are regenerable from transcript.json at any time.
        """
        lec.set_state("noting")
        self._set(job, "noting", 62, "writing study notes")
        try:
            doc = json.loads(lec.transcript_path.read_text())
        except Exception as e:
            log.warning("no transcript to make notes from: %s", e)
            return

        def on_chunk(i: int, n: int) -> None:
            # Progress, never an ETA: generation speed on a machine that may be
            # paging has not been measured and a wrong ETA is worse than none.
            self._set(job, "noting", 62 + int(26 * i / max(1, n)),
                      f"writing study notes… part {i+1} of {n}")

        t0 = time.time()
        nd = notes.generate(doc, progress_cb=on_chunk)
        m = lec.manifest()
        md = notes.to_markdown(nd, m.get("title") or lec.id,
                               m.get("recovery_note"))
        storage.atomic_write_json(lec.path / "notes.json", nd)
        storage.atomic_write_text(lec.notes_path, md)
        if not (lec.path / "notes.raw.md").exists():
            storage.atomic_write_text(lec.path / "notes.raw.md", md)
        n_bul = sum(len(s["bullets"]) for s in nd.get("sections") or [])
        lec.update(notes={"file": "notes.md", "engine": nd.get("engine"),
                          "fallback": nd.get("fallback"),
                          "bullets": n_bul, "terms": len(nd.get("terms") or []),
                          "quiz": len(nd.get("quiz") or []),
                          "generated_utc": storage.utcnow(),
                          "compute_s": round(time.time() - t0, 1)})
        self._set(job, "noting", 90, f"{n_bul} notes written")

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
            self._set(job, "archiving", 99, "archive already in place")
            archive.release_work(lec)
            return
        self._set(job, "archiving", 92, "compressing audio for long-term storage")
        before = lec.master_wav.stat().st_size if lec.master_wav.is_file() else 0
        info = archive.archive_lecture(lec)
        after = info["bytes"]
        ratio = (before / after) if after else 0
        self._set(job, "archiving", 99,
                  f"archived to {after/1e6:.1f} MB ({ratio:.1f}x smaller)")


PIPELINE = Pipeline()
