"""Transcribe while recording: the live draft built segment by segment.

A fake engine and synthetic WAV segments only -- never the microphone, never
whisper or fluidaudio.   Run: python -m unittest discover -s tests"""
import importlib, json, logging, os, shutil, sys, tempfile, threading, time, unittest, wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GB = 1024 ** 3
RATE = 16000


def fresh(tmp):
    os.environ["LECTERN_ROOT"] = str(Path(tmp) / "Lectures")
    os.environ["LECTERN_SUPPORT"] = str(Path(tmp) / "support")
    for name in [m for m in sys.modules if m.startswith("lectern")]:
        del sys.modules[name]
    config = importlib.import_module("lectern.config")
    config.ensure_dirs()
    return config


def write_wav(path, seconds, value=1):
    """Every sample is `value`, so a fake engine can tell segments apart."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(int(value).to_bytes(2, "little") * int(round(RATE * seconds)))


def wait_for(pred, timeout=5.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


class FakeEngine:
    """Stands in for transcribe.run_whisper. Every segment yields two words:
    "<segment-name>" at 0-200 ms and "end." at 400-500 ms, segment-relative.
    `fail` names segments that raise; `block` names segments that hang until
    their (fake) process is killed."""

    def __init__(self, transcribe, fail=(), block=()):
        self.transcribe = transcribe
        self.fail, self.block = set(fail), set(block)
        self.calls = []
        self.started = threading.Event()
        self.killed = threading.Event()

    def __call__(self, wav, out_prefix, *, progress_cb=None, model=None,
                 threads=None, proc_cb=None):
        # The live worker hands over a clean copy of the segment; which one
        # it is shows in its samples (segment n is written with value n + 1).
        with wave.open(str(wav)) as w:
            first = int.from_bytes(w.readframes(1), "little")
        name = f"seg-{first - 1:06d}.wav"
        self.calls.append(name)
        if name in self.block:
            proc = SimpleNamespace(kill=self.killed.set, poll=lambda: None)
            if proc_cb:
                proc_cb(proc)
            self.started.set()
            self.killed.wait(10)
            raise self.transcribe.TranscriptionError("whisper-cli exited -9")
        if name in self.fail:
            raise self.transcribe.TranscriptionError("whisper-cli exited 1")
        out = Path(str(out_prefix) + ".json")
        out.write_text(json.dumps({"transcription": [{"tokens": [
            {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}, "p": 1.0},
            {"text": " " + name, "offsets": {"from": 0, "to": 200}, "p": 0.9},
            {"text": " end", "offsets": {"from": 400, "to": 480}, "p": 0.8},
            {"text": ".", "offsets": {"from": 480, "to": 500}, "p": 0.95},
        ]}]}))
        return out


class Base(unittest.TestCase):
    RAM = 32 * GB

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = fresh(self.tmp)
        self.storage = importlib.import_module("lectern.storage")
        self.Lecture = self.storage.Lecture
        self.live = importlib.import_module("lectern.live")
        self.pipeline = importlib.import_module("lectern.pipeline")
        self.transcribe = importlib.import_module("lectern.transcribe")
        self.concat = importlib.import_module("lectern.concat")
        ram = mock.patch.object(self.pipeline.hardware, "effective_ram_bytes",
                                lambda *a, **k: self.RAM)
        ram.start()
        self.addCleanup(ram.stop)
        self.LIVE = self.live.LiveTranscriber()
        self.lec = self.Lecture.create(title="live")

    def seg(self, n, seconds):
        p = self.lec.work / f"seg-{n:06d}.wav"
        write_wav(p, seconds, value=n + 1)
        return p

    def engine(self, **kw):
        eng = FakeEngine(self.transcribe, **kw)
        p = mock.patch.object(self.live.transcribe, "run_whisper", eng)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.stop_live)      # runs first: no worker outlives the fake
        return eng

    def stop_live(self):
        for lid in list(self.LIVE._sessions):
            self.LIVE.finish(lid, timeout=0)
        t = self.LIVE._thread
        if t is not None:
            t.join(5)

    def draft(self):
        return json.loads(self.lec.draft_path.read_text())


# ------------------------------------------------------------------ timeline
class Timeline(Base):
    def test_offsets_match_concat_across_unequal_segments_and_pause_resume(self):
        # Recording: a full segment, then a short one cut by a pause.
        self.seg(0, 3.0)
        self.seg(1, 1.7)
        eng = self.engine()
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)          # Pause: both are final
        wait_for(lambda: len(eng.calls) == 2 and self.lec.draft_path.is_file()
                 and self.draft()["complete"], what="draft after pause")
        # Continue: numbering carries on, the last segment is still growing.
        self.seg(2, 2.5)
        self.seg(3, 0.9)
        self.LIVE.begin(self.lec.id)
        wait_for(lambda: len(eng.calls) == 3, what="seg 2 while recording")
        time.sleep(0.1)
        self.assertEqual(eng.calls, ["seg-000000.wav", "seg-000001.wav", "seg-000002.wav"],
                         "the growing segment must not be transcribed")
        self.LIVE.capture_ended(self.lec.id)          # Stop
        self.LIVE.finish(self.lec.id, timeout=5)
        self.assertEqual(len(eng.calls), 4)

        man = self.concat.concat_segments(self.lec.segments(), self.lec.master_wav)
        starts, acc = [], 0
        for s in man["segments"]:
            starts.append(round(acc * 1000 / RATE))
            acc += s["frames"]
        d = self.draft()
        self.assertTrue(d["draft"])
        self.assertTrue(d["complete"])
        self.assertEqual(d["duration_ms"], man["duration_ms"])
        w = d["words"]
        self.assertEqual(w["w"], ["seg-000000.wav", "end.", "seg-000001.wav", "end.",
                                  "seg-000002.wav", "end.", "seg-000003.wav", "end."])
        self.assertEqual(w["t0"][0::2], starts)                 # 0, 3000, 4700, 7200
        self.assertEqual(w["t0"][1::2], [s + 400 for s in starts])
        self.assertEqual(w["t1"][1::2], [s + 500 for s in starts])
        self.assertEqual(starts, [0, 3000, 4700, 7200])
        self.assertEqual([s["t0"] for s in d["sents"]], starts)   # one sentence per segment

    def test_segment_frames_are_concats_frames(self):
        p = self.seg(0, 1.2345)
        dest = self.lec.work / "copy.wav"
        frames = self.concat.copy_segment(p, dest)
        man = self.concat.concat_segments([p], self.lec.master_wav)
        self.assertEqual(frames, man["segments"][0]["frames"])

    def test_resume_after_restart_continues_a_matching_draft(self):
        self.seg(0, 1.0)
        self.seg(1, 1.0)
        eng = self.engine()
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)
        self.LIVE.finish(self.lec.id, timeout=5)
        self.assertEqual(len(eng.calls), 2)
        # A new backend process (fresh transcriber) resumes the same lecture.
        self.seg(2, 1.5)
        later = self.live.LiveTranscriber()
        later.begin(self.lec.id)
        later.capture_ended(self.lec.id)
        later.finish(self.lec.id, timeout=5)
        self.assertEqual(eng.calls[2:], ["seg-000002.wav"], "earlier segments were reused")
        self.assertEqual(self.draft()["words"]["t0"][4], 2000)

    def test_a_draft_that_no_longer_matches_is_started_afresh(self):
        self.seg(0, 1.0)
        eng = self.engine()
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)
        self.LIVE.finish(self.lec.id, timeout=5)
        self.seg(0, 2.0)                       # e.g. re-seeded from the archive
        later = self.live.LiveTranscriber()
        later.begin(self.lec.id)
        later.capture_ended(self.lec.id)
        later.finish(self.lec.id, timeout=5)
        self.assertEqual(len(eng.calls), 2)
        self.assertEqual(self.draft()["duration_ms"], 2000)


# ------------------------------------------------------------- robustness
class Robustness(Base):
    def test_failed_segment_leaves_a_usable_draft_and_switches_off(self):
        self.seg(0, 1.0)
        self.seg(1, 1.0)
        self.seg(2, 1.0)
        eng = self.engine(fail={"seg-000001.wav"})
        with self.assertLogs("lectern.live", logging.WARNING) as cm:
            self.LIVE.begin(self.lec.id)
            self.LIVE.capture_ended(self.lec.id)
            self.LIVE.finish(self.lec.id, timeout=5)
        self.assertTrue(any("switched off" in m for m in cm.output), cm.output)
        self.assertEqual(eng.calls, ["seg-000000.wav", "seg-000001.wav"],
                         "no further segments after a failure")
        d = self.draft()
        self.assertEqual(d["words"]["w"], ["seg-000000.wav", "end."])
        self.assertTrue(d["live"]["off"])
        self.assertFalse(d["complete"])
        # And it stays off when the same recording is continued.
        self.seg(3, 1.0)
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)
        self.LIVE.finish(self.lec.id, timeout=5)
        self.assertEqual(len(eng.calls), 2)

    def test_an_engine_slower_than_the_audio_switches_off_after_two_runs(self):
        for n in range(4):
            self.seg(n, 0.5)
        self.seg(4, 0.2)                                 # still growing
        eng = self.engine()
        slow = eng.__call__

        def sluggish(*a, **k):
            time.sleep(0.05)                             # 0.05 s for 0.5 s of audio...
            return slow(*a, **k)
        with mock.patch.object(self.live.transcribe, "run_whisper", sluggish), \
             mock.patch.object(self.live, "MIN_JUDGED_AUDIO_S", 0), \
             mock.patch.object(self.live, "SLOW_RATIO", 0.01), \
             self.assertLogs("lectern.live", logging.WARNING) as cm:   # ...is "slow" at 0.01
            self.LIVE.begin(self.lec.id)
            wait_for(lambda: not self.LIVE.busy(self.lec.id), what="switch-off")
        self.assertTrue(any("too slow" in m for m in cm.output), cm.output)
        self.assertEqual(len(eng.calls), self.live.SLOW_RUNS_TO_SWITCH_OFF)
        self.assertEqual(len(self.draft()["words"]["w"]), 2 * self.live.SLOW_RUNS_TO_SWITCH_OFF,
                         "the segments it did finish stay in the draft")

    def test_one_slow_run_is_forgiven(self):
        for n in range(3):
            self.seg(n, 0.5)
        self.seg(3, 0.2)
        eng = self.engine()
        real, runs = eng.__call__, []

        def first_slow(*a, **k):
            runs.append(1)
            if len(runs) == 1:
                time.sleep(0.05)
            return real(*a, **k)
        with mock.patch.object(self.live.transcribe, "run_whisper", first_slow), \
             mock.patch.object(self.live, "MIN_JUDGED_AUDIO_S", 0), \
             mock.patch.object(self.live, "SLOW_RATIO", 0.05):
            self.LIVE.begin(self.lec.id)
            wait_for(lambda: len(eng.calls) == 3, what="all finished segments")
            self.LIVE.capture_ended(self.lec.id)
            self.LIVE.finish(self.lec.id, timeout=5)
        self.assertTrue(self.draft()["complete"])
        self.assertIsNone(self.draft()["live"]["off"])

    def test_back_to_back_lectures_catch_up_after_the_stage_lock_frees(self):
        # Lecture A's pipeline holds the stage lock for several segments' worth
        # of lecture B: that is waiting, not falling behind.
        self.seg(0, 1.0)
        self.seg(1, 0.5)                                 # growing
        eng = self.engine()
        with mock.patch.object(self.live, "POLL_S", 0.02):
            with self.pipeline.STAGE_LOCK:
                self.LIVE.begin(self.lec.id)
                for n in range(1, 6):                    # B keeps recording
                    self.seg(n, 1.0)
                    self.seg(n + 1, 0.3)
                    time.sleep(0.1)
                self.assertEqual(eng.calls, [])
                self.assertTrue(self.LIVE.busy(self.lec.id), "still on, just waiting")
            wait_for(lambda: len(eng.calls) == 6, what="catch-up after release")
            self.LIVE.capture_ended(self.lec.id)             # B stops
            self.LIVE.finish(self.lec.id, timeout=5)
        self.assertEqual(eng.calls, [f"seg-{n:06d}.wav" for n in range(7)])
        d = self.draft()
        self.assertTrue(d["complete"])
        self.assertIsNone(d["live"]["off"])
        self.assertEqual(d["words"]["t0"][0::2], [n * 1000 for n in range(7)])

    def test_waits_while_another_lecture_holds_the_stage_lock(self):
        self.seg(0, 1.0)
        eng = self.engine()
        with mock.patch.object(self.live, "POLL_S", 0.05):
            with self.pipeline.STAGE_LOCK:
                self.LIVE.begin(self.lec.id)
                self.LIVE.capture_ended(self.lec.id)
                time.sleep(0.3)
                self.assertEqual(eng.calls, [])
            wait_for(lambda: eng.calls == ["seg-000000.wav"], what="run after release")

    def test_turned_off_in_settings(self):
        self.seg(0, 1.0)
        eng = self.engine()
        self.live.set_enabled(False)
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)
        self.LIVE.finish(self.lec.id, timeout=5)
        self.assertEqual(eng.calls, [])
        self.assertFalse(self.lec.draft_path.exists())

    def test_wait_idle_kills_a_stuck_live_job(self):
        self.seg(0, 1.0)
        eng = self.engine(block={"seg-000000.wav"})
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)
        self.assertTrue(eng.started.wait(5))
        t0 = time.monotonic()
        self.LIVE.wait_idle(timeout=0.2)
        self.assertLess(time.monotonic() - t0, 2)
        self.assertTrue(eng.killed.is_set())


class Defaults(Base):
    def test_off_by_default_on_small_memory(self):
        with mock.patch.object(self.pipeline.hardware, "effective_ram_bytes", lambda *a, **k: 8 * GB):
            self.assertFalse(self.live.enabled())
            self.live.set_enabled(True)                 # an explicit choice wins
            self.assertTrue(self.live.enabled())
        self.live.set_enabled(None)
        self.assertTrue(self.live.enabled())            # 32 GB here
        self.live.set_enabled(False)
        self.assertFalse(self.live.enabled())

    def test_settings_endpoint(self):
        server = importlib.import_module("lectern.server")
        c = server.app.test_client()
        r = c.get("/api/settings/live")
        self.assertEqual(r.get_json()["enabled"], True)
        r = c.post("/api/settings/live", json={"enabled": False})
        self.assertEqual(r.get_json()["enabled"], False)
        self.assertFalse(self.live.enabled())
        self.assertEqual(c.post("/api/settings/live", json={"enabled": "x"}).status_code, 400)


# ----------------------------------------------------------- stop + final
class FakeProc:
    def __init__(self):
        self.pid, self.returncode, self.stdout = 999999, None, []

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class StopAndFinal(Base):
    def test_stop_is_not_delayed_by_a_slow_live_transcription(self):
        recorder = importlib.import_module("lectern.recorder")
        self.seg(0, 1.0)
        self.seg(1, 0.5)                                 # growing
        eng = self.engine(block={"seg-000000.wav"})
        rec = recorder.Recorder()
        stack = [mock.patch.object(recorder.live, "LIVE", self.LIVE),
                 mock.patch.object(self.pipeline.live, "LIVE", self.LIVE),
                 mock.patch.object(recorder.Recorder, "_start_caffeinate", lambda s, pid: None),
                 mock.patch.object(recorder.Recorder, "_stop_file_for", staticmethod(lambda i: None)),
                 mock.patch.object(self.live, "FINISH_TIMEOUT_S", 0.5)]
        for p in stack:
            p.start()
            self.addCleanup(p.stop)
        with rec._lock:
            rec._track(self.lec, FakeProc(), SimpleNamespace(name="USB Mic", index=0), 0)
        self.assertTrue(eng.started.wait(5), "live transcription of seg 0 started")

        t0 = time.monotonic()
        rec.stop()
        self.assertLess(time.monotonic() - t0, 1.0, "Stop waited on the live job")

        reached = threading.Event()
        p = self.pipeline.Pipeline()
        with mock.patch.object(p, "_concat", lambda j, l: None), \
             mock.patch.object(p, "_transcribe", lambda j, l: reached.set()), \
             mock.patch.object(p, "_archive", lambda j, l: None), \
             mock.patch.object(p, "_notes", lambda j, l: None):
            t0 = time.monotonic()
            p.submit(self.lec.id)
            self.assertTrue(reached.wait(5), "pipeline never reached transcription")
            wait_for(lambda: (p.job(self.lec.id) or {}).get("done"), what="job done")
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertTrue(eng.killed.is_set(), "the stuck live job was stopped")

    def test_final_transcript_replaces_the_draft(self):
        server = importlib.import_module("lectern.server")
        server._STARTUP_DONE.set()
        c = server.app.test_client()
        self.seg(0, 1.0)
        self.seg(1, 1.0)
        self.engine()
        self.LIVE.begin(self.lec.id)
        self.LIVE.capture_ended(self.lec.id)
        self.LIVE.finish(self.lec.id, timeout=5)

        with c.get(f"/api/lecture/{self.lec.id}/transcript") as r:
            self.assertEqual(r.status_code, 200)
            self.assertTrue(r.get_json()["draft"])
        summ = c.get(f"/api/lecture/{self.lec.id}").get_json()
        self.assertEqual((summ["has_transcript"], summ["has_draft"]), (False, True))

        # The full-file pass, with a fake engine of its own.
        self.concat.concat_segments(self.lec.segments(), self.lec.master_wav)
        self.lec.update(audio={"duration_ms": 2000})

        def final(wav, prefix, *, progress_cb=None):
            out = Path(str(prefix) + ".json")
            out.write_text(json.dumps({"transcription": [{"tokens": [
                {"text": " Final", "offsets": {"from": 10, "to": 300}, "p": 1.0},
                {"text": " words.", "offsets": {"from": 1200, "to": 1500}, "p": 1.0}]}]}))
            return out, "fake"
        p = self.pipeline.Pipeline()
        job = self.pipeline.Job(self.lec.id)
        with mock.patch.object(self.pipeline.transcribe, "run_with_fallback", final):
            p._transcribe(job, self.lec)

        with c.get(f"/api/lecture/{self.lec.id}/transcript") as r:
            doc = r.get_json()
        self.assertNotIn("draft", doc)
        self.assertEqual(doc["words"]["w"], ["Final", "words."])
        self.assertFalse(self.lec.draft_path.exists())
        summ = c.get(f"/api/lecture/{self.lec.id}").get_json()
        self.assertEqual((summ["has_transcript"], summ["has_draft"]), (True, False))


class Recovery(Base):
    def _interrupted(self, with_draft):
        lec = self.Lecture.create(title="crash")
        write_wav(lec.work / "seg-000000.wav", 3.0)
        write_wav(lec.work / "seg-000001.wav", 4.0)
        if with_draft:
            (lec.work / "draft.json").write_text('{"draft": true, "words": {"w": []}}')
            (lec.work / "draft.raw.json").write_text("{not json")
            (lec.work / "live-seg.wav").write_bytes(b"RIFF")
        return lec

    def test_recovery_is_the_same_with_or_without_a_draft(self):
        recovery = importlib.import_module("lectern.recovery")
        results = []
        for with_draft in (False, True):
            lec = self._interrupted(with_draft)
            self.assertIn(lec.id, [l.id for l in recovery.scan_interrupted()])
            res = recovery.recover(lec)
            m = self.Lecture.load(lec.id).manifest()
            results.append((res, m["state"], m["audio"]["duration_ms"],
                            lec.transcript_path.exists()))
        self.assertEqual(results[0], results[1])
        self.assertFalse(results[1][3], "a draft is never a transcript")

    def test_released_work_with_a_draft_is_fully_purged(self):
        recovery = importlib.import_module("lectern.recovery")
        archive = importlib.import_module("lectern.archive")
        lec = self._interrupted(True)
        dest = archive.release_work(lec)
        recovery._purge_work_dir(dest)
        self.assertFalse(dest.exists(), list(dest.iterdir()) if dest.exists() else None)


if __name__ == "__main__":
    unittest.main()
