"""Pre-warming the Core ML engines, and honest progress for Parakeet.

A fake engine only -- never the microphone, never whisper or fluidaudio.
Run: python -m unittest discover -s tests"""
import importlib, json, os, shutil, stat, sys, tempfile, threading, time, unittest, wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_models_hw import fresh_modules


def wait_for(pred, timeout=5.0, what="condition"):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


class FakeEngine:
    """Stands in for transcribe.run_whisper. block=True hangs until its (fake)
    process is killed, then fails the way a killed engine does."""

    def __init__(self, transcribe, block=False, fail=False):
        self.transcribe, self.block, self.fail = transcribe, block, fail
        self.calls = []
        self.started = threading.Event()
        self.killed = threading.Event()

    def __call__(self, wav, out_prefix, *, progress_cb=None, model=None,
                 threads=None, proc_cb=None):
        with wave.open(str(wav)) as w:                 # a real, readable clip
            self.calls.append({"model": model, "threads": threads,
                               "seconds": w.getnframes() / w.getframerate()})
        if proc_cb:
            proc_cb(SimpleNamespace(kill=self.killed.set, poll=lambda: None))
        self.started.set()
        if self.block:
            self.killed.wait(10)
            raise self.transcribe.TranscriptionError("fluidaudiocli exited -9")
        if self.fail:
            raise self.transcribe.TranscriptionError("fluidaudiocli exited 3\nANE busy")
        out = Path(str(out_prefix) + ".json")
        out.write_text('{"transcription": []}')
        return out


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        engine = Path(self.tmp) / "fluidaudiocli"
        engine.write_text("#!/bin/sh\nexit 0\n")
        engine.chmod(engine.stat().st_mode | stat.S_IXUSR)
        self.engine = engine
        self.models = fresh_modules(self.tmp, LECTERN_MODEL_DIR=str(Path(self.tmp) / "models"),
                                    LECTERN_FLUIDAUDIO=str(engine), LECTERN_WARMUP="1")
        self.models.LEGACY_DIR = self.models.BENCH_DIR = Path(self.tmp) / "none"
        self.config = importlib.import_module("lectern.config")
        self.config.ensure_dirs()
        self.transcribe = importlib.import_module("lectern.transcribe")
        self.warmup = importlib.import_module("lectern.warmup")
        self.warmup._machine_id = lambda: "test-mac"
        self.warmup.BUSY_RETRY_S = 0.05
        self.warmup.WATCH_S = 0.02
        self.pipeline = importlib.import_module("lectern.pipeline")
        self.live = importlib.import_module("lectern.live")
        self.recorder = importlib.import_module("lectern.recorder")
        self.pk = self.models.MODELS_DIR / "parakeet-tdt-0.6b-v3-coreml"
        self.pk.mkdir(parents=True)
        (self.pk / "config.json").write_text("{}")
        self.models.save_settings(whisper_model="parakeet-tdt-0.6b-v3-coreml")
        self.W = self.warmup.Warmer()

    def use(self, engine):
        self.transcribe.run_whisper = engine
        return engine


class WhichModels(Base):
    def test_coreml_folder_and_whisper_with_encoder_need_warming(self):
        w = self.warmup
        self.assertTrue(w.uses_coreml(self.pk))
        q4 = self.models.MODELS_DIR / "ggml-large-v3-turbo-q4_0.bin"
        q4.write_bytes(b"lmgg")
        self.assertFalse(w.uses_coreml(q4))                 # no encoder beside it yet
        # whisper.cpp drops the "-q4_0" suffix when it looks for the encoder
        (self.models.MODELS_DIR / "ggml-large-v3-turbo-encoder.mlmodelc").mkdir()
        self.assertTrue(w.uses_coreml(q4))
        plain = self.models.MODELS_DIR / "ggml-small.en-q5_1.bin"
        plain.write_bytes(b"lmgg")
        self.assertFalse(w.uses_coreml(plain))

    def test_key_changes_with_app_version_engine_and_model(self):
        w = self.warmup
        k = w.key(self.pk)
        self.assertEqual(k, w.key(self.pk))
        with mock.patch.object(self.config, "VERSION", "9.9.9"):
            self.assertNotEqual(k, w.key(self.pk))
        self.engine.write_text("#!/bin/sh\n# a newer build\nexit 0\n")
        k2 = w.key(self.pk)
        self.assertNotEqual(k, k2)
        os.utime(self.pk, ns=(1, 1))                        # the model folder was replaced
        self.assertNotEqual(k2, w.key(self.pk))


class Warms(Base):
    def test_warms_the_active_model_once_and_records_it(self):
        eng = self.use(FakeEngine(self.transcribe))
        self.W.request()
        wait_for(lambda: self.W.idle(), what="warm-up to finish")
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual(eng.calls[0]["model"], self.pk)
        self.assertLessEqual(eng.calls[0]["seconds"], 5)    # a short clip, not a lecture
        self.assertTrue(self.warmup.is_warmed(self.pk))
        self.assertTrue(self.warmup.RECORD_FILE.is_file())
        self.assertFalse(any(self.warmup.WORK_DIR.glob("*")))   # clip and output removed
        self.W.request()                                   # already warmed: nothing runs
        wait_for(lambda: self.W.idle(), what="second request")
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual(self.W.status()["state"], "ready")

    def test_a_new_app_version_warms_again(self):
        eng = self.use(FakeEngine(self.transcribe))
        self.W.request()
        wait_for(lambda: self.W.idle() and len(eng.calls) == 1)
        with mock.patch.object(self.config, "VERSION", "9.9.9"):
            self.W.request()
            wait_for(lambda: self.W.idle() and len(eng.calls) == 2, what="re-warm")

    def test_whisper_model_without_coreml_is_left_alone(self):
        eng = self.use(FakeEngine(self.transcribe))
        shutil.rmtree(self.pk)
        small = self.models.BY_ID["small.en"]
        with open(self.models.MODELS_DIR / small.file, "wb") as f:
            f.truncate(small.size_mb * 1_000_000)
        self.models.save_settings(whisper_model="small.en")
        self.W.request()
        wait_for(lambda: self.W.idle())
        self.assertEqual(eng.calls, [])
        self.assertEqual(self.W.status()["state"], "idle")

    def test_a_downloaded_model_that_is_not_active_is_warmed_too(self):
        eng = self.use(FakeEngine(self.transcribe))
        other = self.models.MODELS_DIR / "other-coreml"
        other.mkdir()
        self.W.request(self.pk)
        self.W.request(other)
        wait_for(lambda: self.W.idle() and len(eng.calls) == 2)
        self.assertEqual({c["model"] for c in eng.calls}, {self.pk, other})

    def test_failure_only_logs_and_is_not_retried_in_this_run(self):
        eng = self.use(FakeEngine(self.transcribe, fail=True))
        with self.assertLogs("lectern.warmup", "WARNING") as logs:
            self.W.request()
            wait_for(lambda: self.W.idle())
        self.assertIn("ANE busy", "\n".join(logs.output))
        st = self.W.status()
        self.assertEqual(st["state"], "failed")
        self.assertIn("exited 3", st["error"])
        self.assertFalse(self.warmup.is_warmed(self.pk))
        self.W.request()
        wait_for(lambda: self.W.idle())
        self.assertEqual(len(eng.calls), 1)


class StaysOutOfTheWay(Base):
    def test_waits_while_a_transcription_holds_the_stage_lock(self):
        eng = self.use(FakeEngine(self.transcribe))
        release = threading.Event()
        held = threading.Event()

        def hold():
            with self.pipeline.STAGE_LOCK:
                held.set()
                release.wait(10)
        threading.Thread(target=hold, daemon=True).start()
        held.wait(2)
        self.W.request()
        wait_for(lambda: self.W.status()["state"] == "waiting", what="waiting state")
        time.sleep(0.2)
        self.assertEqual(eng.calls, [])
        release.set()
        wait_for(lambda: self.W.idle() and len(eng.calls) == 1, what="warm-up after release")

    def test_waits_while_recording(self):
        eng = self.use(FakeEngine(self.transcribe))
        rec = {"id": "lec-1"}
        self.recorder.RECORDER.current_lecture_id = lambda: rec["id"]
        self.W.request()
        wait_for(lambda: self.W.status()["state"] == "waiting")
        time.sleep(0.2)
        self.assertEqual(eng.calls, [])
        rec["id"] = None
        wait_for(lambda: self.W.idle() and len(eng.calls) == 1)

    def test_waits_while_a_live_draft_is_in_progress(self):
        eng = self.use(FakeEngine(self.transcribe))
        self.live.LIVE.busy_any = lambda: True
        self.W.request()
        wait_for(lambda: self.W.status()["state"] == "waiting")
        self.assertEqual(eng.calls, [])

    def test_preempt_kills_it_and_it_is_retried_later(self):
        eng = self.use(FakeEngine(self.transcribe, block=True))
        self.W.request()
        self.assertTrue(eng.started.wait(2))
        self.assertEqual(self.W.status()["state"], "warming")
        hold = threading.Event()

        def real_work():                       # what pipeline._transcribe does
            with self.pipeline.STAGE_LOCK:
                self.W.preempt()
                self.assertTrue(eng.killed.is_set())
                self.assertFalse(self.W.engine_running())
                hold.wait(10)
        t = threading.Thread(target=real_work, daemon=True)
        t.start()
        wait_for(lambda: self.W.status()["state"] == "waiting", what="deferred")
        self.assertFalse(self.warmup.is_warmed(self.pk))
        eng.block = False
        hold.set()
        t.join(2)
        wait_for(lambda: self.W.idle() and self.warmup.is_warmed(self.pk), what="retry")
        self.assertEqual(len(eng.calls), 2)

    def test_recording_that_starts_mid_warmup_kills_it(self):
        eng = self.use(FakeEngine(self.transcribe, block=True))
        rec = {"id": None}
        self.recorder.RECORDER.current_lecture_id = lambda: rec["id"]
        self.W.request()
        self.assertTrue(eng.started.wait(2))
        rec["id"] = "lec-1"
        self.assertTrue(eng.killed.wait(2))
        wait_for(lambda: self.W.status()["state"] == "waiting")
        self.assertFalse(self.warmup.is_warmed(self.pk))

    def test_preempt_with_nothing_running_returns_at_once(self):
        t0 = time.monotonic()
        self.W.preempt()
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_start_delay_does_not_block_the_caller(self):
        self.use(FakeEngine(self.transcribe))
        t0 = time.monotonic()
        self.W.request(delay=5)
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_pipeline_makes_the_warmup_yield_before_its_engine(self):
        order = []
        self.warmup.WARMER.preempt = lambda *a, **k: order.append("preempt")
        self.live.LIVE.wait_idle = lambda *a, **k: order.append("live idle")

        def engine(*a, **k):
            order.append("engine")
            raise RuntimeError("stop here")
        self.transcribe.run_with_fallback = engine
        lec = mock.MagicMock()
        lec.duration_ms.return_value = 1000
        with self.assertRaises(RuntimeError):
            self.pipeline.Pipeline()._transcribe(self.pipeline.Job("x"), lec)
        self.assertEqual(order, ["live idle", "preempt", "engine"])

    def test_live_makes_the_warmup_yield_before_its_engine(self):
        order = []
        self.warmup.WARMER.preempt = lambda *a, **k: order.append("preempt")

        def engine(*a, **k):
            order.append("engine")
            raise self.transcribe.TranscriptionError("stop here")
        self.transcribe.run_whisper = engine
        lec = mock.MagicMock()
        seg = Path(self.tmp) / "seg-000000.wav"
        with wave.open(str(seg), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\0\0" * 16000)
        lec.work = Path(self.tmp)
        s = self.live._Session("x")
        self.live.LiveTranscriber()._transcribe_one(s, lec, seg)
        self.assertEqual(order, ["preempt", "engine"])


class Exposed(Base):
    def test_status_and_system_report_the_warmup(self):
        server = importlib.import_module("lectern.server")
        server.warmup.WARMER._set(state="warming", model="parakeet-tdt-0.6b-v3-coreml")
        c = server.app.test_client()
        self.assertEqual(c.get("/api/system").get_json()["warmup"]["state"], "warming")
        self.assertEqual(c.get("/api/status").get_json()["warmup"]["state"], "warming")


class ParakeetProgress(Base):
    """fluidaudiocli reports nothing while it runs, so the job says so instead
    of sitting at a fixed percentage."""

    FAKE = "#!/bin/sh\nout=\"\"; prev=\"\"\nfor a in \"$@\"; do [ \"$prev\" = \"--output-json\" ] && out=\"$a\"; prev=\"$a\"; done\n" \
           "echo '{\"wordTimings\":[{\"word\":\"Hi.\",\"startTime\":0.5,\"endTime\":0.9}]}' > \"$out\"\n"

    def test_run_whisper_says_no_percentage_then_done(self):
        self.engine.write_text(self.FAKE)
        wav = Path(self.tmp) / "a.wav"
        wav.write_bytes(b"RIFF")
        seen = []
        self.transcribe.run_whisper(wav, Path(self.tmp) / "w.raw", progress_cb=seen.append)
        self.assertEqual(seen, [None, 100])

    def test_job_is_indeterminate_while_the_engine_gives_no_percentage(self):
        P = self.pipeline.Pipeline()
        job = self.pipeline.Job("x")
        P._set(job, "transcribing", 10, "transcribing…", indeterminate=True)
        d = job.as_dict()
        self.assertTrue(d["indeterminate"])
        P._set(job, "transcribing", 40, "transcribing… 55%")
        self.assertFalse(job.as_dict()["indeterminate"])


if __name__ == "__main__":
    unittest.main()
