"""Recording the Mac's own sound (YouTube, Zoom, any app) instead of the mic.

The ScreenCaptureKit helper itself only builds and runs on a Mac; these tests
cover everything around it: choosing the source, the helper's command line,
resuming, and the guarantee that system audio never silently falls back to the
microphone.   Run: python -m unittest discover -s tests"""
import importlib, os, shutil, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fresh(tmp):
    os.environ["LECTERN_ROOT"] = str(Path(tmp) / "Lectures")
    os.environ["LECTERN_SUPPORT"] = str(Path(tmp) / "support")
    for name in [m for m in sys.modules if m.startswith("lectern")]:
        del sys.modules[name]
    config = importlib.import_module("lectern.config")
    config.ensure_dirs()
    return config


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.config = fresh(self.tmp)
        self.audio = importlib.import_module("lectern.audio")
        self.recorder = importlib.import_module("lectern.recorder")
        self.storage = importlib.import_module("lectern.storage")
        self.models = importlib.import_module("lectern.models")

    def mac(self, version="14.5"):
        """Pretend to be a Mac with the native helper."""
        return [
            mock.patch.object(self.audio.pc, "IS_MAC", True),
            mock.patch.object(self.audio.capture, "portable", lambda: False),
            mock.patch("platform.mac_ver", lambda: (version, ("", "", ""), "")),
        ]


class Support(Base):
    def supported(self, version):
        ps = self.mac(version)
        for p in ps:
            p.start(); self.addCleanup(p.stop)
        return self.audio.system_audio_supported()

    def test_needs_macos_13(self):
        self.assertTrue(self.supported("13.0"))
        self.assertTrue(self.supported("26.1"))
        self.assertFalse(self.supported("12.7"))

    def test_not_on_windows_or_linux(self):
        with mock.patch.object(self.audio.pc, "IS_MAC", False):
            self.assertFalse(self.audio.system_audio_supported())
            with self.assertRaises(self.audio.PreflightError) as e:
                self.audio.pick_device("system")
            self.assertEqual(e.exception.kind, "no_system_audio")

    def test_default_is_the_mic_until_chosen_and_never_where_unsupported(self):
        self.assertEqual(self.audio.default_source(), "mic")
        self.models.save_settings(record_source="system")
        with mock.patch.object(self.audio.pc, "IS_MAC", False):
            self.assertEqual(self.audio.default_source(), "mic")     # not a silent recording
        for p in self.mac():
            p.start(); self.addCleanup(p.stop)
        self.assertEqual(self.audio.default_source(), "system")
        self.assertTrue(self.audio.is_system(self.audio.pick_device()))

    def test_explicit_mic_beats_a_system_default(self):
        self.models.save_settings(record_source="system")
        mic = self.audio.Device(0, "USB Mic")
        with mock.patch.object(self.audio, "resolve_device", lambda fresh=False: mic):
            self.assertIs(self.audio.pick_device("mic"), mic)


class HelperCommand(Base):
    def lecture(self):
        return self.storage.Lecture.create("Sys")

    def test_system_source_uses_source_flag_and_no_mic_or_lid(self):
        lec = self.lecture()
        for p in self.mac():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(self.recorder.capture, "available", lambda: Path("/x/lectern-capture")):
            argv = self.recorder.Recorder()._argv(lec, self.audio.SYSTEM_DEVICE)
        self.assertEqual(argv[:3], ["/x/lectern-capture", "--source", "system"])
        self.assertNotIn("--device", argv)
        self.assertNotIn("--pause-on-lid", argv)      # a lid close is covered by sleep handling

    def test_mic_source_is_unchanged(self):
        lec = self.lecture()
        mic = self.audio.Device(0, "MacBook Air Microphone")
        with mock.patch.object(self.recorder.capture, "portable", lambda: False), \
             mock.patch.object(self.recorder.pc, "IS_MAC", True), \
             mock.patch.object(self.recorder.capture, "available", lambda: Path("/x/lectern-capture")):
            argv = self.recorder.Recorder()._argv(lec, mic)
        self.assertEqual(argv[:3], ["/x/lectern-capture", "--device", "MacBook Air Microphone"])
        self.assertNotIn("--source", argv)
        self.assertIn("--pause-on-lid", argv)

    def test_never_falls_back_to_recording_the_mic(self):
        lec = self.lecture()
        # no native helper (not built / portable mode): ffmpeg or the portable
        # helper would record the microphone, so it must refuse instead
        with mock.patch.object(self.recorder.capture, "available", lambda: None):
            with self.assertRaises(self.audio.PreflightError):
                self.recorder.Recorder()._argv(lec, self.audio.SYSTEM_DEVICE)
        with mock.patch.object(self.recorder.capture, "portable", lambda: True):
            with self.assertRaises(self.audio.PreflightError):
                self.recorder.Recorder()._argv(lec, self.audio.SYSTEM_DEVICE)
            with self.assertRaises(self.audio.PreflightError):
                self.recorder._require_capture_helper(True)

    def test_status_names_the_source(self):
        self.assertEqual(self.audio.SYSTEM_DEVICE.name, "System audio")
        self.assertFalse(self.recorder._builtin_mic(self.audio.SYSTEM_DEVICE.name))


class Resume(Base):
    def test_a_paused_system_recording_resumes_as_system_audio(self):
        lec = self.storage.Lecture.create("Paused sys", state="paused")
        self.storage.atomic_write_json(lec.work / "recording.json", {"device_name": "System audio"})
        for p in self.mac():
            p.start(); self.addCleanup(p.stop)
        with mock.patch.object(self.audio, "list_audio_devices",
                               side_effect=AssertionError("must not enumerate microphones")):
            d = self.recorder.Recorder._resume_device(lec, "paused")
        self.assertTrue(self.audio.is_system(d))

    def test_a_paused_system_recording_does_not_become_a_mic_one(self):
        lec = self.storage.Lecture.create("Paused sys2", state="paused")
        self.storage.atomic_write_json(lec.work / "recording.json", {"device_name": "System audio"})
        with mock.patch.object(self.audio.pc, "IS_MAC", False):
            # unsupported now: take the slow, tested path (which will refuse or
            # use the default), never match a microphone by accident
            self.assertIsNone(self.recorder.Recorder._resume_device(lec, "paused"))


class ResumeSource(Base):
    """A finished lecture has no recording.json (.work is gone), so the source
    lives on the manifest. It must win over the saved default."""

    def resume_device(self, default=None, **manifest):
        lec = self.storage.Lecture.create("Done", state="complete")   # before "being a Mac"
        if manifest:
            lec.update(**manifest)
        if default:
            self.models.save_settings(record_source=default)
        p = mock.patch.object(self.audio, "system_audio_supported", lambda: True)
        p.start(); self.addCleanup(p.stop)     # not "IS_MAC": that also switches fsync to macOS-only calls
        got = {}
        mic = self.audio.Device(0, "USB Mic")
        rec = self.recorder.Recorder()
        with mock.patch.object(self.recorder, "_require_capture_helper", lambda *a, **k: None), \
             mock.patch.object(self.recorder.audio, "resolve_device", lambda fresh=False: mic), \
             mock.patch.object(self.recorder.audio, "probe", lambda d: got.update(probed=d)), \
             mock.patch.object(self.recorder.Recorder, "_seed_from_archive_if_needed", staticmethod(lambda l: 0)), \
             mock.patch.object(self.recorder.Recorder, "_spawn",
                               lambda self_, l, d, start_number=0: got.update(device=d) or (mock.Mock(), 0)), \
             mock.patch.object(self.recorder.Recorder, "_track", lambda *a, **k: {}):
            rec.resume(lec.id)
        return got["device"]

    def test_finished_system_lecture_resumes_as_system_even_if_default_is_mic(self):
        d = self.resume_device(capture_source="system")
        self.assertEqual(self.audio.default_source(), "mic")
        self.assertTrue(self.audio.is_system(d))

    def test_finished_mic_lecture_stays_mic_even_if_default_is_system(self):
        self.assertEqual(self.resume_device("system", capture_source="mic").name, "USB Mic")

    def test_imports_and_old_lectures_count_as_microphone(self):
        # an imported lecture keeps a dict under "source"; it must not be read as ours
        self.assertEqual(self.resume_device("system", source={"kind": "import", "filename": "a.mp4"}).name, "USB Mic")

    def test_start_records_the_source_on_the_lecture(self):
        lec = self.storage.Lecture.create("S")
        with mock.patch.object(self.recorder.Recorder, "_argv", lambda *a, **k: ["true"]), \
             mock.patch.object(self.recorder.subprocess, "Popen", return_value=mock.Mock(pid=1)):
            self.recorder.Recorder()._spawn(lec, self.audio.SYSTEM_DEVICE)
        self.assertEqual(lec.manifest()["capture_source"], "system")
        with mock.patch.object(self.recorder.Recorder, "_argv", lambda *a, **k: ["true"]), \
             mock.patch.object(self.recorder.subprocess, "Popen", return_value=mock.Mock(pid=1)):
            self.recorder.Recorder()._spawn(lec, self.audio.Device(0, "USB Mic"))
        self.assertEqual(lec.manifest()["capture_source"], "mic")


class Probe(Base):
    def test_silence_is_ok_for_system_audio_but_a_denied_permission_is_not(self):
        import subprocess, wave
        calls = []

        def fake_run(argv, **kw):
            calls.append(argv)
            out = argv[argv.index("--out") + 1] % 0
            with wave.open(out, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 24000)            # 1.5 s of digital silence
            return subprocess.CompletedProcess(argv, 0, "", "")

        with mock.patch.object(self.audio.capture, "available", lambda: Path("/x/h")), \
             mock.patch.object(self.audio.subprocess, "run", fake_run):
            res = self.audio.probe(self.audio.SYSTEM_DEVICE)
        self.assertTrue(res["silent"])
        self.assertEqual(calls[0][1:3], ["--source", "system"])

        def denied(argv, **kw):
            return subprocess.CompletedProcess(argv, 3, "", "Operation not permitted: ...")
        with mock.patch.object(self.audio.capture, "available", lambda: Path("/x/h")), \
             mock.patch.object(self.audio.subprocess, "run", denied):
            with self.assertRaises(self.audio.PreflightError) as e:
                self.audio.probe(self.audio.SYSTEM_DEVICE)
        self.assertEqual(e.exception.kind, "permission")
        self.assertIn("Screen & System Audio Recording", e.exception.detail)


class Api(Base):
    def setUp(self):
        super().setUp()
        self.server = importlib.import_module("lectern.server")
        self.client = self.server.app.test_client()

    def test_source_setting_round_trips_and_is_refused_where_unsupported(self):
        r = self.client.get("/api/source").get_json()
        self.assertEqual((r["source"], r["system_supported"]), ("mic", False))   # Linux CI
        bad = self.client.post("/api/source", json={"source": "system"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.client.post("/api/source", json={"source": "bogus"}).status_code, 400)
        for p in self.mac():
            p.start(); self.addCleanup(p.stop)
        ok = self.client.post("/api/source", json={"source": "system"}).get_json()
        self.assertEqual(ok["source"], "system")
        self.assertEqual(self.client.get("/api/source").get_json()["source"], "system")
        self.client.post("/api/source", json={"source": "mic"})
        self.assertEqual(self.client.get("/api/source").get_json()["source"], "mic")

    def test_start_passes_the_chosen_source_and_rejects_unknown_ones(self):
        seen = {}
        self.server.RECORDER.start = lambda *a, **k: seen.update(k) or {"recording": True}
        with mock.patch.object(self.server, "_MOVE_LOCK", mock.MagicMock()):
            self.client.post("/api/record/start", json={"title": "t", "source": "system"})
            self.assertEqual(seen["source"], "system")
            self.client.post("/api/record/start", json={"title": "t"})
            self.assertIsNone(seen["source"])                 # saved default
            r = self.client.post("/api/record/start", json={"title": "t", "source": "tv"})
            self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
