"""Regressions for the v0.1.0 review fixes: cross-site requests, trash age,
unique ids across classes, model-file validation, notes regenerate state.
Run: python -m unittest discover -s tests"""
import importlib, json, os, shutil, sys, tempfile, time, unittest
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
        self.storage = importlib.import_module("lectern.storage")


class CrossSite(Base):
    def setUp(self):
        super().setUp()
        self.server = importlib.import_module("lectern.server")
        self.client = self.server.app.test_client()
        self.port = self.config.PORT

    def test_foreign_host_is_refused(self):
        # DNS rebinding: an attacker's hostname pointed at 127.0.0.1
        r = self.client.get("/api/lectures", headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(r.status_code, 403)

    def test_cross_origin_post_is_refused(self):
        r = self.client.post("/api/record/stop", headers={
            "Host": f"127.0.0.1:{self.port}", "Origin": "https://evil.example"})
        self.assertEqual(r.status_code, 403)

    def test_cross_site_post_without_origin_is_refused(self):
        r = self.client.post("/api/record/stop", headers={
            "Host": f"127.0.0.1:{self.port}", "Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)

    def test_same_origin_post_is_allowed(self):
        host = f"127.0.0.1:{self.port}"
        r = self.client.post("/api/record/stop",
                             headers={"Host": host, "Origin": f"http://{host}"})
        self.assertNotEqual(r.status_code, 403)

    def test_non_dict_json_is_a_clean_error(self):
        host = f"127.0.0.1:{self.port}"
        r = self.client.post("/api/models/select", json=[1, 2],
                             headers={"Host": host, "Origin": f"http://{host}"})
        self.assertLess(r.status_code, 500)
        r = self.client.post("/api/models/download", json={"id": ["x"]},
                             headers={"Host": host, "Origin": f"http://{host}"})
        self.assertEqual(r.status_code, 404)


class Trash(Base):
    def test_old_lecture_is_not_purged_right_after_delete(self):
        lec = self.storage.Lecture.create("Old lecture", state="complete")
        (lec.path / "audio.m4a").write_bytes(b"audio")
        month_ago = time.time() - 30 * 86400
        os.utime(lec.path, (month_ago, month_ago))
        dest = lec.trash()
        self.assertLess(time.time() - dest.stat().st_mtime, 60)
        recovery = importlib.import_module("lectern.recovery")
        recovery.sweep_trash()
        self.assertTrue((dest / "audio.m4a").is_file())


class UniqueIds(Base):
    def test_same_title_same_minute_in_two_classes(self):
        a = self.storage.Lecture.create("Imported lecture", folder="BIO", state="complete")
        b = self.storage.Lecture.create("Imported lecture", folder="CHEM", state="complete")
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(self.storage.Lecture.load(b.id).path, b.path)


class ModelFile(Base):
    def test_html_page_is_never_installed(self):
        models = importlib.import_module("lectern.models")
        tiny = models.BY_ID["tiny.en"]
        part = Path(self.tmp) / "x.part"
        part.write_bytes(b"<html>" + b" " * (tiny.size_mb * 1_000_000))
        with self.assertRaises(models.DownloadError):
            models._check_model_file(part, tiny)
        self.assertFalse(part.exists())


class NotesRegenerate(Base):
    def test_state_is_restored_after_regenerate(self):
        pipeline = importlib.import_module("lectern.pipeline")
        lec = self.storage.Lecture.create("Done", state="complete")
        p = pipeline.Pipeline()
        job = mock.Mock(lecture_id=lec.id, percent=0, error="", done=False)

        def fake_notes(job, lec):
            lec.set_state("noting")

        with mock.patch.object(p, "_notes", fake_notes), \
             mock.patch.object(p, "_set", lambda *a, **k: None):
            p._run_notes(job)
        self.assertEqual(self.storage.Lecture.load(lec.id).manifest()["state"], "complete")


if __name__ == "__main__":
    unittest.main()


class QuickRecord(Base):
    def setUp(self):
        super().setUp()
        self.server = importlib.import_module("lectern.server")
        self.client = self.server.app.test_client()

    def test_auto_title_is_date_and_time(self):
        from datetime import datetime
        self.assertEqual(self.server.quick_title(datetime(2026, 10, 26, 14, 5)), "102626 2:05 PM")
        self.assertEqual(self.server.quick_title(datetime(2026, 1, 2, 0, 0)), "010226 12:00 AM")

    def test_toggle_starts_without_dialog_or_mic_test(self):
        calls = {}

        def start(title, course="", folder="", *, skip_probe=False):
            calls.update(title=title, skip_probe=skip_probe)
            return {"recording": True, "lecture_id": "x"}

        with mock.patch.object(self.server.RECORDER, "status", lambda: {"recording": False}), \
             mock.patch.object(self.server.RECORDER, "start", start):
            r = self.client.post("/api/record/toggle")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["action"], "started")
        self.assertTrue(calls["skip_probe"])
        self.assertRegex(calls["title"], r"^\d{6} \d{1,2}:\d{2} [AP]M$")

    def test_toggle_stops_when_recording(self):
        with mock.patch.object(self.server.RECORDER, "status", lambda: {"recording": True}), \
             mock.patch.object(self.server.RECORDER, "stop", lambda: {"lecture_id": "x", "stopped": True}), \
             mock.patch.object(self.server.PIPELINE, "submit", lambda i: None):
            r = self.client.post("/api/record/toggle")
        self.assertEqual(r.get_json()["action"], "stopped")

    def test_shortcut_default_and_validation(self):
        self.assertEqual(self.client.get("/api/shortcut").get_json(),
                         {"enabled": True, "key": "L", "mods": ["cmd"]})
        bad = [{"key": "L", "mods": []}, {"key": "L", "mods": ["shift"]},
               {"key": "LL", "mods": ["cmd"]}, {"key": "L", "mods": ["hyper"]}]
        for b in bad:
            self.assertEqual(self.client.post("/api/shortcut", json=b).status_code, 400, b)
        r = self.client.post("/api/shortcut", json={"key": "r", "mods": ["shift", "ctrl"]})
        self.assertEqual(r.get_json(), {"enabled": True, "key": "R", "mods": ["ctrl", "shift"]})
        self.assertEqual(self.client.get("/api/shortcut").get_json()["key"], "R")


class FastResume(Base):
    def test_paused_lecture_reuses_its_mic_without_a_test(self):
        recorder = importlib.import_module("lectern.recorder")
        audio = importlib.import_module("lectern.audio")
        lec = self.storage.Lecture.create("Paused", state="paused")
        self.storage.atomic_write_json(lec.work / "recording.json", {"device_name": "MacBook Air Microphone"})
        devs = [audio.Device(0, "BlackHole 2ch"), audio.Device(1, "MacBook Air Microphone")]
        with mock.patch.object(audio, "list_audio_devices", lambda use_cache=True: devs):
            d = recorder.Recorder._resume_device(lec, "paused")
            self.assertEqual(d.name, "MacBook Air Microphone")
            # a finished lecture, or a mic that is gone, takes the slow, tested path
            self.assertIsNone(recorder.Recorder._resume_device(lec, "complete"))
        with mock.patch.object(audio, "list_audio_devices", lambda use_cache=True: devs[:1]):
            self.assertIsNone(recorder.Recorder._resume_device(lec, "paused"))


class Recommendation(Base):
    def setUp(self):
        super().setUp()
        self.models = importlib.import_module("lectern.models")

    def hw(self, **kw):
        base = {"os": "Darwin", "accelerator": "metal", "ram_total_mb": 8192,
                "cpu_cores": 8, "device": {"name": "MacBook Air", "chip": "Apple M1"}}
        return {**base, **kw}

    def test_device_label_names_the_machine(self):
        self.assertEqual(self.models.device_label(self.hw()), "MacBook Air · Apple M1 · 8 GB memory")
        self.assertEqual(self.models.device_label({"device": {}}), "this computer")

    def test_recommendation_explains_itself_with_a_time_estimate(self):
        hw = self.hw()
        rec = self.models.recommend(hw)
        self.assertEqual(rec.id, "large-v3-turbo-q5_0")
        why = self.models.recommend_why(rec, hw)
        self.assertIn("Apple chip with 8 GB", why)
        self.assertIn(f"about {self.models.minutes_per_hour(rec, hw)} min", why)

    def test_weak_pc_gets_a_light_model_and_slower_estimates(self):
        weak = self.hw(os="Windows", accelerator="cpu", ram_total_mb=4096, cpu_cores=2,
                       device={"name": None, "chip": "Intel Celeron"})
        rec = self.models.recommend(weak)
        self.assertEqual(rec.id, "base.en")
        turbo = self.models.BY_ID["large-v3-turbo-q5_0"]
        self.assertGreater(self.models.minutes_per_hour(turbo, weak),
                           self.models.minutes_per_hour(turbo, self.hw()))

    def test_device_never_includes_serial_numbers(self):
        hardware = importlib.import_module("lectern.hardware")
        self.assertEqual(set(hardware.device()), {"name", "chip"})
