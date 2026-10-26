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
