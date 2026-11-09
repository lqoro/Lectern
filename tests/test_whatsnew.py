"""The "what's new" panel shows once per version, never on a brand-new install.
Run: python -m unittest discover -s tests"""
import importlib, os, shutil, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class WhatsNew(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        os.environ["LECTERN_ROOT"] = str(Path(tmp) / "Lectures")
        os.environ["LECTERN_SUPPORT"] = str(Path(tmp) / "support")
        for name in [m for m in sys.modules if m.startswith("lectern")]:
            del sys.modules[name]
        self.config = importlib.import_module("lectern.config")
        self.config.ensure_dirs()
        self.models = importlib.import_module("lectern.models")
        self.client = importlib.import_module("lectern.server").app.test_client()

    def get(self):
        return self.client.get("/api/whatsnew").get_json()

    def test_a_new_install_sees_nothing_and_is_marked_up_to_date(self):
        self.assertFalse(self.get()["show"])
        self.assertEqual(self.models.load_settings()["seen_version"], self.config.VERSION)

    def test_an_existing_user_sees_the_notes_once(self):
        self.models.save_settings(notice_accepted="1")      # used Lectern before the update
        self.assertTrue(self.get()["show"])
        self.assertTrue(self.client.post("/api/whatsnew/seen").get_json()["ok"])
        self.assertFalse(self.get()["show"])

    def test_the_next_version_shows_again(self):
        self.models.save_settings(notice_accepted="1", seen_version="0.0.1")
        self.assertTrue(self.get()["show"])

    def test_this_version_has_notes_to_show(self):
        notes = Path(__file__).resolve().parent.parent / "web" / "whatsnew" / f"{self.config.VERSION}.html"
        self.assertTrue(notes.is_file(), f"write {notes.name} before releasing")


if __name__ == "__main__":
    unittest.main()
