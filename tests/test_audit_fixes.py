"""Regressions for the 2026-10-30 audit: recording paths, Windows-safe names
and ids, busy-lecture guards, phone-mode access, recovery of the last step,
model checksums, one server per port, and transcript normalisation.
Run: python -m unittest discover -s tests"""
import importlib, json, os, shutil, socket, sys, tempfile, unittest, wave
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.test_release_fixes import Base  # noqa: E402


class RecordingPaths(Base):
    def test_percent_in_class_name_survives_the_segment_pattern(self):
        recorder = importlib.import_module("lectern.recorder")
        lec = self.storage.Lecture.create("Quiz review", folder="100% Math")
        pattern = recorder.Recorder._seg_pattern(lec)
        self.assertEqual(pattern % 3, str(lec.work / "seg-000003.wav"))

    def test_duration_ignores_an_unpatched_header(self):
        concat = importlib.import_module("lectern.concat")
        p = Path(self.tmp) / "seg.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"\0\0" * 16000)            # one second
        data = bytearray(p.read_bytes())
        data[4:8] = data[40:44] = b"\xff\xff\xff\xff"  # a writer killed mid-segment
        p.write_bytes(data)
        self.assertEqual(concat.wav_duration_ms(p), 1000)


class NamesAndIds(Base):
    def test_lecture_ids_that_could_escape_are_refused(self):
        for bad in ("x\\..\\..\\Documents", "C:\\foo", "../x", ".hidden", "a b"):
            with self.assertRaises(ValueError, msg=bad):
                self.storage.Lecture.load(bad)

    def test_folder_names_are_safe_on_windows(self):
        clean = self.storage.clean_folder
        self.assertEqual(clean('Why? "Intro" <CS>|*'), "Why Intro CS")
        self.assertEqual(clean("con"), "con_")
        self.assertEqual(clean("COM1.notes"), "COM1.notes_")
        self.assertEqual(clean("Console"), "Console")

    def test_case_only_rename_and_create(self):
        self.storage.create_folder("bio")
        self.assertEqual(self.storage.create_folder("BIO"), "bio")
        self.assertEqual(self.storage.rename_folder("bio", "Bio"), "Bio")
        self.assertEqual([f["name"] for f in self.storage.list_folders()], ["Bio"])


class BusyGuards(Base):
    def setUp(self):
        super().setUp()
        self.server = importlib.import_module("lectern.server")
        self.server._STARTUP_DONE.set()
        self.client = self.server.app.test_client()
        host = f"127.0.0.1:{self.config.PORT}"
        self.h = {"Host": host, "Origin": f"http://{host}"}

    def test_process_is_refused_while_importing(self):
        lec = self.storage.Lecture.create("Big video", state="importing")
        r = self.client.post(f"/api/lecture/{lec.id}/process", headers=self.h)
        self.assertEqual(r.status_code, 409)
        self.assertEqual(lec.manifest()["state"], "importing")

    def test_notes_need_a_transcript(self):
        lec = self.storage.Lecture.create("No words", state="failed")
        r = self.client.post(f"/api/lecture/{lec.id}/notes/regenerate", headers=self.h)
        self.assertEqual(r.status_code, 409)

    def test_delete_of_a_locked_lecture_is_a_clean_409(self):
        lec = self.storage.Lecture.create("Open in player", state="complete")
        with mock.patch.object(self.storage.Lecture, "trash",
                               side_effect=PermissionError("WinError 32")):
            r = self.client.delete(f"/api/lecture/{lec.id}", headers=self.h)
        self.assertEqual(r.status_code, 409)


class PhoneMode(Base):
    def setUp(self):
        super().setUp()
        self.config.HOST = "0.0.0.0"
        self.server = importlib.import_module("lectern.server")
        self.client = self.server.app.test_client()
        self.key = self.config.access_token()
        self.lan = {"environ_base": {"REMOTE_ADDR": "192.168.1.20"},
                    "headers": {"Host": f"192.168.1.5:{self.config.PORT}"}}

    def test_a_phone_needs_the_key(self):
        self.assertEqual(self.client.get("/api/lectures", **self.lan).status_code, 403)
        self.assertEqual(self.client.get("/api/lectures?k=wrong", **self.lan).status_code, 403)
        r = self.client.get(f"/api/lectures?k={self.key}", **self.lan)
        self.assertEqual(r.status_code, 200)
        self.assertIn(self.server.COOKIE + "=", r.headers.get("Set-Cookie", ""))
        self.assertEqual(self.client.get("/api/lectures", **self.lan).status_code, 200)

    def test_this_computer_needs_no_key(self):
        r = self.client.get("/api/status", headers={"Host": f"127.0.0.1:{self.config.PORT}"})
        self.assertEqual(r.status_code, 200)

    def test_a_rebinding_page_on_this_computer_still_needs_the_key(self):
        r = self.client.get("/api/lectures", headers={"Host": "evil.example"})
        self.assertEqual(r.status_code, 403)


class Recovery(Base):
    def setUp(self):
        super().setUp()
        self.recovery = importlib.import_module("lectern.recovery")

    def test_crash_after_releasing_work_finishes_the_lecture(self):
        lec = self.storage.Lecture.create("Almost done", state="archiving")
        lec.archive_path.write_bytes(b"m4a")
        shutil.rmtree(lec.work)
        self.recovery.run_startup_recovery(lambda _id: None)
        self.assertEqual(lec.manifest()["state"], "complete")

    def test_a_running_job_is_left_alone(self):
        lec = self.storage.Lecture.create("Being transcribed", state="transcribing")
        submitted = []
        self.recovery.run_startup_recovery(submitted.append, is_running=lambda _id: True)
        self.assertEqual(submitted, [])
        self.assertEqual(lec.manifest()["state"], "transcribing")


class ModelChecksum(Base):
    def test_a_damaged_download_is_not_installed(self):
        models = importlib.import_module("lectern.models")
        p = Path(self.tmp) / "m.bin.part"
        p.write_bytes(b"lmgg" + b"\0" * (1_000_000 - 4))
        good = models._sha256(p)
        m = models.Model("t", "m.bin", 1, 1, 1, "t", "t", sha256="0" * 64)
        with self.assertRaises(models.DownloadError):
            models._check_model_file(p, m)
        self.assertFalse(p.exists())
        p.write_bytes(b"lmgg" + b"\0" * (1_000_000 - 4))
        models._check_model_file(p, models.Model("t", "m.bin", 1, 1, 1, "t", "t", sha256=good))


class DownloadCertificates(Base):
    def test_downloads_trust_the_bundled_certificate_authorities(self):
        models = importlib.import_module("lectern.models")
        self.assertGreater(models._ssl_context().cert_store_stats()["x509_ca"], 50)


class OneServer(Base):
    def test_a_second_copy_does_not_start(self):
        main = importlib.import_module("lectern.__main__")
        held = socket.socket()
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        self.addCleanup(held.close)
        with mock.patch.object(self.config, "PORT", held.getsockname()[1]):
            self.assertIsNone(main._claim_port())


class Normalize(Base):
    def test_tokens_become_words_with_times(self):
        transcribe = importlib.import_module("lectern.transcribe")
        raw = Path(self.tmp) / "w.json"
        tok = lambda text, a, b: {"text": text, "offsets": {"from": a, "to": b}, "p": 0.9}
        raw.write_text(json.dumps({"transcription": [{"tokens": [
            tok("[_BEG_]", 0, 0), tok(" Mito", 0, 300), tok("chondria", 300, 700),
            tok(" are", 700, 900), tok(" small", 900, 1300), tok(".", 1300, 1350)]}]}))
        doc = transcribe.normalize(raw, 2000)
        self.assertEqual(doc["words"]["w"], ["Mitochondria", "are", "small."])
        self.assertEqual(doc["words"]["t0"][0], 0)
        self.assertEqual(doc["words"]["t0"][1], 700)


if __name__ == "__main__":
    unittest.main()
