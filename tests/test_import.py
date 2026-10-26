"""File import: extraction, rejection, upload endpoint, restart recovery.
Needs ffmpeg/ffprobe; no microphone, whisper or network.
Run: python -m unittest discover -s tests"""
import importlib, io, json, os, shutil, subprocess, sys, tempfile, unittest, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
HAVE_FFMPEG = Path(FFMPEG).exists() or bool(shutil.which("ffmpeg"))


def fresh(tmp):
    os.environ["LECTERN_ROOT"] = str(Path(tmp) / "Lectures")
    os.environ["LECTERN_SUPPORT"] = str(Path(tmp) / "support")
    for name in [m for m in sys.modules if m.startswith("lectern")]:
        del sys.modules[name]
    config = importlib.import_module("lectern.config")
    config.ensure_dirs()
    return config


def make_video(path, secs, audio=True):
    """A 48 kHz stereo AAC track in an mp4 with a tiny video stream: the shape
    of a typical downloaded lecture, nothing like the 16 kHz mono we keep."""
    argv = [FFMPEG, "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s=64x64:d={secs}"]
    if audio:
        argv += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={secs}",
                 "-ac", "2", "-c:a", "aac", "-shortest"]
    argv += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(argv, check=True)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Importing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        fresh(self.tmp)
        self.storage = importlib.import_module("lectern.storage")
        self.importer = importlib.import_module("lectern.importer")

    def new_import(self, src):
        lec = self.storage.Lecture.create("t", state="importing",
                                          source={"kind": "import"})
        shutil.copy(src, self.importer.source_path(lec))
        return lec

    def test_extract_is_16k_mono_and_full_length(self):
        src = Path(self.tmp) / "lec.mp4"
        make_video(src, 12)
        lec = self.new_import(src)
        info = self.importer.probe(self.importer.source_path(lec))
        self.assertAlmostEqual(info["duration_s"], 12, delta=0.2)
        pcts = []
        seg = self.importer.extract(lec, duration_s=info["duration_s"], progress_cb=pcts.append)
        with wave.open(str(seg)) as w:
            self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))
            self.assertAlmostEqual(w.getnframes(), 12 * 16000, delta=16000 * 0.05)
        self.assertEqual(seg.stat().st_size, 44 + 2 * w.getnframes())   # 44-byte header
        self.assertEqual(lec.segments(), [seg])                           # pipeline sees it
        self.assertFalse((lec.work / self.importer.PART_NAME).exists())

    def test_video_without_sound_is_rejected(self):
        src = Path(self.tmp) / "silent.mp4"
        make_video(src, 2, audio=False)
        with self.assertRaisesRegex(self.importer.ImportError_, "no sound track"):
            self.importer.probe(src)

    def test_garbage_is_rejected(self):
        src = Path(self.tmp) / "notes.pdf"
        src.write_bytes(b"%PDF-1.4 not a video")
        with self.assertRaises(self.importer.ImportError_):
            self.importer.probe(src)

    def test_interrupted_import_restarts_on_startup(self):
        src = Path(self.tmp) / "lec.mp4"
        make_video(src, 2)
        lec = self.new_import(src)
        orphan = self.storage.Lecture.create("u", state="importing")   # upload never finished
        recovery = importlib.import_module("lectern.recovery")
        normal, imports = [], []
        recovery.run_startup_recovery(normal.append, imports.append)
        self.assertEqual(imports, [lec.id])
        self.assertEqual(normal, [])
        self.assertEqual(orphan.manifest()["state"], "failed")


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
class Endpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        fresh(self.tmp)
        self.server = importlib.import_module("lectern.server")
        self.server._STARTUP_DONE.set()
        self.submitted = []
        self.server.PIPELINE.submit_import = lambda lid: self.submitted.append(lid) or {"stage": "queued"}
        self.client = self.server.app.test_client()

    def post(self, data, **headers):
        return self.client.post("/api/import?title=Week%208&folder=BIO&filename=w8.mp4",
                                data=data, headers={"content-type": "application/octet-stream", **headers})

    def test_upload_queues_an_import(self):
        src = Path(self.tmp) / "lec.mp4"
        make_video(src, 3)
        r = self.post(src.read_bytes())
        self.assertEqual(r.status_code, 200, r.get_json())
        lid = r.get_json()["lecture_id"]
        self.assertEqual(self.submitted, [lid])
        Lecture = importlib.import_module("lectern.storage").Lecture
        lec = Lecture.load(lid)
        self.assertEqual(lec.folder, "BIO")
        m = lec.manifest()
        self.assertEqual((m["title"], m["state"], m["source"]["filename"]), ("Week 8", "importing", "w8.mp4"))
        self.assertEqual(Path(lec.work / "import-source").read_bytes(), src.read_bytes())

    def test_non_media_never_reaches_the_pipeline(self):
        r = self.post(b"hello, not a video")
        self.assertEqual(r.status_code, 415)
        self.assertEqual(self.submitted, [])

    def test_truncated_upload_never_reaches_the_pipeline(self):
        # Claims 1 MB, sends 10 bytes: the client went away mid-upload.
        r = self.client.post("/api/import?title=x", input_stream=io.BytesIO(b"0123456789"),
                             headers={"content-type": "application/octet-stream"},
                             environ_overrides={"CONTENT_LENGTH": str(1 << 20)})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.submitted, [])


if __name__ == "__main__":
    unittest.main()
