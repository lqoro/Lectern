"""Parakeet-on-CoreML path: FluidAudio JSON -> whisper shape, and a CoreML model folder counts as installed.
No FluidAudio binary or model needed.   Run: python -m unittest discover -s tests"""
import hashlib, dataclasses, http.server, io, json, os, stat, sys, tempfile, threading, time, unittest, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_models_hw import fresh_modules, fake_model


class FluidAudio(unittest.TestCase):
    def test_conversion_feeds_normalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_modules(tmp)
            from lectern import transcribe
            src, dst = Path(tmp) / "fa.json", Path(tmp) / "out.json"
            src.write_text(json.dumps({"wordTimings": [
                {"word": "Uh", "startTime": 0.08, "endTime": 0.24, "confidence": 0.9},
                {"word": "ChatGPT", "startTime": 0.4, "endTime": 0.88, "confidence": 0.8},
                {"word": "works.", "startTime": 1.0, "endTime": 1.2, "confidence": 0.99}]}))
            transcribe._fluidaudio_to_whisper_json(src, dst)
            w = transcribe.normalize(dst, 2000)["words"]
            self.assertEqual(w["w"], ["ChatGPT", "works."])          # filler dropped
            self.assertEqual(w["t0"], [220, 820])                    # 400 and 1000 ms minus the 180 ms lead

    def test_empty_word_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            fresh_modules(tmp)
            from lectern import transcribe
            src, dst = Path(tmp) / "fa.json", Path(tmp) / "out.json"
            src.write_text(json.dumps({"wordTimings": None}))
            transcribe._fluidaudio_to_whisper_json(src, dst)
            self.assertEqual(json.loads(dst.read_text())["transcription"][0]["tokens"], [])

    def test_model_folder_is_installed_only_with_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = fresh_modules(tmp)
            m = models.BY_ID["parakeet-tdt-0.6b-v3-coreml"]
            d = models.MODELS_DIR / m.file
            d.mkdir(parents=True)
            self.assertIsNone(models.installed_path(m))
            (d / "config.json").write_text("{}")
            self.assertEqual(models.installed_path(m), d)


FAKE_CLI = """#!/bin/sh
# stands in for fluidaudiocli: records its arguments, writes the --output-json file
out=""; prev=""
for a in "$@"; do [ "$prev" = "--output-json" ] && out="$a"; prev="$a"; done
echo "$@" > "$FAKE_ARGS"
[ -n "$FAKE_FAIL" ] && { echo "ANE busy" >&2; exit 3; }
echo '{"wordTimings":[{"word":"Hello","startTime":0.5,"endTime":0.9,"confidence":0.9}]}' > "$out"
"""


def fake_cli(tmp):
    p = Path(tmp) / "fluidaudiocli"
    p.write_text(FAKE_CLI)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


class RunsParakeet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.models = fresh_modules(self.tmp, LECTERN_MODEL_DIR=str(Path(self.tmp) / "models"),
                                    LECTERN_FLUIDAUDIO=str(fake_cli(self.tmp)),
                                    FAKE_ARGS=str(Path(self.tmp) / "args.txt"))
        self.models.LEGACY_DIR = self.models.BENCH_DIR = Path(self.tmp) / "none"
        os.environ.pop("FAKE_FAIL", None)
        self.pk = self.models.MODELS_DIR / "parakeet-tdt-0.6b-v3-coreml"
        self.pk.mkdir(parents=True)
        (self.pk / "config.json").write_text("{}")
        self.wav = Path(self.tmp) / "a.wav"
        self.wav.write_bytes(b"RIFF")
        self.transcribe = __import__("importlib").import_module("lectern.transcribe")

    def test_builds_the_command_and_cleans_up(self):
        out = self.transcribe.run_whisper(self.wav, Path(self.tmp) / "w.raw")
        args = (Path(self.tmp) / "args.txt").read_text().split()
        self.assertEqual(args[0], "transcribe")
        self.assertIn("--local-model-dir", args)
        self.assertIn(str(self.pk), args)
        self.assertIn("--word-timestamps", args)
        self.assertEqual(self.transcribe.normalize(out, 2000)["words"]["w"], ["Hello"])
        self.assertFalse((Path(self.tmp) / "w.raw.fluidaudio.json").exists())

    def test_failure_names_the_program_and_keeps_its_message(self):
        os.environ["FAKE_FAIL"] = "1"
        with self.assertRaises(self.transcribe.TranscriptionError) as cm:
            self.transcribe.run_whisper(self.wav, Path(self.tmp) / "w.raw")
        self.assertIn("fluidaudiocli exited 3", str(cm.exception))
        self.assertIn("ANE busy", str(cm.exception))

    def test_failed_parakeet_falls_back_to_an_installed_whisper_model(self):
        os.environ["FAKE_FAIL"] = "1"
        small = self.models.BY_ID["small.en"]
        fake_model(self.models, small)
        calls = []
        real = self.transcribe.run_whisper
        def spy(wav, prefix, *, progress_cb=None, model=None):
            calls.append(model)
            if model is None:
                return real(wav, prefix, progress_cb=progress_cb)
            Path(str(prefix) + ".json").write_text("{}")
            return Path(str(prefix) + ".json")
        self.transcribe.run_whisper = spy
        out, used = self.transcribe.run_with_fallback(self.wav, Path(self.tmp) / "w.raw")
        self.assertEqual(used, "small.en")
        self.assertEqual(calls[1], self.models.installed_path(small))

    def test_no_fallback_without_a_whisper_model(self):
        os.environ["FAKE_FAIL"] = "1"
        with self.assertRaises(self.transcribe.TranscriptionError):
            self.transcribe.run_with_fallback(self.wav, Path(self.tmp) / "w.raw")

    def test_unexpected_json_shape_is_a_clear_error(self):
        src, dst = Path(self.tmp) / "x.json", Path(self.tmp) / "y.json"
        src.write_text(json.dumps({"wordTimings": [{"text": "hi"}]}))
        with self.assertRaises(self.transcribe.TranscriptionError):
            self.transcribe._fluidaudio_to_whisper_json(src, dst)


class UpgradeOffer(unittest.TestCase):
    def test_offered_only_to_someone_with_another_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = fresh_modules(tmp, LECTERN_MODEL_DIR=str(Path(tmp) / "models"))
            m.LEGACY_DIR = m.BENCH_DIR = Path(tmp) / "none"
            q4 = m.BY_ID["large-v3-turbo-q4_0"]
            m._recommended = lambda: q4
            self.assertIsNone(m.upgrade_offer())                    # nothing installed: first-run banner instead
            fake_model(m, m.BY_ID["small.en"])
            self.assertEqual(m.upgrade_offer(), q4.id)
            (m.MODELS_DIR / q4.file).write_bytes(b"x")              # the file alone is not the bundle
            self.assertEqual(m.upgrade_offer(), q4.id)
            enc = m.MODELS_DIR / "ggml-large-v3-turbo-encoder.mlmodelc"
            enc.mkdir()
            (enc / "model.mil").write_text("x")
            self.assertIsNone(m.upgrade_offer())                    # already has both parts
            m._recommended = lambda: m.BY_ID["large-v3-turbo-q5_0"]
            (enc / "model.mil").unlink()
            self.assertIsNone(m.upgrade_offer())                    # recommendation is an ordinary file


def zip_bytes(files, folder="parakeet-tdt-0.6b-v3-coreml"):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, data in files.items():
            z.writestr(f"{folder}/{name}" if folder else name, data)
    return b.getvalue()


class DownloadsParakeet(unittest.TestCase):
    def serve(self, payload):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(h):
                h.send_response(200)
                h.send_header("Content-Length", str(len(payload)))
                h.end_headers()
                h.wfile.write(payload)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_port}"

    def run_download(self, payload, model_id="parakeet-tdt-0.6b-v3-coreml"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        m = fresh_modules(tmp, LECTERN_MODEL_BASE_URL=self.serve(payload),
                          LECTERN_MODEL_DIR=str(Path(tmp) / "models"))
        m.LEGACY_DIR = m.BENCH_DIR = Path(tmp) / "none"
        pk = m.BY_ID[model_id] = dataclasses.replace(
            m.BY_ID[model_id], size_mb=len(payload) / 1e6,
            sha256=hashlib.sha256(payload).hexdigest())
        m.DOWNLOADER.start(pk.id)
        for _ in range(200):
            if not m.DOWNLOADER.status().get("running"):
                break
            time.sleep(0.05)
        return m, pk

    def test_zip_is_unpacked_into_a_model_folder(self):
        m, pk = self.run_download(zip_bytes({"config.json": "{}", "Encoder_v2.mlmodelc/w.bin": "x" * 50}))
        self.assertEqual(m.DOWNLOADER.status().get("error"), "")
        self.assertEqual(m.installed_path(pk), m.MODELS_DIR / pk.file)
        self.assertTrue((m.MODELS_DIR / pk.file / "Encoder_v2.mlmodelc" / "w.bin").is_file())
        self.assertEqual(sorted(p.name for p in m.MODELS_DIR.iterdir()), [pk.file])   # no .part, no .unzipping

    def test_zip_without_config_is_not_installed(self):
        m, pk = self.run_download(zip_bytes({"other.txt": "hi"}))
        self.assertIn("missing files", m.DOWNLOADER.status()["error"])
        self.assertIsNone(m.installed_path(pk))

    def test_zip_that_climbs_out_of_the_folder_is_refused(self):
        m, pk = self.run_download(zip_bytes({"../../evil.txt": "x", "config.json": "{}"}))
        self.assertIn("unsafe", m.DOWNLOADER.status()["error"])
        self.assertFalse((m.MODELS_DIR.parent / "evil.txt").exists())
        self.assertIsNone(m.installed_path(pk))

    def test_q4_bundle_puts_the_file_and_encoder_side_by_side(self):
        # whisper.cpp finds the encoder by name next to the model, so both must land at the top level.
        enc = "ggml-large-v3-turbo-encoder.mlmodelc"
        m, q4 = self.run_download(zip_bytes({"ggml-large-v3-turbo-q4_0.bin": "b" * 100,
                                             f"{enc}/model.mil": "m", f"{enc}/weights/weight.bin": "w"},
                                            folder=None), "large-v3-turbo-q4_0")
        self.assertEqual(m.DOWNLOADER.status().get("error"), "")
        self.assertEqual(m.installed_path(q4), m.MODELS_DIR / q4.file)
        self.assertTrue((m.MODELS_DIR / enc / "weights" / "weight.bin").is_file())
        self.assertEqual(sorted(p.name for p in m.MODELS_DIR.iterdir()), [enc, q4.file])

    def test_q4_bundle_without_the_encoder_is_not_installed(self):
        m, q4 = self.run_download(zip_bytes({"ggml-large-v3-turbo-q4_0.bin": "b" * 100}, folder=None),
                                  "large-v3-turbo-q4_0")
        self.assertIn("missing files", m.DOWNLOADER.status()["error"])
        self.assertIsNone(m.installed_path(q4))


if __name__ == "__main__":
    unittest.main()
