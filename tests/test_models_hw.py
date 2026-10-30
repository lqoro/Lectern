"""Model selection, download, hardware heuristics, FIFO stage lock, notes guards.
No network, microphone, ollama or whisper needed.   Run: python -m unittest discover -s tests"""
import dataclasses, hashlib, http.server, importlib, os, shutil, sys, tempfile, threading, time, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def fresh_modules(tmp, **env):
    """Import lectern.* against an isolated support/models dir."""
    os.environ["LECTERN_SUPPORT"] = str(Path(tmp) / "support")
    os.environ["LECTERN_ROOT"] = str(Path(tmp) / "Lectures")
    os.environ.pop("LECTERN_MODEL", None)
    os.environ.update(env)
    for name in [m for m in sys.modules if m.startswith("lectern")]:
        del sys.modules[name]
    return importlib.import_module("lectern.models")


def fake_model(models, model, where=None, short=False):
    d = Path(where or models.MODELS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    size = model.size_mb * 1_000_000 // (3 if short else 1)
    with open(d / model.file, "wb") as f:
        f.truncate(size)


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.m = fresh_modules(self.tmp, LECTERN_MODEL_DIR=str(Path(self.tmp) / "models"))
        self.m.LEGACY_DIR = Path(self.tmp) / "legacy"           # never touch the real ones
        self.m.BENCH_DIR = Path(self.tmp) / "bench"

    def test_nothing_installed(self):
        self.assertIsNone(self.m.active_path())
        self.assertEqual(self.m.active()[2], "none installed")

    def test_truncated_file_is_not_installed(self):
        fake_model(self.m, self.m.BY_ID["base.en"], short=True)
        self.assertEqual(self.m.installed(), [])

    def test_legacy_dir_still_found(self):
        fake_model(self.m, self.m.BY_ID["large-v3-turbo-q5_0"], where=self.m.LEGACY_DIR)
        self.assertEqual(self.m.active()[1].id, "large-v3-turbo-q5_0")

    def test_user_choice_beats_recommendation(self):
        for mid in ("base.en", "small.en"):
            fake_model(self.m, self.m.BY_ID[mid])
        self.m.save_settings(whisper_model="base.en")
        self.assertEqual(self.m.active()[1].id, "base.en")
        self.m.save_settings(whisper_model=None)
        self.assertIn(self.m.active()[1].id, ("base.en", "small.en"))

    def test_full_precision_file_counts_as_installed(self):
        small = self.m.BY_ID["small.en"]
        d = self.m.BENCH_DIR; d.mkdir(parents=True)
        with open(d / "ggml-small.en.bin", "wb") as f:
            f.truncate(488 * 1_000_000)
        self.assertEqual(self.m.installed_path(small).name, "ggml-small.en.bin")

    def test_env_override(self):
        os.environ["LECTERN_MODEL"] = "/some/file.bin"
        self.assertEqual(self.m.active()[0], Path("/some/file.bin"))

    def test_recommendations(self):
        r = lambda ram, accel, cores: self.m.recommend(
            {"ram_total_mb": ram, "accelerator": accel, "cpu_cores": cores}).id
        self.assertEqual(r(8192, "metal", 8), "large-v3-turbo-q5_0")
        self.assertEqual(r(4096, "metal", 8), "small.en")
        self.assertEqual(r(16384, "cpu", 8), "large-v3-turbo-q5_0")
        self.assertEqual(r(8192, "cpu", 8), "small.en")
        self.assertEqual(r(8192, "cpu", 2), "base.en")
        self.assertEqual(r(2048, "cpu", 2), "tiny.en")

    def test_windows_needs_twice_the_ram_of_a_mac(self):
        r = lambda os_name, gb, accel, cores: self.m.recommend({
            "os": os_name, "ram_total_mb": gb * 1024, "accelerator": accel, "cpu_cores": cores,
            "ram_effective_mb": self.m.hardware.effective_ram_bytes(gb * 2**30, os_name) // 2**20}).id
        # A 16 GB PC gets what an 8 GB Mac gets; an 8 GB PC what a 4 GB Mac gets.
        self.assertEqual(r("Windows", 16, "cpu", 8), r("Darwin", 8, "cpu", 8))
        self.assertEqual(r("Windows", 8, "cpu", 8), "base.en")
        self.assertEqual(r("Windows", 32, "cpu", 8), "large-v3-turbo-q5_0")
        self.assertEqual(r("Windows", 8, "cuda", 8), "small.en")
        self.assertEqual(r("Linux", 8, "cpu", 8), "small.en")
        self.assertEqual(self.m.hardware.effective_ram_bytes(8 * 2**30, "Darwin"), 8 * 2**30)
        self.assertIsNone(self.m.hardware.effective_ram_bytes(0, "Windows"))

    def test_big_model_does_not_fit_small_machine(self):
        self.assertFalse(self.m.fits(self.m.BY_ID["large-v3-q5_0"], 4 * 2**30))
        self.assertTrue(self.m.fits(self.m.BY_ID["base.en"], 4 * 2**30))


class Download(unittest.TestCase):
    def serve(self, payload):
        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(h):
                start = 0
                rng = h.headers.get("Range")
                if rng:
                    start = int(rng.split("=")[1].rstrip("-"))
                    h.send_response(206)
                else:
                    h.send_response(200)
                body = payload[start:]
                h.send_header("Content-Length", str(len(body)))
                h.end_headers()
                h.wfile.write(body)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        return f"http://127.0.0.1:{srv.server_port}"

    def wait(self, dl):
        for _ in range(200):
            if not dl.status().get("running"):
                return dl.status()
            time.sleep(0.05)
        self.fail("download did not finish")

    def test_download_and_resume(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        size = 32_000_000                      # tiny.en q5_1 catalogue size
        payload = b"lmgg" + b"x" * (size - 4)
        url = self.serve(payload)
        m = fresh_modules(tmp, LECTERN_MODEL_BASE_URL=url,
                          LECTERN_MODEL_DIR=str(Path(tmp) / "models"))
        m.LEGACY_DIR = Path(tmp) / "legacy"
        m.BENCH_DIR = Path(tmp) / "bench"
        # The fake file is not the real model: give the catalogue its checksum.
        tiny = m.BY_ID["tiny.en"] = dataclasses.replace(
            m.BY_ID["tiny.en"], sha256=hashlib.sha256(payload).hexdigest())
        # a previous interrupted attempt left the first 30 MB behind
        m.MODELS_DIR.mkdir(parents=True)
        (m.MODELS_DIR / (tiny.file + ".part")).write_bytes(b"lmgg" + b"x" * 29_999_996)
        m.DOWNLOADER.start("tiny.en")
        st = self.wait(m.DOWNLOADER)
        self.assertEqual(st.get("error"), "")
        self.assertTrue(st.get("done"))
        self.assertEqual((m.MODELS_DIR / tiny.file).stat().st_size, size)
        self.assertFalse((m.MODELS_DIR / (tiny.file + ".part")).exists())
        self.assertIsNotNone(m.installed_path(tiny))

    def test_short_download_is_never_installed(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        url = self.serve(b"x" * 1000)
        m = fresh_modules(tmp, LECTERN_MODEL_BASE_URL=url,
                          LECTERN_MODEL_DIR=str(Path(tmp) / "models"))
        m.LEGACY_DIR = Path(tmp) / "legacy"
        m.BENCH_DIR = Path(tmp) / "bench"
        # Content-Length (1000) matches what was sent, so this installs a tiny
        # file; the installed() size floor must still refuse to treat it as a model.
        m.DOWNLOADER.start("tiny.en")
        self.wait(m.DOWNLOADER)
        self.assertEqual(m.installed(), [])


class Threads(unittest.TestCase):
    def test_threads_bounds_and_override(self):
        hw = importlib.import_module("lectern.hardware")
        os.environ.pop("LECTERN_THREADS", None)
        self.assertTrue(1 <= hw.whisper_threads() <= 8)
        os.environ["LECTERN_THREADS"] = "3"
        self.assertEqual(hw.whisper_threads(), 3)
        os.environ.pop("LECTERN_THREADS")


class Fifo(unittest.TestCase):
    def test_turns_in_arrival_order(self):
        fresh_modules(tempfile.mkdtemp())
        pipeline = importlib.import_module("lectern.pipeline")
        lock, order = pipeline.FifoLock(), []
        with lock:                                   # hold it so everyone queues
            ts = []
            for i in range(6):
                t = threading.Thread(target=lambda i=i: (lock.__enter__(), order.append(i), lock.__exit__()))
                t.start(); ts.append(t); time.sleep(0.05)
        for t in ts:
            t.join(5)
        self.assertEqual(order, list(range(6)))


class NotesGuards(unittest.TestCase):
    def transcript(self, n=60):
        words, sents = [], []
        for s in range(n):
            w0 = len(words)
            words += [f"concept{s}", "explains", f"mechanism{s}", "regulating", f"enzyme{s%7}", "activity."]
            sents.append({"sid": s + 1, "w0": w0, "wn": 6, "t0": s * 10000, "t1": s * 10000 + 9000})
        return {"words": {"w": words}, "sents": sents, "duration_ms": n * 10000}

    def setUp(self):
        fresh_modules(tempfile.mkdtemp())
        self.notes = importlib.import_module("lectern.notes")

    def run_generate(self, total_gb, free_gb, mac):
        n = self.notes
        n.available = lambda: True
        n.hardware.total_ram_bytes = lambda: int(total_gb * 2**30)
        n.hardware.available_ram_bytes = lambda: int(free_gb * 2**30)
        n.hardware.pc.IS_MAC = mac
        called = []
        n._generate_llm = lambda *a, **k: called.append(1) or {"engine": "llm", "fallback": None}
        doc = n.generate(self.transcript())
        return called, doc

    def test_small_machine_skips_llm(self):
        called, doc = self.run_generate(4, 3, mac=False)
        self.assertEqual(called, [])
        self.assertIn("too little memory", doc["fallback"])

    def test_low_free_ram_skips_llm_off_mac(self):
        called, doc = self.run_generate(8, 1, mac=False)
        self.assertEqual(called, [])
        self.assertIn("not enough free memory", doc["fallback"])

    def test_windows_needs_16gb_for_the_notes_model(self):
        sysname = self.notes.hardware.platform.system
        self.notes.hardware.platform.system = lambda: "Windows"
        try:
            called, doc = self.run_generate(8, 4, mac=False)
            self.assertEqual(called, [])
            self.assertIn("too little memory", doc["fallback"])
            called, _ = self.run_generate(16, 4, mac=False)
            self.assertEqual(called, [1])
        finally:
            self.notes.hardware.platform.system = sysname

    def test_mac_with_low_reported_free_ram_still_uses_llm(self):
        called, _ = self.run_generate(8, 1, mac=True)     # the real 8 GB M1 case
        self.assertEqual(called, [1])

    def test_slow_model_falls_back_for_the_rest(self):
        n = self.notes
        n.available = lambda: True
        n.hardware.total_ram_bytes = lambda: 16 * 2**30
        n.hardware.available_ram_bytes = lambda: 16 * 2**30
        n.hardware.pc.IS_MAC = False
        n.CHUNK_TOKENS = 60                         # many small chunks
        n.NOTES_BUDGET_S = 1                        # absurdly small budget
        def slow(system, user, schema, **kw):
            time.sleep(0.6)
            sids = schema["properties"]["bullets"]["items"]["properties"]["cites"]["items"]["enum"] \
                if "bullets" in schema.get("properties", {}) else []
            return {"topic": "", "bullets": []} if sids else {"summary": "", "quiz": []}
        n._chat = slow
        doc = n.generate(self.transcript(120))
        self.assertIsNotNone(doc.get("partial_from_ms"))
        self.assertTrue(doc["sections"], "extractive tail should cover the remainder")
        self.assertTrue(all(s["t0"] >= doc["partial_from_ms"] for s in doc["sections"]))
        md = n.to_markdown(doc, "t")
        self.assertIn("too slow on this computer", md)


if __name__ == "__main__":
    unittest.main()
