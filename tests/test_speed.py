"""Everyday speed without weaker safety: pipeline order and locks, preflight
reuse, startup caches, request-path caches, and the small leak/log fixes.
No microphone, whisper, ollama or network.   Run: python -m unittest discover -s tests"""
import gc, importlib, json, os, shutil, subprocess, sys, tempfile, threading, time, unittest, warnings, wave
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
        self.Lecture = self.storage.Lecture


def write_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))


# ------------------------------------------------------------------ pipeline
GB = 1024 ** 3


class PipelineOrder(Base):
    RAM = 32 * GB                        # room for whisper and the notes model

    def setUp(self):
        super().setUp()
        self.pipeline = importlib.import_module("lectern.pipeline")
        self.p = self.pipeline.Pipeline()
        ram = mock.patch.object(self.pipeline.hardware, "effective_ram_bytes",
                                lambda *a, **k: self.RAM)
        ram.start()
        self.addCleanup(ram.stop)

    def wait_done(self, *ids, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            if all((self.p.job(i) or {}).get("done") for i in ids):
                return
            time.sleep(0.01)
        self.fail(f"jobs did not finish: {[self.p.job(i) for i in ids]}")

    def test_archive_runs_before_notes(self):
        lec = self.Lecture.create("Order", state="paused")
        order = []
        stages = {name: (lambda name: lambda job, lec: order.append(name))(name)
                  for name in ("_concat", "_transcribe", "_archive", "_notes")}
        with mock.patch.multiple(self.p, **stages):
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertEqual(order, ["_concat", "_transcribe", "_archive", "_notes"])
        self.assertEqual(self.Lecture.load(lec.id).manifest()["state"], "complete")

    def test_notes_do_not_hold_up_the_next_lecture(self):
        a = self.Lecture.create("A", state="paused")
        b = self.Lecture.create("B", state="paused")
        release = threading.Event()
        b_archived = threading.Event()
        noop = lambda job, lec: None

        def notes(job, lec):
            if lec.id == a.id:
                release.wait(10)

        def archive(job, lec):
            if lec.id == b.id:
                b_archived.set()

        with mock.patch.multiple(self.p, _concat=noop, _transcribe=noop,
                                 _archive=archive, _notes=notes):
            self.p.submit(a.id)
            time.sleep(0.1)
            self.p.submit(b.id)
            # B is transcribed and archived while A is still writing notes.
            self.assertTrue(b_archived.wait(5))
            self.assertFalse(self.p.job(a.id)["done"])
            release.set()
            self.wait_done(a.id, b.id)

    def test_rewrite_notes_does_not_wait_for_another_transcription(self):
        a = self.Lecture.create("A", state="paused")
        b = self.Lecture.create("B", state="complete")
        b.transcript_path.write_text("{}")
        release = threading.Event()
        noop = lambda job, lec: None
        with mock.patch.multiple(self.p, _concat=noop, _archive=noop, _notes=noop,
                                 _transcribe=lambda job, lec: release.wait(10)):
            self.p.submit(a.id)
            time.sleep(0.1)
            self.p.submit_notes(b.id)
            self.wait_done(b.id, timeout=5)
            self.assertFalse(self.p.job(a.id)["done"])
            release.set()
            self.wait_done(a.id)

    def test_only_one_transcription_at_a_time(self):
        lecs = [self.Lecture.create(f"L{i}", state="paused") for i in range(3)]
        live, peak, lock = [0], [0], threading.Lock()
        noop = lambda job, lec: None

        def transcribe(job, lec):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1

        with mock.patch.multiple(self.p, _concat=noop, _archive=noop, _notes=noop,
                                 _transcribe=transcribe):
            for l in lecs:
                self.p.submit(l.id)
            self.wait_done(*[l.id for l in lecs])
        self.assertEqual(peak[0], 1)

    def test_a_failure_to_record_failure_is_logged(self):
        lec = self.Lecture.create("Broken", state="paused")

        def concat(job, l):
            raise RuntimeError("disk on fire")

        real = self.Lecture.set_state

        def set_state(self_, state, **kw):
            if state == "failed":
                raise OSError("read-only disk")
            return real(self_, state, **kw)

        with mock.patch.object(self.p, "_concat", concat), \
             mock.patch.object(self.Lecture, "set_state", set_state), \
             self.assertLogs("lectern.pipeline", "ERROR") as logs:
            self.p.submit(lec.id)
            self.wait_done(lec.id)
        self.assertTrue(any("failed" in m for m in logs.output))


class SmallMachine(PipelineOrder):
    """Below the threshold whisper and the notes model never run together."""
    RAM = 8 * GB

    def test_threshold_and_unknown_ram(self):
        pl = self.pipeline
        for ram, shared in ((8 * GB, False), (pl.SHARED_RAM_MIN_BYTES - 1, False),
                            (pl.SHARED_RAM_MIN_BYTES, True), (16 * GB, True),
                            (None, False), (0, False)):
            with mock.patch.object(pl.hardware, "effective_ram_bytes", lambda *a, **k: ram):
                self.assertEqual(pl.notes_share_ram(), shared, ram)

        def boom(*a, **k):
            raise OSError("sysctl failed")
        with mock.patch.object(pl.hardware, "effective_ram_bytes", boom), \
             self.assertLogs("lectern.pipeline", "ERROR"):
            self.assertFalse(pl.notes_share_ram())

    def test_notes_hold_up_the_next_transcription(self):
        a = self.Lecture.create("A", state="paused")
        b = self.Lecture.create("B", state="paused")
        in_notes, release, b_transcribed = threading.Event(), threading.Event(), threading.Event()
        noop = lambda job, lec: None

        def notes(job, lec):
            if lec.id == a.id:
                in_notes.set()
                release.wait(10)

        def transcribe(job, lec):
            if lec.id == b.id:
                b_transcribed.set()

        with mock.patch.multiple(self.p, _concat=noop, _transcribe=transcribe,
                                 _archive=noop, _notes=notes):
            self.p.submit(a.id)
            self.assertTrue(in_notes.wait(5))
            self.p.submit(b.id)
            self.assertFalse(b_transcribed.wait(0.3))   # waits for A's notes
            release.set()
            self.wait_done(a.id, b.id)
        self.assertTrue(b_transcribed.is_set())

    def test_rewrite_notes_waits_for_another_transcription(self):
        a = self.Lecture.create("A", state="paused")
        b = self.Lecture.create("B", state="complete")
        b.transcript_path.write_text("{}")
        in_whisper, release, b_noted = threading.Event(), threading.Event(), threading.Event()
        noop = lambda job, lec: None

        def transcribe(job, lec):
            in_whisper.set()
            release.wait(10)

        def notes(job, lec):
            if lec.id == b.id:
                b_noted.set()

        with mock.patch.multiple(self.p, _concat=noop, _archive=noop,
                                 _transcribe=transcribe, _notes=notes):
            self.p.submit(a.id)
            self.assertTrue(in_whisper.wait(5))
            self.p.submit_notes(b.id)
            self.assertFalse(b_noted.wait(0.3))         # waits for A's whisper
            release.set()
            self.wait_done(a.id, b.id)
        self.assertTrue(b_noted.is_set())
        self.assertEqual(self.Lecture.load(b.id).manifest()["state"], "complete")

    # The big-machine expectations do not apply here.
    test_notes_do_not_hold_up_the_next_lecture = None
    test_rewrite_notes_does_not_wait_for_another_transcription = None


class NotesRecovery(Base):
    """Startup recovery for the archive-before-notes order."""

    def setUp(self):
        super().setUp()
        self.recovery = importlib.import_module("lectern.recovery")

    def archived(self, title, *states):
        lec = self.Lecture.create(title, state="paused")
        for s in states:
            lec.set_state(s)
        lec.archive_path.write_bytes(b"aac")
        lec.transcript_path.write_text("{}")
        (lec.work / "stage").unlink(missing_ok=True)
        for p in lec.work.iterdir():
            p.unlink()
        lec.work.rmdir()                 # archive installed and .work released
        return lec

    def run_recovery(self):
        full, notes = [], []
        self.recovery.run_startup_recovery(full.append, None, lambda _id: False,
                                           submit_notes=notes.append)
        return full, notes

    def test_crash_while_writing_notes_after_archive_requeues_notes(self):
        lec = self.archived("n", "concatenating", "transcribing", "archiving", "noting")
        full, notes = self.run_recovery()
        self.assertEqual((full, notes), ([], [lec.id]))
        self.assertEqual(self.Lecture.load(lec.id).manifest()["state"], "complete")

    def test_crash_between_archive_and_notes_requeues_notes(self):
        lec = self.archived("a", "concatenating", "transcribing", "archiving")
        full, notes = self.run_recovery()
        self.assertEqual((full, notes), ([], [lec.id]))
        self.assertEqual(self.Lecture.load(lec.id).manifest()["state"], "complete")

    def test_old_order_archiving_crash_is_simply_finished(self):
        lec = self.archived("o", "concatenating", "transcribing", "noting", "archiving")
        full, notes = self.run_recovery()
        self.assertEqual((full, notes), ([], []))
        self.assertEqual(self.Lecture.load(lec.id).manifest()["state"], "complete")

    def test_crash_during_a_rewrite_restores_the_lecture(self):
        lec = self.archived("r", "complete", "noting")
        full, notes = self.run_recovery()
        self.assertEqual(full, [])
        self.assertEqual(self.Lecture.load(lec.id).manifest()["state"], "complete")
        failed = self.archived("f", "failed", "noting")
        self.run_recovery()
        self.assertEqual(self.Lecture.load(failed.id).manifest()["state"], "failed")

    def test_old_order_notes_crash_with_unarchived_audio_reruns_everything(self):
        # Notes used to run before archive: .work still holds the only audio.
        lec = self.Lecture.create("old", state="paused")
        write_wav(lec.work / "seg-000000.wav", 6)
        for s in ("concatenating", "transcribing", "noting"):
            lec.set_state(s)
        lec.archive_path.write_bytes(b"an older archive")     # e.g. a resumed lecture
        full, notes = self.run_recovery()
        self.assertEqual((full, notes), ([lec.id], []))
        self.assertTrue((self.Lecture.load(lec.id).work / "seg-000000.wav").is_file())


# --------------------------------------------------------- preflight reuse
class PreflightReuse(Base):
    def setUp(self):
        super().setUp()
        self.audio = importlib.import_module("lectern.audio")
        self.recorder = importlib.import_module("lectern.recorder")
        self.server = importlib.import_module("lectern.server")
        self.devs = [self.audio.Device(0, "MacBook Air Microphone")]
        self.lists, self.probes = [], []
        A = self.audio

        def list_devices(timeout=30.0, use_cache=True):
            self.lists.append(use_cache)
            return self.devs

        def probe(device, seconds=1.8):
            self.probes.append(device)
            return {"rms": 100, "peak": 900, "frames": 28800, "seconds": 1.8}

        R = self.recorder.Recorder
        for target, name, value in (
                (A, "list_audio_devices", list_devices), (A, "_probe", probe),
                (self.recorder, "_require_capture_helper", lambda: None),
                (self.storage, "free_bytes", lambda: 10 ** 12),
                (R, "_spawn", lambda self_, lec, dev, start_number=0: (None, 0)),
                (R, "_track", lambda self_, lec, proc, dev, off: {"device": dev.name})):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)

    def preflight(self):
        r = self.server.app.test_client().post("/api/preflight").get_json()
        self.assertTrue(r["ok"])

    def test_start_right_after_preflight_skips_the_second_listing_and_probe(self):
        self.preflight()
        self.assertEqual(len(self.probes), 1)
        self.lists.clear()
        self.recorder.Recorder().start("t")
        self.assertEqual(len(self.probes), 1)                 # no second test clip
        self.assertNotIn(False, self.lists)                   # no forced fresh listing

    def test_an_old_preflight_is_not_reused(self):
        self.preflight()
        with mock.patch.object(self.audio, "PREFLIGHT_REUSE_S", 0):
            self.recorder.Recorder().start("t")
        self.assertEqual(len(self.probes), 2)
        self.assertIn(False, self.lists)

    def test_a_different_device_is_tested_again(self):
        self.preflight()
        self.devs = [self.audio.Device(0, "USB Mic"), self.audio.Device(1, "MacBook Air Microphone")]
        self.audio._cache = None
        self.recorder.Recorder().start("t")
        self.assertEqual(len(self.probes), 2)

    def test_a_preflight_is_used_once(self):
        self.preflight()
        self.recorder.Recorder().start("t")
        self.recorder.Recorder().start("t2")
        self.assertEqual(len(self.probes), 2)

    def test_failed_preflight_is_not_reused(self):
        self.preflight()

        def bad(device, seconds=1.8):
            raise self.audio.PreflightError("silent", "zeros")
        with mock.patch.object(self.audio, "_probe", bad):
            r = self.server.app.test_client().post("/api/preflight").get_json()
        self.assertFalse(r["ok"])
        self.assertIsNone(self.audio.recent_preflight())

    def test_hotkey_start_still_lists_fresh(self):
        self.preflight()
        self.lists.clear()
        self.recorder.Recorder().start("t", skip_probe=True)
        self.assertEqual(self.lists, [False])
        self.assertEqual(len(self.probes), 1)


# ------------------------------------------------------------------ startup
class BootSeconds(Base):
    def test_sysctl_runs_once_per_process(self):
        pc = importlib.import_module("lectern.platform_compat")
        calls = []
        with mock.patch.object(pc, "boot_seconds", lambda: calls.append(1) or 1234):
            self.assertEqual([self.storage.boot_seconds() for _ in range(3)], [1234] * 3)
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.storage.boot_seconds(fresh=True), 1234)
            self.assertEqual(len(calls), 2)

    def test_unknown_boot_time_is_not_cached(self):
        pc = importlib.import_module("lectern.platform_compat")
        calls = []
        with mock.patch.object(pc, "boot_seconds", lambda: calls.append(1) or 0):
            self.storage.boot_seconds()
            self.storage.boot_seconds()
        self.assertEqual(len(calls), 2)


class TrashSweep(Base):
    def setUp(self):
        super().setUp()
        self.recovery = importlib.import_module("lectern.recovery")
        self.archive = importlib.import_module("lectern.archive")
        lec = self.Lecture.create("Swept", state="complete")
        lec.archive_path.write_bytes(b"aac" * 100)
        self.lec = lec
        self.entry = self.config.TRASH_DIR / f"{lec.id}__work"
        self.entry.mkdir()
        (self.entry / "seg-000000.wav").write_bytes(b"x")
        self.decodes = []
        p = mock.patch.object(self.archive, "_ffprobe",
                              lambda path: {"streams": [{"codec_name": "aac"}]})
        p.start()
        self.addCleanup(p.stop)

    def decoder(self, ok):
        def run(argv, **kw):
            self.decodes.append(argv)
            return subprocess.CompletedProcess(argv, 0 if ok else 1, "",
                                               "" if ok else "corrupt packet")
        return mock.patch.object(self.recovery.subprocess, "run", run)

    def test_decode_runs_at_low_priority(self):
        pc = importlib.import_module("lectern.platform_compat")
        with self.decoder(True):
            self.recovery.sweep_trash()
        self.assertEqual(self.decodes[0][0], pc.low_priority(["ffmpeg"])[0][0])
        self.assertFalse(self.entry.exists())

    def test_a_failed_entry_is_not_decoded_again_until_it_changes(self):
        with self.decoder(False):
            self.recovery.sweep_trash()
            self.recovery.sweep_trash()
        self.assertEqual(len(self.decodes), 1)
        self.assertTrue((self.entry / "seg-000000.wav").is_file())   # never deleted
        # The archive is replaced (e.g. a re-archive): verify it again.
        time.sleep(0.01)
        self.lec.archive_path.write_bytes(b"aac" * 101)
        with self.decoder(True):
            self.recovery.sweep_trash()
        self.assertEqual(len(self.decodes), 2)
        self.assertFalse(self.entry.exists())

    def test_the_failure_memory_survives_a_restart(self):
        with self.decoder(False):
            self.recovery.sweep_trash()
        fresh(self.tmp)
        recovery = importlib.import_module("lectern.recovery")
        archive = importlib.import_module("lectern.archive")
        runs = []
        with mock.patch.object(archive, "_ffprobe", lambda p: {"streams": [{"codec_name": "aac"}]}), \
             mock.patch.object(recovery.subprocess, "run", lambda a, **k: runs.append(a)):
            res = recovery.sweep_trash()
        self.assertEqual((runs, res["dirs_kept"]), ([], 1))
        self.assertTrue(self.entry.is_dir())


# -------------------------------------------------------------- requests
class EnsureDirsOnce(Base):
    def test_directories_are_created_once_per_root(self):
        made = []
        real = Path.mkdir

        def mkdir(self_, *a, **k):
            made.append(self_)
            return real(self_, *a, **k)
        with mock.patch.object(Path, "mkdir", mkdir):
            self.config.ensure_dirs()
            self.storage.Lecture.all()
            self.storage.list_folders()
        self.assertEqual(made, [])

    def test_a_deleted_trash_folder_comes_back(self):
        shutil.rmtree(self.config.TRASH_DIR)
        self.config.ensure_dirs()
        self.assertTrue((self.config.TRASH_DIR / ".metadata_never_index").is_file())

    def test_a_new_root_gets_its_folders(self):
        other = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, other, True)
        config = fresh(other)
        self.assertTrue(config.TRASH_DIR.is_dir())
        self.assertTrue((config.LECTURES_ROOT / "README.txt").is_file())


class FolderCache(Base):
    def count_scans(self):
        calls = []
        real = self.storage._scan_folder_dirs
        p = mock.patch.object(self.storage, "_scan_folder_dirs",
                              lambda: calls.append(1) or real())
        p.start()
        self.addCleanup(p.stop)
        return calls

    def test_repeated_loads_reuse_the_folder_list(self):
        lec = self.Lecture.create("In a class", folder="BIO", state="complete")
        calls = self.count_scans()
        for _ in range(5):
            self.assertEqual(self.Lecture.load(lec.id).path, lec.path)
        self.assertLessEqual(len(calls), 1)

    def test_folder_changes_are_seen_at_once(self):
        lec = self.Lecture.create("Mover", state="complete")
        self.Lecture.load(lec.id)
        self.storage.create_folder("CHEM")
        self.assertIn("CHEM", [f["name"] for f in self.storage.list_folders()])
        lec.move_to_folder("CHEM")
        self.assertEqual(self.Lecture.load(lec.id).folder, "CHEM")
        self.storage.rename_folder("CHEM", "Chemistry")
        self.assertEqual(self.Lecture.load(lec.id).folder, "Chemistry")

    def test_a_folder_made_outside_the_app_is_found(self):
        lec = self.Lecture.create("Finder", state="complete")
        self.Lecture.load(lec.id)                       # warm the cache
        (self.config.LECTURES_ROOT / "Physics").mkdir()
        os.rename(lec.path, self.config.LECTURES_ROOT / "Physics" / lec.id)
        self.assertEqual(self.Lecture.load(lec.id).folder, "Physics")


class MoveSweeper(Base):
    def setUp(self):
        super().setUp()
        self.server = importlib.import_module("lectern.server")
        self.server._STARTUP_DONE.set()
        self.client = self.server.app.test_client()

    def test_sweeper_only_scans_while_a_move_is_queued(self):
        w = self.server._MOVES_WANTED
        self.assertTrue(w.is_set())            # moves left from a previous run
        self.server._apply_pending_moves()
        self.assertFalse(w.is_set())           # nothing queued: the sweeper sleeps
        lec = self.Lecture.create("Busy")      # "recording": the move is deferred
        r = self.client.post(f"/api/lecture/{lec.id}/folder", json={"folder": "BIO"})
        self.assertTrue(r.get_json()["deferred"])
        self.assertTrue(w.is_set())
        self.server._apply_pending_moves()
        self.assertTrue(w.is_set())            # still busy, still queued
        lec.set_state("paused")
        self.server._apply_pending_moves()
        self.assertEqual(self.Lecture.load(lec.id).folder, "BIO")
        self.assertFalse(w.is_set())


class StorageReport(Base):
    def test_app_folder_is_walked_once(self):
        usage = importlib.import_module("lectern.usage")
        walked = []
        real = usage.dir_size

        def dir_size(path, **kw):
            walked.append(Path(path))
            return real(path, **kw) if Path(path) != usage.REPO_DIR else 1234
        with mock.patch.object(usage, "dir_size", dir_size), \
             mock.patch.object(usage, "_notes_model", lambda: None):
            a = usage.report(force=True)
            b = usage.report(force=True)
        self.assertEqual(walked.count(usage.REPO_DIR), 1)
        self.assertEqual([p["key"] for p in a["parts"]], [p["key"] for p in b["parts"]])


# ------------------------------------------------------------ small fixes
class SmallFixes(Base):
    def test_legal_docs_leave_no_open_files(self):
        server = importlib.import_module("lectern.server")
        client = server.app.test_client()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            for doc in ("terms", "privacy", "notices", "license"):
                self.assertEqual(client.get(f"/api/legal/{doc}").status_code, 200)
            gc.collect()
        self.assertEqual([w for w in caught if issubclass(w.category, ResourceWarning)], [])

    def test_progress_reader_failure_is_logged(self):
        recorder = importlib.import_module("lectern.recorder")

        class Boom:
            def __iter__(self):
                raise ValueError("I/O operation on closed file")
        rec = recorder.Recorder()
        with self.assertLogs("lectern.recorder", "WARNING"):
            rec._pump_progress(mock.Mock(stdout=Boom()))


if __name__ == "__main__":
    unittest.main()
