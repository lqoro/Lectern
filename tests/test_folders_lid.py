"""Class moves and the lid/sleep pause, without a microphone.
Run:  python -m unittest discover -s tests -v"""
import json, os, shutil, sys, tempfile, unittest
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="lectern-test-")
os.environ["LECTERN_ROOT"] = os.path.join(_TMP, "Lectures")
os.environ["LECTERN_SUPPORT"] = os.path.join(_TMP, "support")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lectern import config, recorder, recovery, server, storage   # noqa: E402
from lectern.storage import Lecture                                 # noqa: E402

config.ensure_dirs()


def _new(title="t", folder=""):
    return Lecture.create(title=title, folder=folder)


class FakeProc:
    """Popen stand-in: running until signalled, or until `exit_after` polls."""
    def __init__(self, exit_after=None, returncode=0):
        self.pid, self.returncode = 999999, None
        self._left, self._rc = exit_after, returncode

    def poll(self):
        if self.returncode is None and self._left is not None:
            self._left -= 1
            if self._left <= 0:
                self.returncode = self._rc
        return self.returncode

    def send_signal(self, sig):
        self.returncode = 0

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class Supervise(unittest.TestCase):
    def run_supervised(self, device, proc, clocks=None, lid=False, log=b"", offset=0):
        lec = _new("sup")
        (lec.work / "record.stderr.log").write_bytes(log)
        rec = recorder.Recorder()
        rec._proc = proc
        rec._state = recorder.RecordingState(
            lecture_id=lec.id, pid=proc.pid, device_name=device, device_index=0,
            started_monotonic=0, started_utc=storage.utcnow(), log_offset=offset)
        ticks = iter(clocks or [])
        lid_calls = []

        def fake_clocks():
            return next(ticks, (0.0, 0.0))

        def fake_lid():
            lid_calls.append(1)
            return lid
        with mock.patch.object(recorder.time, "sleep", lambda s: None), \
             mock.patch.object(recorder, "_clocks", fake_clocks), \
             mock.patch.object(recorder, "lid_closed", fake_lid), \
             mock.patch.object(recorder, "_WAKE_GRACE_SECONDS", 0), \
             mock.patch.object(recorder, "_LID_POLL_SECONDS", 0):
            rec._supervise(adopted=False)
        return Lecture.load(lec.id).manifest(), rec, lid_calls

    def test_sleep_pauses_and_releases(self):
        m, rec, _ = self.run_supervised("USB Mic", FakeProc(), clocks=[(0, 0), (60, 0)])
        self.assertEqual((m["state"], m["pause_reason"]), ("paused", recorder.REASON_SLEEP))
        self.assertIsNone(rec.current_lecture_id())
        self.assertTrue(rec.status()["last_stop"]["paused"])

    def test_lid_poll_pauses_builtin_mic(self):
        with mock.patch.object(recorder.pc, "IS_MAC", True):
            m, _, _ = self.run_supervised("MacBook Air Microphone", FakeProc(), lid=True)
        self.assertEqual(m["pause_reason"], recorder.REASON_LID)

    def test_usb_mic_ignores_lid_and_crash_is_labelled(self):
        m, _, calls = self.run_supervised("USB Mic", FakeProc(exit_after=3, returncode=1),
                                          lid=True)
        self.assertEqual(calls, [])
        self.assertEqual((m["state"], m["pause_reason"]), ("paused", recorder.REASON_CRASH))

    def test_helper_own_lid_exit(self):
        m, _, _ = self.run_supervised("USB Mic", FakeProc(exit_after=1, returncode=75),
                                      log=b"stopped (lid closed)\n")
        self.assertEqual(m["pause_reason"], recorder.REASON_LID)

    def test_earlier_session_log_does_not_explain_a_crash(self):
        log = b"stopped (lid closed)\n"
        m, _, _ = self.run_supervised("USB Mic", FakeProc(exit_after=1, returncode=1),
                                      log=log, offset=len(log))
        self.assertEqual(m["pause_reason"], recorder.REASON_CRASH)

    def test_disk_full_pauses_instead_of_stranding_it(self):
        with mock.patch.object(recorder.storage, "free_bytes", lambda: 0):
            m, rec, _ = self.run_supervised("USB Mic", FakeProc())
        self.assertEqual((m["state"], m["pause_reason"]), ("paused", recorder.REASON_DISK))
        self.assertIsNone(rec.current_lecture_id())

    def test_old_watcher_leaves_a_resumed_session_alone(self):
        lec = _new("stale")
        rec = recorder.Recorder()
        new = FakeProc()

        def state():
            return recorder.RecordingState(
                lecture_id=lec.id, pid=1, device_name="USB Mic", device_index=0,
                started_monotonic=0, started_utc=storage.utcnow())
        rec._proc, rec._state = FakeProc(), state()
        ticks = iter([(0, 0), (60, 0)])          # this watcher saw a sleep gap

        def pause_and_continue(_secs):           # ...then Pause + Continue land
            rec._proc, rec._state = new, state()
        with mock.patch.object(recorder.time, "sleep", pause_and_continue), \
             mock.patch.object(recorder, "_clocks", lambda: next(ticks, (60, 0))), \
             mock.patch.object(recorder, "_WAKE_GRACE_SECONDS", 0):
            rec._supervise(adopted=False)
        self.assertIsNone(new.returncode)        # the new writer was never signalled
        self.assertIs(rec._proc, new)

    def test_pause_after_the_helper_stopped_for_the_lid_keeps_that_reason(self):
        lec = _new("late")
        (lec.work / "record.stderr.log").write_bytes(b"stopped (lid closed)\n")
        rec = recorder.Recorder()
        rec._proc = FakeProc()
        rec._proc.returncode = 75
        rec._state = recorder.RecordingState(
            lecture_id=lec.id, pid=1, device_name="MacBook Air Microphone",
            device_index=0, started_monotonic=0, started_utc=storage.utcnow())
        self.assertEqual(rec.pause("user")["reason"], recorder.REASON_LID)
        self.assertEqual(Lecture.load(lec.id).manifest()["pause_reason"], recorder.REASON_LID)

    def test_writer_that_cannot_start_is_not_left_recording(self):
        dev = mock.Mock()
        dev.name, dev.index = "USB Mic", 0
        with mock.patch.object(recorder.audio, "resolve_device", lambda fresh=True: dev), \
             mock.patch.object(recorder, "_require_capture_helper", lambda: None), \
             mock.patch.object(recorder.subprocess, "Popen", side_effect=OSError("gone")):
            with self.assertRaises(OSError):
                recorder.Recorder().start("boom", skip_probe=True)
        lec = next(l for l in Lecture.all() if l.id.endswith("_boom"))
        self.assertEqual(lec.manifest()["state"], "failed")


class Storage(unittest.TestCase):
    def test_stale_object_never_recreates_a_moved_lecture(self):
        lec = _new("ghost")
        stale = Lecture(lec.path)
        old = lec.path
        lec.move_to_folder("RELI")
        with self.assertRaises(FileNotFoundError):
            stale.update(title="x")
        self.assertFalse(old.exists())

    def test_class_name_with_a_split_emoji_is_still_a_valid_folder(self):
        name = storage.clean_folder("a" * 59 + "\ud83d")       # half an emoji
        self.assertEqual(name, "a" * 59)
        _new("emoji").move_to_folder(name + "\ud83d")           # must not raise

    def test_noop_move_clears_queued_move(self):
        lec = _new("noop", folder="RELI")
        lec.set_pending_folder("RELI")
        lec.move_to_folder("RELI")
        self.assertIsNone(lec.pending_folder())


class Recovery(unittest.TestCase):
    def _orphan(self, log):
        lec = _new("orphan")
        (lec.work / "record.stderr.log").write_bytes(log)
        storage.atomic_write_json(lec.work / "recording.json",
                                  {"pid": 0, "boot_sec": 0, "argv_token": lec.id,
                                   "log_offset": 0})
        return lec

    def test_self_paused_writer_stays_paused(self):
        lec = self._orphan(b"stopped (lid closed)\n")
        submitted = []
        recovery.run_startup_recovery(submitted.append)
        m = Lecture.load(lec.id).manifest()
        self.assertEqual((m["state"], m["pause_reason"]), ("paused", recorder.REASON_LID))
        self.assertNotIn(lec.id, submitted)


class Trash(unittest.TestCase):
    def _released(self, *names):
        lecs = [_new(n) for n in names]
        for l in lecs:
            l.set_state("complete")
            w = config.TRASH_DIR / f"{l.id}__work"
            w.mkdir()
            (w / "seg-000000.wav").write_bytes(b"x")
        return lecs

    def test_lecture_filed_during_reverify_is_still_released(self):
        a, b = self._released("tra", "trb")

        def reverify(lec):
            if lec.id == a.id:
                Lecture.load(a.id).move_to_folder("Physics")   # user files it meanwhile
            return True
        with mock.patch.object(recovery, "_reverify", reverify):
            recovery.sweep_trash()
        for l in (a, b):
            self.assertFalse((config.TRASH_DIR / f"{l.id}__work").exists())
        self.assertIn("wav_released_utc", Lecture.load(a.id).manifest()["audio"])
        self.assertEqual(Lecture.load(a.id).folder, "Physics")

    def test_one_failure_does_not_stop_the_sweep(self):
        a, b = self._released("trc", "trd")

        def reverify(lec):
            if lec.id == a.id:
                raise RuntimeError("decode failed")
            return True
        with mock.patch.object(recovery, "_reverify", reverify):
            recovery.sweep_trash()
        self.assertTrue((config.TRASH_DIR / f"{a.id}__work").exists())
        self.assertFalse((config.TRASH_DIR / f"{b.id}__work").exists())


class Moves(unittest.TestCase):
    def setUp(self):
        self.c = server.app.test_client()
        server._STARTUP_DONE.set()

    def test_moves_wait_for_startup_recovery(self):
        lec = _new("early")
        lec.set_state("paused")
        server._STARTUP_DONE.clear()
        r = self.c.post(f"/api/lecture/{lec.id}/folder", json={"folder": "crju"}).get_json()
        self.assertTrue(r["deferred"])
        self.assertEqual(server._apply_pending_moves(), 0)
        server._STARTUP_DONE.set()
        server._apply_pending_moves()
        self.assertEqual(Lecture.load(lec.id).folder, "crju")

    def test_recording_on_disk_counts_as_busy(self):
        lec = _new("rec")                       # state "recording", no writer here
        r = self.c.post(f"/api/lecture/{lec.id}/folder", json={"folder": "RELI"}).get_json()
        self.assertTrue(r["deferred"])
        lec.set_state("paused")
        server._apply_pending_moves()
        self.assertEqual(Lecture.load(lec.id).folder, "RELI")

    def test_record_start_holds_the_move_lock(self):
        held = []

        def start(*a, **k):
            held.append(server._MOVE_LOCK.locked())
            return {"recording": True}
        with mock.patch.object(server.RECORDER, "start", start):
            self.c.post("/api/record/start", json={"title": "x", "folder": "RELI"})
        self.assertEqual(held, [True])

    def test_resume_refused_while_processing(self):
        lec = _new("proc")
        lec.set_state("complete")
        with mock.patch.object(server.PIPELINE, "is_running", lambda i: i == lec.id):
            r = self.c.post("/api/record/resume", json={"lecture_id": lec.id})
        self.assertEqual(r.status_code, 409)


def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
