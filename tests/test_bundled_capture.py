"""Lectern.app ships the mic helper prebuilt in LECTERN_BIN_DIR. It must be
used as is, never rebuilt (people who download the app have no compiler).
Run:  python -m unittest discover -s tests -v"""
import os, stat, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lectern import capture, config  # noqa: E402


def fake_helper(d: Path, executable: bool = True) -> Path:
    p = d / config.CAPTURE_BIN.name
    p.write_text("#!/bin/sh\nexit 0\n")
    if executable:
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return p


class BundledHelper(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def env(self, **extra):
        return mock.patch.dict(os.environ, {"LECTERN_CAPTURE": "swift", **extra})

    def test_shipped_helper_is_used_without_building(self):
        helper = fake_helper(self.dir)
        with self.env(LECTERN_BIN_DIR=str(self.dir)), \
                mock.patch.object(capture.subprocess, "run") as run:
            self.assertEqual(capture.ensure_built(), helper)
            self.assertEqual(capture.available(), helper)
            run.assert_not_called()                       # no swiftc

    def test_later_bin_dir_entries_are_searched(self):
        empty = Path(tempfile.mkdtemp())
        helper = fake_helper(self.dir)
        with self.env(LECTERN_BIN_DIR=os.pathsep.join([str(empty), str(self.dir)])):
            self.assertEqual(capture.bundled(), helper)

    def test_non_executable_file_is_ignored(self):
        fake_helper(self.dir, executable=False)
        with self.env(LECTERN_BIN_DIR=str(self.dir)):
            self.assertIsNone(capture.bundled())

    def test_no_bin_dir_means_no_bundled_helper(self):
        with self.env(LECTERN_BIN_DIR=""):
            self.assertIsNone(capture.bundled())

    def test_portable_mode_still_wins(self):
        fake_helper(self.dir)
        with self.env(LECTERN_BIN_DIR=str(self.dir), LECTERN_CAPTURE="python"):
            self.assertIsNone(capture.ensure_built())
            self.assertIsNone(capture.available())


if __name__ == "__main__":
    unittest.main()
