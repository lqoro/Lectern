"""Tests for the portable recorder, using its built-in synthetic source so no
microphone is needed.   Run:  python -m unittest discover -s tests -v"""
import glob, os, signal, subprocess, sys, tempfile, time, unittest, wave
try:
    import numpy as np
except ImportError:          # the portable recorder needs numpy; Mac-only installs don't
    np = None

HELPER = os.path.join(os.path.dirname(__file__), "..", "lectern", "capture_py.py")
SEG_BYTES = 44 + 60 * 16000 * 2          # 1,920,044


def record(rate, secs, channels=1, speed=200, seg=60):
    d = tempfile.mkdtemp()
    subprocess.run([sys.executable, HELPER, "--source", "synthetic",
                    "--synthetic-rate", str(rate), "--synthetic-speed", str(speed),
                    "--synthetic-channels", str(channels), "--duration", str(secs),
                    "--out", d + "/seg-%06d.wav", "--segment-seconds", str(seg)],
                   check=True, capture_output=True, timeout=120)
    return d


@unittest.skipIf(np is None, 'numpy not installed')
class Segments(unittest.TestCase):
    def check(self, rate, secs, channels=1):
        d = record(rate, secs, channels)
        files = sorted(glob.glob(d + "/seg-*.wav"))
        chunks = []
        for f in files[:-1]:
            self.assertEqual(os.path.getsize(f), SEG_BYTES)      # sample-exact cut
        for f in files:
            with wave.open(f) as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()),
                                 (16000, 1, 2))
                chunks.append(np.frombuffer(w.readframes(w.getnframes()), "<i2"))
        x = np.concatenate(chunks).astype(float)
        self.assertEqual(len(x), secs * 16000)                   # nothing dropped
        spec = np.abs(np.fft.rfft(x[16000:160000] * np.hanning(144000)))
        self.assertAlmostEqual(np.argmax(spec) * 16000 / 144000, 440, delta=1.5)
        self.assertLess(np.max(np.abs(np.diff(x[400:]))), 1500)  # no glitches at seams

    def test_16k(self): self.check(16000, 150)
    def test_48k(self): self.check(48000, 130)
    def test_44k1(self): self.check(44100, 130)
    def test_stereo(self): self.check(48000, 70, channels=2)
    def test_8k(self): self.check(8000, 70)


@unittest.skipIf(np is None, 'numpy not installed')
class Stopping(unittest.TestCase):
    def spawn(self, d, *extra):
        return subprocess.Popen([sys.executable, HELPER, "--source", "synthetic",
                                 "--synthetic-rate", "16000", "--out", d + "/seg-%06d.wav", *extra],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def frames(self, d):
        with wave.open(sorted(glob.glob(d + "/seg-*.wav"))[0]) as w:
            return w.getnframes()

    @unittest.skipIf(sys.platform.startswith("win"), "signals are not used on Windows")
    def test_sigint_patches_header(self):
        d = tempfile.mkdtemp(); p = self.spawn(d); time.sleep(2.5)
        p.send_signal(signal.SIGINT); p.communicate(timeout=10)
        self.assertEqual(p.returncode, 0); self.assertGreater(self.frames(d), 16000)

    def test_stop_file(self):
        d = tempfile.mkdtemp(); sf = d + "/STOP"
        p = self.spawn(d, "--stop-file", sf); time.sleep(2.5); open(sf, "w").close()
        p.communicate(timeout=10)
        self.assertEqual(p.returncode, 0); self.assertGreater(self.frames(d), 16000)


if __name__ == "__main__":
    unittest.main()
