"""lectern-capture (portable): lossless microphone capture into fixed-length
WAV segments on Windows, Linux and macOS.

This is the cross-platform twin of lectern_capture.swift and keeps the same
contract with recorder.py, so everything downstream (concat, transcribe,
archive, recovery) is unchanged:

  * writes <pattern % n> as 16 kHz mono s16le WAV with a 44-byte header; the
    RIFF/data sizes are 0xFFFFFFFF while a segment is open and patched when it
    closes, so a crash leaves a file that stdlib `wave` still reads
  * segments are cut by SAMPLE COUNT, so a 60 s segment is always exactly
    1,920,044 bytes and nothing is dropped between them
  * the audio callback only copies bytes into a queue; a separate writer
    thread does all disk I/O, so a slow disk can never make the sound card
    drop input
  * prints `out_time_us=N` / `progress=continue` on stdout ONLY when audio
    advanced, so a vanished device shows up as a stall
  * stops cleanly on SIGINT/SIGTERM, on the existence of --stop-file (the
    portable way, and the only one that works for a detached Windows process),
    or after --duration seconds; the open segment's header is patched first
  * exits 3 if the microphone cannot be opened (permission denied, no device),
    4 if the audio library is missing

If the device will not open at 16 kHz mono it is opened at its native rate and
converted with a streaming low-pass + linear-interpolation resampler.
Run `python -m lectern.capture_py --list-devices` for a JSON device list.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import struct
import sys
import threading
import time
from pathlib import Path

OUT_RATE = 16000
HEADER_BYTES = 44
EXIT_DEVICE, EXIT_NO_LIB = 3, 4


def log(msg: str) -> None:
    print("lectern-capture: " + msg, file=sys.stderr, flush=True)


def emit(s: str) -> None:
    try:                      # stdout may be a dead pipe; never let that kill us
        os.write(1, s.encode())
    except OSError:
        pass


# ------------------------------------------------------------- WAV output
_FMT = struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, OUT_RATE, OUT_RATE * 2, 2, 16)
HEADER_OPEN = b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + _FMT + \
    b"data" + struct.pack("<I", 0xFFFFFFFF)
assert len(HEADER_OPEN) == HEADER_BYTES


class SegmentWriter:
    def __init__(self, pattern: str, seg_samples: int, start_number: int):
        self.pattern, self.seg_samples, self.n = pattern, seg_samples, start_number
        self.f = None
        self.in_seg = 0
        self.total = 0            # samples written this run

    def _open(self) -> None:
        self.f = open(self.pattern % self.n, "wb", buffering=0)   # unbuffered
        self.f.write(HEADER_OPEN)
        self.in_seg = 0

    def _close(self) -> None:
        if self.f is None:
            return
        nbytes = self.in_seg * 2
        try:
            self.f.seek(4)
            self.f.write(struct.pack("<I", 36 + nbytes))
            self.f.seek(40)
            self.f.write(struct.pack("<I", nbytes))
        finally:
            self.f.close()
            self.f = None
            self.n += 1

    def write(self, pcm: bytes) -> None:
        mv = memoryview(pcm)[: len(pcm) - (len(pcm) % 2)]
        while len(mv):
            if self.f is None:
                self._open()
            room = (self.seg_samples - self.in_seg) * 2
            chunk = mv[:room]
            self.f.write(chunk)
            self.in_seg += len(chunk) // 2
            self.total += len(chunk) // 2
            mv = mv[len(chunk):]
            if self.in_seg >= self.seg_samples:
                self._close()

    def close(self) -> None:
        self._close()


# -------------------------------------------------------------- resampling
class Resampler:
    """Streaming mono int16 -> 16 kHz int16 (windowed-sinc low-pass, then
    linear interpolation at a carried fractional phase). Output length is
    within one sample of in_samples * 16000 / in_rate over any run length."""

    def __init__(self, in_rate: int, taps: int = 63):
        import numpy as np
        self.np = np
        self.step = in_rate / OUT_RATE
        fc = min(0.45 * OUT_RATE, 0.45 * in_rate) / in_rate     # cycles/sample
        m = taps - 1
        n = np.arange(taps) - m / 2
        h = 2 * fc * np.sinc(2 * fc * n) * np.blackman(taps)
        self.h = h / h.sum()
        self.tail = np.zeros(m, dtype=np.float64)
        self.buf = np.zeros(0, dtype=np.float64)
        self.pos = 0.0

    def process(self, x_int16) -> bytes:
        np = self.np
        x = x_int16.astype(np.float64)
        full = np.concatenate([self.tail, x])
        self.tail = full[len(full) - (len(self.h) - 1):]
        y = np.convolve(full, self.h, mode="valid")              # len(x) samples
        self.buf = np.concatenate([self.buf, y])
        avail = len(self.buf) - 1
        if avail <= self.pos:
            return b""
        k = int(np.ceil((avail - self.pos) / self.step))
        idx = self.pos + self.step * np.arange(k)
        i0 = np.floor(idx).astype(np.int64)
        fr = idx - i0
        out = self.buf[i0] * (1 - fr) + self.buf[i0 + 1] * fr
        new_pos = self.pos + k * self.step
        drop = int(new_pos)
        self.buf = self.buf[drop:]
        self.pos = new_pos - drop
        return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()


# ----------------------------------------------------------------- sources
def _sd():
    try:
        import sounddevice as sd
        return sd
    except Exception as e:                                       # ImportError or PortAudio OSError
        log(f"audio library unavailable: {e}")
        log("install it with:  pip install sounddevice numpy")
        sys.exit(EXIT_NO_LIB)


def _preferred_hostapi(sd) -> int | None:
    """One physical mic shows up once per host API on Windows (MME, DirectSound,
    WASAPI...). WASAPI/Core Audio are the least lossy, MME truncates names."""
    prefer = ("Windows WASAPI", "Core Audio", "ALSA", "PulseAudio", "JACK Audio Connection Kit")
    apis = sd.query_hostapis()
    names = [a["name"] for a in apis]
    for p in prefer:
        if p in names:
            return names.index(p)
    return None


def list_input_devices() -> list[dict]:
    sd = _sd()
    pref = _preferred_hostapi(sd)
    default_in = sd.default.device[0]
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        if pref is not None and d["hostapi"] != pref and i != default_in:
            continue
        out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"],
                    "default_rate": int(d["default_samplerate"]),
                    "default": i == default_in})
    return out


def pick_device(name: str | None) -> int | None:
    """Resolve a device NAME to a PortAudio index right now (indices move when
    USB mics come and go). None -> the system default input."""
    devs = list_input_devices()
    if name:
        for d in devs:
            if d["name"] == name:
                return d["index"]
        for d in devs:
            if name.lower() in d["name"].lower():
                return d["index"]
        log(f"device {name!r} not found; using the system default input")
    for d in devs:
        if d["default"]:
            return d["index"]
    return devs[0]["index"] if devs else None


class SoundDeviceSource:
    """Feeds (bytes) 16 kHz mono int16 into `q`. Tries 16 kHz first, then the
    device's native rate with our own resampler."""

    def __init__(self, device_name: str | None, q: "queue.SimpleQueue[bytes]"):
        self.sd = _sd()
        self.q = q
        self.stream = None
        self.resampler = None
        self.channels = 1
        self.rate = OUT_RATE
        self.overflows = 0
        self._last_warn = 0.0
        idx = pick_device(device_name)
        if idx is None:
            log("no input device found")
            sys.exit(EXIT_DEVICE)
        info = self.sd.query_devices(idx)
        self.name = info["name"]
        native = int(info["default_samplerate"])
        maxch = int(info["max_input_channels"])
        attempts = [(OUT_RATE, 1), (OUT_RATE, min(2, maxch)),
                    (native, 1), (native, min(2, maxch))]
        err = None
        for rate, ch in dict.fromkeys(attempts):
            try:
                self.stream = self._open(idx, rate, ch)
                self.rate, self.channels = rate, ch
                break
            except Exception as e:
                err = e
        if self.stream is None:
            log(f"could not open {self.name!r}: {err}")
            log("check that a microphone is connected and that this app is allowed to use it "
                "(Windows: Settings > Privacy & security > Microphone)")
            sys.exit(EXIT_DEVICE)
        if self.rate != OUT_RATE:
            self.resampler = Resampler(self.rate)
            log(f"capturing at {self.rate} Hz x{self.channels}; converting to {OUT_RATE} Hz")

    def _open(self, idx, rate, ch):
        sd = self.sd
        extra = None
        try:
            if sys.platform.startswith("win") and hasattr(sd, "WasapiSettings"):
                api = sd.query_hostapis(sd.query_devices(idx)["hostapi"])["name"]
                if "WASAPI" in api:
                    extra = sd.WasapiSettings(auto_convert=True)
        except Exception:
            extra = None
        s = sd.RawInputStream(samplerate=rate, device=idx, channels=ch, dtype="int16",
                              callback=self._cb, extra_settings=extra)
        s.start()
        return s

    def _cb(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
            now = time.monotonic()
            if now - self._last_warn > 1.0:
                self._last_warn = now
                log(f"audio status: {status}")
        self.q.put(bytes(indata))

    def convert(self, raw: bytes) -> bytes:
        if self.channels == 1 and self.resampler is None:
            return raw
        import numpy as np
        x = np.frombuffer(raw, dtype="<i2")
        if self.channels > 1:
            x = x[: len(x) - len(x) % self.channels].reshape(-1, self.channels).mean(axis=1)
            x = x.astype("<i2")
        return self.resampler.process(x) if self.resampler else x.tobytes()

    def close(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


class SyntheticSource:
    """A deterministic tone at an arbitrary rate, for tests and CI where no
    microphone exists. Paced faster than real time via --synthetic-speed."""

    def __init__(self, q, rate: int, speed: float, channels: int = 1):
        self.q, self.rate, self.speed, self.channels = q, rate, speed, channels
        self.name = f"Synthetic tone ({rate} Hz)"
        self.resampler = Resampler(rate) if rate != OUT_RATE else None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        import numpy as np
        block = max(1, self.rate // 100)          # 10 ms
        n = 0
        while not self._stop.is_set():
            t = (np.arange(block) + n) / self.rate
            x = (8000 * np.sin(2 * np.pi * 440 * t)).astype("<i2")
            n += block
            if self.channels > 1:
                x = np.repeat(x[:, None], self.channels, axis=1).reshape(-1)
            self.q.put(x.tobytes())
            time.sleep(0.01 / self.speed)

    def convert(self, raw: bytes) -> bytes:
        import numpy as np
        x = np.frombuffer(raw, dtype="<i2")
        if self.channels > 1:
            x = x.reshape(-1, self.channels).mean(axis=1).astype("<i2")
        return self.resampler.process(x) if self.resampler else x.tobytes()

    def close(self):
        self._stop.set()


# ------------------------------------------------------------------- main
def _keep_awake_windows() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lectern-capture")
    ap.add_argument("--device")
    ap.add_argument("--out", default="")
    ap.add_argument("--segment-seconds", type=int, default=60)
    ap.add_argument("--start-number", type=int, default=0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--stop-file", default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--source", choices=("sounddevice", "synthetic"),
                    default=os.environ.get("LECTERN_CAPTURE_SOURCE", "sounddevice"))
    ap.add_argument("--synthetic-rate", type=int, default=48000)
    ap.add_argument("--synthetic-speed", type=float,
                    default=float(os.environ.get("LECTERN_SYNTH_SPEED", "1")))
    ap.add_argument("--synthetic-channels", type=int, default=1)
    a = ap.parse_args(argv)

    if a.list_devices:
        print(json.dumps(list_input_devices()))
        return 0
    if not a.out or "%" not in a.out:
        log("--out must be a pattern containing %d, e.g. seg-%06d.wav")
        return 2

    stop = threading.Event()

    def _stop(*_):
        stop.set()
    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _stop)
            except (ValueError, OSError):
                pass
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    q: "queue.SimpleQueue[bytes]" = queue.SimpleQueue()
    if a.source == "synthetic":
        src = SyntheticSource(q, a.synthetic_rate, a.synthetic_speed, a.synthetic_channels)
    else:
        src = SoundDeviceSource(a.device, q)
    log(f"recording from {src.name!r}")
    _keep_awake_windows()

    writer = SegmentWriter(a.out, a.segment_seconds * OUT_RATE, a.start_number)
    max_samples = int(a.duration * OUT_RATE) if a.duration else None
    last_emit = 0.0
    last_total = -1
    stop_file = Path(a.stop_file) if a.stop_file else None

    def pump(timeout: float) -> bool:
        nonlocal last_emit, last_total
        try:
            raw = q.get(timeout=timeout)
        except queue.Empty:
            return False
        pcm = src.convert(raw)
        if max_samples is not None:
            room = max_samples - writer.total
            pcm = pcm[: max(0, room) * 2]
        if pcm:
            writer.write(pcm)
        now = time.monotonic()
        if writer.total != last_total and now - last_emit >= 0.5:
            last_total, last_emit = writer.total, now
            emit(f"out_time_us={int(writer.total * 1_000_000 / OUT_RATE)}\nprogress=continue\n")
        return True

    try:
        while not stop.is_set():
            pump(0.25)
            if stop_file is not None and stop_file.exists():
                break
            if max_samples is not None and writer.total >= max_samples:
                break
        src.close()
        while pump(0.05):                        # drain what the device already gave us
            pass
    finally:
        writer.close()
        emit(f"out_time_us={int(writer.total * 1_000_000 / OUT_RATE)}\nprogress=end\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
