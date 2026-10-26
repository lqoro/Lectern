"""Speech-model catalogue, selection and download.

Lectern no longer assumes a model file already sits in some other app's folder.
It looks in its own models directory first, then in the legacy location, picks
the best installed model that suits this machine, and can fetch a smaller or
larger one on request. Nothing is downloaded unless the user asks for it.

The `ram_mb` and `size_mb` figures are the ones measured by bench/bench.py on an
M1 with 8 GB (5-minute excerpt). The CPU-only recommendations are heuristics,
not measurements: the benchmark has no CPU-only run yet.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import config, hardware, storage

log = logging.getLogger(__name__)

BASE_URL = os.environ.get(
    "LECTERN_MODEL_BASE_URL",
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main")

LEGACY_DIR = Path.home() / ".localflow" / "models"
# Models fetched by bench/bench.py --download live here; use them rather than
# downloading the same files a second time.
BENCH_DIR = Path(__file__).resolve().parent.parent / "bench" / "models"
MODELS_DIR = Path(os.environ.get("LECTERN_MODEL_DIR", config.SUPPORT_DIR / "models"))
SETTINGS_FILE = config.SUPPORT_DIR / "settings.json"
_MB = 1_000_000          # catalogue sizes are decimal MB, as on the download page


@dataclass(frozen=True)
class Model:
    id: str
    file: str               # preferred file (what a download fetches)
    size_mb: int            # that file on disk
    ram_mb: int             # peak RAM while transcribing (measured, M1)
    quality: int            # 1 (lowest) .. 5, for ordering only
    label: str
    note: str
    # Other files that are the same model and may already be on disk, as
    # (filename, size_mb). Used when found; never downloaded.
    alts: tuple[tuple[str, int], ...] = ()


# Ordered best-quality first. Measured 2026-10-26 on an M1 (bench/results.md):
# the q5_1 builds of tiny/base/small are ~60% smaller, 15-25% faster and use
# 30-45% less RAM than the full-precision files, with no loss on the
# agreement measure, so they are what Lectern downloads. CPU-only speeds on
# the M1's CPU: tiny 74x, base 30x, small 10x, turbo 3.1x, large 1.2x real time.
CATALOG: tuple[Model, ...] = (
    Model("large-v3-q5_0", "ggml-large-v3-q5_0.bin", 1081, 1507, 5,
          "Most accurate", "Slowest. Needs a strong machine with 16 GB or more."),
    Model("large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin", 574, 736, 4,
          "Balanced (default)", "Near-large accuracy at about twice the speed."),
    Model("small.en", "ggml-small.en-q5_1.bin", 190, 382, 3,
          "Light", "Good for older or low-memory computers. English only.",
          alts=(("ggml-small.en.bin", 488),)),
    Model("base.en", "ggml-base.en-q5_1.bin", 60, 208, 2,
          "Very light", "Fast on weak machines; more mistakes. English only.",
          alts=(("ggml-base.en.bin", 148),)),
    Model("tiny.en", "ggml-tiny.en-q5_1.bin", 32, 185, 1,
          "Minimal", "For very old machines. Expect many mistakes. English only.",
          alts=(("ggml-tiny.en.bin", 78),)),
)
BY_ID = {m.id: m for m in CATALOG}


# --------------------------------------------------------------- settings
def load_settings() -> dict:
    try:
        data = json.loads(SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(**fields) -> dict:
    cur = load_settings()
    for k, v in fields.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    storage.atomic_write_json(SETTINGS_FILE, cur, indent=2)
    return cur


# ------------------------------------------------------------ discovery
def search_dirs() -> list[Path]:
    return [MODELS_DIR, LEGACY_DIR, BENCH_DIR]


def installed_path(model: Model) -> Path | None:
    for fname, size_mb in ((model.file, model.size_mb), *model.alts):
        for d in search_dirs():
            p = d / fname
            # A partial or truncated file must not count as installed.
            if p.is_file() and p.stat().st_size >= size_mb * _MB * 0.9:
                return p
    return None


def installed() -> list[Model]:
    return [m for m in CATALOG if installed_path(m)]


def fits(model: Model, total_ram_bytes: int | None) -> bool:
    """Whisper plus the OS and a browser: keep the model under ~35% of RAM."""
    if not total_ram_bytes:
        return model.quality <= 3
    return model.ram_mb * 2**20 <= 0.35 * total_ram_bytes


def recommend(hw: dict | None = None) -> Model:
    hw = hw or hardware.snapshot()
    ram_gb = (hw.get("ram_total_mb") or 0) / 1024
    accel = hw.get("accelerator") or "cpu"
    cores = hw.get("cpu_cores") or 2
    if accel != "cpu":
        if ram_gb >= 8:
            return BY_ID["large-v3-turbo-q5_0"]
        return BY_ID["small.en"]
    # CPU only. Turbo ran at 3.1x real time on the M1's CPU (an hour of
    # lecture in ~20 min); an older CPU is slower, so only a big one gets it.
    if ram_gb >= 16 and cores >= 8:
        return BY_ID["large-v3-turbo-q5_0"]
    if ram_gb >= 8 and cores >= 6:
        return BY_ID["small.en"]
    if ram_gb >= 4:
        return BY_ID["base.en"]
    return BY_ID["tiny.en"]


def active() -> tuple[Path | None, Model | None, str]:
    """The model file to transcribe with, plus why it was chosen.

    Order: LECTERN_MODEL (an explicit file) -> the user's saved choice ->
    the recommended model if installed -> the best installed model that fits
    this machine -> the smallest installed model. None if nothing is installed.
    """
    env = os.environ.get("LECTERN_MODEL")
    if env:
        return Path(env), None, "LECTERN_MODEL"
    chosen = BY_ID.get(load_settings().get("whisper_model", ""))
    if chosen and (p := installed_path(chosen)):
        return p, chosen, "your choice"
    hw = hardware.snapshot()
    rec = recommend(hw)
    if p := installed_path(rec):
        return p, rec, "recommended for this computer"
    have = installed()
    ok = [m for m in have if fits(m, (hw.get("ram_total_mb") or 0) * 2**20 or None)]
    if ok:
        m = ok[0]
        return installed_path(m), m, "best installed model that fits this computer"
    if have:
        m = have[-1]
        return installed_path(m), m, "only model installed"
    return None, None, "none installed"


def active_path() -> Path | None:
    return active()[0]


def active_name() -> str:
    p, m, _ = active()
    return m.id if m else (p.name if p else "none")


def describe() -> dict:
    hw = hardware.snapshot()
    rec = recommend(hw)
    p, cur, why = active()
    total = (hw.get("ram_total_mb") or 0) * 2**20 or None
    return {
        "hardware": hw,
        "recommended": rec.id,
        "active": {"id": cur.id if cur else None,
                   "file": p.name if p else None, "why": why},
        "models": [{
            "id": m.id, "label": m.label, "note": m.note,
            "size_mb": m.size_mb, "ram_mb": m.ram_mb,
            "installed": installed_path(m) is not None,
            "fits": fits(m, total),
            "recommended": m.id == rec.id,
        } for m in CATALOG],
        "download": DOWNLOADER.status(),
    }


# --------------------------------------------------------------- download
class DownloadError(RuntimeError):
    pass


class Downloader:
    """One download at a time, resumable, written to .part then renamed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {}
        self._cancel = threading.Event()

    def status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def start(self, model_id: str) -> dict:
        model = BY_ID.get(model_id)
        if model is None:
            raise KeyError(model_id)
        with self._lock:
            if self._state.get("running"):
                return dict(self._state)
            self._cancel.clear()
            self._state = {"id": model_id, "running": True, "bytes": 0,
                           "total": model.size_mb * _MB, "error": "",
                           "started": time.time()}
        threading.Thread(target=self._run, args=(model,), daemon=True).start()
        return self.status()

    def cancel(self) -> None:
        self._cancel.set()

    def _set(self, **kw) -> None:
        with self._lock:
            self._state.update(kw)

    def _run(self, model: Model) -> None:
        try:
            self._download(model)
            self._set(running=False, done=True)
        except Exception as e:
            log.warning("model download failed: %s", e)
            self._set(running=False, error=str(e)[:300])

    def _download(self, model: Model) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / model.file
        part = dest.with_suffix(dest.suffix + ".part")
        have = part.stat().st_size if part.is_file() else 0

        need = model.size_mb * _MB - have
        free = hardware_free(MODELS_DIR)
        if free is not None and free < need + 200 * _MB:
            raise DownloadError(
                f"not enough disk space: need about {need // _MB + 200} MB free")

        req = urllib.request.Request(f"{BASE_URL}/{model.file}")
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            resp = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:           # already complete?
                _check_model_file(part, model)
                part.replace(dest)
                return
            raise DownloadError(f"download refused: HTTP {e.code}") from e
        except OSError as e:
            raise DownloadError(f"could not reach the download server: {e}") from e

        with resp:
            resumed = resp.status == 206
            if have and not resumed:             # server ignored Range: restart
                have = 0
            length = resp.headers.get("Content-Length")
            total = (int(length) + have) if length else model.size_mb * _MB
            self._set(total=total, bytes=have)
            with open(part, "ab" if have else "wb") as f:
                while True:
                    if self._cancel.is_set():
                        raise DownloadError("cancelled (partial file kept so it can resume)")
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    have += len(block)
                    self._set(bytes=have)
                f.flush()
                os.fsync(f.fileno())

        # A short read must never be installed as a model.
        if have != total:
            raise DownloadError(
                f"download incomplete ({have} of {total} bytes); run it again to resume")
        _check_model_file(part, model)
        part.replace(dest)


_GGML_MAGIC = b"lmgg"            # whisper.cpp model files start with this
_SIZE_TOLERANCE = 0.05


def _check_model_file(path: Path, model: Model) -> None:
    """Refuse to install something that is not this model, e.g. a captive
    portal's HTML page or a corrupt partial file. The partial file is removed
    so the next attempt starts clean."""
    size = path.stat().st_size
    expected = model.size_mb * _MB
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != _GGML_MAGIC or abs(size - expected) > expected * _SIZE_TOLERANCE:
        path.unlink(missing_ok=True)
        raise DownloadError(
            "the downloaded file is not a valid speech model (a Wi-Fi login page "
            "or a network error?). Try again on a working connection.")


def hardware_free(path: Path) -> int | None:
    import shutil
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


DOWNLOADER = Downloader()
