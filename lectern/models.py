"""Speech-model catalogue, selection and download.

Lectern no longer assumes a model file already sits in some other app's folder.
It looks in its own models directory first, then in the legacy location, picks
the best installed model that suits this machine, and can fetch a smaller or
larger one on request. Nothing is downloaded unless the user asks for it.

The `ram_mb` and `size_mb` figures are the ones measured with a benchmark script on an
M1 with 8 GB (5-minute excerpt). The CPU-only recommendations are heuristics,
not measurements: the benchmark has no CPU-only run yet.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import ssl
import threading
import platform
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import config, hardware, storage


def _ssl_context() -> ssl.SSLContext:
    """The packaged Mac app has no system certificate store of its own, so
    HTTPS downloads trust the CA bundle that ships inside it (certifi)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

log = logging.getLogger(__name__)

BASE_URL = os.environ.get(
    "LECTERN_MODEL_BASE_URL",
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main")

# Parakeet is a folder of CoreML models, shipped as one zip on a Lectern GitHub release.
ASSET_URL = os.environ.get("LECTERN_MODEL_BASE_URL",
                           "https://github.com/lqoro/Lectern/releases/download/models-v1")

LEGACY_DIR = Path.home() / ".localflow" / "models"
# A developer's benchmark downloads (bench/models, not in the repo) live here; use them rather than
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
    # Measured speed, x real time, on an M1 (developer benchmark, 5-min lecture excerpt): with the
    # Apple GPU, and CPU-only. Used for "an hour of lecture takes ~N min".
    xrt_gpu: float = 0.0
    xrt_cpu: float = 0.0
    # SHA-256 of `file` as published (Hugging Face LFS oid). A download that
    # does not match is never installed.
    sha256: str = ""
    # Set for a model published as a zip on a Lectern GitHub release (CoreML needs a folder, which a
    # single Hugging Face file cannot be): `size_mb` and `sha256` then describe the zip, and
    # `required` lists the paths (relative to MODELS_DIR) that must exist once it is unpacked.
    asset: str = ""
    required: tuple[str, ...] = ()
    apple_only: bool = False         # runs only on Apple Silicon (the Neural Engine)
    min_macos: int = 0

    @property
    def is_folder(self) -> bool:
        """The model itself is a CoreML folder (Parakeet), not a ggml file."""
        return not self.file.endswith(".bin")


# large-v3-turbo-q4_0-coreml.zip: the q4_0 file (474 MB) plus the CoreML encoder folder (1.27 GB).
Q4_ZIP_MB = 1586
Q4_ZIP_SHA256 = "4d88d390ebf3766d6ffcb5a1a7447b5d813459c615968480fefa99239f37bcd7"

# Ordered best-quality first. Measured 2026-10-26 on an M1:
# the q5_1 builds of tiny/base/small are ~60% smaller, 15-25% faster and use
# 30-45% less RAM than the full-precision files, with no loss on the
# agreement measure, so they are what Lectern downloads. CPU-only speeds on
# the M1's CPU: tiny 74x, base 30x, small 10x, turbo 3.1x, large 1.2x real time.
CATALOG: tuple[Model, ...] = (
    Model("large-v3-q5_0", "ggml-large-v3-q5_0.bin", 1081, 1507, 5,
          "Most accurate", "Slowest. Needs a strong machine with 16 GB or more.",
          xrt_gpu=6.1, xrt_cpu=1.2,
          sha256="d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"),
    # The same turbo model in 4-bit, with its CoreML encoder beside it so whisper.cpp runs the
    # encoder on the Neural Engine. Measured on an M4 (10 min of speech): 16.5 s against 28.0 s for
    # the q5_0 file on Metal, 0.75 GB RAM; word agreement with the f16 model 99.5% (q5_0: 99.3%).
    # Both files are made by us (whisper-quantize, Core ML conversion), so they are hosted as one zip.
    Model("large-v3-turbo-q4_0", "ggml-large-v3-turbo-q4_0.bin", Q4_ZIP_MB, 760, 4,
          "Balanced", "Near-large accuracy, about 1.7 times faster than the compatible model "
          "below. Uses the Neural Engine of Apple chips.",
          xrt_gpu=15.3, apple_only=True,
          asset="large-v3-turbo-q4_0-coreml.zip",
          required=("ggml-large-v3-turbo-q4_0.bin", "ggml-large-v3-turbo-encoder.mlmodelc/model.mil"),
          sha256=Q4_ZIP_SHA256),
    Model("large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin", 574, 736, 4,
          "Balanced (compatible)", "Same accuracy, smaller download, works on every computer.",
          xrt_gpu=9.0, xrt_cpu=3.1,
          sha256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"),
    # NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0), CoreML conversion by FluidInference, run by
    # bin/fluidaudiocli on the Neural Engine. Apple Silicon and macOS 14+ only. Recommended there
    # when the engine is on disk (see recommend()). Measured 2026-11-11, 10 min of lecture, warm:
    # M1 Pro 3.6 s / 0.16 GB, M4 2.2 s, against 26-34 s for the Whisper turbo builds; 9.8% word
    # disagreement with Whisper large-v3 (turbo q5_0: 7.2%, q4_0: 8.5%).
    Model("parakeet-tdt-0.6b-v3-coreml", "parakeet-tdt-0.6b-v3-coreml", 555, 400, 4,
          "Fastest (Parakeet)", "Several times faster than the Whisper models, with slightly more "
          "mistakes. Uses the Neural Engine of Apple chips. Covers 25 European languages, English included.",
          xrt_gpu=153.0, xrt_cpu=0.0, asset="parakeet-tdt-0.6b-v3-coreml.zip",
          required=("parakeet-tdt-0.6b-v3-coreml/config.json",),
          apple_only=True, min_macos=14,
          sha256="dc345e55f08452b3cdefd23602b32f906322a6d95387249fafb5a97fd51a853b"),
    Model("small.en", "ggml-small.en-q5_1.bin", 190, 382, 3,
          "Light", "Good for older or low-memory computers. English only.",
          alts=(("ggml-small.en.bin", 488),), xrt_gpu=23.2, xrt_cpu=10.4,
          sha256="bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30"),
    Model("base.en", "ggml-base.en-q5_1.bin", 60, 208, 2,
          "Very light", "Fast on weak machines; more mistakes. English only.",
          alts=(("ggml-base.en.bin", 148),), xrt_gpu=57.2, xrt_cpu=29.8,
          sha256="4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f"),
    Model("tiny.en", "ggml-tiny.en-q5_1.bin", 32, 185, 1,
          "Minimal", "For very old machines. Expect many mistakes. English only.",
          alts=(("ggml-tiny.en.bin", 78),), xrt_gpu=72.9, xrt_cpu=73.7,
          sha256="c77c5766f1cef09b6b7d47f21b546cbddd4157886b3b5d6d4f709e91e66c7c2b"),
)
BY_ID = {m.id: m for m in CATALOG}
PARAKEET_ID = "parakeet-tdt-0.6b-v3-coreml"


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
            if model.asset:                       # only a model once everything in the zip is unpacked
                if all((d / r).exists() for r in model.required):
                    return p
                continue
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


def effective_mb(hw: dict) -> int:
    """RAM in Mac terms (see hardware.RAM_FACTOR); the tiers below are Mac-measured."""
    return hw.get("ram_effective_mb") or hw.get("ram_total_mb") or 0


def _macos_major(hw: dict) -> int:
    try:
        return int(str(hw.get("macos") or "0").split(".")[0])
    except ValueError:
        return 0


def available(model: Model, hw: dict) -> bool:
    """Whether this computer can run `model` at all (the list shows only what it can)."""
    if model.apple_only and hw.get("accelerator") != "metal":
        return False
    return not model.min_macos or _macos_major(hw) >= model.min_macos


def needs_newer_macos(model: Model, hw: dict) -> bool:
    """An Apple Silicon Mac that could run `model` if it had a newer macOS.
    The list shows these greyed out, with the reason, instead of hiding them."""
    return (not available(model, hw) and hw.get("accelerator") == "metal"
            and bool(model.min_macos))


def fluidaudio_present() -> bool:
    """The Parakeet engine (fluidaudiocli) is an executable file on disk."""
    cli = config.FLUIDAUDIO_CLI
    return os.path.isfile(cli) and os.access(cli, os.X_OK)


def recommend(hw: dict | None = None) -> Model:
    """Parakeet on an Apple chip with macOS 14+ when its engine is there; Whisper otherwise."""
    hw = hw or hardware.snapshot()
    pk = BY_ID[PARAKEET_ID]
    if available(pk, hw) and fluidaudio_present():
        return pk
    return recommend_whisper(hw)


def recommend_whisper(hw: dict) -> Model:
    """The Whisper model that suits this machine; also the fallback when Parakeet cannot run."""
    ram_gb = effective_mb(hw) / 1024
    accel = hw.get("accelerator") or "cpu"
    cores = hw.get("cpu_cores") or 2
    if accel != "cpu":
        if ram_gb >= 8:
            return BY_ID["large-v3-turbo-q4_0" if accel == "metal" else "large-v3-turbo-q5_0"]
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


# Real lectures ran slower than the 5-minute benchmark (a loaded machine,
# memory shared with the notes model), so estimates are padded, and an
# unknown non-Apple CPU is assumed to be half an M1's CPU.
_REAL_WORLD_SLOWDOWN = 1.5
_UNKNOWN_CPU_FACTOR = 0.5


def minutes_per_hour(m: Model, hw: dict) -> int:
    """Rough minutes to transcribe one hour of lecture with `m` here."""
    accel = hw.get("accelerator") or "cpu"
    if accel == "metal":
        xrt = m.xrt_gpu
    else:
        xrt = m.xrt_cpu * (1.0 if hw.get("os") == "Darwin" else _UNKNOWN_CPU_FACTOR)
    if xrt <= 0:
        return 0
    return max(1, round(60 / xrt * _REAL_WORLD_SLOWDOWN))


def device_label(hw: dict) -> str:
    """ "MacBook Air · Apple M1 · 8 GB memory" -- whatever is known."""
    d = hw.get("device") or {}
    ram = hw.get("ram_total_mb")
    parts = [d.get("name"), d.get("chip"),
             f"{round(ram / 1024)} GB memory" if ram else None]
    if hw.get("accelerator") == "cuda":
        parts.append("NVIDIA GPU")
    return " · ".join(p for p in parts if p) or "this computer"


def recommend_why(m: Model, hw: dict) -> str:
    """One sentence on why `m` suits this machine."""
    ram_gb = round((hw.get("ram_total_mb") or 0) / 1024)
    cores = hw.get("cpu_cores") or 0
    mins = minutes_per_hour(m, hw)
    took = f" An hour of lecture should take about {mins} min to transcribe." if mins else ""
    if m.id == PARAKEET_ID:
        base = ("Apple chip with macOS 14 or later: Parakeet runs on the Neural Engine and is several "
                "times faster than Whisper, though slightly less accurate. It covers 25 European "
                "languages, English included. Whisper models stay available if you prefer them.")
    elif hw.get("accelerator") == "metal":
        base = (f"Apple chip with {ram_gb} GB: enough for the balanced model, which uses the Neural Engine."
                if m.id == "large-v3-turbo-q4_0" else
                f"Apple chip with {ram_gb} GB: enough for the balanced model, which runs on the GPU."
                if m.id == "large-v3-turbo-q5_0" else
                f"Apple chip with {ram_gb} GB: a lighter model leaves memory for your other apps.")
    elif m.id == "large-v3-turbo-q5_0":
        base = f"{cores} CPU cores and {ram_gb} GB: strong enough for the balanced model without a GPU."
    else:
        base = (f"No supported GPU, {cores} CPU cores and {ram_gb} GB: "
                f"the {m.label.lower()} model keeps transcription reasonably quick.")
    if hw.get("os") == "Windows":
        base += " Windows needs about twice the memory a Mac does for the same model."
    return base + took


def active() -> tuple[Path | None, Model | None, str]:
    """The model file to transcribe with, plus why it was chosen.

    Order: LECTERN_MODEL (an explicit file) -> the user's saved choice ->
    the recommended model if installed -> the recommended Whisper model if installed (so a user
    who has not downloaded Parakeet keeps the model they had) -> the best installed model that
    fits this machine -> the smallest installed model. None if nothing is installed.
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
    if p := installed_path(wrec := recommend_whisper(hw)):
        return p, wrec, "recommended for this computer"
    have = installed()
    ok = [m for m in have if fits(m, effective_mb(hw) * 2**20 or None)]
    if ok:
        m = ok[0]
        return installed_path(m), m, "best installed model that fits this computer"
    if have:
        m = have[-1]
        return installed_path(m), m, "only model installed"
    return None, None, "none installed"


@functools.lru_cache(maxsize=1)
def _recommended() -> Model:                 # the hardware does not change while the app runs
    return recommend()


def upgrade_offer() -> str | None:
    """Id of the recommended model when it is a hosted download (not what an older Lectern fetched)
    that this computer does not have yet, and some other model is installed. None otherwise."""
    rec = _recommended()
    if rec.asset and installed_path(rec) is None and installed():
        return rec.id
    return None


def active_path() -> Path | None:
    return active()[0]


def active_name() -> str:
    p, m, _ = active()
    return m.id if m else (p.name if p else "none")


def describe() -> dict:
    hw = hardware.snapshot()
    rec = recommend(hw)
    p, cur, why = active()
    total = effective_mb(hw) * 2**20 or None
    return {
        "hardware": hw,
        "device_label": device_label(hw),
        "recommended": rec.id,
        "recommended_label": rec.label,
        "recommended_size_mb": rec.size_mb,
        "recommended_why": recommend_why(rec, hw),
        "active": {"id": cur.id if cur else None,
                   "file": p.name if p else None, "why": why},
        "models": [{
            "id": m.id, "label": m.label, "note": m.note,
            "size_mb": m.size_mb, "ram_mb": m.ram_mb,
            "installed": installed_path(m) is not None,
            "fits": fits(m, total),
            "recommended": m.id == rec.id,
            "min_per_hour": minutes_per_hour(m, hw),
            "needs_macos": m.min_macos if needs_newer_macos(m, hw) else 0,
        } for m in CATALOG
            if available(m, hw) or installed_path(m) or needs_newer_macos(m, hw)],
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
            return
        try:
            from . import warmup                 # late: warmup imports this module
            warmup.WARMER.request(installed_path(model))
        except Exception:
            log.exception("could not queue the engine warm-up for %s", model.id)

    def _download(self, model: Model) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dest = MODELS_DIR / model.file
        part = MODELS_DIR / ((model.asset or model.file) + ".part")
        have = part.stat().st_size if part.is_file() else 0

        need = model.size_mb * _MB - have + (model.size_mb * _MB if model.asset else 0)   # zip + unzipped
        free = hardware_free(MODELS_DIR)
        if free is not None and free < need + 200 * _MB:
            raise DownloadError(
                f"not enough disk space: need about {need // _MB + 200} MB free")

        url = f"{ASSET_URL}/{model.asset}" if model.asset else f"{BASE_URL}/{model.file}"
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            resp = urllib.request.urlopen(req, timeout=30, context=_ssl_context())
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:           # already complete?
                _install(part, dest, model)
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
        _install(part, dest, model)


def _install(part: Path, dest: Path, model: Model) -> None:
    _check_model_file(part, model)
    if not model.asset:
        part.replace(dest)
        return
    _unzip_model(part, model)
    part.unlink(missing_ok=True)


def _unzip_model(zpath: Path, model: Model) -> None:
    """Unzip into a temporary folder, check everything is there, then move each top-level item into
    the models folder, so a half-unzipped model never counts as installed."""
    tmp = MODELS_DIR / (model.asset + ".unzipping")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(zpath) as z:
            root = tmp.resolve()
            for name in z.namelist():            # refuse paths that climb out of the folder
                if not (tmp / name).resolve().is_relative_to(root):
                    raise DownloadError("the downloaded model has an unsafe file path")
            z.extractall(tmp)
        if not all((tmp / r).exists() for r in model.required):
            raise DownloadError("the downloaded model is missing files. Try the download again.")
        for item in tmp.iterdir():
            target = MODELS_DIR / item.name
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            item.rename(target)
    except (zipfile.BadZipFile, OSError) as e:
        zpath.unlink(missing_ok=True)
        raise DownloadError(f"could not unpack the model: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_ZIP_MAGIC = b"PK\x03\x04"
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
    if magic != (_ZIP_MAGIC if model.asset else _GGML_MAGIC) or abs(size - expected) > expected * _SIZE_TOLERANCE:
        path.unlink(missing_ok=True)
        raise DownloadError(
            "the downloaded file is not a valid speech model (a Wi-Fi login page "
            "or a network error?). Try again on a working connection.")
    if model.sha256 and _sha256(path) != model.sha256:
        # e.g. a resumed .part spliced onto bytes of a different revision.
        path.unlink(missing_ok=True)
        raise DownloadError(
            "the downloaded model is damaged (its checksum does not match). "
            "Press Download again to fetch a fresh copy.")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hardware_free(path: Path) -> int | None:
    import shutil
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


DOWNLOADER = Downloader()
