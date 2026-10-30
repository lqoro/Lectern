"""Build the self-contained Windows beta zip from any OS (no Windows needed).

    python3 packaging/build_windows.py            # -> dist/Lectern-<ver>-windows-x64-beta.zip

The zip needs nothing installed on the PC: it carries the official Python
embeddable runtime, the Windows wheels of every dependency, ffmpeg/ffprobe
(LGPL build) and whisper-cli (whisper.cpp). Unzip, double-click Lectern.bat.

Downloads are cached in build/cache and checked against pinned SHA-256s once
known (see PINS); a changed upstream file stops the build instead of shipping.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "windows"
CACHE = ROOT / "build" / "cache"
DIST = ROOT / "dist"

PY_VER = "3.12.10"  # last 3.12 release with Windows binaries
PINS = {
    "python": f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip",
    "whisper": "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-x64.zip",
    # BtbN prunes dated builds after a few weeks; when this 404s, move to a
    # newer autobuild tag and refresh its hash in windows-pins.json.
    "ffmpeg": "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-27-18-38/"
              "ffmpeg-n8.1.3-24-g92611900bb-win64-lgpl-shared-8.1.zip",
}
SHA_FILE = Path(__file__).with_name("windows-pins.json")
MSVC_RUNTIME = "14.51.36231"
MSVC_DLLS = ("msvcp140.dll", "vcomp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")

# What of the repo goes into the app. Mac-only launchers and Swift helpers stay out.
APP_FILES = ["LICENSE", "README.md", "PRIVACY.md", "TERMS.md", "THIRD_PARTY_NOTICES.md", "requirements.txt"]
APP_DIRS = ["lectern", "web"]
MAC_ONLY = {"lectern_capture.swift", "lectern_menubar.swift"}

LAUNCHER = r"""@echo off
rem Lectern for Windows (beta). Everything it needs is in this folder.
cd /d "%~dp0"
set "LECTERN_BIN_DIR=%~dp0bin"
set "PYTHONDONTWRITEBYTECODE=1"
title Lectern
echo Starting Lectern... keep this window open while you use it.
echo Your lectures are saved in %USERPROFILE%\Lectures
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8477/"
"%~dp0python\python.exe" -m lectern
if errorlevel 1 (
  echo.
  echo Lectern stopped with an error. Copy the text above into a bug report:
  echo https://github.com/lqoro/Lectern/issues
  pause
)
"""

README_BETA = """Lectern for Windows - BETA
===========================

1. Unzip this folder somewhere permanent (e.g. Documents\\Lectern).
2. Double-click Lectern.bat. Your browser opens Lectern.
   Windows SmartScreen may warn about an unknown publisher: More info > Run anyway.
3. In Settings, download the speech model it recommends.
4. Allow microphone access if Windows asks
   (Settings > Privacy & security > Microphone > Let desktop apps access your microphone).

Recordings and notes are plain files in %USERPROFILE%\\Lectures.
AI notes are optional: install Ollama (https://ollama.com) and run
    ollama pull qwen2.5:3b
Without it you still get transcripts and simple key-sentence notes.

This is a beta: the Windows build has been checked by automated tests,
not yet on many real PCs. Please report problems at
https://github.com/lqoro/Lectern/issues
"""


def fetch(name: str, url: str, pins: dict) -> Path:
    dest = CACHE / url.rsplit("/", 1)[1]
    if not dest.exists():
        print(f"  downloading {name}: {url}")
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if pins.get(name) and pins[name] != digest:
        sys.exit(f"{name}: SHA-256 changed ({digest}); check upstream, then update {SHA_FILE.name}")
    pins[name] = digest
    return dest


def unzip_members(src: Path, out: Path, keep) -> int:
    """Extract members for which keep(name) returns a relative target, flattened."""
    n = 0
    with zipfile.ZipFile(src) as z:
        for m in z.infolist():
            if m.is_dir():
                continue
            target = keep(m.filename)
            if target:
                p = out / target
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(z.read(m))
                n += 1
    return n


def build_python(dist: Path, py_zip: Path) -> None:
    pydir = dist / "python"
    with zipfile.ZipFile(py_zip) as z:
        z.extractall(pydir)
    # The ._pth file replaces sys.path: stdlib zip, the runtime, our wheels, and
    # the app folder (one level up) so `python -m lectern` finds the package.
    tag = "".join(PY_VER.split(".")[:2])
    (pydir / f"python{tag}._pth").write_text(
        f"python{tag}.zip\n.\nLib\\site-packages\n..\nimport site\n", encoding="ascii")
    site = pydir / "Lib" / "site-packages"
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                    "--target", str(site), "--platform", "win_amd64", "--python-version", tag,
                    "--implementation", "cp", "--only-binary=:all:", "--no-compile",
                    "-r", str(ROOT / "requirements.txt")], check=True)
    for junk in site.glob("*/__pycache__"):
        shutil.rmtree(junk)
    shutil.rmtree(site / "bin", ignore_errors=True)


def build_bin(dist: Path, whisper_zip: Path, ffmpeg_zip: Path) -> None:
    bindir = dist / "bin"
    # whisper.cpp: the CLI and its DLLs (ggml, whisper) only.
    n = unzip_members(whisper_zip, bindir, lambda f: (
        Path(f).name if Path(f).name in ("whisper-cli.exe",) or f.endswith(".dll") else None))
    if not (bindir / "whisper-cli.exe").exists():
        sys.exit("whisper-cli.exe not found in the whisper.cpp zip")
    # ffmpeg shared build: ffmpeg/ffprobe plus the libav* DLLs; no ffplay.
    n += unzip_members(ffmpeg_zip, bindir, lambda f: (
        Path(f).name if "/bin/" in f and Path(f).name != "ffplay.exe" else None))
    n += unzip_members(ffmpeg_zip, dist / "licenses" / "ffmpeg", lambda f: (
        Path(f).name if Path(f).name.upper().startswith("LICENSE") else None))
    # whisper.cpp links the MSVC C++ runtime and OpenMP (MSVCP140, VCOMP140),
    # which a fresh Windows install does not have. Ship them app-local, as
    # Microsoft's redistributable terms allow.
    wheel_dir = CACHE / "msvc"
    if not any(wheel_dir.glob(f"msvc_runtime-{MSVC_RUNTIME}-*.whl")):
        subprocess.run([sys.executable, "-m", "pip", "download", "--quiet", "--disable-pip-version-check",
                        f"msvc-runtime=={MSVC_RUNTIME}", "--platform", "win_amd64", "--only-binary=:all:",
                        "--python-version", "312", "--no-deps", "-d", str(wheel_dir)], check=True)
    wheel = next(wheel_dir.glob(f"msvc_runtime-{MSVC_RUNTIME}-*.whl"))
    n += unzip_members(wheel, bindir, lambda f: (
        Path(f).name if "/Scripts/" in f and Path(f).name.lower() in MSVC_DLLS else None))
    missing = [d for d in MSVC_DLLS if not (bindir / d).exists()]
    if missing:
        sys.exit(f"MSVC runtime DLLs missing: {missing}")
    print(f"  bin: {n} files")


def copy_app(dist: Path) -> None:
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.split("\n"))
    for f in APP_FILES:
        shutil.copy2(ROOT / f, dist / f)
    for d in APP_DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and rel in tracked and p.name not in MAC_ONLY:
                (dist / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dist / rel)
    (dist / "Lectern.bat").write_bytes(LAUNCHER.replace("\n", "\r\n").encode("ascii"))
    (dist / "READ ME FIRST.txt").write_bytes(README_BETA.replace("\n", "\r\n").encode("ascii"))


def version() -> str:
    for line in (ROOT / "lectern" / "__init__.py").read_text().splitlines():
        if "__version__" in line:
            return line.split("=")[1].strip().strip("\"'")
    return "0"


def write_licences(dist: Path) -> None:
    """Full licence texts next to Lectern.bat; the app shows them in Settings > Legal."""
    lic = dist / "licenses"
    for f in ("whisper.cpp-LICENSE", "portaudio-LICENSE.txt"):
        shutil.copy2(ROOT / "packaging" / "licenses" / f, lic / f)
    subprocess.run([sys.executable, ROOT / "packaging" / "collect_licenses.py",
                    "--site-packages", dist / "python" / "Lib" / "site-packages",
                    "--python-license", dist / "python" / "LICENSE.txt",
                    "--out", dist / "THIRD_PARTY_LICENSES.txt",
                    "--extra", f"FFmpeg (BtbN lgpl-shared build)={lic / 'ffmpeg' / 'LICENSE.txt'}"],
                   check=True)


def main() -> None:
    pins = json.loads(SHA_FILE.read_text()) if SHA_FILE.exists() else {}
    src = {name: fetch(name, url, pins) for name, url in PINS.items()}
    SHA_FILE.write_text(json.dumps(pins, indent=2) + "\n")

    name = f"Lectern-{version()}-windows-x64-beta"
    dist = BUILD / name
    shutil.rmtree(dist, ignore_errors=True)
    dist.mkdir(parents=True)
    print("==> app"); copy_app(dist)
    print("==> python"); build_python(dist, src["python"])
    print("==> tools"); build_bin(dist, src["whisper"], src["ffmpeg"])
    print("==> licences"); write_licences(dist)

    DIST.mkdir(exist_ok=True)
    out = DIST / f"{name}.zip"
    out.unlink(missing_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(dist.rglob("*")):
            if p.is_file():
                z.write(p, Path(name) / p.relative_to(dist))
    print(f"==> {out.relative_to(ROOT)}  ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
