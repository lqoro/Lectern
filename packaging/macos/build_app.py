"""Build Lectern.app and a test DMG on a Mac. No Apple developer account needed.

    .venv/bin/python packaging/macos/build_app.py            # app + DMG, ad-hoc signed
    .venv/bin/python packaging/macos/build_app.py --no-dmg   # just dist/Lectern.app

Everything a student would otherwise install with Homebrew goes inside the app,
so they download one file and run nothing in Terminal:

    Lectern.app/Contents/
      MacOS/LecternMenu              the app (lectern_menubar.swift): starts the
                                     server, shows Lectern in its own window and
                                     the menu bar mic, owns the microphone permission
      Resources/backend/lectern-server/   the Python server, frozen by PyInstaller
      Resources/bin/                 whisper-cli, ffmpeg, ffprobe, lectern-capture
      Resources/licenses/            Lectern's licence and the bundled tools' licences

The tools are built from pinned sources here, not copied from Homebrew: a
Homebrew binary links against /opt/homebrew libraries that a student's Mac does
not have. ffmpeg is configured without GPL parts and without auto-detected
libraries, so the result is LGPL and links only macOS system libraries.
whisper.cpp is built with GGML_NATIVE=OFF: a native build tuned for this Mac's
chip can crash on an older one.

Signing: by default everything is signed ad hoc ("-"), which is enough to run
it on this Mac. With a Developer ID (after joining the Apple Developer Program)
pass --identity "Developer ID Application: ..." to sign with the hardened
runtime, a secure timestamp and entitlements.plist; notarizing is a separate
step, see packaging/macos/README.md.

Needs: Xcode command line tools and Homebrew cmake (the script says what is
missing), and about 15 minutes for the first build. Tool
builds are cached in build/macos/ and reused while the pinned versions match.
"""
from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lectern import config  # noqa: E402

HERE = Path(__file__).resolve().parent
BUILD = REPO / "build" / "macos"
DIST = REPO / "dist"
APP = DIST / "Lectern.app"

# Pinned so a release is reproducible. whisper.cpp 1.9.1 is the version the
# flags in config.py / transcribe.py were measured against (see README).
WHISPER_TAG = os.environ.get("LECTERN_WHISPER_TAG", "v1.9.1")
FFMPEG_TAG = os.environ.get("LECTERN_FFMPEG_TAG", "n8.1.3")
# The commit each tag must resolve to: a tag moved upstream stops the build
# instead of shipping different source under the same name.
COMMITS = {"v1.9.1": "f049fff95a089aa9969deb009cdd4892b3e74916",
           "n8.1.3": "1041abdc962f4cc4f394aa8de9dc5236c0c3b9e7"}
PYINSTALLER = "pyinstaller>=6.10,<7"
MIN_MACOS = "12.0"
ARCH = "arm64"

BUNDLE_ID = "app.lectern"
EXE_NAME = "LecternMenu"
MIC_REASON = "Lectern records your lectures on this Mac. Audio never leaves it."

# Never ship a tool that loads a library the student's Mac does not have.
SYSTEM_LIB_PREFIXES = ("/usr/lib/", "/System/Library/", "@rpath/", "@loader_path/",
                       "@executable_path/")


# ------------------------------------------------------------------ helpers
def say(msg: str) -> None:
    print(f"==> {msg}", flush=True)


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def need(tool: str, hint: str) -> None:
    if not shutil.which(tool):
        raise SystemExit(f"{tool} is missing. {hint}")


def set_aside(path: Path) -> None:
    """Move an old build output out of the way. Like the app itself (see
    recovery.py), this script never rmtrees: delete build/macos/old by hand."""
    old = BUILD / "old"
    old.mkdir(parents=True, exist_ok=True)
    path.rename(old / f"{path.name}-{time.strftime('%Y%m%d-%H%M%S')}")


def clone(url: str, tag: str, dest: Path) -> Path:
    """Shallow checkout of one tag, reused if it is already there."""
    stamp = dest / ".lectern-tag"
    if stamp.is_file() and stamp.read_text().strip() == tag:
        return dest
    if dest.exists():
        set_aside(dest)
    run(["git", "clone", "--depth", "1", "--branch", tag, url, dest])
    head = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"],
                                   text=True).strip()
    want = COMMITS.get(tag)
    if want is None:
        say(f"warning: {tag} has no pinned commit ({head}); add it to COMMITS")
    elif head != want:
        raise SystemExit(f"{url} tag {tag} is now {head}, expected {want}. "
                         "Check the upstream change before building.")
    stamp.write_text(tag + "\n")
    return dest


def cached(out: Path, recipe: list) -> bool:
    """Reuse a built tool only if it was built with exactly this recipe (the
    tag, the commit and every flag); a changed flag rebuilds."""
    key = out.parent / ".lectern-recipe"
    want = "\n".join(str(x) for x in recipe)
    if out.is_file() and key.is_file() and key.read_text() == want:
        return True
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(want)          # written now; a failed build leaves no `out`
    return False


def deployment_env() -> dict:
    return {**os.environ, "MACOSX_DEPLOYMENT_TARGET": MIN_MACOS}


def check_links(binary: Path) -> None:
    out = subprocess.run(["otool", "-L", str(binary)], capture_output=True,
                         text=True, check=True).stdout.splitlines()[1:]
    bad = [line.strip() for line in out
           if line.strip() and not line.strip().startswith(SYSTEM_LIB_PREFIXES)]
    if bad:
        raise SystemExit(f"{binary.name} links libraries a student's Mac will not have:\n  "
                         + "\n  ".join(bad))


# ------------------------------------------------------------------- tools
def build_whisper() -> Path:
    out = BUILD / f"whisper-{WHISPER_TAG}" / "whisper-cli"
    flags = ["-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_OSX_ARCHITECTURES={ARCH}",
             f"-DCMAKE_OSX_DEPLOYMENT_TARGET={MIN_MACOS}",
             "-DBUILD_SHARED_LIBS=OFF", "-DGGML_NATIVE=OFF",
             "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON",
             "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_SERVER=OFF"]
    if cached(out, [WHISPER_TAG, COMMITS.get(WHISPER_TAG), *flags]):
        return out
    say(f"Building whisper.cpp {WHISPER_TAG} (Metal, static, portable CPU flags)")
    src = clone("https://github.com/ggml-org/whisper.cpp.git", WHISPER_TAG,
                BUILD / "src" / f"whisper.cpp-{WHISPER_TAG}")
    bdir = src / "build-lectern"
    run(["cmake", "-S", src, "-B", bdir, *flags], env=deployment_env())
    run(["cmake", "--build", bdir, "--config", "Release", "-j", os.cpu_count() or 4,
         "--target", "whisper-cli"], env=deployment_env())
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bdir / "bin" / "whisper-cli", out)
    shutil.copy2(src / "LICENSE", out.parent / "whisper.cpp-LICENSE")
    return out


def build_ffmpeg() -> tuple[Path, Path]:
    stage = BUILD / f"ffmpeg-{FFMPEG_TAG}"
    ff, fp = stage / "bin" / "ffmpeg", stage / "bin" / "ffprobe"
    cflags = f"-mmacosx-version-min={MIN_MACOS} -arch {ARCH}"
    # No --enable-gpl / --enable-nonfree: the build stays LGPL-2.1-or-later.
    # --disable-autodetect stops configure picking up Homebrew libraries.
    # audiotoolbox = the aac_at encoder (config.ARCHIVE_CODEC); avfoundation =
    # the recorder's last-resort fallback when the Swift helper is missing.
    flags = [f"--prefix={stage}", "--disable-autodetect",
             "--enable-audiotoolbox", "--enable-avfoundation", "--enable-zlib",
             "--disable-doc", "--disable-ffplay", "--disable-network",
             "--enable-static", "--disable-shared",
             f"--extra-cflags={cflags}", f"--extra-ldflags={cflags}"]
    if cached(ff, [FFMPEG_TAG, COMMITS.get(FFMPEG_TAG), *flags]) and fp.is_file():
        return ff, fp
    say(f"Building FFmpeg {FFMPEG_TAG} (LGPL, system libraries only)")
    src = clone("https://github.com/FFmpeg/FFmpeg.git", FFMPEG_TAG,
                BUILD / "src" / f"ffmpeg-{FFMPEG_TAG}")
    run(["./configure", *flags], cwd=src, env=deployment_env())
    run(["make", "-j", os.cpu_count() or 4], cwd=src, env=deployment_env())
    run(["make", "install"], cwd=src, env=deployment_env())
    for name in ("COPYING.LGPLv2.1", "LICENSE.md"):
        shutil.copy2(src / name, stage / f"ffmpeg-{name}")
    return ff, fp


def build_swift(src: Path, out: Path) -> Path:
    say(f"Compiling {src.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["/usr/bin/xcrun", "swiftc", "-O", "-swift-version", "5",
         "-target", f"{ARCH}-apple-macos{MIN_MACOS}", "-o", out, src])
    return out


def collect_licenses(py: Path) -> Path:
    """Full licence texts of everything bundled, for Resources/licenses and the
    app's Settings > Legal. The build venv holds exactly what PyInstaller freezes."""
    out = BUILD / "THIRD_PARTY_LICENSES.txt"
    site, stdlib = (subprocess.check_output(
        [py, "-c", f"import sysconfig; print(sysconfig.get_path('{k}'))"], text=True).strip()
        for k in ("purelib", "stdlib"))
    ff = BUILD / "src" / f"ffmpeg-{FFMPEG_TAG}"
    run([sys.executable, REPO / "packaging" / "collect_licenses.py",
         "--site-packages", site, "--python-license", Path(stdlib) / "LICENSE.txt",
         "--out", out,
         "--extra", f"FFmpeg {FFMPEG_TAG} (LICENSE.md)={ff / 'LICENSE.md'}",
         "--extra", f"FFmpeg {FFMPEG_TAG} (GNU LGPL 2.1)={ff / 'COPYING.LGPLv2.1'}"])
    return out


def build_backend() -> Path:
    """Freeze the server with PyInstaller into a one-folder build."""
    say("Freezing the Python server with PyInstaller")
    venv = BUILD / "venv"
    py = venv / "bin" / "python"
    if not py.is_file():
        run([sys.executable, "-m", "venv", venv])
    run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run([py, "-m", "pip", "install", "--quiet", "-r", REPO / "requirements.txt", PYINSTALLER])
    work, dist = BUILD / "pyinstaller", BUILD / "pyinstaller-dist"
    data = [(REPO / "web", "web"), (collect_licenses(py), ".")] + [
        (REPO / name, ".") for name in
        ("LICENSE", "THIRD_PARTY_NOTICES.md", "PRIVACY.md", "TERMS.md")]
    cmd = [py, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
           "--name", "lectern-server", "--target-architecture", ARCH,
           "--workpath", work, "--distpath", dist, "--specpath", work,
           "--paths", REPO, "--collect-submodules", "lectern"]
    for src, dest in data:
        # server.py finds web/ and the legal files two levels above its own
        # module, which in a PyInstaller build is the bundle's root folder.
        cmd += ["--add-data", f"{src}:{dest}"]
    run(cmd + [HERE / "server_entry.py"])
    return dist / "lectern-server"


def make_icon(dest: Path) -> None:
    """AppIcon.icns from the web app's 512 px icon."""
    src = REPO / "web" / "icon-512.png"
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "AppIcon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                if px > 512:
                    continue
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                run(["sips", "-z", px, px, src, "--out", iconset / name],
                    stdout=subprocess.DEVNULL)
        run(["iconutil", "-c", "icns", iconset, "-o", dest])


# ----------------------------------------------------------------- assemble
def info_plist() -> dict:
    return {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": "Lectern",
        "CFBundleDisplayName": "Lectern",
        "CFBundleExecutable": EXE_NAME,
        "CFBundleIconFile": "AppIcon",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": config.VERSION,
        "CFBundleVersion": config.VERSION,
        "LSMinimumSystemVersion": MIN_MACOS,
        "LSUIElement": False,                    # a Dock app with its own window
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "MIT License. See Resources/licenses.",
        "NSMicrophoneUsageDescription": MIC_REASON,
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    }


def assemble(menu: Path, capture: Path, backend: Path,
             whisper: Path, ffmpeg: Path, ffprobe: Path) -> None:
    say(f"Assembling {APP.relative_to(REPO)}")
    if APP.exists():
        set_aside(APP)
    contents = APP / "Contents"
    res = contents / "Resources"
    bin_dir, lic = res / "bin", res / "licenses"
    for d in (contents / "MacOS", bin_dir, lic):
        d.mkdir(parents=True)
    shutil.copy2(menu, contents / "MacOS" / EXE_NAME)
    shutil.copytree(backend, res / "backend" / "lectern-server", symlinks=True)
    for tool in (whisper, ffmpeg, ffprobe, capture):
        check_links(tool)
        shutil.copy2(tool, bin_dir / tool.name)
    for f in (REPO / "LICENSE", REPO / "THIRD_PARTY_NOTICES.md",
              BUILD / "THIRD_PARTY_LICENSES.txt", REPO / "PRIVACY.md", REPO / "TERMS.md",
              whisper.parent / "whisper.cpp-LICENSE",
              *sorted(ffmpeg.parent.parent.glob("ffmpeg-*"))):
        if f.is_file():
            shutil.copy2(f, lic / f.name)
    make_icon(res / "AppIcon.icns")
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(info_plist(), f)


def is_macho(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            magic = f.read(4)
    except OSError:
        return False
    return magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe",
                     b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")


def sign(identity: str) -> None:
    """Sign inside out: every nested binary first, the bundle last."""
    hardened = identity != "-"
    say(f"Signing ({'Developer ID, hardened runtime' if hardened else 'ad hoc'})")
    base = ["/usr/bin/codesign", "--force", "--sign", identity]
    if hardened:
        base += ["--options", "runtime", "--timestamp"]
    nested = [p for p in (APP / "Contents" / "Resources").rglob("*")
              if p.is_file() and not p.is_symlink() and is_macho(p)]
    for p in sorted(nested, key=lambda p: len(p.parts), reverse=True):
        extra = []
        if hardened and os.access(p, os.X_OK) and p.suffix not in (".so", ".dylib"):
            extra = ["--entitlements", HERE / "entitlements.plist"]
        run(base + extra + [p], stdout=subprocess.DEVNULL)
    ent = ["--entitlements", HERE / "entitlements.plist"] if hardened else []
    run(base + ent + [APP])
    run(["/usr/bin/codesign", "--verify", "--deep", "--strict", APP])


def smoke_test() -> None:
    """Start the bundled server against throwaway folders and ask it the
    questions the app depends on. Never touches ~/Lectures."""
    say("Smoke test: bundled server, web UI, legal docs, tools")
    res = APP / "Contents" / "Resources"
    port = 18477
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "LECTERN_BIN_DIR": str(res / "bin"),
               "LECTERN_PORT": str(port),
               "LECTERN_ROOT": str(Path(tmp) / "Lectures"),
               "LECTERN_SUPPORT": str(Path(tmp) / "support")}
        proc = subprocess.Popen([res / "backend" / "lectern-server" / "lectern-server"],
                                env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        try:
            def get(path: str) -> bytes:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}",
                                            timeout=3) as r:
                    return r.read()
            for _ in range(120):
                try:
                    get("/api/status")
                    break
                except OSError:
                    if proc.poll() is not None:
                        raise SystemExit("bundled server exited:\n" + proc.stderr.read()[-3000:])
                    time.sleep(0.5)
            else:
                raise SystemExit("bundled server did not answer within 60 s")
            assert b"<html" in get("/").lower(), "web UI not bundled"
            assert b"MIT License" in get("/api/legal/license"), "legal docs not bundled"
        finally:
            proc.terminate()
            proc.wait(timeout=15)
    run([res / "bin" / "whisper-cli", "--help"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    enc = subprocess.run([res / "bin" / "ffmpeg", "-hide_banner", "-encoders"],
                         capture_output=True, text=True, check=True).stdout
    assert " aac_at " in enc, "ffmpeg was built without the AudioToolbox AAC encoder"
    say("Smoke test passed")


def make_dmg() -> Path:
    dmg = DIST / f"Lectern-{config.VERSION}.dmg"
    say(f"Making {dmg.relative_to(REPO)}")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "Lectern"
        stage.mkdir()
        shutil.copytree(APP, stage / APP.name, symlinks=True)
        (stage / "Applications").symlink_to("/Applications")
        dmg.unlink(missing_ok=True)
        run(["hdiutil", "create", "-volname", "Lectern", "-srcfolder", stage,
             "-ov", "-format", "UDZO", dmg])
    return dmg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--identity", default="-",
                    help='codesign identity; "-" (default) signs ad hoc')
    ap.add_argument("--no-dmg", action="store_true", help="skip the disk image")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the smoke test")
    a = ap.parse_args()

    if sys.platform != "darwin":
        raise SystemExit("Lectern.app can only be built on a Mac.")
    if os.uname().machine != ARCH:
        raise SystemExit(f"Build on an Apple Silicon Mac ({ARCH}); this one is {os.uname().machine}.")
    need("git", "Install Apple's command line tools: xcode-select --install")
    need("cmake", "Install it with: brew install cmake")
    BUILD.mkdir(parents=True, exist_ok=True)

    whisper = build_whisper()
    ffmpeg, ffprobe = build_ffmpeg()
    capture = build_swift(config.CAPTURE_SRC, BUILD / "swift" / config.CAPTURE_BIN.name)
    menu = build_swift(REPO / "lectern" / "lectern_menubar.swift", BUILD / "swift" / EXE_NAME)
    backend = build_backend()
    assemble(menu, capture, backend, whisper, ffmpeg, ffprobe)
    sign(a.identity)
    if not a.skip_smoke:
        smoke_test()
    print(f"\nBuilt {APP}")
    if not a.no_dmg:
        print(f"Built {make_dmg()}")
    if a.identity == "-":
        print("Signed ad hoc: fine on this Mac. Other Macs will block it until it is\n"
              "signed with a Developer ID and notarized (packaging/macos/README.md).")


if __name__ == "__main__":
    main()
