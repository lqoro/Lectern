"""Build, install and remove the macOS menu bar helper (global start/stop key).

    python -m lectern.menubar install     # build "Lectern Menu.app", start it at login
    python -m lectern.menubar uninstall   # stop it and remove both

The helper is a tiny Swift status-bar app (lectern_menubar.swift). It is a real
.app bundle, not a bare binary, because macOS only grants microphone access to
an app that declares why it wants it (NSMicrophoneUsageDescription): the
server it starts, and the mic helper under that, inherit it as their
"responsible" app. Login start is a per-user LaunchAgent, no admin rights.
"""
from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from . import config

APP_NAME = "Lectern Menu"
BUNDLE_ID = "app.lectern.menubar"
EXE_NAME = "LecternMenu"
SRC = Path(__file__).with_name("lectern_menubar.swift")
REPO = Path(__file__).resolve().parent.parent
APP_DIR = Path.home() / "Applications" / f"{APP_NAME}.app"
AGENT = Path.home() / "Library" / "LaunchAgents" / f"{BUNDLE_ID}.plist"


def _info_plist() -> dict:
    return {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleExecutable": EXE_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": config.VERSION,
        "CFBundleVersion": config.VERSION,
        "LSMinimumSystemVersion": "12.0",
        "LSUIElement": True,                      # menu bar only, no Dock icon
        "NSMicrophoneUsageDescription":
            "Lectern records your lectures on this Mac. Audio never leaves it.",
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
    }


def build(dest: Path = APP_DIR) -> Path:
    if sys.platform != "darwin":
        raise SystemExit("The menu bar helper is macOS only.")
    macos = dest / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    exe = macos / EXE_NAME
    tmp = exe.with_suffix(".building")
    proc = subprocess.run(
        ["/usr/bin/xcrun", "swiftc", "-O", "-swift-version", "5",
         "-o", str(tmp), str(SRC)],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise SystemExit("Could not build the menu bar helper "
                         "(install Apple's command line tools: xcode-select --install)\n"
                         + proc.stderr[-2000:])
    os.replace(tmp, exe)
    with open(dest / "Contents" / "Info.plist", "wb") as f:
        plistlib.dump(_info_plist(), f)
    # Ad-hoc signature over the whole bundle so macOS ties the microphone
    # permission to this app consistently.
    subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(dest)],
                   capture_output=True, check=False)
    return exe


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["/bin/launchctl", *args], capture_output=True, text=True)


def install() -> None:
    exe = build()
    AGENT.parent.mkdir(parents=True, exist_ok=True)
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{BUNDLE_ID}")       # replace a running copy
    with open(AGENT, "wb") as f:
        plistlib.dump({
            "Label": BUNDLE_ID,
            "ProgramArguments": [str(exe), "--repo", str(REPO),
                                 "--port", str(config.PORT)],
            "RunAtLoad": True,
            "KeepAlive": {"SuccessfulExit": False},      # restart after a crash, not after Quit
            "ProcessType": "Interactive",
        }, f)
    r = _launchctl("bootstrap", domain, str(AGENT))
    if r.returncode != 0:
        raise SystemExit(f"Installed, but could not start it now: {r.stderr.strip()}\n"
                         "It will start at your next login.")
    print(f"Installed {APP_DIR} and started it (look for the mic icon in the menu bar).")
    print("Start/stop shortcut: ⌘L by default. Change it in Lectern ▸ Settings (⚙).")


def uninstall() -> None:
    _launchctl("bootout", f"gui/{os.getuid()}/{BUNDLE_ID}")
    AGENT.unlink(missing_ok=True)
    if APP_DIR.is_dir() and (APP_DIR / "Contents" / "MacOS" / EXE_NAME).exists():
        shutil.rmtree(APP_DIR)        # only our own bundle, checked by its executable
    print("Removed the Lectern menu bar helper.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "install":
        install()
    elif cmd == "uninstall":
        uninstall()
    elif cmd == "build":
        print(build())
    else:
        raise SystemExit("usage: python -m lectern.menubar install|uninstall|build")
