"""Sign a Lectern disk image and write the Sparkle update feed for it.

    .venv/bin/python packaging/macos/make_appcast.py dist/Lectern-0.2.1.dmg \\
        --notes packaging/macos/notes/0.2.1.html --out ~/lectern-site/appcast.xml

The signature comes from the update-signing private key in this Mac's login
keychain (made once with Sparkle's `generate_keys`; the matching public key is
UPDATE_PUBLIC_KEY in build_app.py). Installed copies of Lectern download the
feed, find a version newer than theirs, and refuse any archive whose signature
does not match their built-in public key.

The feed lists only the newest version: Lectern has no delta updates and an
older copy always jumps straight to the latest. Publish order matters: upload
the disk image to the GitHub release first, then deploy the feed, so no one is
offered a download that is not there yet.

Use --url-base to point the disk image URL somewhere else (a local test server).
"""
from __future__ import annotations

import argparse
import email.utils
import plistlib
import re
import subprocess
import sys
import tempfile
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_app  # noqa: E402  (also puts the repo on sys.path)
from lectern import config  # noqa: E402

RELEASES = "https://github.com/lqoro/Lectern/releases/download"


def app_version_in(dmg: Path) -> str:
    """The version of the Lectern.app inside the image. A feed that promises a
    different one would offer the same update again after every install."""
    with tempfile.TemporaryDirectory() as tmp:
        mount = Path(tmp) / "mnt"
        mount.mkdir()
        build_app.run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", mount, dmg],
                      stdout=subprocess.DEVNULL)
        try:
            info = plistlib.loads((mount / "Lectern.app" / "Contents" / "Info.plist").read_bytes())
        finally:
            subprocess.run(["hdiutil", "detach", "-force", mount], capture_output=True)
    return info["CFBundleVersion"]


def sign(dmg: Path) -> tuple[str, int]:
    tool = build_app.fetch_sparkle() / "bin" / "sign_update"
    out = subprocess.run([tool, dmg], capture_output=True, text=True, check=True).stdout
    sig = re.search(r'sparkle:edSignature="([^"]+)"', out)
    length = re.search(r'length="(\d+)"', out)
    if not (sig and length):
        raise SystemExit(f"sign_update printed something unexpected:\n{out}")
    return sig.group(1), int(length.group(1))


def feed_xml(version: str, notes_html: str, url: str, signature: str, length: int) -> str:
    """The appcast for one release (the only item in the feed)."""
    if "]]>" in notes_html:
        raise SystemExit("The release notes may not contain ']]>'.")
    return f"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>Lectern for Mac</title>
    <link>https://lectern-ten.vercel.app</link>
    <description>Updates for Lectern, the lecture recorder that stays on your Mac.</description>
    <language>en</language>
    <item>
      <title>Lectern {escape(version)}</title>
      <pubDate>{email.utils.formatdate(usegmt=True)}</pubDate>
      <sparkle:version>{escape(version)}</sparkle:version>
      <sparkle:shortVersionString>{escape(version)}</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>{build_app.MIN_MACOS}</sparkle:minimumSystemVersion>
      <description><![CDATA[{notes_html}]]></description>
      <enclosure url="{escape(url, quote=True)}"
                 sparkle:edSignature="{escape(signature, quote=True)}" length="{length}"
                 type="application/x-apple-diskimage"/>
    </item>
  </channel>
</rss>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dmg", type=Path)
    ap.add_argument("--notes", type=Path, required=True,
                    help="HTML fragment shown in the update window")
    ap.add_argument("--out", type=Path, required=True, help="where to write appcast.xml")
    ap.add_argument("--url-base", help=f"folder URL of the disk image (default {RELEASES}/v<version>)")
    a = ap.parse_args()

    version = config.VERSION
    if a.dmg.name != f"Lectern-{version}.dmg":
        raise SystemExit(f"{a.dmg.name} is not Lectern-{version}.dmg (config.VERSION).")
    inside = app_version_in(a.dmg)
    if inside != version:
        raise SystemExit(f"The app inside {a.dmg.name} is version {inside}, not {version}.")

    sig, length = sign(a.dmg)
    base = (a.url_base or f"{RELEASES}/v{version}").rstrip("/")
    feed = feed_xml(version, a.notes.read_text().strip(), f"{base}/{a.dmg.name}", sig, length)
    a.out.expanduser().write_text(feed)
    print(f"Wrote {a.out} for Lectern {version} ({length:,} bytes, signed)")


if __name__ == "__main__":
    main()
