"""Write THIRD_PARTY_LICENSES.txt: the full licence text of everything a Lectern
download bundles, so the app can show it and the licences' notice terms are met.

    python packaging/collect_licenses.py --site-packages DIR --python-license FILE \
        --out FILE [--extra "Name=path/to/LICENSE" ...]

Python packages are read from their *.dist-info folders under --site-packages
(no imports, so this works on a Windows tree built from macOS). Only the
packages in SHIPPED are included; build-only tools (pip, setuptools) are not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# What the frozen/embedded server actually ships: requirements.txt and their
# dependencies, plus PyInstaller (its bootloader is inside the Mac app).
SHIPPED = ("flask", "werkzeug", "jinja2", "itsdangerous", "click", "markupsafe",
           "blinker", "waitress", "sounddevice", "cffi", "pycparser", "numpy",
           "psutil", "pyinstaller")

ALWAYS = (("PortAudio (inside sounddevice)", HERE / "licenses" / "portaudio-LICENSE.txt"),
          ("whisper.cpp and ggml", HERE / "licenses" / "whisper.cpp-LICENSE"))

RULE = "=" * 78


def dist_info(site: Path, name: str) -> Path | None:
    for d in sorted(site.glob("*.dist-info")):
        if d.name.split("-")[0].lower().replace("_", "-") == name:
            return d
    return None


def licence_files(d: Path) -> list[Path]:
    return [p for p in sorted(d.rglob("*")) if p.is_file()
            and ("licenses" in p.relative_to(d).parts
                 or p.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE")))]


def section(title: str, files: list[Path]) -> str:
    body = "\n\n".join(f.read_text(encoding="utf-8", errors="replace").strip() for f in files)
    return f"{RULE}\n{title}\n{RULE}\n\n{body}\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-packages", type=Path, required=True)
    ap.add_argument("--python-license", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", action="append", default=[], help='"Title=path"')
    a = ap.parse_args()

    parts = ["Lectern bundles the third-party software below. Each part keeps its own "
             "licence, reproduced here in full. A summary is in THIRD_PARTY_NOTICES.md.\n",
             section("Python", [a.python_license])]
    missing = []
    for name in SHIPPED:
        d = dist_info(a.site_packages, name)
        if d is None:
            continue                      # not part of this build (e.g. PyInstaller on Windows)
        files = licence_files(d)
        if not files:
            missing.append(name)
            continue
        version = d.name[:-len(".dist-info")].split("-", 1)[1]
        parts.append(section(f"{d.name.split('-')[0]} {version}", files))
    for title, path in [*ALWAYS, *(e.split("=", 1) for e in a.extra)]:
        path = Path(path)
        if not path.is_file():
            missing.append(title)
            continue
        parts.append(section(title, [path]))
    if missing:
        print(f"licence text not found for: {', '.join(missing)}", file=sys.stderr)
        return 1
    a.out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {a.out} ({len(parts) - 1} components)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
