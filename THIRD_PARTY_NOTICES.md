# Third-party software and models

Lectern itself is released under the MIT License (see `LICENSE`). The Mac app
and the Windows beta bundle the components listed in the first two sections.
When you run Lectern from source instead, nothing is bundled: you install the
tools yourself (with `install.sh`, Homebrew, pip or Ollama), and each one stays
under its own licence.

## Included in the Mac app (Lectern.app)

The Mac app (built with `packaging/macos/build_app.py`) bundles these, unmodified
and built from their official sources. The full licence texts are inside the app
under `Contents/Resources/licenses` (all of them together in
`THIRD_PARTY_LICENSES.txt`), and in the app under **Settings ▸ Legal ▸ Full
licence texts**.

| Component | Version | Licence |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (incl. ggml) | v1.9.1 | MIT |
| [FFmpeg](https://ffmpeg.org) (`ffmpeg`, `ffprobe`) | n8.1.3, built without GPL or non-free parts, as separate programs (not linked into Lectern) | LGPL-2.1-or-later |
| [Python](https://www.python.org) | 3.12 | PSF License |
| Python packages (table below), frozen with [PyInstaller](https://pyinstaller.org) | | as listed; PyInstaller's bootloader is GPL-2.0 with an exception that allows bundling |
| [PortAudio](https://www.portaudio.com) (inside sounddevice) | | MIT-style |
| [Sparkle](https://sparkle-project.org) (in-app updates; Mac app only) | 2.10.0, with its XPC services left out | MIT, with the licences of the few components it includes (its `LICENSE` file is in the app) |

## Included in the Windows beta (zip)

The Windows beta (built with `packaging/build_windows.py`) bundles these. Licence
texts are in the `licenses` folder and `THIRD_PARTY_LICENSES.txt` next to
`Lectern.bat`, and in the app under **Settings ▸ Legal**.

| Component | Version | Licence |
|---|---|---|
| [Python](https://www.python.org) embeddable package | 3.12.10 | PSF License |
| Python packages (table below) | | as listed |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) Windows binaries | b5454 | MIT |
| [FFmpeg](https://ffmpeg.org) LGPL shared build by [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds) | n8.1.3-24-g92611900bb | LGPL-2.1-or-later |
| Microsoft Visual C++ runtime (`msvcp140.dll`, `vcomp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`) | 14.51 | © Microsoft Corporation; redistributable files under Microsoft's Visual C++ Redistributable terms |

### FFmpeg source code

FFmpeg is LGPL software, and its complete source is available for exactly the
versions Lectern ships:

- **Mac app:** tag `n8.1.3`, <https://github.com/FFmpeg/FFmpeg/tree/n8.1.3>,
  configured as in `packaging/macos/build_app.py`.
- **Windows beta:** BtbN's `lgpl-shared` build of commit `92611900bb`
  (n8.1.3-24-g92611900bb),
  <https://github.com/FFmpeg/FFmpeg/tree/92611900bb112123e9bf997fdcc2d78e4c4922fc>.
  The scripts that built it, which pin the libraries it includes, are
  <https://github.com/BtbN/FFmpeg-Builds/tree/b4a2b1d7cc241786eb8e2b588a7ade40cad48bbd>
  (release `autobuild-2026-10-27-18-38`).

A copy of the FFmpeg source is attached to every Lectern release that bundles
it. Lectern runs FFmpeg as separate programs, so you can replace them with your
own build.

Speech models and the notes model are never bundled. They are downloaded only
when you ask.

## Installed separately (not distributed with Lectern)

| Component | Used for | Licence |
|---|---|---|
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (incl. ggml) | speech-to-text | MIT |
| [OpenAI Whisper](https://github.com/openai/whisper) model weights, as ggml files from [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) | speech-to-text models, downloaded only when you press Download | MIT |
| [FFmpeg](https://ffmpeg.org) (Homebrew build) | reading imported files, audio encode/decode | GPL-3.0-or-later as built by Homebrew (FFmpeg itself is LGPL-2.1-or-later with optional GPL parts) |
| [Ollama](https://ollama.com) | runs the optional study-notes model | MIT |
| [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (`qwen2.5:3b`) | optional study notes | **Qwen Research License** — permits research and evaluation use only; commercial use needs a licence from Alibaba Cloud. Read it before installing with `./install.sh --notes`. Without it, Lectern falls back to notes built from the lecture's own sentences. |

## Python packages (installed with pip from `requirements.txt`)

| Package | Licence |
|---|---|
| Flask, Werkzeug, Jinja2, itsdangerous, click, MarkupSafe | BSD-3-Clause |
| blinker | MIT |
| Waitress | Zope Public License 2.1 |
| sounddevice (bundles PortAudio on macOS and Windows) | MIT (PortAudio: MIT-style) |
| NumPy | BSD-3-Clause (bundled libraries have their own permissive licences) |
| psutil | BSD-3-Clause |
| certifi (CA certificate bundle) | MPL-2.0 |
| cffi, pycparser (used by sounddevice) | MIT, BSD-3-Clause |

## Apple frameworks

The macOS microphone helper and menu bar helper are compiled on your Mac from
source in this repository and use Apple's system frameworks (AVFoundation,
AppKit, Carbon, IOKit) under Apple's terms for macOS.

## Trademarks

Apple, Mac and macOS are trademarks of Apple Inc. Windows is a trademark of
Microsoft Corporation. "Whisper" is associated with OpenAI, "Qwen" with Alibaba
Cloud, "FFmpeg" with the FFmpeg project and "Ollama" with Ollama Inc. Lectern is
an independent project and is not affiliated with or endorsed by any of them, or
by any school or university.

Licence details were checked on 2026-10-30. If you spot an error, please
[open an issue](https://github.com/lqoro/Lectern/issues/new?template=bug.yml).
