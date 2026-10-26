# Third-party software and models

Lectern itself is released under the MIT License (see `LICENSE`). It uses the
software and models below. Lectern does **not** include or redistribute any of
the separately installed tools or models: you install them yourself (with
`install.sh`, Homebrew, pip or Ollama), and each one stays under its own licence.

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

## Apple frameworks

The macOS microphone helper and menu bar helper are compiled on your Mac from
source in this repository and use Apple's system frameworks (AVFoundation,
AppKit, Carbon, IOKit) under Apple's terms for macOS.

## Trademarks

"Whisper" is associated with OpenAI, "Qwen" with Alibaba Cloud, "FFmpeg" with
the FFmpeg project and "Ollama" with Ollama Inc. Lectern is an independent
project and is not affiliated with or endorsed by any of them, or by any
school or university.

Licence details were checked on 2026-10-26. If you spot an error, please
[open an issue](https://github.com/lqoro/Lectern/issues/new?template=bug.yml).
