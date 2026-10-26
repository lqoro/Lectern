#!/bin/bash
# One-time setup on macOS. Safe to re-run.
#   ./install.sh            speech tools + Python env
#   ./install.sh --notes    also install Ollama and the small notes model (~2 GB)
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname)" != "Darwin" ]; then
  echo "This script is for macOS. On Windows/Linux see README: 'Windows and Linux (experimental)'."
  exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is needed first: https://brew.sh (then run this script again)."
  exit 1
fi

echo "==> Installing whisper.cpp, ffmpeg and Python 3.12 (Homebrew)"
brew install whisper-cpp ffmpeg python@3.12

PY="$(brew --prefix python@3.12)/bin/python3.12"
if [ ! -x .venv/bin/python ]; then
  echo "==> Creating the Python environment (.venv)"
  "$PY" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

if [ "${1:-}" = "--notes" ]; then
  echo "==> Installing Ollama and the notes model (qwen2.5:3b, ~2 GB)"
  brew install ollama
  brew services start ollama >/dev/null 2>&1 || true
  ollama pull qwen2.5:3b
else
  echo "    (Skipped AI notes. Without Ollama you still get transcripts and simple extractive notes."
  echo "     Run './install.sh --notes' to add them; it needs about 8 GB of RAM to be useful.)"
fi

chmod +x Lectern.command Lectern-on-Phone.command run.sh
echo
echo "Done. Start Lectern with:  ./Lectern.command"
echo "First launch: open Settings (⚙) and download the speech model it recommends for this computer."
